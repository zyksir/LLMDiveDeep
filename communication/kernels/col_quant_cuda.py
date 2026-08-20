"""CUDA kernels for the Lamport collectives (see comm.py for protocol).

Everything latency-critical is hand-written CUDA:

  * VECTORIZED volatile poll: ``ld.volatile.global.v4.u32`` + one
    ``__vcmpeq2`` per word - 8 bf16 sentinel checks per instruction.
    (A Triton prototype compiled the poll to scalar 16-bit loads and
    was deleted; this is why the comm kernels are CUDA-only.)
  * BATCHED sentinel checks: the reduce polls issue the loads of all
    ``world`` peer rows before testing any of them, so waits overlap
    instead of chaining up to ``world`` NVLink latencies.
  * PDL (programmatic dependent launch, sm90+): kernels sync at entry
    and trigger at exit and are launched with the programmatic-stream-
    serialization attribute, hiding ~1-2 us of launch latency - the
    same trick flashinfer's oneshot AR uses (launch_with_pdl).

The sentinel poll IS the synchronization (no barrier anywhere): a call
pushes before it polls, so seeing any of a peer's round-n data implies
that peer finished round n-1 entirely; 3-slot rotation makes the
clear/write/read of consecutive calls race-free (see comm.py).

Requirements: 16B-aligned base pointers, cols and row stride multiples
of 8 bf16 (the fc1 AG slice and all bench_comm shapes satisfy this).
"""

from __future__ import annotations

import os

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp8.h>
#include <cstdint>

#define SENT2 0x80008000u  // two bf16 -0.0 sentinels per word

// Programmatic dependent launch (sm90+): the next kernel's launch
// setup overlaps this kernel's tail. Sync FIRST (before reading any
// regular-memory input or metadata produced by the previous kernel);
// trigger LAST (release semantics - all this CTA's stores are visible
// to whoever passed the corresponding sync). flashinfer launches its
// oneshot AR exactly this way (launch_with_pdl=True); without it our
// small-message collectives pay ~1-2 us of exposed launch latency the
// AR does not.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define GRID_DEP_SYNC() cudaGridDependencySynchronize()
#define TRIGGER_PDL() cudaTriggerProgrammaticLaunchCompletion()
#else
#define GRID_DEP_SYNC()
#define TRIGGER_PDL()
#endif

__device__ __forceinline__ uint4 ld_volatile_v4(const uint4 *p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void st_volatile_v4(uint4 *p, uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// zero out any 16-bit lane equal to the sentinel (-0.0 -> +0.0)
__device__ __forceinline__ uint32_t canon(uint32_t w) {
    return w & ~__vcmpeq2(w, SENT2);
}

__device__ __forceinline__ bool has_sentinel(uint4 v) {
    return (__vcmpeq2(v.x, SENT2) | __vcmpeq2(v.y, SENT2) |
            __vcmpeq2(v.z, SENT2) | __vcmpeq2(v.w, SENT2)) != 0u;
}

// WORLD is a compile-time template parameter (dispatched on the host):
// with a runtime bound the per-rank push/poll loops cannot unroll, so
// the "batched" sentinel loads compile to a serial loop and the kernel
// trails TRT-LLM's (templated) oneshot AR by ~1-2 us at small sizes.
// clear_hist[cta][slot] remembers how many vectors round n wrote into
// slot n%3, so the re-sentinel pass (step 2) clears exactly what the
// previous call wrote instead of the whole slot. Clearing slot_bytes
// every call made the AG scale with the INSTANCE capacity, not the
// message: at [64, 10752] the full-slot clear alone was ~1.3 MB of
// stores and the AG lost to the flashinfer one-shot AR it should beat
// by 8x on wire bytes. (rs_cols already had this via meta[2].)
template <int WORLD>
__global__ void ag_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ rounds,
    int *__restrict__ clear_hist,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    int rows, int cols_v, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes,
    int rank)
{
    GRID_DEP_SYNC();
    const int out_cols_v = WORLD * cols_v;
    const int64_t write_v = (int64_t)rows * out_cols_v;
    __shared__ int s_base, s_clear;
    if (threadIdx.x == 0) {
        s_base = atomicAdd(rounds + blockIdx.x, 1);
        int *h = clear_hist + blockIdx.x * 3;
        s_clear = h[(s_base + 2) % 3];
        h[s_base % 3] = (int)write_v;
    }
    __syncthreads();
    const int base = s_base;
    const int64_t clear_v = s_clear;
    const int64_t off = data_off + (int64_t)(base % 3) * slot_bytes;
    const int64_t off_c = data_off + (int64_t)((base + 2) % 3) * slot_bytes;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthr = gridDim.x * blockDim.x;
    const int64_t numel_v = (int64_t)rows * cols_v;

    // 1) push my shard into every peer's slot (canonicalized)
    for (int64_t i = tid; i < numel_v; i += nthr) {
        const int row = i / cols_v;
        const int col = i - (int64_t)row * cols_v;
        uint4 v = x[row * x_stride_v + col];
        v.x = canon(v.x); v.y = canon(v.y);
        v.z = canon(v.z); v.w = canon(v.w);
        const int64_t dst = (int64_t)row * out_cols_v
                            + (int64_t)rank * cols_v + col;
        #pragma unroll
        for (int r = 0; r < WORLD; ++r) {
            uint4 *peer = reinterpret_cast<uint4 *>(
                (char *)buf_ptrs[r] + off);
            st_volatile_v4(peer + dst, v);
        }
    }

    // 2) re-sentinel the slot consumed last call (local writes only)
    uint4 *mine_c = reinterpret_cast<uint4 *>(
        (char *)buf_ptrs[rank] + off_c);
    const uint4 sent = {SENT2, SENT2, SENT2, SENT2};
    for (int64_t i = tid; i < clear_v; i += nthr)
        st_volatile_v4(mine_c + i, sent);

    // 3) vectorized poll + copy to the regular out tensor
    uint4 *mine = reinterpret_cast<uint4 *>((char *)buf_ptrs[rank] + off);
    const int64_t total_v = (int64_t)rows * out_cols_v;
    for (int64_t i = tid; i < total_v; i += nthr) {
        uint4 v;
        do { v = ld_volatile_v4(mine + i); } while (has_sentinel(v));
        out[i] = v;
    }
    TRIGGER_PDL();
}

// ------------------------------------------------------------------
// All-gather with the MXFP8 activation quantize FUSED on the write-out
// (out = e4m3 bytes + ue8m0 per-32 block scales instead of bf16). The
// transport is UNCHANGED (bf16 on wire - decode messages are latency-
// bound, and bf16 keeps the sentinel protocol intact); only step 3
// changes: one thread owns one 32-element scale block (4 vectors),
// polls its 4 vectors batched, then quantizes. The recipe is a
// bit-exact replica of TRT-LLM's cvt_warp_fp16_to_mxfp8
// (quantization.cuh): sf = e8m0(amax * rcp.approx(448)) rounded UP,
// out = e4m3(x * rcp.approx(2^sf)) satfinite, scale 0 when amax == 0.
// ------------------------------------------------------------------

__device__ __forceinline__ float rcp_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint64_t e4m3x8(const float2 (&f)[4]) {
    union { uint64_t u; __nv_fp8x2_e4m3 e[4]; } o;
    #pragma unroll
    for (int k = 0; k < 4; ++k) o.e[k] = __nv_fp8x2_e4m3(f[k]);
    return o.u;
}

template <int WORLD>
__global__ void ag_mxfp8_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ rounds,
    int *__restrict__ clear_hist,
    const uint4 *__restrict__ x, uint64_t *__restrict__ out,
    uint8_t *__restrict__ sf,
    int rows, int cols_v, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes,
    int rank)
{
    GRID_DEP_SYNC();
    const int out_cols_v = WORLD * cols_v;
    const int64_t write_v = (int64_t)rows * out_cols_v;
    __shared__ int s_base, s_clear;
    if (threadIdx.x == 0) {
        s_base = atomicAdd(rounds + blockIdx.x, 1);
        int *h = clear_hist + blockIdx.x * 3;
        s_clear = h[(s_base + 2) % 3];
        h[s_base % 3] = (int)write_v;
    }
    __syncthreads();
    const int base = s_base;
    const int64_t clear_v = s_clear;
    const int64_t off = data_off + (int64_t)(base % 3) * slot_bytes;
    const int64_t off_c = data_off + (int64_t)((base + 2) % 3) * slot_bytes;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthr = gridDim.x * blockDim.x;
    const int64_t numel_v = (int64_t)rows * cols_v;

    // 1) push my shard into every peer's slot (identical to ag_lamport)
    for (int64_t i = tid; i < numel_v; i += nthr) {
        const int row = i / cols_v;
        const int col = i - (int64_t)row * cols_v;
        uint4 v = x[row * x_stride_v + col];
        v.x = canon(v.x); v.y = canon(v.y);
        v.z = canon(v.z); v.w = canon(v.w);
        const int64_t dst = (int64_t)row * out_cols_v
                            + (int64_t)rank * cols_v + col;
        #pragma unroll
        for (int r = 0; r < WORLD; ++r) {
            uint4 *peer = reinterpret_cast<uint4 *>(
                (char *)buf_ptrs[r] + off);
            st_volatile_v4(peer + dst, v);
        }
    }

    // 2) re-sentinel the slot consumed last call
    uint4 *mine_c = reinterpret_cast<uint4 *>(
        (char *)buf_ptrs[rank] + off_c);
    const uint4 sent = {SENT2, SENT2, SENT2, SENT2};
    for (int64_t i = tid; i < clear_v; i += nthr)
        st_volatile_v4(mine_c + i, sent);

    // 3) poll one 32-element block per thread (4 vectors, batched
    //    sentinel check), then quantize to e4m3 + one ue8m0 scale
    uint4 *mine = reinterpret_cast<uint4 *>((char *)buf_ptrs[rank] + off);
    const int64_t nblk = (int64_t)rows * (out_cols_v / 4);
    for (int64_t b = tid; b < nblk; b += nthr) {
        const int64_t v0 = b * 4;
        uint4 v[4];
        bool done = false;
        while (!done) {
            done = true;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ld_volatile_v4(mine + v0 + j);
                done &= !has_sentinel(v[j]);
            }
        }
        float2 f[16];
        float vmax = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t *w = &v[j].x;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                f[j * 4 + k] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&w[k]));
                vmax = fmaxf(vmax, fmaxf(fabsf(f[j * 4 + k].x),
                                         fabsf(f[j * 4 + k].y)));
            }
        }
        __nv_fp8_e8m0 t;
        t.__x = __nv_cvt_float_to_e8m0(
            vmax * rcp_ftz(448.0f), __NV_SATFINITE, cudaRoundPosInf);
        sf[b] = t.__x;
        const float scale =
            (vmax != 0.f) ? rcp_ftz(static_cast<float>(t)) : 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 q[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                q[k].x = f[j * 4 + k].x * scale;
                q[k].y = f[j * 4 + k].y * scale;
            }
            out[v0 + j] = e4m3x8(q);
        }
    }
    TRIGGER_PDL();
}

__device__ __forceinline__ void acc_u4(float2 acc[4], uint4 v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 b =
            *reinterpret_cast<const __nv_bfloat162 *>(&w[j]);
        const float2 f = __bfloat1622float2(b);
        acc[j].x += f.x;
        acc[j].y += f.y;
    }
}

// Reduce-scatter (out = my chunk of the rank-sum). Sentinel poll IS
// the synchronization; no barrier anywhere.
template <int WORLD>
__global__ void rs_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ rounds,
    int *__restrict__ clear_hist,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    int64_t chunk_v,
    int64_t data_off, int64_t slot_bytes,
    int rank)
{
    GRID_DEP_SYNC();
    const int64_t write_v = (int64_t)WORLD * chunk_v;
    __shared__ int s_base, s_clear;
    if (threadIdx.x == 0) {
        s_base = atomicAdd(rounds + blockIdx.x, 1);
        int *h = clear_hist + blockIdx.x * 3;
        s_clear = h[(s_base + 2) % 3];
        h[s_base % 3] = (int)write_v;
    }
    __syncthreads();
    const int base = s_base;
    const int64_t clear_v = s_clear;
    const int64_t off = data_off + (int64_t)(base % 3) * slot_bytes;
    const int64_t off_c = data_off + (int64_t)((base + 2) % 3) * slot_bytes;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthr = gridDim.x * blockDim.x;

    // 1) push: my contribution to each peer's slot at row `rank`
    //    (peer r gets my chunk r). Self IS included: skipping it
    //    (reading own chunk from x in step 3) was measured SLOWER -
    //    the r==rank branch in these unrolled loops costs more than
    //    the local slot round-trip it saves.
    for (int64_t i = tid; i < chunk_v; i += nthr) {
        #pragma unroll
        for (int r = 0; r < WORLD; ++r) {
            uint4 v = x[(int64_t)r * chunk_v + i];
            v.x = canon(v.x); v.y = canon(v.y);
            v.z = canon(v.z); v.w = canon(v.w);
            uint4 *peer = reinterpret_cast<uint4 *>(
                (char *)buf_ptrs[r] + off);
            st_volatile_v4(peer + (int64_t)rank * chunk_v + i, v);
        }
    }

    // 2) re-sentinel the slot consumed last call
    uint4 *mine_c = reinterpret_cast<uint4 *>(
        (char *)buf_ptrs[rank] + off_c);
    const uint4 sent = {SENT2, SENT2, SENT2, SENT2};
    for (int64_t i = tid; i < clear_v; i += nthr)
        st_volatile_v4(mine_c + i, sent);

    // 3) poll the WORLD rows of my slot with a BATCHED sentinel check
    //    (issue all world loads, then test together - the serial
    //    do/while-per-rank variant chained up to world NVLink
    //    latencies and lost to the flashinfer AR at small sizes),
    //    accumulate fp32, store bf16
    uint4 *mine = reinterpret_cast<uint4 *>((char *)buf_ptrs[rank] + off);
    for (int64_t i = tid; i < chunk_v; i += nthr) {
        uint4 vals[WORLD];
        bool done = false;
        while (!done) {
            done = true;
            #pragma unroll
            for (int s = 0; s < WORLD; ++s) {
                vals[s] = ld_volatile_v4(mine + (int64_t)s * chunk_v + i);
                done &= !has_sentinel(vals[s]);
            }
        }
        float2 acc[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int s = 0; s < WORLD; ++s)
            acc_u4(acc, vals[s]);
        uint4 o;
        uint32_t *ow = &o.x;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162 b = __float22bfloat162_rn(acc[j]);
            ow[j] = *reinterpret_cast<const uint32_t *>(&b);
        }
        out[i] = o;
    }
    TRIGGER_PDL();
}

// launch with the PDL attribute (pairs with GRID_DEP_SYNC/TRIGGER_PDL
// in the kernels)
template <typename K, typename... Args>
static void launch_pdl(K kernel, int grid, int block, Args... args)
{
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

// dispatch the runtime world size to the compile-time template
#define DISPATCH_WORLD(world, ...)                                     \
    switch (world) {                                                   \
        case 2: { constexpr int kWorld = 2; __VA_ARGS__; break; }      \
        case 4: { constexpr int kWorld = 4; __VA_ARGS__; break; }      \
        case 8: { constexpr int kWorld = 8; __VA_ARGS__; break; }      \
        default: TORCH_CHECK(false, "unsupported world size ", world); \
    }

torch::Tensor ag_lamport(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block)
{
    const int rows = x.size(0), cols = x.size(1);
    const int64_t stride = x.stride(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(cols % 8 == 0 && stride % 8 == 0,
                "need 16B-aligned rows");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    TORCH_CHECK(clear_hist.numel() == rounds.numel() * 3);
    auto out = torch::empty({rows, world * cols}, x.options());
    DISPATCH_WORLD(world,
        launch_pdl(ag_lamport_kernel<kWorld>,
            (int)rounds.numel(), (int)block,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            clear_hist.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            rows, (int)(cols / 8), stride / 8,
            data_off, slot_bytes, (int)rank));
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> ag_mxfp8(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block)
{
    const int rows = x.size(0), cols = x.size(1);
    const int64_t stride = x.stride(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(cols % 8 == 0 && stride % 8 == 0,
                "need 16B-aligned rows");
    TORCH_CHECK((world * cols) % 32 == 0, "gathered row % 32 != 0");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    TORCH_CHECK(clear_hist.numel() == rounds.numel() * 3);
    auto out = torch::empty({rows, world * cols},
                            x.options().dtype(torch::kFloat8_e4m3fn));
    auto sf = torch::empty({rows, world * cols / 32},
                           x.options().dtype(torch::kUInt8));
    DISPATCH_WORLD(world,
        launch_pdl(ag_mxfp8_lamport_kernel<kWorld>,
            (int)rounds.numel(), (int)block,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            clear_hist.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint64_t *>(out.data_ptr()),
            reinterpret_cast<uint8_t *>(sf.data_ptr()),
            rows, (int)(cols / 8), stride / 8,
            data_off, slot_bytes, (int)rank));
    return {out, sf};
}

// Column reduce-scatter: FlashInfer oneshot Lamport AR layout
// (allreduce_fusion_kernel_oneshot_lamport) with push/poll shrunk to
// the local column chunk - world x less wire than AR.
//
// Slot rotation matches FI: rounds[0]=flag, rounds[1]=entry counter,
// rounds[2]=previous clear size in bytes (not per-CTA counters), so
// the launch grid can vary per call and still agree on the slot.
//
// Poll matches FI: load all world ranks, then check sentinels together
// (memory-level parallelism), instead of spinning one peer at a time.
//
// (Fused-RMSNorm, deferred-scale, and fused-MoE-finalize variants of
// this kernel existed and were REMOVED: every one of them lost to the
// flashinfer fused AR+norm tail - see moe_optimization.md Appendix A.)
template <int WORLD>
__global__ void rs_cols_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ meta,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    int rows, int chunk_v, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes,
    int rank)
{
    GRID_DEP_SYNC();
    // meta: [flag, counter, clear_bytes]
    __shared__ int s_flag, s_clear;
    if (threadIdx.x == 0) {
        s_flag = meta[0];
        s_clear = meta[2];
        atomicAdd(meta + 1, 1);
    }
    __syncthreads();
    const int flag = s_flag;
    const int64_t off = data_off + (int64_t)(flag % 3) * slot_bytes;
    const int64_t off_c = data_off + (int64_t)((flag + 2) % 3) * slot_bytes;
    const int64_t tot = (int64_t)rows * chunk_v;
    const int64_t clear_v = (int64_t)s_clear / 16;

    // ---- FI-style: one block owns one token (stride by grid) ----
    for (int t = (int)blockIdx.x; t < rows; t += (int)gridDim.x) {
        // 1) push: peer r gets my r-th column slice of token t
        for (int c = (int)threadIdx.x; c < chunk_v; c += (int)blockDim.x) {
            const int64_t idx = (int64_t)t * chunk_v + c;
            #pragma unroll
            for (int r = 0; r < WORLD; ++r) {
                uint4 v = x[(int64_t)t * x_stride_v
                            + (int64_t)r * chunk_v + c];
                v.x = canon(v.x); v.y = canon(v.y);
                v.z = canon(v.z); v.w = canon(v.w);
                uint4 *peer = reinterpret_cast<uint4 *>(
                    (char *)buf_ptrs[r] + off);
                st_volatile_v4(peer + (int64_t)rank * tot + idx, v);
            }
        }
    }

    // 2) clear the slot the previous call consumed (size-tracked)
    {
        uint4 *mine_c = reinterpret_cast<uint4 *>(
            (char *)buf_ptrs[rank] + off_c);
        const uint4 sent = {SENT2, SENT2, SENT2, SENT2};
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int nthr = gridDim.x * blockDim.x;
        for (int64_t i = tid; i < clear_v; i += nthr)
            st_volatile_v4(mine_c + i, sent);
    }

    uint4 *mine = reinterpret_cast<uint4 *>(
        (char *)buf_ptrs[rank] + off);

    for (int t = (int)blockIdx.x; t < rows; t += (int)gridDim.x) {
        // 3) FI batched poll: issue all ranks, then check together
        for (int c = (int)threadIdx.x; c < chunk_v; c += (int)blockDim.x) {
            const int64_t idx = (int64_t)t * chunk_v + c;
            uint4 vals[WORLD];
            bool done = false;
            while (!done) {
                done = true;
                #pragma unroll
                for (int r = 0; r < WORLD; ++r) {
                    vals[r] = ld_volatile_v4(
                        mine + (int64_t)r * tot + idx);
                    done &= !has_sentinel(vals[r]);
                }
            }
            float2 acc[4] = {{0.f, 0.f}, {0.f, 0.f},
                             {0.f, 0.f}, {0.f, 0.f}};
            #pragma unroll
            for (int r = 0; r < WORLD; ++r)
                acc_u4(acc, vals[r]);
            uint4 o;
            uint32_t *ow = &o.x;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162 b = __float22bfloat162_rn(acc[j]);
                ow[j] = *reinterpret_cast<const uint32_t *>(&b);
            }
            out[idx] = o;
        }
    }

    // FI-style flag advance: block 0 waits until every CTA entered
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (meta[1] != (int)gridDim.x) {}
        meta[0] = (flag + 1) % 3;
        meta[2] = (int)((int64_t)WORLD * tot * 16);
        meta[1] = 0;
    }
    TRIGGER_PDL();
}


torch::Tensor rs_cols(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world)
{
    const int rows = x.size(0), cols = x.size(1);
    const int64_t stride = x.stride(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(cols % (world * 8) == 0 && stride % 8 == 0,
                "need 16B-aligned chunks");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    TORCH_CHECK(rounds.numel() >= 3, "rounds needs [flag,ctr,clear]");
    const int chunk_cols = cols / world;
    const int chunk_v = chunk_cols / 8;
    const int64_t numel_v = (int64_t)rows * chunk_v;
    TORCH_CHECK(numel_v * world * 16 <= slot_bytes, "slot too small");
    // FI-like launch: one CTA per token, block covers the column chunk.
    // Cap grid by SM count so large B still saturates without over-sub.
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount,
                           x.get_device());
    if (sm < 1) sm = 1;
    int grid = rows < sm ? rows : sm;
    if (grid < 1) grid = 1;
    const int block = 256;
    // clear_bytes (meta[2]) must be primed by the host to slot_bytes
    // before the first call; do NOT read it back here (graph-unsafe).
    auto out = torch::empty({rows, chunk_cols}, x.options());
    DISPATCH_WORLD(world,
        launch_pdl(rs_cols_lamport_kernel<kWorld>, grid, block,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            rows, chunk_v, stride / 8,
            data_off, slot_bytes, (int)rank));
    return out;
}

torch::Tensor rs_lamport(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block)
{
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    const int64_t numel = x.numel();
    TORCH_CHECK(numel % world == 0);
    const int64_t chunk = numel / world;
    TORCH_CHECK(chunk % 8 == 0, "need 16B-aligned chunks");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    TORCH_CHECK(clear_hist.numel() == rounds.numel() * 3);
    auto out = torch::empty({chunk}, x.options());
    DISPATCH_WORLD(world,
        launch_pdl(rs_lamport_kernel<kWorld>,
            (int)rounds.numel(), (int)block,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            clear_hist.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            chunk / 8, data_off, slot_bytes,
            (int)rank));
    return out;
}

// ------------------------------------------------------------------
// Copy-engine (CE / DMA) collectives support. The data movement is
// cudaMemcpy2DAsync peer copies issued from the host - the DMA
// engines do the transfer, ZERO SMs, so it overlaps compute by
// construction and has no Lamport re-sentinel cost at large sizes.
//
// Sync = ONE kernel per collective, placed AFTER the copies (push
// orientation: every rank only SENDS, so the only thing to wait for
// is arrival of the peers' pushes):
//
//   * publish "my round-r pushes are complete" - correct by stream
//     order, the kernel runs only after this rank's memcpys finished;
//   * wait for every peer's publication (the unavoidable data-arrival
//     rendezvous, gating only the CONSUMER of the slot, never the
//     copies);
//   * bump the device round counter (graph replay safe: no host
//     state, fixed memcpy node addresses, single data slot).
//
// There is NO pre-copy barrier. Slot-reuse safety comes from stream
// order + pipeline symmetry: my round r+1 copies are issued after my
// round-r consumer (same stream), and every rank runs the same
// pipeline, so by the time anyone pushes round r+1 into my slot my
// consumer has had one full compute phase to finish reading round r.
//
// Two thread layouts (pick by measurement): one warp (thread t owns
// peer t) or one thread (loops over peers; batches the poll loads).
// ------------------------------------------------------------------

__device__ __forceinline__ void st_release_sys(int *p, int v) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v));
}

__device__ __forceinline__ int ld_acquire_sys(const int *p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template <int WORLD>
__global__ void ce_sync_warp_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ round_buf,
    int64_t flag_off, int rank)
{
    GRID_DEP_SYNC();
    const int r = *round_buf;
    if (threadIdx.x < WORLD) {
        int *peer = reinterpret_cast<int *>(
            (char *)buf_ptrs[threadIdx.x] + flag_off) + rank;
        st_release_sys(peer, r);
        const int *mine = reinterpret_cast<const int *>(
            (char *)buf_ptrs[rank] + flag_off) + threadIdx.x;
        // nanosleep backoff: the spin can last a full DMA copy time
        // when overlapped with compute, and a tight .sys-acquire poll
        // loop measurably slows a concurrent GEMM (+15%); backing off
        // makes the wait free while adding <1 us of signal latency
        while (ld_acquire_sys(mine) < r) { __nanosleep(256); }
    }
    __syncwarp();
    if (threadIdx.x == 0) *round_buf = r + 1;
    TRIGGER_PDL();
}

template <int WORLD>
__global__ void ce_sync_1t_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ round_buf,
    int64_t flag_off, int rank)
{
    GRID_DEP_SYNC();
    const int r = *round_buf;
    #pragma unroll
    for (int p = 0; p < WORLD; ++p) {
        int *peer = reinterpret_cast<int *>(
            (char *)buf_ptrs[p] + flag_off) + rank;
        st_release_sys(peer, r);
    }
    const int *mine = reinterpret_cast<const int *>(
        (char *)buf_ptrs[rank] + flag_off);
    bool done = false;
    while (!done) {
        done = true;
        int v[WORLD];
        #pragma unroll
        for (int p = 0; p < WORLD; ++p)
            v[p] = ld_acquire_sys(mine + p);
        #pragma unroll
        for (int p = 0; p < WORLD; ++p)
            done &= (v[p] >= r);
        if (!done) __nanosleep(256);
    }
    *round_buf = r + 1;
    TRIGGER_PDL();
}

void ce_sync(torch::Tensor buf_ptrs, torch::Tensor round_buf,
             int64_t flag_off, int64_t rank, int64_t world,
             int64_t threads)
{
    DISPATCH_WORLD(world,
        if (threads == 1)
            launch_pdl(ce_sync_1t_kernel<kWorld>, 1, 1,
                       (const int64_t *)buf_ptrs.data_ptr<int64_t>(),
                       round_buf.data_ptr<int>(), flag_off, (int)rank);
        else
            launch_pdl(ce_sync_warp_kernel<kWorld>, 1, 32,
                       (const int64_t *)buf_ptrs.data_ptr<int64_t>(),
                       round_buf.data_ptr<int>(), flag_off, (int)rank));
}

// pitched device-to-device copy on the current stream (UVA resolves
// peer pointers; the driver routes it through a copy engine)
void ce_copy2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
               int64_t width, int64_t height)
{
    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        reinterpret_cast<void *>(dst), (size_t)dpitch,
        reinterpret_cast<const void *>(src), (size_t)spitch,
        (size_t)width, (size_t)height, cudaMemcpyDefault,
        at::cuda::getCurrentCUDAStream()));
}

// SM copy kernel for the LOCAL (same-device) pitched copy. A local
// cudaMemcpy2DAsync is executed by the driver's SM copy path and it
// does NOT co-schedule with a saturating kernel - the profiler shows
// it queued until the concurrent GEMM nearly drains, which stalls the
// arrival sync and serializes the whole CE collective with compute.
// A regular small kernel (like ce_sync) slots into SMs as blocks of
// the big kernel retire, so the local shard placement is done here.
__global__ void local_copy2d_kernel(
    char *__restrict__ dst, int64_t dpitch,
    const char *__restrict__ src, int64_t spitch,
    int64_t width, int64_t height)
{
    GRID_DEP_SYNC();
    const int64_t wvec = width / 16;  // width%16==0 checked host-side
    const int64_t total = wvec * height;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const int64_t row = i / wvec, col = (i % wvec) * 16;
        *reinterpret_cast<uint4 *>(dst + row * dpitch + col) =
            *reinterpret_cast<const uint4 *>(src + row * spitch + col);
    }
    TRIGGER_PDL();
}

void ce_local2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
                int64_t width, int64_t height)
{
    if ((width | dst | src | dpitch | spitch) % 16 == 0) {
        const int64_t total = (width / 16) * height;
        int grid = (int)((total + 255) / 256);
        grid = grid < 1 ? 1 : (grid > 64 ? 64 : grid);
        launch_pdl(local_copy2d_kernel, grid, 256,
                   reinterpret_cast<char *>(dst), dpitch,
                   reinterpret_cast<const char *>(src), spitch,
                   width, height);
    } else {
        ce_copy2d(dst, dpitch, src, spitch, width, height);
    }
}

// ONE pybind crossing for a whole collective's copy set: jobs is a CPU
// int64 tensor [N, 6] of (dst, dpitch, src, spitch, width, height);
// destinations inside [local_lo, local_hi) route through the SM copy
// kernel exactly like the Python issue() dispatch. Eager host cost of
// the per-peer loop measured ~4.2us/call x 28 calls per SP prefill
// iteration - this folds each collective's 8 calls into 1.
void ce_copy2d_batch(torch::Tensor jobs, int64_t local_lo,
                     int64_t local_hi)
{
    TORCH_CHECK(jobs.device().is_cpu() && jobs.dtype() == torch::kLong &&
                jobs.is_contiguous() && jobs.numel() % 6 == 0,
                "jobs must be a contiguous CPU int64 [N,6] tensor");
    const int64_t *j = jobs.data_ptr<int64_t>();
    const int64_t n = jobs.numel() / 6;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t *e = j + i * 6;
        if (e[0] >= local_lo && e[0] < local_hi)
            ce_local2d(e[0], e[1], e[2], e[3], e[4], e[5]);
        else
            ce_copy2d(e[0], e[1], e[2], e[3], e[4], e[5]);
    }
}

// ------------------------------------------------------------------
// Whole-collective host paths in C++ (one pybind crossing each): job
// computation, copy issue, sync and the slab reduce all happen here.
// World is a RUNTIME arg for the pointer math (any TP) and reaches the
// templated ce_sync via its own DISPATCH_WORLD (TP = 2 / 4 / 8).
// Semantics are byte-identical to comm.py's push_rows / reduce_rows /
// all_gather_rows / all_gather (cols) — those remain as the reference
// implementations (B10_CE_CPP=0).
// ------------------------------------------------------------------

static inline void _ce_issue(int64_t dst, int64_t dpitch, int64_t src,
                             int64_t spitch, int64_t width, int64_t height,
                             int64_t lo, int64_t hi)
{
    if (dst >= lo && dst < hi)
        ce_local2d(dst, dpitch, src, spitch, width, height);
    else
        ce_copy2d(dst, dpitch, src, spitch, width, height);
}

// all-to-all half of a row RS: push row-block r of [B, W] to rank r.
void ce_push_rows(torch::Tensor ptrs_cpu, torch::Tensor x,
                  int64_t data_off, int64_t slot_bytes,
                  int64_t rank, int64_t world)
{
    TORCH_CHECK(x.is_contiguous() && x.dim() == 2 &&
                x.size(0) % world == 0, "push_rows: [B,W] contiguous, B%world==0");
    const int64_t chunk = x.size(0) / world;
    const int64_t slab = chunk * x.size(1) * (int64_t)x.element_size();
    TORCH_CHECK(world * slab <= slot_bytes, "CE instance too small");
    const int64_t *p = ptrs_cpu.data_ptr<int64_t>();
    const int64_t lo = p[rank], hi = lo + data_off + slot_bytes;
    const int64_t src0 = (int64_t)(intptr_t)x.data_ptr();
    for (int64_t i = 0; i < world; ++i) {
        const int64_t r = (rank + 1 + i) % world;  // self last
        _ce_issue(p[r] + data_off + rank * slab, slab,
                  src0 + r * slab, slab, slab, 1, lo, hi);
    }
}

// sync + fp32-accumulated sum of the world slabs -> my [chunk, cols].
torch::Tensor ce_reduce_rows(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                             torch::Tensor round_buf, int64_t data_off,
                             int64_t rank, int64_t world,
                             int64_t sync_threads,
                             int64_t chunk, int64_t cols)
{
    ce_sync(buf_ptrs, round_buf, 0, rank, world, sync_threads);
    const int64_t *p = ptrs_cpu.data_ptr<int64_t>();
    auto opts = torch::TensorOptions()
        .dtype(torch::kBFloat16).device(buf_ptrs.device());
    auto slabs = torch::from_blob(
        reinterpret_cast<void *>((intptr_t)(p[rank] + data_off)),
        {world, chunk, cols}, opts);
    return slabs.sum(0);
}

// row AG: my [B/world, W] slice -> full [B, W] view of my slot.
torch::Tensor ce_ag_rows(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                         torch::Tensor round_buf, torch::Tensor x,
                         int64_t data_off, int64_t slot_bytes,
                         int64_t rank, int64_t world, int64_t sync_threads)
{
    TORCH_CHECK(x.is_contiguous() && x.dim() == 2, "ag_rows: [rows,W] contiguous");
    const int64_t rows = x.size(0), cols = x.size(1);
    const int64_t slab = rows * cols * (int64_t)x.element_size();
    TORCH_CHECK(world * slab <= slot_bytes, "CE instance too small");
    const int64_t *p = ptrs_cpu.data_ptr<int64_t>();
    const int64_t lo = p[rank], hi = lo + data_off + slot_bytes;
    const int64_t src0 = (int64_t)(intptr_t)x.data_ptr();
    for (int64_t i = 0; i < world; ++i) {
        const int64_t r = (rank + 1 + i) % world;
        _ce_issue(p[r] + data_off + rank * slab, slab,
                  src0, slab, slab, 1, lo, hi);
    }
    ce_sync(buf_ptrs, round_buf, 0, rank, world, sync_threads);
    auto opts = torch::TensorOptions()
        .dtype(x.scalar_type()).device(x.device());
    return torch::from_blob(
        reinterpret_cast<void *>((intptr_t)(p[rank] + data_off)),
        {world * rows, cols}, opts);
}

// column AG: [rows, C] shards (possibly row-strided) -> [rows, world*C].
torch::Tensor ce_ag_cols(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                         torch::Tensor round_buf, torch::Tensor x,
                         int64_t data_off, int64_t slot_bytes,
                         int64_t rank, int64_t world, int64_t sync_threads)
{
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "ag_cols: [rows,C], inner-contig");
    const int64_t rows = x.size(0), cols = x.size(1);
    const int64_t esz = (int64_t)x.element_size();
    const int64_t row_bytes = cols * esz;
    const int64_t dpitch = world * row_bytes;
    TORCH_CHECK(rows * dpitch <= slot_bytes, "CE instance too small");
    const int64_t *p = ptrs_cpu.data_ptr<int64_t>();
    const int64_t lo = p[rank], hi = lo + data_off + slot_bytes;
    const int64_t src0 = (int64_t)(intptr_t)x.data_ptr();
    const int64_t spitch = x.stride(0) * esz;
    for (int64_t i = 0; i < world; ++i) {
        const int64_t r = (rank + 1 + i) % world;
        _ce_issue(p[r] + data_off + rank * row_bytes, dpitch,
                  src0, spitch, row_bytes, rows, lo, hi);
    }
    ce_sync(buf_ptrs, round_buf, 0, rank, world, sync_threads);
    auto opts = torch::TensorOptions()
        .dtype(x.scalar_type()).device(x.device());
    return torch::from_blob(
        reinterpret_cast<void *>((intptr_t)(p[rank] + data_off)),
        {rows, world * cols}, opts);
}
"""

_CPP_SRC = """
#include <torch/extension.h>
torch::Tensor ag_lamport(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block);
std::tuple<torch::Tensor, torch::Tensor> ag_mxfp8(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block);
torch::Tensor rs_lamport(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block);
torch::Tensor rs_cols(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world);
void ce_sync(torch::Tensor buf_ptrs, torch::Tensor round_buf,
             int64_t flag_off, int64_t rank, int64_t world,
             int64_t threads);
void ce_copy2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
               int64_t width, int64_t height);
void ce_local2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
                int64_t width, int64_t height);
void ce_copy2d_batch(torch::Tensor jobs, int64_t local_lo,
                     int64_t local_hi);
void ce_push_rows(torch::Tensor ptrs_cpu, torch::Tensor x,
                  int64_t data_off, int64_t slot_bytes,
                  int64_t rank, int64_t world);
torch::Tensor ce_reduce_rows(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                             torch::Tensor round_buf, int64_t data_off,
                             int64_t rank, int64_t world,
                             int64_t sync_threads,
                             int64_t chunk, int64_t cols);
torch::Tensor ce_ag_rows(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                         torch::Tensor round_buf, torch::Tensor x,
                         int64_t data_off, int64_t slot_bytes,
                         int64_t rank, int64_t world, int64_t sync_threads);
torch::Tensor ce_ag_cols(torch::Tensor ptrs_cpu, torch::Tensor buf_ptrs,
                         torch::Tensor round_buf, torch::Tensor x,
                         int64_t data_off, int64_t slot_bytes,
                         int64_t rank, int64_t world, int64_t sync_threads);
"""

_mod = None


def get_module():
    """Compile on first use (file-locked, safe under torchrun)."""
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        from common import arch

        # `setdefault` so a build-time prebuild can pin the target without a
        # live device (arch.torch_arch_list() reads B10_FORCE_SM first, and the
        # eager get_device_capability() call this replaced made GPU-less
        # prebuilds impossible).
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch.torch_arch_list())
        _mod = load_inline(
            name=arch.ext_name("k3_comm_cuda"),
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["ag_lamport", "ag_mxfp8",
                       "rs_lamport", "rs_cols",
                       "ce_sync", "ce_copy2d", "ce_local2d",
                       "ce_copy2d_batch", "ce_push_rows",
                       "ce_reduce_rows", "ce_ag_rows", "ce_ag_cols"],
            extra_cuda_cflags=["-O3"],
            extra_include_paths=[],
            verbose=False,
        )
    return _mod
