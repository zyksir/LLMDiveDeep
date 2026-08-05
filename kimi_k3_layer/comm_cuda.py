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

// ------------------------------------------------------------------
// All-gather with the quantize done SENDER-side: each rank MXFP8-
// quantizes its own [rows, cols] shard once (vs every rank quantizing
// the full gathered matrix) and the wire carries e4m3 + ue8m0, HALF
// the bytes of the bf16 transport. Lamport data-is-flag survives
// because the recipe is satfinite: e4m3 output can never be the NaN
// byte 0xFF (negative overflow clamps to 0xFE = -448) and ue8m0
// scales clamp to 0xFE, so 0xFF is a safe sentinel for BOTH regions.
// Slot layout: [payload rows x world*cols e4m3][pad16][scales
// rows x world*cols/32 ue8m0]. The instance must be dedicated and
// initialized with the 0xFF fill (OneShotComm(wire="fp8")): bf16-
// sentinel clears (0x00/0x80 bytes) would read as arrived data.
// NaN activations would break the sentinel - upstream compute never
// produces them (same class of assumption as -0.0 for bf16).
// ------------------------------------------------------------------

__device__ __forceinline__ bool has_ff(uint4 v) {
    return (__vcmpeq4(v.x, 0xFFFFFFFFu) | __vcmpeq4(v.y, 0xFFFFFFFFu) |
            __vcmpeq4(v.z, 0xFFFFFFFFu) | __vcmpeq4(v.w, 0xFFFFFFFFu))
           != 0u;
}

__device__ __forceinline__ void st_volatile_u8(uint8_t *p, uint32_t v) {
    asm volatile("st.volatile.global.u8 [%0], %1;" :: "l"(p), "r"(v));
}

__device__ __forceinline__ uint32_t ld_volatile_u8(const uint8_t *p) {
    uint32_t v;
    asm volatile("ld.volatile.global.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template <int WORLD>
__global__ void ag_qpush_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ rounds,
    int *__restrict__ clear_hist,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    uint8_t *__restrict__ sf,
    int rows, int nblk_row, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes, int64_t sf_off,
    int64_t write_v, int rank)
{
    GRID_DEP_SYNC();
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
    const int64_t nblk_my = (int64_t)rows * nblk_row;
    const int out_row_v = WORLD * nblk_row * 2;  // uint4s per out row

    // 1) quantize MY shard one 32-elt block per thread, push the 32
    //    e4m3 bytes (2 vectors) + 1 scale byte to every peer
    for (int64_t b = tid; b < nblk_my; b += nthr) {
        const int row = b / nblk_row;
        const int cb = b - (int64_t)row * nblk_row;
        const uint4 *src = x + (int64_t)row * x_stride_v + cb * 4;
        float2 f[16];
        float vmax = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 v = src[j];
            const uint32_t *w = &v.x;
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
        const float scale =
            (vmax != 0.f) ? rcp_ftz(static_cast<float>(t)) : 0.0f;
        union { uint4 q[2]; uint64_t d[4]; } u;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 qf[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qf[k].x = f[j * 4 + k].x * scale;
                qf[k].y = f[j * 4 + k].y * scale;
            }
            u.d[j] = e4m3x8(qf);
        }
        const int64_t dst_v = (int64_t)row * out_row_v
                              + ((int64_t)rank * nblk_row + cb) * 2;
        const int64_t dst_s = (int64_t)row * (WORLD * nblk_row)
                              + (int64_t)rank * nblk_row + cb;
        #pragma unroll
        for (int r = 0; r < WORLD; ++r) {
            char *slot = (char *)buf_ptrs[r] + off;
            uint4 *pay = reinterpret_cast<uint4 *>(slot);
            st_volatile_v4(pay + dst_v, u.q[0]);
            st_volatile_v4(pay + dst_v + 1, u.q[1]);
            st_volatile_u8(reinterpret_cast<uint8_t *>(slot + sf_off)
                           + dst_s, (uint32_t)t.__x);
        }
    }

    // 2) re-sentinel (0xFF fill) the slot consumed last call
    uint4 *mine_c = reinterpret_cast<uint4 *>(
        (char *)buf_ptrs[rank] + off_c);
    const uint4 sent = {0xFFFFFFFFu, 0xFFFFFFFFu,
                        0xFFFFFFFFu, 0xFFFFFFFFu};
    for (int64_t i = tid; i < clear_v; i += nthr)
        st_volatile_v4(mine_c + i, sent);

    // 3) poll + copy payload (vectorized), then scales (bytes)
    char *slot = (char *)buf_ptrs[rank] + off;
    uint4 *pay = reinterpret_cast<uint4 *>(slot);
    const int64_t total_v = (int64_t)rows * out_row_v;
    for (int64_t i = tid; i < total_v; i += nthr) {
        uint4 v;
        do { v = ld_volatile_v4(pay + i); } while (has_ff(v));
        out[i] = v;
    }
    uint8_t *ss = reinterpret_cast<uint8_t *>(slot + sf_off);
    const int64_t total_s = (int64_t)rows * WORLD * nblk_row;
    for (int64_t i = tid; i < total_s; i += nthr) {
        uint32_t v;
        do { v = ld_volatile_u8(ss + i); } while (v == 0xFFu);
        sf[i] = (uint8_t)v;
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

std::tuple<torch::Tensor, torch::Tensor> ag_qpush(
    torch::Tensor buf_ptrs, torch::Tensor rounds,
    torch::Tensor clear_hist, torch::Tensor x,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t block)
{
    const int rows = x.size(0), cols = x.size(1);
    const int64_t stride = x.stride(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(cols % 32 == 0, "sender-side blocks need cols % 32");
    TORCH_CHECK(stride % 8 == 0, "need 16B-aligned rows");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    TORCH_CHECK(clear_hist.numel() == rounds.numel() * 3);
    const int64_t pay_bytes = (int64_t)rows * world * cols;
    const int64_t sf_off = (pay_bytes + 15) / 16 * 16;
    const int64_t sf_bytes = pay_bytes / 32;
    TORCH_CHECK(sf_off + sf_bytes <= slot_bytes, "instance too small");
    const int64_t write_v = sf_off / 16 + (sf_bytes + 15) / 16;
    auto out = torch::empty({rows, world * cols},
                            x.options().dtype(torch::kFloat8_e4m3fn));
    auto sf = torch::empty({rows, world * cols / 32},
                           x.options().dtype(torch::kUInt8));
    DISPATCH_WORLD(world,
        launch_pdl(ag_qpush_lamport_kernel<kWorld>,
            (int)rounds.numel(), (int)block,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            clear_hist.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            reinterpret_cast<uint8_t *>(sf.data_ptr()),
            rows, (int)(cols / 32), stride / 8,
            data_off, slot_bytes, sf_off, write_v, (int)rank));
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
// Optional fused RMSNorm: one block owns one token end-to-end so the
// chunk sumsq can block-reduce, then a tiny second Lamport exchange
// assembles the full-row sumsq (world x rows fp32).
template <int WORLD>
__global__ void rs_cols_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ meta,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    const uint4 *__restrict__ norm_w,
    float eps, float inv_n,
    int rows, int chunk_v, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes, int64_t scal_off,
    int rank, int defer)
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
        float sumsq = 0.f;
        // per-thread register cache of the reduced vectors, so the
        // norm scale pass does not re-read `out` from global memory
        // (at decode shapes chunk_v <= blockDim -> one slot suffices)
        uint4 kept[2];
        int nk = 0;
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
                if (norm_w != nullptr) {
                    const float2 f = __bfloat1622float2(b);
                    sumsq += f.x * f.x + f.y * f.y;
                }
                ow[j] = *reinterpret_cast<const uint32_t *>(&b);
            }
            if (nk < 2) kept[nk++] = o;
            out[idx] = o;
        }
        if (norm_w == nullptr)
            continue;

        // block-sum the per-thread chunk sumsq: warp shuffle tree +
        // one cross-warp pass (token owned by this CTA)
        __shared__ float s_warp[9];
        float wsum = sumsq;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            wsum += __shfl_down_sync(0xffffffffu, wsum, o);
        if ((threadIdx.x & 31) == 0)
            s_warp[threadIdx.x >> 5] = wsum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float bs = 0.f;
            const int nwarp = ((int)blockDim.x + 31) >> 5;
            for (int i = 0; i < nwarp; ++i) bs += s_warp[i];
            s_warp[8] = bs;
        }
        __syncthreads();
        const float block_sum = s_warp[8];

        // stage-2 Lamport: exchange per-token chunk sumsq
        if (threadIdx.x < WORLD) {
            volatile float *peer_s = reinterpret_cast<volatile float *>(
                (char *)buf_ptrs[threadIdx.x] + off + scal_off);
            peer_s[(int64_t)rank * rows + t] = block_sum;
        }
        // deferred mode: partials are pushed but NOT awaited here -
        // rs_cols_scale_kernel (or a consumer that folds 1/rms in)
        // polls them off the critical path
        if (defer)
            continue;
        float part = 0.f;
        if (threadIdx.x < WORLD) {
            volatile uint32_t *ss =
                reinterpret_cast<volatile uint32_t *>(
                    (char *)buf_ptrs[rank] + off + scal_off);
            uint32_t w;
            do { w = ss[(int64_t)threadIdx.x * rows + t]; }
            while (w == SENT2);
            part = __uint_as_float(w);
        }
        __shared__ float s_parts[8];
        if (threadIdx.x < WORLD) s_parts[threadIdx.x] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.f;
            #pragma unroll
            for (int r = 0; r < WORLD; ++r) total += s_parts[r];
            s_parts[0] = total;
        }
        __syncthreads();
        const float scale = rsqrtf(s_parts[0] * inv_n + eps);

        int k = 0;
        for (int c = (int)threadIdx.x; c < chunk_v; c += (int)blockDim.x) {
            const int64_t idx = (int64_t)t * chunk_v + c;
            const uint4 o = (k < nk) ? kept[k] : out[idx];
            ++k;
            const uint4 w4 = norm_w[c];
            const uint32_t *ov = &o.x, *wv = &w4.x;
            uint4 n;
            uint32_t *nv = &n.x;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&ov[j]));
                const float2 g = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&wv[j]));
                const float2 rr = {f.x * scale * g.x, f.y * scale * g.y};
                const __nv_bfloat162 b = __float22bfloat162_rn(rr);
                nv[j] = *reinterpret_cast<const uint32_t *>(&b);
            }
            out[idx] = n;
        }
    }

    // FI-style flag advance: block 0 waits until every CTA entered
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (meta[1] != (int)gridDim.x) {}
        meta[0] = (flag + 1) % 3;
        // next clear covers data rows (+ sumsq region when norming)
        const int64_t used = (norm_w != nullptr)
            ? scal_off + (int64_t)WORLD * rows * 4
            : (int64_t)WORLD * tot * 16;
        meta[2] = (int)used;
        meta[1] = 0;
    }
    TRIGGER_PDL();
}

// Deferred-norm follow-up for rs_cols(defer=1): polls the per-token
// sumsq partials the RS kernel pushed (they arrive during this
// kernel's launch gap, so the poll is ~free) and applies the 1/rms
// scale. Reads the slot of the JUST-COMPLETED rs_cols call:
// meta[0] was already advanced, so that slot is (meta[0]+2)%3.
template <int WORLD>
__global__ void rs_cols_scale_kernel(
    const int64_t *__restrict__ buf_ptrs, const int *__restrict__ meta,
    uint4 *__restrict__ out, const uint4 *__restrict__ norm_w,
    float eps, float inv_n, int rows, int chunk_v,
    int64_t data_off, int64_t slot_bytes, int64_t scal_off, int rank)
{
    GRID_DEP_SYNC();
    __shared__ int s_flag;
    if (threadIdx.x == 0)
        s_flag = (meta[0] + 2) % 3;
    __syncthreads();
    const int64_t off = data_off + (int64_t)s_flag * slot_bytes;
    __shared__ float s_parts[WORLD];
    for (int t = (int)blockIdx.x; t < rows; t += (int)gridDim.x) {
        float part = 0.f;
        if (threadIdx.x < WORLD) {
            volatile uint32_t *ss = reinterpret_cast<volatile uint32_t *>(
                (char *)buf_ptrs[rank] + off + scal_off);
            uint32_t w;
            do { w = ss[(int64_t)threadIdx.x * rows + t]; }
            while (w == SENT2);
            part = __uint_as_float(w);
        }
        if (threadIdx.x < WORLD) s_parts[threadIdx.x] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.f;
            #pragma unroll
            for (int r = 0; r < WORLD; ++r) total += s_parts[r];
            s_parts[0] = total;
        }
        __syncthreads();
        const float scale = rsqrtf(s_parts[0] * inv_n + eps);
        for (int c = (int)threadIdx.x; c < chunk_v; c += (int)blockDim.x) {
            const int64_t idx = (int64_t)t * chunk_v + c;
            const uint4 o = out[idx];
            const uint4 w4 = norm_w[c];
            const uint32_t *ov = &o.x, *wv = &w4.x;
            uint4 n;
            uint32_t *nv = &n.x;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&ov[j]));
                const float2 g = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&wv[j]));
                const float2 rr = {f.x * scale * g.x, f.y * scale * g.y};
                const __nv_bfloat162 b = __float22bfloat162_rn(rr);
                nv[j] = *reinterpret_cast<const uint32_t *>(&b);
            }
            out[idx] = n;
        }
        __syncthreads();
    }
    TRIGGER_PDL();
}

// ROW reduce-scatter with fused RMSNorm: each rank ends up with
// rows/world COMPLETE output rows, so the norm sumsq needs no
// cross-RANK exchange (the structural cost rs_cols+norm pays) - only
// a device-local cross-CTA reduction over the row's column segments
// (atomics in L2, ~1us). Grid = chunk_rows x nseg CTAs, CTA
// (t, seg) owns column segment seg of output row t; the push phase is
// grid-strided over everything so small chunk_rows still gets full
// parallelism. scr/cnt are rows-sized fp32/int32 scratch, zeroed by
// the kernel itself after use (graph-replay safe).
// Slot layout: [WORLD src ranks][chunk_rows][row_v] uint4; same
// 3-slot flag rotation via meta as rs_cols. Requires rows % world == 0.
template <int WORLD>
__global__ void rs_rows_lamport_kernel(
    const int64_t *__restrict__ buf_ptrs, int *__restrict__ meta,
    const uint4 *__restrict__ x, uint4 *__restrict__ out,
    const uint4 *__restrict__ norm_w,   // full row (cols) or null
    float *__restrict__ scr, int *__restrict__ cnt,
    float eps, float inv_n,
    int chunk_rows, int row_v, int nseg, int64_t x_stride_v,
    int64_t data_off, int64_t slot_bytes, int rank)
{
    GRID_DEP_SYNC();
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
    const int64_t tot = (int64_t)chunk_rows * row_v;  // per-src block
    const int64_t clear_v = (int64_t)s_clear / 16;

    // 1) push: peer r gets my row block [r*chunk_rows, (r+1)*chunk_rows).
    // Peers are the INNER loop so every dest rank fills uniformly -
    // a peer-outer loop serializes whole blocks per dest and the last
    // dest's poll then waits out the entire push phase.
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < tot; i += (int64_t)gridDim.x * blockDim.x) {
        const int rr = (int)(i / row_v);
        const int cc = (int)(i % row_v);
        #pragma unroll
        for (int r = 0; r < WORLD; ++r) {
            uint4 v = x[(int64_t)(r * chunk_rows + rr) * x_stride_v + cc];
            v.x = canon(v.x); v.y = canon(v.y);
            v.z = canon(v.z); v.w = canon(v.w);
            uint4 *peer = reinterpret_cast<uint4 *>(
                (char *)buf_ptrs[r] + off) + (int64_t)rank * tot;
            st_volatile_v4(peer + i, v);
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

    // 3) CTA (t, seg): poll+reduce my column segment of output row t
    const int t = (int)blockIdx.x % chunk_rows;
    const int seg = (int)blockIdx.x / chunk_rows;
    const int seg_v = (row_v + nseg - 1) / nseg;
    const int c_lo = seg * seg_v;
    const int c_hi = min(row_v, c_lo + seg_v);
    float sumsq = 0.f;
    uint4 kept[2];
    int nk = 0;
    for (int c = c_lo + (int)threadIdx.x; c < c_hi;
         c += (int)blockDim.x) {
        const int64_t idx = (int64_t)t * row_v + c;
        uint4 vals[WORLD];
        bool done = false;
        while (!done) {
            done = true;
            #pragma unroll
            for (int r = 0; r < WORLD; ++r) {
                vals[r] = ld_volatile_v4(mine + (int64_t)r * tot + idx);
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
            if (norm_w != nullptr) {
                const float2 f = __bfloat1622float2(b);
                sumsq += f.x * f.x + f.y * f.y;
            }
            ow[j] = *reinterpret_cast<const uint32_t *>(&b);
        }
        if (nk < 2) kept[nk++] = o;
        out[idx] = o;
    }

    if (norm_w != nullptr) {
        // segment sumsq: warp tree + cross-warp
        __shared__ float s_warp[9];
        float wsum = sumsq;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            wsum += __shfl_down_sync(0xffffffffu, wsum, o);
        if ((threadIdx.x & 31) == 0)
            s_warp[threadIdx.x >> 5] = wsum;
        __syncthreads();
        // cross-CTA (device-local): accumulate into scr[t], arrive on
        // cnt[t], spin until all nseg segments of this row arrived
        if (threadIdx.x == 0) {
            float bs = 0.f;
            const int nwarp = ((int)blockDim.x + 31) >> 5;
            for (int i = 0; i < nwarp; ++i) bs += s_warp[i];
            atomicAdd(scr + t, bs);
            __threadfence();
            atomicAdd(cnt + t, 1);
            volatile int *vc = cnt + t;
            while (*vc < nseg) {}
            s_warp[8] = *reinterpret_cast<volatile float *>(scr + t);
        }
        __syncthreads();
        const float scale = rsqrtf(s_warp[8] * inv_n + eps);

        int k = 0;
        for (int c = c_lo + (int)threadIdx.x; c < c_hi;
             c += (int)blockDim.x) {
            const int64_t idx = (int64_t)t * row_v + c;
            const uint4 o = (k < nk) ? kept[k] : out[idx];
            ++k;
            const uint4 w4 = norm_w[c];
            const uint32_t *ov = &o.x, *wv = &w4.x;
            uint4 n;
            uint32_t *nv = &n.x;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&ov[j]));
                const float2 g = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162 *>(&wv[j]));
                const float2 rr = {f.x * scale * g.x, f.y * scale * g.y};
                const __nv_bfloat162 b = __float22bfloat162_rn(rr);
                nv[j] = *reinterpret_cast<const uint32_t *>(&b);
            }
            out[idx] = n;
        }
        // second arrival round; the last CTA of the row resets the
        // scratch for the next call / graph replay
        if (threadIdx.x == 0) {
            __threadfence();
            const int done = atomicAdd(cnt + t, 1) + 1;
            if (done == 2 * nseg) {
                scr[t] = 0.f;
                __threadfence();
                cnt[t] = 0;
            }
        }
    }

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
    torch::Tensor norm_w, double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t defer)
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
    const int64_t scal_off = (numel_v * world * 16 + 255) / 256 * 256;
    TORCH_CHECK(scal_off + (int64_t)world * rows * 4 <= slot_bytes,
                "slot too small for data + sumsq scalars");
    const bool use_norm = norm_w.numel() > 0;
    if (use_norm)
        TORCH_CHECK(norm_w.is_contiguous()
                    && norm_w.numel() == chunk_cols);
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
            use_norm
                ? reinterpret_cast<const uint4 *>(norm_w.data_ptr())
                : nullptr,
            (float)eps, 1.0f / (float)cols,
            rows, chunk_v, stride / 8,
            data_off, slot_bytes, scal_off, (int)rank,
            (int)(defer != 0)));
    return out;
}

torch::Tensor rs_cols_scale(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor out,
    torch::Tensor norm_w, double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world)
{
    const int rows = out.size(0), chunk_cols = out.size(1);
    TORCH_CHECK(out.is_contiguous()
                && out.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(norm_w.is_contiguous()
                && norm_w.numel() == chunk_cols);
    const int chunk_v = chunk_cols / 8;
    const int64_t numel_v = (int64_t)rows * chunk_v;
    const int64_t scal_off = (numel_v * world * 16 + 255) / 256 * 256;
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount,
                           out.get_device());
    if (sm < 1) sm = 1;
    int grid = rows < sm ? rows : sm;
    if (grid < 1) grid = 1;
    DISPATCH_WORLD(world,
        launch_pdl(rs_cols_scale_kernel<kWorld>, grid, 256,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            reinterpret_cast<const uint4 *>(norm_w.data_ptr()),
            (float)eps, 1.0f / (float)(chunk_cols * world),
            rows, chunk_v, data_off, slot_bytes, scal_off, (int)rank));
    return out;
}

torch::Tensor rs_rows(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor x,
    torch::Tensor norm_w, torch::Tensor scr, torch::Tensor cnt,
    double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world)
{
    const int rows = x.size(0), cols = x.size(1);
    const int64_t stride = x.stride(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(rows % world == 0, "rows must divide by world");
    TORCH_CHECK(cols % 8 == 0 && stride % 8 == 0,
                "need 16B-aligned rows");
    TORCH_CHECK(((uintptr_t)x.data_ptr()) % 16 == 0, "unaligned base");
    const int chunk_rows = rows / (int)world;
    const int row_v = cols / 8;
    TORCH_CHECK((int64_t)chunk_rows * row_v * world * 16 <= slot_bytes,
                "slot too small");
    const bool use_norm = norm_w.numel() > 0;
    if (use_norm)
        TORCH_CHECK(norm_w.is_contiguous() && norm_w.numel() == cols);
    TORCH_CHECK(scr.numel() >= chunk_rows && cnt.numel() >= chunk_rows);
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount,
                           x.get_device());
    if (sm < 1) sm = 1;
    // column segments per row: enough CTAs to saturate at small B, but
    // grid = chunk_rows*nseg must stay co-resident (spin sync)
    int nseg = sm / chunk_rows;
    if (nseg > 8) nseg = 8;
    if (nseg < 1) nseg = 1;
    const int grid = chunk_rows * nseg;
    auto out = torch::empty({chunk_rows, cols}, x.options());
    DISPATCH_WORLD(world,
        launch_pdl(rs_rows_lamport_kernel<kWorld>, grid, 256,
            buf_ptrs.data_ptr<int64_t>(), rounds.data_ptr<int>(),
            reinterpret_cast<const uint4 *>(x.data_ptr()),
            reinterpret_cast<uint4 *>(out.data_ptr()),
            use_norm
                ? reinterpret_cast<const uint4 *>(norm_w.data_ptr())
                : nullptr,
            scr.data_ptr<float>(), cnt.data_ptr<int>(),
            (float)eps, 1.0f / (float)cols,
            chunk_rows, row_v, nseg, stride / 8,
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
std::tuple<torch::Tensor, torch::Tensor> ag_qpush(
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
    torch::Tensor norm_w, double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world,
    int64_t defer);
torch::Tensor rs_cols_scale(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor out,
    torch::Tensor norm_w, double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world);
torch::Tensor rs_rows(
    torch::Tensor buf_ptrs, torch::Tensor rounds, torch::Tensor x,
    torch::Tensor norm_w, torch::Tensor scr, torch::Tensor cnt,
    double eps,
    int64_t data_off, int64_t slot_bytes, int64_t rank, int64_t world);
void ce_sync(torch::Tensor buf_ptrs, torch::Tensor round_buf,
             int64_t flag_off, int64_t rank, int64_t world,
             int64_t threads);
void ce_copy2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
               int64_t width, int64_t height);
void ce_local2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch,
                int64_t width, int64_t height);
"""

_mod = None


def get_module():
    """Compile on first use (file-locked, safe under torchrun)."""
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST",
                              f"{torch.cuda.get_device_capability()[0]}."
                              f"{torch.cuda.get_device_capability()[1]}")
        _mod = load_inline(
            name="k3_comm_cuda",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["ag_lamport", "ag_mxfp8", "ag_qpush",
                       "rs_lamport", "rs_cols",
                       "rs_cols_scale", "rs_rows",
                       "ce_sync", "ce_copy2d", "ce_local2d"],
            extra_cuda_cflags=["-O3"],
            extra_include_paths=[],
            verbose=False,
        )
    return _mod
