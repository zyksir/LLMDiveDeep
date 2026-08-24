"""Fused MoE finalize + one-shot AR + RMSNorm over the LATENT only.

The stock kernel (moefinalize_allreduce_fusion, f17ea3ab32) fuses
finalize+AR+norm but ARs concat([latent|hidden]) and forces the full
fc2_latent_proj afterwards. This kernel ARs ONLY the latent (3584), so
the sharded-fc2 tail (3 us vs 27 us at B=8) keeps working — the missing
piece identified in communication/MOE_FINALIZE_AR.md.

v2 (data-encoded lamport, stock's own protocol):
  - The payload IS the arrival signal: buffers are pre-filled with bf16
    negative zero (0x8000); writers canonicalize -0.0 -> +0.0 so real
    data never contains the sentinel; readers spin per-16B-vector until
    no lane reads 0x8000. No flag words, no __threadfence_system, no
    trailing bump kernel — aligned 16B NVLink stores are single
    transactions, so detection needs no cross-address ordering.
  - 3 rotating slots: round r writes slot r%3; after phase B consumes a
    slot the kernel clears exactly the rows it read back to -0.0 for
    round r+3 (stream order guarantees no peer of round r+3 can write
    before our r+1 push, which follows this kernel's completion).
  - The spin is distributed across all threads (each spins on the
    vectors it will reduce), not serialized on tid 0.
  - The round counter is bumped in-kernel by the last block to finish
    (every block has read the old round before any block can finish),
    so the whole tail is ONE launch and CUDA-graph-safe.

Structure:
  phase A  per-token block: gather this rank's top-k expert rows from the
           unfinalized fc2 output, weighted-sum in fp32, push the bf16
           partial (canonicalized) to EVERY rank's symm slot.
           Then ``griddepcontrol.launch_dependents`` — the next kernel
           (fc2-shard GEMM) launches and prefetches weights during our
           spin (the stock kernel's PDL overlap, reproduced).
  phase B  per-vector lamport spin on all world partials for this token,
           sum in fp32, RMSNorm the row, write [B, LATENT] output, then
           clear the consumed slot rows.

``griddepcontrol.wait`` at entry lets the kernel launch under the expert
GEMM that produces fc2_output.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#define THREADS 512
#define NEG_ZERO_U32 0x80008000u

__device__ __forceinline__ uint4 ld_volatile_v4(uint4 const* p) {
    uint4 q;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
                 : "l"(p) : "memory");
    return q;
}

__device__ __forceinline__ bool has_neg_zero(uint4 const& q) {
    // any of the 8 bf16 lanes == 0x8000 (the not-yet-arrived sentinel)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t const w = (&q.x)[j];
        if ((w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u) return true;
    }
    return false;
}

__global__ void __launch_bounds__(THREADS) finalize_ar_norm_kernel(
    __nv_bfloat16 const* __restrict__ fc2,   // [n_rows, latent]
    int32_t const* __restrict__ idx,          // [B, topk] permuted rows
    float const* __restrict__ scales,         // [B, topk]
    __nv_bfloat16 const* __restrict__ normw,  // [latent]
    __nv_bfloat16* __restrict__ out,          // [B, latent]
    int64_t const* __restrict__ peer_ptrs,    // world data-region ptrs
    int32_t* __restrict__ round_buf,          // [round, done_count]
    int32_t B, int32_t topk, int32_t latent,
    int32_t rank, int32_t world, int64_t slot_stride, int64_t round_bytes,
    float eps, int32_t skip_wait, int32_t use_pdl, int32_t stage,
    int32_t n_rows)
{
#if __CUDA_ARCH__ >= 900
    // griddepcontrol is only defined for PDL-attributed launches
    if (use_pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    int const b = blockIdx.x;
    int const tid = threadIdx.x;
    if (stage >= 9 && b == 0 && tid == 0) printf("[fan] enter rank=%d\n", rank);
    int const round = *round_buf;
    int64_t const slot_off = (int64_t)(round % 3) * round_bytes;
    if (stage == 1) return;

    // ---- phase A: finalize this rank's partial and push to all ranks.
    // partial[c] = sum_k scales[b,k] * fc2[idx[b,k], c]
    extern __shared__ float acc[];  // [latent] fp32
    for (int c = tid; c < latent; c += THREADS) acc[c] = 0.f;
    __syncthreads();
    // k-outer: each expert row is read COALESCED and 16B-vectorized,
    // accumulated into the smem row (the scattered-per-column order was
    // ~2x the whole kernel's budget).
    int const nvec_a = latent / 8;
    for (int k = 0; k < topk; ++k) {
        int const row = idx[b * topk + k];
        // run_moe pads expanded_idx with invalid rows (dropped slots)
        if (row < 0 || row >= n_rows) continue;
        float const w = scales[b * topk + k];
        uint4 const* rowv = reinterpret_cast<uint4 const*>(
            fc2 + (int64_t)row * latent);
        for (int v = tid; v < nvec_a; v += THREADS) {
            uint4 const q = rowv[v];
            __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&q);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[v * 8 + j] += w * __bfloat162float(e[j]);
        }
        // no barrier: each thread owns disjoint acc slots across k
    }
    __syncthreads();
    // pack the bf16 partial once in smem (canonicalized: -0.0 -> +0.0 so
    // the payload never contains the lamport sentinel), then push
    // 16B-vectorized to every rank's slot [rank][b] (self included)
    __shared__ uint16_t packed_row[8192];
    for (int c = tid; c < latent; c += THREADS) {
        uint16_t v = __bfloat16_as_ushort(__float2bfloat16(acc[c]));
        packed_row[c] = (v == 0x8000u) ? 0u : v;
    }
    __syncthreads();
    int const nvec = latent / 8;  // latent % 8 == 0 (3584)
    uint4 const* src_v = reinterpret_cast<uint4 const*>(packed_row);
    for (int p = 0; p < world; ++p) {
        int const pp = (p + rank) % world;  // stagger peers across ranks
        uint4* dst = reinterpret_cast<uint4*>(
            peer_ptrs[pp] + slot_off
            + ((int64_t)rank * B + b) * slot_stride);
        for (int v = tid; v < nvec; v += THREADS) {
            dst[v] = src_v[v];
        }
    }
    if (stage == 2) return;
#if __CUDA_ARCH__ >= 900
    // let the NEXT kernel (fc2-shard GEMM) launch during our spin
    if (use_pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

    if (stage == 3) return;
    // ---- phase B: lamport spin + reduce + norm, distributed over threads.
    // Each thread spins on exactly the vectors it reduces; arrival is
    // encoded in the data (no lane == -0.0), so no flags and no fences.
    __shared__ float ssq_sh;
    float ssq = 0.f;
    for (int v = tid; v < nvec; v += THREADS) {
        float col[8] = {0.f};
        for (int p = 0; p < world; ++p) {
            uint4 const* src = reinterpret_cast<uint4 const*>(
                peer_ptrs[rank] + slot_off
                + ((int64_t)p * B + b) * slot_stride) + v;
            uint4 q = ld_volatile_v4(src);
            if (!skip_wait) {
                while (has_neg_zero(q)) q = ld_volatile_v4(src);
            }
            __nv_bfloat16 const* e =
                reinterpret_cast<__nv_bfloat16 const*>(&q);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                col[j] += __bfloat162float(e[j]);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[v * 8 + j] = col[j];
            ssq += col[j] * col[j];
        }
    }
    // block reduction of sum-of-squares
    __shared__ float red[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1)
        ssq += __shfl_down_sync(0xffffffffu, ssq, o);
    if ((tid & 31) == 0) red[tid >> 5] = ssq;
    __syncthreads();
    if (tid < THREADS / 32) {
        ssq = red[tid];
        // mask covers ONLY the 16 active lanes — a full mask deadlocks
        for (int o = THREADS / 64; o > 0; o >>= 1)
            ssq += __shfl_down_sync(0x0000ffffu, ssq, o);
        if (tid == 0) ssq_sh = ssq;
    }
    __syncthreads();
    float const inv = rsqrtf(ssq_sh / latent + eps);
    for (int c = tid; c < latent; c += THREADS) {
        out[(int64_t)b * latent + c] = __float2bfloat16(
            acc[c] * inv * __bfloat162float(normw[c]));
    }

    // ---- retire the consumed slot: clear exactly the rows this block
    // read back to the sentinel, ready for round+3. Safe: no round+3
    // writer can touch this slot before our round+1 push, which follows
    // this kernel in stream order.
    uint4 const sentinel = {NEG_ZERO_U32, NEG_ZERO_U32,
                            NEG_ZERO_U32, NEG_ZERO_U32};
    for (int p = 0; p < world; ++p) {
        uint4* row = reinterpret_cast<uint4*>(
            peer_ptrs[rank] + slot_off
            + ((int64_t)p * B + b) * slot_stride);
        for (int v = tid; v < nvec; v += THREADS) row[v] = sentinel;
    }
    // last block to finish bumps the round (every block already read the
    // old round before any block can reach this point)
    __syncthreads();
    if (tid == 0) {
        int const done = atomicAdd(round_buf + 1, 1);
        if (done == gridDim.x - 1) {
            round_buf[1] = 0;
            *reinterpret_cast<volatile int32_t*>(round_buf) = round + 1;
        }
    }
}

// ---- PDL follower: sharded-fc2 GEMM whose weight tile is prefetched
// into smem BEFORE griddepcontrol.wait, so the whole 6.4MB weight read
// overlaps the producer's lamport spin (the stock tail's overlap trick,
// which a torch-launched addmm cannot do).
//
// out[b][h] = shared[b][h] + sum_c routed[b][col_off+c] * W[c][h]
// W: [C, H] row-major (the layer's _fc2_shard_t), C=latent/world.
#define FC2_TILE_H 64
#define FC2_THREADS 256
#define FC2_MAX_C 512
#define FC2_TILE_B 32

__global__ void __launch_bounds__(FC2_THREADS) fc2_shard_pdl_kernel(
    __nv_bfloat16 const* __restrict__ routed,  // [B, latent] normed
    __nv_bfloat16 const* __restrict__ shared,  // [B, H]
    __nv_bfloat16 const* __restrict__ W,       // [C, H]
    __nv_bfloat16* __restrict__ out,           // [B, H]
    int32_t B, int32_t latent, int32_t C, int32_t H,
    int32_t col_off, int32_t use_pdl)
{
    extern __shared__ unsigned char fc2_smem[];
    __nv_bfloat16* w_sh = reinterpret_cast<__nv_bfloat16*>(fc2_smem);
    float* r_sh = reinterpret_cast<float*>(
        fc2_smem + (size_t)C * FC2_TILE_H * sizeof(__nv_bfloat16));
    int const tid = threadIdx.x;
    int const h0 = blockIdx.x * FC2_TILE_H;
    // ---- independent prefetch (runs during the producer's spin)
    for (int i = tid; i < C * FC2_TILE_H; i += FC2_THREADS) {
        int const c = i / FC2_TILE_H, h = i % FC2_TILE_H;
        w_sh[c * FC2_TILE_H + h] = W[(int64_t)c * H + h0 + h];
    }
#if __CUDA_ARCH__ >= 900
    if (use_pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    bool first_tile = true;
    // token tiles of 32 keep smem under the 228KB cap at B up to 128
    for (int b0 = 0; b0 < B; b0 += FC2_TILE_B) {
        int const nb = min(FC2_TILE_B, B - b0);
        if (!first_tile) __syncthreads();  // r_sh reuse hazard
        // ---- dependent load: the producer's normed latent slice
        for (int i = tid; i < nb * C; i += FC2_THREADS) {
            int const b = i / C, c = i % C;
            r_sh[b * C + c] = __bfloat162float(
                routed[(int64_t)(b0 + b) * latent + col_off + c]);
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 900
        if (use_pdl && first_tile)
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
        first_tile = false;
        for (int i = tid; i < nb * FC2_TILE_H; i += FC2_THREADS) {
            int const b = i / FC2_TILE_H, h = i % FC2_TILE_H;
            float acc = __bfloat162float(
                shared[(int64_t)(b0 + b) * H + h0 + h]);
            #pragma unroll 4
            for (int c = 0; c < C; ++c)
                acc += r_sh[b * C + c]
                    * __bfloat162float(w_sh[c * FC2_TILE_H + h]);
            out[(int64_t)(b0 + b) * H + h0 + h] = __float2bfloat16(acc);
        }
    }
}

void run_fc2(int64_t routed, int64_t shared, int64_t w, int64_t out,
             int64_t B, int64_t latent, int64_t C, int64_t H,
             int64_t col_off)
{
    TORCH_CHECK(C <= FC2_MAX_C && H % FC2_TILE_H == 0,
                "fc2_shard_pdl shape limits");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    size_t const smem = (size_t)C * FC2_TILE_H * sizeof(__nv_bfloat16)
        + (size_t)min((int)B, FC2_TILE_B) * C * sizeof(float);
    static bool fc2_smem_ok = false;
    if (!fc2_smem_ok) {
        cudaFuncSetAttribute(fc2_shard_pdl_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)(FC2_MAX_C * FC2_TILE_H * 2 + FC2_TILE_B * FC2_MAX_C * 4));
        fc2_smem_ok = true;
    }
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    bool const use_pdl = getenv("K3_FAN_PDL") == nullptr
        || getenv("K3_FAN_PDL")[0] != '0';
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)(H / FC2_TILE_H), 1, 1);
    cfg.blockDim = dim3(FC2_THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = use_pdl ? 1 : 0;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, fc2_shard_pdl_kernel,
        reinterpret_cast<__nv_bfloat16 const*>(routed),
        reinterpret_cast<__nv_bfloat16 const*>(shared),
        reinterpret_cast<__nv_bfloat16 const*>(w),
        reinterpret_cast<__nv_bfloat16*>(out),
        (int32_t)B, (int32_t)latent, (int32_t)C, (int32_t)H,
        (int32_t)col_off, (int32_t)(use_pdl ? 1 : 0)) == cudaSuccess,
        "fc2_shard_pdl launch failed");
}

void run(int64_t fc2, int64_t idx, int64_t scales, int64_t normw,
         int64_t out, torch::Tensor peer_ptrs, torch::Tensor round_buf,
         int64_t B, int64_t topk, int64_t latent, int64_t rank,
         int64_t world, int64_t slot_stride, int64_t round_bytes, double eps,
         int64_t n_rows)

{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    size_t const smem = latent * sizeof(float);
    static bool smem_ok = false;
    if (!smem_ok) {
        cudaFuncSetAttribute(finalize_ar_norm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        smem_ok = true;
    }
    // PDL launch (programmatic stream serialization): lets this kernel
    // start under the producer GEMM (its griddepcontrol.wait gates the
    // data reads) and lets the follower launch during our lamport spin —
    // the same mechanism behind the stock tail's overlap.
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    bool const use_pdl = getenv("K3_FAN_PDL") == nullptr
        || getenv("K3_FAN_PDL")[0] != '0';
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)B, 1, 1);
    cfg.blockDim = dim3(THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = use_pdl ? 1 : 0;
    auto fc2p = reinterpret_cast<__nv_bfloat16 const*>(fc2);
    auto idxp = reinterpret_cast<int32_t const*>(idx);
    auto scp = reinterpret_cast<float const*>(scales);
    auto nwp = reinterpret_cast<__nv_bfloat16 const*>(normw);
    auto outp = reinterpret_cast<__nv_bfloat16*>(out);
    auto ptrs = reinterpret_cast<int64_t const*>(peer_ptrs.data_ptr<int64_t>());
    int32_t Bi = (int32_t)B, Ki = (int32_t)topk, Li = (int32_t)latent;
    int32_t Ri = (int32_t)rank, Wi = (int32_t)world;
    float ef = (float)eps;
    int32_t skipw = (getenv("K3_FAN_SKIP_WAIT") != nullptr
                     && getenv("K3_FAN_SKIP_WAIT")[0] == '1') ? 1 : 0;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, finalize_ar_norm_kernel,
        fc2p, idxp, scp, nwp, outp, ptrs,
        round_buf.data_ptr<int32_t>(), Bi, Ki, Li, Ri, Wi,
        slot_stride, round_bytes, ef, skipw,
        (int32_t)(use_pdl ? 1 : 0),
        (int32_t)(getenv("K3_FAN_STAGE") ? atoi(getenv("K3_FAN_STAGE")) : 0),
        (int32_t)n_rows) == cudaSuccess,
        "finalize_ar_norm launch failed");
}

"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        _MODULE = load_inline(
            name="k3_finalize_ar_norm_v2",
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "void run(int64_t, int64_t, int64_t, int64_t, int64_t, "
                "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, "
                "int64_t, int64_t, int64_t, int64_t, double, int64_t);\n"
                "void run_fc2(int64_t, int64_t, int64_t, int64_t, int64_t, "
                "int64_t, int64_t, int64_t, int64_t);"),
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            functions=["run", "run_fc2"],
        )
    return _MODULE


class FinalizeARNorm:
    """Latent-only fused finalize+AR+RMSNorm over a symm-memory buffer.

    Data-encoded lamport protocol (v2): 3 rotating rounds, payload
    doubles as the arrival signal, slot cleared in-kernel after use.
    ONE kernel launch per call, CUDA-graph safe (device round counter).

    NOTE: the slot layout is [round][writer_rank][token] with the token
    stride tied to the CURRENT batch (rows are packed by B), so all
    ranks must call with the same B — true everywhere in TP decode.
    """

    def __init__(self, group, rank: int, world: int, *,
                 max_tokens: int = 128, latent: int = 3584) -> None:
        self.rank, self.world = rank, world
        self.latent, self.max_tokens = latent, max_tokens
        self.slot_stride = latent * 2  # bf16 bytes, one row per slot
        self.round_bytes = world * max_tokens * self.slot_stride
        # allocation aligned with TRT-LLM's AllReduce workspace policy
        # (K3_PEER_ALLOC: ipc = trt IpcMemory, IMEX-extended cross-node;
        # symm = torch symmetric memory; auto = ipc then symm) — the
        # kernels only consume the pointer table, so the source is
        # swappable. See communication/kernels/peer_alloc.py.
        from communication.kernels.peer_alloc import alloc_peer_buffer
        self.buf, self.peer_ptrs, self._keepalive, self._alloc_kind = (
            alloc_peer_buffer(group, rank, world, 3 * self.round_bytes,
                              sentinel_i16=-32768))
        # [round, done_count]
        self.round_buf = torch.zeros(2, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()

    def __call__(self, fc2: torch.Tensor, idx: torch.Tensor,
                 scales: torch.Tensor, norm_weight: torch.Tensor,
                 eps: float) -> torch.Tensor:
        batch = idx.shape[0]
        assert batch <= self.max_tokens
        out = torch.empty(batch, self.latent, device=fc2.device,
                          dtype=torch.bfloat16)
        _module().run(
            fc2.data_ptr(), idx.data_ptr(), scales.data_ptr(),
            norm_weight.data_ptr(), out.data_ptr(), self.peer_ptrs,
            self.round_buf, batch, idx.shape[1], self.latent, self.rank,
            self.world, self.slot_stride, self.round_bytes, eps,
            fc2.shape[0])
        return out


def fc2_shard_pdl(routed: torch.Tensor, shared: torch.Tensor,
                  w_shard_t: torch.Tensor, col_off: int) -> torch.Tensor:
    """PDL follower GEMM: out = shared + routed[:, col_off:col_off+C] @ W.

    W ([C, H] = the layer's _fc2_shard_t) is prefetched to smem BEFORE
    griddepcontrol.wait, so the weight read overlaps the producer fan
    kernel's lamport spin. Torch addmm cannot do this.
    """
    B, latent = routed.shape
    C, H = w_shard_t.shape
    out = torch.empty(B, H, device=routed.device, dtype=torch.bfloat16)
    _module().run_fc2(
        routed.data_ptr(), shared.data_ptr(), w_shard_t.data_ptr(),
        out.data_ptr(), B, latent, C, H, col_off)
    return out
