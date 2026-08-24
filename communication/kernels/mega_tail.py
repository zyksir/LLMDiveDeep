"""Fan v3 "mega-tail": the ENTIRE b10 MoE decode tail in ONE kernel.

Replaces the serving tail chain (trt finalizeKernel 4.3 us -> trt fused
AR+norm 6.9 -> torch addmm 4.1 -> trt AR 7.2 = 22.5 us kernels + ~8 us
launch gaps = 30.6 us wall p50 at bs8, measured serving trace
2026-08-23) with a single launch:

  phase A  (blocks b < B)  finalize: gather this rank's top-k expert rows
           from the unfinalized fc2 output, weighted-sum fp32, push the
           canonicalized bf16 latent partial to every rank's ROUND-1
           lamport slot (data-encoded arrival, v2 protocol).
  phase B  (blocks b < B)  spin on round-1, reduce fp32, RMSNorm, write
           the normed latent row to a LOCAL scratch (data-encoded ready:
           scratch pre-filled with -0.0 sentinel, writes canonicalized).
  phase C  (ALL blocks)    each block owns a 64-col H-tile; its W tile
           ([448, 64] of the rank's fc2 shard) is prefetched to smem at
           kernel ENTRY (independent of rounds - overlaps A/B). Spin on
           scratch rows, GEMV partial[b, tile] = shared[b, tile] +
           normed[b, cols] @ Wtile, canonicalize, push [B, 64] column
           slices into every rank's ROUND-2 slot.
  phase D  (ALL blocks)    spin on round-2 for the own tile across all
           ranks, reduce fp32, write out[b, tile]; clear consumed slots
           and scratch back to the sentinel; last block bumps the round.

Two data-encoded lamport rounds, zero flags/fences/extra launches,
CUDA-graph safe. Same B-packing invariant as v2 (all ranks call with the
same B; clears cover exactly what was read).
"""
from __future__ import annotations

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
#define TILE_H 64
#define NEG_ZERO_U32 0x80008000u

__device__ __forceinline__ uint4 ld_vol_v4(uint4 const* p) {
    uint4 q;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
                 : "l"(p) : "memory");
    return q;
}
__device__ __forceinline__ bool has_nz(uint4 const& q) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t const w = (&q.x)[j];
        if ((w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u) return true;
    }
    return false;
}
__device__ __forceinline__ uint16_t canon(float v) {
    uint16_t b = __bfloat16_as_ushort(__float2bfloat16(v));
    return (b == 0x8000u) ? 0u : b;
}

// Layout inside the symm buffer (per rank), all offsets in BYTES:
//   round1: slot s in [0,3): s*r1_bytes ... world*Bmax*latent*2 each
//   round2: R2_OFF + s*r2_bytes; rows [writer*Bmax + b] * hidden*2
// The local scratch (normed latent) is a separate LOCAL tensor.
__global__ void __launch_bounds__(THREADS) mega_tail_kernel(
    __nv_bfloat16 const* __restrict__ fc2,     // [n_rows, latent]
    int32_t const* __restrict__ idx,            // [B, topk]
    float const* __restrict__ scales,           // [B, topk]
    __nv_bfloat16 const* __restrict__ normw,    // [latent]
    __nv_bfloat16 const* __restrict__ shared_in,// [B, hidden]
    __nv_bfloat16 const* __restrict__ w2t,      // [width, hidden]
    __nv_bfloat16* __restrict__ scratch,        // [Bmax, latent] local
    __nv_bfloat16* __restrict__ out,            // [B, hidden]
    int64_t const* __restrict__ peer_ptrs,
    int32_t* __restrict__ round_buf,             // [round, done]
    int32_t B, int32_t topk, int32_t latent, int32_t hidden,
    int32_t width, int32_t col_off, int32_t rank, int32_t world,
    int64_t r1_slot_bytes, int64_t r2_off, int64_t r2_slot_bytes,
    int32_t Bmax, float eps, int32_t use_pdl, int32_t n_rows)
{
    extern __shared__ unsigned char smem[];
    // smem: W tile [width, TILE_H] bf16, then fp32 acc row [latent]
    __nv_bfloat16* w_sh = reinterpret_cast<__nv_bfloat16*>(smem);
    float* acc = reinterpret_cast<float*>(
        smem + (size_t)width * TILE_H * sizeof(__nv_bfloat16));
    int const tid = threadIdx.x;
    int const blk = blockIdx.x;
    int const round = *round_buf;
    int const slot = round % 3;
    int const nvec_l = latent / 8;

    // ---- W-tile prefetch: independent of every round, overlaps A+B.
    int const h0 = blk * TILE_H;
    for (int i = tid; i < width * TILE_H; i += THREADS)
        w_sh[i] = w2t[(int64_t)(i / TILE_H) * hidden + h0 + i % TILE_H];
#if __CUDA_ARCH__ >= 900
    if (use_pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    if (blk < B) {
        int const b = blk;
        // ---- phase A: finalize + round-1 push
        for (int c = tid; c < latent; c += THREADS) acc[c] = 0.f;
        __syncthreads();
        for (int k = 0; k < topk; ++k) {
            int const row = idx[b * topk + k];
            if (row < 0 || row >= n_rows) continue;
            float const w = scales[b * topk + k];
            uint4 const* rowv = reinterpret_cast<uint4 const*>(
                fc2 + (int64_t)row * latent);
            for (int v = tid; v < nvec_l; v += THREADS) {
                uint4 const q = rowv[v];
                __nv_bfloat16 const* e =
                    reinterpret_cast<__nv_bfloat16 const*>(&q);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[v * 8 + j] += w * __bfloat162float(e[j]);
            }
        }
        __syncthreads();
        __shared__ __align__(16) uint16_t packed[4096];
        for (int c = tid; c < latent; c += THREADS)
            packed[c] = canon(acc[c]);
        __syncthreads();
        uint4 const* src = reinterpret_cast<uint4 const*>(packed);
        for (int p = 0; p < world; ++p) {
            int const pp = (p + rank) % world;
            uint4* dst = reinterpret_cast<uint4*>(
                peer_ptrs[pp] + (int64_t)slot * r1_slot_bytes
                + ((int64_t)rank * Bmax + b) * latent * 2);
            for (int v = tid; v < nvec_l; v += THREADS) dst[v] = src[v];
        }
        // ---- phase B: round-1 spin + reduce + norm -> scratch
        __shared__ float ssq_sh;
        float ssq = 0.f;
        for (int v = tid; v < nvec_l; v += THREADS) {
            float col[8] = {0.f};
            for (int p = 0; p < world; ++p) {
                uint4 const* s = reinterpret_cast<uint4 const*>(
                    peer_ptrs[rank] + (int64_t)slot * r1_slot_bytes
                    + ((int64_t)p * Bmax + b) * latent * 2) + v;
                uint4 q = ld_vol_v4(s);
                while (has_nz(q)) q = ld_vol_v4(s);
                __nv_bfloat16 const* e =
                    reinterpret_cast<__nv_bfloat16 const*>(&q);
                #pragma unroll
                for (int j = 0; j < 8; ++j) col[j] += __bfloat162float(e[j]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[v * 8 + j] = col[j];
                ssq += col[j] * col[j];
            }
        }
        __shared__ float red[THREADS / 32];
        for (int o = 16; o > 0; o >>= 1)
            ssq += __shfl_down_sync(0xffffffffu, ssq, o);
        if ((tid & 31) == 0) red[tid >> 5] = ssq;
        __syncthreads();
        if (tid < THREADS / 32) {
            ssq = red[tid];
            for (int o = THREADS / 64; o > 0; o >>= 1)
                ssq += __shfl_down_sync(0x0000ffffu, ssq, o);
            if (tid == 0) ssq_sh = ssq;
        }
        __syncthreads();
        float const inv = rsqrtf(ssq_sh / latent + eps);
        // normed row -> LOCAL scratch, canonicalized (data-encoded ready)
        for (int c = tid; c < latent; c += THREADS) {
            reinterpret_cast<uint16_t*>(scratch)[(int64_t)b * latent + c] =
                canon(acc[c] * inv * __bfloat162float(normw[c]));
        }
        // clear own round-1 rows for round+3
        uint4 const sent = {NEG_ZERO_U32, NEG_ZERO_U32,
                            NEG_ZERO_U32, NEG_ZERO_U32};
        for (int p = 0; p < world; ++p) {
            uint4* row = reinterpret_cast<uint4*>(
                peer_ptrs[rank] + (int64_t)slot * r1_slot_bytes
                + ((int64_t)p * Bmax + b) * latent * 2);
            for (int v = tid; v < nvec_l; v += THREADS) row[v] = sent;
        }
    }
#if __CUDA_ARCH__ >= 900
    if (use_pdl)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

    // ---- phase C: token-parallel GEMV over sub-batches of TB tokens.
    // Staging r_all[TB][width] fp32 reuses the acc area (latent*4 bytes
    // >= TB*width*4 for TB=8, width=448).
    {
        float* r_all = acc;
        int const TB = 8;
        __shared__ __align__(16) __nv_bfloat16 tile_out[TB * TILE_H];
        for (int b0 = 0; b0 < B; b0 += TB) {
            int const nb = min(TB, B - b0);
            __syncthreads();
            // parallel data-encoded spin-stage, 16B vectors
            // (width % 8 == 0 and col_off % 8 == 0: both = latent/world)
            int const nvw = width / 8;
            for (int i = tid; i < nb * nvw; i += THREADS) {
                int const b = i / nvw, v = i % nvw;
                uint4 const* sp = reinterpret_cast<uint4 const*>(
                    scratch + (int64_t)(b0 + b) * latent + col_off) + v;
                uint4 q = ld_vol_v4(sp);
                while (has_nz(q)) q = ld_vol_v4(sp);
                __nv_bfloat16 const* e =
                    reinterpret_cast<__nv_bfloat16 const*>(&q);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    r_all[b * width + v * 8 + j] = __bfloat162float(e[j]);
            }
            __syncthreads();
            // one output per thread: (token, col) over nb*TILE_H
            for (int i = tid; i < nb * TILE_H; i += THREADS) {
                int const b = i / TILE_H, h = i % TILE_H;
                float s = __bfloat162float(
                    shared_in[(int64_t)(b0 + b) * hidden + h0 + h]);
                float const* r = r_all + b * width;
                #pragma unroll 4
                for (int c = 0; c < width; ++c)
                    s += r[c] * __bfloat162float(w_sh[c * TILE_H + h]);
                reinterpret_cast<uint16_t*>(tile_out)[i] = canon(s);
            }
            __syncthreads();
            // vectorized push: (token, peer, vec) one 16B store per thread
            int const nv = TILE_H / 8;
            uint4 const* src = reinterpret_cast<uint4 const*>(tile_out);
            for (int i = tid; i < nb * world * nv; i += THREADS) {
                int const b = i / (world * nv);
                int const p = ((i / nv) % world + rank) % world;
                int const v = i % nv;
                reinterpret_cast<uint4*>(
                    peer_ptrs[p] + r2_off + (int64_t)slot * r2_slot_bytes
                    + ((int64_t)rank * Bmax + b0 + b) * hidden * 2
                    + (int64_t)h0 * 2)[v] = src[b * nv + v];
            }
        }
        __syncthreads();
    }

    // ---- phase D: token-parallel round-2 spin + reduce own tile
    {
        int const nv = TILE_H / 8;
        for (int i = tid; i < B * nv; i += THREADS) {
            int const b = i / nv, v = i % nv;
            float col[8] = {0.f};
            for (int p = 0; p < world; ++p) {
                uint4 const* sp = reinterpret_cast<uint4 const*>(
                    peer_ptrs[rank] + r2_off + (int64_t)slot * r2_slot_bytes
                    + ((int64_t)p * Bmax + b) * hidden * 2
                    + (int64_t)h0 * 2) + v;
                uint4 q = ld_vol_v4(sp);
                while (has_nz(q)) q = ld_vol_v4(sp);
                __nv_bfloat16 const* e =
                    reinterpret_cast<__nv_bfloat16 const*>(&q);
                #pragma unroll
                for (int j = 0; j < 8; ++j) col[j] += __bfloat162float(e[j]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                out[(int64_t)b * hidden + h0 + v * 8 + j] =
                    __float2bfloat16(col[j]);
        }
    }
    __syncthreads();
    // clears: round-2 own tile rows; scratch tile (per-token slices)
    {
        uint4 const sent = {NEG_ZERO_U32, NEG_ZERO_U32,
                            NEG_ZERO_U32, NEG_ZERO_U32};
        int const nv = TILE_H / 8;
        for (int i = tid; i < B * world * nv; i += THREADS)
            reinterpret_cast<uint4*>(
                peer_ptrs[rank] + r2_off + (int64_t)slot * r2_slot_bytes
                + ((int64_t)((i / nv) % world) * Bmax + i / (world * nv))
                  * hidden * 2
                + (int64_t)h0 * 2)[i % nv] = sent;
    }
    if (blk < B) {
        // clear own scratch row (the writer clears; readers are done:
        // every tile block passed its phase-D barrier for this round)
        // NOTE: readers of OTHER blocks may still be in phase C for this
        // row -> clear must wait for grid-wide phase-C completion. We
        // piggyback on the round counter: clear happens NEXT call, so
        // here we only clear after last-block arrival (below).
    }
    __syncthreads();
    if (tid == 0) {
        int const done = atomicAdd(round_buf + 1, 1);
        if (done == gridDim.x - 1) {
            round_buf[1] = 0;
            *reinterpret_cast<volatile int32_t*>(round_buf) = round + 1;
        }
    }
    // scratch rows are cleared at the START of the next call by phase-A
    // blocks (before writing): safe because a new call cannot begin
    // before this kernel completes (same stream).
}

__global__ void clear_scratch_kernel(__nv_bfloat16* scratch, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) reinterpret_cast<uint16_t*>(scratch)[i] = 0x8000u;
}

void run_mega(int64_t fc2, int64_t idx, int64_t scales, int64_t normw,
              int64_t shared_in, int64_t w2t, int64_t scratch, int64_t out,
              torch::Tensor peer_ptrs, torch::Tensor round_buf,
              int64_t B, int64_t topk, int64_t latent, int64_t hidden,
              int64_t width, int64_t col_off, int64_t rank, int64_t world,
              int64_t r1_slot_bytes, int64_t r2_off, int64_t r2_slot_bytes,
              int64_t Bmax, double eps, int64_t n_rows, int64_t pre_clear)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    size_t const smem = (size_t)width * 64 * 2 + (size_t)latent * 4;
    static bool ok = false;
    if (!ok) {
        cudaError_t e = cudaFuncSetAttribute(mega_tail_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(e == cudaSuccess, "smem attr: ", cudaGetErrorString(e));
        ok = true;
    }
    if (pre_clear) {
        int64_t n = Bmax * latent;
        clear_scratch_kernel<<<(unsigned)((n + 511) / 512), 512, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(scratch), n);
    }
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    bool const use_pdl = getenv("K3_FAN_PDL") == nullptr
        || getenv("K3_FAN_PDL")[0] != '0';
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)(hidden / TILE_H), 1, 1);
    cfg.blockDim = dim3(THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = use_pdl ? 1 : 0;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, mega_tail_kernel,
        reinterpret_cast<__nv_bfloat16 const*>(fc2),
        reinterpret_cast<int32_t const*>(idx),
        reinterpret_cast<float const*>(scales),
        reinterpret_cast<__nv_bfloat16 const*>(normw),
        reinterpret_cast<__nv_bfloat16 const*>(shared_in),
        reinterpret_cast<__nv_bfloat16 const*>(w2t),
        reinterpret_cast<__nv_bfloat16*>(scratch),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<int64_t const*>(peer_ptrs.data_ptr<int64_t>()),
        round_buf.data_ptr<int32_t>(),
        (int32_t)B, (int32_t)topk, (int32_t)latent, (int32_t)hidden,
        (int32_t)width, (int32_t)col_off, (int32_t)rank, (int32_t)world,
        r1_slot_bytes, r2_off, r2_slot_bytes, (int32_t)Bmax,
        (float)eps, (int32_t)(use_pdl ? 1 : 0), (int32_t)n_rows)
        == cudaSuccess, "mega_tail launch failed: ",
        cudaGetErrorString(cudaGetLastError()));
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        _MODULE = load_inline(
            name="k3_mega_tail_v3",
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "void run_mega(int64_t,int64_t,int64_t,int64_t,int64_t,"
                "int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,"
                "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
                "int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,"
                "int64_t);"),
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            functions=["run_mega"],
        )
    return _MODULE


class MegaTail:
    """One-kernel MoE decode tail: finalize + AR(latent)+norm + fc2-shard
    GEMV + AR(hidden). Data-encoded lamport, 3 rotating slots."""

    def __init__(self, group, rank: int, world: int, *,
                 max_tokens: int = 128, latent: int = 3584,
                 hidden: int = 7168) -> None:
        self.rank, self.world = rank, world
        self.latent, self.hidden, self.max_tokens = latent, hidden, max_tokens
        self.r1_slot = world * max_tokens * latent * 2
        self.r2_slot = world * max_tokens * hidden * 2
        self.r2_off = 3 * self.r1_slot
        total = 3 * self.r1_slot + 3 * self.r2_slot
        self.buf = symm_mem.empty(
            total, dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()))
        self.buf.view(torch.int16).fill_(-32768)
        hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.peer_ptrs = torch.tensor(
            [int(p) for p in hdl.buffer_ptrs], dtype=torch.int64,
            device="cuda")
        self.round_buf = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.scratch = torch.empty(max_tokens, latent, dtype=torch.bfloat16,
                                   device="cuda")
        self.scratch.view(torch.int16).fill_(-32768)
        self._first = True
        torch.cuda.synchronize()
        dist.barrier()

    def __call__(self, fc2: torch.Tensor, idx: torch.Tensor,
                 scales: torch.Tensor, norm_weight: torch.Tensor,
                 shared: torch.Tensor, w2_shard_t: torch.Tensor,
                 col_off: int, eps: float) -> torch.Tensor:
        batch = idx.shape[0]
        assert batch <= self.max_tokens
        width = w2_shard_t.shape[0]
        out = torch.empty(batch, self.hidden, device=fc2.device,
                          dtype=torch.bfloat16)
        _module().run_mega(
            fc2.data_ptr(), idx.data_ptr(), scales.data_ptr(),
            norm_weight.data_ptr(), shared.data_ptr(),
            w2_shard_t.data_ptr(), self.scratch.data_ptr(), out.data_ptr(),
            self.peer_ptrs, self.round_buf, batch, idx.shape[1],
            self.latent, self.hidden, width, col_off, self.rank,
            self.world, self.r1_slot, self.r2_off, self.r2_slot,
            self.max_tokens, eps, fc2.shape[0],
            1)  # pre-clear scratch every call (tiny kernel; see note)
        return out
