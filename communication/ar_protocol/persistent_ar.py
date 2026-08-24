"""AR protocol experiment 1: persistent multi-round AR vs launch-per-AR.

Question: a K3 decode step runs ~279 small one-shot ARs (93 layers x 3).
Each launch-per-AR pays launch overhead + a fresh spin that eats the
CURRENT rank skew. A single persistent kernel looping R rounds inside
(data-encoded lamport, 3 rotating slots — safety induction identical to
fan v2: a rank can lead by at most 2 rounds because round r+3 requires
everyone's r+2 push) removes the launches and lets skew pipeline.

Arms:
  persistent  ONE kernel, R internal rounds, optional per-rank delay
              (emulated compute) before each round's push.
  chain       R launches of a single-round kernel (same protocol), the
              same optional delay injected as a separate tiny kernel
              between launches (launch-per-AR reference).

Both arms produce out = world-sum of the payload each round (correctness
checked once). Payload [B, DIM] bf16 (default 8 x 3584 — the MoE latent
AR shape).
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

__device__ __forceinline__ void delay_ns(int64_t ns) {
    if (ns <= 0) return;
    int64_t start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    int64_t now = start;
    while (now - start < ns)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
}

// one round of data-encoded lamport AR: push canonicalized payload to
// every rank's slot, spin+reduce, write out, clear consumed slot.
__device__ void ar_round(
    __nv_bfloat16 const* __restrict__ src,   // [B, dim]
    __nv_bfloat16* __restrict__ out,          // [B, dim]
    int64_t const* __restrict__ peer_ptrs,
    int32_t b, int32_t dim, int32_t rank, int32_t world,
    int64_t slot_bytes, int32_t slot, int32_t Bmax)
{
    int const tid = threadIdx.x;
    int const nvec = dim / 8;
    // push (canonicalize -0.0 -> +0.0 per 16-bit lane)
    for (int p = 0; p < world; ++p) {
        int const pp = (p + rank) % world;
        uint4* dst = reinterpret_cast<uint4*>(
            peer_ptrs[pp] + (int64_t)slot * slot_bytes
            + ((int64_t)rank * Bmax + b) * dim * 2);
        uint4 const* s = reinterpret_cast<uint4 const*>(
            src + (int64_t)b * dim);
        for (int v = tid; v < nvec; v += THREADS) {
            uint4 q = s[v];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t w = (&q.x)[j];
                if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
                if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
                (&q.x)[j] = w;
            }
            dst[v] = q;
        }
    }
    // spin + reduce
    for (int v = tid; v < nvec; v += THREADS) {
        float col[8] = {0.f};
        for (int p = 0; p < world; ++p) {
            uint4 const* s = reinterpret_cast<uint4 const*>(
                peer_ptrs[rank] + (int64_t)slot * slot_bytes
                + ((int64_t)p * Bmax + b) * dim * 2) + v;
            uint4 q = ld_vol_v4(s);
            while (has_nz(q)) q = ld_vol_v4(s);
            __nv_bfloat16 const* e =
                reinterpret_cast<__nv_bfloat16 const*>(&q);
            #pragma unroll
            for (int j = 0; j < 8; ++j) col[j] += __bfloat162float(e[j]);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            out[(int64_t)b * dim + v * 8 + j] = __float2bfloat16(col[j]);
    }
    // clear consumed rows for slot reuse at round+3
    uint4 const sent = {NEG_ZERO_U32, NEG_ZERO_U32,
                        NEG_ZERO_U32, NEG_ZERO_U32};
    for (int p = 0; p < world; ++p) {
        uint4* row = reinterpret_cast<uint4*>(
            peer_ptrs[rank] + (int64_t)slot * slot_bytes
            + ((int64_t)p * Bmax + b) * dim * 2);
        for (int v = tid; v < nvec; v += THREADS) row[v] = sent;
    }
}

__global__ void __launch_bounds__(THREADS) persistent_ar_kernel(
    __nv_bfloat16 const* __restrict__ src,
    __nv_bfloat16* __restrict__ out,
    int64_t const* __restrict__ peer_ptrs,
    int32_t* __restrict__ round_buf,
    int32_t B, int32_t dim, int32_t rank, int32_t world,
    int64_t slot_bytes, int32_t Bmax, int32_t rounds, int64_t skew_ns)
{
    int const b = blockIdx.x;
    int const round0 = *round_buf;
    for (int r = 0; r < rounds; ++r) {
        // emulated per-rank compute skew before each round's push
        delay_ns(skew_ns * rank);
        ar_round(src, out, peer_ptrs, b, dim, rank, world,
                 slot_bytes, (round0 + r) % 3, Bmax);
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int const done = atomicAdd(round_buf + 1, 1);
        if (done == gridDim.x - 1) {
            round_buf[1] = 0;
            *reinterpret_cast<volatile int32_t*>(round_buf) =
                round0 + rounds;
        }
    }
}

__global__ void __launch_bounds__(THREADS) single_ar_kernel(
    __nv_bfloat16 const* __restrict__ src,
    __nv_bfloat16* __restrict__ out,
    int64_t const* __restrict__ peer_ptrs,
    int32_t* __restrict__ round_buf,
    int32_t B, int32_t dim, int32_t rank, int32_t world,
    int64_t slot_bytes, int32_t Bmax, int64_t skew_ns)
{
    int const b = blockIdx.x;
    int const round = *round_buf;
    delay_ns(skew_ns * rank);
    ar_round(src, out, peer_ptrs, b, dim, rank, world,
             slot_bytes, round % 3, Bmax);
    __syncthreads();
    if (threadIdx.x == 0) {
        int const done = atomicAdd(round_buf + 1, 1);
        if (done == gridDim.x - 1) {
            round_buf[1] = 0;
            *reinterpret_cast<volatile int32_t*>(round_buf) = round + 1;
        }
    }
}

void run_persistent(int64_t src, int64_t out, torch::Tensor peer_ptrs,
                    torch::Tensor round_buf, int64_t B, int64_t dim,
                    int64_t rank, int64_t world, int64_t slot_bytes,
                    int64_t Bmax, int64_t rounds, int64_t skew_ns)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    persistent_ar_kernel<<<(unsigned)B, THREADS, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(src),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<int64_t const*>(peer_ptrs.data_ptr<int64_t>()),
        round_buf.data_ptr<int32_t>(), (int32_t)B, (int32_t)dim,
        (int32_t)rank, (int32_t)world, slot_bytes, (int32_t)Bmax,
        (int32_t)rounds, skew_ns);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "persistent launch");
}

void run_single(int64_t src, int64_t out, torch::Tensor peer_ptrs,
                torch::Tensor round_buf, int64_t B, int64_t dim,
                int64_t rank, int64_t world, int64_t slot_bytes,
                int64_t Bmax, int64_t skew_ns)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    single_ar_kernel<<<(unsigned)B, THREADS, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(src),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<int64_t const*>(peer_ptrs.data_ptr<int64_t>()),
        round_buf.data_ptr<int32_t>(), (int32_t)B, (int32_t)dim,
        (int32_t)rank, (int32_t)world, slot_bytes, (int32_t)Bmax, skew_ns);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "single launch");
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        _MODULE = load_inline(
            name="k3_persistent_ar",
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "void run_persistent(int64_t,int64_t,torch::Tensor,"
                "torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,"
                "int64_t,int64_t,int64_t);\n"
                "void run_single(int64_t,int64_t,torch::Tensor,"
                "torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,"
                "int64_t,int64_t);"),
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            functions=["run_persistent", "run_single"],
        )
    return _MODULE


class PersistentAR:
    def __init__(self, group, rank: int, world: int, *,
                 max_tokens: int = 16, dim: int = 3584) -> None:
        self.rank, self.world, self.dim, self.Bmax = rank, world, dim, max_tokens
        self.slot_bytes = world * max_tokens * dim * 2
        self.buf = symm_mem.empty(3 * self.slot_bytes, dtype=torch.uint8,
                                  device=torch.device(
                                      "cuda", torch.cuda.current_device()))
        self.buf.view(torch.int16).fill_(-32768)
        hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.peer_ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs],
                                      dtype=torch.int64, device="cuda")
        self.round_buf = torch.zeros(2, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()

    def persistent(self, src, out, rounds: int, skew_ns: int = 0) -> None:
        _module().run_persistent(
            src.data_ptr(), out.data_ptr(), self.peer_ptrs, self.round_buf,
            src.shape[0], self.dim, self.rank, self.world, self.slot_bytes,
            self.Bmax, rounds, skew_ns)

    def single(self, src, out, skew_ns: int = 0) -> None:
        _module().run_single(
            src.data_ptr(), out.data_ptr(), self.peer_ptrs, self.round_buf,
            src.shape[0], self.dim, self.rank, self.world, self.slot_bytes,
            self.Bmax, skew_ns)
