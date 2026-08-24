"""Small-M streaming GEMV for the KDA projections (decode M <= 16).

Discovery (2026-08-24): the KDA projection GEMMs run far off their
weight-byte floors at decode M=8 — out_proj [8,1536]x[1536->7168]
measures 12.4 us (torch) / ~9.6 us (serving splitK chain) vs a 3.4 us
DRAM floor; in_proj 18.7 vs 13.9. cublasLt/nvjet have no good tiny-M
kernels for these shapes. A memory-bound GEMV that streams W once with
16B loads and keeps x in smem should sit near the floor.

out[m, n] = sum_k x[m, k] * W[n, k]   (W row-major [N, K], bf16)
Grid: N/ROWS_PER_CTA blocks; each block streams its W rows once,
computing all M outputs for those rows (x staged in smem fp32).
"""
from __future__ import annotations

import torch

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#define THREADS 256
#define MAX_M 16
#define MAX_K 8192

// each WARP owns one W row at a time; lanes stride the K dim with uint4
// loads; per-lane partial dots for all M reduce via shuffles.
template <int M>
__global__ void __launch_bounds__(THREADS) small_m_gemv_kernel(
    __nv_bfloat16 const* __restrict__ x,   // [M, K]
    __nv_bfloat16 const* __restrict__ w,   // [N, K]
    __nv_bfloat16* __restrict__ out,       // [M, N]
    int32_t N, int32_t K)
{
    constexpr int R = 4;  // rows per warp pass: x loads amortize 4x
    int const tid = threadIdx.x;
    int const warp = tid / 32, lane = tid % 32;
    int const warps = THREADS / 32;
    int const nvec = K / 8;
    uint4 const* xv = reinterpret_cast<uint4 const*>(x);
    for (int n0 = (blockIdx.x * warps + warp) * R; n0 < N;
         n0 += gridDim.x * warps * R) {
        int const nr = min(R, N - n0);
        float acc[R][M];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
        for (int v = lane; v < nvec; v += 32) {
            uint4 qw[R];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                if (r < nr)
                    qw[r] = reinterpret_cast<uint4 const*>(
                        w + (int64_t)(n0 + r) * K)[v];
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                uint4 const qx = xv[m * nvec + v];  // L2, amortized over R
                __nv_bfloat16 const* ex =
                    reinterpret_cast<__nv_bfloat16 const*>(&qx);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    __nv_bfloat16 const* e =
                        reinterpret_cast<__nv_bfloat16 const*>(&qw[r]);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[r][m] += __bfloat162float(e[j])
                                   * __bfloat162float(ex[j]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= nr) break;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float sv = acc[r][m];
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    sv += __shfl_down_sync(0xffffffffu, sv, o);
                if (lane == 0)
                    out[(int64_t)m * N + n0 + r] = __float2bfloat16(sv);
            }
        }
    }
}

template <int M>
void launch_gemv(int64_t x, int64_t w, int64_t out, int64_t N, int64_t K,
                 int64_t n_blocks, cudaStream_t stream)
{
    small_m_gemv_kernel<M><<<(unsigned)n_blocks, THREADS, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(x),
        reinterpret_cast<__nv_bfloat16 const*>(w),
        reinterpret_cast<__nv_bfloat16*>(out), (int32_t)N, (int32_t)K);
}

void run_gemv(int64_t x, int64_t w, int64_t out,
              int64_t M, int64_t N, int64_t K, int64_t n_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(K % 8 == 0, "K % 8");
    switch (M) {
        case 1: launch_gemv<1>(x, w, out, N, K, n_blocks, stream); break;
        case 2: launch_gemv<2>(x, w, out, N, K, n_blocks, stream); break;
        case 4: launch_gemv<4>(x, w, out, N, K, n_blocks, stream); break;
        case 8: launch_gemv<8>(x, w, out, N, K, n_blocks, stream); break;
        case 16: launch_gemv<16>(x, w, out, N, K, n_blocks, stream); break;
        default: TORCH_CHECK(false, "unsupported M");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gemv launch failed");
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        _MODULE = load_inline(
            name="k3_small_m_gemv",
            cpp_sources=("#include <torch/extension.h>\n"
                         "void run_gemv(int64_t,int64_t,int64_t,int64_t,"
                         "int64_t,int64_t,int64_t);"),
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            functions=["run_gemv"],
        )
    return _MODULE


def small_m_gemv(x: torch.Tensor, w: torch.Tensor,
                 n_blocks: int = 148) -> torch.Tensor:
    """out = x @ w.T for tiny M; w is [N, K] row-major bf16."""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    _module().run_gemv(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                       M, N, K, n_blocks)
    return out
