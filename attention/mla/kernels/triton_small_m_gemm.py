"""Small-M tall-skinny GEMM via Triton (bottom-line target #3).

Goal: beat cublasLt at the KDA projection shapes at decode M<=16, where
cublasLt reaches only 30-74% of the weight-byte floor (out_proj
[8,1536]x[1536->7168]: 11.1 us vs 3.4 floor; in_proj [8,7168]x[7168->6288]:
18.7 vs 13.9). The naive SIMT GEMV plateaued at 2.1 TB/s (documented in
RESEARCH.md); this attempt uses tl.dot + deep num_stages pipelining so
Triton emits bulk async weight copies.

out = x @ W.T, x [M, K] bf16, W [N, K] bf16 row-major, out [M, N] bf16.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _small_m_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16),
             mask=m_mask & (offs_n[None, :] < N))


_CONFIGS = [
    (BN, BK, w, st)
    for BN in (64, 128, 256)
    for BK in (64, 128)
    for w in (4, 8)
    for st in (3, 4, 5)
]


def small_m_gemm(x: torch.Tensor, w: torch.Tensor,
                 cfg=(128, 64, 4, 4)) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    BN, BK, warps, stages = cfg
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(N, BN),)
    _small_m_gemm_kernel[grid](
        x, w, out, M, N, K, w.stride(0), w.stride(1),
        BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=warps, num_stages=stages,
    )
    return out
