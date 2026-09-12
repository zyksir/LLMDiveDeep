"""Triton fused GEMM with bf16 and raw-fp32 outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _epilogue_store(
    out_ptr,
    acc,
    offs_m,
    n0,
    batch,
    N1: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ntot: tl.constexpr = N1 + N2
    width16: tl.constexpr = N1 + 2 * N2
    m_mask = offs_m[:, None] < batch
    offs_n = n0 + tl.arange(0, BLOCK_N)
    if n0 + BLOCK_N <= N1:
        tl.store(
            out_ptr + offs_m[:, None] * width16 + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=m_mask,
        )
    else:
        i32 = acc.to(tl.int32, bitcast=True)
        lo = (i32 & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        hi = (
            ((i32 >> 16) & 0xFFFF)
            .to(tl.uint16)
            .to(tl.bfloat16, bitcast=True)
        )
        pair = tl.interleave(lo, hi)
        lane = tl.arange(0, 2 * BLOCK_N)
        pcol = n0 + lane // 2
        pwidth = N1 + 2 * (pcol - N1) + lane % 2
        p_mask = (pcol[None, :] >= N1) & (pcol[None, :] < ntot)
        tl.store(
            out_ptr + offs_m[:, None] * width16 + pwidth[None, :],
            pair,
            mask=m_mask & p_mask,
        )
        if n0 < N1:
            tl.store(
                out_ptr + offs_m[:, None] * width16 + offs_n[None, :],
                acc.to(tl.bfloat16),
                mask=m_mask & (offs_n[None, :] < N1),
            )


@triton.jit
def _dual_out_gemm_kernel(
    h_ptr,
    w_ptr,
    desc_h,
    desc_w,
    out_ptr,
    scratch_ptr,
    counter_ptr,
    batch,
    stride_hm,
    TILES_M: tl.constexpr,
    K: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    REDUCE: tl.constexpr,
    EVEN_M: tl.constexpr,
    W_CG: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    ntot: tl.constexpr = N1 + N2
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = pid % TILES_M
    pid_n = pid // TILES_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split: tl.constexpr = K // SPLIT_K
    offs_k = pid_k * k_per_split + tl.arange(0, BLOCK_K)

    even_n: tl.constexpr = ntot % BLOCK_N == 0
    m_mask = offs_m[:, None] < batch
    if even_n:
        scratch_mask = m_mask
    else:
        scratch_mask = m_mask & (offs_n[None, :] < ntot)
    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    modifier: tl.constexpr = ".cg" if W_CG else ""

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(0, k_per_split, BLOCK_K):
        if USE_TMA:
            h = desc_h.load(
                [pid_m * BLOCK_M, pid_k * k_per_split + ki]
            )
            w = desc_w.load(
                [pid_n * BLOCK_N, pid_k * k_per_split + ki]
            ).T
        else:
            if EVEN_M:
                h = tl.load(h_ptrs)
            else:
                h = tl.load(h_ptrs, mask=m_mask, other=0.0)
            if even_n:
                w = tl.load(w_ptrs, cache_modifier=modifier)
            else:
                w = tl.load(
                    w_ptrs,
                    mask=offs_n[None, :] < ntot,
                    other=0.0,
                    cache_modifier=modifier,
                )
            h_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        acc = tl.dot(h, w, acc)

    if REDUCE == 0:
        _epilogue_store(
            out_ptr,
            acc,
            offs_m,
            pid_n * BLOCK_N,
            batch,
            N1=N1,
            N2=N2,
            BLOCK_N=BLOCK_N,
        )
    elif REDUCE == 3:
        scratch_ptrs = (
            scratch_ptr + offs_m[:, None] * ntot + offs_n[None, :]
        )
        tl.atomic_add(scratch_ptrs, acc, mask=scratch_mask)
    else:
        plane = batch * ntot
        base = scratch_ptr + offs_m[:, None] * ntot + offs_n[None, :]
        tl.store(base + pid_k * plane, acc, mask=scratch_mask)
        if REDUCE == 1:
            count = tl.atomic_add(
                counter_ptr + pid, 1, sem="acq_rel", scope="gpu"
            )
            if count == SPLIT_K - 1:
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for split in tl.static_range(SPLIT_K):
                    acc += tl.load(
                        base + split * plane,
                        mask=scratch_mask,
                        other=0.0,
                    )
                tl.store(counter_ptr + pid, 0)
                _epilogue_store(
                    out_ptr,
                    acc,
                    offs_m,
                    pid_n * BLOCK_N,
                    batch,
                    N1=N1,
                    N2=N2,
                    BLOCK_N=BLOCK_N,
                )


@triton.jit
def _reduce_finalize_kernel(
    src_ptr,
    out_ptr,
    total,
    N1: tl.constexpr,
    N2: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ntot: tl.constexpr = N1 + N2
    width16: tl.constexpr = N1 + 2 * N2
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(
            src_ptr + split * total + offsets, mask=mask, other=0.0
        )
    row = offsets // ntot
    col = offsets % ntot
    is_bf16 = col < N1
    tl.store(
        out_ptr + row * width16 + col,
        acc.to(tl.bfloat16),
        mask=mask & is_bf16,
    )
    i32 = acc.to(tl.int32, bitcast=True)
    lo = (i32 & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (
        ((i32 >> 16) & 0xFFFF)
        .to(tl.uint16)
        .to(tl.bfloat16, bitcast=True)
    )
    base = row * width16 + N1 + 2 * (col - N1)
    tl.store(out_ptr + base, lo, mask=mask & (col >= N1))
    tl.store(out_ptr + base + 1, hi, mask=mask & (col >= N1))


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int
    sema: bool = True
    w_cg: bool = False
    tma: bool = False


_BEST: dict[tuple, Config] = {}
_COUNTERS: dict[torch.device, torch.Tensor] = {}
_MAX_TILES = 8192


def _counter_buf(device: torch.device) -> torch.Tensor:
    buf = _COUNTERS.get(device)
    if buf is None:
        buf = torch.zeros(_MAX_TILES, device=device, dtype=torch.int32)
        _COUNTERS[device] = buf
    return buf


def _bm_bucket(batch: int) -> int:
    return max(16, min(128, triton.next_power_of_2(batch)))


def _coarse_candidates(
    bm_bucket: int,
    k_dim: int,
    n1: int,
    n2: int,
    impl: str,
    tma_ok: bool,
) -> list[Config]:
    bms = sorted({bm_bucket, max(16, bm_bucket // 2)})
    out = []

    def add(bm, bn, bk, split_k, sema=True, tma=False):
        if k_dim % (split_k * bk):
            return
        if tma and (not tma_ok or bk > 256):
            return
        if impl == "atomic" and (split_k == 1 or tma):
            return
        out.append(
            Config(
                bm,
                bn,
                bk,
                split_k,
                4,
                3,
                sema=sema and impl != "atomic",
                tma=tma,
            )
        )

    for bm in bms:
        for split_k in (1, 2, 4, 8, 14):
            for tma in (False, True):
                add(bm, 64, 128, split_k, tma=tma)
        for split_k in (1, 2, 4):
            for tma in (False, True):
                add(bm, 64, 256, split_k, tma=tma)
        for split_k in (4, 8):
            add(bm, 64, 64, split_k)
            add(bm, 64, 128, split_k, sema=False)
        if bm >= 64:
            add(bm, 128, 128, 4, tma=True)
            add(bm, 128, 64, 4)
    return out


def _run(
    h: torch.Tensor,
    w_all: torch.Tensor,
    buf: torch.Tensor,
    cfg: Config,
    impl: str,
    k_dim: int,
    n1: int,
    n2: int,
) -> None:
    batch = h.shape[0]
    ntot = n1 + n2
    tiles_m = triton.cdiv(batch, cfg.block_m)
    tiles_n = triton.cdiv(ntot, cfg.block_n)
    grid = (tiles_n * tiles_m, cfg.split_k)
    assert grid[0] <= _MAX_TILES
    counters = _counter_buf(h.device)
    if cfg.tma:
        from triton.tools.tensor_descriptor import TensorDescriptor

        desc_h = TensorDescriptor.from_tensor(
            h, [cfg.block_m, cfg.block_k]
        )
        desc_w = TensorDescriptor.from_tensor(
            w_all, [cfg.block_n, cfg.block_k]
        )
    else:
        desc_h, desc_w = h, w_all
    if cfg.split_k == 1:
        reduce_mode = 0
        scratch = counters
    elif impl == "atomic":
        reduce_mode = 3
        scratch = torch.zeros(
            batch, ntot, device=h.device, dtype=torch.float32
        )
    else:
        reduce_mode = 1 if cfg.sema else 2
        scratch = torch.empty(
            cfg.split_k,
            batch,
            ntot,
            device=h.device,
            dtype=torch.float32,
        )
    _dual_out_gemm_kernel[grid](
        h,
        w_all,
        desc_h,
        desc_w,
        buf,
        scratch,
        counters,
        batch,
        h.stride(0),
        TILES_M=tiles_m,
        K=k_dim,
        N1=n1,
        N2=n2,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        SPLIT_K=cfg.split_k,
        REDUCE=reduce_mode,
        EVEN_M=batch % cfg.block_m == 0,
        W_CG=cfg.w_cg,
        USE_TMA=cfg.tma,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if reduce_mode in (2, 3):
        total = batch * ntot
        block = 1024
        _reduce_finalize_kernel[(triton.cdiv(total, block),)](
            scratch,
            buf,
            total,
            N1=n1,
            N2=n2,
            SPLIT_K=cfg.split_k if reduce_mode == 2 else 1,
            BLOCK=block,
            num_warps=4,
        )


def _graph_time_us(fn, iters: int = 20, repeats: int = 3) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
    except Exception:
        torch.cuda.synchronize()
        graph = None
    best = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is None:
            for _ in range(iters):
                fn()
        else:
            graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / iters)
    return best


def _tune(
    h: torch.Tensor,
    w_all: torch.Tensor,
    impl: str,
    k_dim: int,
    n1: int,
    n2: int,
    bm_bucket: int,
) -> Config:
    buf = torch.empty(
        h.shape[0], n1 + 2 * n2, device=h.device, dtype=torch.bfloat16
    )

    def time_cfg(cfg: Config) -> float:
        def fn():
            _run(h, w_all, buf, cfg, impl, k_dim, n1, n2)

        fn()
        torch.cuda.synchronize()
        return _graph_time_us(fn)

    tma_ok = h.stride(0) % 8 == 0 and k_dim % 8 == 0
    timed: list[tuple[float, Config]] = []
    coarse = _coarse_candidates(
        bm_bucket, k_dim, n1, n2, impl, tma_ok
    )
    if not coarse:
        raise ValueError(f"no valid tiling for N1={n1}, N2={n2}, K={k_dim}")
    for cfg in coarse:
        try:
            timed.append((time_cfg(cfg), cfg))
        except Exception:
            continue
    timed.sort(key=lambda pair: pair[0])
    best_us, best = timed[0]
    for _, base in timed[:3]:
        for num_warps, num_stages in ((4, 4), (4, 5), (8, 3), (8, 4)):
            cfg = Config(
                base.block_m,
                base.block_n,
                base.block_k,
                base.split_k,
                num_warps,
                num_stages,
                sema=base.sema,
                w_cg=base.w_cg,
                tma=base.tma,
            )
            try:
                elapsed_us = time_cfg(cfg)
            except Exception:
                continue
            if elapsed_us < best_us:
                best_us, best = elapsed_us, cfg
    _BEST[(k_dim, n1, n2, bm_bucket, impl)] = best
    return best


def dual_out_gemm_triton(
    h: torch.Tensor,
    w_all: torch.Tensor,
    n1: int,
    n2: int,
    *,
    impl: str = "det",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bf16 and raw-fp32 GEMMs from one shared-weight kernel."""
    assert impl in ("det", "atomic")
    assert h.dtype == torch.bfloat16 and h.dim() == 2 and h.is_contiguous()
    assert w_all.dtype == torch.bfloat16 and w_all.is_contiguous()
    k_dim = h.shape[1]
    assert w_all.shape == (n1 + n2, k_dim)
    assert n1 % 2 == 0, "n1 must be even for the fp32 view alignment"
    batch = h.shape[0]
    bm_bucket = _bm_bucket(batch)
    buf = torch.empty(
        batch, n1 + 2 * n2, device=h.device, dtype=torch.bfloat16
    )
    cfg = _BEST.get((k_dim, n1, n2, bm_bucket, impl))
    if cfg is None:
        cfg = _tune(h, w_all, impl, k_dim, n1, n2, bm_bucket)
    _run(h, w_all, buf, cfg, impl, k_dim, n1, n2)
    return buf[:, :n1], buf[:, n1:].view(torch.float32)


def best_configs() -> dict[tuple, Config]:
    return dict(_BEST)


def bench(
    *,
    batch: int = 16,
    n1: int = 1984,
    n2: int = 896,
    k_dim: int = 7168,
    iters: int = 100,
    repeats: int = 5,
) -> dict[str, float]:
    """Check correctness and graph latency against unfused PyTorch GEMMs."""
    from ._bench import (
        check_regression,
        dual_out_reference,
        graph_time_us,
    )

    torch.manual_seed(0)
    h = torch.randn(batch, k_dim, device="cuda", dtype=torch.bfloat16)
    w_all = torch.randn(
        n1 + n2, k_dim, device="cuda", dtype=torch.bfloat16
    )
    expected = dual_out_reference(h, w_all, n1, n2)
    actual = dual_out_gemm_triton(h, w_all, n1, n2)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.5)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.02, atol=0.5)

    reference_us = graph_time_us(
        lambda: dual_out_reference(h, w_all, n1, n2),
        iters=iters,
        repeats=repeats,
    )
    kernel_us = graph_time_us(
        lambda: dual_out_gemm_triton(h, w_all, n1, n2),
        iters=iters,
        repeats=repeats,
    )
    check_regression(reference_us, kernel_us)
    return {"reference_us": reference_us, "kernel_us": kernel_us}


dual_out_gemm = dual_out_gemm_triton
kernel_fn = dual_out_gemm_triton

__all__ = [
    "Config",
    "bench",
    "best_configs",
    "dual_out_gemm",
    "dual_out_gemm_triton",
]
