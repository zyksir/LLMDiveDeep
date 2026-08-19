#!/usr/bin/env python3
"""Contiguous-input A/B: b10 stride-aware SiTU vs stock TRT SiTUAndMul.

Both inputs are independently allocated, contiguous [T, 1536] bf16 — the
fused-front shared gate/up width at TP8. The b10 kernel is also valid on
row-strided views; this harness only checks that the contiguous case stays
in the same latency band as the default kernel, not that it wins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.kernel_bench import bench_cuda, error_stats  # noqa: E402
from kimi_k3_layer.kernels.situ_triton import situ_and_mul  # noqa: E402

# Fused-front shared gate/up at TP8: 2 * (SHARED_INTER / 8) = 1536.
_GATE_UP_COLS = 1536
_BETA = 1.0
_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def _contiguous(tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        tokens, _GATE_UP_COLS, generator=generator, device="cuda",
    ).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=list(_SIZES))
    args = parser.parse_args()

    from kimi_k3_layer.b10_kimi_k3_kda_layer import _graft

    SiTUAndMul = _graft(
        "tensorrt_llm._torch.modules.situ",
        "_torch/modules/situ.py",
    ).SiTUAndMul
    trt_situ = SiTUAndMul(beta=_BETA, linear_beta=None).cuda()

    print(f"{'T':>6}  {'correct':>7}  {'max_abs':>9}  "
          f"{'trt_situ_compile':>16}  {'b10_situ_triton':>15}  ratio")
    for tokens in args.tokens:
        # Two independent contiguous allocations (no shared storage).
        x_trt = _contiguous(tokens, seed=tokens)
        x_b10 = _contiguous(tokens, seed=tokens + 1_000_003)

        # Correctness on a third contiguous clone of the same values.
        x_check = x_trt.clone()
        trt_out = trt_situ(x_check)
        b10_out = situ_and_mul(x_check, _BETA, None)
        stats = error_stats(b10_out, trt_out, atol=2e-3, rtol=2e-3)

        def trt_fn():
            return trt_situ(x_trt)

        def b10_fn():
            return situ_and_mul(x_b10, _BETA, None)

        # torch.compile / Triton warmup is untimed (per-shape, once).
        trt_fn()
        b10_fn()
        torch.cuda.synchronize()

        trt_us = bench_cuda(trt_fn)
        b10_us = bench_cuda(b10_fn)
        ratio = b10_us / trt_us if trt_us > 0 else float("nan")
        print(
            f"{tokens:6d}  {str(stats['pass']):>7}  {stats['max_abs']:9.2e}  "
            f"{trt_us:14.2f} us  {b10_us:13.2f} us  {ratio:5.2f}x",
            flush=True,
        )
        if not stats["pass"]:
            raise AssertionError(
                f"SiTU mismatch at T={tokens}: max_abs={stats['max_abs']:.3e}"
            )


if __name__ == "__main__":
    main()
