#!/usr/bin/env python3
"""Check and time the Kimi-K3 RMSNorm entry point against PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.kernel_bench import bench_cuda  # noqa: E402
from kimi_k3.config import MOE_LATENT, RMS_EPS  # noqa: E402
from kimi_k3.kernels.rmsnorm import rmsnorm  # noqa: E402


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + RMS_EPS) * weight.float()).to(
        x.dtype
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 128, 4096])
    args = parser.parse_args()
    for tokens in args.tokens:
        x = torch.randn(
            tokens, MOE_LATENT, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(MOE_LATENT, device="cuda", dtype=torch.bfloat16)
        expected = reference(x, weight)
        actual = rmsnorm(x, weight, RMS_EPS)
        if not torch.allclose(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        ):
            error = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f"RMSNorm mismatch at T={tokens}: {error}")
        ref_us = bench_cuda(lambda: reference(x, weight))
        fused_us = bench_cuda(lambda: rmsnorm(x, weight, RMS_EPS))
        print(
            f"T={tokens:<5} correct=True reference={ref_us:.2f} us "
            f"fused={fused_us:.2f} us speedup={ref_us / fused_us:.2f}x"
        )


if __name__ == "__main__":
    main()
