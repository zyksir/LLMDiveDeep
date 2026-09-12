#!/usr/bin/env python3
"""Check and time the merged-projection split against PyTorch operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.kernel_bench import bench_cuda  # noqa: E402
from kimi_k3.config import (  # noqa: E402
    MOE_LATENT,
    NUM_EXPERTS,
    SHARED_INTER,
)
from kimi_k3.kernels.split_projection import (  # noqa: E402
    split_merged_projection_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 80])
    args = parser.parse_args()
    for tokens in args.tokens:
        source_latent = MOE_LATENT
        output_latent = MOE_LATENT
        merged = torch.randn(
            tokens,
            NUM_EXPERTS + source_latent + 2 * SHARED_INTER,
            device="cuda",
            dtype=torch.bfloat16,
        )
        latent = torch.empty(
            tokens, output_latent, device="cuda", dtype=torch.bfloat16
        )
        activation = torch.empty(
            tokens, SHARED_INTER, device="cuda", dtype=torch.bfloat16
        )

        def run():
            block = 1024
            split_merged_projection_kernel[
                (tokens, triton.cdiv(output_latent + SHARED_INTER, block))
            ](
                merged,
                latent,
                activation,
                merged.stride(0),
                E=NUM_EXPERTS,
                L_SRC=source_latent,
                L_OUT=output_latent,
                I=SHARED_INTER,
                LATENT_STRIDE=latent.stride(0),
                ACT_STRIDE=activation.stride(0),
                BLOCK=block,
                num_warps=4,
            )
            return latent, activation

        ref_latent = merged[:, NUM_EXPERTS : NUM_EXPERTS + output_latent]
        gate, up = merged[:, NUM_EXPERTS + source_latent :].chunk(2, dim=-1)
        ref_activation = F.silu(gate.float()).mul(up.float()).to(torch.bfloat16)
        out_latent, out_activation = run()
        if not torch.equal(out_latent, ref_latent):
            raise AssertionError(f"latent split mismatch at tokens={tokens}")
        if not torch.allclose(
            out_activation.float(),
            ref_activation.float(),
            atol=2e-2,
            rtol=2e-2,
        ):
            raise AssertionError(f"activation mismatch at tokens={tokens}")

        ref_us = bench_cuda(
            lambda: (
                merged[:, NUM_EXPERTS : NUM_EXPERTS + output_latent].contiguous(),
                F.silu(gate.float()).mul(up.float()).to(torch.bfloat16),
            )
        )
        fused_us = bench_cuda(run)
        print(
            f"T={tokens:<4} correct=True reference={ref_us:.2f} us "
            f"fused={fused_us:.2f} us speedup={ref_us / fused_us:.2f}x"
        )


if __name__ == "__main__":
    main()
