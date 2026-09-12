#!/usr/bin/env python3
"""Check and time fused AttnRes against a direct PyTorch implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.kernel_bench import bench_cuda  # noqa: E402
from kimi_k3.config import (  # noqa: E402
    HIDDEN,
    NUM_ATTN_RES_BLOCKS,
    RMS_EPS,
)
from kimi_k3.kernels.attn_res_cutedsl import attn_res  # noqa: E402


def reference(
    prefix,
    delta,
    blocks,
    norm_weight,
    projection_weight,
    output_weight,
    num_blocks,
    block_write_idx,
):
    prefix.add_(delta)
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(prefix)
    candidates = torch.cat(
        (blocks[:, :num_blocks], prefix[:, None]), dim=1
    ).float()
    normalized = candidates * torch.rsqrt(
        candidates.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    scores = (normalized * norm_weight.float() * projection_weight.float()).sum(
        dim=-1
    )
    mixed = (
        torch.softmax(scores, dim=-1)[..., None] * candidates
    ).sum(dim=1)
    return (
        mixed
        * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + RMS_EPS)
        * output_weight.float()
    ).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 128])
    args = parser.parse_args()
    for tokens in args.tokens:
        tensors = [
            torch.randn(
                tokens,
                HIDDEN,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        ]
        prefix, delta = tensors
        blocks = torch.randn(
            tokens,
            NUM_ATTN_RES_BLOCKS,
            HIDDEN,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weights = [
            torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        num_blocks = NUM_ATTN_RES_BLOCKS
        write_idx = 2
        ref_prefix, ref_blocks = prefix.clone(), blocks.clone()
        expected = reference(
            ref_prefix,
            delta,
            ref_blocks,
            *weights,
            num_blocks,
            write_idx,
        )
        out_prefix, out_blocks = prefix.clone(), blocks.clone()
        actual = attn_res(
            out_prefix,
            delta,
            out_blocks,
            *weights,
            num_blocks,
            write_idx,
            eps=RMS_EPS,
            out_eps=RMS_EPS,
            reuse_out=False,
        )
        correct = (
            torch.allclose(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
            and torch.equal(out_prefix, ref_prefix)
            and torch.equal(out_blocks, ref_blocks)
        )
        if not correct:
            raise AssertionError(f"AttnRes mismatch at tokens={tokens}")

        ref_us = bench_cuda(
            lambda: reference(
                ref_prefix,
                delta,
                ref_blocks,
                *weights,
                num_blocks,
                write_idx,
            )
        )
        fused_us = bench_cuda(
            lambda: attn_res(
                out_prefix,
                delta,
                out_blocks,
                *weights,
                num_blocks,
                write_idx,
                eps=RMS_EPS,
                out_eps=RMS_EPS,
            )
        )
        print(
            f"T={tokens:<4} correct=True reference={ref_us:.2f} us "
            f"fused={fused_us:.2f} us speedup={ref_us / fused_us:.2f}x"
        )


if __name__ == "__main__":
    main()
