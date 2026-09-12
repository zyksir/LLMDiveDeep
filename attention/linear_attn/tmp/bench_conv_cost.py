#!/usr/bin/env python3
"""Packed conv1d-update cost vs the b10 KDA decode kernel.

Decides whether fusing the width-4 causal conv into the decode kernel is
worth it: if the separate conv launch costs >=10% of the decode kernel at
production shapes, fusion pays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/
from common.kernel_bench import bench_cuda, print_table
from kda.inputs import make_decode_inputs
from kda.kda_decode_register import KDA_DECODE
from linear_attention import Shape

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)


def make_conv_runner(batch: int, heads: int, head_dim: int = 128):
    """The production packed decode conv: one launch over [B, 3*H*K]."""
    dim = 3 * heads * head_dim
    dev = "cuda"
    x = torch.randn(batch, dim, device=dev, dtype=torch.bfloat16)
    # pool stored [slots, state_len, dim], passed dim-contiguous like sglang
    conv_state = torch.zeros(
        batch, 3, dim, device=dev, dtype=torch.bfloat16
    ).transpose(-1, -2)
    weight = 0.1 * torch.randn(dim, 4, device=dev, dtype=torch.bfloat16)
    indices = torch.arange(batch, device=dev, dtype=torch.int32)

    def run():
        return causal_conv1d_update(
            x,
            conv_state,
            weight,
            None,
            activation="silu",
            conv_state_indices=indices,
        )

    return run


def main():
    rows = []
    for heads in (12, 96):
        shape = Shape(heads, heads, 128, 128, "float32")
        for batch in (1, 8, 32, 128):
            inputs = make_decode_inputs(batch, shape, seed=batch)
            built, unavailable = KDA_DECODE.build(
                inputs, shape, only=["b10_kda_decode"]
            )
            if "b10_kda_decode" not in built:
                raise RuntimeError(unavailable["b10_kda_decode"])
            decode_us = bench_cuda(built["b10_kda_decode"])
            conv_us = bench_cuda(make_conv_runner(batch, heads))
            rows.append(
                {
                    "heads": heads,
                    "batch": batch,
                    "conv_us": round(conv_us, 3),
                    "decode_us": round(decode_us, 3),
                    "conv/decode %": round(100 * conv_us / decode_us, 1),
                }
            )
    print_table(rows, title="packed conv1d update vs b10 KDA decode")


if __name__ == "__main__":
    main()
