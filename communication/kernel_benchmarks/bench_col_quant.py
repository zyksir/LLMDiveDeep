#!/usr/bin/env python3
"""Exercise column all-gather with fused MXFP8 write-out."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.collective import Collectives
from communication.backends.col_quant_backend import nccl_all_gather_cols
from communication.kernel_benchmarks.common import (
    exercise,
    init,
    report_latency,
    skip,
)


def main() -> None:
    rank, world = init()
    output_columns = 4096
    if world not in (2, 4, 8) or output_columns % world:
        skip("col_quant benchmark needs TP2/4/8", rank)
        return
    try:
        import tensorrt_llm  # noqa: F401
    except (ImportError, OSError) as error:
        skip(f"TensorRT-LLM custom-op library unavailable: {error}", rank)
        return
    if not hasattr(torch.ops, "trtllm") or not hasattr(
        torch.ops.trtllm, "mxfp8_quantize"
    ):
        skip("torch.ops.trtllm.mxfp8_quantize unavailable", rank)
        return
    rows, cols = 8, output_columns // world
    comm = Collectives(
        dist.group.WORLD,
        rows * output_columns,
        col_ag_max_rows=rows,
        col_ag_max_output_columns=output_columns,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )

    def make(step):
        gen = torch.Generator(device="cuda").manual_seed(3000 * rank + step)
        return (0.1 * torch.randn(
            rows, cols, generator=gen, device="cuda", dtype=torch.bfloat16
        ),)

    def reference(x):
        gathered = nccl_all_gather_cols(x)
        quant, scales = torch.ops.trtllm.mxfp8_quantize(
            gathered.contiguous(), False)
        return quant, scales.view(rows, -1)

    sample = make(0)[0]
    comm.tune_col_ag(sample)
    static = (torch.empty_like(sample),)
    run = lambda x: comm.all_gather_col_quant(x)
    exercise(static, run, reference, make, atol=0.0)
    report_latency(
        "col_quant",
        lambda: run(sample),
        lambda: reference(sample),
        rank,
    )


if __name__ == "__main__":
    main()
