#!/usr/bin/env python3
"""Exercise the B10 copy-engine all-reduce implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.collective import Collectives
from communication.kernel_benchmarks.common import (
    exercise,
    init,
    report_latency,
    skip,
)


def main() -> None:
    rank, world = init()
    if world not in (2, 4, 8):
        skip("B10 copy-engine benchmark needs TP2/4/8", rank)
        return
    rows, hidden = max(16, world * 2), 1024
    comm = Collectives(
        dist.group.WORLD,
        rows * hidden,
        enable_flashinfer=False,
        enable_trt=False,
        enable_b10=True,
        lazy_autotune=False,
    )

    def make(step):
        gen = torch.Generator(device="cuda").manual_seed(1000 * rank + step)
        return (torch.randn(
            rows, hidden, generator=gen, device="cuda", dtype=torch.bfloat16
        ),)

    def reference(x):
        out = x.clone()
        dist.all_reduce(out)
        return out

    static = tuple(torch.empty_like(x) for x in make(0))
    run = lambda x: comm.all_reduce(x, impl="b10_copy_engine")
    exercise(static, run, reference, make, atol=0.25)
    sample = make(99)[0]
    report_latency(
        "b10_copy_engine",
        lambda: run(sample),
        lambda: reference(sample),
        rank,
    )


if __name__ == "__main__":
    main()
