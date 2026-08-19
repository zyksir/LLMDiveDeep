#!/usr/bin/env python3
"""Exercise graph-safe and host-state B10 multimem variants."""

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
    if torch.cuda.get_device_capability()[0] < 10 or world not in (2, 4, 8):
        skip("b10_multimem needs sm100+ and TP2/4/8", rank)
        return
    rows, hidden = max(16, world * 2), 1024
    comm = Collectives(
        dist.group.WORLD,
        rows * hidden,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )
    if not comm._multimem_state():
        skip("b10_multimem backend unavailable", rank)
        return

    def make(step):
        gen = torch.Generator(device="cuda").manual_seed(2000 * rank + step)
        return (torch.randn(
            rows, hidden, generator=gen, device="cuda", dtype=torch.bfloat16
        ),)

    def reference(x):
        out = x.clone()
        dist.all_reduce(out)
        return out

    static = tuple(torch.empty_like(x) for x in make(0))
    safe = lambda x: comm.all_reduce(x, impl="b10_multimem")
    exercise(static, safe, reference, make, atol=0.25)
    unsafe = lambda x: comm.all_reduce(x, impl="b10_multimem:lamport")
    exercise(static, unsafe, reference, make, atol=0.25, graph_safe=False)
    sample = make(99)[0]
    report_latency(
        "b10_multimem",
        lambda: safe(sample),
        lambda: reference(sample),
        rank,
    )
    if rank == 0:
        print("PASS b10_multimem:lamport: eager-safe, graph capture rejected")


if __name__ == "__main__":
    main()
