#!/usr/bin/env python3
"""Exercise the CuTe DSL GEMM+all-reduce kernel."""

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
        skip("CuTe DSL GEMM+AR needs sm100+ and TP2/4/8", rank)
        return
    rows, k, n = 128, 1024, 1024
    comm = Collectives(
        dist.group.WORLD,
        rows * n,
        boundary_max_tokens=rows,
        boundary_gemm_k=k,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )

    def make(step):
        gx = torch.Generator(device="cuda").manual_seed(4000 * rank + step)
        gw = torch.Generator(device="cuda").manual_seed(4100 + step)
        x = 0.02 * torch.randn(
            rows, k, generator=gx, device="cuda", dtype=torch.bfloat16
        )
        w = 0.02 * torch.randn(
            n, k, generator=gw, device="cuda", dtype=torch.bfloat16
        )
        dist.broadcast(w, src=0)
        return x, w

    def reference(x, w):
        out = torch.mm(x, w.T)
        dist.all_reduce(out)
        return out

    static = tuple(torch.empty_like(x) for x in make(0))
    run = lambda x, w: comm.gemm_allreduce(x, w, impl="cutedsl")
    exercise(static, run, reference, make, atol=0.35)
    sample = make(99)
    report_latency(
        "cutedsl_gemm_ar",
        lambda: run(*sample),
        lambda: reference(*sample),
        rank,
    )


if __name__ == "__main__":
    main()
