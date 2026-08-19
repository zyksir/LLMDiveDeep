#!/usr/bin/env python3
"""Exercise the CuTe DSL all-reduce+RMSNorm kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

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
        skip("CuTe DSL AR+norm needs sm100+ and TP2/4/8", rank)
        return
    rows, hidden = 128, 1024
    comm = Collectives(
        dist.group.WORLD,
        rows * hidden,
        boundary_max_tokens=rows,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )

    def make(step):
        gx = torch.Generator(device="cuda").manual_seed(5000 * rank + step)
        gr = torch.Generator(device="cuda").manual_seed(5100 + step)
        x = 0.02 * torch.randn(
            rows, hidden, generator=gx, device="cuda", dtype=torch.bfloat16
        )
        residual = 0.02 * torch.randn(
            rows, hidden, generator=gr, device="cuda", dtype=torch.bfloat16
        )
        gamma = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        dist.broadcast(residual, src=0)
        return x, gamma, residual

    def reference(x, gamma, residual):
        reduced = x.clone()
        dist.all_reduce(reduced)
        new_residual = reduced + residual
        norm = F.rms_norm(
            new_residual.float(), (hidden,), gamma.float(), 1e-5
        ).to(torch.bfloat16)
        return norm, new_residual

    static = tuple(torch.empty_like(x) for x in make(0))
    run = lambda x, gamma, residual: comm.allreduce_norm(
        x, gamma, 1e-5, residual=residual, impl="cutedsl"
    )
    exercise(static, run, reference, make, atol=0.25)
    sample = make(99)
    report_latency(
        "cutedsl_arnorm",
        lambda: run(*sample),
        lambda: reference(*sample),
        rank,
    )


if __name__ == "__main__":
    main()
