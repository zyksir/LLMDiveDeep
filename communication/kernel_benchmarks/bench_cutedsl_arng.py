#!/usr/bin/env python3
"""Exercise the CuTe DSL all-reduce+RMSNorm+GEMM kernel."""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out-features", type=int, default=1024)
    args = ap.parse_args()
    rank, world = init()
    if torch.cuda.get_device_capability()[0] < 10 or world not in (2, 4, 8):
        skip("CuTe DSL AR+norm+GEMM needs sm100+ and TP2/4/8", rank)
        return
    rows, hidden, n = args.rows, args.hidden, args.out_features
    max_rows = args.max_rows or rows
    comm = Collectives(
        dist.group.WORLD,
        max_rows * hidden,
        boundary_max_tokens=max_rows,
        boundary_gemm_n=n,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )

    def make(step):
        gx = torch.Generator(device="cuda").manual_seed(6000 * rank + step)
        common = torch.Generator(device="cuda").manual_seed(6100 + step)
        x = 0.02 * torch.randn(
            rows, hidden, generator=gx, device="cuda", dtype=torch.bfloat16
        )
        residual = 0.02 * torch.randn(
            rows, hidden, generator=common, device="cuda", dtype=torch.bfloat16
        )
        gamma = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        w = 0.02 * torch.randn(
            n, hidden, generator=common, device="cuda", dtype=torch.bfloat16
        )
        dist.broadcast(residual, src=0)
        dist.broadcast(w, src=0)
        return x, residual, gamma, w

    def reference(x, residual, gamma, w):
        reduced = x.clone()
        dist.all_reduce(reduced)
        new_residual = reduced + residual
        norm = F.rms_norm(
            new_residual.float(), (hidden,), gamma.float(), 1e-5
        ).to(torch.bfloat16)
        return torch.mm(norm, w.T), new_residual

    static = tuple(torch.empty_like(x) for x in make(0))
    run = lambda x, residual, gamma, w: comm.allreduce_norm_gemm(
        x, residual, gamma, w, 1e-5, impl="cutedsl"
    )
    exercise(static, run, reference, make, atol=0.4)
    sample = make(99)
    report_latency(
        "cutedsl_arng",
        lambda: run(*sample),
        lambda: reference(*sample),
        rank,
    )


if __name__ == "__main__":
    main()
