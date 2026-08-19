#!/usr/bin/env python3
"""Build rank-specific CuTeDSL AOT objects for Kimi-K3 boundary shapes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.collective import Collectives
from communication.kernel_benchmarks.common import init


def main() -> None:
    rank, world = init()
    if torch.cuda.get_device_capability()[0] < 10 or world not in (2, 4, 8):
        raise RuntimeError("CuTeDSL prebuild requires sm100+ and TP2/4/8")

    max_rows, hidden, gemm_k, gemm_n = 16384, 7168, 896, 6288
    comm = Collectives(
        dist.group.WORLD,
        max_numel=max_rows * hidden,
        max_hidden=hidden,
        boundary_max_tokens=max_rows,
        boundary_gemm_k=gemm_k,
        boundary_gemm_n=gemm_n,
        enable_flashinfer=False,
        enable_trt=False,
        lazy_autotune=False,
    )
    device = torch.device("cuda", rank)
    gamma = torch.ones(hidden, dtype=torch.bfloat16, device=device)

    def build(label, fn) -> None:
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"[aot] {label}: {time.perf_counter() - started:.1f}s",
                  flush=True)

    build(
        "gemm_allreduce",
        lambda: comm.gemm_allreduce(
            torch.zeros(1, gemm_k, dtype=torch.bfloat16, device=device),
            torch.zeros(hidden, gemm_k, dtype=torch.bfloat16, device=device),
            impl="cutedsl",
        ),
    )
    build(
        "allreduce_norm",
        lambda: comm.allreduce_norm(
            torch.zeros(1, hidden, dtype=torch.bfloat16, device=device),
            gamma,
            1e-5,
            residual=torch.zeros(
                1, hidden, dtype=torch.bfloat16, device=device),
            impl="cutedsl",
        ),
    )
    if rank == 0:
        print("PASS CuTeDSL AOT prebuild", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
