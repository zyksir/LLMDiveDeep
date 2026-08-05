#!/usr/bin/env python3
"""Decode benchmark for isolated KDA attention modules (no MoE/collectives).

What this isolates: the *full attention module* around the KDA core —
input/gate projections + short causal conv + KDA recurrence + sigmoid-gated
RMSNorm + output projection. Compare against the kernel-level benches to
attribute decode time to the surrounding GEMMs versus the KDA core itself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from kda.attention_modules import AttentionShape, get_attention_runners
from common.kernel_bench import bench_cuda, plot_rows


def profile(fn: Callable, path: Path, steps: int) -> None:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        acc_events=True,
    ) as prof:
        for _ in range(steps):
            fn()
            prof.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=12,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument("--backends", nargs="*", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--csv", type=Path, default=Path("results/bench_attention_layer.csv")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="output PNG path (default: CSV path with .png suffix)",
    )
    parser.add_argument("--profile-backend")
    parser.add_argument("--profile-batch", type=int, default=32)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument(
        "--trace", type=Path, default=Path("results/kda_attention_trace.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shape = AttentionShape(
        hidden_size=args.hidden_size,
        heads=args.heads,
        head_dim=args.head_dim,
        conv_size=args.conv_size,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Scope: projection + short conv + KDA + gated norm + output projection")
    print(f"{'backend':>26} {'B':>5} {'latency us':>12} {'tok/s':>12}")
    rows = []
    for batch in args.batch_sizes:
        runners, unavailable = get_attention_runners(batch, shape, seed=batch)
        for name, reason in unavailable.items():
            print(f"  unavailable {name}: {reason}")
        for backend, fn in runners.items():
            if args.backends and backend not in args.backends:
                continue
            latency_us = bench_cuda(fn, args.warmup, args.iters, args.repeats)
            row = {
                "backend": backend,
                "batch": batch,
                "latency_us": latency_us,
                "tokens_s": batch * 1e6 / latency_us,
                "hidden_size": shape.hidden_size,
                "heads": shape.heads,
                "head_dim": shape.head_dim,
                "conv_size": shape.conv_size,
                "activation_dtype": "bfloat16",
                "state_dtype": "float32",
            }
            rows.append(row)
            print(
                f"{backend:>26} {batch:5d} {latency_us:12.2f} "
                f"{row['tokens_s']:12.0f}"
            )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {args.csv}")
    if rows:
        figure = plot_rows(
            rows,
            args.figure or args.csv.with_suffix(".png"),
            x="batch",
            y="latency_us",
            series="backend",
            suptitle="Isolated KDA attention module decode",
        )
        print(f"Wrote {figure}")

    if args.profile_backend:
        runners, unavailable = get_attention_runners(args.profile_batch, shape, seed=7)
        if args.profile_backend not in runners:
            raise RuntimeError(
                f"{args.profile_backend!r} unavailable; choices={list(runners)}, "
                f"errors={unavailable}"
            )
        profile(runners[args.profile_backend], args.trace, args.profile_steps)
        print(f"Wrote attention trace: {args.trace}")


if __name__ == "__main__":
    main()
