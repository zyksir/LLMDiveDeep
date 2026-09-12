#!/usr/bin/env python3
"""Verify and time the Kimi-K3 B10 decode kernels against TRT-LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kimi_k3.bench_b10_kimi_k3_kda_layer import (  # noqa: E402
    bootstrap,
    build_layer,
    check_decode_correctness,
    run_layer,
    time_graph,
)
from kimi_k3.config import k3_shard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=["tp1", "tp8"], default="tp8")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["trt_fused", "b10_fused", "b10_overlap"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    bootstrap()
    shard = k3_shard(args.shard)
    check_decode_correctness(shard, layer_idx=1)
    for tokens in args.tokens:
        if tokens > 128:
            parser.error("decode benchmark token counts must be <= 128")
        for backend in args.backends:
            layer = build_layer(
                backend, tokens, shard, layer_idx=1, seed=1000 + tokens
            )
            run = lambda: run_layer(layer)
            latency = time_graph(run, args.warmup, args.iters, args.repeats)
            print(f"T={tokens:<4} {backend:>12}: {latency:.2f} us")


if __name__ == "__main__":
    main()
