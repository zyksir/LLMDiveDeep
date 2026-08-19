#!/usr/bin/env python3
"""Verify and time Kimi-K3 B10 prefill against the TRT-LLM chunk pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kimi_k3_layer.bench_b10_kimi_k3_kda_layer import (  # noqa: E402
    bootstrap,
    build_layer,
    check_prefill_correctness,
    make_prefill_input,
    run_layer,
    time_eager,
)
from kimi_k3_layer.config import k3_shard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=["tp1", "tp8"], default="tp8")
    parser.add_argument("--tokens", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument(
        "--backends", nargs="+", default=["trt_prefill", "b10_prefill"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    bootstrap()
    shard = k3_shard(args.shard)
    if min(args.tokens) <= 128:
        parser.error("prefill benchmark token counts must be > 128")
    check_prefill_correctness(shard, args.tokens[0], layer_idx=1)
    for tokens in args.tokens:
        hidden, cu_seqlens = make_prefill_input(tokens, shard.hidden)
        for backend in args.backends:
            layer = build_layer(
                backend, tokens, shard, layer_idx=1, seed=1000 + tokens
            )
            run = lambda: run_layer(layer, hidden, cu_seqlens)
            latency = time_eager(run, args.warmup, args.iters, args.repeats)
            print(
                f"T={tokens:<6} {backend:>12}: {latency:.2f} us "
                f"({tokens * 1e6 / latency:.0f} tok/s)"
            )


if __name__ == "__main__":
    main()
