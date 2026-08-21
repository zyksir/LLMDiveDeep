#!/usr/bin/env python3
"""B300 retune probe for the fused-front dual-out GEMM tile config.

Why: the TP8 B300 table reproduces B200 within +-1.6% everywhere except
B=256, which is 10.1% slower. The plan is not at fault -- the decode/prefill
boundary, the decode tail and the front *choice* were each swept on this node
and the shipped value already wins. What has NOT been re-decided here is the
config INSIDE the fused front:

    dual_out_gemm_cutedsl.py:995  "offline sweep on B200; fixed here"
    _TUNED: dict[tuple, Config] = {}          # empty -> DEFAULT_CONFIG
    _bucket(batch) -> min(max(b, 1), 128)     # caps at 128 < DECODE_MAX_TOKENS

so every batch > 9 runs DEFAULT_CONFIG (split_k=4, block_n=96, 6 stages),
swept on B200, with no arch in the key. This sweeps the config directly at
the sizes that matter and reports anything that beats the default.

Report-only, single GPU, no collectives:

    python3 debug/probe_dualout_config.py --batches 128 256
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kimi_k3_layer.kernels import dual_out_gemm_cutedsl as dg


def graph_us(fn, iters: int = 50, repeats: int = 5) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop) * 1000 / iters)
    return best


def candidates() -> list[dg.Config]:
    out = [dg.DEFAULT_CONFIG]
    for split_k, block_n, stages in itertools.product(
        (2, 4, 8), (64, 96, 128, 192), (6, 8)
    ):
        cfg = dg.Config(split_k=split_k, block_n=block_n, ab_stages=stages)
        if cfg != dg.DEFAULT_CONFIG:
            out.append(cfg)
    # split_k == 1 unlocks 2-CTA MMA and N-multicast clusters
    for block_n, two_cta, cluster_n in (
        (96, True, 1), (128, True, 1), (128, False, 2), (192, False, 2),
    ):
        out.append(dg.Config(split_k=1, block_n=block_n,
                             use_2cta=two_cta, cluster_n=cluster_n))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    weight = torch.randn(dg.N_TOTAL, dg.HIDDEN, device="cuda",
                         dtype=torch.bfloat16)
    for batch in args.batches:
        h = torch.randn(batch, dg.HIDDEN, device="cuda", dtype=torch.bfloat16)
        reference = dg.dual_out_gemm_cutedsl(
            h, weight, dg.N_BF16, dg.N_GATE, config=dg.DEFAULT_CONFIG)
        rows = []
        for cfg in candidates():
            try:
                got = dg.dual_out_gemm_cutedsl(
                    h, weight, dg.N_BF16, dg.N_GATE, config=cfg)
                # every config must produce the same numbers, not just be fast
                torch.testing.assert_close(got[0], reference[0],
                                           rtol=2e-2, atol=5e-1)
                torch.testing.assert_close(got[1], reference[1],
                                           rtol=2e-2, atol=5e-1)
                us = graph_us(
                    lambda c=cfg: dg.dual_out_gemm_cutedsl(
                        h, weight, dg.N_BF16, dg.N_GATE, config=c),
                    iters=args.iters)
                rows.append((us, cfg, "ok"))
            except Exception as error:  # noqa: BLE001 - probe: report and move on
                rows.append((float("inf"), cfg,
                             f"{type(error).__name__}: {str(error)[:60]}"))
        rows.sort(key=lambda row: row[0])
        base = next(us for us, cfg, _ in rows if cfg == dg.DEFAULT_CONFIG)
        print(f"\n=== B={batch}  (DEFAULT_CONFIG = {base:.2f} us) ===",
              flush=True)
        for us, cfg, status in rows[:10]:
            if status != "ok":
                print(f"  {'--':>8}    {status:<28} "
                      f"sk={cfg.split_k} bn={cfg.block_n}", flush=True)
                continue
            mark = " <- DEFAULT" if cfg == dg.DEFAULT_CONFIG else ""
            print(f"  {us:>8.2f} us  {(base - us) / base * 100:>+6.1f}%  "
                  f"sk={cfg.split_k} bn={cfg.block_n} st={cfg.ab_stages} "
                  f"2cta={int(cfg.use_2cta)} cn={cfg.cluster_n}{mark}",
                  flush=True)


if __name__ == "__main__":
    main()
