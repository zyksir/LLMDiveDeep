#!/usr/bin/env python3
"""How does GEMM time scale with batch for the layer's shapes?

The decode analysis assumes the dense GEMMs are ~FLAT in bs
(weight-read bound: [bs,7168]x[7168,3584] reads the same 51 MB whether
bs is 1 or 80). This sweep measures exactly where that stops being
true for each of the layer's GEMM shapes - the transition bs where
activations/compute start to matter - and therefore how the fc1-shard
and tail crossovers should move for the bs=1..80 target range.

Columns: time, achieved TB/s counting WEIGHT bytes only (flat regime
=> constant time, rising "TB/s" is fictitious once compute-bound),
and achieved TFLOP/s (meaningful in the compute regime).

  CUDA_VISIBLE_DEVICES=<free gpu> python3 debug/gemm_bs_sweep.py
  ... --bs 1 2 4 8 16 32 48 64 80 128 256 1024 4096
"""
import argparse

import torch

SHAPES = [
    ("fc1_full", 7168, 3584),   # the user's asked-for case (51 MB)
    ("fc1_shard", 7168, 448),   # per-rank shard (6.4 MB)
    ("fc2_full", 3584, 7168),   # ref-tail fc2 (51 MB)
    ("fc2_shard", 448, 7168),   # shard-tail slice... K=448 variant
    ("gate", 7168, 896),
    ("merge3", 7168, 2880),     # [gate|fc1shard|shared g/u]
    ("shared_gu", 7168, 1536),
]


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay()
    torch.cuda.synchronize()
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32, 48, 64, 80,
                             128, 256, 512, 1024, 4096])
    args = ap.parse_args()
    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda").manual_seed(7)

    def rn(*s, sc=0.02):
        return (torch.randn(*s, generator=gen, device="cuda",
                            dtype=torch.float32) * sc).to(torch.bfloat16)

    for name, k, n in SHAPES:
        w = rn(n, k)
        wbytes = n * k * 2
        print(f"\n== {name}: [bs,{k}] x [{k},{n}]  "
              f"(weight {wbytes / 1e6:.1f} MB) ==")
        print(f"{'bs':>6} {'us':>9} {'w-TB/s':>7} {'TFLOP/s':>8} "
              f"{'vs bs=1':>8}")
        t1 = None
        for b in args.bs:
            a = rn(b, k, sc=0.1)
            t = bench(lambda: a @ w.T)
            if t1 is None:
                t1 = t
            fl = 2 * b * k * n
            print(f"{b:>6} {t:>9.2f} {wbytes / t / 1e6:>7.2f} "
                  f"{fl / t / 1e9:>8.1f} {t / t1:>7.2f}x")


if __name__ == "__main__":
    main()
