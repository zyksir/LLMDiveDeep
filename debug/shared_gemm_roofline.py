#!/usr/bin/env python3
"""Is the shared-expert chain at its HBM roofline at decode sizes?

The ablation's `-shared` probe says 11-14 us of the shared chain stays
EXPOSED at B=8..16 even with the aux-stream overlap on. Two very
different causes: (a) the chain is at the memory roofline and the bytes
simply cannot hide under the routed path's own DRAM traffic, or (b) the
tall-skinny GEMMs are launch/occupancy-limited and are nowhere near
roofline, in which case making them faster shrinks the exposure.

This times the two shared GEMMs standalone (one GPU, no contention) and
prints achieved bandwidth vs the weight bytes they must read.

  python3 debug/shared_gemm_roofline.py
"""
import torch

HIDDEN = 7168
SHARED_INTER_LOCAL = 6144 // 8  # TP8 shard of 16 x 384
PEAK_TBS = 8.0  # B200 HBM3e


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
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
    torch.cuda.set_device(0)
    dev, dt = "cuda", torch.bfloat16
    gu = torch.randn(2 * SHARED_INTER_LOCAL, HIDDEN, device=dev, dtype=dt)
    dn = torch.randn(HIDDEN, SHARED_INTER_LOCAL, device=dev, dtype=dt)
    # the merged routed input GEMM for reference (gate | fc1-shard)
    inw = torch.randn(896 + 3584 // 8, HIDDEN, device=dev, dtype=dt)
    inw_full = torch.randn(896 + 3584, HIDDEN, device=dev, dtype=dt)

    print(f"{'M':>3} {'gate_up us':>11} {'TB/s':>6} {'down us':>9} "
          f"{'TB/s':>6} {'chain us':>9} | {'in(shard) us':>13} {'TB/s':>6} "
          f"| {'in(full) us':>12} {'TB/s':>6}")
    for m in (1, 2, 4, 8, 16, 32, 64):
        h = torch.randn(m, HIDDEN, device=dev, dtype=dt)
        act = torch.randn(m, SHARED_INTER_LOCAL, device=dev, dtype=dt)
        t_gu = bench(lambda: h @ gu.T)
        t_dn = bench(lambda: act @ dn.T)
        t_in = bench(lambda: h @ inw.T)
        t_if = bench(lambda: h @ inw_full.T)
        bw = lambda w, t: w.numel() * 2 / t / 1e6  # noqa: E731  (TB/s)
        print(f"{m:>3} {t_gu:11.2f} {bw(gu, t_gu):6.2f} {t_dn:9.2f} "
              f"{bw(dn, t_dn):6.2f} {t_gu + t_dn:9.2f} | "
              f"{t_in:13.2f} {bw(inw, t_in):6.2f} | "
              f"{t_if:12.2f} {bw(inw_full, t_if):6.2f}")
    tot = (gu.numel() + dn.numel()) * 2
    print(f"\nshared weight bytes/rank: {tot / 1e6:.1f} MB -> "
          f"{tot / (PEAK_TBS * 1e12) * 1e6:.2f} us at {PEAK_TBS} TB/s")


if __name__ == "__main__":
    main()
