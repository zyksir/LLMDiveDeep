#!/usr/bin/env python3
"""Is FlashKDA's K1 (prepare) / K2 (recurrence) memory- or compute-bound?

FlashKDA's single ``fwd`` binding launches two kernels that cannot be invoked
separately, but the CUDA profiler attributes device time to each. This bench
combines those per-kernel times with an analytic traffic/FLOP model of what
each kernel provably reads, writes, and computes (layout in
``csrc/flash_kda.cpp`` and KDA.md "K1/K2" sections) to report achieved GB/s
and TFLOP/s against machine peaks, plus grid occupancy — enough to classify
each kernel as memory-bound, compute-bound, or (the interesting case)
neither: parallelism/latency-bound.

Per 16-token tile per head (C=16, K=V=128, all operands bf16):

  K1  reads q,k,g tiles (3 x 16x128 bf16) + beta; writes the workspace
      (k_decayed/q_decayed/k_restored 16x128 bf16, g_total 128 fp32,
      INV/Mqk 16x16 bf16)                              ~26 KB
      MMAs: L and Mqk (2 x [16,128]@[128,16]) + Neumann ladder
      (~6 x 16^3 fp16 MMAs) + elementwise               ~0.24 MFLOP
      => arithmetic intensity ~9 FLOP/B  (machine balance ~280)

  K2  reads the workspace + v tile, writes the o tile   ~22 KB
      MMAs: Kd@S0, Qd@S0, Kg^T@Vnew (3 x [16,128]@[128,128])
      + M/Mqk solves (2 x [16,16]@[16,128]) + decay      ~1.77 MFLOP
      => arithmetic intensity ~79 FLOP/B
      (the 128x128 state lives in SMEM for the whole sequence: zero HBM
      traffic per tile; initial read + final write amortized below)

Usage: CUDA_VISIBLE_DEVICES=7 ../.venv/bin/python bench_flashkda_k1k2.py
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from kda_attention import KDA_PREFILL, make_prefill_inputs
from linear_attention import Shape

C, H, K, V = 16, 16, 128, 128
ITERS = 20
# B200 peaks: HBM3e ~8.0 TB/s; dense bf16 ~2.25 PFLOP/s
PEAK_GBPS = 8000.0
PEAK_TFLOPS = 2250.0
SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count

SHAPES = [(1, 8192), (4, 8192), (4, 65536), (16, 65536), (64, 4096), (64, 65536)]


def k1_model(tiles: int):
    """(bytes, flops) for _flash_kda_fwd_prepare over `tiles` tile-heads."""
    rd = 3 * C * K * 2 + C * 2                    # q, k, g bf16 + beta
    wr = 3 * C * K * 2 + K * 4 + 2 * C * C * 2    # kd/qd/krest + g_total + INV/Mqk
    mma = 2 * (2 * C * C * K) + 6 * (2 * C**3)    # L, Mqk + Neumann ladder
    elem = 15 * 2 * C * K                          # norms, gate, cumsum, decays
    return tiles * (rd + wr), tiles * (mma + elem)


def k2_model(tiles: int, seqs: int):
    """(bytes, flops) for _flash_kda_fwd_recurrence."""
    rd = 3 * C * K * 2 + 2 * C * C * 2 + K * 4 + C * V * 2 + C * 2  # ws + v
    wr = C * V * 2                                                   # o tile
    state = seqs * H * K * V * 4 * 2                                 # S0 in + S_T out
    mma = 3 * (2 * C * K * V) + 2 * (2 * C * C * V)                  # KdS0,QdS0,KgVnew + solves
    elem = 3 * 2 * C * V + 2 * K * V                                 # beta/residual + decay
    return tiles * (rd + wr) + state, tiles * (mma + elem)


def profile_flash_kda(batch: int, seq_len: int, shape: Shape):
    inputs = make_prefill_inputs(batch, seq_len, shape, seed=batch + seq_len)
    runners, unavailable = KDA_PREFILL.build(inputs, shape, only=["flash_kda"])
    if "flash_kda" not in runners:
        raise RuntimeError(f"flash_kda unavailable: {unavailable}")
    fn = runners["flash_kda"]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    times = {}
    for evt in prof.key_averages():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if "prepare" in evt.key:
            times["K1"] = evt.self_device_time_total / ITERS
        elif "recurrence" in evt.key:
            times["K2"] = evt.self_device_time_total / ITERS
    del inputs, runners
    torch.cuda.empty_cache()
    return times


def verdict(gb_pct: float, tf_pct: float) -> str:
    if max(gb_pct, tf_pct) < 40:
        return "NEITHER -> latency/parallelism-bound"
    return "MEMORY-bound" if gb_pct > tf_pct else "COMPUTE-bound"


def main():
    shape = Shape(H, H, K, V, "float32")
    hdr = (f"{'shape':>14} {'kern':>4} {'grid':>9} {'us':>8} "
           f"{'GB/s':>7} {'%peak':>6} {'TFLOP/s':>8} {'%peak':>6}  verdict")
    print(f"machine: {SM_COUNT} SMs, peaks {PEAK_GBPS:.0f} GB/s, {PEAK_TFLOPS:.0f} TFLOP/s bf16")
    print(hdr)
    for batch, seq_len in SHAPES:
        try:
            times = profile_flash_kda(batch, seq_len, shape)
        except torch.OutOfMemoryError:
            print(f"{f'B={batch} S={seq_len}':>14}  skipped (OOM)")
            torch.cuda.empty_cache()
            continue
        tiles = batch * seq_len // C * H
        models = {
            "K1": (k1_model(tiles), tiles),              # grid = all tile-heads
            "K2": (k2_model(tiles, batch), batch * H),   # grid = seqs x heads
        }
        for kern in ("K1", "K2"):
            (nbytes, nflops), grid = models[kern]
            us = times.get(kern)
            if us is None:
                print(f"{f'B={batch} S={seq_len}':>14} {kern:>4}  kernel not found in profile")
                continue
            gbps = nbytes / us / 1e3
            tflops = nflops / us / 1e6
            gb_pct, tf_pct = 100 * gbps / PEAK_GBPS, 100 * tflops / PEAK_TFLOPS
            print(f"{f'B={batch} S={seq_len}':>14} {kern:>4} {grid:>9} {us:>8.1f} "
                  f"{gbps:>7.0f} {gb_pct:>5.1f}% {tflops:>8.1f} {tf_pct:>5.1f}%  "
                  f"{verdict(gb_pct, tf_pct)}")


if __name__ == "__main__":
    main()
