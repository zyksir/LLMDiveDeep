#!/usr/bin/env python3
"""FUSE_FB A/B: in-kernel f_b GEMV vs separate cuBLAS GEMV + decode kernel.

Correctness: both compared against an fp32-gate oracle (the fused variant
skips the bf16 rounding of the intermediate, so it should be CLOSER to the
oracle, not merely close to the reference). Timing: CUDA-graph medians.
Runtime: ~60 s including the two per-shape CuTeDSL compiles.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kda.b10.b10_kda_decode_conv_gated_cutedsl import (  # noqa: E402
    kda_decode_conv_gated_raw,
)
from kda.b10.b10_kda_decode_conv_gated_fusefb_cutedsl import (  # noqa: E402
    kda_decode_conv_gated_raw_fusefb,
)

H, K = 12, 128


def graph_time_us(fn, calls=100, repeats=7):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / calls)
    del g
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    print(f"{'B':>5} {'gemv+kernel':>12} {'fusefb':>10} {'delta':>8} "
          f"{'ref-vs-oracle':>14} {'fused-vs-oracle':>16}")
    for B in (1, 2, 4, 8, 16, 32, 64, 128):
        q, k, v = (torch.randn(B, H, K, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        cw = torch.randn(3 * H * K, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        cs = torch.randn(B, 3 * H * K, 3, device="cuda", dtype=torch.bfloat16)
        f_a = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
        w_fb = torch.randn(H * K, K, device="cuda", dtype=torch.bfloat16) * 0.05
        beta = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(H, device="cuda") * 0.1
        dt = (torch.randn(H, K, device="cuda") * 0.1).contiguous()
        S0 = torch.randn(B, H, K, K, device="cuda")
        z = torch.randn(B, H, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(K, device="cuda")

        # oracle gate: fp32 GEMV, no intermediate rounding
        gate32 = (f_a.float() @ w_fb.float().t()).view(B, H, K)
        Sor = S0.clone()
        o_or = kda_decode_conv_gated_raw(
            q, k, v, cw, cs.clone(), gate32.to(torch.bfloat16), beta, A, dt,
            Sor, z, w)  # closest runnable oracle: fp32 GEMV rounded once

        raw_gate = (f_a @ w_fb.t()).view(B, H, K)
        Sr = S0.clone()
        o_ref = kda_decode_conv_gated_raw(
            q, k, v, cw, cs.clone(), raw_gate, beta, A, dt, Sr, z, w)
        Sf = S0.clone()
        o_fus = kda_decode_conv_gated_raw_fusefb(
            q, k, v, cw, cs.clone(), f_a, w_fb, beta, A, dt, Sf, z, w)
        torch.cuda.synchronize()
        d_ref = (o_ref.float() - o_or.float()).abs().max().item()
        d_fus = (o_fus.float() - o_or.float()).abs().max().item()

        csr, csf = cs.clone(), cs.clone()
        S1, S2 = S0.clone(), S0.clone()

        def ref_run():
            rg = (f_a @ w_fb.t()).view(B, H, K)
            kda_decode_conv_gated_raw(q, k, v, cw, csr, rg, beta, A, dt,
                                      S1, z, w)

        def fus_run():
            kda_decode_conv_gated_raw_fusefb(q, k, v, cw, csf, f_a, w_fb,
                                             beta, A, dt, S2, z, w)

        t_ref = graph_time_us(ref_run)
        t_fus = graph_time_us(fus_run)
        print(f"{B:>5} {t_ref:12.2f} {t_fus:10.2f} {t_fus - t_ref:+8.2f} "
              f"{d_ref:14.3e} {d_fus:16.3e}")


if __name__ == "__main__":
    main()
