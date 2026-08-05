#!/usr/bin/env python3
"""Can the vendored CuTeDSL tall-GEMM replace the KDA decode
projections? Standalone: in_proj [B,7168]x[7168,6288] and o_proj
[B,1536]x[1536,7168] at B=1..16, cuBLAS vs kernel, correctness+time."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorrt_llm  # noqa: F401,E402  (not needed, keeps env parity)
from kimi_k3_layer.tail_gemm_cutedsl import moe_tail_gemm  # noqa: E402


def bench(fn, iters=300):
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
    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda").manual_seed(11)

    def rn(*s, sc=0.02):
        return (torch.randn(*s, generator=gen, device="cuda",
                            dtype=torch.float32) * sc).to(torch.bfloat16)

    # in_proj N padded 6288 -> 6336 (=64*99): the layer already
    # slices in_proj_padding off, 48 more cols is free
    shapes = [("in_proj", 7168, 6336), ("o_proj", 1536, 7168)]
    print(f"{'gemm':>8} {'B':>3} {'cublas':>8} {'cutedsl':>8} "
          f"{'save':>6} {'rel_err':>9}")
    for name, k, n in shapes:
        w = rn(n, k)
        for b in (1, 2, 4, 8, 16):
            a = rn(b, k, sc=0.1)
            zero = torch.zeros(b, n, device="cuda", dtype=torch.bfloat16)
            ref = (a.float() @ w.t().float()).to(torch.bfloat16)
            out = moe_tail_gemm(a, w, zero)
            rel = ((out.float() - ref.float()).abs().max()
                   / ref.float().abs().max()).item()
            t_c = bench(lambda: a @ w.T)
            buf = torch.empty_like(zero)
            t_k = bench(lambda: moe_tail_gemm(a, w, zero, out=buf))
            print(f"{name:>8} {b:>3} {t_c:>8.2f} {t_k:>8.2f} "
                  f"{t_c - t_k:>+6.2f} {rel:>9.1e}")


if __name__ == "__main__":
    main()
