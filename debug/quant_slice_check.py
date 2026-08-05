#!/usr/bin/env python3
"""Bit-exactness + speed of quant_slice_mxfp8 vs the production pair
(contiguous-copy + torch.ops.trtllm.mxfp8_quantize) on a STRIDED
merged-GEMM latent slice.

  CUDA_VISIBLE_DEVICES=<free> python3 debug/quant_slice_check.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorrt_llm  # noqa: F401,E402  (registers torch.ops.trtllm)
from kimi_k3_layer.quant_slice import quant_slice_mxfp8  # noqa: E402

E, LAT = 896, 3584


def bench(fn, iters=300):
    for _ in range(30):
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
    gen = torch.Generator(device="cuda").manual_seed(9)
    print(f"{'bs':>4} {'copy+quant':>11} {'fused':>7} {'save':>6} "
          f"{'bitexact':>9}")
    for bs in (1, 2, 4, 8, 16, 32, 64, 80):
        gf = (torch.randn(bs, E + LAT + 1536, generator=gen,
                          device="cuda", dtype=torch.float32) * 0.5
              ).to(torch.bfloat16)
        sl = gf[:, E:E + LAT]  # strided latent slice

        def ref():
            return torch.ops.trtllm.mxfp8_quantize(sl.contiguous(),
                                                   False)

        q_ref, sf_ref = ref()
        q, sf = quant_slice_mxfp8(sl)
        ok = torch.equal(q.view(torch.uint8),
                         q_ref.view(torch.uint8).view(bs, LAT)) and \
            torch.equal(sf.flatten(),
                        sf_ref.view(torch.uint8).flatten())
        t_ref = bench(ref)
        t_new = bench(lambda: quant_slice_mxfp8(sl))
        print(f"{bs:>4} {t_ref:>11.2f} {t_new:>7.2f} "
              f"{t_ref - t_new:>+6.2f} {str(ok):>9}")


if __name__ == "__main__":
    main()
