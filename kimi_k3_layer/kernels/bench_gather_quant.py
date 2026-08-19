#!/usr/bin/env python3
"""gather_quant correctness + performance (1 GPU, ~60 s).

Correctness: BIT-EXACT gate vs the unfused oracle
interleave -> ``torch.ops.trtllm.mxfp8_quantize`` (payload AND scales)
at every size. Performance: CUDA-graph medians, fused one-pass vs the
unfused interleave copy + trtllm quantize.

  CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/kernels/bench_gather_quant.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kimi_k3_layer.kernels.gather_quant import gather_quant_mxfp8  # noqa: E402

WORLD, WIDTH = 8, 448


def graph_us(fn, calls=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / calls)
    del g
    return statistics.median(ts)


def main() -> None:
    import tensorrt_llm  # noqa: F401  (registers trtllm ops)

    torch.manual_seed(0)
    print(f"{'B':>6} {'unfused (copy+quant)':>21} {'fused':>8} "
          f"{'saved':>7} {'bit-exact':>9}")
    for batch in (512, 1024, 2048, 4096, 8192, 16384):
        gathered = torch.randn(WORLD * batch, WIDTH, device="cuda",
                               dtype=torch.bfloat16)
        # oracle: interleave copy then the stock quantize
        def interleave():
            return gathered.view(WORLD, batch, WIDTH) \
                .permute(1, 0, 2).reshape(batch, WORLD * WIDTH)

        # oracle = exactly what quantize_input runs: LINEAR sf layout
        ref_fp8, ref_sf = torch.ops.trtllm.mxfp8_quantize(
            interleave(), False, alignment=32)
        out_fp8, out_sf = gather_quant_mxfp8(gathered, WORLD)
        torch.cuda.synchronize()
        exact = (torch.equal(ref_fp8.view(torch.uint8),
                             out_fp8.view(torch.uint8))
                 and torch.equal(ref_sf.view(torch.uint8).flatten(),
                                 out_sf.flatten()))

        t_unfused = graph_us(
            lambda: torch.ops.trtllm.mxfp8_quantize(
                interleave().contiguous(), False, alignment=32))
        t_fused = graph_us(lambda: gather_quant_mxfp8(gathered, WORLD))
        print(f"{batch:>6} {t_unfused:21.1f} {t_fused:8.1f} "
              f"{t_unfused - t_fused:+7.1f} {str(exact):>9}")
    print("PASS bench_gather_quant")


if __name__ == "__main__":
    main()
