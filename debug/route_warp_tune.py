#!/usr/bin/env python3
"""Quick standalone tune of the route_pack kernel launch config.

The kernel is 16 dependent block-wide max reductions; NUM_WARPS=4 was
tuned on the OLD node. Sweep warps (and stages) here on bf16 strided
logits at the layer's decode sizes.

  python3 debug/route_warp_tune.py
"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorrt_llm  # noqa: F401,E402
from kimi_k3_layer.routing_kimi_k3 import (  # noqa: E402
    _kimik3_route_pack_kernel,
)
from kimi_k3_layer.config import (  # noqa: E402
    NUM_EXPERTS, ROUTED_SCALING, TOP_K,
)


def bench(fn, iters=400):
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
    gen = torch.Generator(device="cuda").manual_seed(7)
    bias = torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.02
    print(f"{'B':>4} " + " ".join(f"nw={w:<2}" for w in (2, 4, 8, 16)))
    for b in (1, 2, 4, 8, 16):
        # strided logits like the layer's merged-GEMM slice
        gf = torch.randn(b, 2880, generator=gen, device="cuda",
                         dtype=torch.float32).to(torch.bfloat16)
        logits = gf[:, :NUM_EXPERTS]
        ids = torch.empty(b, TOP_K, device="cuda", dtype=torch.int32)
        sc = torch.empty(b, TOP_K, device="cuda", dtype=torch.bfloat16)
        row = []
        for nw in (2, 4, 8, 16):
            def fn(nw=nw):
                _kimik3_route_pack_kernel[(b,)](
                    logits, bias, ids, sc, ids, sc,
                    logits.stride(0),
                    E=NUM_EXPERTS, K=TOP_K, SCALE=ROUTED_SCALING,
                    BLOCK=1024, EMIT_PACKED=False, EMIT_FUSED=True,
                    num_warps=nw,
                )
            row.append(bench(fn))
        print(f"{b:>4} " + " ".join(f"{t:5.2f}" for t in row))


if __name__ == "__main__":
    main()
