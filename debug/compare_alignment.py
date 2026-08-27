#!/usr/bin/env python3
"""Side-by-side kernel alignment: TRT serving MoE-sublayer window vs one
LLMDiveDeep bench-layer iteration.

Usage:
  python3 debug/compare_alignment.py \
      /node-storage/var/traces/trt-decode-stock-bs8-isl512-conc8-rank0.json \
      kimi_k3_layer/local_results/moe_tp8_b8_baseline_graph.trace.json
"""
import json
import sys
from collections import Counter

CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def short(name: str) -> str:
    for pat, tag in (
        ("moefinalize_allreduce", "moefinalize+AR (fused)"),
        ("allreduce_fusion", "AR oneshot"),
        ("finalizeKernel", "finalize (plain)"),
        ("routingIndicesBlockScores", "routing blockScores"),
        ("routingIndicesCluster", "routing cluster"),
        ("routingIndicesSmallB", "routing smallB"),
        ("route_radix", "routing radix"),
        ("bmm_MxE4m3", "expert bmm fc1"),
        ("bmm_Bfloat16", "expert bmm fc2"),
        ("quantize_with_block_size", "mxfp8 quantize"),
        ("ag_mxfp8", "col-AG+quant (fused)"),
        ("_situ_and_mul", "situ_and_mul"),
        ("DualOutGemm", "DualOutGemm"),
        ("splitKreduce", "splitK reduce"),
        ("multimem_all_reduce", "multimem AR"),
        ("rmsnorm", "rmsnorm"),
        ("Memcpy", "memcpy"),
    ):
        if pat.lower() in name.lower():
            return tag
    if "nvjet" in name:
        core = name.split("nvjet_sm103_")[1][:28]
        return f"nvjet {core}"
    return name[:36]


def load(path: str):
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ks = sorted((e for e in ev if e.get("cat") in CATS and e.get("dur", 0) > 0),
                key=lambda e: e["ts"])
    return ks


def serve_moe_window(ks):
    """Median MoE sublayer: gate GEMM ... fc2_latent_proj, incl. aux-stream
    shared branch overlapping the window."""
    idx = [i for i, e in enumerate(ks) if "routingIndicesBlockScores" in e["name"]]
    mid = idx[len(idx) // 2]
    anchor_ts = ks[mid]["ts"]
    # window: 30us before blockScores to the fused tail + latent proj after
    t0, t1 = anchor_ts - 40, anchor_ts + 90
    return [e for e in ks if t0 <= e["ts"] <= t1]


def bench_iteration(ks):
    counts = Counter(e["name"] for e in ks)
    modal = Counter(counts.values()).most_common(1)[0][0]
    firsts = {}
    for e in ks:
        if counts[e["name"]] == modal and e["name"] not in firsts:
            firsts[e["name"]] = e["ts"]
    anchor = min(firsts, key=firsts.get)
    ts = [e["ts"] for e in ks if e["name"] == anchor]
    mid = len(ts) // 2
    return [e for e in ks if ts[mid] <= e["ts"] < ts[mid + 1]]


def show(label, events):
    lanes = sorted({e["tid"] for e in events})
    lane_of = {t: i for i, t in enumerate(lanes)}
    print(f"\n== {label}: {len(events)} kernels, "
          f"{sum(e['dur'] for e in events):.1f} us summed ==")
    t0 = min(e["ts"] for e in events)
    for e in events:
        print(f"  t={e['ts']-t0:6.1f} {e['dur']:6.1f}us s{lane_of[e['tid']]}  "
              f"{short(e['name'])}")


def main():
    serve, bench = sys.argv[1], sys.argv[2]
    show("TRT serving (MoE sublayer window)", serve_moe_window(load(serve)))
    show("LLMDiveDeep bench (one layer iteration)", bench_iteration(load(bench)))


if __name__ == "__main__":
    main()
