#!/usr/bin/env python3
"""Extract one steady-state iteration per trace as JSON timeline lanes.

Usage: python3 debug/extract_timeline.py OUT.json PREFIX1 [PREFIX2 ...]
Each PREFIX expands to <PREFIX>_{baseline,opt}_graph.trace.json.
"""
import json
import sys
from collections import Counter, defaultdict

CATS = ("kernel", "gpu_memcpy", "gpu_memset")

BUCKETS = (
    ("front_gemm", ("nvjet", "dualoutgemm", "dual_out", "cutlass_kernel_kimi",
                    "splitkreduce", "sm100_xmma", "matmul", "rmsnorm")),
    ("expert_gemm", ("bmm_", "fused_moe", "moe_gemm", "group_gemm", "situ",
                     "finalize")),
    ("routing", ("routing", "topk", "sort", "radix", "permut", "scan",
                 "histogram", "expandinputrows", "smallb")),
    ("quant_ag", ("ag_mxfp8", "quant", "fp8", "e4m3", "e2m1")),
    ("comm", ("nccl", "allreduce", "all_reduce", "multimem", "oneshot",
              "twoshot", "allgather", "all_gather", "symm", "lamport",
              "cross_device", "ar_fusion")),
    ("other", ()),
)


def bucket(name: str) -> str:
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def one_iteration(path: str) -> dict:
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") in CATS and e.get("dur", 0) > 0]
    kernels.sort(key=lambda e: e["ts"])
    # anchor = a kernel that fires exactly once per iteration. Kernel names
    # cluster at count == n_iterations; take the modal count, then the
    # earliest-starting name at that count so the window starts at the
    # iteration head.
    counts = Counter(e["name"] for e in kernels)
    modal = Counter(counts.values()).most_common(1)[0][0]
    firsts = {}
    for e in kernels:
        if counts[e["name"]] == modal and e["name"] not in firsts:
            firsts[e["name"]] = e["ts"]
    anchor = min(firsts, key=firsts.get)
    anchors = [e["ts"] for e in kernels if e["name"] == anchor]
    # a middle iteration, away from profiler warm-up edges
    mid = len(anchors) // 2
    t0, t1 = anchors[mid], anchors[mid + 1]
    lanes: dict[str, list] = defaultdict(list)
    for e in kernels:
        if t0 <= e["ts"] < t1:
            lanes[str(e.get("tid", "?"))].append(dict(
                t=round(e["ts"] - t0, 1), d=round(e["dur"], 1),
                b=bucket(e["name"]), n=e["name"][:90]))
    # order lanes by busiest first
    ordered = sorted(lanes.items(), key=lambda kv: -sum(k["d"] for k in kv[1]))
    return dict(period_us=round(t1 - t0, 1), n_anchored=len(anchors),
                lanes=[dict(tid=t, kernels=ks) for t, ks in ordered])


def main() -> None:
    out, prefixes = sys.argv[1], sys.argv[2:]
    result = {}
    for p in prefixes:
        for variant in ("baseline", "opt"):
            key = f"{p.rsplit('/', 1)[-1]}_{variant}"
            result[key] = one_iteration(f"{p}_{variant}_graph.trace.json")
            print(key, "period_us", result[key]["period_us"],
                  "lanes", len(result[key]["lanes"]))
    with open(out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
