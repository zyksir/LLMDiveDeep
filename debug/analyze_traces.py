#!/usr/bin/env python3
"""Compare baseline vs opt chrome traces: GPU kernel time by bucket.

Usage: python3 debug/analyze_traces.py kimi_k3_layer/local_results/moe_tp8_b8
(reads <prefix>_baseline_graph.trace.json and <prefix>_opt_graph.trace.json)
"""
import json
import sys
from collections import defaultdict

BUCKETS = (
    ("routing", ("routing", "topk", "sort", "radix", "permut", "scan",
                 "histogram", "expandInputRows")),
    ("gemm_expert", ("fused_moe", "moe_gemm", "group_gemm", "bmm_", "situ",
                     "finalize", "MoE")),
    ("gemm_dense", ("nvjet", "cutlass", "gemm", "sm100_xmma", "matmul")),
    ("quant", ("quant", "fp8", "mxfp8", "e4m3")),
    ("comm", ("nccl", "allreduce", "all_reduce", "multimem", "oneshot",
              "twoshot", "allgather", "all_gather", "cross_device", "symm")),
    ("norm", ("rmsnorm", "layernorm", "norm")),
    ("copy_misc", ("elementwise", "copy", "cat", "memcpy", "memset", "fill",
                   "vectorized", "index", "gather", "scatter")),
)


def bucket(name: str) -> str:
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k.lower() in low for k in keys):
            return label
    return "other"


def load(path: str) -> tuple[dict, dict, float]:
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    by_bucket: dict[str, float] = defaultdict(float)
    by_kernel: dict[str, float] = defaultdict(float)
    total = 0.0
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur", 0.0)
        name = e.get("name", "?")
        by_bucket[bucket(name)] += dur
        by_kernel[name] += dur
        total += dur
    return by_bucket, by_kernel, total


def main() -> None:
    prefix = sys.argv[1]
    base_b, base_k, base_t = load(f"{prefix}_baseline_graph.trace.json")
    opt_b, opt_k, opt_t = load(f"{prefix}_opt_graph.trace.json")
    print(f"== {prefix}  (GPU-summed us across profiled window, all ranks) ==")
    print(f"{'bucket':<12} {'baseline':>10} {'opt':>10} {'delta':>10}")
    for label in sorted(set(base_b) | set(opt_b),
                        key=lambda l: -(base_b.get(l, 0) - opt_b.get(l, 0))):
        b, o = base_b.get(label, 0.0), opt_b.get(label, 0.0)
        print(f"{label:<12} {b:>10.0f} {o:>10.0f} {b - o:>+10.0f}")
    print(f"{'TOTAL':<12} {base_t:>10.0f} {opt_t:>10.0f} {base_t - opt_t:>+10.0f}")
    for tag, kmap in (("baseline", base_k), ("opt", opt_k)):
        print(f"\n-- top kernels: {tag} --")
        for name, dur in sorted(kmap.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {dur:>9.0f}us  {name[:110]}")


if __name__ == "__main__":
    main()
