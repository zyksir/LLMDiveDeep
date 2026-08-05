#!/usr/bin/env python3
"""Per-stage profile of a CUDA-graph-replay MoE trace (baseline or opt).

Graph replays carry no record_function spans, so stages are recovered
from the kernel stream: kernels are matched by name/tile signature
(RULES), and the un-named `nvjet` GEMMs are disambiguated by their
POSITION between anchors that are unambiguous (routing, quantize, bmm,
finalize, collectives). One iteration = one occurrence of the anchor
cycle; we report the mean over the profiled replays.

  python3 kimi_k3_layer/profile_moe_stages.py results/*.trace.json
  python3 kimi_k3_layer/profile_moe_stages.py --json out.json trace...

Output per trace: a per-kernel table (stream, calls/iter, us/iter) and
the wall span of one iteration, plus the aux-stream (overlapped) work
separated from the critical path.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# kernel-name substring -> stage label. First match wins. These cover
# BOTH the baseline (production TRT-LLM modules) and the b10 opt path.
RULES = [
    # --- routing ---
    ("_kimik3_route_pack", "routing (b10 triton)"),
    ("routingIndicesBlockScores", "routing (scores+topk)"),
    ("routingIndicesCluster", "routing (permute)"),
    ("routingIndicesSmallBs", "routing (small-bs)"),
    # --- expert stage (identical kernels in both paths) ---
    ("quantize_with_block_size", "expert quantize (mxfp8)"),
    ("MxE4m3_MxE2m1", "expert bmm1 (fp8xfp4)"),
    ("Bfloat16_MxE2m1", "expert bmm2 (fp8xfp4)"),
    ("finalizeKernel", "expert finalize"),
    ("activation_kernel", "expert activation"),
    # --- shared expert ---
    ("_situ_and_mul", "shared SiTU"),
    ("silu_and_mul", "shared SiTU"),
    # --- collectives ---
    ("ag_mxfp8", "b10 AG+mxfp8 (fc1 shard)"),
    ("ag_lamport", "b10 AG (fc1 shard)"),
    ("rs_cols_lamport", "b10 RS+norm (latent)"),
    ("rs_cols_scale", "b10 RS deferred scale"),
    ("allreduce_fusion", "allreduce (flashinfer)"),
    ("ncclSymkDevKernel", "allreduce (nccl symm)"),
    ("ncclDevKernel", "nccl (harness/barrier)"),
    # --- norms / elementwise glue ---
    ("RMSNorm", "rmsnorm (latent)"),
    ("rms_norm", "rmsnorm (latent)"),
    ("CatArrayBatchedCopy", "cat [latent|shared]"),
    ("splitKreduce", "splitK reduce"),
    ("elementwise_kernel", "elementwise add"),
    ("vectorized_elementwise", "elementwise add"),
    ("direct_copy", "copy/pad"),
    ("Memcpy", "memcpy"),
    ("memcpy", "memcpy"),
]

GEMM_PAT = re.compile(r"nvjet|cutlass|gemm|Gemm|GEMM|sm100|ampere|cublas")


def label(name: str) -> str | None:
    for pat, stage in RULES:
        if pat in name:
            return stage
    if GEMM_PAT.search(name):
        return None  # a GEMM: labelled by position later
    return "other: " + name[:48]


def short(name: str) -> str:
    """Compact kernel identity: keep the family + tile signature."""
    name = name.split("(")[0]
    for key in ("nvjet_", "cutlass", "void "):
        if key in name:
            name = name[name.index(key):]
            break
    return name[:78]


def load_kernels(path: Path):
    t = json.loads(path.read_text())
    ev = [e for e in t["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
    ev.sort(key=lambda e: e["ts"])
    return ev


def split_iters(ev):
    """Cut the stream into iterations at each occurrence of the FIRST
    kernel of the layer (the earliest-recurring kernel on the busiest
    stream). Returns a list of per-iteration event lists."""
    by_stream = defaultdict(list)
    for e in ev:
        by_stream[e["tid"]].append(e)
    main = max(by_stream.values(), key=lambda v: sum(x["dur"] for x in v))
    counts = defaultdict(int)
    for e in main:
        counts[e["name"]] += 1
    n_iter = max(counts.values())
    anchor = next(e["name"] for e in main if counts[e["name"]] == n_iter)
    starts = [e["ts"] for e in main if e["name"] == anchor]
    iters = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else float("inf")
        iters.append([e for e in ev if s <= e["ts"] < end])
    return iters, n_iter, anchor


def summarize(path: Path, drop_first: int = 1):
    ev = load_kernels(path)
    iters, n_iter, anchor = split_iters(ev)
    iters = iters[drop_first:-1] if len(iters) > drop_first + 1 else iters
    n = len(iters)
    if n == 0:
        raise SystemExit(f"{path}: no complete iteration found")

    # stream classification: the stream carrying the anchor is "main"
    main_tid = None
    for e in iters[0]:
        if e["name"] == anchor:
            main_tid = e["tid"]
            break

    agg = defaultdict(lambda: {"dur": 0.0, "n": 0, "stream": ""})
    span = 0.0
    for it in iters:
        span += max(e["ts"] + e["dur"] for e in it) - min(e["ts"] for e in it)
        for e in it:
            stage = label(e["name"])
            key = (stage or "gemm", short(e["name"]),
                   "main" if e["tid"] == main_tid else "aux")
            a = agg[key]
            a["dur"] += e["dur"]
            a["n"] += 1
    rows = []
    for (stage, name, stream), a in agg.items():
        rows.append({
            "stage": stage, "kernel": name, "stream": stream,
            "calls": a["n"] / n, "us": a["dur"] / n,
        })
    rows.sort(key=lambda r: -r["us"])
    return {
        "path": str(path), "iters": n, "anchor": anchor,
        "wall_us": span / n,
        "main_us": sum(r["us"] for r in rows if r["stream"] == "main"),
        "aux_us": sum(r["us"] for r in rows if r["stream"] == "aux"),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = []
    for p in args.traces:
        s = summarize(p)
        out.append(s)
        print(f"\n=== {p.name} ===")
        print(f"iterations analysed: {s['iters']}  "
              f"wall/iter: {s['wall_us']:.1f} us  "
              f"(main-stream kernel time {s['main_us']:.1f}, "
              f"aux-stream {s['aux_us']:.1f})")
        print(f"{'us/it':>7} {'calls':>6} {'str':>4}  {'stage':<26} kernel")
        for r in s["rows"]:
            if r["us"] < 0.05:
                continue
            print(f"{r['us']:7.2f} {r['calls']:6.1f} {r['stream']:>4}  "
                  f"{r['stage']:<26} {r['kernel']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
