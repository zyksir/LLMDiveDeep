#!/usr/bin/env python3
"""Merge phantom CUDA-graph pool lanes into the minimal true lanes.

keep_graph=True graphs (needed by the routing graphpatch) make the torch
profiler attribute kernels to ~50 internal pool streams even though at most
~3 are ever concurrently active. This relabels tids by greedy interval
packing: a kernel joins the first lane whose last kernel ended before it
starts. Timestamps and durations are untouched — iteration-to-iteration
timing stays fully inspectable.

Usage: normalize_trace_streams.py IN.trace.json [OUT.trace.json]
"""
import json
import sys

CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def main() -> None:
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(
        ".trace.json", "_normalized.trace.json")
    with open(src) as f:
        data = json.load(f)
    events = data["traceEvents"]
    ks = sorted((e for e in events if e.get("cat") in CATS
                 and e.get("dur", 0) > 0), key=lambda e: e["ts"])
    lane_end: list[float] = []
    for e in ks:
        for i, end in enumerate(lane_end):
            if end <= e["ts"] + 1e-3:
                lane_end[i] = e["ts"] + e["dur"]
                e["tid"] = 1000 + i
                break
        else:
            e["tid"] = 1000 + len(lane_end)
            lane_end.append(e["ts"] + e["dur"])
    # drop non-kernel rows' thread metadata collisions by leaving them as-is
    with open(dst, "w") as f:
        json.dump(data, f)
    print(f"{src}: {len(lane_end)} true lanes -> {dst}")


if __name__ == "__main__":
    main()
