"""Inject pipeline-stage labels into a graph-replay trace.

Graph replays cannot carry record_function annotations, so this adds a
synthetic "stages" track per GPU stream with one span per kernel,
named by kernel->stage rules (verified against the eager labeled twin
via tmp_label_map.py). Output: <input>_annotated.trace.json.

Usage: python3 annotate_graph_trace.py results/..._opt_graph.trace.json
"""

import json
import sys

# rules verified for the B=64 (routed split pipeline) and B=8 (merged
# input GEMM, +-fc1 shard) opt traces; first substring match wins
RULES = [
    ("_kimik3_route_pack", "moe.routing"),
    ("_situ_and_mul", "moe.shared_act"),
    ("quantize_with_block_size", "moe.latent_quant"),
    ("ag_mxfp8", "moe.fc1_ag_quant"),
    ("_MxE2m1", "moe.experts.bmm"),
    ("routingIndicesCluster", "moe.experts.permute"),
    ("finalizeKernel", "moe.experts.finalize"),
    ("rs_cols_lamport", "moe.rs_latent+norm"),
    ("ag_lamport", "moe.fc1_allgather"),
    ("allreduce_fusion", "moe.allreduce_7168"),
    ("ncclDevKernel", "harness.barrier"),
    ("badd", "moe.tail_fc2_addmm"),
    ("direct_copy", "moe.latent_pad"),
    # B=64 split-pipeline GEMMs
    ("64x14_4x1_v_bz_TNN", "moe.shared_gate_up"),
    ("2x2_2cta_h_bz_splitK_TNT", "moe.gate_gemm"),
    ("2x1_2cta_v_bz_splitK_TNT", "moe.shared_tail_gemm"),
    ("2x2_2cta_h_bz_TNT", "moe.fc1_gemm"),
    # B=8 merged-pipeline GEMMs
    ("64x8_64x16_2x1_v_bz_splitK_TNT", "moe.fused_in_gemm"),
    ("32x64_64x16_4x1_v_bz_splitK_TNN", "moe.fused_in_gemm(fc1shard)"),
    ("64x8_64x16_4x1_v_bz_splitK_TNT", "moe.shared_gate_up"),
    ("64x8_64x16_4x1_v_bz_TNT", "moe.shared_down"),
    ("Memcpy", "memcpy"),
]


def stage_of(name, prev_stage):
    if "splitKreduce" in name:  # belongs to the preceding splitK GEMM
        return prev_stage or "splitKreduce"
    for pat, stage in RULES:
        if pat in name:
            return stage
    return "other:" + name[:40]


def main():
    path = sys.argv[1]
    t = json.load(open(path))
    evs = t["traceEvents"]
    kernels = sorted(
        (e for e in evs if e.get("ph") == "X"
         and e.get("cat") in ("kernel", "gpu_memcpy")),
        key=lambda e: e["ts"])

    prev_by_stream = {}
    out = []
    for e in kernels:
        st = stage_of(e["name"], prev_by_stream.get(e["tid"]))
        prev_by_stream[e["tid"]] = st
        out.append({
            "ph": "X", "cat": "user_annotation", "name": st,
            "pid": e["pid"], "tid": f"stages-{e['tid']}",
            "ts": e["ts"], "dur": e["dur"],
        })
    for tid in {o["tid"] for o in out}:
        out.append({
            "ph": "M", "cat": "__metadata", "name": "thread_name",
            "pid": kernels[0]["pid"], "tid": tid,
            "args": {"name": tid},
        })
    t["traceEvents"] = evs + out
    dst = path.replace(".trace.json", "_annotated.trace.json")
    json.dump(t, open(dst, "w"))
    print(f"wrote {dst} ({len(out)} stage events)")


if __name__ == "__main__":
    main()
