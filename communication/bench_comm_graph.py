#!/usr/bin/env python3
"""Graph-timed communication report (host bubbles excluded by CUDA graphs).

Times every backend of every requested op under CUDA-graph replay (median of
replays, MAX across ranks) and writes one CSV per op — the numbers RESULTS.md
tables are generated from. `bench_comm.py` remains the correctness +
autotune-map harness; its event-window numbers include host overhead and are
not report material.

  mpirun --allow-run-as-root -np 8 python3 communication/bench_comm_graph.py \\
      --ops allreduce --bs 1..16k   # ~40 s per op
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from communication.collective import Collectives  # noqa: E402

OP_NAMES = {
    "allreduce": "all_reduce",
    "allgather": "all_gather",
    "reducescatter": "reduce_scatter",
    "all-to-all": "all_to_all",
    "quantized-allreduce": "quantized_all_reduce",
    "allreduce-norm": "allreduce_norm",
    "gemm-allreduce": "gemm_allreduce",
    "allreduce-norm-gemm": "allreduce_norm_gemm",
}
_WORLD_SIZED = {"all_gather", "reduce_scatter", "all_to_all"}


def _bs_value(token: str) -> int:
    token = token.strip().lower()
    return int(token[:-1]) * 1024 if token.endswith("k") else int(token)


def parse_bs(spec: str) -> list[int]:
    """``A..B`` = pow2 sweep from A to B inclusive (k-suffixes allowed),
    else a comma list. The old parser accepted only the literal ``1..16k``
    and crashed on any other range."""
    if ".." in spec:
        lo_s, _, hi_s = spec.partition("..")
        lo, hi = _bs_value(lo_s), _bs_value(hi_s)
        out = []
        b = lo
        while b <= hi:
            out.append(b)
            b *= 2
        return out
    return [_bs_value(v) for v in spec.split(",")]


def graph_time_us(fn, calls: int, repeats: int = 7) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls)
        dist.barrier()
    del graph
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ops", default="allreduce")
    ap.add_argument("--bs", default="1..16k")
    ap.add_argument("--dims", default="7168")
    ap.add_argument("--enable-b10", action="store_true", default=True)
    ap.add_argument("--enable-cutedsl", action="store_true", default=True)
    ap.add_argument("--gemm-k", type=int, default=896)
    ap.add_argument("--gemm-n", type=int, default=6288)
    ap.add_argument("--out-dir", default="communication/local_result")
    args = ap.parse_args()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29547")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
        device_id=torch.device("cuda", rank),
    )
    ops = [OP_NAMES[o] for o in args.ops.split(",")]
    bs_list = parse_bs(args.bs)
    dims = [int(v) for v in args.dims.split(",")]
    mult = world if any(op in _WORLD_SIZED for op in ops) else 1
    comm = Collectives(
        dist.group.WORLD,
        max_numel=max(bs_list) * max(dims) * mult,
        dtype=torch.bfloat16,
        max_hidden=max(dims),
        flashinfer_max_tokens=max(bs_list),
        enable_b10=args.enable_b10,
        enable_cutedsl=args.enable_cutedsl,
        boundary_gemm_k=args.gemm_k,
        boundary_gemm_n=args.gemm_n,
        boundary_max_tokens=max(bs_list),
    )
    for op in ops:
        rows = []
        candidates = comm._op_candidates(op)
        for dim in dims:
            for bs in bs_list:
                rows_here = bs * dim * (world if op in _WORLD_SIZED else 1)
                if rows_here > comm.max_numel:
                    continue
                calls = 20 if bs <= 256 else (8 if bs <= 2048 else 3)
                for name in candidates:
                    # capacity probe = the op's INPUT operand shape
                    # (matches _make_bench_fn): all_gather takes
                    # [bs, dim]; reduce_scatter/all_to_all take
                    # [bs*world, dim]. Passing the gathered OUTPUT
                    # shape to all_gather's _fits double-counts world
                    # and spuriously capacity-gated bs>=4096.
                    in_rows = bs * (
                        world if op in ("reduce_scatter", "all_to_all")
                        else 1)
                    probe = torch.empty(
                        (in_rows, dim),
                        device="meta", dtype=torch.bfloat16)
                    status, us = "ok", float("nan")
                    try:
                        if not comm._fits(op, name, probe):
                            status = "capacity"
                        else:
                            fn = comm._planner._make_bench_fn(op, name, dim, bs)
                            fn()
                            us = graph_time_us(fn, calls)
                    except Exception as exc:  # noqa: BLE001
                        status = f"fail: {type(exc).__name__}: {str(exc)[:80]}"
                    t = torch.tensor(
                        [0.0 if us != us else us], device="cuda")
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    us_max = float(t.item())
                    if rank == 0:
                        rows.append({
                            "op": op, "dim": dim, "bs": bs, "backend": name,
                            "latency_us": round(us_max, 3)
                            if status == "ok" else "",
                            "status": status,
                        })
                dist.barrier()
        if rank == 0 and rows:
            out = Path(args.out_dir) / f"report_graph_{op}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {out}", flush=True)
    dist.barrier()
    if rank == 0:
        print("PASS bench_comm_graph", flush=True)


if __name__ == "__main__":
    main()
