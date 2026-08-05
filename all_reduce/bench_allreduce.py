"""TP=8 allreduce tactic sweep, following TRT-LLM's allreduce autotuner.

With ``allreduce_strategy=AUTO``, TRT-LLM routes every allreduce through
``torch.ops.trtllm.tunable_allreduce``: the AutoTuner asks ``AllReduceRunner``
for the valid tactics of the input shape, times each one, and caches the
winner per (tp_size, fusion op, num-token bucket). This bench replays exactly
that measurement — same runner, same tactic enumeration, same op call with the
same workspace — but reports EVERY tactic's latency and bus bandwidth instead
of silently keeping one:

  - ``NCCL_SYMMETRIC``  ncclAllReduce over window-registered symmetric buffers
                        (also the tuner's cache-miss fallback)
  - ``NCCL``            plain ncclAllReduce
  - ``ONESHOT``         TRT custom one-shot kernel (valid while the message
                        fits the custom-allreduce workspace)
  - ``TWOSHOT``         TRT custom two-shot kernel (valid when
                        num_tokens >= tp_size, so it appears from tokens=8 up)

Input shapes follow Kimi K3 at TP=8 (hidden 7168, latent 3584, bf16):

  - ``attn_oproj``  [tokens, 7168]   attention o_proj partial sums
  - ``moe_fused``   [tokens, 10752]  cat(routed latent 3584, shared 7168)

Launch (needs the tensorrt_llm venv, see README):

  mpirun -n 8 --allow-run-as-root \
      .venv-trtllm/bin/python all_reduce/bench_allreduce.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.kernel_bench import (  # noqa: E402
    _try_graph_capture,
    print_section,
    print_table,
    report,
)

# Decode has tokens == batch size; the tail of the sweep covers small-prefill /
# large-batch territory. ONESHOT drops out of the tactic list automatically
# (via get_valid_tactics) once tokens*hidden*2 exceeds the 64 MiB custom
# workspace — at these shapes even 2048 tokens (42 MiB) still fits.
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SHAPES = {
    "attn_oproj_h7168": 7168,
    "moe_fused_h10752": 3584 + 7168,
}


def bench_collective(comm, fn, warmup: int, iters: int, repeats: int) -> float:
    """Median us per call. All ranks call this together; an MPI barrier
    aligns launch, then a CUDA-graph replay of `iters` back-to-back
    collectives is event-timed (removes per-launch CPU overhead, which
    otherwise dominates these ~10us kernels). Falls back to an eager loop
    only when capture fails, and only if it fails on EVERY rank in lockstep
    (mismatched capture would deadlock the collective)."""
    from mpi4py import MPI

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    comm.Barrier()
    graph = _try_graph_capture(fn, iters)
    everyone_captured = comm.allreduce(graph is not None, op=MPI.LAND)
    if not everyone_captured:
        graph = None
    samples = []
    for _ in range(repeats):
        comm.Barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iters):
                fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    del graph
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "results" / "bench_allreduce_tp8.csv"),
    )
    args = parser.parse_args()

    from mpi4py import MPI

    from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=world, tp_size=world, rank=rank, gpus_per_node=world
    )

    # Same runner + workspace the tunable_allreduce path hands to the
    # AutoTuner (fusion op NONE — both K3 decode reduces are plain sums).
    runner = AllReduceRunner(
        tp_size=world,
        group=list(mapping.tp_group),
        op=int(AllReduceFusionOp.NONE),
        eps=1e-6,
        trigger_completion_at_end=True,
    )
    workspace = get_allreduce_workspace(mapping)

    bench_rows = []
    check_rows = []
    for shape_name, hidden in SHAPES.items():
        for tokens in TOKEN_COUNTS:
            gen = torch.Generator(device="cuda").manual_seed(1000 * tokens + rank)
            x = torch.randn(
                tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=gen
            )
            inputs = [x, None, None, None, None, workspace]
            tactics = runner.get_valid_tactics(inputs, None)

            # Reduced result must be tactic-independent: plain NCCL is the
            # reference (bf16 sums can differ by reduction order, <= 1 ulp).
            reference = runner.forward(inputs, tactic=AllReduceStrategy.NCCL.value)[
                0
            ].float()
            for tactic in tactics:
                name = AllReduceStrategy(tactic).name.lower()
                out = runner.forward(inputs, tactic=tactic)[0]
                check_rows.append(
                    {
                        "impl": name,
                        "shape": shape_name,
                        "tokens": tokens,
                        "max_abs_vs_nccl": (out.float() - reference)
                        .abs()
                        .max()
                        .item(),
                    }
                )

            msg_bytes = tokens * hidden * 2
            for tactic in tactics:
                name = AllReduceStrategy(tactic).name.lower()
                row = {
                    "impl": name,
                    "shape": shape_name,
                    "tokens": tokens,
                    "msg_kb": round(msg_bytes / 1024, 1),
                }
                try:
                    latency_us = bench_collective(
                        comm,
                        lambda t=tactic: runner.forward(inputs, tactic=t),
                        args.warmup,
                        args.iters,
                        args.repeats,
                    )
                    row["latency_us"] = latency_us
                    # Standard allreduce bus bandwidth: 2(n-1)/n * S / t.
                    row["busbw_gbps"] = (
                        2 * (world - 1) / world * msg_bytes / (latency_us * 1e-6) / 1e9
                    )
                except Exception as exc:  # noqa: BLE001
                    torch.cuda.synchronize()
                    row["error"] = f"{type(exc).__name__}: {exc}"
                bench_rows.append(row)

    # Cross-rank timing skew is real for collectives: keep the max over ranks.
    all_rows = comm.gather(bench_rows, root=0)
    if rank != 0:
        return
    merged = []
    for i, row in enumerate(bench_rows):
        row = dict(row)
        latencies = [
            r[i]["latency_us"] for r in all_rows if "latency_us" in r[i]
        ]
        if latencies:
            row["latency_us"] = max(latencies)
            row["busbw_gbps"] = (
                2 * (world - 1) / world
                * row["msg_kb"] * 1024
                / (row["latency_us"] * 1e-6)
                / 1e9
            )
        merged.append(row)
    for row in merged:
        base = next(
            (
                r["latency_us"]
                for r in merged
                if r["impl"] == "nccl"
                and r["shape"] == row["shape"]
                and r["tokens"] == row["tokens"]
                and "latency_us" in r
            ),
            None,
        )
        if base and "latency_us" in row:
            row["speedup_vs_nccl"] = base / row["latency_us"]

    print_section(
        f"TP={world} allreduce tactic sweep (autotuner candidates), "
        "Kimi K3 decode shapes (bf16)"
    )
    print_table(check_rows, title="correctness vs nccl (bf16 sum, <=1 ulp expected)")
    report(
        merged,
        title="latency (max over ranks, median of repeats)",
        csv_path=args.csv,
        plot=dict(
            x="tokens",
            y="latency_us",
            panel="shape",
            suptitle=f"TP={world} allreduce autotuner tactics, "
            "Kimi K3 decode shapes (B200)",
        ),
    )


if __name__ == "__main__":
    main()
