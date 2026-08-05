"""Dump what the TRT-LLM allreduce autotuner actually selects, per shape.

Runs the real production path — ``AllReduce(strategy=AUTO)`` →
``tunable_allreduce`` → ``AutoTuner.choose_one`` inside an ``autotune()``
context — for both Kimi K3 decode reduce shapes, then prints every cache
entry the tuner created: one winner per (hidden, num-token bucket).

  mpirun -n 8 --allow-run-as-root \
      .venv-trtllm/bin/python all_reduce/autotune_choices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.kernel_bench import print_table  # noqa: E402


def main() -> None:
    from mpi4py import MPI

    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.functional import AllReduceStrategy as StrategyEnum
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=world, tp_size=world, rank=rank, gpus_per_node=world
    )

    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO)
    # Production (_run_autotuner_warmup) sets this up before tuning; the
    # allreduce runner's MERGE strategy silently skips tuning without it.
    AutoTuner.get().setup_distributed_state(mapping)

    # One call per hidden size is enough: the tuner's DynamicTensorSpec sweeps
    # every power-of-2 token bucket up to 8192 internally.
    with autotune():
        for hidden in (7168, 3584 + 7168):
            x = torch.randn(64, hidden, device="cuda", dtype=torch.bfloat16)
            allreduce(x)

    comm.Barrier()
    if rank != 0:
        return

    rows = []
    for key, value in AutoTuner.get().profiling_cache.cache.items():
        runner_id, tactic, min_time = value[0], value[1], value[-1]
        rows.append(
            {
                "cache_key": str(key),
                "tactic": StrategyEnum(tactic).name if tactic >= 0 else "-1",
                "time_ms": min_time,
            }
        )
    rows.sort(key=lambda r: r["cache_key"])
    print_table(rows, title=f"autotuner cache after tuning (TP={world})")


if __name__ == "__main__":
    main()
