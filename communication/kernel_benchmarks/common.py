"""Shared distributed correctness, graph-replay, and timing helpers."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist


def init() -> tuple[int, int]:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
    world = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1))
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29546")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    if not dist.is_initialized():
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda", rank % torch.cuda.device_count()),
        )
    return rank, world


def skip(reason: str, rank: int) -> None:
    if rank == 0:
        print(f"SKIP: {reason}", flush=True)


def _flat(value):
    return value if isinstance(value, tuple) else (value,)


def _snapshot(value):
    values = tuple(item.clone() for item in _flat(value))
    return values if isinstance(value, tuple) else values[0]


def check(actual, expected, *, atol: float, label: str) -> None:
    aa, ee = _flat(actual), _flat(expected)
    assert len(aa) == len(ee), label
    for index, (got, want) in enumerate(zip(aa, ee)):
        error = (got.float() - want.float()).abs().max().item()
        assert error <= atol, f"{label}[{index}] max error {error} > {atol}"


def exercise(
    static_inputs: tuple[torch.Tensor, ...],
    run: Callable,
    reference: Callable,
    make_inputs: Callable[[int], tuple[torch.Tensor, ...]],
    *,
    atol: float,
    graph_safe: bool = True,
    launches: int = 2,
    replays: int = 3,
) -> None:
    for eager_round in range(3):
        values = make_inputs(eager_round)
        check(
            _snapshot(run(*values)),
            reference(*values),
            atol=atol,
            label=f"eager {eager_round}",
        )

    if not graph_safe:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(*static_inputs)
        except (AssertionError, RuntimeError):
            return
        raise AssertionError("capture-unsafe kernel unexpectedly captured")

    for dst, src in zip(static_inputs, make_inputs(10)):
        dst.copy_(src)
    run(*static_inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [_snapshot(run(*static_inputs)) for _ in range(launches)]

    for replay_round in range(replays):
        changing = make_inputs(20 + replay_round)
        for dst, src in zip(static_inputs, changing):
            dst.copy_(src)
        expected = reference(*changing)
        graph.replay()
        torch.cuda.synchronize()
        for launch, output in enumerate(captured):
            check(
                output,
                expected,
                atol=atol,
                label=f"graph replay {replay_round} launch {launch}",
            )


def latency_us(fn: Callable, *, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    value = torch.tensor(
        [start.elapsed_time(end) * 1000.0 / iters],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def report_latency(name: str, run: Callable, reference: Callable, rank: int) -> None:
    custom = latency_us(run)
    trusted = latency_us(reference)
    if rank == 0:
        print(
            f"PASS {name}: custom={custom:.1f}us reference={trusted:.1f}us "
            f"ratio={custom / trusted:.2f}x",
            flush=True,
        )
