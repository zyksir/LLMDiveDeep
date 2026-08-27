#!/usr/bin/env python3
"""Measure how destination-rank locality affects DeepGEMM MegaMoE.

The timed boundary is the unchanged ``fp8_fp4_mega_moe`` call. Inputs and
transformed synthetic weights are allocated once outside timing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batches", default="8,64,256,512,2048,8192,16384")
    parser.add_argument("--route-modes", default="spread,single_rank")
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def init_dist() -> tuple[int, int]:
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29547")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local_rank),
    )
    return rank, world


def make_packed_weight(
    generator: torch.Generator,
    experts: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randint(
        0,
        256,
        (experts, n, k // 2),
        generator=generator,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.int8)
    scale = torch.full(
        (experts, k // 128, n),
        0x7C7C7C7C,
        dtype=torch.int32,
        device="cuda",
    ).transpose(-2, -1)
    return weight, scale


def make_routes(
    mode: str,
    rank: int,
    world: int,
    local_tokens: int,
    experts: int,
    top_k: int,
) -> torch.Tensor:
    experts_per_rank = experts // world
    token = torch.arange(local_tokens, device="cuda", dtype=torch.int64)
    global_token = rank * local_tokens + token

    if mode == "spread":
        if top_k % world:
            raise ValueError("spread routing requires top_k divisible by world size")
        experts_per_destination = top_k // world
        destination = torch.arange(world, device="cuda", dtype=torch.int64)
        local_offset = torch.arange(
            experts_per_destination, device="cuda", dtype=torch.int64
        )
        local_expert = (
            global_token[:, None, None] * experts_per_destination
            + local_offset[None, None, :]
        ) % experts_per_rank
        routes = destination[None, :, None] * experts_per_rank + local_expert
        return routes.reshape(local_tokens, top_k).contiguous()

    if mode == "single_rank":
        if top_k > experts_per_rank:
            raise ValueError("single_rank routing requires top_k <= experts_per_rank")
        destination = global_token % world
        local_expert = (
            global_token[:, None] * top_k
            + torch.arange(top_k, device="cuda", dtype=torch.int64)[None, :]
        ) % experts_per_rank
        return (destination[:, None] * experts_per_rank + local_expert).contiguous()

    raise ValueError(f"unknown route mode: {mode}")


def max_cuda_time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / iters
    value = torch.tensor(elapsed_us, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def main() -> None:
    args = parse_args()
    rank, world = init_dist()
    import deep_gemm

    if args.experts % world:
        raise ValueError("experts must be divisible by world size")
    global_batches = tuple(int(value) for value in args.global_batches.split(","))
    if any(batch % world for batch in global_batches):
        raise ValueError("all global batch sizes must be divisible by world size")
    route_modes = tuple(args.route_modes.split(","))
    max_local_tokens = max(global_batches) // world
    setup_start = time.perf_counter()

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        args.experts,
        max_local_tokens,
        args.top_k,
        args.hidden,
        args.intermediate,
        mma_type="fp8xfp4",
    )
    local_experts = args.experts // world
    generator = torch.Generator(device="cuda").manual_seed(7000 + rank)
    l1_weight = make_packed_weight(
        generator, local_experts, 2 * args.intermediate, args.hidden
    )
    l2_weight = make_packed_weight(
        generator, local_experts, args.hidden, args.intermediate
    )
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        l1_weight, l2_weight
    )
    del l1_weight, l2_weight
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start

    rows = []
    payload_bytes = args.hidden + args.hidden // 32
    for global_batch in global_batches:
        local_tokens = global_batch // world
        x = torch.randn(
            (local_tokens, args.hidden),
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).clamp_(-448, 448).to(torch.float8_e4m3fn)
        x_sf = torch.full(
            (local_tokens, args.hidden // 128),
            0x7F7F7F7F,
            dtype=torch.int32,
            device="cuda",
        )
        weights = torch.full(
            (local_tokens, args.top_k),
            1.0 / args.top_k,
            dtype=torch.float32,
            device="cuda",
        )
        output = torch.empty(
            (local_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
        )

        for mode in route_modes:
            routes = make_routes(
                mode,
                rank,
                world,
                local_tokens,
                args.experts,
                args.top_k,
            )
            buffer.x[:local_tokens].copy_(x)
            buffer.x_sf[:local_tokens].copy_(x_sf)
            buffer.topk_idx[:local_tokens].copy_(routes)
            buffer.topk_weights[:local_tokens].copy_(weights)

            def run() -> None:
                deep_gemm.fp8_fp4_mega_moe(
                    output,
                    transformed_l1,
                    transformed_l2,
                    buffer,
                    activation="swiglu",
                    fast_math=True,
                )

            latency_us = max_cuda_time_us(run, args.warmup, args.iters)
            unique_destination_ranks = world if mode == "spread" else 1
            average_remote_pairs = args.top_k * (world - 1) / world
            average_unique_remote_ranks = (
                world - 1 if mode == "spread" else (world - 1) / world
            )
            rows.append(
                {
                    "global_batch_size": global_batch,
                    "local_tokens": local_tokens,
                    "route_mode": mode,
                    "unique_destination_ranks_per_token": unique_destination_ranks,
                    "latency_us": latency_us,
                    "nominal_current_remote_payload_bytes_per_token": (
                        average_remote_pairs * payload_bytes
                    ),
                    "rank_dedup_remote_payload_bytes_per_token": (
                        average_unique_remote_ranks * payload_bytes
                    ),
                    "finite": bool(torch.isfinite(output).all().item()),
                }
            )
            if rank == 0:
                print(
                    f"global_bs={global_batch:>5} mode={mode:<11} "
                    f"unique_ranks={unique_destination_ranks} "
                    f"latency={latency_us:9.1f} us",
                    flush=True,
                )

    receipt = {
        "deep_gemm_version": deep_gemm.__version__,
        "shape": {
            "world_size": world,
            "num_experts": args.experts,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "input": "FP8 E4M3 + packed UE8M0",
            "weights": "synthetic MXFP4 + packed UE8M0",
            "output": "BF16",
        },
        "timed_boundary": "deep_gemm.fp8_fp4_mega_moe only",
        "warmup": args.warmup,
        "iters": args.iters,
        "setup_seconds": setup_seconds,
        "rows": rows,
    }
    if rank == 0 and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
