#!/usr/bin/env python3
"""Sweep global token counts with one weight load per MoE backend."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from bench_a2a_megamoe_pipeline import (
    _backend_name_from_module,
    _benchmark,
    _build_mapping_from_config,
    _build_module,
    _build_routing_plan,
    _comm_method_name,
    _project_router_logits_for_plan,
    _run_autotune,
    _set_device_from_local_rank,
    dispatch_expert_combine,
)
from bench_moe.specs import (
    ConfigSpec,
    ModelSpec,
    RoutingControlSpec,
)
from _torch.modules.moe.quantize_utils import MXFP4MXFP8QuantizeUtil
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._utils import (
    mpi_allgather,
    mpi_barrier,
    mpi_rank,
    mpi_world_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-batch-sizes",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
        ],
    )
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--weight-cache-dir",
        type=Path,
        default=Path("out/moe_weight_cache"),
    )
    parser.add_argument("--rebuild-weight-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/k3_ep8_a2a_megamoe_bs_sweep.json"),
    )
    return parser.parse_args()


def _per_rank_tokens(global_tokens: int, world_size: int) -> list[int]:
    quotient, remainder = divmod(global_tokens, world_size)
    return [quotient + remainder, *([quotient] * (world_size - 1))]


def _prepare_local_backend_weights(
    self: MXFP4MXFP8QuantizeUtil,
    backend: Any,
    **quant_kwargs: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Generate only this EP rank's backend weights; no unused reference copy."""
    num_elts_per_dtype = torch.iinfo(backend.quant_method.weight_dtype).bits // 4
    hidden_size_in = backend.w3_w1_weight.shape[-1] * num_elts_per_dtype
    hidden_size_out = backend.w2_weight.shape[-2]
    tp_size = getattr(backend, "tp_size", 1)
    intermediate_size = backend.w2_weight.shape[-1] * num_elts_per_dtype * tp_size
    input_hidden_alignment = getattr(
        backend.quant_method,
        "input_hidden_alignment",
        backend.quant_method.weight_alignment,
    )
    backend_kwargs = dict(
        quant_kwargs,
        hidden_size_in=hidden_size_in,
        hidden_size_out=hidden_size_out,
        intermediate_size=intermediate_size,
        input_hidden_alignment=input_hidden_alignment,
        pad_zero_or_val=tp_size > 1,
        bias=self.bias,
    )

    global_num_experts = self.num_experts
    local_num_experts = self.num_local_experts
    backend_name = type(getattr(backend, "backend", backend)).__name__
    cache_dir = Path(os.environ["MOE_BENCH_WEIGHT_CACHE_DIR"])
    cache_name = (
        f"v1_{backend_name}_e{local_num_experts}_hin{hidden_size_in}"
        f"_hout{hidden_size_out}_i{intermediate_size}"
        f"_wa{backend.quant_method.weight_alignment}"
        f"_hia{input_hidden_alignment}_tp{tp_size}.pt"
    )
    cache_path = cache_dir / cache_name
    rebuild_cache = os.environ.get("MOE_BENCH_REBUILD_WEIGHT_CACHE") == "1"
    mpi_barrier()

    if cache_path.exists() and not rebuild_cache:
        local_weights = torch.load(
            cache_path,
            map_location=backend.w3_w1_weight.device,
            weights_only=True,
        )
    else:
        self.num_experts = local_num_experts
        try:
            local_weights = self.create_weights(**backend_kwargs)
        finally:
            self.num_experts = global_num_experts
        if mpi_rank() == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp")
            cpu_weights = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in local_weights.items()
            }
            torch.save(cpu_weights, temporary_path)
            temporary_path.replace(cache_path)
            del cpu_weights
        mpi_barrier()

    expert_offset = mpi_rank() * local_num_experts
    weights = {}
    for key, value in local_weights.items():
        local_expert, suffix = key.split(".", 1)
        weights[f"{int(local_expert) + expert_offset}.{suffix}"] = value
    return weights, weights, {}


def _make_inputs(
    args: argparse.Namespace,
    global_tokens: int,
    rank: int,
    world_size: int,
    routing_method: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    per_rank = _per_rank_tokens(global_tokens, world_size)
    local_tokens = per_rank[rank]
    generator = torch.Generator(device=device).manual_seed(args.seed + global_tokens + rank)
    hidden_states = torch.randn(
        (local_tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    routing_spec = RoutingControlSpec(
        routing_mode="forced",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        seed=args.seed + global_tokens,
    )
    plan = _build_routing_plan(
        routing_spec,
        num_tokens=global_tokens,
        world_size=world_size,
        top_k=args.top_k,
        num_experts=args.num_experts,
        moe_ep_size=world_size,
    )
    router_logits, status, reason = _project_router_logits_for_plan(
        plan,
        src_rank=rank,
        routing_method=routing_method,
        num_experts=args.num_experts,
        top_k=args.top_k,
        experts_per_rank=args.num_experts // world_size,
        moe_ep_size=world_size,
        device=device,
        dtype=torch.bfloat16,
    )
    if status != "exact":
        raise RuntimeError(f"Routing projection failed for batch {global_tokens}: {status}: {reason}")
    return hidden_states, router_logits, per_rank


def _sweep_backend(
    args: argparse.Namespace,
    model: ModelSpec,
    mapping: Any,
    device: torch.device,
    backend: str,
    comm_method: str,
) -> tuple[dict[int, dict[str, Any]], float]:
    rank = mpi_rank()
    world_size = mpi_world_size()
    max_local_tokens = max(
        max(_per_rank_tokens(batch_size, world_size))
        for batch_size in args.global_batch_sizes
    )
    setup_started = time.perf_counter()
    moe = _build_module(
        model,
        mapping,
        max_num_tokens=max(max_local_tokens, 1),
        device=device,
        backend=backend,
        comm_method=comm_method,
    )
    torch.cuda.synchronize()
    setup_seconds = max(mpi_allgather(time.perf_counter() - setup_started))
    try:
        if _backend_name_from_module(moe) != backend:
            raise RuntimeError(f"Requested {backend}, built {_backend_name_from_module(moe)}")
        if backend == "TRTLLM" and _comm_method_name(moe) != "NVLinkOneSided":
            raise RuntimeError(f"Expected NVLinkOneSided, built {_comm_method_name(moe)}")

        results: dict[int, dict[str, Any]] = {}
        for batch_size in args.global_batch_sizes:
            hidden_states, router_logits, all_rank_num_tokens = _make_inputs(
                args,
                batch_size,
                rank,
                world_size,
                moe.routing_method,
                device,
            )
            with torch.inference_mode():
                AutoTuner.get().clear_cache()
                _run_autotune(
                    moe,
                    hidden_states,
                    router_logits,
                    all_rank_num_tokens,
                    fast_autotune=True,
                )
            timing = _benchmark(
                "external" if backend == "TRTLLM" else "megamoe",
                moe,
                lambda: dispatch_expert_combine(
                    moe,
                    hidden_states,
                    router_logits,
                    all_rank_num_tokens,
                ),
                args.warmup,
                args.iters,
            )
            gathered = mpi_allgather(timing)
            results[batch_size] = {
                "per_rank": {f"rank{idx}": value for idx, value in enumerate(gathered)},
                "score_mean_ms": max(value["total"]["mean_ms"] for value in gathered),
                "score_median_ms": max(value["total"]["median_ms"] for value in gathered),
            }
            if rank == 0:
                print(
                    f"{backend} global_bs={batch_size}: "
                    f"{results[batch_size]['score_mean_ms']:.6f} ms"
                )
        return results, setup_seconds
    finally:
        moe.destroy()
        torch.cuda.empty_cache()
        mpi_barrier()


def main() -> None:
    args = parse_args()
    rank = mpi_rank()
    world_size = mpi_world_size()
    if world_size != 8:
        raise RuntimeError(f"This sweep requires EP=8, got world_size={world_size}")
    if any(batch_size <= 0 for batch_size in args.global_batch_sizes):
        raise ValueError("Global batch sizes must be positive")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ["MOE_BENCH_WEIGHT_CACHE_DIR"] = str(args.weight_cache_dir.resolve())
    if args.rebuild_weight_cache:
        os.environ["MOE_BENCH_REBUILD_WEIGHT_CACHE"] = "1"
    else:
        os.environ.pop("MOE_BENCH_REBUILD_WEIGHT_CACHE", None)
    device = torch.device("cuda", _set_device_from_local_rank())
    model = ModelSpec(
        name="kimi_k3_shape",
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
    )
    mapping = _build_mapping_from_config(
        ConfigSpec(backend="TRTLLM", parallel_mode="DEP"),
        world_size,
    )
    AutoTuner.get().setup_distributed_state(mapping)

    original_prepare = MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend
    MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend = _prepare_local_backend_weights
    try:
        external, external_setup_seconds = _sweep_backend(
            args,
            model,
            mapping,
            device,
            backend="TRTLLM",
            comm_method="NVLINK_ONE_SIDED",
        )
        megamoe, megamoe_setup_seconds = _sweep_backend(
            args,
            model,
            mapping,
            device,
            backend="MEGAMOE_DEEPGEMM",
            comm_method="NONE",
        )
    finally:
        MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend = original_prepare

    if rank == 0:
        rows = []
        for batch_size in args.global_batch_sizes:
            external_ms = external[batch_size]["score_mean_ms"]
            megamoe_ms = megamoe[batch_size]["score_mean_ms"]
            rows.append(
                {
                    "global_batch_size": batch_size,
                    "external_mean_ms": external_ms,
                    "megamoe_mean_ms": megamoe_ms,
                    "megamoe_speedup": external_ms / megamoe_ms,
                    "external_median_ms": external[batch_size]["score_median_ms"],
                    "megamoe_median_ms": megamoe[batch_size]["score_median_ms"],
                }
            )
        receipt = {
            "shape": {
                "world_size": world_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "quant": "W4A8_MXFP4_MXFP8",
                "activation": "SwiGLU",
            },
            "routing": "balanced_alltoall projected through native routing",
            "weights": (
                "Synthetic deterministic weights generated only for each EP rank's "
                "112 local experts; reference weights are not generated. Packed local "
                "weights are cached on disk for subsequent runs."
            ),
            "setup_seconds": {
                "external": external_setup_seconds,
                "megamoe": megamoe_setup_seconds,
            },
            "warmup": args.warmup,
            "iters": args.iters,
            "rows": rows,
            "details": {"external": external, "megamoe": megamoe},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
        print(f"Receipt written to {args.output}")


if __name__ == "__main__":
    main()
