#!/usr/bin/env python3
"""Compare external dispatch->expert->combine with fused MegaMoE."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

TRTLLM_REPO = Path(os.environ.get("TRTLLM_REPO", "/node-storage/trt-llm"))
for search_path in (
    TRTLLM_REPO / "tests" / "microbenchmarks",
    TRTLLM_REPO / "tests" / "unittest",
):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from bench_moe.build import (  # noqa: E402
    _backend_name_from_module,
    _build_moe_module,
    _comm_method_name,
)
from bench_moe.case_runner import _run_autotune  # noqa: E402
from bench_moe.mapping import _build_mapping_from_config  # noqa: E402
from bench_moe.routing import (  # noqa: E402
    _build_routing_plan,
    _project_router_logits_for_plan,
)
from bench_moe.specs import (  # noqa: E402
    ConfigSpec,
    ModelSpec,
    RoutingControlSpec,
    WorkloadSpec,
)
from bench_moe.utils import _set_device_from_local_rank  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner  # noqa: E402
from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correctness and latency comparison for TRTLLM+NVLinkOneSided "
            "dispatch/expert/combine versus fused MegaMoEDeepGemm."
        )
    )
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--tokens-per-rank", type=int, default=1)
    parser.add_argument(
        "--comm-method",
        choices=("NVLINK_ONE_SIDED", "ALLGATHER"),
        default="NVLINK_ONE_SIDED",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=0.3)
    parser.add_argument("--rtol", type=float, default=0.15)
    parser.add_argument("--min-close-fraction", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/a2a_megamoe_pipeline_receipt.json"),
    )
    return parser.parse_args()


def dispatch_expert_combine(
    moe: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: list[int],
) -> torch.Tensor:
    """Run the one function being compared.

    External scheduler:
        comm.dispatch -> backend.run_moe -> comm.combine
    MegaMoE scheduler:
        backend.run_moe, with all three stages fused in the kernel
    """
    return moe.forward(
        hidden_states,
        router_logits,
        all_rank_num_tokens=all_rank_num_tokens,
    )


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(values),
    }


@contextmanager
def _record_external_phases(
    moe: torch.nn.Module,
    current_iteration: list[int],
    records: dict[str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]],
) -> Iterator[None]:
    targets = {
        "dispatch": (moe.comm, "dispatch"),
        "expert_forward": (moe.backend, "run_moe"),
        "combine": (moe.comm, "combine"),
    }
    originals: dict[str, tuple[Any, str, Callable[..., Any]]] = {}

    for phase, (owner, method_name) in targets.items():
        original = getattr(owner, method_name)
        originals[phase] = (owner, method_name, original)

        def wrapped(*args, _phase=phase, _original=original, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = _original(*args, **kwargs)
            end.record()
            records[_phase].append((current_iteration[0], start, end))
            return output

        setattr(owner, method_name, wrapped)

    try:
        yield
    finally:
        for owner, method_name, original in originals.values():
            setattr(owner, method_name, original)


def _benchmark(
    name: str,
    moe: torch.nn.Module,
    run: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    mpi_barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    phase_records: dict[str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]] = {
        "dispatch": [],
        "expert_forward": [],
        "combine": [],
    }
    current_iteration = [-1]
    phase_context = (
        _record_external_phases(moe, current_iteration, phase_records)
        if name == "external"
        else ExitStack()
    )

    with torch.inference_mode(), phase_context:
        for idx in range(iters):
            current_iteration[0] = idx
            starts[idx].record()
            run()
            ends[idx].record()

    torch.cuda.synchronize()
    total_times = [starts[idx].elapsed_time(ends[idx]) for idx in range(iters)]
    result: dict[str, Any] = {"total": _stats(total_times)}
    if name == "external":
        phases: dict[str, dict[str, float]] = {}
        for phase, entries in phase_records.items():
            per_iteration = [0.0] * iters
            for iteration, start, end in entries:
                per_iteration[iteration] += start.elapsed_time(end)
            phases[phase] = _stats(per_iteration)
        result["phases"] = phases
    return result


def _correctness(
    external_output: torch.Tensor,
    megamoe_output: torch.Tensor,
    atol: float,
    rtol: float,
    min_close_fraction: float,
) -> dict[str, Any]:
    actual = external_output.float()
    expected = megamoe_output.float()
    difference = (actual - expected).abs()
    threshold = atol + rtol * expected.abs()
    close_fraction = (difference <= threshold).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(),
        expected.flatten(),
        dim=0,
        eps=1e-12,
    ).item()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "passed": finite and close_fraction >= min_close_fraction,
        "finite": finite,
        "close_fraction": close_fraction,
        "required_close_fraction": min_close_fraction,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "cosine_similarity": cosine,
        "external_absmax": actual.abs().max().item(),
        "megamoe_absmax": expected.abs().max().item(),
    }


def _build_module(
    model: ModelSpec,
    mapping,
    max_num_tokens: int,
    device: torch.device,
    backend: str,
    comm_method: str,
) -> torch.nn.Module:
    common = {
        "model": model,
        "mapping": mapping,
        "use_cuda_graph": False,
        "max_num_tokens": max_num_tokens,
        "use_low_precision_moe_combine": False,
        "enable_perfect_router": False,
        "dtype": torch.bfloat16,
        "routing_logits_dtype": torch.bfloat16,
        "device": device,
    }

    config = ConfigSpec(
        backend=backend,
        parallel_mode="DEP",
        comm_method=comm_method,
        cuda_graph=False,
    )
    if comm_method == "NONE":
        os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    else:
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = comm_method
    module, _ = _build_moe_module(
        config=config,
        moe_backend=backend,
        **common,
    )
    os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    return module


def main() -> None:
    args = parse_args()
    rank = mpi_rank()
    world_size = mpi_world_size()
    if world_size < 2:
        raise RuntimeError("Launch under mpirun with at least two ranks; the intended receipt uses -np 8")
    if args.num_experts % world_size:
        raise ValueError("--num-experts must be divisible by MPI world size")
    if args.hidden_size % 512:
        raise ValueError("MegaMoEDeepGemm requires --hidden-size divisible by 512")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    device_index = _set_device_from_local_rank()
    device = torch.device("cuda", device_index)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model = ModelSpec(
        name="kimi_k3_shape",
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
    )
    config = ConfigSpec(backend="TRTLLM", parallel_mode="DEP")
    mapping = _build_mapping_from_config(config, world_size)
    AutoTuner.get().setup_distributed_state(mapping)

    external = megamoe = None
    try:
        external = _build_module(
            model,
            mapping,
            max_num_tokens=max(args.tokens_per_rank, 1),
            device=device,
            backend="TRTLLM",
            comm_method=args.comm_method,
        )
        if _backend_name_from_module(external) != "TRTLLM":
            raise RuntimeError("External path did not construct TRTLLM backend")
        expected_comm_class = {
            "NVLINK_ONE_SIDED": "NVLinkOneSided",
            "ALLGATHER": "AllGatherReduceScatter",
        }[args.comm_method]
        if _comm_method_name(external) != expected_comm_class:
            raise RuntimeError(
                f"External path did not construct {expected_comm_class}: "
                f"{_comm_method_name(external)}"
            )
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        hidden_states = torch.randn(
            (args.tokens_per_rank, args.hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        router_logits = torch.randn(
            (args.tokens_per_rank, args.num_experts),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        all_rank_num_tokens = [args.tokens_per_rank] * world_size

        routing_spec = RoutingControlSpec(
            routing_mode="forced",
            comm_pattern="balanced_alltoall",
            expert_pattern="balanced",
            seed=args.seed,
        )
        workload = WorkloadSpec(
            num_tokens=args.tokens_per_rank * world_size,
            routing_control=routing_spec,
        )
        plan = _build_routing_plan(
            routing_spec,
            num_tokens=workload.num_tokens,
            world_size=world_size,
            top_k=args.top_k,
            num_experts=args.num_experts,
            moe_ep_size=world_size,
        )
        router_logits, projection_status, projection_reason = _project_router_logits_for_plan(
            plan,
            src_rank=rank,
            routing_method=external.routing_method,
            num_experts=args.num_experts,
            top_k=args.top_k,
            experts_per_rank=args.num_experts // world_size,
            moe_ep_size=world_size,
            device=device,
            dtype=torch.bfloat16,
        )
        if projection_status != "exact":
            raise RuntimeError(
                f"Balanced routing projection was not exact: {projection_status}: "
                f"{projection_reason}"
            )
        external_ids, external_scales = external.routing_method.apply(router_logits)

        with torch.inference_mode():
            AutoTuner.get().clear_cache()
            _run_autotune(
                external,
                hidden_states,
                router_logits,
                all_rank_num_tokens,
                fast_autotune=True,
            )
            external_output = dispatch_expert_combine(
                external, hidden_states, router_logits, all_rank_num_tokens
            ).clone()
        torch.cuda.synchronize()
        external_timing = _benchmark(
            "external",
            external,
            lambda: dispatch_expert_combine(
                external, hidden_states, router_logits, all_rank_num_tokens
            ),
            args.warmup,
            args.iters,
        )

        # The two communication runtimes own separate symmetric-memory
        # allocators. Tear down the external module before constructing
        # MegaMoE, while retaining only its output and routing tensors.
        external.destroy()
        external = None
        torch.cuda.empty_cache()
        mpi_barrier()

        megamoe = _build_module(
            model,
            mapping,
            max_num_tokens=max(args.tokens_per_rank, 1),
            device=device,
            backend="MEGAMOE_DEEPGEMM",
            comm_method="NONE",
        )
        if _backend_name_from_module(megamoe) != "MEGAMOE_DEEPGEMM":
            raise RuntimeError("Fused path did not construct MegaMoEDeepGemm")
        megamoe_ids, megamoe_scales = megamoe.routing_method.apply(router_logits)
        routing_ids_equal = torch.equal(external_ids, megamoe_ids)
        routing_scales_equal = torch.equal(external_scales, megamoe_scales)
        if not routing_ids_equal or not routing_scales_equal:
            raise AssertionError(
                "The two routing methods did not produce identical top-k IDs and scales"
            )

        with torch.inference_mode():
            AutoTuner.get().clear_cache()
            _run_autotune(
                megamoe,
                hidden_states,
                router_logits,
                all_rank_num_tokens,
                fast_autotune=True,
            )
            megamoe_output = dispatch_expert_combine(
                megamoe, hidden_states, router_logits, all_rank_num_tokens
            ).clone()
        torch.cuda.synchronize()
        megamoe_timing = _benchmark(
            "megamoe",
            megamoe,
            lambda: dispatch_expert_combine(
                megamoe, hidden_states, router_logits, all_rank_num_tokens
            ),
            args.warmup,
            args.iters,
        )

        local_correctness = _correctness(
            external_output,
            megamoe_output,
            atol=args.atol,
            rtol=args.rtol,
            min_close_fraction=args.min_close_fraction,
        )

        gathered_correctness = mpi_allgather(local_correctness)
        gathered_external_timing = mpi_allgather(external_timing)
        gathered_megamoe_timing = mpi_allgather(megamoe_timing)
        all_passed = all(item["passed"] for item in gathered_correctness)
        external_score_ms = max(item["total"]["mean_ms"] for item in gathered_external_timing)
        megamoe_score_ms = max(item["total"]["mean_ms"] for item in gathered_megamoe_timing)

        receipt = {
            "status": "PASS" if all_passed else "FAIL",
            "comparison": {
                "external": (
                    f"{expected_comm_class} dispatch -> TRTLLM expert forward -> combine"
                ),
                "fused": "MegaMoEDeepGemm fused dispatch+expert+combine",
            },
            "shape": {
                "world_size": world_size,
                "tokens_per_rank": args.tokens_per_rank,
                "global_tokens": args.tokens_per_rank * world_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "quant": "W4A8_MXFP4_MXFP8",
                "activation": "SwiGLU",
            },
            "controls": {
                "input_seed": args.seed,
                "weight_generator_seed": 42,
                "routing": "projected native balanced_alltoall",
                "routing_projection_status": projection_status,
                "routing_projection_reason": projection_reason,
                "routing_ids_equal": routing_ids_equal,
                "routing_scales_equal": routing_scales_equal,
                "same_logical_weights_verified": False,
                "note": (
                    "Each backend independently generates seed-42 test weights in its "
                    "backend-specific padded layout. Equal logical weights have not "
                    "yet been established across those layouts."
                ),
            },
            "correctness_per_rank": {
                f"rank{idx}": item for idx, item in enumerate(gathered_correctness)
            },
            "timing": {
                "warmup": args.warmup,
                "iters": args.iters,
                "external_per_rank": {
                    f"rank{idx}": item for idx, item in enumerate(gathered_external_timing)
                },
                "megamoe_per_rank": {
                    f"rank{idx}": item for idx, item in enumerate(gathered_megamoe_timing)
                },
                "score": {
                    "external_ms": external_score_ms,
                    "megamoe_ms": megamoe_score_ms,
                    "megamoe_speedup": external_score_ms / megamoe_score_ms,
                },
            },
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2), flush=True)
            print(f"Receipt written to {args.output}", flush=True)
        if not all_passed:
            raise AssertionError("External and MegaMoE outputs failed the correctness threshold")
    finally:
        for module in (external, megamoe):
            if module is not None:
                try:
                    module.destroy()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
