import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from common import Shape, make_inputs
from deepgemm_backend import DeepGEMMBackend
from mok_backend import MoKBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mok-comm-sms", type=int, default=24)
    parser.add_argument("--mok-minibatch", type=int, default=4096)
    parser.add_argument("--sweep-mok", action="store_true")
    parser.add_argument("--sweep-warmup", type=int, default=5)
    parser.add_argument("--sweep-iterations", type=int, default=20)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def initialize_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, world_size, device


def compare_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    max_abs = diff.abs().max()
    diff_l2_sq = diff.square().sum(dtype=torch.float64)
    expected_l2_sq = expected_f.square().sum(dtype=torch.float64)
    dot = (actual_f * expected_f).sum(dtype=torch.float64)
    actual_l2_sq = actual_f.square().sum(dtype=torch.float64)
    all_finite = torch.isfinite(actual_f).all() & torch.isfinite(expected_f).all()

    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    for value in (diff_l2_sq, expected_l2_sq, dot, actual_l2_sq):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)

    relative_l2 = math.sqrt(diff_l2_sq.item() / max(expected_l2_sq.item(), 1e-30))
    cosine = dot.item() / math.sqrt(
        max(actual_l2_sq.item() * expected_l2_sq.item(), 1e-30)
    )
    return {
        "max_abs": max_abs.item(),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "all_finite": bool(all_finite.item()),
    }


def rank_max_samples(
    local_samples_ms: list[float],
    device: torch.device,
) -> list[float]:
    local = torch.tensor(local_samples_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.stack(gathered).max(dim=0).values.cpu().tolist()


def benchmark(
    backend,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        result = backend.run()
        result = None

    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        result = backend.run()
        end.record()
        result = None
    torch.cuda.synchronize(device)
    wall_time_s = time.perf_counter() - wall_start
    dist.barrier()

    samples = rank_max_samples(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        device,
    )
    samples_tensor = torch.tensor(samples, dtype=torch.float64)
    return {
        "median_ms": torch.quantile(samples_tensor, 0.5).item(),
        "p20_ms": torch.quantile(samples_tensor, 0.2).item(),
        "p80_ms": torch.quantile(samples_tensor, 0.8).item(),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "wall_time_s": wall_time_s,
        "rank_max_samples_ms": samples,
    }


def pytorch_reference(inputs) -> torch.Tensor:
    source_root = Path(os.environ["MOK_SOURCE_ROOT"])
    spec = importlib.util.spec_from_file_location(
        "mok_reference_utils",
        source_root / "tests" / "utils.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the pinned MoK PyTorch reference")
    reference_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_utils)

    combine, _, _, _, y_shared = reference_utils.run_forward_reference_bf16(
        inputs.x,
        inputs.topk_experts,
        inputs.w_shared_gate,
        inputs.w_shared_up,
        inputs.w_shared_down,
        inputs.w_routed_gate,
        inputs.w_routed_up,
        inputs.w_routed_down,
    )
    return reference_utils.run_fwd_epilogue_reference(
        y_shared,
        combine,
        inputs.router_weights,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, device = initialize_distributed()
    if world_size != 4:
        raise ValueError("This frozen benchmark requires EP4")

    shape = Shape()
    setup_start = time.perf_counter()
    inputs = make_inputs(rank, world_size, device, shape)
    mok = MoKBackend(
        inputs,
        shape,
        fwd_num_comm_sms=args.mok_comm_sms,
        minibatch_size=args.mok_minibatch,
    )
    deepgemm = DeepGEMMBackend(inputs, shape)
    torch.cuda.synchronize(device)
    setup_time_s = time.perf_counter() - setup_start

    first_run_start = time.perf_counter()
    mok_output = mok.run()[0]
    deepgemm_output = deepgemm.run()[0]
    torch.cuda.synchronize(device)
    first_run_time_s = time.perf_counter() - first_run_start
    correctness = {
        "deepgemm_vs_mok": compare_outputs(deepgemm_output, mok_output),
    }

    if args.check_reference:
        reference_start = time.perf_counter()
        reference_output = pytorch_reference(inputs)
        torch.cuda.synchronize(device)
        reference_time_s = time.perf_counter() - reference_start
        correctness["mok_vs_pytorch"] = compare_outputs(mok_output, reference_output)
        correctness["deepgemm_vs_pytorch"] = compare_outputs(
            deepgemm_output,
            reference_output,
        )
        del reference_output
    else:
        reference_time_s = None
    del mok_output, deepgemm_output
    torch.cuda.empty_cache()

    mok_sweep = []
    if args.sweep_mok:
        best_config = None
        best_latency_ms = math.inf
        for fwd_num_comm_sms in range(4, 53, 4):
            for minibatch_size in (2048, 4096, 8192, 16384):
                candidate = MoKBackend(
                    inputs,
                    shape,
                    fwd_num_comm_sms=fwd_num_comm_sms,
                    minibatch_size=minibatch_size,
                )
                result = benchmark(
                    candidate,
                    device,
                    args.sweep_warmup,
                    args.sweep_iterations,
                )
                sweep_entry = {
                    "fwd_num_comm_sms": fwd_num_comm_sms,
                    "minibatch_size": minibatch_size,
                    "median_ms": result["median_ms"],
                    "p20_ms": result["p20_ms"],
                    "p80_ms": result["p80_ms"],
                }
                mok_sweep.append(sweep_entry)
                if result["median_ms"] < best_latency_ms:
                    best_latency_ms = result["median_ms"]
                    best_config = (fwd_num_comm_sms, minibatch_size)
        assert best_config is not None
        mok = MoKBackend(
            inputs,
            shape,
            fwd_num_comm_sms=best_config[0],
            minibatch_size=best_config[1],
        )

    methods = {}
    for backend in (mok, deepgemm):
        methods[backend.name] = benchmark(
            backend,
            device,
            args.warmup,
            args.iterations,
        )

    per_rank_flops = (
        6
        * (shape.tokens_per_rank + shape.tokens_per_rank * shape.topk)
        * shape.hidden
        * shape.intermediate
    )
    for result in methods.values():
        result["tflops_per_rank"] = (
            per_rank_flops / (result["median_ms"] * 1e9)
        )

    report = {
        "contract": {
            "precision": "BF16 inputs, weights, intermediates, and output",
            "activation": "unclamped SwiGLU",
            "shape": shape.__dict__,
            "ep_size": world_size,
            "timed_boundary": (
                "input/routing copies, global schedule or in-kernel scheduling, "
                "dispatch, shared+routed experts, combine, router-weight reduction, "
                "and final BF16 output"
            ),
            "launch_mode": "eager CUDA events; per-sample maximum across ranks",
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
        },
        "provenance": {
            "mok_commit": "22fc95ae6e331a738c4a58a227a8b03cac586e12",
            "deepgemm_commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        },
        "correctness": correctness,
        "mok_quick_sweep": mok_sweep,
        "methods": methods,
        "dev_loop": {
            "setup_time_s": setup_time_s,
            "cold_first_run_time_s": first_run_time_s,
            "reference_time_s": reference_time_s,
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    deepgemm.close()
    mok.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
