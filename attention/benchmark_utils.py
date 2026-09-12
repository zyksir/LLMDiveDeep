"""Shared forward-kernel benchmark utilities. Nothing runs at import time."""

import csv
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import torch


RESULTS = Path(__file__).resolve().parent / "results"


def results_path(path):
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to(RESULTS) or target == RESULTS:
        raise ValueError(f"result files must be under {RESULTS}")
    return target


def environment(device):
    versions = {}
    for package in (
        "flash-attn",
        "flash-attn-4",
        "sglang",
        "sgl-kernel",
        "flashinfer-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return dict(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=gpu,
        packages=json.dumps(versions, sort_keys=True),
    )


def time_forward(fn, device, *, warmup, iters, repeats):
    """Median per-call time; warmup/JIT excluded. Forward wrappers, no backward.

    CUDA events measure stream elapsed time, including launch starvation.
    This is not a CUDA-graph or single-instruction throughput measurement.
    """
    if min(warmup, iters, repeats) < 1:
        raise ValueError("warmup, iters, repeats must be positive")
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(iters):
                    fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / iters)
            else:
                start = time.perf_counter()
                for _ in range(iters):
                    fn()
                samples.append((time.perf_counter() - start) * 1e6 / iters)
    return statistics.median(samples)


def write_rows(rows, path):
    path = results_path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_arguments(parser, family):
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly execute kernels; omitted means help only",
    )
    parser.add_argument(
        "--list", action="store_true", help="list adapter names without running kernels"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--csv", type=Path, default=RESULTS / family / "benchmark.csv")


def runtime_device(args):
    if min(args.warmup, args.iters, args.repeats) < 1:
        raise ValueError("timing counts must be positive")
    results_path(args.csv)  # validate before compiling or running anything
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; request --device cpu for the torch references"
            )
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("this harness supports CPU or CUDA timing")
    torch.manual_seed(args.seed)
    return device
