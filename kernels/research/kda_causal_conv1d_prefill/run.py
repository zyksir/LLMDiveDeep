#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral correctness and performance runner."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch

from common import (
    Backend,
    Shape,
    compare,
    correctness_cases,
    full_matrix,
    hardware_identity,
    main_performance_matrix,
    make_problem,
    time_cuda,
    write_json,
)

PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
BACKEND_MODULES = {
    "trt_native": "backends.trt_native",
    "trt_triton_head": "backends.trt_triton_head",
    "cute": "backends.cute_direct",
}


def load_backends(names: list[str]) -> list[Backend]:
    loaded: list[Backend] = []
    for name in names:
        if name not in BACKEND_MODULES:
            raise ValueError(f"unknown backend {name!r}; choices={sorted(BACKEND_MODULES)}")
        module = importlib.import_module(BACKEND_MODULES[name])
        loaded.append(module.Backend())
    return loaded


def _run_once(backend: Backend, problem) -> tuple[torch.Tensor, torch.Tensor, float]:
    prepared = backend.prepare(problem)
    start = time.perf_counter()
    output = prepared.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, prepared.state(), elapsed


def run_correctness(backends: list[Backend]) -> list[dict[str, object]]:
    native = load_backends(["trt_native"])[0]
    records: list[dict[str, object]] = []
    for case_name, prototype in correctness_cases():
        reference_problem = prototype.clone()
        reference_output, reference_state, reference_first = _run_once(native, reference_problem)
        for backend in backends:
            problem = prototype.clone()
            supported, reason = backend.supports(problem)
            record: dict[str, object] = {
                "case": case_name,
                "shape": prototype.shape.key,
                "backend": backend.name,
                "source": backend.source,
                "supported": supported,
            }
            if not supported:
                record.update({"correct": None, "skip_reason": reason})
            else:
                output, state, first_seconds = _run_once(backend, problem)
                record.update(compare(output, state, reference_output, reference_state))
                record.update(
                    {
                        "first_call_seconds": first_seconds,
                        "reference_first_call_seconds": reference_first,
                    }
                )
            records.append(record)
            print("CHECK " + json.dumps(record, sort_keys=True))
    return records


def run_matrix_correctness(
    backends: list[Backend],
    matrix: str,
) -> list[dict[str, object]]:
    native = load_backends(["trt_native"])[0]
    records: list[dict[str, object]] = []
    for shape in _select_matrix(matrix):
        prototype = make_problem(shape)
        reference_problem = prototype.clone()
        reference_output, reference_state, _ = _run_once(native, reference_problem)
        for backend in backends:
            problem = prototype.clone()
            supported, reason = backend.supports(problem)
            record: dict[str, object] = {
                "shape": shape.key,
                "backend": backend.name,
                "source": backend.source,
                "supported": supported,
            }
            if not supported:
                record.update({"correct": None, "skip_reason": reason})
            else:
                output, state, first_seconds = _run_once(backend, problem)
                record.update(compare(output, state, reference_output, reference_state))
                record["first_call_seconds"] = first_seconds
            records.append(record)
            print("MATRIX_CHECK " + json.dumps(record, sort_keys=True))
    return records


def _select_matrix(name: str) -> list[Shape]:
    if name == "quick":
        return [
            Shape("bf16", 4, 1536, 128, 1),
            Shape("bf16", 4, 1536, 1024, 8),
            Shape("bf16", 4, 3072, 8192, 8),
        ]
    if name == "main":
        return main_performance_matrix()
    if name == "full":
        return full_matrix()
    raise ValueError(name)


def run_benchmark(
    backends: list[Backend],
    matrix: str,
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for shape in _select_matrix(matrix):
        prototype = make_problem(shape)
        for backend in backends:
            problem = prototype.clone()
            supported, reason = backend.supports(problem)
            record: dict[str, object] = {
                "shape": shape.key,
                "dtype": shape.dtype,
                "width": shape.width,
                "channels": shape.channels,
                "tokens": shape.tokens,
                "batch": shape.batch,
                "backend": backend.name,
                "source": backend.source,
                "supported": supported,
            }
            if not supported:
                record.update({"latency_ms": None, "skip_reason": reason})
            else:
                setup_start = time.perf_counter()
                prepared = backend.prepare(problem)
                setup_seconds = time.perf_counter() - setup_start
                first_start = time.perf_counter()
                prepared.run()
                torch.cuda.synchronize()
                first_seconds = time.perf_counter() - first_start
                record.update(time_cuda(prepared, warmup, iterations))
                record.update(
                    {
                        "setup_seconds": setup_seconds,
                        "first_call_seconds": first_seconds,
                        "launch_count": prepared.launch_count,
                    }
                )
            records.append(record)
            print("BENCH " + json.dumps(record, sort_keys=True))
    return records


def measure_machine_floors(iterations: int = 1000) -> dict[str, object]:
    # The tiny sleep is a real one-kernel launch with negligible device work.
    for _ in range(20):
        torch.cuda._sleep(1)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.cuda._sleep(1)
    end.record()
    torch.cuda.synchronize()
    launch_floor_ms = start.elapsed_time(end) / iterations

    # Keep the working set well beyond LLC/L2 so this is a device-memory copy
    # bandwidth measurement rather than a cache-bandwidth measurement.
    copy_bytes = 1024 * 1024 * 1024
    source = torch.empty(copy_bytes // 2, dtype=torch.float16, device="cuda")
    target = torch.empty_like(source)
    for _ in range(5):
        target.copy_(source)
    torch.cuda.synchronize()
    start.record()
    for _ in range(30):
        target.copy_(source)
    end.record()
    torch.cuda.synchronize()
    copy_ms = start.elapsed_time(end) / 30
    copy_read_write_gbps = 2 * copy_bytes / (copy_ms * 1.0e-3) / 1.0e9

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    # SM103 has 128 FP32 lanes/SM. This is an upper bound used only for a lower
    # bound on arithmetic time. Benchmark commands use gpu_run's 1.5 GHz lock;
    # allow explicit override for runs with a different lock.
    assumed_clock_ghz = float(os.environ.get("KDA_SM_CLOCK_GHZ", "1.5"))
    fp32_peak_tflops = (
        properties.multi_processor_count * 128 * 2 * assumed_clock_ghz / 1000
    )
    return {
        "launch_floor_ms": float(launch_floor_ms),
        "copy_kernel_ms": float(copy_ms),
        "copy_payload_bytes": copy_bytes,
        "copy_read_write_gbps": float(copy_read_write_gbps),
        "fp32_peak_tflops": float(fp32_peak_tflops),
        "assumed_sm_clock_ghz": assumed_clock_ghz,
        "sm_count": properties.multi_processor_count,
    }


def sol_lower_bound(shape: Shape, floors: dict[str, object]) -> dict[str, float]:
    element_bytes = 2
    live_sequences = shape.batch
    initial_sequences = (shape.batch + 1) // 2
    mandatory_bytes = (
        2 * shape.tokens * shape.channels * element_bytes
        + shape.channels * shape.width * element_bytes
        + (shape.channels * element_bytes if shape.bias else 0)
        + (initial_sequences + live_sequences)
        * shape.channels
        * (shape.width - 1)
        * element_bytes
        + (shape.batch + 1 + shape.batch) * 4
        + shape.batch
    )
    bandwidth_ms = mandatory_bytes / (float(floors["copy_read_write_gbps"]) * 1.0e9) * 1.0e3
    flops = 2 * shape.tokens * shape.channels * shape.width
    arithmetic_ms = flops / (float(floors["fp32_peak_tflops"]) * 1.0e12) * 1.0e3
    # A dependent FFMA is conservatively modeled as four cycles. The chain is
    # per output and massively parallel; this is a per-chain latency floor.
    serial_width_ms = (
        shape.width * 4 / (float(floors["assumed_sm_clock_ghz"]) * 1.0e9) * 1.0e3
    )
    lower_bound_ms = max(
        float(floors["launch_floor_ms"]),
        bandwidth_ms,
        arithmetic_ms,
        serial_width_ms,
    )
    return {
        "mandatory_bytes": float(mandatory_bytes),
        "bandwidth_floor_ms": bandwidth_ms,
        "arithmetic_flops": float(flops),
        "arithmetic_floor_ms": arithmetic_ms,
        "serial_width_floor_ms": serial_width_ms,
        "defensible_lower_bound_ms": lower_bound_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("correctness", "matrix-correctness", "benchmark", "all", "floors"),
        default="all",
    )
    parser.add_argument(
        "--backends",
        default="trt_native,trt_triton_head",
        help=f"comma-separated names from {sorted(BACKEND_MODULES)}",
    )
    parser.add_argument("--matrix", choices=("quick", "main", "full"), default="quick")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    floors = measure_machine_floors()
    payload: dict[str, object] = {
        "hardware": hardware_identity(),
        "floors": floors,
        "lower_bounds": {
            shape.key: sol_lower_bound(shape, floors)
            for shape in main_performance_matrix()
        },
        "correctness": [],
        "benchmarks": [],
    }
    if args.mode != "floors":
        backends = load_backends([name for name in args.backends.split(",") if name])
        if args.mode in ("correctness", "all"):
            payload["correctness"] = run_correctness(backends)
        if args.mode == "matrix-correctness":
            payload["correctness"] = run_matrix_correctness(backends, args.matrix)
        if args.mode in ("benchmark", "all"):
            payload["benchmarks"] = run_benchmark(
                backends,
                args.matrix,
                args.warmup,
                args.iterations,
            )
    payload["full_run_wall_seconds"] = time.perf_counter() - started
    output = args.output or RESULTS_DIR / f"{args.mode}_{args.matrix}.json"
    write_json(output, payload)
    correct_values = [
        record["correct"]
        for record in payload["correctness"]
        if record.get("correct") is not None
    ]
    all_correct = all(correct_values) if correct_values else None
    correctness_label = "NOT_RUN" if all_correct is None else ("PASS" if all_correct else "FAIL")
    print(f"CORRECT: {correctness_label}")
    print("RESULT: " + json.dumps({"correct": all_correct, "results": str(output)}))
    if all_correct is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
