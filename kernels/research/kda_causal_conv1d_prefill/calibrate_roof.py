#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the attainable bandwidth roof for the conv workload pattern.

The round-2 SOL gate divided mandatory bytes by a 1 GiB contiguous-copy
bandwidth (6.3 TB/s at 1500 MHz). This script measures what a minimal
streaming kernel with the *same* byte volume and *same* access pattern as the
real workload (read [T,D] bf16 rows, write [T,D] rows, ~100-151 MB total) can
actually attain on this device at the locked clock, including the strided
production-row pattern (stride 4752 read as [:, :4608]).

Every candidate roof kernel does strictly less work than the real operation
(no conv, no SiLU, no state), so its bandwidth is a defensible upper bound on
what the real kernel could reach for this size and pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from common import quantiles, write_json  # noqa: E402


def make_stream_copy_launcher(
    tokens: int,
    channels: int,
    token_tile: int,
    vector_width: int,
    threads: int,
):
    """Minimal same-pattern streaming kernel: one CTA per
    (channel block, token tile), one vector load + one vector store per token,
    identical to the conv kernel's memory access skeleton."""

    channel_vectors = channels // vector_width
    channel_blocks = (channel_vectors + threads - 1) // threads
    token_blocks = (tokens + token_tile - 1) // token_tile

    @cute.kernel
    def stream_copy_kernel(mX: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, token_block, _ = cute.arch.block_idx()
        channel_vector = channel_block * threads + tidx
        active = channel_vector * vector_width < channels
        token_base = token_block * token_tile
        for token_inner in cutlass.range_constexpr(token_tile):
            token = token_base + token_inner
            if (token < tokens) & active:
                row = mX[token, None]
                row_vectors = cute.zipped_divide(row, (vector_width,))
                fragment = cute.make_fragment((vector_width,), mX.element_type)
                cute.autovec_copy(row_vectors[(None, channel_vector)], fragment)
                output_row = mOut[token, None]
                output_vectors = cute.zipped_divide(output_row, (vector_width,))
                cute.autovec_copy(fragment, output_vectors[(None, channel_vector)])

    @cute.jit
    def launch(mX: cute.Tensor, mOut: cute.Tensor, stream):
        stream_copy_kernel(mX, mOut).launch(
            grid=[channel_blocks, token_blocks, 1],
            block=[threads, 1, 1],
            stream=stream,
        )

    return launch


def _time_callable(run, warmup: int, iterations: int, rounds: int) -> dict:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return quantiles(samples)


def measure_stream_copy(
    tokens: int,
    channels: int,
    row_stride: int | None,
    token_tile: int,
    vector_width: int,
    threads: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict:
    device = torch.device("cuda")
    if row_stride is None:
        source = torch.randn(tokens, channels, device=device, dtype=torch.bfloat16)
    else:
        buffer = torch.randn(tokens, row_stride, device=device, dtype=torch.bfloat16)
        source = buffer[:, :channels]
    output = torch.empty(tokens, channels, device=device, dtype=torch.bfloat16)

    stride_bytes = source.stride(0) * source.element_size()
    align = 32
    for candidate in (32, 16, 8):
        if source.data_ptr() % candidate == 0 and stride_bytes % candidate == 0:
            align = candidate
            break
    if vector_width * 2 > align:
        return {
            "skipped": True,
            "reason": f"vector {vector_width} needs {vector_width * 2}B align, "
            f"measured {align}B",
        }

    launcher = make_stream_copy_launcher(
        tokens, channels, token_tile, vector_width, threads
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    arguments = (
        from_dlpack(source, assumed_align=align),
        from_dlpack(output, assumed_align=32),
        stream,
    )
    compile_start = time.perf_counter()
    compiled = cute.compile(launcher, *arguments)
    compile_seconds = time.perf_counter() - compile_start

    def run():
        compiled(*arguments)

    run()
    torch.cuda.synchronize()
    if not torch.equal(output, source.contiguous()):
        raise RuntimeError("stream copy produced wrong bytes")

    timing = _time_callable(run, warmup, iterations, rounds)
    traffic_bytes = 2 * tokens * channels * 2
    gbps = traffic_bytes / (timing["median_ms"] * 1e-3) / 1e9
    return {
        "tokens": tokens,
        "channels": channels,
        "row_stride": row_stride,
        "token_tile": token_tile,
        "vector_width": vector_width,
        "threads": threads,
        "measured_align_bytes": align,
        "traffic_bytes": traffic_bytes,
        "latency_quantiles_ms": timing,
        "median_ms": timing["median_ms"],
        "achieved_gbps": gbps,
        "compile_seconds": compile_seconds,
    }


def measure_torch_copy(elements: int, warmup: int, iterations: int, rounds: int) -> dict:
    source = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    target = torch.empty_like(source)

    def run():
        target.copy_(source)

    timing = _time_callable(run, warmup, iterations, rounds)
    traffic_bytes = 2 * elements * 2
    return {
        "elements": elements,
        "traffic_bytes": traffic_bytes,
        "median_ms": timing["median_ms"],
        "latency_quantiles_ms": timing,
        "achieved_gbps": traffic_bytes / (timing["median_ms"] * 1e-3) / 1e9,
    }


def measure_launch_floor(iterations: int = 1000) -> float:
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
    return float(start.elapsed_time(end) / iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR / "results" / "r3_roof_calibration.json",
    )
    args = parser.parse_args()

    payload: dict = {
        "device": torch.cuda.get_device_properties(0).name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "launch_floor_ms": measure_launch_floor(),
        "torch_copies": [],
        "stream_copies": [],
    }

    # Torch flat copies: 1 GiB reference plus the exact workload payloads.
    for elements, tag in (
        (512 * 1024 * 1024, "1GiB_reference"),
        (8192 * 3072, "T8192_D3072_payload"),
        (8192 * 4608, "T8192_D4608_payload"),
        (1024 * 3072, "T1024_D3072_payload"),
    ):
        record = measure_torch_copy(elements, 10, 30 if elements > 1e8 else 300, 3)
        record["tag"] = tag
        payload["torch_copies"].append(record)
        print("TORCH_COPY " + json.dumps(record, sort_keys=True))

    # Same-pattern streaming kernels (contiguous D3072/D1536 and strided
    # production D4608). Sweep the pure-copy mechanism space so the roof is
    # the best attainable for this pattern, not one arbitrary configuration.
    configs = []
    for vector_width in (4, 8, 16):
        configs.append((8192, 3072, None, 12, vector_width, 128))
    configs.extend(
        [
            (8192, 3072, None, 4, 8, 128),
            (8192, 3072, None, 24, 8, 128),
            (8192, 3072, None, 48, 8, 128),
            (8192, 1536, None, 12, 8, 128),
            (1024, 3072, None, 12, 8, 128),
            (128, 3072, None, 12, 8, 128),
            (8192, 4608, 4752, 12, 4, 128),
            (8192, 4608, 4752, 12, 8, 128),
            (8192, 4608, 4752, 12, 16, 128),
            (8192, 4608, 4752, 24, 8, 128),
        ]
    )
    for tokens, channels, row_stride, tile, vector_width, threads in configs:
        record = measure_stream_copy(
            tokens,
            channels,
            row_stride,
            tile,
            vector_width,
            threads,
            args.warmup,
            args.iterations,
            args.rounds,
        )
        payload["stream_copies"].append(record)
        print("STREAM_COPY " + json.dumps(record, sort_keys=True))

    def best_gbps(predicate):
        rates = [
            record["achieved_gbps"]
            for record in payload["stream_copies"]
            if not record.get("skipped") and predicate(record)
        ]
        return max(rates) if rates else None

    payload["calibrated_roofs_gbps"] = {
        "contiguous_T8192_D3072": best_gbps(
            lambda r: r["tokens"] == 8192
            and r["channels"] == 3072
            and r["row_stride"] is None
        ),
        "contiguous_T8192_D1536": best_gbps(
            lambda r: r["tokens"] == 8192 and r["channels"] == 1536
        ),
        "contiguous_T1024_D3072": best_gbps(
            lambda r: r["tokens"] == 1024 and r["channels"] == 3072
        ),
        "contiguous_T128_D3072": best_gbps(
            lambda r: r["tokens"] == 128 and r["channels"] == 3072
        ),
        "strided_T8192_D4608": best_gbps(
            lambda r: r["channels"] == 4608 and r["row_stride"] == 4752
        ),
    }
    write_json(args.output, payload)
    print("RESULT " + json.dumps(payload["calibrated_roofs_gbps"], sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
