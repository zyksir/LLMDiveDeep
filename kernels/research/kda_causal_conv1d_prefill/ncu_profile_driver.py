#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""NCU profiling driver: compiles kernel, warms up, then runs target iterations
within a cudaProfilerApi capture range so ncu captures only the hot path."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import sys
from pathlib import Path

import torch

# Add package dir to path
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from common import Shape, make_problem

BACKEND_MODULES = {
    "trt_triton_head": "backends.trt_triton_head",
    "cute": "backends.cute_direct",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCU profiling driver")
    p.add_argument("--backend", choices=list(BACKEND_MODULES), required=True)
    p.add_argument("--T", type=int, required=True, help="total tokens")
    p.add_argument("--D", type=int, required=True, help="channels")
    p.add_argument("--B", type=int, required=True, help="batch size (1 or 8)")
    p.add_argument("--W", type=int, default=4, help="conv width")
    p.add_argument("--tile", type=int, default=None, help="CuTe token tile (None=auto)")
    p.add_argument(
        "--algorithm",
        choices=("direct", "stream"),
        default="direct",
        help="CuTe algorithm",
    )
    p.add_argument("--vector-width", type=int, default=4)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--silu-mode", choices=("expdiv", "tanh"), default="expdiv")
    p.add_argument("--prefetch", type=int, choices=(0, 1), default=1)
    p.add_argument("--group-loads", action="store_true")
    p.add_argument("--group-span", type=int, choices=(4, 8, 12), default=4)
    p.add_argument(
        "--row-stride",
        type=int,
        default=None,
        help="physical input row stride in elements (production: 4752)",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help="use the production no-bias variant",
    )
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations (unprofiled)")
    p.add_argument("--iterations", type=int, default=3, help="profiled iterations")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load cudaProfilerApi symbols
    try:
        libcudart = ctypes.CDLL("libcudart.so")
    except OSError:
        libcudart = ctypes.CDLL("libcudart.so.1")

    # Environment: tile override for CuTe backend
    if args.tile is not None:
        os.environ["KDA_CUTE_TOKEN_TILE"] = str(args.tile)
    else:
        os.environ["KDA_CUTE_TOKEN_TILE"] = "auto"
    os.environ["KDA_CUTE_ALGORITHM"] = args.algorithm
    os.environ["KDA_CUTE_VECTOR_WIDTH"] = str(args.vector_width)
    os.environ["KDA_CUTE_THREADS"] = str(args.threads)
    os.environ["KDA_CUTE_SILU_MODE"] = args.silu_mode
    os.environ["KDA_CUTE_PREFETCH"] = str(args.prefetch)
    os.environ["KDA_CUTE_GROUP_LOADS"] = "1" if args.group_loads else "0"
    os.environ["KDA_CUTE_GROUP_SPAN"] = str(args.group_span)

    shape = Shape(
        "bf16",
        args.W,
        args.D,
        args.T,
        args.B,
        bias=not args.no_bias,
        row_stride=args.row_stride,
    )
    problem = make_problem(shape)

    # Load backend
    module = importlib.import_module(BACKEND_MODULES[args.backend])
    backend = module.Backend()

    supported, reason = backend.supports(problem)
    if not supported:
        print(f"[PROFILE_DRIVER] Backend {args.backend!r} does not support shape: {reason}", file=sys.stderr)
        sys.exit(1)

    # Prepare (includes JIT compilation for CuTe; Triton lazily compiles on first call)
    print(
        f"[PROFILE_DRIVER] Preparing backend={args.backend} T={args.T} "
        f"D={args.D} B={args.B} algorithm={args.algorithm} tile={args.tile} "
        f"vector_width={args.vector_width} threads={args.threads}",
        flush=True,
    )
    prepared = backend.prepare(problem)

    # First call to trigger Triton JIT compilation (if applicable)
    print("[PROFILE_DRIVER] First call (JIT trigger) ...", flush=True)
    prepared.run()
    torch.cuda.synchronize()

    # Warmup iterations (un-profiled)
    print(f"[PROFILE_DRIVER] Warmup ({args.warmup} iters) ...", flush=True)
    for _ in range(args.warmup):
        prepared.run()
    torch.cuda.synchronize()

    # Profiled iterations under cudaProfilerApi capture range
    print(f"[PROFILE_DRIVER] Starting profiled region ({args.iterations} iters) ...", flush=True)
    libcudart.cudaProfilerStart()
    for _ in range(args.iterations):
        prepared.run()
    torch.cuda.synchronize()
    libcudart.cudaProfilerStop()

    print("[PROFILE_DRIVER] Done.", flush=True)


if __name__ == "__main__":
    main()
