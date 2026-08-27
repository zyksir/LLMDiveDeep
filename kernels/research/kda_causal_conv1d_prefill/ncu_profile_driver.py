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

    shape = Shape("bf16", args.W, args.D, args.T, args.B)
    problem = make_problem(shape)

    # Load backend
    module = importlib.import_module(BACKEND_MODULES[args.backend])
    backend = module.Backend()

    supported, reason = backend.supports(problem)
    if not supported:
        print(f"[PROFILE_DRIVER] Backend {args.backend!r} does not support shape: {reason}", file=sys.stderr)
        sys.exit(1)

    # Prepare (includes JIT compilation for CuTe; Triton lazily compiles on first call)
    print(f"[PROFILE_DRIVER] Preparing backend={args.backend} T={args.T} D={args.D} B={args.B} tile={args.tile}", flush=True)
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
