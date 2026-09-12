"""Minimal harness that runs attn_res_fused_tma for ncu profiling.

Usage:
  python run_ncu_target.py --tokens <T> --nvb 4 [--iters 10]

The script:
 1. Compiles the JIT kernel (warmup iters outside ncu capture range)
 2. Arms the cudaProfiler range
 3. Runs `iters` kernel calls under the profiler window
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SGLANG_PYTHON = ROOT.parent / "sglang-opensource" / "python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SGLANG_PYTHON))

HIDDEN_SIZE = 7168
MAX_BANK_ROWS = 8
EPS = 1.0e-6


def make_inputs(tokens: int, nvb: int):
    gen = torch.Generator(device="cuda").manual_seed(20260729)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32) * scale).to(torch.bfloat16)

    prefix = randn(tokens, HIDDEN_SIZE)
    bank   = randn(tokens, MAX_BANK_ROWS, HIDDEN_SIZE)
    cw     = randn(HIDDEN_SIZE, scale=HIDDEN_SIZE**-0.5).contiguous()
    ow     = randn(HIDDEN_SIZE, scale=0.1).add_(1.0).to(torch.bfloat16).contiguous()
    out    = torch.empty_like(prefix)
    return prefix, bank, cw, ow, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--nvb",    type=int, default=4)
    parser.add_argument("--iters",  type=int, default=10,
                        help="Kernel launches inside the ncu capture window")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Kernel launches before opening the ncu window")
    args = parser.parse_args()

    from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma  # noqa: F401

    prefix, bank, cw, ow, out = make_inputs(args.tokens, args.nvb)

    # Warmup: force JIT compilation before ncu touches the stream
    for _ in range(args.warmup):
        attn_res_fused_tma(prefix, bank, cw, ow, out, args.nvb, EPS)
    torch.cuda.synchronize()

    # Open cudaProfiler window (ncu uses -c cudaProfilerApi)
    torch.cuda.cudart().cudaProfilerStart()

    for _ in range(args.iters):
        attn_res_fused_tma(prefix, bank, cw, ow, out, args.nvb, EPS)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"[harness] T={args.tokens} nvb={args.nvb} done ({args.iters} iters inside ncu window)")


if __name__ == "__main__":
    main()
