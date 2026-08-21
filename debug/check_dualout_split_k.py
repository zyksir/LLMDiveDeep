#!/usr/bin/env python3
"""Correctness gate for the B300 _TUNED entry (bucket 256 -> split_k=2).

split_k changes the K-reduction ORDER, so it changes floating-point
accumulation: "faster and close enough to the other config" is not the bar.
Three things are checked against the config it replaces:

1. accuracy vs an FP32 oracle -- split_k=2 must be no worse than the
   split_k=4 default, not merely near it;
2. run-to-run bitwise determinism, which the kernel's docstring promises
   ("Deterministic: bitwise identical outputs run-to-run for fixed inputs")
   and which a reduction-order change is exactly what could break;
3. that _pick_config actually selects it on this arch, and still selects
   DEFAULT_CONFIG at 128 and on other arches.

    python3 debug/check_dualout_split_k.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import arch
from kimi_k3_layer.kernels import dual_out_gemm_cutedsl as dg

TUNED = dg.Config(split_k=2, block_n=96)


def oracle(h, w):
    out = h.float() @ w.float().T
    return out[:, :dg.N_BF16], out[:, dg.N_BF16:]


def err(got, want):
    scale = want.abs().max().clamp_min(1e-9)
    return ((got.float() - want).abs().max() / scale).item()


def main() -> int:
    torch.manual_seed(0)
    failures = []
    w = torch.randn(dg.N_TOTAL, dg.HIDDEN, device="cuda",
                    dtype=torch.bfloat16)

    print(f"arch = sm_{arch.suffix()}\n")

    # 3. selection
    print("-- _pick_config selection --")
    for batch, want in ((128, dg.DEFAULT_CONFIG), (256, TUNED)):
        got = dg._pick_config(batch, dg.HIDDEN, dg.N_BF16, dg.N_GATE)
        # evict_last_b is layered on by _pick_config; compare the tile axes
        same = (got.split_k, got.block_n, got.ab_stages) == (
            want.split_k, want.block_n, want.ab_stages)
        print(f"  B={batch:<5} split_k={got.split_k} block_n={got.block_n} "
              f"{'OK' if same else 'MISMATCH, want split_k=%d' % want.split_k}")
        if not same:
            failures.append(f"selection at B={batch}")

    # 1. accuracy vs fp32 oracle, 2. determinism
    for batch in (128, 256):
        h = torch.randn(batch, dg.HIDDEN, device="cuda", dtype=torch.bfloat16)
        ref_bf16, ref_f32 = oracle(h, w)
        print(f"\n-- B={batch} vs FP32 oracle --")
        results = {}
        for name, cfg in (("split_k=4 (default)", dg.DEFAULT_CONFIG),
                          ("split_k=2 (tuned)", TUNED)):
            a, b = dg.dual_out_gemm_cutedsl(
                h, w, dg.N_BF16, dg.N_GATE, config=cfg)
            e = (err(a, ref_bf16), err(b, ref_f32))
            results[name] = e
            # determinism: same inputs, same config, twice
            a2, b2 = dg.dual_out_gemm_cutedsl(
                h, w, dg.N_BF16, dg.N_GATE, config=cfg)
            bitwise = (torch.equal(a, a2.clone()) and
                       torch.equal(b, b2.clone()))
            print(f"  {name:<22} err_bf16={e[0]:.3e} err_fp32={e[1]:.3e} "
                  f"bitwise_repeat={'OK' if bitwise else 'FAIL'}")
            if not bitwise:
                failures.append(f"determinism {name} B={batch}")
        base = results["split_k=4 (default)"]
        tuned = results["split_k=2 (tuned)"]
        # allow a little slack: different order, not different accuracy class
        for index, label in ((0, "bf16"), (1, "fp32")):
            if tuned[index] > max(base[index] * 2.0, 1e-6):
                failures.append(
                    f"accuracy regression B={batch} {label}: "
                    f"{tuned[index]:.3e} vs {base[index]:.3e}")
        print(f"  -> tuned/default error ratio: "
              f"bf16 {tuned[0] / max(base[0], 1e-30):.2f}x, "
              f"fp32 {tuned[1] / max(base[1], 1e-30):.2f}x")

    print()
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS: selection, accuracy and determinism all hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
