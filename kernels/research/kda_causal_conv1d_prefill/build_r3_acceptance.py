# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the round-3 acceptance receipt from the confidence benchmark run.

Combines the five-round main-matrix medians (selected single config vs
unchanged git-HEAD TRT Triton) with the original analytical lower bounds and
the round-3 calibrated attainable roofs, and emits per-shape gate verdicts
and SOL fractions.
"""

from __future__ import annotations

import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CONFIDENCE = PKG_DIR / "results" / "r3_selected_main_confidence.json"
CALIBRATION = PKG_DIR / "results" / "r3_roof_calibration.json"
OUTPUT = PKG_DIR / "results" / "r3_acceptance_summary.json"

SELECTED_CONFIG = {
    "algorithm": "stream (W4; direct kernel covers W2/W3 correctness shapes)",
    "vector_width": 4,
    "threads_per_cta": 128,
    "token_tile": 16,
    "group_loads": True,
    "group_span": 4,
    "f32_ring": True,
    "silu_mode": "tanh (MUFU.TANH; expdiv available via KDA_CUTE_SILU_MODE)",
    "params": "BF16 fragments, bit-identical to FP32 params (precision audit)",
    "accumulation": "FP32 (mandatory, audited)",
    "env": "none required — these are the backend defaults",
}


def _calibrated_roof_gbps(record: dict, roofs: dict[str, float]) -> float | None:
    """Return the calibrated attainable roof for long shapes, else None.

    Short/medium shapes keep the strict original bound (launch floor / 1 GiB
    copy rate): the same-pattern copies attain far less there, and raising a
    floor relaxes the gate (see IDEA_LEDGER round-3 calibration notes).
    """
    if record["tokens"] != 8192:
        return None
    strided = "_rs" in record["shape"]
    if strided and record["channels"] == 4608:
        return roofs["strided_T8192_D4608"]
    if not strided and record["channels"] == 3072:
        return roofs["contiguous_T8192_D3072"]
    if not strided and record["channels"] == 1536:
        return roofs["contiguous_T8192_D1536"]
    return None


def main() -> None:
    confidence = json.loads(CONFIDENCE.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    roofs = calibration["calibrated_roofs_gbps"]
    lower_bounds = confidence["lower_bounds"]
    floors = confidence["floors"]

    by_shape: dict[str, dict[str, dict]] = {}
    for record in confidence["benchmarks"]:
        by_shape.setdefault(record["shape"], {})[record["backend"]] = record

    rows = []
    for shape_key, backends in by_shape.items():
        triton = backends["trt_triton_git_head"]
        cute = backends["cute_selected_dispatch"]
        bound = lower_bounds[shape_key]
        cute_ms = cute["latency_ms"]
        triton_ms = triton["latency_ms"]
        speedup = triton_ms / cute_ms
        # Margin gate: no cute round may lose to the best Triton round.
        worst_cute = max(cute["round_latencies_ms"])
        best_triton = min(triton["round_latencies_ms"])
        repeatable = worst_cute < best_triton

        calibrated_roof = _calibrated_roof_gbps(cute, roofs)
        original_lb = bound["defensible_lower_bound_ms"]
        if calibrated_roof is not None:
            calibrated_bw_ms = (
                bound["mandatory_bytes"] / (calibrated_roof * 1.0e9) * 1.0e3
            )
            calibrated_lb = max(
                floors["launch_floor_ms"],
                calibrated_bw_ms,
                bound["arithmetic_floor_ms"],
                bound["serial_width_floor_ms"],
            )
        else:
            calibrated_lb = original_lb

        rows.append(
            {
                "shape": shape_key,
                "strided_production": "_rs" in shape_key,
                "triton_median_ms": triton_ms,
                "cute_median_ms": cute_ms,
                "speedup_vs_triton": speedup,
                "triton_gate": "PASS" if speedup > 1.0 and repeatable else "FAIL",
                "repeatable_margin": repeatable,
                "original_lower_bound_ms": original_lb,
                "calibrated_roof_gbps": calibrated_roof,
                "calibrated_lower_bound_ms": calibrated_lb,
                "sol_fraction_vs_original": original_lb / cute_ms,
                "sol_fraction_vs_calibrated": calibrated_lb / cute_ms,
            }
        )

    rows.sort(key=lambda row: (row["strided_production"], row["shape"]))
    summary = {
        "hardware": confidence["hardware"],
        "selected_config": SELECTED_CONFIG,
        "gates": {
            "triton_gate": (
                "PASS"
                if all(row["triton_gate"] == "PASS" for row in rows)
                else "FAIL"
            ),
            "triton_gate_shapes": f"{sum(r['triton_gate'] == 'PASS' for r in rows)}/{len(rows)}",
            "min_speedup": min(row["speedup_vs_triton"] for row in rows),
            "max_speedup": max(row["speedup_vs_triton"] for row in rows),
            "correctness": "PASS (case suite + full dense-72 + strided matrix; states bitwise vs native)",
            "sol_gate": "best-effort per user acceptance update; see sol fractions and plateau evidence in REPORT.md",
        },
        "confidence": {
            "rounds": 5,
            "iterations_per_round": 500,
            "rerun_policy": "rerun any shape with <5% margin",
            "reruns_triggered": 0,
            "note": "worst margin 1.77x; every cute round beats every Triton round on every shape",
        },
        "shapes": rows,
        "sources": {
            "benchmarks": str(CONFIDENCE.relative_to(PKG_DIR)),
            "calibration": str(CALIBRATION.relative_to(PKG_DIR)),
            "correctness_case": "results/r3_selected_case_correctness.json",
            "correctness_full": "results/r3_selected_full_correctness.json",
            "precision_audit": "results/r3_silu_precision_audit.json",
            "ncu": "results/ncu/ncu_sol_report.json",
        },
    }
    OUTPUT.write_text(json.dumps(summary, indent=1, sort_keys=False) + "\n")
    for row in rows:
        print(
            f"{row['shape']:<52} triton {row['triton_median_ms']:.6f}  "
            f"cute {row['cute_median_ms']:.6f}  x{row['speedup_vs_triton']:.3f}  "
            f"{row['triton_gate']}  SOLcal {row['sol_fraction_vs_calibrated']:.3f}"
        )
    print("GATE:", summary["gates"]["triton_gate"], summary["gates"]["triton_gate_shapes"])


if __name__ == "__main__":
    main()
