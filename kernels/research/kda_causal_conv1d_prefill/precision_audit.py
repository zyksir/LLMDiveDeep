#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Round-3 precision audit: SiLU formulation and parameter-cache numerics.

Enforces the user precision directive:
  1. FP32 accumulation is mandatory (code-audited; this script additionally
     proves the BF16 parameter-fragment cache is bit-identical to the
     FP32-parameter path, because source weights/bias are BF16 and the
     BF16->FP32 conversion at the multiply is exact).
  2. The tanh SiLU reformulation is validated against the unchanged TRT
     native kernel on every W4 correctness case (normal, unscaled,
     adversarial extremes, short, padded, strided production, fp16), with
     measured max abs/rel error reported next to the expdiv form.
  3. Both formulations' error numbers are emitted so the selection is
     documented, never silent.

W2/W3 shapes dispatch to the direct kernel, which only implements expdiv;
the SiLU-mode choice therefore only exists for the W4 streaming kernel
audited here.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from common import (  # noqa: E402
    ATOL,
    RTOL,
    Shape,
    compare,
    correctness_cases,
    hardware_identity,
    make_problem,
    production_shape,
    write_json,
)

BASE_ENV = {
    "KDA_CUTE_ALGORITHM": "stream",
    "KDA_CUTE_TOKEN_TILE": "12",
    "KDA_CUTE_VECTOR_WIDTH": "4",
    "KDA_CUTE_THREADS": "128",
    "KDA_CUTE_PREFETCH": "1",
    "KDA_CUTE_L2_PREFETCH_BLOCKS": "0",
}

# (variant tag, env overrides). Config-invariance variants prove that tile /
# thread / prefetch choices do not change numerics (same FP32 accumulation
# order per output in every variant).
VARIANTS = {
    "expdiv_bf16params": {"KDA_CUTE_SILU_MODE": "expdiv", "KDA_CUTE_FP32_PARAMS": "0"},
    "expdiv_fp32params": {"KDA_CUTE_SILU_MODE": "expdiv", "KDA_CUTE_FP32_PARAMS": "1"},
    "tanh_bf16params": {"KDA_CUTE_SILU_MODE": "tanh", "KDA_CUTE_FP32_PARAMS": "0"},
    "tanh_fp32params": {"KDA_CUTE_SILU_MODE": "tanh", "KDA_CUTE_FP32_PARAMS": "1"},
}
CONFIG_INVARIANCE_VARIANTS = {
    "tanh_tile20": {
        "KDA_CUTE_SILU_MODE": "tanh",
        "KDA_CUTE_FP32_PARAMS": "0",
        "KDA_CUTE_TOKEN_TILE": "20",
    },
    "tanh_t64": {
        "KDA_CUTE_SILU_MODE": "tanh",
        "KDA_CUTE_FP32_PARAMS": "0",
        "KDA_CUTE_THREADS": "64",
    },
    "tanh_nopf": {
        "KDA_CUTE_SILU_MODE": "tanh",
        "KDA_CUTE_FP32_PARAMS": "0",
        "KDA_CUTE_PREFETCH": "0",
    },
}
CONFIG_INVARIANCE_CASES = (
    "bf16_w4_normal",
    "fp16_w4_adversarial",
    "bf16_w4_strided_prod_adversarial",
)


def _make_backend(overrides: dict[str, str]):
    for key, value in {**BASE_ENV, **overrides}.items():
        os.environ[key] = value
    module = importlib.import_module("backends.cute_direct")
    return module.Backend()


def _run(backend, prototype):
    problem = prototype.clone()
    supported, reason = backend.supports(problem)
    if not supported:
        return None, None, reason
    prepared = backend.prepare(problem)
    output = prepared.run()
    torch.cuda.synchronize()
    return output, prepared.state(), ""


def _delta(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    af, bf = a.float(), b.float()
    absolute = (af - bf).abs()
    relative = absolute / bf.abs().clamp_min(1.0e-6)
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float(absolute.max().item()),
        "max_rel": float(relative.max().item()),
    }


def main() -> None:
    native = importlib.import_module("backends.trt_native").Backend()
    backends = {tag: _make_backend(env) for tag, env in VARIANTS.items()}
    invariance_backends = {
        tag: _make_backend(env) for tag, env in CONFIG_INVARIANCE_VARIANTS.items()
    }

    cases = [
        (name, prototype)
        for name, prototype in correctness_cases()
        if prototype.shape.width == 4
    ]
    # Long shapes: exercise the exact gate geometry (contiguous and strided).
    cases.append(
        (
            "bf16_w4_long_dense",
            make_problem(Shape("bf16", 4, 3072, 8192, 8), seed=51),
        )
    )
    cases.append(
        ("bf16_w4_long_strided_prod", make_problem(production_shape(8192, 8), seed=52))
    )

    records: list[dict[str, object]] = []
    for case_name, prototype in cases:
        reference_problem = prototype.clone()
        reference_prepared = native.prepare(reference_problem)
        reference_output = reference_prepared.run()
        torch.cuda.synchronize()
        reference_state = reference_prepared.state()

        outputs: dict[str, torch.Tensor] = {}
        states: dict[str, torch.Tensor] = {}
        record: dict[str, object] = {
            "case": case_name,
            "shape": prototype.shape.key,
            "tolerances": {
                "rtol": RTOL,
                "atol": ATOL[reference_output.dtype],
            },
            "vs_native": {},
        }
        for tag, backend in backends.items():
            output, state, reason = _run(backend, prototype)
            if output is None:
                record["vs_native"][tag] = {"skipped": reason}
                continue
            outputs[tag] = output
            states[tag] = state
            record["vs_native"][tag] = compare(
                output, state, reference_output, reference_state
            )

        if "tanh_bf16params" in outputs and "expdiv_bf16params" in outputs:
            record["tanh_vs_expdiv_output"] = _delta(
                outputs["tanh_bf16params"], outputs["expdiv_bf16params"]
            )
        for mode in ("expdiv", "tanh"):
            a, b = f"{mode}_bf16params", f"{mode}_fp32params"
            if a in outputs and b in outputs:
                record[f"{mode}_bf16params_vs_fp32params"] = {
                    "output_bitwise_equal": bool(
                        torch.equal(outputs[a], outputs[b])
                    ),
                    "state_bitwise_equal": bool(torch.equal(states[a], states[b])),
                }

        if case_name in CONFIG_INVARIANCE_CASES and "tanh_bf16params" in outputs:
            invariance: dict[str, object] = {}
            for tag, backend in invariance_backends.items():
                output, state, reason = _run(backend, prototype)
                if output is None:
                    invariance[tag] = {"skipped": reason}
                    continue
                invariance[tag] = {
                    "output_bitwise_equal_vs_tile12": bool(
                        torch.equal(output, outputs["tanh_bf16params"])
                    ),
                    "state_bitwise_equal_vs_tile12": bool(
                        torch.equal(state, states["tanh_bf16params"])
                    ),
                }
            record["config_invariance"] = invariance

        records.append(record)
        print("AUDIT " + json.dumps(record, sort_keys=True), flush=True)

    def worst(tag: str, field: str) -> float:
        values = [
            record["vs_native"][tag][field]
            for record in records
            if isinstance(record["vs_native"].get(tag), dict)
            and field in record["vs_native"][tag]
        ]
        return max(values) if values else float("nan")

    summary = {
        "worst_max_abs_vs_native": {
            tag: worst(tag, "max_abs") for tag in VARIANTS
        },
        "worst_max_rel_vs_native": {
            tag: worst(tag, "max_rel") for tag in VARIANTS
        },
        "all_within_tolerance": {
            tag: all(
                record["vs_native"][tag].get("correct", True)
                for record in records
                if isinstance(record["vs_native"].get(tag), dict)
                and "correct" in record["vs_native"][tag]
            )
            for tag in VARIANTS
        },
        "all_states_bitwise_exact": {
            tag: all(
                record["vs_native"][tag].get("state_correct", True)
                for record in records
                if isinstance(record["vs_native"].get(tag), dict)
                and "state_correct" in record["vs_native"][tag]
            )
            for tag in VARIANTS
        },
        "fp32params_bitwise_identity": all(
            record.get(f"{mode}_bf16params_vs_fp32params", {}).get(
                "output_bitwise_equal", True
            )
            and record.get(f"{mode}_bf16params_vs_fp32params", {}).get(
                "state_bitwise_equal", True
            )
            for record in records
            for mode in ("expdiv", "tanh")
        ),
        "worst_tanh_vs_expdiv_max_abs": max(
            (
                record["tanh_vs_expdiv_output"]["max_abs"]
                for record in records
                if "tanh_vs_expdiv_output" in record
            ),
            default=float("nan"),
        ),
        "worst_tanh_vs_expdiv_max_rel": max(
            (
                record["tanh_vs_expdiv_output"]["max_rel"]
                for record in records
                if "tanh_vs_expdiv_output" in record
            ),
            default=float("nan"),
        ),
        "config_invariance_all_bitwise": all(
            entry.get("output_bitwise_equal_vs_tile12", True)
            and entry.get("state_bitwise_equal_vs_tile12", True)
            for record in records
            for entry in record.get("config_invariance", {}).values()
            if isinstance(entry, dict) and "skipped" not in entry
        ),
    }
    payload = {
        "hardware": hardware_identity(),
        "base_config": BASE_ENV,
        "records": records,
        "summary": summary,
    }
    write_json(PACKAGE_DIR / "results" / "r3_silu_precision_audit.json", payload)
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
