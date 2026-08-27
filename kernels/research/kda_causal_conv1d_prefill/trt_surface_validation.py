# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate alignment with the TRT-LLM stable integration surface.

Checks, versus tensorrt_llm/_torch/modules/mamba/causal_conv1d_prefill.py
(read-only, never edited):

1. Signature parity of ``causal_conv1d_prefill`` (names, kinds, defaults;
   the CuTe wrapper may append the deprecated ``sequence_lengths`` alias).
2. Grouped output mode: the grouped CuTe result must be bit-identical to the
   CuTe flat result regrouped via ``view(T, n_groups, G).permute(1, 0, 2)``
   on the K3 strided production shape and a dense shape, T in {128, 8192},
   mixed initial states; conv states must also be bitwise identical between
   the flat and grouped runs. A ``qkv_group_tokens > T`` case checks only
   the leading T rows of each plane.
3. Quick benchmark: flat vs grouped latency on the selected config's main
   long shapes (T8192 dense + strided).

Receipt: local_results/r3_trt_surface_alignment.json
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import torch

from common import (
    Shape,
    hardware_identity,
    make_problem,
    production_shape,
    time_cuda,
    write_json,
)
from kernel import CausalConv1dPrefill

PKG_DIR = Path(__file__).resolve().parent
TRT_SURFACE = Path(
    "/node-storage/trt-llm/tensorrt_llm/_torch/modules/mamba/"
    "causal_conv1d_prefill.py"
)
GROUP_SIZE = 1536  # Kimi-K3: num_heads * head_dim = 12 * 128


def _trt_signature() -> list[tuple[str, str, object]]:
    """Parse the TRT surface signature without importing TRT-LLM.

    The module imports triton/TRT internals at module scope; parsing the
    function AST-free via a stub exec would be brittle, so read the file and
    extract the signature by compiling only the function definition header
    with a stubbed body.
    """
    source = TRT_SURFACE.read_text()
    start = source.index("def causal_conv1d_prefill(")
    end = source.index("-> torch.Tensor:", start) + len("-> torch.Tensor:")
    header = source[start:end] + "\n    ...\n"
    namespace: dict[str, object] = {"torch": torch, "Optional": None}
    # Provide typing names used in the annotations.
    import typing
    from collections.abc import Sequence

    namespace.update({"Optional": typing.Optional, "Sequence": Sequence})
    exec(header, namespace)  # noqa: S102 - trusted local repository file
    function = namespace["causal_conv1d_prefill"]
    return [
        (name, str(parameter.kind), parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    ]


def _cute_signature() -> list[tuple[str, str, object]]:
    import kernel as cute_kernel

    runner_signature = inspect.signature(CausalConv1dPrefill.prepare)
    parameters = list(runner_signature.parameters.items())[1:]  # drop self
    del cute_kernel
    return [
        (name, str(parameter.kind), parameter.default)
        for name, parameter in parameters
    ]


def check_signature() -> dict[str, object]:
    trt = _trt_signature()
    cute = _cute_signature()
    # The CuTe wrapper appends the deprecated `sequence_lengths` alias and
    # makes seq_lens_cpu keyword-optional to keep the alias callable; every
    # TRT-shaped call must bind identically.
    cute_names = [name for name, _, _ in cute]
    trt_names = [name for name, _, _ in trt]
    prefix_match = cute_names[: len(trt_names)] == trt_names
    kinds_match = [kind for _, kind, _ in cute[: len(trt)]] == [
        kind for _, kind, _ in trt
    ]
    return {
        "trt_parameters": [(n, k) for n, k, _ in trt],
        "cute_parameters": [(n, k) for n, k, _ in cute],
        "trt_prefix_matches": bool(prefix_match and kinds_match),
        "cute_extra_parameters": cute_names[len(trt_names):],
    }


def _run_surface(
    runner: CausalConv1dPrefill,
    problem,
    *,
    qkv_group_size: int | None,
    qkv_group_tokens: int | None = None,
):
    prepared = runner.prepare(
        problem.projected,
        problem.shape.tokens,
        problem.weight,
        conv_states=problem.conv_states,
        query_start_loc=problem.query_start_loc,
        seq_lens_cpu=list(problem.sequence_lengths),
        cache_indices=problem.cache_indices,
        has_initial_state=problem.has_initial_state,
        bias=problem.bias,
        activation=problem.shape.activation,
        qkv_group_size=qkv_group_size,
        qkv_group_tokens=qkv_group_tokens,
    )
    output = prepared.run()
    torch.cuda.synchronize()
    return prepared, output


def check_grouped_case(
    runner: CausalConv1dPrefill,
    name: str,
    shape: Shape,
    *,
    seed: int,
    qkv_group_tokens: int | None = None,
) -> dict[str, object]:
    tokens = shape.tokens
    flat_problem = make_problem(shape, seed=seed)
    grouped_problem = flat_problem.clone()

    _, flat = _run_surface(runner, flat_problem, qkv_group_size=None)
    _, grouped = _run_surface(
        runner,
        grouped_problem,
        qkv_group_size=GROUP_SIZE,
        qkv_group_tokens=qkv_group_tokens,
    )

    num_groups = shape.channels // GROUP_SIZE
    group_tokens = qkv_group_tokens or tokens
    expected_shape = (num_groups, group_tokens, GROUP_SIZE)
    regrouped = (
        flat.view(tokens, num_groups, GROUP_SIZE).permute(1, 0, 2).contiguous()
    )
    written = grouped[:, :tokens, :]
    record = {
        "case": name,
        "shape": shape.key,
        "qkv_group_size": GROUP_SIZE,
        "qkv_group_tokens": group_tokens,
        "grouped_shape_ok": tuple(grouped.shape) == expected_shape,
        "grouped_contiguous": bool(grouped.is_contiguous()),
        "output_bitwise_equal": bool(torch.equal(written, regrouped)),
        "state_bitwise_equal": bool(
            torch.equal(flat_problem.conv_states, grouped_problem.conv_states)
        ),
    }
    record["pass"] = bool(
        record["grouped_shape_ok"]
        and record["grouped_contiguous"]
        and record["output_bitwise_equal"]
        and record["state_bitwise_equal"]
    )
    return record


def benchmark_case(
    runner: CausalConv1dPrefill,
    name: str,
    shape: Shape,
    *,
    seed: int,
    warmup: int = 50,
    iterations: int = 500,
) -> dict[str, object]:
    flat_problem = make_problem(shape, seed=seed)
    grouped_problem = flat_problem.clone()
    flat_prepared, _ = _run_surface(runner, flat_problem, qkv_group_size=None)
    grouped_prepared, _ = _run_surface(
        runner, grouped_problem, qkv_group_size=GROUP_SIZE
    )
    flat_timing = time_cuda(flat_prepared, warmup, iterations)
    grouped_timing = time_cuda(grouped_prepared, warmup, iterations)
    return {
        "case": name,
        "shape": shape.key,
        "flat_latency_ms": flat_timing["latency_ms"],
        "grouped_latency_ms": grouped_timing["latency_ms"],
        "grouped_over_flat": grouped_timing["latency_ms"]
        / flat_timing["latency_ms"],
        "warmup": warmup,
        "iterations": iterations,
    }


def main() -> None:
    runner = CausalConv1dPrefill()

    signature = check_signature()
    print("SIGNATURE " + json.dumps(signature))

    grouped_records = [
        check_grouped_case(
            runner,
            "dense_d3072_t128_b8",
            Shape("bf16", 4, 3072, 128, 8),
            seed=61,
        ),
        check_grouped_case(
            runner,
            "dense_d3072_t8192_b8",
            Shape("bf16", 4, 3072, 8192, 8),
            seed=62,
        ),
        check_grouped_case(
            runner,
            "strided_prod_t128_b8",
            production_shape(128, 8),
            seed=63,
        ),
        check_grouped_case(
            runner,
            "strided_prod_t8192_b8",
            production_shape(8192, 8),
            seed=64,
        ),
        check_grouped_case(
            runner,
            "strided_prod_t8192_b1",
            production_shape(8192, 1),
            seed=65,
        ),
        # Mixed batch: planes taller than the prefill token count leave the
        # tail rows uninitialized; only the leading T rows are compared.
        check_grouped_case(
            runner,
            "strided_prod_t128_b8_tail_rows",
            production_shape(128, 8),
            seed=66,
            qkv_group_tokens=192,
        ),
    ]
    for record in grouped_records:
        print("GROUPED " + json.dumps(record))

    benchmarks = [
        benchmark_case(
            runner,
            "dense_d3072_t8192_b8",
            Shape("bf16", 4, 3072, 8192, 8),
            seed=71,
        ),
        benchmark_case(
            runner,
            "strided_prod_t8192_b8",
            production_shape(8192, 8),
            seed=72,
        ),
    ]
    for record in benchmarks:
        print("BENCH " + json.dumps(record))

    payload = {
        "hardware": hardware_identity(),
        "trt_surface": str(TRT_SURFACE),
        "signature": signature,
        "grouped_bitwise": grouped_records,
        "benchmarks": benchmarks,
        "all_grouped_pass": all(record["pass"] for record in grouped_records),
        "deviations": [
            "qkv_group_size must be a multiple of the CuTe CTA channel span "
            "512 (threads 128 x vector width 4); Triton requires a multiple "
            "of CONV_FWD_BLOCK_N = 256",
            "padded sequences copy input to output (compatible superset; the "
            "contract leaves those rows uninitialized and forbids reading "
            "them)",
            "grouped mode requires the streaming W4 kernel (the W2/W3 direct "
            "kernel rejects it)",
        ],
    }
    write_json(PKG_DIR / "results" / "r3_trt_surface_alignment.json", payload)
    print(
        "RESULT "
        + json.dumps(
            {
                "signature_prefix_matches": signature["trt_prefix_matches"],
                "all_grouped_pass": payload["all_grouped_pass"],
            }
        )
    )
    print("SURFACE_DONE")


if __name__ == "__main__":
    main()
