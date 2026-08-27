# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Shared input generation, validation, timing, and result serialization."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Callable, Protocol

import torch

PAD_SLOT_ID = -1
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
ATOL = {
    torch.float16: 1.0e-2,
    torch.bfloat16: 1.0e-1,
}
RTOL = 1.0e-2


@dataclasses.dataclass(frozen=True)
class Shape:
    dtype: str
    width: int
    channels: int
    tokens: int
    batch: int
    bias: bool = True
    activation: str | None = "silu"
    pattern: str = "normal"
    # Physical row stride of the projected input in elements. None means
    # contiguous rows (stride == channels). The Kimi-K3 production case is a
    # [T, 4752] in_proj buffer consumed as projected[:, :4608].
    row_stride: int | None = None

    @property
    def key(self) -> str:
        activation = self.activation or "none"
        stride_tag = "" if self.row_stride is None else f"_rs{self.row_stride}"
        return (
            f"{self.dtype}_w{self.width}_d{self.channels}_t{self.tokens}"
            f"_b{self.batch}_bias{int(self.bias)}_{activation}_{self.pattern}"
            f"{stride_tag}"
        )


@dataclasses.dataclass
class Problem:
    shape: Shape
    projected: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    conv_states: torch.Tensor
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    sequence_lengths: tuple[int, ...]
    # TRT integration-surface grouped output mode: when qkv_group_size is set
    # the output is [channels // G, group_tokens, G] contiguous (token-major
    # within each channel-group plane, groups outermost) with only the leading
    # num_prefill_tokens rows of each plane written.
    qkv_group_size: int | None = None
    qkv_group_tokens: int | None = None

    def clone(self) -> "Problem":
        return Problem(
            shape=self.shape,
            projected=clone_preserving_view(self.projected),
            weight=self.weight.clone(),
            bias=None if self.bias is None else self.bias.clone(),
            conv_states=self.conv_states.clone(),
            query_start_loc=self.query_start_loc.clone(),
            cache_indices=self.cache_indices.clone(),
            has_initial_state=self.has_initial_state.clone(),
            sequence_lengths=self.sequence_lengths,
            qkv_group_size=self.qkv_group_size,
            qkv_group_tokens=self.qkv_group_tokens,
        )


@dataclasses.dataclass
class Prepared:
    run: Callable[[], torch.Tensor]
    output: Callable[[], torch.Tensor]
    state: Callable[[], torch.Tensor]
    launch_count: int | None


class Backend(Protocol):
    name: str
    source: str

    def supports(self, problem: Problem) -> tuple[bool, str]: ...

    def prepare(self, problem: Problem) -> Prepared: ...


def clone_preserving_view(tensor: torch.Tensor) -> torch.Tensor:
    """Clone a tensor, preserving strided-view geometry.

    ``Tensor.clone()`` on a non-dense view (for example ``buffer[:, :4608]`` of
    a ``[T, 4752]`` buffer) silently materializes a contiguous copy, which
    would drop the production row stride. Clone the underlying storage instead
    and rebuild the identical view.
    """
    base = tensor._base
    if base is None:
        return tensor.clone()
    return base.clone().as_strided(
        tensor.size(),
        tensor.stride(),
        tensor.storage_offset(),
    )


def _uneven_lengths(total: int, batch: int) -> tuple[int, ...]:
    if batch == 1:
        return (total,)
    if batch != 8:
        raise ValueError(f"only batch 1 and 8 are contracted, got {batch}")
    # Positive, deliberately uneven, includes a short sequence for T=128.
    base = [1, 3, 7, 11, 17, 23, 29]
    scale = max(1, total // (sum(base) * 3))
    first = [value * scale for value in base]
    remainder = total - sum(first)
    while remainder <= 0:
        scale -= 1
        first = [value * scale for value in base]
        remainder = total - sum(first)
    return tuple([*first, remainder])


def _pattern_tensors(
    shape: Shape,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = DTYPES[shape.dtype]
    projected_shape = (shape.tokens + 5, shape.channels)
    weight_shape = (shape.channels, shape.width)
    bias_shape = (shape.channels,)
    if shape.pattern == "normal":
        projected = torch.randn(projected_shape, generator=generator, device=device, dtype=dtype) * 0.5
        weight = torch.randn(weight_shape, generator=generator, device=device, dtype=dtype) * 0.2
        bias = torch.randn(bias_shape, generator=generator, device=device, dtype=dtype) * 0.1
    elif shape.pattern == "unscaled":
        projected = torch.randn(projected_shape, generator=generator, device=device, dtype=dtype)
        weight = torch.randn(weight_shape, generator=generator, device=device, dtype=dtype)
        bias = torch.randn(bias_shape, generator=generator, device=device, dtype=dtype)
    elif shape.pattern == "adversarial":
        token = torch.arange(shape.tokens + 5, device=device, dtype=torch.float32)[:, None]
        channel = torch.arange(shape.channels, device=device, dtype=torch.float32)[None, :]
        projected = (((token + channel) % 2) * 2 - 1).mul(7.0).to(dtype)
        tap = torch.arange(shape.width, device=device, dtype=torch.float32)[None, :]
        weight = (((channel.T + tap) % 2) * 2 - 1).mul(0.75).to(dtype)
        bias = torch.linspace(-3.0, 3.0, shape.channels, device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown input pattern: {shape.pattern}")
    projected = projected.contiguous()
    if shape.row_stride is not None:
        if shape.row_stride < shape.channels:
            raise ValueError("row_stride must be at least channels")
        # Mirror production: the in_proj GEMM writes a [T, row_stride] buffer
        # and the convolution consumes projected[:, :channels]. The padding
        # columns are filled with large sentinel values so any kernel that
        # wrongly assumes contiguous rows produces detectably wrong output.
        buffer = torch.full(
            (shape.tokens + 5, shape.row_stride),
            777.0,
            device=device,
            dtype=dtype,
        )
        buffer[:, : shape.channels] = projected
        projected = buffer[:, : shape.channels]
    return projected, weight.contiguous(), bias.contiguous()


def make_problem(
    shape: Shape,
    *,
    seed: int = 1234,
    padded: bool = False,
    short: bool = False,
    mixed_initial_state: bool = True,
    permuted_slots: bool = True,
    device: str | torch.device = "cuda",
) -> Problem:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    projected, weight, generated_bias = _pattern_tensors(shape, generator, device)
    bias = generated_bias if shape.bias else None

    if short:
        if shape.tokens < shape.batch:
            raise ValueError("short case requires at least one token per sequence")
        lengths = [1] * (shape.batch - 1)
        lengths.append(shape.tokens - sum(lengths))
        sequence_lengths = tuple(lengths)
    else:
        sequence_lengths = _uneven_lengths(shape.tokens, shape.batch)

    query_start_loc = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_slots = max(shape.batch + 3, 11)
    if permuted_slots:
        fixed = [5, 2, 9, 0, 7, 1, 8, 3]
        slots = fixed[: shape.batch]
    else:
        slots = list(range(shape.batch))
    if padded:
        slots[max(0, shape.batch // 2)] = PAD_SLOT_ID
    cache_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    has_initial_state = torch.tensor(
        [mixed_initial_state and index % 2 == 0 for index in range(shape.batch)],
        dtype=torch.bool,
        device=device,
    )
    conv_states = (
        torch.randn(
            (num_slots, shape.channels, shape.width - 1),
            generator=generator,
            device=device,
            dtype=DTYPES[shape.dtype],
        )
        * 0.5
    ).contiguous()
    return Problem(
        shape=shape,
        projected=projected,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        sequence_lengths=sequence_lengths,
    )


# Kimi-K3 KDA production geometry: packed QKV (3 x 1536, num_heads=12,
# head_dim=128) consumed as a row-strided view of the [T, 4752] in_proj output.
PRODUCTION_CHANNELS = 4608
PRODUCTION_ROW_STRIDE = 4752


def production_shape(tokens: int, batch: int, *, bias: bool = False, width: int = 4,
                     dtype: str = "bf16", pattern: str = "normal") -> Shape:
    return Shape(
        dtype,
        width,
        PRODUCTION_CHANNELS,
        tokens,
        batch,
        bias=bias,
        activation="silu",
        pattern=pattern,
        row_stride=PRODUCTION_ROW_STRIDE,
    )


def full_matrix() -> list[Shape]:
    dense = [
        Shape(dtype, width, channels, tokens, batch)
        for dtype in ("fp16", "bf16")
        for width in (2, 3, 4)
        for channels in (1536, 3072)
        for tokens in (128, 1024, 8192)
        for batch in (1, 8)
    ]
    strided = [
        production_shape(tokens, batch, width=width, dtype=dtype)
        for dtype in ("fp16", "bf16")
        for width in (2, 3, 4)
        for tokens in (128, 1024, 8192)
        for batch in (1, 8)
    ]
    return dense + strided


def main_performance_matrix() -> list[Shape]:
    dense = [
        Shape("bf16", 4, channels, tokens, batch)
        for channels in (1536, 3072)
        for tokens in (128, 1024, 8192)
        for batch in (1, 8)
    ]
    strided = [
        production_shape(tokens, batch)
        for tokens in (128, 1024, 8192)
        for batch in (1, 8)
    ]
    return dense + strided


def production_performance_matrix() -> list[Shape]:
    return [
        production_shape(tokens, batch)
        for tokens in (128, 1024, 8192)
        for batch in (1, 8)
    ]


def correctness_cases() -> list[tuple[str, Problem]]:
    cases: list[tuple[str, Problem]] = []
    for dtype in ("fp16", "bf16"):
        for width in (2, 3, 4):
            base = Shape(dtype, width, 1536, 128, 8, pattern="normal")
            cases.append((f"{dtype}_w{width}_normal", make_problem(base, seed=10 + width)))
    cases.extend(
        [
            (
                "bf16_w4_unscaled",
                make_problem(Shape("bf16", 4, 1536, 128, 8, pattern="unscaled"), seed=31),
            ),
            (
                "fp16_w4_adversarial",
                make_problem(Shape("fp16", 4, 1536, 128, 8, pattern="adversarial"), seed=32),
            ),
            (
                "bf16_w4_short",
                make_problem(Shape("bf16", 4, 1536, 16, 8), seed=33, short=True),
            ),
            (
                "bf16_w4_padded",
                make_problem(Shape("bf16", 4, 1536, 128, 8), seed=34, padded=True),
            ),
            (
                "bf16_w4_no_bias_no_activation",
                make_problem(
                    Shape("bf16", 4, 1536, 128, 8, bias=False, activation=None),
                    seed=35,
                ),
            ),
            # Kimi-K3 production strided-view coverage (D=4608, stride 4752).
            (
                "bf16_w4_strided_prod_normal",
                make_problem(production_shape(128, 8), seed=41),
            ),
            (
                "bf16_w4_strided_prod_bias",
                make_problem(production_shape(128, 8, bias=True), seed=42),
            ),
            (
                "bf16_w4_strided_prod_adversarial",
                make_problem(production_shape(128, 8, pattern="adversarial"), seed=43),
            ),
            (
                "bf16_w4_strided_prod_short",
                make_problem(production_shape(16, 8), seed=44, short=True),
            ),
            (
                "bf16_w4_strided_prod_padded",
                make_problem(production_shape(128, 8), seed=45, padded=True),
            ),
            (
                "fp16_w4_strided_prod_normal",
                make_problem(
                    production_shape(128, 8, dtype="fp16"),
                    seed=46,
                ),
            ),
        ]
    )
    return cases


def compare(
    output: torch.Tensor,
    state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
) -> dict[str, float | bool]:
    output_f = output.float()
    reference_f = reference_output.float()
    absolute = (output_f - reference_f).abs()
    relative = absolute / reference_f.abs().clamp_min(1.0e-6)
    output_ok = torch.allclose(
        output,
        reference_output,
        rtol=RTOL,
        atol=ATOL[output.dtype],
    )
    state_ok = torch.equal(state, reference_state)
    return {
        "correct": bool(output_ok and state_ok),
        "output_correct": bool(output_ok),
        "state_correct": bool(state_ok),
        "max_abs": float(absolute.max().item()),
        "max_rel": float(relative.max().item()),
    }


def time_cuda(prepared: Prepared, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        prepared.run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        prepared.run()
    end.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    latency_ms = start.elapsed_time(end) / iterations
    return {
        "latency_ms": float(latency_ms),
        "wall_seconds": float(wall_seconds),
        "warmup": warmup,
        "iterations": iterations,
    }


def quantiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        raise ValueError("samples must be non-empty")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        alpha = position - lower
        return ordered[lower] * (1 - alpha) + ordered[upper] * alpha

    return {
        "min_ms": ordered[0],
        "median_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "max_ms": ordered[-1],
    }


def hardware_identity() -> dict[str, object]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
