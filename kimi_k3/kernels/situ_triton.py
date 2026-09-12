"""Stride-aware SiTU-and-mul (Triton).

Same math as trt-llm's ``tensorrt_llm/_torch/modules/situ.py`` (the
grafted ``SiTUAndMul``), with one difference: the input may be a
row-strided view (``gate_up.stride(1) == 1``), so the fuse3 front's
``merged[:, width:]`` slice feeds the activation directly instead of
paying trt-llm's ``gate_up.contiguous()`` copy ([B, 1536] bf16 on the
overlap stream, every fused decode step).

Bit-exact with the trt-llm kernel: identical per-element formula and
fp32 math (``beta * tanh(gate/beta) * sigmoid(gate) * up``), only the
gate/up addressing gains a runtime row stride.
Probe: local_debug/situ_strided_probe.py. Capture-safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["situ_and_mul"]

_SMALL_BLOCK_SIZE = 128
_LARGE_BLOCK_SIZE = 256
_LARGE_INPUT_NUMEL_THRESHOLD = 65536


@triton.jit
def _situ_and_mul_strided_kernel(
    gate_up_ptr,
    output_ptr,
    output_numel,
    row_stride,
    half_width: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_numel
    row_offsets = output_offsets // half_width
    column_offsets = output_offsets - row_offsets * half_width
    gate_offsets = row_offsets.to(tl.int64) * row_stride + column_offsets

    gate = tl.load(gate_up_ptr + gate_offsets, mask=output_mask).to(tl.float32)
    up = tl.load(gate_up_ptr + gate_offsets + half_width,
                 mask=output_mask).to(tl.float32)
    gate = beta * libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * libdevice.tanh(up / linear_beta)
    tl.store(output_ptr + output_offsets, gate * up, mask=output_mask)


def situ_and_mul(
    gate_up: torch.Tensor,
    beta: float,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """``situ(gate) * up`` over the last dim split in half. ``gate_up``
    is [rows, 2*half_width], row-strided allowed."""
    assert gate_up.is_cuda and gate_up.dim() == 2
    assert gate_up.stride(1) == 1
    half_width = gate_up.shape[-1] // 2
    output = torch.empty(
        (gate_up.shape[0], half_width),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    if output.numel() == 0:
        return output

    if output.numel() < _LARGE_INPUT_NUMEL_THRESHOLD:
        block_size = _SMALL_BLOCK_SIZE
        num_warps = 4
    else:
        block_size = _LARGE_BLOCK_SIZE
        num_warps = 8
    grid = (triton.cdiv(output.numel(), block_size),)
    _situ_and_mul_strided_kernel[grid](
        gate_up,
        output,
        output.numel(),
        gate_up.stride(0),
        half_width,
        beta,
        linear_beta if linear_beta is not None else 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
