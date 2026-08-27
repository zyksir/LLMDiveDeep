# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Standalone production wrapper for the direct CuTe DSL kernel."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from backends.cute_direct import CuTeDirectBackend
from common import DTYPES, Problem, Shape


class CausalConv1dPrefill:
    """Cached compiler and launcher for packed KDA causal-conv1d prefill."""

    def __init__(self) -> None:
        self._backend = CuTeDirectBackend()

    def prepare(
        self,
        projected: torch.Tensor,
        num_prefill_tokens: int,
        weight: torch.Tensor,
        *,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        sequence_lengths: Sequence[int] | None = None,
    ):
        """Allocate output, compile/cache, and return a one-launch invocation.

        Call this outside the timed region, then invoke ``prepared.run()`` for
        kernel-only execution. If ``sequence_lengths`` is omitted, the CPU
        mirror is derived from ``query_start_loc`` during this setup call.
        """
        if projected.ndim != 2 or projected.stride(1) != 1:
            raise ValueError("projected must be [T_total,D] with channel stride 1")
        if not projected.is_cuda or projected.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("projected must be CUDA FP16 or BF16")
        if not 0 <= num_prefill_tokens <= projected.shape[0]:
            raise ValueError("num_prefill_tokens is out of range")
        channels = projected.shape[1]
        if weight.ndim != 2 or weight.shape[0] != channels or weight.shape[1] not in (2, 3, 4):
            raise ValueError("weight must have shape [D,W] with W in {2,3,4}")
        if weight.dtype != projected.dtype or weight.stride(1) != 1:
            raise ValueError("weight must match projected dtype and have width stride 1")
        width = weight.shape[1]
        batch = query_start_loc.numel() - 1
        if batch < 1 or cache_indices.shape != (batch,) or has_initial_state.shape != (batch,):
            raise ValueError("metadata shapes must be [B+1], [B], [B]")
        if query_start_loc.dtype != torch.int32 or cache_indices.dtype != torch.int32:
            raise ValueError("query_start_loc and cache_indices must be int32")
        if has_initial_state.dtype != torch.bool:
            raise ValueError("has_initial_state must be bool")
        if conv_states.ndim != 3 or conv_states.shape[1:] != (channels, width - 1):
            raise ValueError("conv_states must have shape [slots,D,W-1]")
        if conv_states.dtype != projected.dtype:
            raise ValueError("conv_states must match projected dtype")
        if bias is not None and (bias.shape != (channels,) or bias.dtype != projected.dtype):
            raise ValueError("bias must have shape [D] and match projected dtype")
        if activation not in (None, "silu", "swish"):
            raise ValueError("activation must be None, 'silu', or 'swish'")

        if sequence_lengths is None:
            starts = query_start_loc.detach().cpu().tolist()
            sequence_lengths = [end - start for start, end in zip(starts, starts[1:])]
        sequence_lengths = tuple(int(length) for length in sequence_lengths)
        if len(sequence_lengths) != batch or sum(sequence_lengths) != num_prefill_tokens:
            raise ValueError("sequence_lengths must contain B lengths summing to T")

        dtype_name = next(name for name, dtype in DTYPES.items() if dtype == projected.dtype)
        shape = Shape(
            dtype=dtype_name,
            width=width,
            channels=channels,
            tokens=num_prefill_tokens,
            batch=batch,
            bias=bias is not None,
            activation=activation,
        )
        problem = Problem(
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
        supported, reason = self._backend.supports(problem)
        if not supported:
            raise ValueError(reason)
        return self._backend.prepare(problem)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Prepare, launch once, and return output.

        Prefer ``prepare`` when compilation/allocation must be excluded from
        timing.
        """
        return self.prepare(*args, **kwargs).run()


_DEFAULT_RUNNER = CausalConv1dPrefill()


def causal_conv1d_prefill(*args, **kwargs) -> torch.Tensor:
    """Convenience wrapper backed by a process-wide compile cache."""
    return _DEFAULT_RUNNER(*args, **kwargs)
