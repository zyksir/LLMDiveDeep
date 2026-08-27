# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Standalone production wrapper for the direct CuTe DSL kernel.

Call-compatible with the TRT-LLM stable integration surface
``tensorrt_llm/_torch/modules/mamba/causal_conv1d_prefill.py`` (the module
written so the underlying kernel can be swapped "Triton today, CuTeDSL
later"): same argument names/order, same validation errors, same grouped
``qkv_group_size`` / ``qkv_group_tokens`` output layout. A caller can swap
the Triton ``causal_conv1d_prefill`` for this one without edits.

Contract deviations (documented, see README):

- ``qkv_group_size`` must be a multiple of the CuTe CTA channel span
  (threads x vector width = 128 x 4 = 512 by default), stricter than the
  Triton launch channel block of 256. The Kimi-K3 production group
  ``G = num_heads * head_dim = 1536 = 3 x 512`` satisfies both.
- Padded sequences (``cache_indices == PAD_SLOT_ID``) copy input to output
  instead of leaving the rows uninitialized — a compatible superset of the
  contract, which only promises callers must not read those rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from backends.cute_direct import CuTeDirectBackend
from common import DTYPES, Problem, Shape

# CuTe grouped-store granularity: each CTA owns threads * vector_width
# contiguous channels, and the grouped plane shift is applied per CTA, so
# qkv_group_size must be a multiple of this span (cf. Triton's
# CONV_FWD_BLOCK_N = 256).
CUTE_GROUP_CHANNEL_SPAN = 512


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
        seq_lens_cpu: Optional[Sequence[int]] = None,
        cache_indices: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
        qkv_group_size: Optional[int] = None,
        qkv_group_tokens: Optional[int] = None,
        sequence_lengths: Optional[Sequence[int]] = None,
    ):
        """Allocate output, compile/cache, and return a one-launch invocation.

        Call this outside the timed region, then invoke ``prepared.run()`` for
        kernel-only execution. ``sequence_lengths`` is a deprecated alias of
        ``seq_lens_cpu``.
        """
        if seq_lens_cpu is None:
            seq_lens_cpu = sequence_lengths
        elif sequence_lengths is not None:
            raise ValueError(
                "pass either seq_lens_cpu or its deprecated alias "
                "sequence_lengths, not both"
            )
        # TRT-surface parity validations (same conditions and messages as
        # tensorrt_llm/_torch/modules/mamba/causal_conv1d_prefill.py).
        if projected.ndim != 2:
            raise ValueError(
                f"projected must be 2D, got shape {tuple(projected.shape)}"
            )
        if projected.stride(1) != 1:
            raise ValueError(
                f"projected must have contiguous columns, got strides "
                f"{projected.stride()}"
            )
        if num_prefill_tokens < 0 or num_prefill_tokens > projected.shape[0]:
            raise ValueError(
                f"num_prefill_tokens must be in [0, {projected.shape[0]}], "
                f"got {num_prefill_tokens}"
            )
        if projected.shape[1] != weight.shape[0]:
            raise ValueError(
                f"projected dim {projected.shape[1]} != weight dim "
                f"{weight.shape[0]}"
            )
        if seq_lens_cpu is None or not len(seq_lens_cpu):
            raise ValueError(
                "seq_lens_cpu must contain at least one prefill sequence"
            )
        if sum(seq_lens_cpu) != num_prefill_tokens:
            raise ValueError(
                f"seq_lens_cpu sums to {sum(seq_lens_cpu)}, expected "
                f"{num_prefill_tokens}"
            )
        dim = projected.shape[1]
        if qkv_group_size is not None:
            if dim % qkv_group_size != 0 or (
                qkv_group_size % CUTE_GROUP_CHANNEL_SPAN != 0
            ):
                raise ValueError(
                    f"qkv_group_size={qkv_group_size} must divide dim={dim} "
                    f"and be a multiple of the launch channel block "
                    f"{CUTE_GROUP_CHANNEL_SPAN}"
                )
            group_tokens = (
                num_prefill_tokens if qkv_group_tokens is None else qkv_group_tokens
            )
            if group_tokens < num_prefill_tokens:
                raise ValueError(
                    f"qkv_group_tokens={group_tokens} must be >= "
                    f"num_prefill_tokens={num_prefill_tokens}"
                )

        # CuTe-backend validations beyond the shared surface.
        if not projected.is_cuda or projected.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("projected must be CUDA FP16 or BF16")
        if weight.ndim != 2 or weight.shape[1] not in (2, 3, 4):
            raise ValueError("weight must have shape [D,W] with W in {2, 3, 4}")
        if weight.dtype != projected.dtype or weight.stride(1) != 1:
            raise ValueError(
                "weight must match projected dtype and have width stride 1"
            )
        width = weight.shape[1]
        batch = query_start_loc.numel() - 1
        if batch < 1:
            raise ValueError("query_start_loc must describe at least one sequence")
        if query_start_loc.dtype != torch.int32:
            raise ValueError("query_start_loc must be int32")
        if cache_indices is None:
            # Triton-surface semantics: without cache indices, sequence i
            # owns state row i.
            cache_indices = torch.arange(
                batch, dtype=torch.int32, device=projected.device
            )
        elif cache_indices.shape != (batch,) or cache_indices.dtype != torch.int32:
            raise ValueError("cache_indices must be int32 with shape [B]")
        if has_initial_state is None:
            # Triton-surface semantics: no sequence consumes cached history.
            has_initial_state = torch.zeros(
                batch, dtype=torch.bool, device=projected.device
            )
        elif (
            has_initial_state.shape != (batch,)
            or has_initial_state.dtype != torch.bool
        ):
            raise ValueError("has_initial_state must be bool with shape [B]")
        if conv_states.ndim != 3 or conv_states.shape[1:] != (dim, width - 1):
            raise ValueError("conv_states must have shape [slots,D,W-1]")
        if conv_states.dtype != projected.dtype:
            raise ValueError("conv_states must match projected dtype")
        if bias is not None and (
            bias.shape != (dim,) or bias.dtype != projected.dtype
        ):
            raise ValueError("bias must have shape [D] and match projected dtype")
        if activation not in (None, "silu", "swish"):
            raise ValueError("activation must be None, 'silu', or 'swish'")

        seq_lens_cpu = tuple(int(length) for length in seq_lens_cpu)
        if len(seq_lens_cpu) != batch:
            raise ValueError(
                "seq_lens_cpu must contain one length per query_start_loc "
                "sequence"
            )

        dtype_name = next(
            name for name, dtype in DTYPES.items() if dtype == projected.dtype
        )
        shape = Shape(
            dtype=dtype_name,
            width=width,
            channels=dim,
            tokens=num_prefill_tokens,
            batch=batch,
            bias=bias is not None,
            activation=activation,
            row_stride=(
                None
                if projected.stride(0) == dim
                else projected.stride(0)
            ),
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
            sequence_lengths=seq_lens_cpu,
            qkv_group_size=qkv_group_size,
            qkv_group_tokens=qkv_group_tokens,
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
    """Convenience wrapper backed by a process-wide compile cache.

    Same call signature as the TRT-LLM integration surface's
    ``causal_conv1d_prefill``.
    """
    return _DEFAULT_RUNNER(*args, **kwargs)
