# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row-major prefill interface for the fused causal convolution.

This is the stable integration surface for the KDA prefill conv path:
callers pass token-major tensors and receive contiguous token-major
output from exactly one kernel launch. Swapping the underlying kernel
(Triton today, CuTeDSL later) only changes the call inside this module.
"""

from collections.abc import Sequence
from typing import Optional

import torch
import triton
import triton.language as tl

from .causal_conv1d_triton import CONV_FWD_BLOCK_N, _causal_conv1d_fwd_kernel
from .causal_conv1d_triton import causal_conv1d_fn as causal_conv1d_triton


@triton.jit()
def b10_kda_conv1d_prefill_kernel(
    # b10_-named inlining wrapper: the trace shows THIS symbol for the KDA
    # one-launch prefill conv, distinguishable from other models' launches
    # of the stock kernel.
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QKV_GROUP_SIZE: tl.constexpr = 0,
    QKV_GROUP_TOKENS: tl.int32 = 0,
):
    if QKV_GROUP_SIZE > 0:
        # Grouped store: the output is [dim // G, group_tokens, G] contiguous
        # instead of token-major [T, dim]. The host guarantees G % BLOCK_N == 0,
        # so every channel block lies inside one group and the layout change
        # reduces to a per-program base shift; with stride_o_dim=1,
        # stride_o_token=G the inlined store then lands at
        # group*G*group_tokens + token*G + (channel % G). group_tokens may
        # exceed seqlen (mixed batches: the host splices decode-token output
        # into the tail rows of each group plane after this kernel).
        group = (tl.program_id(2) * BLOCK_N) // QKV_GROUP_SIZE
        o_ptr = o_ptr + group.to(tl.int64) * QKV_GROUP_SIZE * (QKV_GROUP_TOKENS - 1)
    _causal_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        initial_states_ptr,
        cache_indices_ptr,
        has_initial_states_ptr,
        query_start_loc_ptr,
        o_ptr,
        dim,
        seqlen,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS,
        KERNEL_WIDTH,
        SILU_ACTIVATION,
        HAS_INITIAL_STATES,
        HAS_CACHE,
        IS_CONTINUOUS_BATCHING,
        USE_PAD_SLOT,
        NP2_STATELEN,
        BLOCK_M,
        BLOCK_N,
    )


def causal_conv1d_prefill(
    projected: torch.Tensor,
    num_prefill_tokens: int,
    weight: torch.Tensor,
    *,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    qkv_group_size: Optional[int] = None,
    qkv_group_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Apply packed-varlen causal convolution without layout-copy kernels.

    Args:
        projected: Input in token-major layout ``[num_tokens, dim]``. May be a
            row-strided view (e.g. one chunk of a packed in_proj ``split``);
            only ``stride(1) == 1`` is required.
        num_prefill_tokens: Number of leading tokens to process.
        weight: Depthwise convolution weights in ``[dim, width]`` layout.
        conv_states: Slot-indexed convolution state cache, updated in place.
        query_start_loc: GPU cumulative prefill sequence lengths.
        seq_lens_cpu: CPU prefill sequence lengths used to size the launch grid.
        cache_indices: State-cache slot for each prefill sequence.
        has_initial_state: Whether each sequence consumes its cached state.
        bias: Optional per-channel bias.
        activation: ``None``, ``"silu"``, or ``"swish"``.
        qkv_group_size: When set (e.g. ``num_heads * head_dim``), the output
            is returned grouped as ``[dim // qkv_group_size,
            group_tokens, qkv_group_size]`` contiguous — token-major
            within each channel group with groups outermost — so a q/k/v
            consumer can view-split it without a permute+contiguous copy.
            Must divide ``dim`` and be a multiple of the launch's channel
            block (``CONV_FWD_BLOCK_N``).
        qkv_group_tokens: Token rows allocated per group plane (defaults to
            ``num_prefill_tokens``). Setting it larger leaves the tail rows
            of every plane uninitialized so a mixed-batch caller can splice
            decode-token output in without reallocating.

    Returns:
        Convolution output in contiguous token-major layout
        ``[num_prefill_tokens, dim]``, or the grouped layout above when
        ``qkv_group_size`` is set (only the leading ``num_prefill_tokens``
        rows of each group plane are written).

    The transposes below only change tensor metadata: the kernel reads the
    token-major storage through a channel-major view and writes into an
    explicitly allocated token-major output through the same trick, so this
    function launches exactly one GPU kernel regardless of how ``projected``
    is strided.

    Padded sequences (``cache_indices == pad_slot_id``) are skipped by the
    kernel and their output rows are left uninitialized — callers must not
    read them.
    """
    if projected.ndim != 2:
        raise ValueError(f"projected must be 2D, got shape {tuple(projected.shape)}")
    if projected.stride(1) != 1:
        raise ValueError(
            f"projected must have contiguous columns, got strides {projected.stride()}"
        )
    if num_prefill_tokens < 0 or num_prefill_tokens > projected.shape[0]:
        raise ValueError(
            f"num_prefill_tokens must be in [0, {projected.shape[0]}], got {num_prefill_tokens}"
        )
    if projected.shape[1] != weight.shape[0]:
        raise ValueError(f"projected dim {projected.shape[1]} != weight dim {weight.shape[0]}")
    if not len(seq_lens_cpu):
        raise ValueError("seq_lens_cpu must contain at least one prefill sequence")
    if sum(seq_lens_cpu) != num_prefill_tokens:
        raise ValueError(f"seq_lens_cpu sums to {sum(seq_lens_cpu)}, expected {num_prefill_tokens}")

    dim = projected.shape[1]
    prefill = projected[:num_prefill_tokens].transpose(0, 1)
    # Result allocated here (not inside the kernel): empty_like on a
    # non-dense view would silently fall back to a channel-major allocation
    # and reintroduce the transpose-materialization copy downstream.
    if qkv_group_size is None:
        output = torch.empty(
            num_prefill_tokens,
            dim,
            dtype=projected.dtype,
            device=projected.device,
        )
        out_view = output.transpose(0, 1)
        kernel_meta = None
    else:
        if dim % qkv_group_size != 0 or qkv_group_size % CONV_FWD_BLOCK_N != 0:
            raise ValueError(
                f"qkv_group_size={qkv_group_size} must divide dim={dim} and be "
                f"a multiple of the launch channel block {CONV_FWD_BLOCK_N}"
            )
        group_tokens = num_prefill_tokens if qkv_group_tokens is None else qkv_group_tokens
        if group_tokens < num_prefill_tokens:
            raise ValueError(
                f"qkv_group_tokens={group_tokens} must be >= "
                f"num_prefill_tokens={num_prefill_tokens}"
            )
        output = torch.empty(
            dim // qkv_group_size,
            group_tokens,
            qkv_group_size,
            dtype=projected.dtype,
            device=projected.device,
        )
        # 2D descriptor over the grouped storage: base pointer plus
        # (stride_dim=1, stride_token=G) for the kernel; the group plane
        # offset is applied per program inside the b10 kernel.
        out_view = torch.as_strided(
            output,
            (dim, num_prefill_tokens),
            (1, qkv_group_size),
        )
        kernel_meta = {
            "QKV_GROUP_SIZE": qkv_group_size,
            "QKV_GROUP_TOKENS": group_tokens,
        }
    causal_conv1d_triton(
        prefill,
        weight,
        bias,
        conv_states,
        query_start_loc,
        list(seq_lens_cpu),
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        out=out_view,
        kernel=b10_kda_conv1d_prefill_kernel,
        kernel_meta=kernel_meta,
    )
    return output
