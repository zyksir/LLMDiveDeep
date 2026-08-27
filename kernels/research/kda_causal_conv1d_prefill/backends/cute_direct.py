# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct token-major CuTe DSL causal-conv1d candidate."""

from __future__ import annotations

import math
import os
import time

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from common import PAD_SLOT_ID, Prepared, Problem

F32 = cutlass.Float32
I32 = cutlass.Int32
THREADS = 128
VEC = 8


def measured_alignment_bytes(tensor: torch.Tensor, cap: int = 32) -> int:
    """Largest power-of-two alignment provable for every row start.

    Derived from the actual base pointer and the physical row stride, so
    row-strided views (production stride 4752) are handled per-measurement
    instead of by assumption.
    """
    align = math.gcd(cap, tensor.data_ptr())
    if tensor.ndim >= 2:
        align = math.gcd(align, tensor.stride(0) * tensor.element_size())
    return align


def _prefetch_l2(gptr):
    """Pull one global cache line toward L2 without a register dependency."""
    _llvm.inline_asm(
        None,
        [gptr.toint().ir_value()],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
    )


def make_launcher(
    total_tokens: int,
    channels: int,
    batch: int,
    max_sequence_length: int,
    width: int,
    token_tile: int,
    has_bias: bool,
    use_silu: bool,
):
    """Build one specialized one-launch row-major kernel."""

    state_len = width - 1
    channel_vectors = channels // VEC
    channel_blocks = (channel_vectors + THREADS - 1) // THREADS
    token_blocks = (max_sequence_length + token_tile - 1) // token_tile

    @cute.kernel
    def causal_conv1d_prefill_kernel(
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: cute.Tensor,
        mState: cute.Tensor,
        mQueryStart: cute.Tensor,
        mCacheIndices: cute.Tensor,
        mHasInitial: cute.Tensor,
        mOutput: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, token_block, sequence = cute.arch.block_idx()
        channel_vector = channel_block * THREADS + tidx
        channel_base = channel_vector * VEC
        active_channel = channel_base < channels
        sequence_start = mQueryStart[sequence]
        sequence_end = mQueryStart[sequence + 1]
        sequence_length = sequence_end - sequence_start
        local_token_base = token_block * token_tile
        slot = mCacheIndices[sequence]

        weights = cute.make_fragment((VEC, width), F32)
        bias_values = cute.make_fragment((VEC,), F32)
        for vector_index in cutlass.range_constexpr(VEC):
            bias_values[vector_index] = F32(0.0)
            for tap in cutlass.range_constexpr(width):
                weights[vector_index, tap] = F32(0.0)
        if active_channel:
            for vector_index in cutlass.range_constexpr(VEC):
                channel = channel_base + vector_index
                if cutlass.const_expr(has_bias):
                    bias_values[vector_index] = mBias[channel].to(F32)
                for tap in cutlass.range_constexpr(width):
                    weights[vector_index, tap] = mWeight[channel, tap].to(F32)

        initial_state = cute.make_fragment((VEC, state_len), mX.element_type)
        for vector_index in cutlass.range_constexpr(VEC):
            for state_index in cutlass.range_constexpr(state_len):
                initial_state[vector_index, state_index] = mX.element_type(0.0)

        # Only chunk zero consumes cached history. Snapshot it into registers
        # before updating the slot so there is no cross-CTA read/write race.
        if (token_block == 0) & active_channel & (slot != PAD_SLOT_ID):
            has_initial = mHasInitial[sequence]
            if has_initial:
                for vector_index in cutlass.range_constexpr(VEC):
                    for state_index in cutlass.range_constexpr(state_len):
                        initial_state[vector_index, state_index] = mState[
                            slot,
                            channel_base + vector_index,
                            state_index,
                        ]

            for state_index in cutlass.range_constexpr(state_len):
                source_local = sequence_length - state_len + state_index
                state_fragment = cute.make_fragment((VEC,), mX.element_type)
                if source_local >= 0:
                    source_row = mX[sequence_start + source_local, None]
                    source_vectors = cute.zipped_divide(source_row, (VEC,))
                    cute.autovec_copy(
                        source_vectors[(None, channel_vector)],
                        state_fragment,
                    )
                else:
                    for vector_index in cutlass.range_constexpr(VEC):
                        state_value = mX.element_type(0.0)
                        if has_initial:
                            state_value = initial_state[
                                vector_index,
                                state_index + sequence_length,
                            ]
                        state_fragment[vector_index] = state_value
                for vector_index in cutlass.range_constexpr(VEC):
                    mState[
                        slot,
                        channel_base + vector_index,
                        state_index,
                    ] = state_fragment[vector_index]

        for token_inner in cutlass.range_constexpr(token_tile):
            local_token = local_token_base + token_inner
            token = sequence_start + local_token
            if (local_token < sequence_length) & active_channel:

                output_row = mOutput[token, None]
                output_vectors = cute.zipped_divide(output_row, (VEC,))
                output_fragment = cute.make_fragment((VEC,), mOutput.element_type)

                if slot == PAD_SLOT_ID:
                    input_row = mX[token, None]
                    input_vectors = cute.zipped_divide(input_row, (VEC,))
                    cute.autovec_copy(input_vectors[(None, channel_vector)], output_fragment)
                    cute.autovec_copy(output_fragment, output_vectors[(None, channel_vector)])
                else:
                    accumulator = cute.make_fragment((VEC,), F32)
                    for vector_index in cutlass.range_constexpr(VEC):
                        accumulator[vector_index] = bias_values[vector_index]

                    for tap in cutlass.range_constexpr(width):
                        history_local_token = local_token - state_len + tap
                        input_fragment = cute.make_fragment((VEC,), mX.element_type)
                        if history_local_token >= 0:
                            history_token = sequence_start + history_local_token
                            history_row = mX[history_token, None]
                            history_vectors = cute.zipped_divide(history_row, (VEC,))
                            cute.autovec_copy(
                                history_vectors[(None, channel_vector)],
                                input_fragment,
                            )
                        else:
                            for vector_index in cutlass.range_constexpr(VEC):
                                input_fragment[vector_index] = initial_state[
                                    vector_index,
                                    local_token + tap,
                                ]
                        for vector_index in cutlass.range_constexpr(VEC):
                            accumulator[vector_index] = (
                                accumulator[vector_index]
                                + input_fragment[vector_index].to(F32)
                                * weights[vector_index, tap]
                            )

                    for vector_index in cutlass.range_constexpr(VEC):
                        value = accumulator[vector_index]
                        if cutlass.const_expr(use_silu):
                            denominator = F32(1.0) + cute.math.exp(
                                F32(0.0) - value,
                                fastmath=True,
                            )
                            value = value / denominator
                        output_fragment[vector_index] = value.to(mOutput.element_type)
                    cute.autovec_copy(output_fragment, output_vectors[(None, channel_vector)])

    @cute.jit
    def launch(
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: cute.Tensor,
        mState: cute.Tensor,
        mQueryStart: cute.Tensor,
        mCacheIndices: cute.Tensor,
        mHasInitial: cute.Tensor,
        mOutput: cute.Tensor,
        stream,
    ):
        causal_conv1d_prefill_kernel(
            mX,
            mWeight,
            mBias,
            mState,
            mQueryStart,
            mCacheIndices,
            mHasInitial,
            mOutput,
        ).launch(
            grid=[channel_blocks, token_blocks, batch],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    return launch


def make_streaming_launcher(
    total_tokens: int,
    channels: int,
    batch: int,
    max_sequence_length: int,
    width: int,
    token_tile: int,
    has_bias: bool,
    use_silu: bool,
    vector_width: int,
    threads: int,
    prefetch: bool,
    fp32_params: bool,
    l2_prefetch_blocks: int,
    sequence_lengths: tuple[int, ...],
    row_stride: int,
    silu_mode: str = "expdiv",
    group_loads: bool = False,
    group_span: int = 4,
    f32_ring: bool = False,
):
    """Build a lower-register kernel with a four-phase circular token window."""

    del total_tokens
    state_len = width - 1
    channel_vectors = channels // vector_width
    channel_blocks = (channel_vectors + threads - 1) // threads
    del max_sequence_length
    sequence_block_starts: list[int] = []
    total_token_blocks = 0
    for sequence_length in sequence_lengths:
        sequence_block_starts.append(total_token_blocks)
        total_token_blocks += (sequence_length + token_tile - 1) // token_tile
    token_groups = token_tile // 4

    @cute.kernel
    def causal_conv1d_prefill_streaming_kernel(
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: cute.Tensor,
        mState: cute.Tensor,
        mQueryStart: cute.Tensor,
        mCacheIndices: cute.Tensor,
        mHasInitial: cute.Tensor,
        mOutput: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, flat_token_block, _ = cute.arch.block_idx()
        sequence = I32(0)
        token_block = flat_token_block
        for sequence_index in cutlass.range_constexpr(batch):
            if flat_token_block >= sequence_block_starts[sequence_index]:
                sequence = I32(sequence_index)
                token_block = (
                    flat_token_block - sequence_block_starts[sequence_index]
                )
        channel_vector = channel_block * threads + tidx
        channel_base = channel_vector * vector_width
        active_channel = channel_base < channels
        sequence_start = mQueryStart[sequence]
        sequence_end = mQueryStart[sequence + 1]
        sequence_length = sequence_end - sequence_start
        local_token_base = token_block * token_tile
        slot = mCacheIndices[sequence]

        if cutlass.const_expr(fp32_params):
            weights = cute.make_fragment((vector_width, width), F32)
            bias_values = cute.make_fragment((vector_width,), F32)
        else:
            weights = cute.make_fragment(
                (vector_width, width),
                mX.element_type,
            )
            bias_values = cute.make_fragment(
                (vector_width,),
                mX.element_type,
            )
        for vector_index in cutlass.range_constexpr(vector_width):
            bias_values[vector_index] = bias_values.element_type(0.0)
            for tap in cutlass.range_constexpr(width):
                weights[vector_index, tap] = weights.element_type(0.0)
        if active_channel:
            for vector_index in cutlass.range_constexpr(vector_width):
                channel = channel_base + vector_index
                if cutlass.const_expr(has_bias):
                    bias_values[vector_index] = mBias[channel].to(
                        bias_values.element_type
                    )
                for tap in cutlass.range_constexpr(width):
                    weights[vector_index, tap] = mWeight[channel, tap].to(
                        weights.element_type
                    )

        # Four slots make the rotation return to the same static mapping after
        # each four-token runtime-loop iteration. Widths below four simply
        # leave the oldest slots unused by the tap loop.
        #
        # With f32_ring, resident window values are kept pre-converted to FP32
        # (BF16->FP32 is exact, so state writebacks converted back remain
        # bitwise identical). NCU shows the ALU pipe is the top compute
        # consumer (~64-66%), largely per-tap operand conversions; converting
        # once at load removes up to width-1 redundant converts per value.
        if cutlass.const_expr(f32_ring):
            ring_type = F32
        else:
            ring_type = mX.element_type
        ring = [
            cute.make_fragment((vector_width,), ring_type),
            cute.make_fragment((vector_width,), ring_type),
            cute.make_fragment((vector_width,), ring_type),
            cute.make_fragment((vector_width,), ring_type),
        ]
        for ring_index in cutlass.range_constexpr(4):
            for vector_index in cutlass.range_constexpr(vector_width):
                ring[ring_index][vector_index] = ring_type(0.0)

        if (
            (local_token_base < sequence_length)
            & active_channel
            & (slot != PAD_SLOT_ID)
        ):
            if token_block == 0:
                has_initial = mHasInitial[sequence]
                if has_initial:
                    for history_index in cutlass.range_constexpr(state_len):
                        for vector_index in cutlass.range_constexpr(vector_width):
                            ring[history_index][vector_index] = mState[
                                slot,
                                channel_base + vector_index,
                                history_index,
                            ].to(ring_type)
            else:
                for history_index in cutlass.range_constexpr(state_len):
                    history_token = (
                        sequence_start
                        + local_token_base
                        - state_len
                        + history_index
                    )
                    history_row = mX[history_token, None]
                    history_vectors = cute.zipped_divide(
                        history_row,
                        (vector_width,),
                    )
                    if cutlass.const_expr(f32_ring):
                        history_staging = cute.make_fragment(
                            (vector_width,),
                            mX.element_type,
                        )
                        cute.autovec_copy(
                            history_vectors[(None, channel_vector)],
                            history_staging,
                        )
                        for vector_index in cutlass.range_constexpr(
                            vector_width
                        ):
                            ring[history_index][vector_index] = history_staging[
                                vector_index
                            ].to(ring_type)
                    else:
                        cute.autovec_copy(
                            history_vectors[(None, channel_vector)],
                            ring[history_index],
                        )

        # Chunk zero has snapshotted every cached value that output can consume.
        # Update final state now; later chunks read only immutable input rows.
        if (token_block == 0) & active_channel & (slot != PAD_SLOT_ID):
            has_initial = mHasInitial[sequence]
            for state_index in cutlass.range_constexpr(state_len):
                source_local = sequence_length - state_len + state_index
                state_fragment = cute.make_fragment(
                    (vector_width,),
                    mX.element_type,
                )
                for vector_index in cutlass.range_constexpr(vector_width):
                    state_fragment[vector_index] = mX.element_type(0.0)
                if source_local >= 0:
                    source_row = mX[sequence_start + source_local, None]
                    source_vectors = cute.zipped_divide(
                        source_row,
                        (vector_width,),
                    )
                    cute.autovec_copy(
                        source_vectors[(None, channel_vector)],
                        state_fragment,
                    )
                elif has_initial:
                    for history_index in cutlass.range_constexpr(state_len):
                        if source_local == history_index - state_len:
                            for vector_index in cutlass.range_constexpr(vector_width):
                                state_fragment[vector_index] = ring[history_index][
                                    vector_index
                                ].to(mX.element_type)
                for vector_index in cutlass.range_constexpr(vector_width):
                    mState[
                        slot,
                        channel_base + vector_index,
                        state_index,
                    ] = state_fragment[vector_index]

        # The runtime outer loop prevents code-size growth for larger token
        # tiles. The static four-token inner phase keeps all ring indices
        # compile-time constants and avoids local-memory indexed arrays.
        if cutlass.const_expr(group_loads):
            # Group-batched pipeline: the group_span row loads of a token
            # group are independent (each fills its own fragment), so issuing
            # them back-to-back before any compute multiplies the bytes in
            # flight per warp by the span. NCU shows the four-phase
            # load->compute->store loop is long-scoreboard bound (~69% of
            # stall cycles), i.e. warps starve waiting on single in-flight
            # global loads.
            for token_group in cutlass.range(token_tile // group_span, unroll=1):
                group_token_base = local_token_base + token_group * group_span
                current = [
                    cute.make_fragment((vector_width,), ring_type)
                    for _ in range(group_span)
                ]
                staging = [
                    cute.make_fragment((vector_width,), mX.element_type)
                    for _ in range(group_span if f32_ring else 0)
                ]
                for phase in cutlass.range_constexpr(group_span):
                    local_token = group_token_base + phase
                    if (local_token < sequence_length) & active_channel:
                        if cutlass.const_expr(l2_prefetch_blocks > 0):
                            lanes_per_cache_line = 64 // vector_width
                            future_local_token = (
                                local_token + l2_prefetch_blocks * token_tile
                            )
                            if (
                                (tidx % lanes_per_cache_line == 0)
                                & (future_local_token < sequence_length)
                            ):
                                future_offset = (
                                    (sequence_start + future_local_token)
                                    * row_stride
                                    + channel_base
                                )
                                _prefetch_l2(mX.iterator + future_offset)
                        input_row = mX[sequence_start + local_token, None]
                        input_vectors = cute.zipped_divide(
                            input_row,
                            (vector_width,),
                        )
                        if cutlass.const_expr(f32_ring):
                            cute.autovec_copy(
                                input_vectors[(None, channel_vector)],
                                staging[phase],
                            )
                        else:
                            cute.autovec_copy(
                                input_vectors[(None, channel_vector)],
                                current[phase],
                            )
                # Conversions run after every group load has been issued so
                # they never serialize the load stream.
                if cutlass.const_expr(f32_ring):
                    for phase in cutlass.range_constexpr(group_span):
                        local_token = group_token_base + phase
                        if (local_token < sequence_length) & active_channel:
                            for vector_index in cutlass.range_constexpr(
                                vector_width
                            ):
                                current[phase][vector_index] = staging[phase][
                                    vector_index
                                ].to(ring_type)
                for phase in cutlass.range_constexpr(group_span):
                    local_token = group_token_base + phase
                    token = sequence_start + local_token
                    if (local_token < sequence_length) & active_channel:
                        output_row = mOutput[token, None]
                        output_vectors = cute.zipped_divide(
                            output_row,
                            (vector_width,),
                        )
                        output_fragment = cute.make_fragment(
                            (vector_width,),
                            mOutput.element_type,
                        )
                        if slot == PAD_SLOT_ID:
                            for vector_index in cutlass.range_constexpr(
                                vector_width
                            ):
                                # BF16->FP32->BF16 roundtrips are exact, so
                                # pad passthrough stays bitwise identical.
                                output_fragment[vector_index] = current[phase][
                                    vector_index
                                ].to(mOutput.element_type)
                        else:
                            accumulator = cute.make_fragment(
                                (vector_width,),
                                F32,
                            )
                            for vector_index in cutlass.range_constexpr(
                                vector_width
                            ):
                                accumulator[vector_index] = bias_values[
                                    vector_index
                                ].to(F32)
                            for tap in cutlass.range_constexpr(width):
                                source_index = phase + tap
                                if cutlass.const_expr(source_index < state_len):
                                    source = ring[source_index]
                                else:
                                    source = current[source_index - state_len]
                                for vector_index in cutlass.range_constexpr(
                                    vector_width
                                ):
                                    accumulator[vector_index] = (
                                        accumulator[vector_index]
                                        + source[vector_index].to(F32)
                                        * weights[vector_index, tap].to(F32)
                                    )
                            values = accumulator.load()
                            if cutlass.const_expr(use_silu):
                                if cutlass.const_expr(silu_mode == "tanh"):
                                    half = values * F32(0.5)
                                    values = half + half * cute.math.tanh(
                                        half,
                                        fastmath=True,
                                    )
                                else:
                                    denominator = F32(1.0) + cute.math.exp(
                                        F32(0.0) - values,
                                        fastmath=True,
                                    )
                                    values = values / denominator
                            output_fragment.store(
                                values.to(mOutput.element_type)
                            )
                        cute.autovec_copy(
                            output_fragment,
                            output_vectors[(None, channel_vector)],
                        )
                # Rotate the history window: the last state_len tokens of this
                # group seed the next group's taps. Beyond-sequence groups
                # copy garbage that is provably never consumed (every later
                # token in this tile is also beyond the sequence end).
                for history_index in cutlass.range_constexpr(state_len):
                    for vector_index in cutlass.range_constexpr(vector_width):
                        ring[history_index][vector_index] = current[
                            group_span - state_len + history_index
                        ][vector_index]
        else:
            for token_group in cutlass.range(token_groups, unroll=1):
                group_token_base = local_token_base + token_group * 4
                prefetched_input = cute.make_fragment(
                    (vector_width,),
                    mX.element_type,
                )
                for vector_index in cutlass.range_constexpr(vector_width):
                    prefetched_input[vector_index] = mX.element_type(0.0)
                for phase in cutlass.range_constexpr(4):
                    local_token = group_token_base + phase
                    token = sequence_start + local_token
                    if (local_token < sequence_length) & active_channel:
                        if cutlass.const_expr(l2_prefetch_blocks > 0):
                            lanes_per_cache_line = 64 // vector_width
                            future_local_token = (
                                local_token + l2_prefetch_blocks * token_tile
                            )
                            if (
                                (tidx % lanes_per_cache_line == 0)
                                & (future_local_token < sequence_length)
                            ):
                                future_offset = (
                                    (sequence_start + future_local_token)
                                    * row_stride
                                    + channel_base
                                )
                                _prefetch_l2(mX.iterator + future_offset)
                        output_row = mOutput[token, None]
                        output_vectors = cute.zipped_divide(
                            output_row,
                            (vector_width,),
                        )
                        output_fragment = cute.make_fragment(
                            (vector_width,),
                            mOutput.element_type,
                        )
                        input_row = mX[token, None]
                        input_vectors = cute.zipped_divide(
                            input_row,
                            (vector_width,),
                        )

                        if slot == PAD_SLOT_ID:
                            cute.autovec_copy(
                                input_vectors[(None, channel_vector)],
                                output_fragment,
                            )
                        else:
                            current_slot = (state_len + phase) % width
                            if cutlass.const_expr((not prefetch) or phase == 0):
                                cute.autovec_copy(
                                    input_vectors[(None, channel_vector)],
                                    ring[current_slot],
                                )
                            else:
                                for vector_index in cutlass.range_constexpr(
                                    vector_width
                                ):
                                    ring[current_slot][vector_index] = (
                                        prefetched_input[vector_index]
                                    )
                            if cutlass.const_expr(prefetch and phase < 3):
                                next_local_token = local_token + 1
                                if next_local_token < sequence_length:
                                    next_row = mX[
                                        sequence_start + next_local_token, None
                                    ]
                                    next_vectors = cute.zipped_divide(
                                        next_row,
                                        (vector_width,),
                                    )
                                    cute.autovec_copy(
                                        next_vectors[(None, channel_vector)],
                                        prefetched_input,
                                    )
                            accumulator = cute.make_fragment(
                                (vector_width,),
                                F32,
                            )
                            for vector_index in cutlass.range_constexpr(
                                vector_width
                            ):
                                accumulator[vector_index] = bias_values[
                                    vector_index
                                ].to(F32)
                            for tap in cutlass.range_constexpr(width):
                                ring_slot = (phase + tap) % width
                                for vector_index in cutlass.range_constexpr(
                                    vector_width
                                ):
                                    accumulator[vector_index] = (
                                        accumulator[vector_index]
                                        + ring[ring_slot][vector_index].to(F32)
                                        * weights[vector_index, tap].to(F32)
                                    )
                            values = accumulator.load()
                            if cutlass.const_expr(use_silu):
                                if cutlass.const_expr(silu_mode == "tanh"):
                                    # silu(x) = 0.5*x*(1 + tanh(x/2)); fastmath
                                    # tanh lowers to one MUFU.TANH instead of
                                    # an exp plus a full-precision divide
                                    # sequence.
                                    half = values * F32(0.5)
                                    values = half + half * cute.math.tanh(
                                        half,
                                        fastmath=True,
                                    )
                                else:
                                    denominator = F32(1.0) + cute.math.exp(
                                        F32(0.0) - values,
                                        fastmath=True,
                                    )
                                    values = values / denominator
                            output_fragment.store(
                                values.to(mOutput.element_type)
                            )
                        cute.autovec_copy(
                            output_fragment,
                            output_vectors[(None, channel_vector)],
                        )

    @cute.jit
    def launch(
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: cute.Tensor,
        mState: cute.Tensor,
        mQueryStart: cute.Tensor,
        mCacheIndices: cute.Tensor,
        mHasInitial: cute.Tensor,
        mOutput: cute.Tensor,
        stream,
    ):
        causal_conv1d_prefill_streaming_kernel(
            mX,
            mWeight,
            mBias,
            mState,
            mQueryStart,
            mCacheIndices,
            mHasInitial,
            mOutput,
        ).launch(
            grid=[channel_blocks, total_token_blocks, 1],
            block=[threads, 1, 1],
            stream=stream,
        )

    return launch


class CuTeDirectBackend:
    name = "cute_direct_row_major"
    source = "backends/cute_direct.py"

    def __init__(self) -> None:
        self.algorithm = os.environ.get("KDA_CUTE_ALGORITHM", "auto")
        if self.algorithm not in ("auto", "direct", "stream"):
            raise ValueError("KDA_CUTE_ALGORITHM must be auto, direct, or stream")
        token_tile = os.environ.get("KDA_CUTE_TOKEN_TILE", "auto")
        self.token_tile = None if token_tile == "auto" else int(token_tile)
        if self.token_tile not in (None, 4, 8, 12, 16, 20, 24, 32, 64, 128):
            raise ValueError(
                "KDA_CUTE_TOKEN_TILE must be auto or one of "
                "4, 8, 12, 16, 20, 24, 32, 64, 128"
            )
        # Round-3 selected single configuration (see REPORT.md): stream W4,
        # vector width 4, 128 threads, token tile 16, group-batched loads with
        # span 4, FP32 ring, tanh SiLU, BF16 parameter fragments (proven
        # bit-identical to FP32 parameters), FP32 accumulation.
        self.vector_width = int(os.environ.get("KDA_CUTE_VECTOR_WIDTH", "4"))
        self.threads = int(os.environ.get("KDA_CUTE_THREADS", "128"))
        self.prefetch = os.environ.get("KDA_CUTE_PREFETCH", "1") == "1"
        self.fp32_params = os.environ.get("KDA_CUTE_FP32_PARAMS", "0") == "1"
        self.l2_prefetch_blocks = int(
            os.environ.get("KDA_CUTE_L2_PREFETCH_BLOCKS", "0")
        )
        if self.l2_prefetch_blocks < 0:
            raise ValueError("KDA_CUTE_L2_PREFETCH_BLOCKS must be nonnegative")
        self.silu_mode = os.environ.get("KDA_CUTE_SILU_MODE", "tanh")
        if self.silu_mode not in ("expdiv", "tanh"):
            raise ValueError("KDA_CUTE_SILU_MODE must be expdiv or tanh")
        self.group_loads = os.environ.get("KDA_CUTE_GROUP_LOADS", "1") == "1"
        self.group_span = int(os.environ.get("KDA_CUTE_GROUP_SPAN", "4"))
        if self.group_span not in (4, 8, 12):
            raise ValueError("KDA_CUTE_GROUP_SPAN must be one of 4, 8, 12")
        self.f32_ring = os.environ.get("KDA_CUTE_F32_RING", "1") == "1"
        if self.f32_ring and not self.group_loads:
            # F32 ring only exists on the group-loads path; follow the
            # explicit group-loads opt-out instead of failing.
            self.f32_ring = False
        if self.vector_width not in (2, 4, 8, 16):
            raise ValueError("KDA_CUTE_VECTOR_WIDTH must be one of 2, 4, 8, 16")
        if self.threads not in (32, 64, 128, 256):
            raise ValueError("KDA_CUTE_THREADS must be one of 32, 64, 128, 256")
        if self.algorithm == "stream":
            self.name = (
                f"cute_stream_v{self.vector_width}_threads{self.threads}"
                f"_tile{self.token_tile or 'auto'}_prefetch{int(self.prefetch)}"
                f"_fp32params{int(self.fp32_params)}"
                f"_l2pf{self.l2_prefetch_blocks}"
                f"_silu{self.silu_mode}"
                f"_gl{int(self.group_loads)}"
                f"_gs{self.group_span}"
                f"_fr{int(self.f32_ring)}"
            )
        elif self.algorithm == "auto":
            self.name = "cute_selected_dispatch"
        self._cache: dict[tuple[object, ...], object] = {}
        self.compile_seconds: dict[tuple[object, ...], float] = {}

    def _algorithm(self, problem: Problem) -> str:
        if self.algorithm != "auto":
            return self.algorithm
        return "stream" if problem.shape.width == 4 else "direct"

    def _token_tile(self, problem: Problem) -> int:
        if self.token_tile is not None:
            return self.token_tile
        if self._algorithm(problem) == "stream":
            return 16
        return 16 if max(problem.sequence_lengths) > 1024 else 4

    def _effective_vector_width(self, problem: Problem) -> int:
        """Clamp the requested vector width to what pointer/stride alignment
        provably permits (verified, never assumed) for every streamed tensor."""
        element_size = problem.projected.element_size()
        align = min(
            measured_alignment_bytes(problem.projected),
            measured_alignment_bytes(problem.conv_states),
        )
        vector_width = self.vector_width
        while vector_width > 1 and vector_width * element_size > align:
            vector_width //= 2
        return vector_width

    def supports(self, problem: Problem) -> tuple[bool, str]:
        shape = problem.shape
        if shape.dtype not in ("fp16", "bf16"):
            return False, f"unsupported dtype {shape.dtype}"
        if shape.width not in (2, 3, 4):
            return False, f"unsupported width {shape.width}"
        if problem.projected.stride(1) != 1:
            return False, "projected channel stride must be 1"
        algorithm = self._algorithm(problem)
        if algorithm == "stream" and shape.width != 4:
            return False, "streaming four-phase ring currently specializes W4"
        if algorithm == "stream" and self._token_tile(problem) % 4:
            return False, "streaming token tile must be divisible by four"
        if (
            algorithm == "stream"
            and self.group_loads
            and self._token_tile(problem) % self.group_span
        ):
            return False, "streaming token tile must be divisible by group span"
        vector_width = (
            self._effective_vector_width(problem)
            if algorithm == "stream"
            else VEC
        )
        if shape.channels % vector_width:
            return False, f"channels must be divisible by {vector_width}"
        if shape.activation not in (None, "silu", "swish"):
            return False, f"unsupported activation {shape.activation}"
        return True, ""

    @staticmethod
    def _stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def _arguments(self, problem: Problem, output: torch.Tensor):
        bias = problem.bias
        if bias is None:
            bias = torch.empty(1, dtype=problem.projected.dtype, device=problem.projected.device)
        stream = self._stream()
        # Alignment is measured from real pointers and physical row strides so
        # strided production views never receive an unproven alignment claim.
        input_align = measured_alignment_bytes(problem.projected)
        output_align = measured_alignment_bytes(output)
        if input_align < 16 or output_align < 16:
            raise ValueError(
                "projected/output row alignment below 16 bytes is not supported"
            )
        arguments = (
            from_dlpack(
                problem.projected[: problem.shape.tokens],
                assumed_align=input_align,
            ),
            from_dlpack(problem.weight, assumed_align=16),
            from_dlpack(bias, assumed_align=16),
            from_dlpack(problem.conv_states, assumed_align=16),
            from_dlpack(problem.query_start_loc, assumed_align=4),
            from_dlpack(problem.cache_indices, assumed_align=4),
            from_dlpack(problem.has_initial_state, assumed_align=1),
            from_dlpack(output, assumed_align=output_align),
            stream,
        )
        return arguments

    def _compiled(self, problem: Problem, arguments):
        shape = problem.shape
        token_tile = self._token_tile(problem)
        algorithm = self._algorithm(problem)
        vector_width = (
            self._effective_vector_width(problem)
            if algorithm == "stream"
            else VEC
        )
        row_stride = problem.projected.stride(0)
        key = (
            shape.dtype,
            shape.width,
            shape.channels,
            shape.tokens,
            shape.batch,
            max(problem.sequence_lengths),
            problem.sequence_lengths,
            shape.bias,
            shape.activation,
            token_tile,
            algorithm,
            vector_width,
            self.threads,
            self.prefetch,
            self.fp32_params,
            self.l2_prefetch_blocks,
            row_stride,
            self.silu_mode,
            self.group_loads,
            self.group_span,
            self.f32_ring,
        )
        compiled = self._cache.get(key)
        if compiled is None:
            launcher_factory = (
                make_streaming_launcher
                if algorithm == "stream"
                else make_launcher
            )
            launcher_kwargs = {
                "total_tokens": shape.tokens,
                "channels": shape.channels,
                "batch": shape.batch,
                "max_sequence_length": max(problem.sequence_lengths),
                "width": shape.width,
                "token_tile": token_tile,
                "has_bias": shape.bias,
                "use_silu": shape.activation in ("silu", "swish"),
            }
            if algorithm == "stream":
                launcher_kwargs.update(
                    {
                        "vector_width": vector_width,
                        "threads": self.threads,
                        "prefetch": self.prefetch,
                        "fp32_params": self.fp32_params,
                        "l2_prefetch_blocks": self.l2_prefetch_blocks,
                        "sequence_lengths": problem.sequence_lengths,
                        "row_stride": row_stride,
                        "silu_mode": self.silu_mode,
                        "group_loads": self.group_loads,
                        "group_span": self.group_span,
                        "f32_ring": self.f32_ring,
                    }
                )
            launcher = launcher_factory(**launcher_kwargs)
            start = time.perf_counter()
            compiled = cute.compile(launcher, *arguments)
            self.compile_seconds[key] = time.perf_counter() - start
            self._cache[key] = compiled
        return compiled

    def prepare(self, problem: Problem) -> Prepared:
        output = torch.empty(
            (problem.shape.tokens, problem.shape.channels),
            dtype=problem.projected.dtype,
            device=problem.projected.device,
        )
        arguments = self._arguments(problem, output)
        compiled = self._compiled(problem, arguments)

        def run() -> torch.Tensor:
            compiled(*arguments)
            return output

        return Prepared(
            run=run,
            output=lambda: output,
            state=lambda: problem.conv_states,
            launch_count=1,
        )


Backend = CuTeDirectBackend
