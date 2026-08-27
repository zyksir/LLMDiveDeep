# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct token-major CuTe DSL causal-conv1d candidate."""

from __future__ import annotations

import os
import time

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from common import PAD_SLOT_ID, Prepared, Problem

F32 = cutlass.Float32
I32 = cutlass.Int32
THREADS = 128
VEC = 8


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


class CuTeDirectBackend:
    name = "cute_direct_row_major"
    source = "backends/cute_direct.py"

    def __init__(self) -> None:
        token_tile = os.environ.get("KDA_CUTE_TOKEN_TILE", "auto")
        self.token_tile = None if token_tile == "auto" else int(token_tile)
        if self.token_tile not in (None, 4, 8, 16):
            raise ValueError("KDA_CUTE_TOKEN_TILE must be auto or one of 4, 8, 16")
        self._cache: dict[tuple[object, ...], object] = {}
        self.compile_seconds: dict[tuple[object, ...], float] = {}

    def _token_tile(self, problem: Problem) -> int:
        if self.token_tile is not None:
            return self.token_tile
        return 16 if max(problem.sequence_lengths) > 1024 else 4

    def supports(self, problem: Problem) -> tuple[bool, str]:
        shape = problem.shape
        if shape.dtype not in ("fp16", "bf16"):
            return False, f"unsupported dtype {shape.dtype}"
        if shape.width not in (2, 3, 4):
            return False, f"unsupported width {shape.width}"
        if shape.channels % VEC:
            return False, f"channels must be divisible by {VEC}"
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
        arguments = (
            from_dlpack(problem.projected[: problem.shape.tokens], assumed_align=16),
            from_dlpack(problem.weight, assumed_align=16),
            from_dlpack(bias, assumed_align=16),
            from_dlpack(problem.conv_states, assumed_align=16),
            from_dlpack(problem.query_start_loc, assumed_align=4),
            from_dlpack(problem.cache_indices, assumed_align=4),
            from_dlpack(problem.has_initial_state, assumed_align=1),
            from_dlpack(output, assumed_align=16),
            stream,
        )
        return arguments

    def _compiled(self, problem: Problem, arguments):
        shape = problem.shape
        token_tile = self._token_tile(problem)
        key = (
            shape.dtype,
            shape.width,
            shape.channels,
            shape.tokens,
            shape.batch,
            max(problem.sequence_lengths),
            shape.bias,
            shape.activation,
            token_tile,
        )
        compiled = self._cache.get(key)
        if compiled is None:
            launcher = make_launcher(
                total_tokens=shape.tokens,
                channels=shape.channels,
                batch=shape.batch,
                max_sequence_length=max(problem.sequence_lengths),
                width=shape.width,
                token_tile=token_tile,
                has_bias=shape.bias,
                use_silu=shape.activation in ("silu", "swish"),
            )
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
