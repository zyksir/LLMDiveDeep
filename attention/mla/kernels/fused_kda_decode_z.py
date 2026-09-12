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

import functools
import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx
from tensorrt_llm._torch.modules.stochastic_rounding import (
    store_state,
    validate_fp16_stochastic_rounding,
)
from tensorrt_llm.logger import logger


class _KdaDecodeLaunchConfig(NamedTuple):
    value_tile: int
    num_warps: int
    num_stages: int


_KDA_DECODE_FALLBACK_CONFIG = _KdaDecodeLaunchConfig(
    value_tile=32,
    num_warps=4,
    num_stages=2,
)
_KDA_DECODE_AUTOTUNE_CONFIGS = tuple(
    triton.Config(
        {"VALUE_TILE": value_tile, "NUM_VALUE_TILES": 128 // value_tile},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for value_tile, num_warps, num_stages in (
        (16, 2, 1),
        (16, 4, 1),
        (32, 2, 1),
        (32, 4, 1),
        (32, 4, 2),
        (64, 2, 1),
        (64, 4, 1),
    )
)
_Z_SCRATCH: dict = {}
_KDA_DECODE_CONFIG_CACHE: dict[
    tuple[int, int, int, int, torch.dtype, bool], _KdaDecodeLaunchConfig
] = {}


@functools.cache
def _supports_pdl(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] >= 9


@triton.jit
def _conv4_silu_update(
    input_ptr,
    weight_ptr,
    conv_state_ptr,
    token_idx,
    state_idx,
    channel_offsets,
    channel_mask,
    stride_input_token: tl.constexpr,
    stride_weight_channel: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_conv_state_slot: tl.constexpr,
    stride_conv_state_channel: tl.constexpr,
    stride_conv_state_width: tl.constexpr,
):
    """Apply a width-four causal convolution and advance its raw-input cache."""
    state_base = (
        conv_state_ptr
        + state_idx * stride_conv_state_slot
        + channel_offsets * stride_conv_state_channel
    )
    state_0 = tl.load(state_base, mask=channel_mask, other=0.0).to(tl.float32)
    state_1 = tl.load(
        state_base + stride_conv_state_width,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    state_2 = tl.load(
        state_base + 2 * stride_conv_state_width,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    current = tl.load(
        input_ptr + token_idx * stride_input_token + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)

    weight_base = weight_ptr + channel_offsets * stride_weight_channel
    weight_0 = tl.load(weight_base, mask=channel_mask, other=0.0).to(tl.float32)
    weight_1 = tl.load(
        weight_base + stride_weight_width,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight_2 = tl.load(
        weight_base + 2 * stride_weight_width,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight_3 = tl.load(
        weight_base + 3 * stride_weight_width,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)

    accumulator = state_0 * weight_0
    accumulator += state_1 * weight_1
    accumulator += state_2 * weight_2
    accumulator += current * weight_3

    tl.store(state_base, state_1, mask=channel_mask)
    tl.store(
        state_base + stride_conv_state_width,
        state_2,
        mask=channel_mask,
    )
    tl.store(
        state_base + 2 * stride_conv_state_width,
        current,
        mask=channel_mask,
    )

    activated = accumulator * tl.sigmoid(accumulator)
    # Match the separate BF16 causal-convolution output before KDA consumes it.
    return activated.to(tl.bfloat16).to(tl.float32)


@triton.heuristics({"USE_STOCHASTIC_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.jit(do_not_specialize=["batch_size"])
def _fused_kda_decode_kernel(
    sq_scratch_ptr,
    counter_ptr,
    input_ptr,
    conv_weight_ptr,
    conv_state_ptr,
    raw_gate_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    state_ptr,
    rand_seed_ptr,
    output_gate_ptr,
    norm_weight_ptr,
    output_ptr,
    lower_bound,
    scale,
    norm_eps,
    num_state_slots,
    batch_size,
    stride_input_token: tl.constexpr,
    stride_conv_weight_channel: tl.constexpr,
    stride_conv_weight_width: tl.constexpr,
    stride_conv_state_slot: tl.constexpr,
    stride_conv_state_channel: tl.constexpr,
    stride_conv_state_width: tl.constexpr,
    stride_raw_gate_token: tl.constexpr,
    stride_raw_gate_head: tl.constexpr,
    stride_raw_gate_channel: tl.constexpr,
    stride_raw_beta_token: tl.constexpr,
    stride_raw_beta_head: tl.constexpr,
    stride_state_index: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_output_gate_token: tl.constexpr,
    stride_output_gate_head: tl.constexpr,
    stride_output_gate_channel: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_head: tl.constexpr,
    stride_output_channel: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    VALUE_TILE: tl.constexpr,
    NUM_VALUE_TILES: tl.constexpr,
    STATE_IS_FP16: tl.constexpr,
    USE_STOCHASTIC_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Fuse width-four convolution, single-token KDA, and gated RMSNorm."""
    if USE_STOCHASTIC_ROUNDING:
        tl.static_assert(
            STATE_IS_FP16,
            "KDA stochastic rounding requires an FP16 recurrent state",
        )
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    state_idx = tl.load(
        state_indices_ptr + token_idx * stride_state_index,
    ).to(tl.int64)

    channel_offsets = tl.arange(0, BLOCK_DIM)
    channel_mask = channel_offsets < HEAD_DIM
    output_base = output_ptr + token_idx * stride_output_token + head_idx * stride_output_head
    if state_idx < 0:
        z_offsets = tile_idx * VALUE_TILE + tl.arange(0, VALUE_TILE)
        tl.store(
            output_base + z_offsets * stride_output_channel,
            0.0,
            mask=z_offsets < HEAD_DIM,
        )
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    tl.device_assert(state_idx < num_state_slots, "KDA state index out of bounds")

    projection_size = NUM_HEADS * HEAD_DIM
    head_channel_offsets = head_idx * HEAD_DIM + channel_offsets
    q = _conv4_silu_update(
        input_ptr,
        conv_weight_ptr,
        conv_state_ptr,
        token_idx,
        state_idx,
        head_channel_offsets,
        channel_mask,
        stride_input_token,
        stride_conv_weight_channel,
        stride_conv_weight_width,
        stride_conv_state_slot,
        stride_conv_state_channel,
        stride_conv_state_width,
    )
    k = _conv4_silu_update(
        input_ptr,
        conv_weight_ptr,
        conv_state_ptr,
        token_idx,
        state_idx,
        projection_size + head_channel_offsets,
        channel_mask,
        stride_input_token,
        stride_conv_weight_channel,
        stride_conv_weight_width,
        stride_conv_state_slot,
        stride_conv_state_channel,
        stride_conv_state_width,
    )
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * scale
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    raw_gate = tl.load(
        raw_gate_ptr
        + token_idx * stride_raw_gate_token
        + head_idx * stride_raw_gate_head
        + channel_offsets * stride_raw_gate_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    dt_bias = tl.load(
        dt_bias_ptr + head_idx * HEAD_DIM + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    decay_rate = tl.exp(tl.load(A_log_ptr + head_idx).to(tl.float32))
    decay = tl.exp(lower_bound * tl.sigmoid(decay_rate * (raw_gate + dt_bias)))
    beta = tl.sigmoid(
        tl.load(
            raw_beta_ptr + token_idx * stride_raw_beta_token + head_idx * stride_raw_beta_head,
        ).to(tl.float32)
    )

    state_head_base = state_ptr + state_idx * stride_state_slot + head_idx * stride_state_head
    output_square_sum = 0.0
    if True:
        value_tile_idx = tile_idx
        value_offsets = value_tile_idx * VALUE_TILE + tl.arange(0, VALUE_TILE)
        value_mask = value_offsets < HEAD_DIM
        value_channel_offsets = 2 * projection_size + head_idx * HEAD_DIM + value_offsets
        v = _conv4_silu_update(
            input_ptr,
            conv_weight_ptr,
            conv_state_ptr,
            token_idx,
            state_idx,
            value_channel_offsets,
            value_mask,
            stride_input_token,
            stride_conv_weight_channel,
            stride_conv_weight_width,
            stride_conv_state_slot,
            stride_conv_state_channel,
            stride_conv_state_width,
        )

        state_offsets = (
            value_offsets[:, None] * stride_state_value
            + channel_offsets[None, :] * stride_state_key
        )
        state_mask = value_mask[:, None] & channel_mask[None, :]
        state = tl.load(
            state_head_base + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        state *= decay[None, :]
        delta = v - tl.sum(state * k[None, :], axis=1)
        state += (delta * beta)[:, None] * k[None, :]
        recurrent_output = tl.sum(state * q[None, :], axis=1)

        random_tile_idx = (state_idx * NUM_HEADS + head_idx) * NUM_VALUE_TILES + value_tile_idx
        store_state(
            state_head_base + state_offsets,
            state,
            state_mask,
            rand_seed_ptr,
            random_tile_idx,
            USE_STOCHASTIC_ROUNDING=USE_STOCHASTIC_ROUNDING,
            PHILOX_ROUNDS=PHILOX_ROUNDS,
        )
        rounded_output = recurrent_output.to(tl.bfloat16).to(tl.float32)
        tl.store(
            output_base + value_offsets * stride_output_channel,
            rounded_output,
            mask=value_mask,
        )
        output_square_sum += tl.sum(
            rounded_output * rounded_output,
            axis=0,
        )

    # cross-tile reduction: accumulate this tile's square-sum, and the
    # LAST tile to arrive finalizes the whole head (reads all raw
    # outputs, applies rms*gate). acq_rel atomics order the raw stores.
    slot = token_idx * NUM_HEADS + head_idx
    tl.atomic_add(sq_scratch_ptr + slot, output_square_sum)
    arrived = tl.atomic_add(counter_ptr + slot, 1) + 1
    if arrived < NUM_VALUE_TILES:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return
    total_sq = tl.load(sq_scratch_ptr + slot)
    # self-clean for the next call (CUDA-graph safe)
    tl.store(sq_scratch_ptr + slot, 0.0)
    tl.store(counter_ptr + slot, 0)
    inverse_rms = tl.rsqrt(total_sq / HEAD_DIM + norm_eps)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    for value_tile_idx in tl.static_range(0, NUM_VALUE_TILES):
        value_offsets = value_tile_idx * VALUE_TILE + tl.arange(0, VALUE_TILE)
        value_mask = value_offsets < HEAD_DIM
        recurrent_output = tl.load(
            output_base + value_offsets * stride_output_channel,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        norm_weight = tl.load(
            norm_weight_ptr + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        output_gate = tl.load(
            output_gate_ptr
            + token_idx * stride_output_gate_token
            + head_idx * stride_output_gate_head
            + value_offsets * stride_output_gate_channel,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        normalized = recurrent_output * inverse_rms * norm_weight * tl.sigmoid(output_gate)
        tl.store(
            output_base + value_offsets * stride_output_channel,
            normalized,
            mask=value_mask,
        )


_autotuned_fused_kda_decode_kernel = triton.autotune(
    configs=list(_KDA_DECODE_AUTOTUNE_CONFIGS),
    key=[
        "batch_size",
        "NUM_HEADS",
        "HEAD_DIM",
        "STATE_IS_FP16",
        "USE_STOCHASTIC_ROUNDING",
    ],
    cache_results=True,
)(_fused_kda_decode_kernel)


def fused_kda_decode(
    projected_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: float,
    norm_eps: float = 1e-5,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
) -> torch.Tensor:
    """Run Kimi K3's fused single-token Triton decode path.

    The operation mutates ``conv_state`` and ``state`` in place and returns
    ``[1, batch, heads, head_dim]`` BF16 output after gated RMSNorm.
    """
    if projected_qkv.ndim != 2 or projected_qkv.stride(1) != 1:
        raise ValueError("projected_qkv must be row-major [batch, 3 * heads * head_dim]")
    if projected_qkv.dtype != torch.bfloat16:
        raise ValueError("fused KDA decode requires BF16 projections")

    batch_size, packed_size = projected_qkv.shape
    if batch_size <= 0:
        raise ValueError("fused KDA decode requires at least one token")
    if state.ndim != 4:
        raise ValueError("state must be a 4D [slots, heads, value_dim, key_dim] tensor")
    num_state_slots, num_heads, value_dim, key_dim = state.shape
    if value_dim != key_dim or packed_size != 3 * num_heads * key_dim:
        raise ValueError("projected_qkv and state have incompatible KDA dimensions")
    if key_dim != 128:
        raise ValueError("fused KDA decode currently supports head_dim=128")
    if state.dtype not in (torch.float16, torch.float32):
        raise ValueError("fused KDA decode requires an FP16 or FP32 recurrent state")
    validate_fp16_stochastic_rounding(
        state,
        rand_seed,
        philox_rounds,
        operation="fused KDA decode",
    )
    if state.stride(-1) != 1:
        raise ValueError("state must be contiguous in its key dimension")

    if conv_weight.shape != (packed_size, 4) or conv_weight.dtype != torch.bfloat16:
        raise ValueError("conv_weight must be BF16 [3 * heads * head_dim, 4]")
    if conv_state.shape != (num_state_slots, packed_size, 3):
        raise ValueError("conv_state must have shape [slots, 3 * heads * head_dim, 3]")
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("fused KDA decode requires a BF16 convolution state")

    if raw_gate.shape != (1, batch_size, num_heads, key_dim):
        raise ValueError("raw_gate must have shape [1, batch, heads, head_dim]")
    if raw_beta.shape != (1, batch_size, num_heads):
        raise ValueError("raw_beta must have shape [1, batch, heads]")
    if raw_gate.dtype != torch.bfloat16 or raw_beta.dtype != torch.bfloat16:
        raise ValueError("raw_gate and raw_beta must be BF16")
    if A_log.shape != (num_heads,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [heads]")
    if dt_bias.shape != (num_heads * key_dim,) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 [heads * head_dim]")
    if state_indices.shape != (batch_size,) or state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be int32 [batch]")
    if output_gate.shape != (batch_size, num_heads * value_dim):
        raise ValueError("output_gate must have shape [batch, heads * head_dim]")
    if output_gate.dtype != torch.bfloat16:
        raise ValueError("output_gate must be BF16")
    if norm_weight.shape != (value_dim,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError("norm_weight must be BF16 [head_dim]")
    if not -5.0 <= lower_bound < 0.0:
        raise ValueError("lower_bound must be in the range [-5, 0)")
    if norm_eps <= 0.0:
        raise ValueError("norm_eps must be positive")

    tensors = (
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
    )
    if not projected_qkv.is_cuda or any(
        not tensor.is_cuda or tensor.device != projected_qkv.device for tensor in tensors
    ):
        raise ValueError("all fused KDA decode tensors must share one CUDA device")
    device_index = projected_qkv.device.index
    if device_index is None:
        raise RuntimeError("fused KDA decode could not resolve its CUDA device")

    raw_gate_view = raw_gate[0]
    raw_beta_view = raw_beta[0]
    output_gate_view = output_gate.view(batch_size, num_heads, value_dim)
    output = projected_qkv.new_empty(
        1,
        batch_size,
        num_heads,
        value_dim,
    )

    block_dim = triton.next_power_of_2(key_dim)
    launch_pdl = _supports_pdl(device_index)

    def launch(
        *,
        launch_config: _KdaDecodeLaunchConfig | None,
        launch_conv_state: torch.Tensor,
        launch_state_indices: torch.Tensor,
        launch_state: torch.Tensor,
        launch_output: torch.Tensor,
        use_pdl: bool,
    ) -> None:
        kernel_args = {
            "input_ptr": projected_qkv,
            "conv_weight_ptr": conv_weight,
            "conv_state_ptr": launch_conv_state,
            "raw_gate_ptr": raw_gate_view,
            "raw_beta_ptr": raw_beta_view,
            "A_log_ptr": A_log,
            "dt_bias_ptr": dt_bias,
            "state_indices_ptr": launch_state_indices,
            "state_ptr": launch_state,
            "rand_seed_ptr": rand_seed,
            "output_gate_ptr": output_gate_view,
            "norm_weight_ptr": norm_weight,
            "output_ptr": launch_output,
            "lower_bound": lower_bound,
            "scale": 1.0 / math.sqrt(key_dim),
            "norm_eps": norm_eps,
            "num_state_slots": launch_state.shape[0],
            "batch_size": batch_size,
            "stride_input_token": projected_qkv.stride(0),
            "stride_conv_weight_channel": conv_weight.stride(0),
            "stride_conv_weight_width": conv_weight.stride(1),
            "stride_conv_state_slot": launch_conv_state.stride(0),
            "stride_conv_state_channel": launch_conv_state.stride(1),
            "stride_conv_state_width": launch_conv_state.stride(2),
            "stride_raw_gate_token": raw_gate_view.stride(0),
            "stride_raw_gate_head": raw_gate_view.stride(1),
            "stride_raw_gate_channel": raw_gate_view.stride(2),
            "stride_raw_beta_token": raw_beta_view.stride(0),
            "stride_raw_beta_head": raw_beta_view.stride(1),
            "stride_state_index": launch_state_indices.stride(0),
            "stride_state_slot": launch_state.stride(0),
            "stride_state_head": launch_state.stride(1),
            "stride_state_value": launch_state.stride(2),
            "stride_state_key": launch_state.stride(3),
            "stride_output_gate_token": output_gate_view.stride(0),
            "stride_output_gate_head": output_gate_view.stride(1),
            "stride_output_gate_channel": output_gate_view.stride(2),
            "stride_output_token": launch_output.stride(1),
            "stride_output_head": launch_output.stride(2),
            "stride_output_channel": launch_output.stride(3),
            "NUM_HEADS": num_heads,
            "HEAD_DIM": key_dim,
            "BLOCK_DIM": block_dim,
            "STATE_IS_FP16": state.dtype == torch.float16,
            "PHILOX_ROUNDS": philox_rounds if rand_seed is not None else 0,
            "launch_pdl": use_pdl,
        }
        n_tiles = (launch_config.value_tile if launch_config else 16)
        num_z = triton.cdiv(key_dim, n_tiles)
        # cached, self-cleaning scratch (the finalizing tile writes the
        # slot back to zero) — no per-call allocation or memset.
        key = (projected_qkv.device.index, batch_size, num_heads)
        pair = _Z_SCRATCH.get(key)
        if pair is None:
            pair = (torch.zeros(batch_size * num_heads,
                                dtype=torch.float32,
                                device=projected_qkv.device),
                    torch.zeros(batch_size * num_heads,
                                dtype=torch.int32,
                                device=projected_qkv.device))
            _Z_SCRATCH[key] = pair
        kernel_args["sq_scratch_ptr"] = pair[0]
        kernel_args["counter_ptr"] = pair[1]
        grid = (batch_size, num_heads, num_z)
        _fused_kda_decode_kernel[grid](
            **kernel_args,
            VALUE_TILE=n_tiles,
            NUM_VALUE_TILES=num_z,
            num_warps=2,
            num_stages=1,
        )

    with custom_device_ctx(device_index):
        config_key = (
            device_index,
            batch_size,
            num_heads,
            key_dim,
            state.dtype,
            rand_seed is not None,
        )
        # grid.z variant: fixed tile config (16/2/1), no autotune pass.
        launch_config = _KDA_DECODE_CONFIG_CACHE.get(config_key)
        if launch_config is None:
            launch_config = _KDA_DECODE_FALLBACK_CONFIG

        launch(
            launch_config=launch_config,
            launch_conv_state=conv_state,
            launch_state_indices=state_indices,
            launch_state=state,
            launch_output=output,
            use_pdl=launch_pdl,
        )
    return output
