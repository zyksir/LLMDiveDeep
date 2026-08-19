# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Code based on the Kraken project, Copyright (c) Meta Platforms, Inc.
# (BSD-3-Clause)
"""Two-shot per-group INT8/FP8 AllReduce for NVIDIA NVLink GPUs.

Vendored from vLLM PR #39783 (commit c77c662a) with local imports and a
TP8 H200-derived fallback schedule. The Triton kernel body is unchanged.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from . import quantized_ar_barrier as ptx_utils

MAX_BLOCK_SIZE = 16384
DEFAULT_GROUP_SIZE = 256


@triton.jit
def _two_shot_quantized_allreduce_kernel(
    buf_ptrs_dev,
    signal_pad_ptrs,
    input_ptr,
    output_ptr,
    numel,
    num_groups_total: tl.constexpr,
    data_offset: tl.constexpr,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    QMAX: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_P2P: tl.constexpr = False,
):
    pid = tl.program_id(0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    ptrs = buf_ptrs_dev.to(tl.pointer_type(tl.uint64))
    my_buf = tl.load(ptrs + rank).to(tl.pointer_type(tl.uint8))

    qtype = tl.float8e4nv if USE_FP8 else tl.int8
    my_scales = my_buf.to(tl.pointer_type(tl.float32))
    my_data = (my_buf + data_offset).to(tl.pointer_type(qtype))
    my_data = tl.multiple_of(my_data, 16)

    group_ids = tl.arange(0, GROUPS_PER_BLOCK)
    elem_ids = tl.arange(0, GROUP_SIZE)

    # Phase 1: quantize local input into the symmetric buffer.
    num_compute_blocks = tl.cdiv(numel, BLOCK_SIZE)
    blk_id = pid
    while blk_id < num_compute_blocks:
        blk_off = blk_id * BLOCK_SIZE
        first_group = blk_id * GROUPS_PER_BLOCK
        offsets_2d = (
            blk_off
            + group_ids[:, None] * GROUP_SIZE
            + elem_ids[None, :]
        )
        mask_2d = offsets_2d < numel
        x = tl.load(
            input_ptr + offsets_2d, mask=mask_2d, other=0.0
        ).to(tl.float32)
        grp_amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)
        grp_scales = QMAX / grp_amax
        scale_mask = (first_group + group_ids) < num_groups_total
        tl.store(
            my_scales + first_group + group_ids,
            grp_scales,
            mask=scale_mask,
        )
        x_scaled = tl.clamp(x * grp_scales[:, None], -QMAX, QMAX)
        offsets_1d = blk_off + tl.arange(0, BLOCK_SIZE)
        mask_1d = offsets_1d < numel
        tl.store(
            my_data + offsets_1d,
            tl.reshape(x_scaled, [BLOCK_SIZE]).to(qtype),
            mask=mask_1d,
        )
        blk_id += tl.num_programs(0)

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Phase 2: reduce this rank's stripe and requantize the result.
    block_start = pid * stride_per_program
    while block_start < numel:
        stripe_off = block_start + rank * BLOCK_SIZE
        first_group = stripe_off // GROUP_SIZE
        offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc_2d = tl.zeros(
            [GROUPS_PER_BLOCK, GROUP_SIZE], dtype=tl.float32
        )

        for i in tl.static_range(world_size):
            peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
            peer_scales = tl.load(
                peer.to(tl.pointer_type(tl.float32))
                + first_group
                + group_ids,
                mask=(first_group + group_ids) < num_groups_total,
                other=1.0,
            )
            peer_data = (peer + data_offset).to(
                tl.pointer_type(qtype)
            )
            peer_data = tl.multiple_of(peer_data, 16)
            qvals = tl.load(peer_data + offsets, mask=mask, other=0.0)
            vals_2d = tl.reshape(
                qvals.to(tl.float32),
                [GROUPS_PER_BLOCK, GROUP_SIZE],
            )
            acc_2d += vals_2d / peer_scales[:, None]

        out_amax = tl.maximum(tl.max(tl.abs(acc_2d), axis=1), 1e-12)
        out_scales = QMAX / out_amax
        result_2d = tl.clamp(
            acc_2d * out_scales[:, None], -QMAX, QMAX
        )
        result_q = tl.reshape(result_2d, [BLOCK_SIZE]).to(qtype)

        if USE_P2P:
            tl.store(
                my_scales
                + num_groups_total
                + first_group
                + group_ids,
                out_scales,
                mask=(first_group + group_ids) < num_groups_total,
            )
            tl.store(my_data + offsets, result_q, mask=mask)
        else:
            for i in tl.static_range(world_size):
                peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
                tl.store(
                    peer.to(tl.pointer_type(tl.float32))
                    + num_groups_total
                    + first_group
                    + group_ids,
                    out_scales,
                    mask=(first_group + group_ids) < num_groups_total,
                )
                peer_data = (peer + data_offset).to(
                    tl.pointer_type(qtype)
                )
                peer_data = tl.multiple_of(peer_data, 16)
                tl.store(peer_data + offsets, result_q, mask=mask)

        result_bf16 = tl.reshape(acc_2d, [BLOCK_SIZE]).to(tl.bfloat16)
        tl.store(output_ptr + offsets, result_bf16, mask=mask)
        block_start += tl.num_programs(0) * stride_per_program

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Phase 3: dequantize every other rank's completed stripe.
    if USE_P2P:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = block_start + r * BLOCK_SIZE
                    first_group = stripe_off // GROUP_SIZE
                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < numel
                    reducer = tl.load(ptrs + r).to(
                        tl.pointer_type(tl.uint8)
                    )
                    reducer_data = (reducer + data_offset).to(
                        tl.pointer_type(qtype)
                    )
                    reducer_data = tl.multiple_of(reducer_data, 16)
                    qvals = tl.load(
                        reducer_data + offsets, mask=mask, other=0.0
                    )
                    vals_2d = tl.reshape(
                        qvals.to(tl.float32),
                        [GROUPS_PER_BLOCK, GROUP_SIZE],
                    )
                    r_out_scales = tl.load(
                        reducer.to(tl.pointer_type(tl.float32))
                        + num_groups_total
                        + first_group
                        + group_ids,
                        mask=(
                            first_group + group_ids
                        ) < num_groups_total,
                        other=1.0,
                    )
                    result_2d = vals_2d / r_out_scales[:, None]
                    tl.store(
                        output_ptr + offsets,
                        tl.reshape(
                            result_2d, [BLOCK_SIZE]
                        ).to(tl.bfloat16),
                        mask=mask,
                    )
            block_start += tl.num_programs(0) * stride_per_program
    else:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = block_start + r * BLOCK_SIZE
                    first_group = stripe_off // GROUP_SIZE
                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < numel
                    qvals = tl.load(
                        my_data + offsets, mask=mask, other=0.0
                    )
                    vals_2d = tl.reshape(
                        qvals.to(tl.float32),
                        [GROUPS_PER_BLOCK, GROUP_SIZE],
                    )
                    r_out_scales = tl.load(
                        my_scales
                        + num_groups_total
                        + first_group
                        + group_ids,
                        mask=(
                            first_group + group_ids
                        ) < num_groups_total,
                        other=1.0,
                    )
                    result_2d = vals_2d / r_out_scales[:, None]
                    tl.store(
                        output_ptr + offsets,
                        tl.reshape(
                            result_2d, [BLOCK_SIZE]
                        ).to(tl.bfloat16),
                        mask=mask,
                    )
            block_start += tl.num_programs(0) * stride_per_program


def compute_layout(
    numel: int,
    group_size: int,
    max_block_size: int = MAX_BLOCK_SIZE,
) -> tuple[int, int, int]:
    num_groups_total = triton.cdiv(numel, group_size)
    num_out_groups = num_groups_total + max_block_size // group_size
    num_scale_slots = num_groups_total + num_out_groups
    scale_header = num_scale_slots * 4
    data_offset = ((scale_header + 15) // 16) * 16
    return num_groups_total, data_offset, data_offset + numel


class QuantizedAllReduceState:
    """One bounded symmetric buffer reusable by all supported token counts."""

    def __init__(
        self,
        max_numel: int,
        group_size: int,
        device: torch.device,
        group,
    ) -> None:
        self.world_size = dist.get_world_size(group)
        self.max_numel = max_numel
        self.group_size = group_size
        (
            self.num_groups_total,
            self.data_offset,
            self.packed_size,
        ) = compute_layout(max_numel, group_size)
        self.buf = symm_mem.empty(
            (self.packed_size,), dtype=torch.uint8, device=device
        )
        self.hdl = symm_mem.rendezvous(self.buf, group=group)

    def get_metadata(self, numel: int) -> tuple[int, int]:
        if numel == self.max_numel:
            return self.num_groups_total, self.data_offset
        return compute_layout(numel, self.group_size)[:2]


def _tp8_config(numel: int) -> tuple[int, int, bool]:
    """H200 TP8 gs256 schedule from vLLM; B200 is benchmarked explicitly."""
    if numel < 2_097_152:
        return 1024, 4, False
    if numel < 4_194_304:
        return 2048, 8, False
    if numel < 8_388_608:
        return 4096, 16, False
    if numel < 16_777_216:
        return 4096, 16, False
    if numel < 134_217_728:
        return 4096, 16, True
    return 16384, 16, True


def two_shot_quantized_allreduce(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    state: QuantizedAllReduceState,
    *,
    use_fp8: bool,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Run the vLLM/Kraken two-shot algorithm into preallocated output."""
    if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_contiguous():
        raise ValueError("quantized AllReduce requires contiguous BF16 input")
    numel = input_tensor.numel()
    if numel % 8:
        raise ValueError("quantized AllReduce numel must be divisible by 8")
    if numel > state.max_numel:
        raise ValueError("quantized AllReduce state is too small")
    ws = state.world_size
    block_size, num_warps, use_p2p = _tp8_config(numel)
    if block_size * ws > numel:
        raise ValueError(
            f"quantized AllReduce needs at least {block_size * ws} elements"
        )

    groups_per_block = block_size // group_size
    qmax = 448.0 if use_fp8 else 127.0
    ngt, doff = state.get_metadata(numel)
    stride = block_size * ws
    num_blocks = min(triton.cdiv(numel, stride), 132)
    flat_out = output.view(-1)[:numel]
    _two_shot_quantized_allreduce_kernel[(num_blocks,)](
        state.hdl.buffer_ptrs_dev,
        state.hdl.signal_pad_ptrs_dev,
        input_tensor,
        flat_out,
        numel=numel,
        num_groups_total=ngt,
        data_offset=doff,
        stride_per_program=stride,
        rank=state.hdl.rank,
        world_size=ws,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        QMAX=qmax,
        USE_FP8=use_fp8,
        USE_P2P=use_p2p,
        num_warps=num_warps,
    )
    return flat_out.view_as(input_tensor)
