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

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.causal_conv1d_prefill import causal_conv1d_prefill
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import extract_transpose_prefill_slice

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for causal-conv kernels",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize(
    "seq_lens_cpu",
    [
        [1],
        [17, 8, 3],
        [128, 64, 32, 16],
    ],
)
def test_causal_conv1d_prefill_matches_layout_copy_pipeline(
    dtype: torch.dtype,
    width: int,
    seq_lens_cpu: list[int],
) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_prefill_tokens = sum(seq_lens_cpu)
    dim = 384
    padding = 16

    projected_storage = torch.randn(
        num_prefill_tokens + 5,
        dim + padding,
        dtype=dtype,
        device=device,
    )
    projected = projected_storage[:, :dim]
    assert not projected.is_contiguous()

    weight = torch.randn(dim, width, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.arange(len(seq_lens_cpu), dtype=torch.int32, device=device)
    has_initial_state = torch.tensor(
        [(index % 2) == 0 for index in range(len(seq_lens_cpu))],
        dtype=torch.bool,
        device=device,
    )
    initial_states = torch.randn(
        len(seq_lens_cpu) + 2,
        dim,
        width - 1,
        dtype=dtype,
        device=device,
    )
    expected_states = initial_states.clone()
    actual_states = initial_states.clone()

    expected = extract_transpose_prefill_slice(
        projected,
        num_prefill_tokens,
        0,
        dim,
    )
    expected = causal_conv1d_fn(
        expected,
        weight,
        bias,
        conv_states=expected_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    ).transpose(0, 1)

    actual = causal_conv1d_prefill(
        projected,
        num_prefill_tokens,
        weight,
        bias=bias,
        conv_states=actual_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    assert actual.shape == (num_prefill_tokens, dim)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)


def test_causal_conv1d_prefill_skips_padded_sequences() -> None:
    """Padded sequences produce uninitialized output rows (documented contract);
    only non-padded rows and conv states are compared against the native path."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_lens_cpu = [7, 5, 3]
    num_prefill_tokens = sum(seq_lens_cpu)
    dim = 384
    width = 4

    projected = torch.randn(num_prefill_tokens, dim, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device) * 0.1
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.tensor([1, PAD_SLOT_ID, 0], dtype=torch.int32, device=device)
    has_initial_state = torch.ones(len(seq_lens_cpu), dtype=torch.bool, device=device)
    initial_states = torch.randn(2, dim, width - 1, dtype=dtype, device=device)
    expected_states = initial_states.clone()
    actual_states = initial_states.clone()

    expected = extract_transpose_prefill_slice(
        projected,
        num_prefill_tokens,
        0,
        dim,
    )
    expected = causal_conv1d_fn(
        expected,
        weight,
        conv_states=expected_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    ).transpose(0, 1)

    actual = causal_conv1d_prefill(
        projected,
        num_prefill_tokens,
        weight,
        conv_states=actual_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    boundaries = [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()]
    for seq_idx, slot in enumerate(cache_indices.tolist()):
        if slot == PAD_SLOT_ID:
            continue
        start, end = boundaries[seq_idx], boundaries[seq_idx + 1]
        torch.testing.assert_close(
            actual[start:end], expected[start:end], rtol=1e-2, atol=1e-2
        )
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)


@pytest.mark.parametrize("qkv_group_tokens_extra", [0, 5])
def test_causal_conv1d_prefill_grouped_output(qkv_group_tokens_extra: int) -> None:
    """Grouped output [G_planes, group_tokens, G] must match the token-major
    output regrouped, writing only the leading num_prefill_tokens rows."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_lens_cpu = [17, 8, 7]
    num_prefill_tokens = sum(seq_lens_cpu)
    qkv_group_size = 256
    dim = 3 * qkv_group_size
    width = 4

    projected = torch.randn(num_prefill_tokens, dim, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device) * 0.1
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.arange(len(seq_lens_cpu), dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(len(seq_lens_cpu), dtype=torch.bool, device=device)
    conv_states = torch.zeros(
        len(seq_lens_cpu), dim, width - 1, dtype=dtype, device=device
    )

    common = dict(
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )
    flat = causal_conv1d_prefill(
        projected,
        num_prefill_tokens,
        weight,
        conv_states=conv_states.clone(),
        **common,
    )
    grouped_states = conv_states.clone()
    grouped = causal_conv1d_prefill(
        projected,
        num_prefill_tokens,
        weight,
        conv_states=grouped_states,
        qkv_group_size=qkv_group_size,
        qkv_group_tokens=num_prefill_tokens + qkv_group_tokens_extra,
        **common,
    )

    num_groups = dim // qkv_group_size
    assert grouped.shape == (
        num_groups,
        num_prefill_tokens + qkv_group_tokens_extra,
        qkv_group_size,
    )
    assert grouped.is_contiguous()
    expected_grouped = (
        flat.view(num_prefill_tokens, num_groups, qkv_group_size)
        .permute(1, 0, 2)
        .contiguous()
    )
    torch.testing.assert_close(
        grouped[:, :num_prefill_tokens], expected_grouped, rtol=0, atol=0
    )
