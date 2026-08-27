# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unchanged TensorRT native transpose + CUDA + materialize pipeline."""

from __future__ import annotations

import torch

from common import PAD_SLOT_ID, Prepared, Problem


class TRTNativeBackend:
    name = "trt_native_pipeline"
    source = (
        "TensorRT-LLM extract_transpose_prefill_slice + "
        "torch.ops.trtllm.causal_conv1d_fwd"
    )

    def __init__(self) -> None:
        # Imports register the extension and expose the unchanged transpose helper.
        import tensorrt_llm  # noqa: F401
        from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
            extract_transpose_prefill_slice,
        )

        self._extract = extract_transpose_prefill_slice

    def supports(self, problem: Problem) -> tuple[bool, str]:
        del problem
        return True, ""

    def _pipeline(self, problem: Problem) -> torch.Tensor:
        # BASELINE BODY: keep this equivalent to the production KDA path.
        channel_major = self._extract(
            problem.projected,
            problem.shape.tokens,
            0,
            problem.shape.channels,
        )
        torch.ops.trtllm.causal_conv1d_fwd(
            channel_major,
            problem.weight,
            problem.bias,
            problem.conv_states,
            problem.query_start_loc,
            problem.cache_indices,
            problem.has_initial_state,
            problem.shape.activation in ("silu", "swish"),
            PAD_SLOT_ID,
        )
        return channel_major.transpose(0, 1).contiguous()

    def prepare(self, problem: Problem) -> Prepared:
        latest: list[torch.Tensor | None] = [None]

        def run() -> torch.Tensor:
            latest[0] = self._pipeline(problem)
            return latest[0]

        return Prepared(
            run=run,
            output=lambda: latest[0] if latest[0] is not None else run(),
            state=lambda: problem.conv_states,
            launch_count=3,
        )


Backend = TRTNativeBackend
