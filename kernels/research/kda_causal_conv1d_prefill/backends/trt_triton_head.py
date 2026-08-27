# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""One-launch TRT Triton baseline loaded exactly from git HEAD."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

import torch

from common import PAD_SLOT_ID, Prepared, Problem

TRT_REPOSITORY = Path("/node-storage/trt-llm")
SOURCE_PATH = "tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py"
EXPECTED_SHA256 = "68860408e57d076765d1e0be13cc87b9a3811ea9424de670d1c39821e5332432"


def _load_head_module() -> ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"HEAD:{SOURCE_PATH}"],
        cwd=TRT_REPOSITORY,
    )
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"git HEAD Triton source changed: expected {EXPECTED_SHA256}, got {digest}"
        )
    cache_path = Path(tempfile.gettempdir()) / f"trt_causal_conv1d_triton_{digest}.py"
    if not cache_path.exists() or cache_path.read_bytes() != source:
        cache_path.write_bytes(source)
    module_name = f"_kda_trt_triton_head_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, cache_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build module spec for {cache_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TRTTritonHeadBackend:
    name = "trt_triton_git_head"
    source = f"git HEAD:{SOURCE_PATH}@{EXPECTED_SHA256}"

    def __init__(self) -> None:
        import tensorrt_llm  # noqa: F401

        self._module = _load_head_module()

    def supports(self, problem: Problem) -> tuple[bool, str]:
        if bool((problem.cache_indices == PAD_SLOT_ID).any().item()):
            return False, "git-HEAD Triton leaves padded output uninitialized"
        return True, ""

    def _pipeline(self, problem: Problem) -> torch.Tensor:
        # Metadata transpose only: physical storage remains token-major.
        channel_view = problem.projected[: problem.shape.tokens].transpose(0, 1)
        output_view = self._module.causal_conv1d_fn(
            channel_view,
            problem.weight,
            problem.bias,
            problem.conv_states,
            problem.query_start_loc,
            list(problem.sequence_lengths),
            cache_indices=problem.cache_indices,
            has_initial_state=problem.has_initial_state,
            activation=problem.shape.activation,
            pad_slot_id=PAD_SLOT_ID,
        )
        output = output_view.transpose(0, 1)
        if not output.is_contiguous():
            raise RuntimeError("git-HEAD Triton output did not preserve token-major storage")
        return output

    def prepare(self, problem: Problem) -> Prepared:
        latest: list[torch.Tensor | None] = [None]

        def run() -> torch.Tensor:
            latest[0] = self._pipeline(problem)
            return latest[0]

        return Prepared(
            run=run,
            output=lambda: latest[0] if latest[0] is not None else run(),
            state=lambda: problem.conv_states,
            launch_count=1,
        )


Backend = TRTTritonHeadBackend
