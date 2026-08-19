"""Unchanged TRT-LLM post-top-k routing pipeline via FlashInfer moe_utils.

This is the production permutation machinery behind the trtllm-gen /
cute-DSL MoE paths (flashinfer `moe_sort` -> `routingDeepSeek::run` with
precomputed top-k IDs; scoring is skipped, so the pipeline is
policy-independent). The compiled module is FlashInfer's own JIT artifact;
this adapter only binds preallocated buffers and calls the exported entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import (
    LOCAL_EXPERT_START,
    NUM_EXPERTS,
    NUM_LOCAL_EXPERTS,
    TILE_TOKENS,
    TOP_K,
    Prepared,
    max_num_tiles,
    max_permuted_tokens,
)

NAME = "trt_moe_sort"
KIND = "permute"
PROVENANCE = {
    "repository": "installed flashinfer wheel (bundled TensorRT-LLM sources)",
    "symbol": "flashinfer_moe_sort -> moe::dev::routing::routingDeepSeek::run (post-top-k)",
    "source": "flashinfer/data/csrc/moe_utils_binding.cu + trtllm_backend routing kernels",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
}

_MODULE = None


def load() -> None:
    global _MODULE
    if _MODULE is None:
        import flashinfer
        from flashinfer.fused_moe.cute_dsl.moe_utils import _get_moe_utils_module

        PROVENANCE["flashinfer_version"] = flashinfer.__version__
        _MODULE = _get_moe_utils_module()


def prepare(batch: int, device, *, ids=None, weights=None, **_) -> Prepared:
    assert _MODULE is not None, "call load() first"
    if ids is None:
        ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)
    if weights is None:
        weights = torch.empty((batch, TOP_K), dtype=torch.float32, device=device)
    tiles = max_num_tiles(batch)
    permuted = max_permuted_tokens(batch)
    out = {
        "tile_to_expert": torch.empty(tiles, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.empty(tiles, dtype=torch.int32, device=device),
        "expanded_to_permuted": torch.empty(batch * TOP_K, dtype=torch.int32, device=device),
        "permuted_to_expanded": torch.empty(permuted, dtype=torch.int32, device=device),
        "padded_size": torch.empty(1, dtype=torch.int32, device=device),
        "num_tiles": torch.empty(1, dtype=torch.int32, device=device),
    }
    expert_counts = torch.empty(2 * NUM_EXPERTS, dtype=torch.int32, device=device)

    def run() -> None:
        _MODULE.flashinfer_moe_sort(
            ids.data_ptr(),
            weights.data_ptr(),
            batch,
            NUM_EXPERTS,
            TOP_K,
            LOCAL_EXPERT_START,
            NUM_LOCAL_EXPERTS,
            TILE_TOKENS,
            False,  # use_pdl
            out["tile_to_expert"].data_ptr(),
            out["tile_to_mn_limit"].data_ptr(),
            out["expanded_to_permuted"].data_ptr(),
            out["permuted_to_expanded"].data_ptr(),
            out["padded_size"].data_ptr(),
            out["num_tiles"].data_ptr(),
            expert_counts.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )

    return Prepared(inputs={"ids": ids, "weights": weights}, outputs=out, run=run)
