"""Unchanged upstream vLLM grouped_topk Tier<896,16> (prebuilt, compile-free)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import NUM_EXPERTS, TOP_K, Prepared

NAME = "vllm_grouped_topk"
KIND = "route"
PROVENANCE = {
    "repository": "https://github.com/vllm-project/vllm",
    "commit": "d3fafe0c27f9666a06675858738aaeab949da0f5",
    "symbol": "csrc/libtorch_stable/moe/grouped_topk_kernels.cu invokeNoAuxTc Tier<896,16>",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
    "artifact": "prebuilt cache (kernel_research vllm_grouped_topk baseline)",
}


def load() -> None:
    from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.baselines.vllm_grouped_topk import (
        backend,
    )

    backend.load()


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.baselines.vllm_grouped_topk import (
        backend,
    )

    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
    weights = torch.empty((batch, TOP_K), dtype=torch.float32, device=device)
    ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)

    def run() -> None:
        backend.run_into(logits, bias, weights, ids)

    return Prepared(
        inputs={"logits": logits, "bias": bias},
        outputs={"ids": ids, "weights": weights},
        run=run,
    )
