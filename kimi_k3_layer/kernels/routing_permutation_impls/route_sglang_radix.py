"""Unchanged upstream SGLang RouteRadixKernel (prebuilt, compile-free load)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import NUM_EXPERTS, TOP_K, Prepared

NAME = "sgl_radix"
KIND = "route"
PROVENANCE = {
    "repository": "https://github.com/sgl-project/sglang",
    "commit": "7b5410c999be60489c71636443ea036596fe1432",
    "symbol": "moe/route_radix.cuh::RouteRadixKernel via moe_route_radix",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
    "artifact": "prebuilt cache (kernel_research sglang_radix baseline)",
}


def load() -> None:
    from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.sglang_radix_prebuilt import (
        load_sglang_radix,
    )

    load_sglang_radix()


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.sglang_radix_prebuilt import (
        route_radix_into,
    )

    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
    weights = torch.empty((batch, TOP_K), dtype=torch.float32, device=device)
    ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)

    def run() -> None:
        route_radix_into(logits, bias, weights, ids, sorted=False)

    return Prepared(
        inputs={"logits": logits, "bias": bias},
        outputs={"ids": ids, "weights": weights},
        run=run,
    )
