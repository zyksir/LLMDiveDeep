"""LLMDiveDeep candidate v2: B route CTAs + last-CTA permutation epilogue.

One kernel: each CTA routes one token with block-scope byte-radix top-16
(4 keys/lane over 256 threads), the last-arriving CTA (device counter) runs
the permutation. Removes the second launch and keeps route parallelism.
Shares the compiled extension with fused_llmdd_v1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import (
    NUM_EXPERTS,
    TOP_K,
    Prepared,
    max_num_tiles,
    max_permuted_tokens,
)
from kimi_k3_layer.kernels.routing_permutation_impls import fused_llmdd_v1

NAME = "b10_fused_v2"
KIND = "fused"
PROVENANCE = {
    "repository": "LLMDiveDeep (candidate, not a baseline)",
    "symbol": "fused_llmdd_v1.cu::fusedV2Kernel",
    "role": "candidate",
    "idea": "block-radix route per token CTA + last-CTA shared-memory "
    "permutation epilogue; removes the second launch and the exposed "
    "permutation critical path of the two-kernel composition",
}

MAX_BATCH = 128
WEIGHTS_DTYPE = torch.float32


def load() -> None:
    fused_llmdd_v1.load()


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    module = fused_llmdd_v1.module()
    if batch > MAX_BATCH:
        raise NotImplementedError(f"candidate supports B<={MAX_BATCH}")
    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    tiles = max_num_tiles(batch)
    permuted = max_permuted_tokens(batch)
    out = {
        "ids": torch.empty((batch, TOP_K), dtype=torch.int32, device=device),
        "weights": torch.empty((batch, TOP_K), dtype=torch.float32, device=device),
        "expanded_to_permuted": torch.empty(batch * TOP_K, dtype=torch.int32, device=device),
        "permuted_to_expanded": torch.empty(permuted, dtype=torch.int32, device=device),
        "permuted_to_token": torch.empty(permuted, dtype=torch.int32, device=device),
        "tile_to_expert": torch.empty(tiles, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.empty(tiles, dtype=torch.int32, device=device),
        "padded_size": torch.empty(1, dtype=torch.int32, device=device),
        "num_tiles": torch.empty(1, dtype=torch.int32, device=device),
    }

    def run(ablate: int = 0) -> None:
        module.fused_route_permute_v2(
            logits, bias, counter, out["ids"], out["weights"],
            out["expanded_to_permuted"], out["permuted_to_expanded"],
            out["permuted_to_token"], out["tile_to_expert"],
            out["tile_to_mn_limit"], out["padded_size"], out["num_tiles"],
            ablate,
        )

    return Prepared(
        inputs={"logits": logits, "bias": bias},
        outputs=out,
        run=run,
        padding_filled=frozenset({"permuted_to_expanded"}),
    )
