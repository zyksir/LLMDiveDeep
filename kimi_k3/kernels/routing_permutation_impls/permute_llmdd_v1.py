"""LLMDiveDeep candidate: single-CTA permutation from IDs with parallel scan.

Shares the compiled extension with fused_llmdd_v1 (one build, shape-generic).
Pairs with the unchanged SGLang radix route as the composed region candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3.kernels.routing_permutation import (
    TOP_K,
    Prepared,
    max_num_tiles,
    max_permuted_tokens,
)
from kimi_k3.kernels.routing_permutation_impls import fused_llmdd_v1

NAME = "b10_permute_v1"
KIND = "permute"
PROVENANCE = {
    "repository": "LLMDiveDeep (candidate, not a baseline)",
    "symbol": "fused_llmdd_v1.cu::permuteFromIdsKernel",
    "role": "candidate",
    "idea": "one 512-thread CTA: shared histogram + one warp-level "
    "two-quantity exclusive scan + parallel tiles/fill/scatter; removes the "
    "multi-phase cluster pipeline moe_sort pays and the fork kernel's "
    "narrower parallelism",
}


def load() -> None:
    fused_llmdd_v1.load()


def prepare(batch: int, device, *, ids=None, weights=None, use_pdl=False, **_) -> Prepared:
    module = fused_llmdd_v1.module()
    if ids is None:
        ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)
    tiles = max_num_tiles(batch)
    out = {
        "expanded_to_permuted": torch.empty(batch * TOP_K, dtype=torch.int32, device=device),
        "permuted_to_expanded": torch.empty(
            max_permuted_tokens(batch), dtype=torch.int32, device=device
        ),
        "permuted_to_token": torch.empty(
            max_permuted_tokens(batch), dtype=torch.int32, device=device
        ),
        "tile_to_expert": torch.empty(tiles, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.empty(tiles, dtype=torch.int32, device=device),
        "padded_size": torch.empty(1, dtype=torch.int32, device=device),
        "num_tiles": torch.empty(1, dtype=torch.int32, device=device),
    }

    def run() -> None:
        module.permute_from_ids(
            ids, out["expanded_to_permuted"], out["permuted_to_expanded"],
            out["permuted_to_token"], out["tile_to_expert"],
            out["tile_to_mn_limit"], out["padded_size"], out["num_tiles"],
            use_pdl,
        )

    return Prepared(
        inputs={"ids": ids},
        outputs=out,
        run=run,
        padding_filled=frozenset({"permuted_to_expanded"}),
    )
