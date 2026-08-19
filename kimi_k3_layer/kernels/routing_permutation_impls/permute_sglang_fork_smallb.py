"""Baseten-fork small-B permutation kernel (migration reference, default off).

This is the current deployment branch's permutation (basetenlabs/sglang
fit-KDA `routing_graphpatch.py` kernel body, prebuilt unchanged). It is not
upstream open source, so it is excluded from the primary matrix.
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
    NUM_LOCAL_EXPERTS,
    TILE_TOKENS,
    TOP_K,
    Prepared,
    max_num_tiles,
    max_permuted_tokens,
)

NAME = "b10_fork_smallb_permute"
KIND = "permute"
PROVENANCE = {
    "repository": "https://github.com/basetenlabs/sglang",
    "commit": "65d6bb9dadee5716a61c02c3d09653784d925fae",
    "symbol": "kimi_k3/kernels/routing_graphpatch.py small-B kernel",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
    "role": "fork migration reference; not an upstream baseline",
}

MAX_BATCH = 128
_EXTENSION = None


def load() -> None:
    global _EXTENSION
    if _EXTENSION is None:
        from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.baselines import (
            sglang_permutation,
        )

        _EXTENSION = sglang_permutation.load()


def prepare(batch: int, device, *, ids=None, weights=None, **_) -> Prepared:
    assert _EXTENSION is not None, "call load() first"
    if batch > MAX_BATCH:
        raise NotImplementedError(f"small-B kernel supports B<={MAX_BATCH}")
    if ids is None:
        ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)
    tiles = max_num_tiles(batch)
    out = {
        "expanded_to_permuted": torch.empty(batch * TOP_K, dtype=torch.int32, device=device),
        "permuted_to_token": torch.empty(
            max_permuted_tokens(batch), dtype=torch.int32, device=device
        ),
        "tile_to_expert": torch.empty(tiles, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.empty(tiles, dtype=torch.int32, device=device),
        "padded_size": torch.empty(1, dtype=torch.int32, device=device),
        "num_tiles": torch.empty(1, dtype=torch.int32, device=device),
    }

    def run() -> None:
        _EXTENSION.launch_common(
            ids,
            batch,
            TOP_K,
            LOCAL_EXPERT_START,
            NUM_LOCAL_EXPERTS,
            TILE_TOKENS,
            out["tile_to_expert"],
            out["tile_to_mn_limit"],
            out["padded_size"],
            out["num_tiles"],
            out["expanded_to_permuted"],
            out["permuted_to_token"],
        )

    return Prepared(
        inputs={"ids": ids},
        outputs=out,
        run=run,
        padding_filled=frozenset({"permuted_to_token"}),
    )
