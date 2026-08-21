"""LLMDiveDeep candidate: fused decode route+permute in one CTA (B<=128).

See fused_llmdd_v1.cu for the hypothesis and semantics. Compiled once via
torch cpp_extension (shape-generic over B; no per-shape JIT), cached in
TORCH_EXTENSIONS_DIR.
"""

from __future__ import annotations

import os
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

NAME = "b10_fused_v1"
KIND = "fused"
PROVENANCE = {
    "repository": "LLMDiveDeep (candidate, not a baseline)",
    "symbol": "fused_llmdd_v1.cu::fusedRoutePermuteKernel",
    "role": "candidate",
    "idea": "one-CTA warp-radix top-16 + shared-memory histogram/scan/maps; "
    "removes one launch and the IDs global round-trip vs the unfused region",
}

MAX_BATCH = 128
WEIGHTS_DTYPE = torch.float32
_MODULE = None


def load() -> None:
    """Build/load the probe extension for THIS device's arch.

    Two arch hazards, both of which bit on a GB300 (the module built for
    sm_100a, loaded cleanly, then killed the CUDA context at its first launch
    with "no kernel image is available" -- and because the caller only guards
    load() in a try/except, that surfaced later as an unrelated failure in
    torch.randn):

    * the arch list has to come from the device, not a literal;
    * ``cpp_extension`` keys its build directory on name + source hash and NOT
      on the target arch, so the name has to carry the arch or a B200 build is
      silently reused on a B300 -- that is what ``arch.ext_name`` is for.

    ``TORCH_CUDA_ARCH_LIST`` is restored afterwards; leaving it set leaks this
    module's target onto every later build in the process.
    """
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load as load_ext

        from common import arch

        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch.torch_arch_list()
        try:
            _MODULE = load_ext(
                name=arch.ext_name("llmdd_fused_route_permute_v1"),
                sources=[str(Path(__file__).with_suffix(".cu"))],
                extra_cuda_cflags=["-O3", *arch.nvcc_gencode()],
                verbose=False,
            )
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def module():
    assert _MODULE is not None, "call load() first"
    return _MODULE


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    assert _MODULE is not None, "call load() first"
    if batch > MAX_BATCH:
        raise NotImplementedError(f"candidate supports B<={MAX_BATCH}")
    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
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
        _MODULE.fused_route_permute(
            logits, bias, out["ids"], out["weights"],
            out["expanded_to_permuted"], out["permuted_to_expanded"],
            out["permuted_to_token"], out["tile_to_expert"],
            out["tile_to_mn_limit"], out["padded_size"], out["num_tiles"],
            ablate,
        )

    return Prepared(
        inputs={"logits": logits, "bias": bias},
        outputs=out,
        run=run,
        padding_filled=frozenset({"permuted_to_expanded", "permuted_to_token"}),
    )
