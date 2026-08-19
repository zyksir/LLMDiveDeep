"""TensorRT-LLM public noaux_tc_op (supplemental route-only reference).

Not the production Kimi-K3 front. The op allocates its outputs internally, so
this adapter charges one copy into stable buffers inside the timed region.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import NUM_EXPERTS, TOP_K, Prepared

NAME = "trt_noaux"
KIND = "route"
PROVENANCE = {
    "repository": "installed tensorrt_llm wheel",
    "symbol": "torch.ops.trtllm.noaux_tc_op",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
    "role": "supplemental route-only; not the production fused front",
}


def load() -> None:
    import tensorrt_llm  # noqa: F401 - registers torch.ops.trtllm

    if not hasattr(torch.ops.trtllm, "noaux_tc_op"):
        raise RuntimeError("torch.ops.trtllm.noaux_tc_op is unavailable")


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
    weights = torch.empty((batch, TOP_K), dtype=torch.float32, device=device)
    ids = torch.empty((batch, TOP_K), dtype=torch.int32, device=device)

    def run() -> None:
        w, i = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
        weights.copy_(w)
        ids.copy_(i)

    return Prepared(
        inputs={"logits": logits, "bias": bias},
        outputs={"ids": ids, "weights": weights},
        run=run,
        charged_note="op allocates outputs; copy into stable buffers is timed",
    )
