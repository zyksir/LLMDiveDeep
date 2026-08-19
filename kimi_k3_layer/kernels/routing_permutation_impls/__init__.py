"""Registry of routing/permutation implementations, one module per file.

Each module exposes: NAME, KIND ("route" | "permute" | "fused"), PROVENANCE,
load() -> None, and prepare(batch, device, **bound_buffers) -> Prepared.
"""

from __future__ import annotations

import importlib

# name -> module path; default matrix is open-source/native only.
DEFAULT = {
    "sgl_radix": ".route_sglang_radix",
    "vllm_grouped_topk": ".route_vllm_grouped_topk",
    "trt_moe_sort": ".permute_trtllm_moe_sort",
    "trt_routing_custom": ".fused_trtllm_routing_custom",
}

# Supplemental impls that must run in their OWN process: importing the
# tensorrt_llm wheel loads routing template symbols that clash with the
# flashinfer-built modules (same weak symbols, different source versions) and
# corrupts moe_sort/routingCustom dispatch. Run via --impls trt_noaux.
SUPPLEMENTAL = {
    "trt_noaux": ".route_trtllm_noaux",
}

# Baseten-fork migration reference; excluded unless explicitly requested.
FORK = {
    "b10_fork_smallb_permute": ".permute_sglang_fork_smallb",
}

# LLMDiveDeep candidates (never baselines); include via --include-candidates.
CANDIDATE = {
    "b10_fused_v1": ".fused_llmdd_v1",
    "b10_fused_v2": ".fused_llmdd_v2",
    "b10_permute_v1": ".permute_llmdd_v1",
    "b10_permute_v1_pdl": ".permute_llmdd_v1_pdl",
}


def load_impl(name: str):
    table = {**DEFAULT, **SUPPLEMENTAL, **FORK, **CANDIDATE}
    module = importlib.import_module(table[name], __package__)
    assert module.NAME == name, (module.NAME, name)
    return module
