"""Unchanged TensorRT-LLM fused Kimi-K3 routing front (routingCustom::run).

Raw FP32 logits + bias -> top-k weights, packed score/idx, and the complete
permutation ABI in one fused region (single CTA for B<=4, one cluster for
B<=256, multi-kernel above). Kernel sources are the FlashInfer-bundled
TensorRT-LLM files, compiled unchanged by FlashInfer's own JIT together with
the thin launcher in `trtllm_routing_custom_binding.cu`. The Data
configuration mirrors the production DeepSeekV3 no-groups branch.

Output note: in raw-scores mode the kernel emits no INT32 ID tensor; the
`topk_packed` buffer is internal scratch whose layout is not a consumable
ABI. The production consumer reads weights + maps/descriptors only, so
correctness uses the order-insensitive `check_fused_region`.
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

NAME = "trt_routing_custom"
KIND = "fused"
PROVENANCE = {
    "repository": "installed flashinfer wheel (bundled TensorRT-LLM sources)",
    "symbol": "moe::dev::routing::routingCustom::run (SigmoidBias + ScaledSumNormalize)",
    "source": "flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
    "license": "Apache-2.0",
    "kernel_body_unchanged": True,
    "adapter": "trtllm_routing_custom_binding.cu (launcher only, untimed setup)",
}

WEIGHTS_DTYPE = torch.float32  # set to torch.bfloat16 for the production output mode

_MODULE = None


def load() -> None:
    global _MODULE
    if _MODULE is not None:
        return
    import flashinfer
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.moe_utils import gen_moe_utils_module

    # Ensure the trtllm-gen BMM export headers/symlinks exist (same setup the
    # bundled moe_utils module performs); its own build is cached.
    gen_moe_utils_module()

    csrc = jit_env.FLASHINFER_CSRC_DIR
    flags = [
        "-DTLLM_GEN_EXPORT_INTERFACE",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
    ] + current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10])
    spec = gen_jit_spec(
        "llmdd_trtllm_routing_custom",
        [
            Path(__file__).with_name("trtllm_routing_custom_binding.cu"),
            csrc / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
            csrc / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
        ],
        extra_cuda_cflags=flags,
        extra_include_paths=[
            csrc,
            csrc / "nv_internal",
            csrc / "nv_internal" / "include",
            jit_env.FLASHINFER_CUBIN_DIR,  # trtllmGen_bmm_export headers (symlinked)
        ],
    )
    PROVENANCE["flashinfer_version"] = flashinfer.__version__
    _MODULE = spec.build_and_load()


def prepare(batch: int, device, *, logits=None, bias=None, **_) -> Prepared:
    assert _MODULE is not None, "call load() first"
    if logits is None:
        logits = torch.empty((batch, NUM_EXPERTS), dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.empty((NUM_EXPERTS,), dtype=torch.float32, device=device)
    tiles = max_num_tiles(batch)
    permuted = max_permuted_tokens(batch)
    packed_bytes = 4 if WEIGHTS_DTYPE == torch.bfloat16 else 8
    out = {
        "weights": torch.empty((batch, TOP_K), dtype=WEIGHTS_DTYPE, device=device),
        "topk_packed": torch.empty(
            batch * TOP_K * packed_bytes, dtype=torch.uint8, device=device
        ),
        "expanded_to_permuted": torch.empty(batch * TOP_K, dtype=torch.int32, device=device),
        "permuted_to_expanded": torch.empty(permuted, dtype=torch.int32, device=device),
        "permuted_to_token": torch.empty(permuted, dtype=torch.int32, device=device),
        "tile_to_expert": torch.empty(tiles, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.empty(tiles, dtype=torch.int32, device=device),
        "padded_size": torch.empty(1, dtype=torch.int32, device=device),
        "num_tiles": torch.empty(1, dtype=torch.int32, device=device),
    }
    expert_counts = torch.empty(2 * NUM_EXPERTS, dtype=torch.int32, device=device)

    def run() -> None:
        _MODULE.llmdd_routing_custom_from_scores(
            logits.data_ptr(),
            bias.data_ptr(),
            out["weights"].data_ptr(),
            out["topk_packed"].data_ptr(),
            expert_counts.data_ptr(),
            out["padded_size"].data_ptr(),
            out["expanded_to_permuted"].data_ptr(),
            out["permuted_to_expanded"].data_ptr(),
            out["permuted_to_token"].data_ptr(),
            out["tile_to_expert"].data_ptr(),
            out["tile_to_mn_limit"].data_ptr(),
            out["num_tiles"].data_ptr(),
            batch,
            NUM_EXPERTS,
            TOP_K,
            TILE_TOKENS,
            LOCAL_EXPERT_START,
            NUM_LOCAL_EXPERTS,
            1.0,  # route_scale
            True,  # input_is_fp32
            WEIGHTS_DTYPE == torch.float32,
            False,  # use_pdl
            torch.cuda.current_stream().cuda_stream,
        )

    return Prepared(inputs={"logits": logits, "bias": bias}, outputs=out, run=run)
