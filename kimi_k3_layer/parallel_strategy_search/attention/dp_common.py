"""Shared pieces for the DP-attention variants bench (MLA-only, Kimi-K3 shape).

Everything TRT-LLM is IMPORTED unchanged (KimiMLAAttention/MLA, KVCacheManager,
TRTLLM attention metadata, helix helpers). This module only builds instances,
fills deterministic weights (sliced consistently for any head sharding), and
prepares per-variant cache/metadata runtimes.

Weight consistency contract: every module instance, whatever its sharding,
holds slices of the SAME logical per-name tensors (seeded generator per name),
and the decode absorption weights (k_b_proj_trans / v_b_proj) are DERIVED from
the logical kv_b_proj exactly like TRT-LLM's DeepseekV3 loader
(modeling_deepseekv3.py load_kv_b_proj_and_k_b_proj_trans), so the context and
decode paths compute the same math and all variants are cross-comparable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import torch

# Kimi-K3 MLA dims (checkpoint config).
HIDDEN_SIZE = 7168
NUM_HEADS = 96
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
RMS_NORM_EPS = 1e-5
TOKENS_PER_BLOCK = 32
WEIGHT_STD = 0.02
WEIGHT_SEED = 20260829


def build_model_config(mapping: Any) -> Any:
    """ModelConfig[KimiLinearConfig] for a single MLA layer (layer_idx 0)."""
    from tensorrt_llm._torch.configs import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig

    pretrained_config = KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        vocab_size=163840,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=33792,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=1 << 20,
        rms_norm_eps=RMS_NORM_EPS,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "full_attn_layers": [1],  # 1-based: layer 0 is the MLA layer
            "head_dim": 128,
            "kda_layers": [],
            "num_heads": NUM_HEADS,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=QuantConfig(),
        allreduce_strategy="AUTO",
    )


def _named_generator(name: str, device: torch.device) -> torch.Generator:
    digest = hashlib.sha256(name.encode()).digest()
    seed = WEIGHT_SEED ^ int.from_bytes(digest[:6], "little")
    return torch.Generator(device=device).manual_seed(seed)


_LOGICAL_CACHE: dict[str, torch.Tensor] = {}


def logical_weight(name: str, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Deterministic full (unsharded) logical tensor for a given name."""
    key = f"{name}:{shape}"
    if key not in _LOGICAL_CACHE:
        _LOGICAL_CACHE[key] = (
            torch.randn(shape, dtype=torch.float32, device=device,
                        generator=_named_generator(name, device)) * WEIGHT_STD
        ).to(torch.bfloat16)
    return _LOGICAL_CACHE[key]


def fill_mla_weights(
    mla: torch.nn.Module,
    *,
    device: torch.device,
    q_shard: tuple[int, int],
    o_shard: tuple[int, int],
) -> None:
    """Fill an MLA instance with slices of the logical weights.

    q_shard = (rank, count) for the Q-side head shard (q_b_proj, kv_b_proj,
    k_b_proj_trans -> num_heads_tp heads).
    o_shard = (rank, count) for the output-side head shard (v_b_proj, o_proj
    input, g_proj output -> num_heads_tp_cp heads).
    """
    qr, qc = q_shard
    orank, ocount = o_shard
    heads_q = NUM_HEADS // qc
    heads_o = NUM_HEADS // ocount

    kv_a = logical_weight(
        "kv_a_proj_with_mqa",
        (Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE),
        device,
    )
    q_b = logical_weight("q_b_proj", (NUM_HEADS, QK_HEAD_DIM, Q_LORA_RANK), device)
    kv_b = logical_weight(
        "kv_b_proj", (NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK), device
    )
    o_w = logical_weight("o_proj", (HIDDEN_SIZE, NUM_HEADS, V_HEAD_DIM), device)
    g_w = logical_weight("g_proj", (NUM_HEADS, V_HEAD_DIM, HIDDEN_SIZE), device)

    with torch.no_grad():
        mla.kv_a_proj_with_mqa.weight.copy_(kv_a)
        mla.q_a_layernorm.weight.fill_(1.0)
        mla.kv_a_layernorm.weight.fill_(1.0)

        q_heads = q_b[qr * heads_q : (qr + 1) * heads_q]  # [heads_q, 192, q_lora]
        mla.q_b_proj.weight.copy_(q_heads.reshape(heads_q * QK_HEAD_DIM, Q_LORA_RANK))

        kv_heads = kv_b[qr * heads_q : (qr + 1) * heads_q]  # [heads_q, 256, kv_lora]
        mla.kv_b_proj.weight.copy_(
            kv_heads.reshape(heads_q * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK)
        )
        # Absorption weights derived exactly like the DeepseekV3 loader.
        k_nope = kv_heads[:, :QK_NOPE_HEAD_DIM, :]  # [heads_q, 128, 512]
        mla.k_b_proj_trans.copy_(k_nope.transpose(1, 2))  # [heads_q, 512, 128]

        v_heads = kv_b[orank * heads_o : (orank + 1) * heads_o, QK_NOPE_HEAD_DIM:, :]
        mla.v_b_proj.copy_(v_heads)  # [heads_o, 128, 512]

        o_slice = o_w[:, orank * heads_o : (orank + 1) * heads_o, :]
        mla.o_proj.weight.copy_(o_slice.reshape(HIDDEN_SIZE, heads_o * V_HEAD_DIM))

        g_slice = g_w[orank * heads_o : (orank + 1) * heads_o]
        mla.g_proj.weight.copy_(g_slice.reshape(heads_o * V_HEAD_DIM, HIDDEN_SIZE))


def make_mla(
    *,
    mapping: Any,
    device: torch.device,
    mapping_with_cp: Optional[Any] = None,
    reduce_output: bool = False,
    q_shard: tuple[int, int],
    o_shard: tuple[int, int],
) -> torch.nn.Module:
    """Build one KimiMLAAttention with deterministic sliced weights."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiMLAAttention

    model_config = build_model_config(mapping)
    mla = KimiMLAAttention(
        model_config,
        layer_idx=0,
        aux_stream=torch.cuda.Stream(),
        reduce_output=reduce_output,
        output_gate_aux_stream=torch.cuda.Stream(),
        mapping_with_cp=mapping_with_cp,
    )
    if getattr(model_config, "skip_create_weights_in_init", False):
        mla.create_weights()
    mla.cuda(device)
    # Direct forward_impl path: no extra-attrs weakref dance needed.
    mla.register_to_config = False
    fill_mla_weights(mla, device=device, q_shard=q_shard, o_shard=o_shard)
    return mla


def make_kv_cache_manager(
    *,
    mapping: Any,
    batch: int,
    local_seq_len: int,
    device_seq_extra: int = 1,
) -> Any:
    """KVCacheManager for one MLA layer holding `local_seq_len` tokens/req."""
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    max_seq = local_seq_len + device_seq_extra
    blocks_per_seq = (max_seq + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    return KVCacheManager(
        KvCacheConfig(
            max_tokens=blocks_per_seq * TOKENS_PER_BLOCK * batch,
            enable_block_reuse=False,
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seq,
        max_batch_size=batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.bfloat16)),
    )


def make_metadata(
    *,
    kv_cache_manager: Any,
    batch: int,
    phase: str,
    seq_len: int,
    cached: int,
    mapping: Optional[Any] = None,
    request_ids: Optional[list[int]] = None,
) -> Any:
    """Plain (non-helix) TRTLLM attention metadata for prefill or decode."""
    from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    metadata_cls = get_attention_backend("TRTLLM").Metadata
    if request_ids is None:
        request_ids = list(range(batch))
    if phase == "prefill":
        seq_lens = [seq_len] * batch
        num_contexts = batch
        cached_list = [0] * batch
        max_tokens = batch * seq_len
    else:
        seq_lens = [1] * batch
        num_contexts = 0
        cached_list = [cached] * batch
        max_tokens = max(batch, 32)
    metadata = metadata_cls(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int, device="cpu"),
        num_contexts=num_contexts,
        request_ids=request_ids,
        prompt_lens=[cached if phase == "decode" else seq_len] * batch,
        max_num_requests=batch,
        max_num_tokens=max_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=cached_list
        ),
        mapping=mapping,
    )
    metadata.prepare()
    return metadata


# Context fills are setup-only (never timed); chunk them so the dense-prefill
# attention workspace stays small even at large batch x ISL (GPU7 headroom).
CTX_FILL_CHUNK_REQS = 4

# Perf-only mode: skip the real context fill (KV stays zero-filled). Kernel
# shapes, cache layouts, and memory traffic are identical to a real history;
# only the VALUES are synthetic — same tradeoff as block4_demo decode. Set
# from the driver's --skip-ctx-fill; correctness checks are meaningless then.
SKIP_CTX_FILL = False


def add_requests(kv_cache_manager: Any, batch: int, tokens: int, is_gen: bool) -> None:
    kv_cache_manager.add_dummy_requests(
        list(range(batch)), token_nums=[tokens] * batch, is_gen=is_gen
    )


def hidden_batch(tokens: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn((tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device,
                    generator=generator)
        * 0.5
    )


def hidden_for_reqs(
    lo: int, hi: int, isl: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Deterministic per-request history slice [(hi-lo)*isl, hidden].

    Generated on demand so no rank ever materializes the full global history
    (B x ISL x hidden reaches tens of GB at large sweeps).
    """
    outs = []
    for b in range(lo, hi):
        generator = torch.Generator(device=device).manual_seed(seed * 100003 + b)
        outs.append(
            torch.randn((isl, HIDDEN_SIZE), dtype=torch.bfloat16, device=device,
                        generator=generator)
            * 0.5
        )
    return torch.cat(outs)
