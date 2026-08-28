"""Four Kimi-K3 transformer blocks (3 KDA + 1 MLA) with the tp_baseline MoE.

Block pattern is K3's real 3:1 interleave (config.json of the K3 checkpoint:
``kda_layers=[1,2,3,...]``, ``full_attn_layers=[4,8,...]``), so the demo's four
blocks are KDA, KDA, KDA, MLA. Attention modules are IMPORTED unchanged from
the installed TRT-LLM build via ``_kimi_attention_factory``
(``tensorrt_llm/_torch/models/modeling_kimi_linear.py``); the residual/norm
wiring is ADAPTED from ``DeepseekV3DecoderLayer`` (fused add+RMSNorm chain).
The MoE sublayer of every block is the unchanged ``tp_baseline``
``KimiK3MoELayerBaseline`` (TRTLLMGenFusedMoE routed path + NVLinkOneSided A2A,
GatedMLP shared experts).

One code path; the ONLY thing ``mode`` changes is configuration:

- ``tp``  — every rank holds the full global token set. Attention modules are
  built with ``Mapping(tp_size=world)`` (heads sharded, o_proj allreduce);
  shared experts are TP-sharded + allreduce; routed experts run EP+A2A on this
  rank's token slice and the local result is allgathered back to full.
- ``cp``  — every rank owns ``batch/world`` whole sequences. Attention modules
  are built with a single-rank ``Mapping`` (full weights, zero collectives);
  shared experts hold full weights (no collective); routed experts run the
  identical EP+A2A on the local tokens. The ONLY cross-rank communication is
  the MoE A2A dispatch/combine.

Weights: attention weights are randomly initialized (this is a perf/comm demo;
there is no K3 checkpoint on disk). Norm weights are set to 1, ``A_log``/
``dt_bias`` to stable constants, and projections to N(0, 0.02) so every
forward stays finite. The MoE weights are the deterministic cached tp_baseline
weights (unchanged).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

# K3 dims from the real checkpoint config (/node-storage/var/kimi-k3-config).
K3_HIDDEN_SIZE = 7168
K3_NUM_ATTENTION_HEADS = 96
K3_Q_LORA_RANK = 1536
K3_KV_LORA_RANK = 512
K3_QK_NOPE_HEAD_DIM = 128
K3_QK_ROPE_HEAD_DIM = 64
K3_V_HEAD_DIM = 128
K3_KDA_HEAD_DIM = 128
K3_KDA_NUM_HEADS = 96
K3_KDA_CONV_KERNEL = 4
K3_RMS_NORM_EPS = 1e-5
K3_VOCAB_SIZE = 163840

NUM_BLOCKS = 4
KDA_LAYERS_1BASED = [1, 2, 3]  # config lists are 1-based; is_kda_layer adds 1
MLA_LAYERS_1BASED = [4]

WEIGHT_SEED = 20260828


def build_attention_model_config(*, mode: str, world_size: int, rank: int) -> Any:
    """ModelConfig[KimiLinearConfig] for the 4 demo blocks, per parallel mode.

    ``tp``: Mapping(world_size=W, tp_size=W) — KDA/MLA heads sharded W-ways,
    row projections allreduce, exactly the finalized TRT-LLM TP wiring.
    ``cp``: single-rank Mapping — full attention weights, no collectives; the
    process still participates in the world-sized MoE A2A (the MoE module gets
    its own DEP mapping from bench_moe, unchanged from tp_baseline).
    """
    from tensorrt_llm._torch.configs import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    if mode not in ("tp", "cp"):
        raise ValueError(f"mode must be 'tp' or 'cp', got {mode!r}")
    pretrained_config = KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        vocab_size=K3_VOCAB_SIZE,
        hidden_size=K3_HIDDEN_SIZE,
        intermediate_size=33792,  # unused (MoE replaces dense MLP) but real
        num_hidden_layers=NUM_BLOCKS,
        num_attention_heads=K3_NUM_ATTENTION_HEADS,
        num_key_value_heads=K3_NUM_ATTENTION_HEADS,
        q_lora_rank=K3_Q_LORA_RANK,
        kv_lora_rank=K3_KV_LORA_RANK,
        qk_nope_head_dim=K3_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=K3_QK_ROPE_HEAD_DIM,
        v_head_dim=K3_V_HEAD_DIM,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=65536,
        rms_norm_eps=K3_RMS_NORM_EPS,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "full_attn_layers": MLA_LAYERS_1BASED,
            "head_dim": K3_KDA_HEAD_DIM,
            "kda_layers": KDA_LAYERS_1BASED,
            "num_heads": K3_KDA_NUM_HEADS,
            "short_conv_kernel_size": K3_KDA_CONV_KERNEL,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    if mode == "tp":
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
    else:
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=QuantConfig(),  # attention runs unquantized BF16 (as K3)
        allreduce_strategy="AUTO",
    )


def _randomize_attention_weights(module: nn.Module, seed: int) -> None:
    """Deterministic, numerically-safe random init for the attention blocks.

    Norm scales -> 1, KDA ``A_log`` -> log(2) (stable decay), ``dt_bias`` -> 0,
    every other floating parameter -> N(0, 0.02). Perf-only fidelity: kernel
    shapes/layouts match the real model; values are synthetic.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not param.is_floating_point():
                continue
            leaf = name.rsplit(".", 1)[-1]
            if "norm" in name.lower() and leaf == "weight":
                param.fill_(1.0)
            elif leaf == "A_log":
                param.fill_(0.6931471824645996)  # log(2)
            elif leaf == "dt_bias":
                param.zero_()
            else:
                param.normal_(0.0, 0.02, generator=generator)


class KimiK3DemoBlock(nn.Module):
    """One K3 transformer block: RMSNorm -> KDA/MLA -> RMSNorm -> MoE.

    Residual wiring follows the fused add+RMSNorm chain of
    ``DeepseekV3DecoderLayer`` (which K3 reuses): the block receives
    ``(hidden, residual)`` and returns ``(moe_output, residual)``; the next
    block's input_layernorm performs the pending residual add.
    """

    def __init__(
        self,
        *,
        attn_model_config: Any,
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
        output_gate_aux_stream: torch.cuda.Stream,
        moe_layer: nn.Module,
        mode: str,
        rank: int,
    ) -> None:
        super().__init__()
        from tensorrt_llm._torch.models.modeling_kimi_linear import (
            _kimi_attention_factory,
        )
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        config = attn_model_config.pretrained_config
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)
        self.mode = mode
        self.rank = rank
        self.self_attn = _kimi_attention_factory(
            model_config=attn_model_config,
            layer_idx=layer_idx,
            aux_stream=aux_stream,
            reduce_output=(mode == "tp" and attn_model_config.mapping.tp_size > 1),
            output_gate_aux_stream=output_gate_aux_stream,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.moe = moe_layer

    # Separate methods so the benchmark can wrap each phase with CUDA events.
    def attn_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attn_metadata: Any,
        mamba_metadata: Any,
    ) -> torch.Tensor:
        if self.is_kda:
            return self.self_attn(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                mamba_metadata=mamba_metadata,
            )
        return self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        *,
        position_ids: torch.Tensor,
        attn_metadata: Any,
        mamba_metadata: Any,
        router_logits_local: torch.Tensor,
        all_rank_num_tokens: list[int],
        gather_routed,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.attn_forward(
            hidden_states,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            mamba_metadata=mamba_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.mode == "tp":
            # Full replicated tokens in; full tokens out.
            shared_partial = self.moe.shared_forward(hidden_states)
            shared_full = self.moe.allreduce_forward(shared_partial)
            offset = sum(all_rank_num_tokens[: self.rank])
            local = hidden_states[offset : offset + all_rank_num_tokens[self.rank]]
            routed_local = self.moe.routed_forward(
                local, router_logits_local, all_rank_num_tokens
            )
            routed_full = gather_routed(routed_local, all_rank_num_tokens)
            hidden_states = shared_full + routed_full
        else:
            # Local tokens in; local tokens out. MoE A2A is the only comm.
            shared_local = self.moe.shared_forward(hidden_states)
            routed_local = self.moe.routed_forward(
                hidden_states, router_logits_local, all_rank_num_tokens
            )
            hidden_states = shared_local + routed_local
        return hidden_states, residual


class KimiK3Block4Stack(nn.Module):
    """The 4-block demo stack. TP and CP share this exact code path."""

    def __init__(
        self,
        *,
        mode: str,
        world_size: int,
        rank: int,
        device: torch.device,
        moe_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.world_size = world_size
        self.rank = rank
        self.attn_model_config = build_attention_model_config(
            mode=mode, world_size=world_size, rank=rank
        )
        aux_stream = torch.cuda.Stream()
        output_gate_aux_stream = torch.cuda.Stream()
        self.blocks = nn.ModuleList(
            [
                KimiK3DemoBlock(
                    attn_model_config=self.attn_model_config,
                    layer_idx=layer_idx,
                    aux_stream=aux_stream,
                    output_gate_aux_stream=output_gate_aux_stream,
                    moe_layer=moe_layer,
                    mode=mode,
                    rank=rank,
                )
                for layer_idx in range(NUM_BLOCKS)
            ]
        )
        self.cuda(device)
        _randomize_attention_weights(self, WEIGHT_SEED)
        for block in self.blocks:
            if hasattr(block.self_attn, "post_load_weights"):
                block.self_attn.post_load_weights()

    def gather_routed(
        self, routed_local: torch.Tensor, all_rank_num_tokens: list[int]
    ) -> torch.Tensor:
        """TP-mode allgather restoring the full routed output (wrappable)."""
        from tensorrt_llm._torch.distributed import allgather

        return allgather(
            routed_local,
            self.attn_model_config.mapping,
            dim=0,
            sizes=all_rank_num_tokens,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attn_metadata: Any,
        mamba_metadata: Any,
        router_logits_local: torch.Tensor,
        all_rank_num_tokens: list[int],
    ) -> torch.Tensor:
        residual = None
        for block in self.blocks:
            hidden_states, residual = block(
                hidden_states,
                residual,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                mamba_metadata=mamba_metadata,
                router_logits_local=router_logits_local,
                all_rank_num_tokens=all_rank_num_tokens,
                gather_routed=self.gather_routed,
            )
        return hidden_states + residual

    def attention_weight_bytes(self) -> int:
        """Per-GPU attention+norm parameter bytes (MoE accounted separately)."""
        seen: set[int] = set()
        total = 0
        for param in list(self.parameters()) + list(self.buffers()):
            if not param.is_cuda or param.data_ptr() in seen:
                continue
            seen.add(param.data_ptr())
            total += param.numel() * param.element_size()
        return total
