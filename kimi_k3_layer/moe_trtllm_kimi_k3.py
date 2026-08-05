"""Kimi-K3 MoE baseline on the REAL TensorRT-LLM modules.

Runs inside the ``trt-dev`` container (official
``nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23`` image, workspace
mounted at ``/workspace``): ``KimiK3MoE`` instantiates the actual
``NemotronHMOE`` - real ``DeepseekV3Gate``, real ``Linear`` latent
projections, real CUTLASS/TRTLLM expert backends, real ``AllReduce`` -
so baseline numbers ARE TensorRT-LLM, not an emulation.

Structured for upstreaming (nothing in the trt-llm tree is modified):
this class is what the (unreleased) ``modeling_kimi_k3.KimiK3MoE``
parameterizes on top of the public 1.3.0rc23 ``NemotronHMOE`` -
Swiglu routed experts, gated SwiGLU shared expert (production K3 uses
SiTU; SwiGLU is the public activation with identical layout and cost),
the Stable-LatentMoE RMSNorm on the reduced latent, and the fused
latent+shared TP reduction (one [B, 3584+7168] message reduced BEFORE
the norm + fc2, replacing the parent's post-fc2 reduction which is
invalid under the nonlinear latent norm).

EXPERT BACKEND: ``k3_model_config(..., moe_backend=...)`` selects it.
'TRTLLM' is the PRODUCTION configuration - exactly what
``modeling_kimi_k3.KimiK3MoE`` hard-requires: ``ConfigurableMoE ->
TRTLLMGenFusedMoE`` with packed ``W4A8_MXFP4_MXFP8`` expert weights
(the published K3 QAT recipe: MXFP4 weights, MXFP8 activations;
random-initialized packed nibbles + ue8m0 block scales, no
checkpoint needed), routing in-kernel from the TRTLLM-flavored
``DeepseekV3Gate``, and ``SiTUAndMul`` (grafted from the checkout)
as the shared-expert activation. Residual deviations from the real
checkpoint: SiTU betas are placeholders (cost is beta-independent)
and the trtllm-gen kernel runs a Swiglu expert epilogue (production
1.3.0rc19 @ 496bc01f8e runs in-kernel SiTU from a PRIVATE FlashInfer
cubin pool - FLASHINFER_PRIVATE_CUBIN_DIR - unavailable here; same
elementwise epilogue cost).
'CUTLASS' keeps the bf16 ``CutlassFusedMoE`` comparison stack that
the b10 opt path currently drives (same expert kernels in baseline
and opt, isolating the non-expert work).

OP BACKEND: production K3 forces the FLASHINFER op backend (the SiTU
path early-returns use_flashinfer=True in 496bc01f8e), so production
expert + routing kernels come from FlashInfer's trtllm-gen cubin pool
- notably the SPLIT routing pipeline (routingIndicesBlockScoresKernel
+ routingIndicesClusterKernel, ~24 us) instead of rc23's monolithic
41-us routingIndicesClusterKernel. ``_force_flashinfer_op_backend``
replicates that (default; BENCH_MOE_OP_BACKEND=trtllm reverts to the
native torch.ops.trtllm path for comparison). Verified bit-exact vs
native at B=1/8 (B=64 differs only via bf16-vs-fp32 top-k tie-breaks
in the split pipeline - also true in production).

The optimized subclass lives in ``moe_b10_kimi_k3.py``; import both
via ``bench_moe_kimi_k3.py`` (it bootstraps ``tensorrt_llm`` before
this package so the container's installed runtime stays authoritative).
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE
from tensorrt_llm._torch.modules.fused_moe import (
    MoEWeightLoadingMode,
    create_moe,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import AuxStreamType, EventType
from tensorrt_llm.mapping import Mapping
from transformers import PretrainedConfig

from .config import (
    HIDDEN,
    MOE_INTER,
    MOE_LATENT,
    N_GROUP,
    NUM_EXPERTS,
    NUM_SHARED_EXPERTS,
    RMS_EPS,
    ROUTED_SCALING,
    TOP_K,
    TOPK_GROUP,
)
from .kda_trtllm_kimi_k3 import _graft

# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------


def k3_pretrained_config() -> PretrainedConfig:
    """The MoE-relevant slice of the Kimi-K3 checkpoint config."""
    cfg = PretrainedConfig()
    cfg.architectures = ["KimiK3ForCausalLM"]
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = MOE_INTER * NUM_SHARED_EXPERTS
    cfg.torch_dtype = torch.bfloat16
    cfg.num_attention_heads = 96
    cfg.num_hidden_layers = 1
    cfg.moe_intermediate_size = MOE_INTER
    cfg.moe_latent_size = MOE_LATENT
    cfg.n_routed_experts = NUM_EXPERTS
    cfg.num_experts_per_tok = TOP_K
    cfg.n_shared_experts = NUM_SHARED_EXPERTS
    cfg.moe_shared_expert_intermediate_size = MOE_INTER
    cfg.n_group = N_GROUP
    cfg.topk_group = TOPK_GROUP
    cfg.routed_scaling_factor = ROUTED_SCALING
    cfg.mlp_bias = False
    cfg.rms_norm_eps = RMS_EPS
    # SiTU betas are checkpoint values we don't have; the activation's
    # cost is beta-independent (same tanh+sigmoid epilogue), so any
    # positive value gives production-identical kernels.
    cfg.activation_situ_beta = 1.0
    cfg.activation_situ_linear_beta = None
    return cfg


def k3_model_config(rank: int, world: int,
                    moe_backend: str = "CUTLASS") -> ModelConfig:
    """TP-`world` mapping with EP experts (896/tp experts local, full
    384 intermediate - moe_tp's 384/8=48 cannot tile trtllm-gen bf16).

    moe_backend='TRTLLM' selects the PRODUCTION K3 expert stack
    (TRTLLMGenFusedMoE + packed W4A16_MXFP4, in-kernel routing) - the
    configuration modeling_kimi_k3.KimiK3MoE hard-requires.
    """
    mapping = Mapping(
        world_size=world,
        tp_size=world,
        rank=rank,
        gpus_per_node=world,
        moe_ep_size=world,
        moe_tp_size=1,
    )
    # honest-baseline knob: production runs AUTO (-> NCCL_SYMMETRIC on
    # this box); BENCH_ALLREDUCE_STRATEGY sweeps alternatives (ONESHOT,
    # TWOSHOT, MIN_LATENCY, NCCL, ...) so the baseline is the FASTEST
    # configuration reachable by flipping supported knobs, not a straw
    # man. Applies to baseline and opt alike (both use self.allreduce).
    return ModelConfig(
        pretrained_config=k3_pretrained_config(),
        mapping=mapping,
        moe_backend=moe_backend,
        allreduce_strategy=os.environ.get(
            "BENCH_ALLREDUCE_STRATEGY", "AUTO"),
    )


def _force_flashinfer_op_backend(backend) -> None:
    """Swap a TRTLLMGenFusedMoE onto the FlashInfer op backend.

    Production K3 (1.3.0rc19 @ 496bc01f8e) always takes this path (its
    SiTU epilogue lives in a private FlashInfer cubin pool), so the
    production expert + routing kernels are FlashInfer's. The rc23
    wheel gates it off for DeepSeekV3 routing, hence the direct swap.
    One shim: run_moe hands the op backend a FLAT mxfp8 scale (the
    native op's layout); flashinfer 0.6.15's autotuner asserts
    [num_tokens, hidden/32]. Same bytes - reshape at the boundary
    (production's private flashinfer has no such assert).
    """
    from tensorrt_llm._torch.modules.fused_moe.moe_op_backend import (
        get_op_backend,
    )

    fi = get_op_backend("flashinfer")
    _orig_run = fi.run_fp4_block_scale_moe

    def _run_2d_sf(router_logits, routing_bias, hidden_states,
                   hidden_states_scale, *args, **kwargs):
        if hidden_states_scale is not None \
                and hidden_states_scale.dim() == 1:
            hidden_states_scale = hidden_states_scale.view(
                hidden_states.shape[0], -1)
        return _orig_run(router_logits, routing_bias, hidden_states,
                         hidden_states_scale, *args, **kwargs)

    fi.run_fp4_block_scale_moe = _run_2d_sf
    backend.use_flashinfer = True
    backend.op_backend = fi


# --------------------------------------------------------------------------
# the K3 port
# --------------------------------------------------------------------------


class KimiK3MoE(NemotronHMOE):
    """Kimi-K3 latent MoE on public 1.3.0rc23 NemotronHMOE."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        reduce_output: bool = False,
    ) -> None:
        # the parent always gets a known-good bf16 CUTLASS config: its
        # experts are discarded below, and its dense Linears (latent
        # projections) are bf16 in production too (quant-excluded)
        super().__init__(
            replace(model_config, moe_backend="CUTLASS"),
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            reduce_output=reduce_output,
        )
        config = model_config.pretrained_config
        self.moe_backend_name = model_config.moe_backend.upper()
        production = self.moe_backend_name == "TRTLLM"

        # K3 experts are gated (SiTU in production; the installed
        # trtllm-gen kernel's Swiglu epilogue is the same cost). The
        # installed parent hard-codes Relu2, so rebuild gate + experts.
        self.activation_type = ActivationType.Swiglu
        self.experts = None
        torch.cuda.empty_cache()
        if production:
            # gate flavor depends on the MoE backend (routing runs
            # inside the trtllm-gen kernel from raw logits)
            self.gate = DeepseekV3Gate(
                self.hidden_size,
                self.num_experts,
                top_k=self.top_k,
                n_group=self.moe_n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
                dtype=config.torch_dtype,
                fuse_routing_kernel=True,
                apply_routing=False,
                moe_backend="TRTLLM",
            )
            # production expert config: MXFP4 weights + MXFP8
            # activations (the published K3 QAT recipe; one of the two
            # modes modeling_kimi_k3.KimiK3MoE hard-requires)
            experts_config = replace(
                model_config,
                quant_config=QuantConfig(
                    quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
                moe_backend="TRTLLM",
            )
        else:
            experts_config = model_config
        # the parent's create_moe registered this layer_idx; deregister
        # so the rebuild does not trip the duplicate-layer assert
        moe_layers = model_config.extra_attrs.get("moe_layers", {})
        moe_layers.pop(str(layer_idx), None)
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=experts_config,
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            bias=self.mlp_bias,
            activation_type=ActivationType.Swiglu,
        )
        if production and os.environ.get(
                "BENCH_MOE_OP_BACKEND", "flashinfer") == "flashinfer":
            _force_flashinfer_op_backend(
                getattr(self.experts, "backend", self.experts))

        # production (496bc01f8e) constructs AllReduce WITHOUT dtype;
        # the rc23 parent passes dtype (pre-building fused MNNVL paths).
        # Rebuild to match the production construction exactly.
        if self.allreduce is not None:
            from tensorrt_llm._torch.distributed import AllReduce
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

        # K3 shared expert is a gated GatedMLP (parent: relu2 MLP).
        # Production activation is SiTUAndMul (grafted from the
        # checkout); the CUTLASS comparison mode keeps F.silu so the
        # b10 opt kernels (which fuse SwiGLU) stay output-comparable.
        if production:
            situ = _graft(
                "tensorrt_llm._torch.modules.situ",
                "_torch/modules/situ.py",
            )
            shared_activation = situ.SiTUAndMul(
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
            )
        else:
            shared_activation = F.silu
        shared_inter = (
            config.moe_shared_expert_intermediate_size
            * config.n_shared_experts
        )
        self.shared_experts = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_inter,
            bias=self.mlp_bias,
            activation=shared_activation,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=False,
            is_shared_expert=True,
        )

        # Stable LatentMoE: RMSNorm on the fully reduced routed latent
        # right before fc2 (kimi_k3 config: latent_moe_use_norm=true).
        self.latent_norm = RMSNorm(
            hidden_size=self.moe_hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """K3 flow: gate -> fc1 -> experts (PARTIAL latent) || shared
        (PARTIAL) -> ONE fused allreduce of cat([latent, shared]) ->
        split -> latent RMSNorm -> fc2 -> shared + routed.

        The latent norm is nonlinear, so unlike the parent the TP
        reduction must happen at the latent (before norm + fc2); fusing
        the shared output into the same message keeps it one AR.
        """
        orig_shape = hidden_states.shape
        h2d = hidden_states.view(-1, self.hidden_dim)
        all_rank_num_tokens = kwargs.get(
            "all_rank_num_tokens",
            getattr(attn_metadata, "all_rank_num_tokens", None),
        )

        def _routed():
            with torch.profiler.record_function("moe.base_gate"):
                router_logits = self.gate(h2d)
            with torch.profiler.record_function("moe.base_fc1_latent"):
                latent = self.fc1_latent_proj(h2d)
            with torch.profiler.record_function("moe.base_experts"):
                return self.experts(
                    latent,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False,
                )

        def _shared():
            with torch.profiler.record_function("moe.base_shared"):
                return self.shared_experts(h2d)

        routed_out, shared_out = maybe_execute_in_parallel(
            _routed,
            _shared,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream_shared,
            disable_on_compile=True,
        )
        routed_out = routed_out.view(-1, self.moe_hidden_size)

        if self.allreduce is not None:
            with torch.profiler.record_function("moe.base_allreduce"):
                reduced = self.allreduce(
                    torch.cat((routed_out, shared_out), dim=-1),
                    all_reduce_params=kwargs.get("all_reduce_params"),
                )
                routed_out, shared_out = torch.split(
                    reduced,
                    (self.moe_hidden_size, self.hidden_dim),
                    dim=-1,
                )

        with torch.profiler.record_function("moe.base_latent_norm"):
            routed_out = self.latent_norm(routed_out)
        with torch.profiler.record_function("moe.base_fc2_latent"):
            routed_out = self.fc2_latent_proj(routed_out)
        with torch.profiler.record_function("moe.base_add"):
            return (shared_out + routed_out).view(orig_shape)
