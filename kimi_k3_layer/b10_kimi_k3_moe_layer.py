"""Kimi-K3 latent-MoE reference and measured B10 layer.

The optimized path owns no communication implementation: every collective is
issued through one :class:`communication.collective.Collectives` instance.

Zero-copy AR staging: prefill tails (and the EXP packed decode tail)
produce their AR operands directly into ``Collectives.symm_input``
views when the autotuned pick is a symm-staged impl
(``Collectives.symm_staged``), deleting the stage-in DtoD copies. The
sharded tail partitions one staging buffer between its two ARs —
routed at [0, B*LATENT), shared above it — so the overlap-stream down
GEMM cannot race the main stream's routed AR (see
``_split_ar_staging``).

Column-slice RMSNorm (``kimi_k3_layer.kernels.rmsnorm_cutedsl``): the
latent norm in the optimized tails reduces the full row but only
writes the columns its consumer reads — the fc2-shard tails get just
this rank's [B, LATENT/world] window (7/8 less write; contiguous
addmm operand), and the packed tails feed ``packed[:, :LATENT]``
strided instead of paying flashinfer's ``.contiguous()`` copy. The
CuTeDSL kernel (single gmem pass, register-resident window) beats the
Triton port ``rmsnorm_cols`` at every shape (window 20.8 vs 28.3 us,
strided 36.3 vs 51.1 us at B=16384); Triton stays vendored as the
reference/fallback. The baseline path keeps its own norm and is
untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE
from tensorrt_llm._torch.modules.fused_moe import MoEWeightLoadingMode, create_moe
from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import AuxStreamType, EventType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from kimi_k3_layer.capabilities import Capabilities
from kimi_k3_layer.kernels.dual_out_gemm_cutedsl import dual_out_gemm_cutedsl
from kimi_k3_layer.kernels.gather_quant import gather_quant_mxfp8
from kimi_k3_layer.kernels.rmsnorm_cutedsl import rmsnorm_column_slice_cutedsl
from kimi_k3_layer.kernels.routing import (
    radix_available,
    route_radix_for_trtllm_gen,
)
from kimi_k3_layer.kernels.situ_triton import situ_and_mul
from kimi_k3_layer.config import (
    HIDDEN,
    MOE_INTER,
    MOE_LATENT,
    N_GROUP,
    NUM_EXPERTS,
    NUM_SHARED_EXPERTS,
    RMS_EPS,
    ROUTED_SCALING,
    SHARED_INTER,
    TOP_K,
    TOPK_GROUP,
)
from kimi_k3_layer.b10_kimi_k3_kda_layer import _graft

SiTUAndMul = _graft(
    "tensorrt_llm._torch.modules.situ",
    "_torch/modules/situ.py",
).SiTUAndMul

SMALL_BATCH_MAX_TOKENS = 32
# 256 since Aug-19 (was 128): the decode path still beats the baseline
# at 192/256 (see agent/moe_optimization.md), and extending the cap
# removes half of the awkward 129..2048 fall-back-to-baseline window.
DECODE_MAX_TOKENS = 256
# Aug-11 re-search (post route-side-stream/evict-last/zero-copy, and
# reinforced by the routing-indices graph patch): the multimem tail wins
# from B=16 up (was <=16 sharded). Both flips hold with the graph patch
# off, so they are drift from the Aug-11 optimizations, not patch
# artifacts. See kimi_k3/local_results/strategy_update_aug11.md.
# The fused fc1+gate+shared_up front wins at EVERY decode size (Aug-11
# moved its boundary 112 -> 128 = DECODE_MAX_TOKENS; Aug-19 one-axis
# A/B in the shipped config: fused 81.2/100.2 us vs separate-sharded
# 89.8/111.5 at B=64/128), so decode uses it unconditionally and the
# former FUSED_FC1_SHARED_GATE_MAX_TOKENS boundary was deleted.
SHARDED_FC2_MAX_TOKENS = 8
# No baseline window since Aug-19: the FULL-fc1 + FC2_SHARD-tail prefill
# beats the baseline at every size in the former 257..2048 zone
# (+16.9/+22.6/+17.2% at 512/1024/2048, graph-timed; see
# agent/moe_optimization.md §3). The knob remains for EXP overrides.
PREFILL_BASELINE_MAX_TOKENS = DECODE_MAX_TOKENS
# 2048 since Aug-19 (was 4096): the sharded path's gather moved to the
# b10 DMA copy-engine mover (zero SM — hides under gate/routing/shared,
# see communication/RESULTS.md overlap study). Graph-timed vs
# FULL+FC2_SHARD: +30/+33/+60 us at 2048/4096/8192; FULL still wins at
# 512-1024 (the DMA fixed cost exceeds the fc1 FLOP cut there).
PREFILL_SHARDED_MIN_TOKENS = 2048
PREFILL_NATIVE_EXPERT_MIN_TOKENS = 8192


class LayerMode(str, Enum):
    EXP = "exp"
    DEPLOY = "deploy"


class DecodeFront(str, Enum):
    # The three fronts the measured plan uses: the fused dual-out GEMM
    # for all decode sizes, the separate fronts for prefill (sharded
    # from 4096 tokens, full below). Two former variants were removed
    # as measured losers and stay research-only in kernels/: the
    # Triton dual-out GEMM (kernels/dual_out_gemm_triton.py, slower
    # than the CuTeDSL kernel at every size) and a fc1+gate-only fuse
    # without the shared rows (dominated by this full fuse).
    FUSED_FC1_SHARED_GATE_CUTE = "fused_fc1_shared_gate_cute"
    SEPARATE_SHARDED_FC1 = "separate_sharded_fc1"
    SEPARATE_FULL_FC1 = "separate_full_fc1"


class Routing(str, Enum):
    # Unchanged SGLang RouteRadixKernel (byte-radix top-k counting).
    # Measured fastest at every batch size on B200 (CUDA-graph A/B vs
    # the removed CUTE_EXACT: 3.4 vs 4.7 us at B=1, 46 vs 50 us at
    # 16384) with bit-identical expert IDs; see
    # kimi_k3_layer/kernels/routing_permutation_results.md.
    RADIX = "radix"
    REFERENCE = "reference"
    # (The CuTeDSL and Triton routing kernels measured slower than
    # radix at EVERY size and are deliberately NOT layer backends;
    # they remain research-only material in kernels/routing_cutedsl.py
    # and kernels/routing_triton.py.)


class DecodeTail(str, Enum):
    # The two measured tails (fc2 shard <= 8 tokens, multimem full-fc2
    # above) plus the baseline-style packed tail kept for EXP ablations.
    # A fourth variant (FULL_FC2_SHARED_REDUCE: full fc2 with the shared
    # AR chained as a plain flashinfer/nccl collective) was removed —
    # never selected by any measured plan; the multimem tail dominates
    # it wherever full fc2 applies.
    SHARDED_FC2_OUTPUT_REDUCE = "sharded_fc2_output_reduce"
    FULL_FC2_MULTIMEM_SHARED_REDUCE = "full_fc2_multimem_shared_reduce"
    PACKED_LATENT_SHARED_REDUCE = "packed_latent_shared_reduce"


class PrefillFC1(str, Enum):
    FULL = "full"
    SHARDED = "sharded"


class PrefillTail(str, Enum):
    """Tail of the FULL-fc1 prefill path (the SHARDED path always uses
    the fc2-shard tail). PACKED: one AR of cat([latent | shared]) then
    the full fc2. FC2_SHARD: AR(latent) -> column-window norm -> this
    rank's fc2 shard addmm_ into the shared partial -> ONE output AR
    that reduces shared and fc2 partials together — same total wire,
    1/8 the fc2 FLOPs, one extra collective latency."""

    PACKED = "packed"
    FC2_SHARD = "fc2_shard"


class ExpertBackend(str, Enum):
    FLASHINFER = "flashinfer"
    NATIVE = "native"


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete EXP pipeline selection; no partial mutable flags."""

    enabled: bool = True
    decode_front: DecodeFront = DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
    routing: Routing = Routing.RADIX
    decode_tail: DecodeTail = DecodeTail.SHARDED_FC2_OUTPUT_REDUCE
    # Routing, the fc1 column-AG+quantize, and the shared chain all
    # depend only on the front's dual_out output; with this flag the
    # routing kernel moves to its own side stream so all three run
    # concurrently instead of routing+AG serializing on the main stream.
    route_on_side_stream: bool = False
    prefill_fc1: PrefillFC1 = PrefillFC1.FULL
    prefill_tail: PrefillTail = PrefillTail.PACKED
    prefill_expert_backend: ExpertBackend = ExpertBackend.FLASHINFER
    prefill_overlap_shared_branch: bool = True

    @classmethod
    def all_off(cls) -> "ExperimentConfig":
        return cls(enabled=False)

    def with_axis(self, axis: str, value: Any) -> "ExperimentConfig":
        if axis not in self.__dataclass_fields__ or axis == "enabled":
            raise ValueError(f"unknown sweep axis {axis!r}")
        return replace(self, **{axis: value})


def measured_config(
    tokens: int,
    *,
    prefill_baseline_max_tokens: int = PREFILL_BASELINE_MAX_TOKENS,
    prefill_native_expert_min_tokens: int = PREFILL_NATIVE_EXPERT_MIN_TOKENS,
) -> ExperimentConfig:
    """Frozen Aug-11 plan, including both measured decode boundaries."""
    if tokens <= DECODE_MAX_TOKENS:
        return ExperimentConfig(
            decode_front=DecodeFront.FUSED_FC1_SHARED_GATE_CUTE,
            routing=Routing.RADIX,
            decode_tail=(
                DecodeTail.SHARDED_FC2_OUTPUT_REDUCE
                if tokens <= SHARDED_FC2_MAX_TOKENS
                else DecodeTail.FULL_FC2_MULTIMEM_SHARED_REDUCE),
            # Uniform 4.5-5.7 us win across B=8..128 (route_overlap A/B,
            # Aug-11): the routing kernel leaves the main-stream critical
            # path, where it used to serialize with the column-AG.
            route_on_side_stream=True,
        )
    if tokens <= prefill_baseline_max_tokens:
        return ExperimentConfig.all_off()
    sharded = tokens >= PREFILL_SHARDED_MIN_TOKENS
    # Prefill configs set ONLY the prefill_* axes (+ routing): the
    # prefill paths never read decode_front/decode_tail, so those stay
    # at their dataclass defaults instead of pretending to matter.
    return ExperimentConfig(
        routing=Routing.RADIX,
        prefill_fc1=PrefillFC1.SHARDED if sharded else PrefillFC1.FULL,
        # fc2-shard tail everywhere (the SHARDED path carries it
        # structurally): measured +17..+26% over baseline at 512..4095
        # vs the packed tail's +2..+6%.
        prefill_tail=PrefillTail.FC2_SHARD,
        prefill_expert_backend=(
            ExpertBackend.NATIVE
            if tokens >= prefill_native_expert_min_tokens
            else ExpertBackend.FLASHINFER),
    )


def k3_pretrained_config() -> PretrainedConfig:
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
    cfg.activation_situ_beta = 1.0
    cfg.activation_situ_linear_beta = None
    return cfg


def k3_model_config(rank: int, world: int,
                    moe_backend: str = "TRTLLM") -> ModelConfig:
    mapping = Mapping(
        world_size=world,
        tp_size=world,
        rank=rank,
        gpus_per_node=world,
        moe_ep_size=world,
        moe_tp_size=1,
    )
    return ModelConfig(
        pretrained_config=k3_pretrained_config(),
        mapping=mapping,
        moe_backend=moe_backend,
        allreduce_strategy=os.environ.get("BENCH_ALLREDUCE_STRATEGY", "AUTO"),
    )


def _force_flashinfer_op_backend(backend) -> None:
    """Select FlashInfer and restore scales flattened by TRT-LLM 1.3."""
    from tensorrt_llm._torch.modules.fused_moe.moe_op_backend import get_op_backend

    flashinfer = get_op_backend("flashinfer")
    if not getattr(flashinfer, "_kimi_k3_2d_scale_compat", False):
        original = flashinfer.run_fp4_block_scale_moe

        def run_2d_scale(router_logits, routing_bias, hidden_states,
                         hidden_states_scale, *args, **kwargs):
            if hidden_states_scale is not None and hidden_states_scale.dim() == 1:
                hidden_states_scale = hidden_states_scale.view(
                    hidden_states.shape[0], -1)
            return original(router_logits, routing_bias, hidden_states,
                            hidden_states_scale, *args, **kwargs)

        flashinfer.run_fp4_block_scale_moe = run_2d_scale
        flashinfer._kimi_k3_2d_scale_compat = True
    backend.use_flashinfer = True
    backend.op_backend = flashinfer


class KimiK3MoEReference(NemotronHMOE):
    """TRT-LLM Kimi-K3 reference whose TP reduction uses Collectives."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        reduce_output: bool = False,
        *,
        collectives=None,
    ) -> None:
        super().__init__(
            replace(model_config, moe_backend="CUTLASS"),
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            reduce_output=reduce_output,
        )
        config = model_config.pretrained_config
        self.activation_type = ActivationType.Swiglu
        self.experts = None
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
        experts_config = replace(
            model_config,
            quant_config=QuantConfig(
                quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
            moe_backend="TRTLLM",
        )
        model_config.extra_attrs.get("moe_layers", {}).pop(str(layer_idx), None)
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
        _force_flashinfer_op_backend(
            getattr(self.experts, "backend", self.experts))

        # The inherited transport is deliberately not part of this layer.
        self.allreduce = None
        shared_activation = SiTUAndMul(
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
        )
        self.shared_experts = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=(
                config.moe_shared_expert_intermediate_size
                * config.n_shared_experts),
            bias=self.mlp_bias,
            activation=shared_activation,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=False,
            is_shared_expert=True,
        )
        self.latent_norm = RMSNorm(
            hidden_size=self.moe_hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self._collectives = collectives

    def attach_collectives(self, collectives) -> None:
        self._collectives = collectives

    def _comm(self):
        if self._collectives is None:
            if self.mapping.tp_size != 1:
                raise RuntimeError("TP>1 requires a Collectives instance")
            from communication.collective import Collectives
            self._collectives = Collectives(None, 0)
        return self._collectives

    def baseline_forward(self, hidden_states: torch.Tensor,
                         attn_metadata=None, **kwargs) -> torch.Tensor:
        """The one reference path, also used by EXP all-off."""
        original_shape = hidden_states.shape
        h = hidden_states.view(-1, self.hidden_dim)
        all_rank_tokens = kwargs.get(
            "all_rank_num_tokens",
            getattr(attn_metadata, "all_rank_num_tokens", None),
        )

        def routed_branch():
            logits = self.gate(h)
            latent = self.fc1_latent_proj(h)
            return self.experts(
                latent,
                logits,
                all_rank_num_tokens=all_rank_tokens,
                use_dp_padding=False,
            )

        def shared_branch():
            return self.shared_experts(h)

        routed, shared = maybe_execute_in_parallel(
            routed_branch,
            shared_branch,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream_shared,
            disable_on_compile=True,
        )
        routed = routed.view(-1, self.moe_hidden_size)
        packed = self._comm().all_reduce(torch.cat((routed, shared), dim=-1))
        routed, shared = torch.split(
            packed, (self.moe_hidden_size, self.hidden_dim), dim=-1)
        return (shared + self.fc2_latent_proj(
            self.latent_norm(routed))).view(original_shape)

    def forward(self, hidden_states: torch.Tensor, attn_metadata=None,
                **kwargs) -> torch.Tensor:
        return self.baseline_forward(hidden_states, attn_metadata, **kwargs)


@dataclass
class _FrontResult:
    logits: torch.Tensor
    latent: torch.Tensor | None = None
    shard: torch.Tensor | None = None
    shared_gate_up: torch.Tensor | None = None


class B10KimiK3MoELayer(KimiK3MoEReference):
    """Measured B10 layer with immutable DEPLOY and explicit EXP modes."""

    OPT_MAX_TOKENS = DECODE_MAX_TOKENS

    def __init__(
        self,
        *args,
        mode: LayerMode | str = LayerMode.DEPLOY,
        prefill_baseline_max_tokens: int = PREFILL_BASELINE_MAX_TOKENS,
        prefill_native_expert_min_tokens: int = PREFILL_NATIVE_EXPERT_MIN_TOKENS,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mode = LayerMode(mode)
        self.prefill_baseline_max_tokens = prefill_baseline_max_tokens
        self.prefill_native_expert_min_tokens = prefill_native_expert_min_tokens
        self._exp_config = None
        self._optimized = False
        # Probed lazily, NOT here: the arch query needs a live CUDA context and
        # this layer is constructed on CPU (the caller `.cuda()`s it after).
        # init_optimized re-probes with the process group for rank agreement.
        self._capabilities: Capabilities | None = None

    def init_optimized(self, *, max_batch: int = DECODE_MAX_TOKENS,
                       collectives=None) -> None:
        if collectives is not None:
            self.attach_collectives(collectives)
        comm = self._comm()
        self._world, self._rank = comm.world, comm.rank
        self._max_batch = max_batch

        fc1 = self.fc1_latent_proj.weight.data
        fc2 = self.fc2_latent_proj.weight.data
        width = MOE_LATENT // self._world
        cols = slice(self._rank * width, (self._rank + 1) * width)
        self._latent_width = width
        self._fc1_full = fc1
        self._fc1_shard = fc1[cols]  # contiguous row view, no second copy
        self._fc2_t = fc2.T.contiguous()
        self.fc2_latent_proj.weight.data = self._fc2_t.T
        self._fc2_shard_t = self._fc2_t[cols]
        self._shared_gate_up = self.shared_experts.gate_up_proj.weight.data
        self._shared_down = self.shared_experts.down_proj.weight.data
        # Vendored stride-aware SiTU (kimi_k3.kernels.situ_and_mul):
        # bit-exact with the grafted trt-llm SiTUAndMul, but consumes
        # the fuse3 front's merged[:, width:] slice directly instead
        # of paying its .contiguous() copy.
        act = self.shared_experts.activation
        self._situ_beta = act.beta
        self._situ_linear_beta = act.linear_beta
        self._norm_weight = self.latent_norm.weight.data
        self._gate_bias_f32 = (
            self.gate.e_score_correction_bias.data.float().contiguous())
        # Load the prebuilt radix routing module now (no first-forward
        # JIT); Routing.RADIX is the measured default everywhere.
        if not radix_available():
            raise RuntimeError(
                "radix routing artifact missing - build it via "
                "kimi_k3_layer/kernels/routing.py (kernel_research "
                "sglang_radix_prebuilt)")

        backend = getattr(self.experts, "backend", self.experts)
        if type(backend).__name__ != "TRTLLMGenFusedMoE":
            raise TypeError("B10 layer requires TRTLLMGenFusedMoE")
        if not getattr(backend, "has_w4a8_mxfp4_mxfp8", False):
            raise TypeError("B10 layer requires W4A8 MXFP4/MXFP8 experts")
        self._gen_backend = backend
        from tensorrt_llm._torch.modules.fused_moe.moe_op_backend import (
            get_op_backend,
        )
        self._native_op_backend = get_op_backend("trtllm")

        # Capability probe, now that the expert backend exists. `situ_experts`
        # is read off TRT-LLM's OWN verdict rather than re-derived here: SiTU is
        # NOT signalled by activation_type -- `_is_situ_activation` is
        # (activation_type == Swiglu) AND pretrained_config.hidden_act ==
        # "situ" (fused_moe_trtllm_gen.py:231), so a layer passing Swiglu still
        # runs SiTU on a real K3 checkpoint. Duplicating that predicate here
        # would drift; reading the flag cannot. Absent on stock TRT-LLM (no
        # SiTU support at all), where False is correct.
        # Probed with the group so every rank runs the same stages -- a
        # collective-bearing stage enabled on some ranks only deadlocks.
        self._capabilities = Capabilities.probe(
            getattr(comm, "group", None),
            situ_experts=bool(
                getattr(self._gen_backend, "_is_situ_activation", False)))
        self._capabilities.log_once(self._rank)

        device = fc1.device
        self._overlap_stream = torch.cuda.Stream(device=device)
        self._overlap_fork = torch.cuda.Event()
        self._overlap_join = torch.cuda.Event()
        self._route_stream = torch.cuda.Stream(device=device)
        self._route_fork = torch.cuda.Event()
        self._route_join = torch.cuda.Event()
        self._prefill_stream = torch.cuda.Stream(device=device, priority=-1)
        self._prefill_fork = torch.cuda.Event()
        self._prefill_join = torch.cuda.Event()

        self._fused_fc1_shared_gate_weight = None
        # DEPLOY materializes only its measured fused-front copy. EXP builds
        # alternate copies lazily when an explicit config needs them.
        self._ensure_front_weight(
            DecodeFront.FUSED_FC1_SHARED_GATE_CUTE)
        self._optimized = True

    def set_experiment_config(self,
                              config: ExperimentConfig | None) -> None:
        if self.mode is LayerMode.DEPLOY:
            raise RuntimeError("DEPLOY has no mutable sweep configuration")
        if config is not None and not isinstance(config, ExperimentConfig):
            raise TypeError("set a complete ExperimentConfig or None")
        self._exp_config = config
        if config is not None and config.enabled:
            self._ensure_front_weight(config.decode_front)

    def experiment_configs(self, axis: str, values) -> tuple[ExperimentConfig, ...]:
        if self.mode is LayerMode.DEPLOY:
            raise RuntimeError("sweeps are available only in EXP")
        base = self._exp_config or measured_config(1)
        return tuple(base.with_axis(axis, value) for value in values)

    @property
    def capabilities(self) -> Capabilities:
        """Stage availability for this part; probed on first use, then cached.

        ``init_optimized`` replaces this with a rank-agreed probe. Reading it
        before then (single-GPU work, EXP sweeps) is fine: the arch gap table is
        deterministic, so every rank derives the same answer anyway -- the
        all-reduce exists to catch operator overrides that differ per rank.
        """
        if self._capabilities is None:
            self._capabilities = Capabilities.probe()
        return self._capabilities

    def _config(self, tokens: int) -> ExperimentConfig:
        # Capability filter applies to EXP too: an explicit sweep config that
        # names a stage this part has no kernel for would otherwise fail at
        # dispatch instead of being reported as a downgrade.
        if self.mode is LayerMode.EXP and self._exp_config is not None:
            return self.capabilities.filter(self._exp_config)
        return self.capabilities.filter(measured_config(
            tokens,
            prefill_baseline_max_tokens=self.prefill_baseline_max_tokens,
            prefill_native_expert_min_tokens=(
                self.prefill_native_expert_min_tokens),
        ))

    def _ensure_front_weight(self, front: DecodeFront) -> None:
        if (front is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
                and self._fused_fc1_shared_gate_weight is None):
            self._fused_fc1_shared_gate_weight = torch.cat((
                self._fc1_shard,
                self._shared_gate_up,
                self.gate.weight.data,
            )).contiguous()
            self._fused_fc1_shared_gate_width = (
                self._fc1_shard.shape[0] + self._shared_gate_up.shape[0])

    # ------------------------------------------------------------ front

    def _front(self, h: torch.Tensor, cfg: ExperimentConfig) -> _FrontResult:
        front = cfg.decode_front
        if front is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE:
            self._ensure_front_weight(front)
            merged, logits = dual_out_gemm_cutedsl(
                h, self._fused_fc1_shared_gate_weight,
                self._fused_fc1_shared_gate_width,
                self.gate.weight.shape[0])
            width = self._fc1_shard.shape[0]
            return _FrontResult(
                logits=logits,
                shard=merged[:, :width],
                shared_gate_up=merged[:, width:],
            )
        logits = self.gate(h)
        if front is DecodeFront.SEPARATE_FULL_FC1:
            return _FrontResult(logits=logits, latent=h @ self._fc1_full.T)
        return _FrontResult(logits=logits, shard=h @ self._fc1_shard.T)

    def _gather_quantize(self, front: _FrontResult):
        if front.latent is not None:
            return front.latent
        return self._comm().all_gather_col_quant(front.shard)

    # ---------------------------------------------------------- routing

    def _route(self, logits: torch.Tensor, routing: Routing):
        if routing is Routing.REFERENCE:
            ids, scales = self.experts.routing_method.apply(
                logits.contiguous())
            # run_moe's precomputed handoff requires bf16 scales (the
            # other routings emit bf16 natively via fmt="trtllm_gen").
            return ids, scales.to(torch.bfloat16)
        assert routing is Routing.RADIX, routing
        return route_radix_for_trtllm_gen(logits, self._gate_bias_f32)

    # ----------------------------------------------------------- expert

    def _with_native_backend(self, fn):
        original = self._gen_backend.op_backend
        self._gen_backend.op_backend = self._native_op_backend
        try:
            return fn()
        finally:
            self._gen_backend.op_backend = original

    def _experts(self, latent, ids_scales,
                 backend: ExpertBackend = ExpertBackend.NATIVE,
                 out: torch.Tensor | None = None):
        if isinstance(latent, tuple):
            x, x_scale = latent
        elif backend is ExpertBackend.NATIVE:
            x, x_scale = self._with_native_backend(
                lambda: self._gen_backend.quantize_input(
                    latent.contiguous(), post_quant_comm=False))
        else:
            x, x_scale = self._gen_backend.quantize_input(
                latent.contiguous(), post_quant_comm=False)
        if x_scale is not None and x_scale.dim() == 1:
            x_scale = x_scale.view(x.shape[0], -1)
        ids, scales = ids_scales
        # out: zero-copy AR staging — finalize writes the routed
        # output straight into the latent AR's symm staging view
        # (run_moe's moe_output caller buffer).
        if backend is ExpertBackend.NATIVE:
            return self._with_native_backend(
                lambda: self._gen_backend.run_moe(
                    x, ids, scales, x_sf=x_scale, moe_output=out))
        return self._gen_backend.run_moe(
            x, ids, scales, x_sf=x_scale, moe_output=out)

    # ----------------------------------------------------- shared branch

    def _shared_inline(self, h, gate_up=None, out=None):
        gate_up = (h @ self._shared_gate_up.T
                   if gate_up is None else gate_up)
        act = situ_and_mul(
            gate_up, self._situ_beta, self._situ_linear_beta)
        if out is None:
            return act @ self._shared_down.T
        # Zero-copy AR staging: the down GEMM lands directly in the
        # symm staging view the output AR will consume (out may be a
        # column slice of a packed staging view — strided is fine).
        return torch.mm(act, self._shared_down.T, out=out)

    def _fork_shared(self, h, *, gate_up=None, multimem: bool = False,
                     out=None):
        # out with multimem is safe: the producer GEMM and the pinned
        # AR run back-to-back on the overlap stream, and the staging
        # region sits above the [B, LATENT] partition (see
        # _decode_shared_staging).
        current = torch.cuda.current_stream()
        result = {}
        self._overlap_fork.record(current)
        self._overlap_fork.wait(self._overlap_stream)
        with torch.cuda.stream(self._overlap_stream):
            shared = self._shared_inline(h, gate_up, out)
            if multimem:
                shared = self._comm().all_reduce(
                    shared, impl="torch_symm:multimem")
                if self._comm().world > 1:
                    # multimem_all_reduce_ is in-place on the symm
                    # staging view; evacuate it HERE, hidden under the
                    # expert stage, so the tail can accumulate the fc2
                    # GEMM in place on a private tensor instead of
                    # paying addmm's exposed materialization DtoD
                    # (~2.1 us at B=32 on the main stream).
                    shared = shared.clone()
            result["value"] = shared
        self._overlap_join.record(self._overlap_stream)
        return result

    def _join_shared(self, result):
        self._overlap_join.wait(torch.cuda.current_stream())
        return result["value"]

    # ------------------------------------------------------------- tail

    def _tail(self, routed, shared_ref, tail: DecodeTail):
        comm = self._comm()
        if tail is DecodeTail.PACKED_LATENT_SHARED_REDUCE:
            shared = self._join_shared(shared_ref)
            dst = self._packed_ar_staging(routed.shape[0])
            if dst is not None:
                # Zero-copy staging: pack straight into the AR's symm
                # buffer (kills torch.cat's fresh tensor + stage-in;
                # safe here — under the packed tail the fork never
                # chains an AR, so nothing else stages concurrently).
                dst[:, :MOE_LATENT].copy_(routed)
                dst[:, MOE_LATENT:].copy_(shared)
                packed = comm.all_reduce(dst)
            else:
                packed = comm.all_reduce(
                    torch.cat((routed, shared), dim=-1))
            # The kernel consumes the packed column slice
            # strided — no .contiguous() copy of [B, LATENT].
            reduced = rmsnorm_column_slice_cutedsl(
                packed[:, :MOE_LATENT], self._norm_weight, RMS_EPS)
            return torch.addmm(
                packed[:, MOE_LATENT:], reduced, self._fc2_t)

        reduced = comm.allreduce_norm(
            routed, self._norm_weight, RMS_EPS)
        shared = self._join_shared(shared_ref)
        if tail is DecodeTail.SHARDED_FC2_OUTPUT_REDUCE:
            cols = slice(
                self._rank * self._latent_width,
                (self._rank + 1) * self._latent_width)
            shared.addmm_(reduced[:, cols], self._fc2_shard_t)
            return comm.all_reduce(shared)
        # The multimem tail retains the complete projection above B=16.
        # In-place accumulate: `shared` is always a private tensor here
        # (multimem symm views are evacuated in _fork_shared; the
        # flashinfer/nccl ARs and the world==1 down GEMM allocate fresh
        # outputs), so skipping torch.addmm avoids its materialization
        # copy of [B, HIDDEN] on the main stream.
        shared.addmm_(reduced, self._fc2_t)
        return shared

    # ------------------------------------------------------------ decode

    def _decode_shared_staging(self, tokens: int,
                               impl: str | None = None):
        """Zero-copy staging view for the decode shared AR (the
        fc2_shard tail's output AR, or ref_2ar_mm's pinned multimem
        AR), or None when the impl reads its operand directly or the
        region does not fit. The view sits at element offset
        tokens*LATENT — above any [B, LATENT] region a concurrently
        resolved norm-reduce AR could stage at offset 0 (the same
        partition rule as ``_split_ar_staging``). Shape-only and
        capture-safe."""
        comm = self._comm()
        if comm.world == 1:
            return None
        pick = impl or comm.pick("all_reduce", (tokens, HIDDEN))
        offset = tokens * MOE_LATENT
        if not comm.symm_staged(pick) \
                or offset + tokens * HIDDEN > comm.max_numel:
            return None
        return comm.symm_input((tokens, HIDDEN), impl=pick, offset=offset)

    def _decode(self, hidden_states: torch.Tensor,
                cfg: ExperimentConfig) -> torch.Tensor:
        h = hidden_states.view(-1, self.hidden_dim)
        batch = h.shape[0]
        tail = cfg.decode_tail
        front = self._front(h, cfg)
        multimem_shared = (
            tail is DecodeTail.FULL_FC2_MULTIMEM_SHARED_REDUCE)
        # Zero-copy shared-AR staging: the forked down GEMM produces
        # straight into the AR's symm buffer, so its stage-in copy
        # disappears (fc2_shard: the main-stream output AR after
        # addmm_; ref_2ar_mm: the overlap-stream pinned multimem AR).
        if multimem_shared:
            shared_dst = self._decode_shared_staging(
                batch, "torch_symm:multimem")
        elif tail is DecodeTail.SHARDED_FC2_OUTPUT_REDUCE:
            shared_dst = self._decode_shared_staging(batch)
        else:
            shared_dst = None

        # Front -> fc1 gather/quantize -> routing -> experts -> shared -> tail.
        # Only issue order changes around the independent middle stages.
        if cfg.route_on_side_stream:
            # 3-way overlap: routing (side stream), shared chain
            # (overlap stream), and the column-AG+quantize (main) are
            # mutually independent — all consume only the front. The
            # join lands right before _experts, whose permute-cluster
            # kernel is the first consumer of the ids.
            self._route_fork.record(torch.cuda.current_stream())
            self._route_fork.wait(self._route_stream)
            with torch.cuda.stream(self._route_stream):
                ids_scales = self._route(front.logits, cfg.routing)
            self._route_join.record(self._route_stream)
            shared_ref = self._fork_shared(
                h, gate_up=front.shared_gate_up,
                multimem=multimem_shared, out=shared_dst)
            latent = self._gather_quantize(front)
            self._route_join.wait(torch.cuda.current_stream())
        elif multimem_shared:
            shared_ref = self._fork_shared(
                h, gate_up=front.shared_gate_up, multimem=True,
                out=shared_dst)
            latent = self._gather_quantize(front)
            ids_scales = self._route(front.logits, cfg.routing)
        elif batch <= SMALL_BATCH_MAX_TOKENS:
            ids_scales = self._route(front.logits, cfg.routing)
            shared_ref = self._fork_shared(
                h, gate_up=front.shared_gate_up, out=shared_dst)
            latent = self._gather_quantize(front)
        else:
            latent = self._gather_quantize(front)
            ids_scales = self._route(front.logits, cfg.routing)
            shared_ref = self._fork_shared(
                h, gate_up=front.shared_gate_up, out=shared_dst)

        routed = self._experts(latent, ids_scales)
        output = self._tail(routed, shared_ref, tail)
        return output.view(hidden_states.shape)

    # ----------------------------------------------------------- prefill

    def _packed_ar_staging(self, tokens: int) -> torch.Tensor | None:
        """Zero-copy staging view for a packed [B, LATENT+HIDDEN] AR,
        or None when the pick reads its operand directly (nothing to
        save) or the operand does not fit the staging buffer. Shape-
        only and capture-safe (Collectives.pick)."""
        comm = self._comm()
        if comm.world == 1:
            return None
        width = MOE_LATENT + HIDDEN
        pick = comm.pick("all_reduce", (tokens, width))
        if not comm.symm_staged(pick) or tokens * width > comm.max_numel:
            return None
        return comm.symm_input((tokens, width), impl=pick)

    def _split_ar_staging(self, tokens: int):
        """Zero-copy staging views for the sharded prefill tail's two
        ARs: (routed [B, LATENT], shared [B, HIDDEN]) — either may be
        None (pick reads its operand directly, or no capacity).

        LOAD-BEARING region partition: when both picks stage into the
        SAME buffer, the shared operand lives at element offset
        B*LATENT — above the routed AR's [0, B*LATENT) region. The
        forked shared down GEMM writes its region from the overlap
        stream while the main stream stages and reduces the routed AR
        in the same buffer; DISJOINT offsets are what make that safe
        (and the routed AR's result is consumed by the rmsnorm, on the
        main stream, before the shared AR runs)."""
        comm = self._comm()
        if comm.world == 1:
            return None, None
        pick_lat = comm.pick("all_reduce", (tokens, MOE_LATENT))
        pick_out = comm.pick("all_reduce", (tokens, HIDDEN))
        routed_dst = shared_dst = None
        if comm.symm_staged(pick_lat) \
                and tokens * MOE_LATENT <= comm.max_numel:
            routed_dst = comm.symm_input(
                (tokens, MOE_LATENT), impl=pick_lat)
        offset = (tokens * MOE_LATENT
                  if comm.staging_overlaps(pick_lat, pick_out) else 0)
        if comm.symm_staged(pick_out) \
                and offset + tokens * HIDDEN <= comm.max_numel:
            shared_dst = comm.symm_input(
                (tokens, HIDDEN), impl=pick_out, offset=offset)
        return routed_dst, shared_dst

    def _prefill_full(self, hidden_states: torch.Tensor,
                      cfg: ExperimentConfig) -> torch.Tensor:
        h = hidden_states.view(-1, self.hidden_dim)
        if cfg.prefill_tail is PrefillTail.FC2_SHARD:
            # fc2-shard tail on the full-fc1 front: same wire as the
            # packed tail (AR 3584 + AR 7168 == packed AR 10752) but
            # 1/8 the fc2 FLOPs. Measured graph-time win at 512..4095
            # tokens; output bit-identical to _prefill_sharded's tail.
            routed_dst, shared_dst = self._split_ar_staging(h.shape[0])
            shared_ref = (
                self._fork_shared(h, out=shared_dst)
                if cfg.prefill_overlap_shared_branch else None)
            ids_scales = self._route(self.gate(h), cfg.routing)
            routed = self._experts(
                h @ self._fc1_full.T, ids_scales,
                cfg.prefill_expert_backend, out=routed_dst)
            reduced = rmsnorm_column_slice_cutedsl(
                self._comm().all_reduce(routed), self._norm_weight,
                RMS_EPS, self._rank * self._latent_width,
                self._latent_width)
            shared = (self._join_shared(shared_ref)
                      if shared_ref is not None
                      else self._shared_inline(h, out=shared_dst))
            shared.addmm_(reduced, self._fc2_shard_t)
            return self._comm().all_reduce(shared).view(
                hidden_states.shape)
        packed_dst = self._packed_ar_staging(h.shape[0])
        shared_dst = (packed_dst[:, MOE_LATENT:]
                      if packed_dst is not None else None)
        shared_ref = (
            self._fork_shared(h, out=shared_dst)
            if cfg.prefill_overlap_shared_branch else None)
        ids_scales = self._route(self.gate(h), cfg.routing)
        routed = self._experts(
            h @ self._fc1_full.T, ids_scales, cfg.prefill_expert_backend)
        shared = (self._join_shared(shared_ref)
                  if shared_ref is not None
                  else self._shared_inline(h, out=shared_dst))
        if packed_dst is not None:
            # Zero-copy packed AR: the shared half was produced in
            # place by the down GEMM; only the routed half is copied
            # (run_moe cannot finalize into a strided column slice) —
            # torch.cat's fresh tensor AND the AR stage-in are gone.
            packed_dst[:, :MOE_LATENT].copy_(routed)
            packed = self._comm().all_reduce(packed_dst)
        else:
            packed = self._comm().all_reduce(
                torch.cat((routed, shared), dim=-1))
        # The kernel consumes the packed column slice strided —
        # no .contiguous() copy of [B, LATENT].
        reduced = rmsnorm_column_slice_cutedsl(
            packed[:, :MOE_LATENT], self._norm_weight, RMS_EPS)
        return torch.addmm(
            packed[:, MOE_LATENT:], reduced, self._fc2_t).view(
            hidden_states.shape)

    def _prefill_sharded(self, hidden_states: torch.Tensor,
                         cfg: ExperimentConfig) -> torch.Tensor:
        h = hidden_states.view(-1, self.hidden_dim)
        current = torch.cuda.current_stream()
        shard = h @ self._fc1_shard.T
        logits = self.gate(h)

        # Multimem gather on the prefill stream. In-layer A/B across
        # gather impls (3 repeats each): multimem beats the DMA mover
        # +37/+4/+217 us at 2048/4096/16384 and is stable where DMA
        # flaps at 16K; DMA wins only at 8192 (-20 us, conceded for
        # the one-rule simplicity — open corner). The gather returns
        # rank-major [world*B, width] blocks as a view of the symm
        # staging; the fused gather_quant kernel below interleaves and
        # quantizes it in one pass.
        self._prefill_fork.record(current)
        self._prefill_fork.wait(self._prefill_stream)
        with torch.cuda.stream(self._prefill_stream):
            gathered = self._comm().all_gather(
                shard, impl="torch_symm:multimem")
        self._prefill_join.record(self._prefill_stream)

        # Zero-copy AR staging (the column AG above owns its own
        # Lamport buffers and never touches these regions).
        routed_dst, shared_dst = self._split_ar_staging(h.shape[0])
        shared_ref = (
            self._fork_shared(h, out=shared_dst)
            if cfg.prefill_overlap_shared_branch else None)
        self._prefill_join.wait(current)
        # Fused interleave + MXFP8 quantize (kernels/gather_quant.py):
        # one read of the rank-major blocks, one write of the fp8
        # payload + swizzled scales — bit-exact vs interleave-copy +
        # trtllm mxfp8_quantize and 5..119 us faster at 512..16384
        # (kernels/gather_quant_results.md). Evacuates the gather's
        # symm-staging view before the tail ARs touch the pool.
        latent = gather_quant_mxfp8(gathered, self._world)
        ids_scales = self._route(logits, cfg.routing)
        routed = self._experts(
            latent, ids_scales, cfg.prefill_expert_backend,
            out=routed_dst)
        # Column-window norm: the fc2 shard consumes only this rank's
        # [B, latent_width] slice of the normalized latent, so skip
        # 7/8 of the norm's output write (row reduction still spans
        # the full row).
        reduced = rmsnorm_column_slice_cutedsl(
            self._comm().all_reduce(routed), self._norm_weight, RMS_EPS,
            self._rank * self._latent_width, self._latent_width)
        shared = (self._join_shared(shared_ref)
                  if shared_ref is not None
                  else self._shared_inline(h, out=shared_dst))
        shared.addmm_(reduced, self._fc2_shard_t)
        return self._comm().all_reduce(shared).view(hidden_states.shape)

    def forward(self, hidden_states: torch.Tensor, attn_metadata=None,
                **kwargs) -> torch.Tensor:
        if not self._optimized:
            return self.baseline_forward(
                hidden_states, attn_metadata, **kwargs)
        tokens = hidden_states.view(-1, self.hidden_dim).shape[0]
        cfg = self._config(tokens)
        if not cfg.enabled:
            return self.baseline_forward(
                hidden_states, attn_metadata, **kwargs)
        if tokens <= min(DECODE_MAX_TOKENS, self._max_batch):
            return self._decode(hidden_states, cfg)
        if cfg.prefill_fc1 is PrefillFC1.SHARDED:
            return self._prefill_sharded(hidden_states, cfg)
        return self._prefill_full(hidden_states, cfg)


# Short aliases for callers migrating from the old benchmark.
KimiK3MoE = KimiK3MoEReference
KimiK3MoEB10 = B10KimiK3MoELayer

__all__ = [
    "B10KimiK3MoELayer",
    "DecodeFront",
    "DecodeTail",
    "ExpertBackend",
    "ExperimentConfig",
    "KimiK3MoE",
    "KimiK3MoEB10",
    "KimiK3MoEReference",
    "LayerMode",
    "PrefillFC1",
    "Routing",
    "k3_model_config",
    "measured_config",
]
