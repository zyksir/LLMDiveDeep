"""Self-contained Kimi-K3 MoE production blocks for trace alignment.

Three blocks, one per serving stack/revision, all running on this bench's
weights and launch conventions (see ``bench_b10_kimi_k3_moe_layer.py``):

* :class:`TrtKimiK3MoEBlock` — the TRT-LLM fork's production decode MoE,
  pinned to the b10-1.3.0rc19 deployment revision.
* :class:`TrtRc25KimiK3MoEBlock` — upstream NVIDIA TensorRT-LLM
  v1.3.0rc25's K3 decode MoE forward (``KimiK3MoERuntime``), replicated
  against the installed rc19 runtime's modules.
* :class:`SglKimiK3MoEBlock` — sglang's Kimi-K3 decode MoE staged on the
  vendored sglang kernels (``kimi_k3/kernels/sgl_copied_kernels``).

The former bench research classes (KimiK3MoEReference, B10KimiK3MoELayer,
KimiK3StockPlusFront and the ExperimentConfig machinery) were removed
2026-09-10 after the B10 research concluded; their measured plans live in
``kimi_k3/local_results/`` and the git history.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_k3 import (
    KimiK3MoE as TrtKimiK3MoE,
)
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.utils import AuxStreamType, EventType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from kimi_k3.config import (
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
    cfg.latent_moe_use_norm = True
    cfg.mlp_bias = False
    cfg.rms_norm_eps = RMS_EPS
    # Match the released checkpoint (moonshotai/Kimi-K3 config.json) and
    # feat/k3 serving exactly: hidden_act="situ" makes TRT-LLM's
    # _is_situ_activation TRUE, so the experts run the SiTU cubins the
    # production engine runs -- with the old defaults the bench baseline
    # ran SwiGlu experts and was not measuring feat/k3's path (see the
    # arch-gaps note in kimi_k3/README.md). Requires a FlashInfer with
    # ActivationType.Situ (>= the fork's v0.6.18rc1 pin).
    cfg.hidden_act = "situ"
    cfg.activation_situ_beta = 4.0
    cfg.activation_situ_linear_beta = 25.0
    return cfg


def k3_model_config(rank: int, world: int,
                    moe_backend: str = "TRTLLM",
                    moe_mode: str = "ep") -> ModelConfig:
    """moe_mode: "ep" = whole experts per rank (production);
    "tp" = every expert on every rank, inter-dim sliced 1/world —
    identical weight bytes per step at uniform routing, but perfectly
    load-balanced (no expert-imbalance rank skew)."""
    ep, mtp = (world, 1) if moe_mode == "ep" else (1, world)
    mapping = Mapping(
        world_size=world,
        tp_size=world,
        rank=rank,
        gpus_per_node=min(world, torch.cuda.device_count()),
        moe_ep_size=ep,
        moe_tp_size=mtp,
    )
    # Keep the replicated latent projections and shared experts in BF16 while
    # selecting the released checkpoint's packed MXFP4 format for routed
    # experts. NemotronHMOE consumes this per-expert override directly.
    expert_quant = QuantConfig(
        quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8,
    )
    return ModelConfig(
        pretrained_config=k3_pretrained_config(),
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            "model.layers.0.mixer.experts.": expert_quant,
        },
        moe_backend=moe_backend,
        allreduce_strategy=os.environ.get("BENCH_ALLREDUCE_STRATEGY", "AUTO"),
    )


def _bench_doctor_logits(logits: torch.Tensor) -> torch.Tensor:
    """Bench-only routing realism hooks.

    K3_DEBUG_CONC_ROUTING=1   collapse to the first TOP_K experts
                              (dummy-weight serving's degenerate limit).
    K3_BENCH_EXPERT_IMBALANCE=S  bias the experts owned by
                              K3_BENCH_HEAVY_RANKS (default "5,6") by
                              log(S): heavy ranks receive ~S x the
                              routed load, reproducing the measured
                              serving spread (bmm medians 7.2->15.2 us
                              across ranks at S~2) with the skew INSIDE
                              the expert stage where it belongs.
    """
    if os.environ.get("K3_DEBUG_CONC_ROUTING", "0") == "1":
        logits = logits.clone()
        logits[:, TOP_K:] = logits[:, TOP_K:] - 1e4
        return logits
    imb = float(os.environ.get("K3_BENCH_EXPERT_IMBALANCE", "0") or 0)
    if imb > 1.0:
        import math
        world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "8"))
        heavy = [int(r) for r in os.environ.get(
            "K3_BENCH_HEAVY_RANKS", "5,6").split(",")]
        e_local = NUM_EXPERTS // world
        logits = logits.clone()
        for r in heavy:
            logits[:, r * e_local:(r + 1) * e_local] += math.log(imb)
    return logits


class TrtKimiK3MoEBlock(TrtKimiK3MoE):
    """Self-contained production Kimi-K3 MoE block pinned to b10-1.3.0rc19.

    Tracks TRT-LLM commit d9fa74fd94778047f701f708fdb82035beccd674
    ("kimi k3 optimization", tip of origin/optimized/k3,
    tensorrt_llm.__version__ == "1.3.0rc19") — the revision shipped in the
    deployment image
    ``baseten/dynamo-cache-aware-routing:trtllm-d9fa74fd94-d9d1a655f-c08257485a``
    (the image's modeling_kimi_k3.py and modeling_nemotron_h.py were
    byte-compared against that commit's git blobs, 2026-09-10).

    It subclasses the CURRENT checkout's KimiK3MoE (12e51f84e1, 2026-09-08)
    because the default (TRTLLM_KIMI_B10_MOE=0) forward path is equivalent
    between the two revisions: NemotronHMOE.forward is byte-identical, and
    the only diffs are (a) rc19's TRTLLM_KIMI_B10_MOE env switch, removed at
    HEAD, (b) HEAD's multi-node MNNVL MoEAllReduce workspace — identical
    behavior on a single node, and (c) a FlashinferOpBackend pre-packed
    topk_ids fast path present only at rc19, never exercised here because
    this path hands RAW router logits to the experts op (in-kernel routing).

    The block constructs the module exactly the way KimiK3DecoderLayer does
    in production — reduce_output = (tp_size > 1 and not attention-DP) — so
    NemotronHMOE.forward natively runs the rc19 production path through
    TRT-LLM's own AllReduce/MoEAllReduce ops.
    Decode (tokens <= max_fused_moe_tokens): experts(do_finalize=False) ->
    ONE fused finalize+AR+rmsnorm+concat kernel -> full fc2_latent_proj.
    Beyond the fused-epilogue cap (or when the epilogue is unavailable):
    packed AR of cat([latent | shared]) -> latent_norm -> fc2_latent_proj.
    defer_shared_output_add=False so the block returns one summed tensor
    (production defers the shared+routed add into the next AttnRes; the
    math is identical).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        *,
        collectives=None,
    ) -> None:
        mapping = model_config.mapping
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            # Production wiring (modeling_kimi_k3.KimiK3DecoderLayer):
            # has_tp_reduce = not enable_attention_dp and tp_size > 1.
            reduce_output=(not mapping.enable_attention_dp
                           and mapping.tp_size > 1),
            defer_shared_output_add=False,
        )
        # Harness plumbing parity with SglKimiK3MoEBlock: the production
        # forward reduces through TRT-LLM's own AllReduce/MoEAllReduce, so
        # the Collectives handle is held only for bench-side interop.
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

    def forward(self, hidden_states: torch.Tensor, attn_metadata=None,
                **kwargs) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = SimpleNamespace(
                all_rank_num_tokens=kwargs.get("all_rank_num_tokens"),
            )
        return super().forward(hidden_states, attn_metadata, **kwargs)


# Same A/B escape hatch (and env name) as upstream rc25's
# modeling_kimi_linear.py: "1" restores the plain Linear call for the K3
# latent down/up projections instead of the min-latency fused GEMM op.
# Read once at import, exactly like upstream.
_RC25_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)


class TrtRc25KimiK3MoEBlock(TrtKimiK3MoEBlock):
    """Upstream NVIDIA TensorRT-LLM v1.3.0rc25 K3 decode MoE, on rc19 ops.

    Replicates ``KimiK3MoERuntime.forward`` from upstream tag
    ``v1.3.0rc25`` (commit 785c948197b55267260fec3f7f52e47000888d0a,
    ``tensorrt_llm/_torch/models/modeling_kimi_linear.py``) inside this
    harness, executed against the installed rc19 runtime
    (b10 fork d9fa74fd94, the deployment image). REPLICATED LOGIC, not an
    alias: rc25's forward is structurally different from rc19's, so the
    two cannot share a code path (and one process cannot import two
    tensorrt_llm versions).

    rc19 -> rc25 upstream differences for the K3 MoE path (diffed
    2026-09-10 against the fork's ``modeling_kimi_k3.py``/
    ``modeling_nemotron_h.py`` at d9fa74fd94):

    * Upstream has no ``modeling_kimi_k3.py``/``KimiK3ForCausalLM``. K3 is
      ``KimiK3ForConditionalGeneration`` (VL) whose text backbone is
      ``KimiLinearForCausalLM``; the MoE block is ``KimiK3MoERuntime``
      backed by the unified ConfigurableMoE stack.
    * DECODE TAIL: rc25 has NO fused finalize+AR+RMSNorm+concat epilogue
      (rc19's ``MoEAllReduce.finalize_allreduce_rmsnorm_concat`` decode
      path for tokens <= max_fused_moe_tokens is absent upstream). The
      experts always finalize in-kernel (``do_finalize`` defaults True),
      then ONE AllReduce over ``cat([shared 7168 | latent 3584])`` ->
      split -> RMSNorm(latent.contiguous()) -> up-projection -> add.
      This is rc19's beyond-the-cap fallback branch — with the concat
      order swapped to shared-first — promoted to the only path. The
      combined AR runs through the experts' own ``MoE.all_reduce``
      (identically constructed in both revisions).
    * Latent projections: rc25 routes the fc1/fc2 latent GEMMs through
      ``torch.ops.trtllm.dsv3_fused_a_gemm_op`` (byte-identical thop op
      in rc19 and rc25; its custom min-latency kernel covers only
      7168->2112, so K3's 7168->3584 / 3584->7168 shapes take the op's
      internal ``cublas_mm_out`` path — a different cuBLAS pick than the
      TRT-LLM Linear wrapper rc19 uses). ``TLLM_K3_DISABLE_MIN_LATENCY_
      LATENT_PROJ=1`` restores the Linear call, as upstream.
    * Router: rc25's ``KimiK3MoEGate.compute_logits`` (bf16-stored gate
      weight -> ``dsv3_router_gemm_op`` with fp32 out when attention-DP
      is off) computes exactly what rc19's ``DeepseekV3Gate.forward``
      already computes on this config — same op, same dtypes — so the
      inherited gate module IS the rc25 router here.
    * MoE parallelism: rc25 defaults to EP-only (moe_tp=1,
      moe_ep=tp_size) unless the user overrides — the bench's "ep" mode.
    * Removed at rc25: rc19's ``TRTLLM_KIMI_B10_MOE`` switch and the
      FlashinferOpBackend pre-packed topk_ids fast path.
    * New at rc25, not replicated (all default-off under TP): opt-in FP8
      block-scale weight reads for the replicated projections
      (``KIMI_K3_FP8_WEIGHT_READ*``), NVFP4 routed-expert checkpoints,
      MegaMoE backends, per-layer checkpoint quant resolution.

    NAMED DEVIATIONS (rc25 behavior not reachable on the rc19 runtime):

    * experts engine — inherited from rc19 ``KimiK3MoE.__init__``
      (``hidden_act="situ"`` -> ``_is_situ_activation`` -> SiTU cubins)
      instead of rc25's explicit ``trtllm_gen_activation_type=SiTu`` +
      alpha/beta ``create_moe`` kwargs, which rc19's factory lacks. Same
      trtllm-gen SiTU cubins, same beta values (from the config).
    * router bias dtype — rc19's TRTLLM-backend gate stores
      ``e_score_correction_bias`` in bf16; rc25's gate keeps fp32. Both
      are consumed by the same in-kernel trtllm-gen routing.
    * the inherited ``__init__`` may still construct rc19's MoEAllReduce
      fused-epilogue workspace (env ``TLLM_K3_FUSED_MOE_FINALIZE_
      ALLREDUCE`` default on); this forward never uses it.
    * ``lora_params`` plumbing dropped (always None in this bench), and
      routing sees the bench realism hook ``_bench_doctor_logits``
      (identity unless its env knobs are set) like the Sgl block.
    """

    def _rc25_latent_proj(self, x: torch.Tensor, projection) -> torch.Tensor:
        """rc25 ``KimiK3MoERuntime._routed_projection``: the min-latency
        fused GEMM op over the projection's weight (bias-free, bf16); the
        op itself falls back to cuBLAS for K3's non-2112 out dims."""
        if _RC25_DISABLE_MIN_LATENCY_LATENT_PROJ:
            return projection(x)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            x, projection.weight.t(), None, None)

    def forward(self, hidden_states: torch.Tensor, attn_metadata=None,
                **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        h = hidden_states.view(-1, self.hidden_dim)
        all_rank_num_tokens = kwargs.get(
            "all_rank_num_tokens",
            getattr(attn_metadata, "all_rank_num_tokens", None))

        # rc25 _use_combined_all_reduce: direct MoE-TP leaves both branches
        # as partials for one concatenated all-reduce.
        use_combined_all_reduce = (not self.mapping.enable_attention_dp
                                   and self.mapping.tp_size > 1)
        moe_all_reduce = (self.experts.all_reduce
                          if use_combined_all_reduce else None)
        if use_combined_all_reduce and moe_all_reduce is None:
            raise RuntimeError(
                "rc25 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False.")

        logits = _bench_doctor_logits(self.gate(h))

        def _routed_output():
            routed_in = self._rc25_latent_proj(h, self.fc1_latent_proj)
            # rc25 always finalizes in-kernel (do_finalize defaults True);
            # no fused finalize-AR epilogue exists upstream.
            y = self.experts(
                routed_in,
                logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            if use_combined_all_reduce:
                return y
            # Communication-backed / single-rank paths return a complete
            # routed result: norm + up-projection inline.
            y = self.latent_norm(y)
            return self._rc25_latent_proj(y, self.fc2_latent_proj)

        # Shared experts overlap the routed chain on the MoeShared side
        # stream, exactly rc25's maybe_execute_in_parallel wiring (the
        # inherited event pair and aux stream are the same machinery
        # KimiK3MoERuntime allocates for itself).
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            lambda: self.shared_experts(h),
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream_shared,
            disable_on_compile=True,
        )

        if use_combined_all_reduce:
            combined = moe_all_reduce(
                torch.cat((shared_out, routed_out), dim=-1))
            shared_out, routed_latent = torch.split(
                combined,
                (self.hidden_dim, self.moe_hidden_size),
                dim=-1,
            )
            # The column split is a strided view; the fused RMSNorm expects
            # a dense last dimension (rc25 comment, kept verbatim).
            routed_latent = self.latent_norm(routed_latent.contiguous())
            routed_out = self._rc25_latent_proj(
                routed_latent, self.fc2_latent_proj)
        return (routed_out + shared_out).view(orig_shape)


class _SglWeight(torch.nn.Module):
    """Bare replicated-weight holder (sglang ReplicatedLinear surface: just
    a ``.weight`` for the bench init to copy into; the owning block applies
    it via F.linear)."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False)


class _SglMoEGate(torch.nn.Module):
    """kimi_k3.py MoEGate surface: bf16 router weight + fp32 noaux_tc
    correction bias. Logits are produced in fp32 (linear + cast here; the
    serving gate GEMM writes fp32 directly — the cast is a known one-kernel
    residual of keeping the front unfused)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16),
            requires_grad=False)
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.empty(NUM_EXPERTS, dtype=torch.float32),
            requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(h, self.weight).float()


class _SglRMSNorm(torch.nn.Module):
    """sglang RMSNorm surface (weight + variance_epsilon). Applied through
    F.rms_norm on the fallback tail only; the fused tail norms in-kernel."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(dim, dtype=torch.bfloat16), requires_grad=False)
        self.variance_epsilon = eps


class _SglSharedExperts(torch.nn.Module):
    """kimi_k3.py KimiK3MLP: TP-sharded SiTU MLP with reduce_results=False
    (gate_up GEMM -> vendored sglang situ_and_mul -> down GEMM partial).
    Weight layout matches the bench init: gate rows then up rows along
    dim 0 of ``gate_up_proj.weight``."""

    def __init__(self, intermediate_local: int, beta: float,
                 linear_beta: float) -> None:
        super().__init__()
        self.gate_up_proj = _SglWeight(2 * intermediate_local, HIDDEN)
        self.down_proj = _SglWeight(HIDDEN, intermediate_local)
        self._beta = float(beta)
        self._linear_beta = float(linear_beta)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        from .kernels.sgl_adapters import activation as sgl_act
        gate_up = torch.nn.functional.linear(h, self.gate_up_proj.weight)
        act = sgl_act.situ_and_mul(
            gate_up, None, self._beta, self._linear_beta)
        return torch.nn.functional.linear(act, self.down_proj.weight)


class SglKimiK3MoEBlock(torch.nn.Module):
    """sglang's Kimi-K3 decode MoE block, on this bench's weights/collectives.

    Standalone ``torch.nn.Module`` — no TRT-LLM model inheritance — following
    sglang commit f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e:
    ``python/sglang/srt/models/kimi_k3.py`` (KimiK3MoE) and
    ``python/sglang/srt/layers/k3_ar_fusion.py``. All non-experts weights
    (gate, fc1/fc2 latent projections, latent norm, shared experts) are
    plain parameters owned here with the kimi_k3.py structure; the sglang
    kernels are vendored under ``kimi_k3/kernels/sgl_copied_kernels``
    (see its README for per-file provenance) and reached lazily through the
    ``kernels/sgl_adapters`` package — the single sanctioned surface over
    the vendored sglang stack.

    Stage mapping (kimi_k3.py -> here):

    * front — MoEGate fp32 logits + ``routed_expert_down_proj`` (here
      ``gate`` + ``fc1_latent_proj``). sglang's decode path merges the two
      GEMMs into one fused-front weight (``_forward_fused``); this block
      keeps them separate like ``_forward_unfused`` — the merged-front idea
      is a bench front concern already covered by B10's dual-out front.
    * route + activation quant — the vendored
      ``ops/moe/moe_route_quant_fused.py`` kernel, sglang's
      ``route_quant_handoff`` hit (``_route_quant_fuse_eligible`` ->
      staged fused kernel): ONE launch producing the renormalized top-16,
      the trtllm ``(expert<<16)|bf16(weight)`` packed ids, and the group-32
      MXFP8 latent quant (x_q + row-major UE8M0 scales). Consumed the way
      ``mxfp4.py::_prepare_flashinfer_mxfp8_activations`` consumes the
      handoff: packed ids with ``token_final_scales=None`` take the op
      backend's pre-packed fast path (deployment rc19 revision; no eager
      lshift/bitwise-or repack) and the pre-quantized (x, x_sf) skips
      ``quantize_with_block_size``.
    * routed experts — a composed TRT-LLM ``TRTLLMGenFusedMoE`` engine
      (``create_moe`` with this bench's W4A8 MXFP4xMXFP8 quant config), the
      same trtllm-gen SiTU cubins sglang reaches through flashinfer's
      ``trtllm_fp4_block_scale_routed_moe``. NAMED DEVIATION: sglang calls
      flashinfer directly; this block holds the TRT engine as an experts
      submodule because the bench weight init (MXFP4 shuffle, scale layout,
      ``post_load_weights``) is built on its parameter surface. It runs with
      ``do_finalize=False`` when the deferred-finalize AR is active.
    * shared experts — sglang ``KimiK3MLP`` (SiTU, TP-sharded) forked onto
      the MoeShared aux stream, sglang's SBO ``issue_shared`` overlap.
    * tail — sglang's selection logic: when the lazily resolved
      CustomAllReduceV2 state is live (``kernels.sgl_adapters.comm.
      get_sgl_ar_state`` on first forward — sglang's ``k3_ar_fusion.
      _get_state`` pattern; or injected via ``attach_sgl_ar``) and the
      [T, 3584] bf16 latent fits the 512 KB push window
      (``finalize_push_fits``), the
      trtllm-gen finalize is DEFERRED and fused into the 1shot push AR +
      RMSNorm (vendored ``finalize_all_reduce_push_norm`` — the rank-local
      latent never materializes); otherwise AR(latent) -> RMSNorm ->
      fc2_latent_proj -> AR(shared) -> add, exactly
      ``KimiK3MoE._forward_unfused``'s plain-TP latent tail.

    Graceful degradation (no faked numerics): sglang's fused-AR kernels need
    the CustomAllReduceV2 multicast workspaces (push/pull planes), which the
    block stands up itself on first forward through
    ``get_sgl_ar_state()`` (collective; all ranks reach forward in
    lockstep). When that resolves None — no NVLS multicast, unsupported
    arch/world size, world == 1 — the block runs the fallback tail through
    the bench ``Collectives`` instance. Even in the fused path the
    shared-expert AR stays on ``Collectives``/push (sglang reduces it with a
    low-SM NVLS pull on a symm-buffer slice, which needs the same
    workspaces); the reduced value is identical.
    """

    # k3_ar_fusion.py _PUSH_MAX_BYTES: push wins below ~0.5 MB on
    # B200x8/GB300; the finalize-fused AR is push-only.
    _PUSH_MAX_BYTES = 512 * 1024
    # kNormDim in the vendored ar_fusion.cuh; the fused norm hardcodes
    # this latent row width.
    _NORM_DIM = 3584

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int = 0,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
        reduce_output: bool = False,
        *,
        collectives=None,
    ) -> None:
        super().__init__()
        cfg = model_config.pretrained_config
        self.mapping = model_config.mapping
        self.layer_idx = layer_idx
        self.hidden_dim = int(cfg.hidden_size)
        self.moe_hidden_size = int(cfg.moe_latent_size)
        # All-reduces are explicit in forward; reduce_output is accepted
        # for constructor parity with TrtKimiK3MoEBlock (True <=> TP>1).
        self.reduce_output = reduce_output
        self._collectives = collectives

        self.gate = _SglMoEGate()
        self.fc1_latent_proj = _SglWeight(MOE_LATENT, HIDDEN)
        self.fc2_latent_proj = _SglWeight(HIDDEN, MOE_LATENT)
        self.latent_norm = _SglRMSNorm(MOE_LATENT, float(cfg.rms_norm_eps))
        self.shared_experts = _SglSharedExperts(
            SHARED_INTER // self.mapping.tp_size,
            cfg.activation_situ_beta,
            cfg.activation_situ_linear_beta,
        )
        self.experts = self._build_experts_engine(
            model_config, layer_idx, aux_stream_dict)

        # sglang SBO: the shared branch runs on the MoeShared side stream.
        self._shared_stream = (
            aux_stream_dict.get(AuxStreamType.MoeShared)
            if aux_stream_dict else None)

        # Fused-AR tail state (kernels.sgl_adapters.comm.SglArState): None
        # until the first forward resolves it via get_sgl_ar_state() or a
        # caller injects one via attach_sgl_ar(). _sgl_ar_resolved
        # distinguishes "not probed yet" from "probed: unavailable".
        self._sgl_ar_state = None
        self._sgl_ar_resolved = False
        self._sgl_max_push_bytes = 0
        self._sgl_backend = None

    def _build_experts_engine(self, model_config, layer_idx, aux_stream_dict):
        """Composed routed-experts engine (the NAMED DEVIATION above):
        TRT-LLM ``create_moe`` wired the way KimiK3MoE/NemotronHMOE wires
        it, with the per-expert quant config resolved from
        ``quant_config_dict`` by the same key-prefix rule."""
        from dataclasses import replace as dc_replace

        from tensorrt_llm._torch.modules.fused_moe import (
            MoEWeightLoadingMode, create_moe)
        from tensorrt_llm._torch.modules.fused_moe.routing import (
            DeepSeekV3MoeRoutingMethod)
        from tensorrt_llm._torch.utils import ActivationType

        moe_model_config = model_config
        if model_config.quant_config_dict is not None:
            experts_prefix = f"model.layers.{layer_idx}.mixer.experts."
            for name, quant_cfg in model_config.quant_config_dict.items():
                if name.startswith(experts_prefix):
                    moe_model_config = dc_replace(
                        model_config, quant_config=quant_cfg)
                    break
        # run_moe only reads top_k off the routing method here (routing and
        # scaling happen in the vendored fused kernel), but the engine is
        # constructed with the true K3 method for parity with serving.
        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=TOP_K,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            routed_scaling_factor=float(ROUTED_SCALING),
            callable_e_score_correction_bias=(
                lambda: self.gate.e_score_correction_bias),
        )
        return create_moe(
            routing_method=routing_method,
            num_experts=NUM_EXPERTS,
            hidden_size=self.moe_hidden_size,
            intermediate_size=MOE_INTER,
            aux_stream_dict=aux_stream_dict,
            dtype=model_config.pretrained_config.torch_dtype,
            reduce_results=False,
            model_config=moe_model_config,
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            bias=False,
            activation_type=ActivationType.Swiglu,
        )

    # ----------------------------------------------------- bench plumbing

    def attach_collectives(self, collectives) -> None:
        self._collectives = collectives

    def _comm(self):
        if self._collectives is None:
            if self.mapping.tp_size != 1:
                raise RuntimeError("TP>1 requires a Collectives instance")
            from communication.collective import Collectives
            self._collectives = Collectives(None, 0)
        return self._collectives

    # ---------------------------------------------------- fused-AR setup

    def attach_sgl_ar(self, state, *, max_push_bytes=None) -> None:
        """Inject a prebuilt fused-AR state (overrides the lazy self-init).

        ``state`` is the ``SglArState`` from
        ``kernels.sgl_adapters.comm`` (``build_sgl_ar_state`` /
        ``get_sgl_ar_state``); it owns the CustomAllReduceV2 workspaces.
        Normally the block resolves this itself on first forward — the hook
        remains for tests/benches injecting a custom-built state (e.g.
        non-default push-slot sizes).
        """
        if self.moe_hidden_size != self._NORM_DIM:
            raise RuntimeError(
                "sglang ar_fusion hardcodes the K3 latent width "
                f"{self._NORM_DIM}; this layer has {self.moe_hidden_size}")
        from .kernels.sgl_adapters import all_reduce as sgl_ar
        # Idempotent for the same communicator; the state constructors
        # already registered it.
        sgl_ar.register_comm(state.comm)
        self._sgl_ar_state = state
        self._sgl_ar_resolved = True
        self._sgl_max_push_bytes = int(
            max_push_bytes if max_push_bytes is not None
            else state.max_push_size)

    def _ensure_sgl_ar(self) -> None:
        """Resolve the fused-AR state once, lazily, on first forward —
        exactly how sglang's fused tail reaches ``k3_ar_fusion._get_state``.
        All ranks reach forward in lockstep, so the collective build inside
        ``get_sgl_ar_state`` is safe; an "unavailable" verdict is cached and
        the block keeps its documented fallback tail."""
        if self._sgl_ar_resolved:
            return
        self._sgl_ar_resolved = True
        from .kernels.sgl_adapters import comm as sgl_comm
        state = sgl_comm.get_sgl_ar_state()
        if state is not None:
            self.attach_sgl_ar(state)

    def _finalize_push_fits(self, num_tokens: int) -> bool:
        """k3_ar_fusion.finalize_push_fits: the deferred-finalize AR is
        push-only, so the [T, 3584] bf16 latent must fit a push slot."""
        if self._sgl_ar_state is None:
            return False
        nbytes = num_tokens * self._NORM_DIM * 2
        return nbytes <= min(self._PUSH_MAX_BYTES, self._sgl_max_push_bytes)

    # -------------------------------------------------------- components

    def _sgl_gen_backend(self):
        if self._sgl_backend is None:
            backend = getattr(self.experts, "backend", self.experts)
            if type(backend).__name__ != "TRTLLMGenFusedMoE":
                raise TypeError(
                    "SglKimiK3MoEBlock requires TRTLLMGenFusedMoE experts "
                    "(the trtllm-gen MXFP4 runner sglang uses via "
                    "flashinfer)")
            self._sgl_backend = backend
        return self._sgl_backend

    def _sgl_route_quant(self, logits: torch.Tensor,
                         routed_input: torch.Tensor):
        """sglang's fused MoE-front prep: ONE vendored launch = DSv3-style
        noaux_tc routing (sigmoid + fp32 bias top-16, renorm +
        routed_scaling in-kernel), the trtllm ``(expert<<16)|bf16(weight)``
        id pack, and the group-32 MXFP8 quant of the latent activations —
        the ``route_quant_fused_kernel`` a route_quant_handoff hit runs in
        serving.

        Returns run_moe's routed-input 4-tuple ``(token_selected_experts,
        token_final_scales, x, x_sf)``. On the fused hit that is
        ``(packed, None, x_q fp8, scales viewed fp8)``: scales=None routes
        the rc19 op backend to its pre-packed fast path (no eager
        lshift/bitwise-or repack) and the pre-quantized x skips
        ``quantize_with_block_size``, so the kernel sequence matches the
        serving baseline.
        """
        from .kernels.sgl_adapters import moe as sgl_moe

        bias = self.gate.e_score_correction_bias
        scores = logits[:, :NUM_EXPERTS]
        if sgl_moe.route_quant_fused_covered(scores, bias, TOP_K,
                                             routed_input):
            _w, _ids, packed, x_q, x_s = (
                sgl_moe.route_quant_fused(
                    scores, bias, routed_input, TOP_K,
                    renormalize=True,
                    routed_scaling_factor=float(ROUTED_SCALING),
                    apply_scale=True,
                ))
            # x_s is the row-major packed UE8M0 buffer (int32 [M, 28])
            # viewed as the 2D fp8 tensor run_moe expects — byte-identical
            # to sglang's ``x_scale.view(torch.float8_e4m3fn)`` handoff.
            return packed, None, x_q, x_s.view(torch.float8_e4m3fn)

        # Documented fallback — the same misses that fall through to the
        # unfused chain in sglang's route_quant_handoff (fused kernel's
        # decode cap M>64, non-3584 rows, misaligned base/stride): vendored
        # RouteRadixKernel for routing plus the engine's own quantize_input.
        # run_moe then packs the ids eagerly, mirroring sglang's separate
        # PackTopkIds step on that path.
        if not sgl_moe.route_radix_covered(scores, bias, TOP_K):
            raise RuntimeError(
                "vendored route kernels do not cover this input "
                f"(shape {tuple(scores.shape)}, dtype {scores.dtype})")
        weights, ids = sgl_moe.route_radix(
            scores, bias, TOP_K,
            renormalize=True,
            routed_scaling_factor=float(ROUTED_SCALING),
            apply_scale=True,
            sorted=False,
        )
        backend = self._sgl_gen_backend()
        x, x_sf = backend.quantize_input(
            routed_input.contiguous(), post_quant_comm=False)
        if x_sf is not None and x_sf.dim() == 1:
            x_sf = x_sf.view(x.shape[0], -1)
        return ids, weights.to(torch.bfloat16), x, x_sf

    def _fork_shared(self, h: torch.Tensor) -> dict:
        """sglang's SBO ``issue_shared``: run the shared SiTU MLP on the
        side stream so it overlaps the routed experts; joined at the tail
        via ``_join_shared``."""
        if self._shared_stream is None:
            return {"value": self.shared_experts(h)}
        main = torch.cuda.current_stream()
        self._shared_stream.wait_stream(main)
        ref = {}
        with torch.cuda.stream(self._shared_stream):
            ref["value"] = self.shared_experts(h)
            ref["event"] = self._shared_stream.record_event()
        return ref

    def _join_shared(self, ref: dict) -> torch.Tensor:
        event = ref.get("event")
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        return ref["value"]

    # ------------------------------------------------------------ forward

    def forward(self, hidden_states: torch.Tensor, attn_metadata=None,
                **kwargs) -> torch.Tensor:
        original_shape = hidden_states.shape
        h = hidden_states.view(-1, self.hidden_dim)
        num_tokens = h.shape[0]
        comm = self._comm()

        # Front (kimi_k3.py _forward_unfused): the router logits and the
        # latent down projection both read h; routing sees the same bench
        # realism hook every other path in this file applies.
        logits = _bench_doctor_logits(self.gate(h))
        routed_input = torch.nn.functional.linear(
            h, self.fc1_latent_proj.weight)

        self._ensure_sgl_ar()
        defer_finalize = self._finalize_push_fits(num_tokens)

        ids, scales, x, x_sf = self._sgl_route_quant(logits, routed_input)
        shared_ref = self._fork_shared(h)
        routed = self._sgl_gen_backend().run_moe(
            x, ids, scales, x_sf=x_sf, do_finalize=not defer_finalize)
        shared = self._join_shared(shared_ref).view(-1, self.hidden_dim)

        if defer_finalize:
            from .kernels.sgl_adapters import all_reduce as sgl_ar
            gemm2_out, expert_scale_factor, expanded_idx = routed
            latent = torch.empty(
                num_tokens, self.moe_hidden_size,
                dtype=torch.bfloat16, device=h.device)
            # Finalize (top-k weighted unpermute) folded into the 1shot push
            # AR's staging pass, RMSNorm over every latent row — kimi_k3.py
            # _forward_fused's defer_finalize arm.
            sgl_ar.finalize_all_reduce_push_norm(
                self._sgl_ar_state.world_size,
                latent,
                gemm2_out.view(-1, self.moe_hidden_size),
                expanded_idx.view(-1).to(torch.int32),
                expert_scale_factor.view(num_tokens, -1).to(torch.bfloat16),
                self.latent_norm.weight,
                self.latent_norm.variance_epsilon,
            )
            # Shared partial: sglang pulls it low-SM on a symm-buffer slice;
            # without those buffers, push it when the message fits, else the
            # bench collective (identical reduced value either way).
            shared = shared.contiguous()
            shared_bytes = shared.numel() * shared.element_size()
            if shared_bytes <= min(self._PUSH_MAX_BYTES,
                                   self._sgl_max_push_bytes):
                shared = sgl_ar.all_reduce_push_res(
                    self._sgl_ar_state.world_size, shared)
            else:
                shared = comm.all_reduce(shared)
        else:
            # kimi_k3.py _forward_unfused latent tail under plain TP:
            # TP-partial routed sums reduce in latent space BEFORE the norm
            # (sum(norm(x_i)) != norm(sum(x_i))); the shared partial reduces
            # in hidden space.
            routed = routed[0] if isinstance(routed, tuple) else routed
            routed = routed.view(-1, self.moe_hidden_size)
            if comm.world > 1:
                routed = comm.all_reduce(routed)
                shared = comm.all_reduce(shared)
            latent = torch.nn.functional.rms_norm(
                routed, (self.moe_hidden_size,),
                self.latent_norm.weight, self.latent_norm.variance_epsilon)

        # fc2_latent_proj (sglang routed_expert_up_proj) is replicated, so
        # the routed output is fully reduced after the latent AR; one add
        # joins the branches (kimi_k3.py's _add3 with no prefix_sum).
        # torch.matmul over the Linear wrapper for the same reason the
        # removed bench baseline routed fc2 through torch.matmul: the
        # trtllm Linear's cublasLt pick in this harness is a 35 us NNT
        # variant vs production's ~11 us splitK class.
        out = torch.matmul(latent, self.fc2_latent_proj.weight.t())
        return (shared + out).view(original_shape)


__all__ = [
    "SglKimiK3MoEBlock",
    "TrtKimiK3MoEBlock",
    "TrtRc25KimiK3MoEBlock",
    "k3_model_config",
    "k3_pretrained_config",
]
