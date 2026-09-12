"""Self-contained Kimi-K3 MLA production blocks for MTP1 decode alignment.

Two blocks, one per serving stack, mirroring the KDA MTP1 blocks in
``b10_kimi_k3_kda_layer.py`` (same constructor pattern, collectives
plumbing, AttnRes-front + AR-tail ownership and record spans):

* :class:`TrtKimiK3MlaBlock` — the rc19 fork's MLA attention layer
  (``KimiMLAAttention``) at MTP1 decode, running the fork's own module
  against a bench-owned KV cache manager + TRTLLM attention metadata.
* :class:`SglKimiK3MlaBlock` — sglang's K3 MLA decode path
  (``kimi_k3.py::KimiK3MLAAttention`` over ``DeepseekV2AttentionMLA``'s
  absorb path and the ``trtllm_mla`` backend), staged as a standalone
  ``nn.Module`` on the vendored sglang kernels reached through
  ``kernels/sgl_adapters`` plus the installed ``flashinfer`` wheel.

Trace contract: ``traces/mtp1_decode_mla_layer_kernels.md`` (extracted from
``trt_rc19_decode_mtp1_bs1_rank0.trace.json.gz`` and
``sglang_v0.5.18_decode_mtp1_bs1_rank0.trace.json.gz``; raw sequences in
``traces/extracted_{trt_rc19,sglang}.txt``). Workload: 35 cached tokens +
MTP1 verify (M = 2 tokens/request), TP8 rank 0.

MLA dims (checkpoint config.json, /node-storage/var/kimi-k3-config): 96
q-heads global (12 local at tp8), q_lora 1536, kv_lora 512, nope 128,
rope 64, v 128; fused-A width 2112; absorbed q head dim 576. K3 MLA is
NoPE (``mla_use_nope``) with the sigmoid output gate
(``mla_use_output_gate``); ``full_attn_layers = [4, 8, ..., 88, 92, 93]``
(1-based) — the fourth layer of every four-layer window.

KV cache (bench-owned fixed slots, mirroring how the KDA blocks own their
conv/SSM pools): each request holds one 64-token page of latent KV
([kv_lora | k_rope] = 576 per token) — bf16 through the fork's
KVCacheManager for the TRT block, fp8-e4m3 token-rows for the sglang
block. Every forward replays the same fixed decode step: 35 cached rows,
verify tokens written at rows 35..36. Kernel shapes/traffic match the
baseline step; cache CONTENTS are synthetic (KDA-block convention).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from kimi_k3.b10_kimi_k3_kda_layer import MTP_WIDTH, _graft
from kimi_k3.config import (
    ATTN_RES_BLOCK_SIZE,
    HIDDEN,
    MLA_ABSORB_Q_DIM,
    MLA_FUSED_A_DIM,
    MLA_KV_LORA_RANK,
    MLA_Q_LORA_RANK,
    MLA_QK_HEAD_DIM,
    MLA_QK_NOPE_HEAD_DIM,
    MLA_QK_ROPE_HEAD_DIM,
    MLA_V_HEAD_DIM,
    NUM_ATTN_RES_BLOCKS,
    NUM_LAYERS,
    RMS_EPS,
    K3Shard,
    is_kda_layer,
)

#: Baseline workload context (35 input tokens) — the fixed decode step both
#: baseline traces were captured at.
DEFAULT_CONTEXT_LEN = 35
#: Paged-KV page size shared by both stacks at this operating point (the
#: sglang fmha template says ``PagedKvDenseP64``; the TRT bench cache
#: manager uses the same 64, as block4_demo/the fork unittests do).
TOKENS_PER_BLOCK = 64
_MTP_MAX_TOKENS = 8


class _KimiK3MlaMtpBlock(nn.Module):
    """Shared MTP1 (target-verify, M = 2 tokens/request) MLA block machinery.

    Owns what both stacks share: the AttnRes-stream state
    (``prefix_sum``/``delta``/``block_residual`` + input-norm weight, same
    buffers and layer-geometry rules as ``_KimiK3KdaMtpBlock``), the fixed
    decode-step geometry (context length, static positions/slots) and the
    collectives plumbing. Subclasses own their weight surfaces entirely —
    the two stacks' parameter layouts differ (fork module vs. plain
    parameters), unlike the KDA file where one weight set serves both.

    ``batch`` keeps the file's token-count convention: TOTAL tokens M
    (multiple of ``MTP_WIDTH``); requests = batch // 2. Primary alignment
    case is batch=2 (1 request), tp8 shard.

    ``layer_idx`` must be an MLA (non-KDA) layer; the default 3 (1-based 4)
    is the first MLA layer of the real interleave. Like the KDA blocks it
    also needs a non-empty AttnRes bank (``prev_valid_blocks >= 1``), which
    every real MLA layer satisfies (the earliest, 1-based 4, sits above the
    layer-0 block write).
    """

    RECORD_SPAN = "KimiK3MlaMtp"
    #: Collectives.all_reduce impl for the TP tail; subclasses pin the
    #: backend whose kernel matches their baseline trace.
    AR_IMPL = "auto"

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        *,
        layer_idx: int = 3,
        context_len: int = DEFAULT_CONTEXT_LEN,
        device: str = "cuda",
        collectives=None,
    ) -> None:
        super().__init__()
        if batch % MTP_WIDTH:
            raise ValueError(
                f"MTP1 blocks need batch % {MTP_WIDTH} == 0 total tokens"
            )
        if not MTP_WIDTH <= batch <= _MTP_MAX_TOKENS:
            raise ValueError(
                f"MTP1 blocks cover {MTP_WIDTH}..{_MTP_MAX_TOKENS} tokens "
                "(the traces' small-M GEMM window)"
            )
        if not 0 <= layer_idx < NUM_LAYERS or is_kda_layer(layer_idx):
            raise ValueError(
                f"layer {layer_idx} is not an MLA layer in Kimi-K3 "
                "(MLA = 1-based multiples of 4, plus 92/93)"
            )
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if context_len + MTP_WIDTH > TOKENS_PER_BLOCK:
            raise ValueError(
                "single-page bench cache covers context_len + 2 <= "
                f"{TOKENS_PER_BLOCK} tokens"
            )

        self.shard = shard
        self.batch = batch
        self.num_requests = batch // MTP_WIDTH
        self.layer_idx = layer_idx
        self.context_len = context_len
        self.kv_len = context_len + MTP_WIDTH
        self._collectives = collectives

        self.is_block_write_layer = layer_idx % ATTN_RES_BLOCK_SIZE == 0
        self.block_write_idx = layer_idx // ATTN_RES_BLOCK_SIZE
        self.prev_valid_blocks = (
            layer_idx + ATTN_RES_BLOCK_SIZE - 1
        ) // ATTN_RES_BLOCK_SIZE
        if self.prev_valid_blocks < 1:
            raise ValueError(
                "MTP1 blocks need a non-empty AttnRes bank (layer_idx >= 1)"
            )

        dev = torch.device(device)
        self.register_buffer(
            "prefix_sum",
            torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "delta",
            torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "block_residual",
            torch.randn(
                batch, NUM_ATTN_RES_BLOCKS, HIDDEN,
                device=dev, dtype=torch.bfloat16,
            ),
        )
        self.input_norm_weight = nn.Parameter(
            torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
        )
        # Fixed decode-step geometry: token t of request r sits at cache row
        # r * TOKENS_PER_BLOCK + context_len + t; positions are
        # [context_len, context_len + 1] per request.
        positions = torch.arange(
            context_len, context_len + MTP_WIDTH, device=dev,
            dtype=torch.int64,
        ).repeat(self.num_requests)
        self.register_buffer("position_ids", positions.unsqueeze(0))

    def attach_collectives(self, collectives) -> None:
        self._collectives = collectives

    def _reduce(self, partial: torch.Tensor) -> torch.Tensor:
        """TP tail (KDA-block convention): single-GPU tp-shard math means no
        AR kernel at world <= 1; at world > 1 the AR routes through
        ``AR_IMPL``."""
        if self._collectives is not None and self._collectives.world > 1:
            return self._collectives.all_reduce(partial, impl=self.AR_IMPL)
        return partial

    def decode(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        hidden: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is not None or cu_seqlens is not None:
            raise ValueError("MTP1 dispatch does not accept prefill inputs")
        with torch.profiler.record_function(self.RECORD_SPAN):
            return self.decode()


def _randomize_module_weights(module: nn.Module, seed: int) -> None:
    """block4_demo's deterministic, numerically-safe init: norm scales -> 1,
    every other floating parameter -> N(0, 0.02)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not param.is_floating_point():
                continue
            if "norm" in name.lower() and name.rsplit(".", 1)[-1] == "weight":
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.02, generator=generator)


class TrtKimiK3MlaBlock(_KimiK3MlaMtpBlock):
    """Kimi-K3 MLA MTP1-verify block following the TRT-LLM rc19 stack.

    Aligns to ``traces/trt_rc19_decode_mtp1_bs1_rank0.trace.json.gz``
    (native rc19 server, checkout ``12e51f84e1`` / deployment fork
    ``d9fa74fd94``, image ``44d66979``; MLA per-layer sequence in
    ``traces/mtp1_decode_mla_layer_kernels.md``). Source of the attention
    middle: the INSTALLED runtime's ``modeling_kimi_linear.KimiMLAAttention``
    — the block composes the fork's own module rather than re-implementing
    it, the same way ``TrtKimiK3MoEBlock`` composes ``KimiK3MoE`` — so
    contract positions 2–18 (fused-A GEMM, gate/q_b GEMMs, flashinfer
    RMSNorms, absorb BMMs, ``mla_rope_generation``'s identity-rope cache
    assign, the FlashInfer CuTeDSL monolithic MLA decode, the aten
    sigmoid+mul gate, o_proj) are launched by the production code itself.
    Position 1 is the grafted checkout ``attn_res_fwd_online_v2`` (same
    module the KDA blocks graft); position 19 is TRT's pattern-0
    ``allreduce_fusion_kernel_oneshot_lamport`` reached through
    ``Collectives.all_reduce(impl="trt")`` at world > 1.

    Runtime plumbing (built lazily on first decode, CUDA required): a
    bench-owned ``KVCacheManager`` (SELFKONLY, num_kv_heads=1,
    head_dim=576, bf16, ``layer_mask`` mapping this block's ``layer_idx``
    onto pool layer 0) plus a prepared ``TrtllmAttentionMetadata`` for the
    fixed step — generation-only, ``seq_lens=[2]`` per request,
    ``num_cached_tokens_per_seq=[context_len]`` — the
    ``block4_demo/runtime.py`` / ``test_kimi_linear.py`` driving pattern.

    Named deviations (beyond the shared fixed-step note in the module
    docstring):

    * ``reduce_output=False`` + bench ``Collectives`` tail instead of the
      module's in-module AR — identical kernel at world > 1, none at
      world = 1 (shared bench convention).
    * The cache manager gets a single-rank ``Mapping`` (its pools are
      rank-local; MLA's ``num_kv_heads=1`` is TP-replicated) while the
      attention module gets the tp-shard mapping for weight sharding.
    * ``model_config.spec_config`` is a minimal namespace exposing
      ``tokens_per_gen_step = MTP_WIDTH`` — the only field
      ``KimiMLAAttention`` reads — standing in for the server's MTP config.
    * Contract positions 11–12 (fill + memcpy32) are flashinfer wrapper
      bookkeeping and follow the installed wheel (see the contract md).
    """

    RECORD_SPAN = "TrtKimiK3Mla"
    AR_IMPL = "trt"
    WEIGHT_SEED = 20260910

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        *,
        layer_idx: int = 3,
        context_len: int = DEFAULT_CONTEXT_LEN,
        device: str = "cuda",
        collectives=None,
        aux_stream: torch.cuda.Stream | None = None,
        output_gate_aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(
            shard,
            batch,
            layer_idx=layer_idx,
            context_len=context_len,
            device=device,
            collectives=collectives,
        )
        dev = torch.device(device)
        attn_res_mod = _graft(
            "tensorrt_llm._torch.modules.attn_res",
            "_torch/modules/attn_res.py",
        )
        self.attn_res = attn_res_mod.AttnRes(
            HIDDEN, RMS_EPS, dtype=torch.bfloat16, device=dev
        )
        self._model_config = self._build_model_config()
        self.aux_stream = aux_stream or torch.cuda.Stream(device=dev)
        self.output_gate_aux_stream = (
            output_gate_aux_stream or torch.cuda.Stream(device=dev)
        )
        from tensorrt_llm._torch.models.modeling_kimi_linear import (
            KimiMLAAttention,
        )

        with torch.device(dev):
            self.mla = KimiMLAAttention(
                self._model_config,
                layer_idx=self.layer_idx,
                aux_stream=self.aux_stream,
                reduce_output=False,
                output_gate_aux_stream=self.output_gate_aux_stream,
            )
        self._kv_cache_manager = None
        self._attn_metadata = None
        self.reset_parameters()

    def _build_model_config(self):
        """block4_demo's attention ModelConfig, restricted to this layer.

        ``linear_attn_config`` lists every layer up to ``layer_idx`` so the
        fork's ``is_kda_layer`` sees the real interleave; only this block's
        (MLA) layer is ever constructed.
        """
        from tensorrt_llm._torch.configs import KimiLinearConfig
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.mapping import Mapping
        from tensorrt_llm.models.modeling_utils import QuantConfig

        num_layers = self.layer_idx + 1
        one_based = range(1, num_layers + 1)
        pretrained_config = KimiLinearConfig(
            architectures=["KimiLinearForCausalLM"],
            vocab_size=163840,
            hidden_size=HIDDEN,
            intermediate_size=33792,  # unused (MoE model) but real
            num_hidden_layers=num_layers,
            num_attention_heads=self.shard.heads_global,
            num_key_value_heads=self.shard.heads_global,
            q_lora_rank=MLA_Q_LORA_RANK,
            kv_lora_rank=MLA_KV_LORA_RANK,
            qk_nope_head_dim=MLA_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=MLA_QK_ROPE_HEAD_DIM,
            v_head_dim=MLA_V_HEAD_DIM,
            mla_use_nope=True,
            mla_use_output_gate=True,
            max_position_embeddings=65536,
            rms_norm_eps=RMS_EPS,
            torch_dtype=torch.bfloat16,
            linear_attn_config={
                "full_attn_layers": [i for i in one_based if i % 4 == 0],
                "head_dim": 128,
                "kda_layers": [i for i in one_based if i % 4 != 0],
                "num_heads": self.shard.heads_global,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
        )
        rank = self._collectives.rank if self._collectives is not None else 0
        mapping = Mapping(
            world_size=self.shard.tp_size,
            tp_size=self.shard.tp_size,
            rank=rank,
            gpus_per_node=min(
                self.shard.tp_size, max(torch.cuda.device_count(), 1)
            ),
        )
        return ModelConfig(
            pretrained_config=pretrained_config,
            mapping=mapping,
            quant_config=QuantConfig(),  # BF16 attention, as K3 serves it
            # The only spec field KimiMLAAttention reads; production carries
            # the full MTP decoding config here.
            spec_config=SimpleNamespace(tokens_per_gen_step=MTP_WIDTH),
            allreduce_strategy="AUTO",
        )

    def reset_parameters(self) -> None:
        _randomize_module_weights(self.mla, self.WEIGHT_SEED)
        with torch.no_grad():
            self.input_norm_weight.fill_(1.0)
            nn.init.ones_(self.attn_res.norm_weight)
            nn.init.normal_(self.attn_res.proj_weight, std=0.02)
        # Build k_b_proj_trans / v_b_proj (the absorb BMM weights) from the
        # freshly initialized kv_b_proj, exactly as serving weight load does.
        if hasattr(self.mla, "post_load_weights"):
            self.mla.post_load_weights()

    # ------------------------------------------------------ bench runtime

    def _ensure_runtime(self) -> None:
        """Cache manager + prepared metadata for the fixed decode step
        (lazily; the block4_demo/test_kimi_linear driving pattern)."""
        if self._attn_metadata is not None:
            return
        from tensorrt_llm._torch import metadata as metadata_lib
        from tensorrt_llm._torch.attention_backend import (
            utils as attention_utils,
        )
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            CacheTypeCpp,
            DataType,
            KVCacheManager,
        )
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig
        from tensorrt_llm.mapping import Mapping

        requests = self.num_requests
        max_seq_len = self.context_len + TOKENS_PER_BLOCK
        self._kv_cache_manager = KVCacheManager(
            KvCacheConfig(
                max_tokens=requests * max_seq_len,
                enable_block_reuse=False,
            ),
            CacheTypeCpp.SELFKONLY,
            num_layers=self.layer_idx + 1,
            layer_mask=[False] * self.layer_idx + [True],
            num_kv_heads=1,
            head_dim=MLA_ABSORB_Q_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=requests,
            # Rank-local pools; TP does not shard MLA's single latent head.
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        request_ids = list(range(1, requests + 1))
        self._kv_cache_manager.add_dummy_requests(
            request_ids, token_nums=[self.kv_len] * requests
        )
        self._kv_cache_manager.get_buffers(self.layer_idx).zero_()

        metadata_cls = attention_utils.get_attention_backend(
            "TRTLLM"
        ).Metadata
        self._attn_metadata = metadata_cls(
            seq_lens=torch.tensor(
                [MTP_WIDTH] * requests, dtype=torch.int
            ),
            num_contexts=0,
            kv_cache_params=metadata_lib.KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[self.context_len] * requests,
            ),
            kv_cache_manager=self._kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=[self.context_len] * requests,
            max_num_requests=requests,
            max_num_tokens=max(32, self.batch),
        )
        self._attn_metadata.prepare()

    def shutdown(self) -> None:
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.shutdown()
            self._kv_cache_manager = None
            self._attn_metadata = None

    def _attn_res_hidden(self) -> torch.Tensor:
        with torch.profiler.record_function("mla.attn_res"):
            return self.attn_res(
                self.prefix_sum,
                self.block_residual,
                delta=self.delta,
                num_blocks=self.prev_valid_blocks,
                block_write_idx=(
                    self.block_write_idx if self.is_block_write_layer else -1
                ),
                output_norm_weight=self.input_norm_weight,
                output_norm_eps=RMS_EPS,
            )

    def decode(self) -> torch.Tensor:
        self._ensure_runtime()
        hidden = self._attn_res_hidden()  # contract kernel 1
        with torch.profiler.record_function("mla.fork_module"):
            partial = self.mla(  # contract kernels 2-18
                position_ids=self.position_ids,
                hidden_states=hidden,
                attn_metadata=self._attn_metadata,
            )
        return self._reduce(partial)  # contract kernel 19


class SglKimiK3MlaBlock(_KimiK3MlaMtpBlock):
    """Kimi-K3 MLA MTP1-verify block following the sglang implementation.

    Aligns to ``traces/sglang_v0.5.18_decode_mtp1_bs1_rank0.trace.json.gz``
    (MLA per-layer sequence in ``traces/mtp1_decode_mla_layer_kernels.md``).
    Source: sglang @ ``f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`` —
    ``srt/models/kimi_k3.py::KimiK3MLAAttention`` (skip_rope=True NoPE,
    ``_precompute_output_gate`` on the gate alt stream, fused output gate)
    over ``srt/models/deepseek_common/attention_forward_methods/
    forward_mla.py::forward_absorb_*`` and ``srt/layers/attention/
    trtllm_mla_backend.py`` (fp8 KV, fused set-KV+concat-q, trtllm-gen
    decode). Device code comes from ``kernels/sgl_copied_kernels`` through
    the ``kernels/sgl_adapters`` surface, plus the installed ``flashinfer``
    wheel — the same wheel sglang's RMSNorm and fmha dispatch to.

    Per-layer kernels (contract positions): (1) ``attn_res_fused_tma`` on
    the pre-added prefix, (2) vendored JIT ``fused_a_gemm``
    [q_a|kv_a|k_rope], (3) g_proj on the TGV GEMM, issued on the gate
    stream (sglang's gate alt stream), (4) flashinfer RMSNorm q_a
    (main stream), (5) flashinfer RMSNorm kv_a (aux stream), (6) q_b TGV,
    (7) q_nope absorb BMM ``torch.bmm`` (nvjet), (8) vendored fused
    fp8 quantize + KV scatter + q concat, (9) flashinfer trtllm-gen fp8 MLA
    decode, (10) v absorb BMM, (11) vendored fused output gate,
    (12) o_proj TGV, (13) fused push AR + residual fold.

    Deviations beyond the shared fixed-step note:

    * Kernel #13 runs GENUINELY when the lazily resolved CustomAllReduceV2
      state is live (``kernels.sgl_adapters.comm.get_sgl_ar_state`` on
      first decode, or injected via :meth:`attach_sgl_ar`); otherwise
      ``Collectives.all_reduce`` at world > 1 (none at world = 1) plus ONE
      explicit add into the persistent ``next_prefix`` buffer — the
      ``SglKimiK3KdaBlock`` convention.
    * Kernel #8's baseline name is the v0.5.18 Triton kernel; the vendored
      commit ships the CUDA JIT ``sglang::set_mla_kv_concat_q_fp8`` —
      same single launch, same fusion boundary (see the contract md).
    * ``w_kc``/``w_vc`` (the absorb BMM factors sglang derives from
      ``kv_b_proj`` at weight load) are owned directly as parameters.
    * The gate GEMM and kv_a norm always run on their side streams; sglang
      gates those forks on CUDA-graph capture mode (the KDA blocks'
      aux-stream convention — kernels and order identical either way).
    * fp8 scales: the bench pool carries no checkpoint ``k_scale``, so
      ``bmm1_scale = 1/sqrt(192)`` with q_scale = k_scale = 1 — exactly
      sglang's ``_compute_decode_bmm1_scale`` on a scale-less checkpoint.
    """

    RECORD_SPAN = "SglKimiK3Mla"
    WEIGHT_SEED = 20260911
    #: sglang trtllm_mla backend workspace (flashinfer trtllm-gen fmha).
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        *,
        layer_idx: int = 3,
        context_len: int = DEFAULT_CONTEXT_LEN,
        device: str = "cuda",
        collectives=None,
        aux_stream: torch.cuda.Stream | None = None,
        gate_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(
            shard,
            batch,
            layer_idx=layer_idx,
            context_len=context_len,
            device=device,
            collectives=collectives,
        )
        dev = torch.device(device)
        heads = shard.heads_local

        # kimi_k3.py weight surfaces (attn-TP sharded like serving).
        self.fused_qkv_a_proj_with_mqa = nn.Linear(
            HIDDEN, MLA_FUSED_A_DIM, bias=False,
            device=dev, dtype=torch.bfloat16,
        )
        self.q_b_proj = nn.Linear(
            MLA_Q_LORA_RANK, heads * MLA_QK_HEAD_DIM, bias=False,
            device=dev, dtype=torch.bfloat16,
        )
        self.g_proj = nn.Linear(
            HIDDEN, heads * MLA_V_HEAD_DIM, bias=False,
            device=dev, dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(
            heads * MLA_V_HEAD_DIM, HIDDEN, bias=False,
            device=dev, dtype=torch.bfloat16,
        )
        # Absorb BMM factors (sglang builds these from kv_b_proj at load).
        self.w_kc = nn.Parameter(
            torch.empty(
                heads, MLA_QK_NOPE_HEAD_DIM, MLA_KV_LORA_RANK,
                device=dev, dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w_vc = nn.Parameter(
            torch.empty(
                heads, MLA_KV_LORA_RANK, MLA_V_HEAD_DIM,
                device=dev, dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.q_a_layernorm_weight = nn.Parameter(
            torch.ones(MLA_Q_LORA_RANK, device=dev, dtype=torch.bfloat16)
        )
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(MLA_KV_LORA_RANK, device=dev, dtype=torch.bfloat16)
        )
        # AttnRes weights + the precomputed TMA product (SglKimiK3KdaBlock
        # convention).
        self.attn_res_norm_weight = nn.Parameter(
            torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
        )
        self.attn_res_proj_weight = nn.Parameter(
            torch.empty(
                NUM_ATTN_RES_BLOCKS, device=dev, dtype=torch.bfloat16
            )
        )
        self.register_buffer(
            "attn_res_cw",
            torch.empty(
                NUM_ATTN_RES_BLOCKS * HIDDEN,
                device=dev, dtype=torch.bfloat16,
            ).view(NUM_ATTN_RES_BLOCKS, HIDDEN),
        )
        self.register_buffer(
            "next_prefix", torch.zeros_like(self.prefix_sum)
        )

        # fp8 latent-KV pool: token rows [requests * page, 576], one page
        # per request (sglang's token-granular MLA pool viewed paged by the
        # decode kernel). Row 0 is sglang's reserved padding slot; the
        # bench's write rows start at context_len >= 1 so it is never hit.
        requests = self.num_requests
        self.register_buffer(
            "kv_buffer",
            torch.zeros(
                requests * TOKENS_PER_BLOCK, MLA_ABSORB_Q_DIM,
                device=dev, dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "block_tables",
            torch.arange(requests, device=dev, dtype=torch.int32).view(
                requests, 1
            ),
        )
        self.register_buffer(
            "seq_lens",
            torch.full(
                (requests,), self.kv_len, device=dev, dtype=torch.int32
            ),
        )
        row0 = (
            torch.arange(requests, device=dev, dtype=torch.int64)
            * TOKENS_PER_BLOCK
            + context_len
        )
        self.register_buffer(
            "cache_locs",
            (
                row0.unsqueeze(1)
                + torch.arange(MTP_WIDTH, device=dev, dtype=torch.int64)
            ).reshape(-1),
        )
        self.register_buffer(
            "fmha_workspace",
            torch.zeros(
                self._WORKSPACE_BYTES, device=dev, dtype=torch.uint8
            ),
        )
        self._multi_ctas_counter = None
        self._decode_extra_kwargs = None

        self.aux_stream = aux_stream or torch.cuda.Stream(device=dev)
        self.gate_stream = gate_stream or torch.cuda.Stream(device=dev)

        self._sgl_world = 0  # 0 = fallback tail; live AR state enables push
        self._sgl_ar_state = None
        self._sgl_ar_resolved = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for linear in (
                self.fused_qkv_a_proj_with_mqa,
                self.q_b_proj,
                self.g_proj,
                self.o_proj,
            ):
                nn.init.normal_(linear.weight, std=0.02)
            nn.init.normal_(self.w_kc, std=0.02)
            nn.init.normal_(self.w_vc, std=0.02)
            self.q_a_layernorm_weight.fill_(1.0)
            self.kv_a_layernorm_weight.fill_(1.0)
            self.input_norm_weight.fill_(1.0)
            self.attn_res_norm_weight.fill_(1.0)
            nn.init.normal_(self.attn_res_proj_weight, std=0.02)
            # attn_res_fused_tma consumes score_norm_weight *
            # score_proj_weight precomputed (attn_residual.get_cw).
            self.attn_res_cw.copy_(
                (
                    self.attn_res_norm_weight.float().unsqueeze(0)
                    * self.attn_res_proj_weight.float().unsqueeze(1)
                ).to(torch.bfloat16)
            )

    # ------------------------------------------------------- fused-AR tail

    def attach_sgl_ar(self, state) -> None:
        """Inject a prebuilt fused-AR state (``SglArState`` from
        ``kernels.sgl_adapters.comm``); normally resolved lazily on first
        decode. The o_proj partial ([T, 7168] bf16, 28 KB at M=2) fits the
        tuned push slot."""
        from .kernels.sgl_adapters import all_reduce as sgl_ar

        sgl_ar.register_comm(state.comm)
        self._sgl_ar_state = state
        self._sgl_ar_resolved = True
        self._sgl_world = int(state.world_size)

    def _ensure_sgl_ar(self) -> None:
        if self._sgl_ar_resolved:
            return
        self._sgl_ar_resolved = True
        from .kernels.sgl_adapters import comm as sgl_comm

        state = sgl_comm.get_sgl_ar_state()
        if state is not None:
            self.attach_sgl_ar(state)

    # ----------------------------------------------------------- sections

    def _attn_res_hidden(self) -> torch.Tensor:
        from .kernels.sgl_adapters.attn_res import attn_res_fused_tma

        out = torch.empty_like(self.prefix_sum)
        with torch.profiler.record_function("mla.attn_res_tma"):
            attn_res_fused_tma(
                self.prefix_sum,
                self.block_residual,
                self.attn_res_cw,
                self.input_norm_weight.detach(),
                out,
                self.prev_valid_blocks,
                RMS_EPS,
                write_prefix=self.is_block_write_layer,
            )
        return out

    def _fmha_extra_kwargs(self) -> dict:
        """sglang's trtllm-gen extra kwargs, resolved once against the
        installed flashinfer signature (versions before the fi_golden
        counter-buffer split own the semaphores inside the workspace)."""
        if self._decode_extra_kwargs is not None:
            return self._decode_extra_kwargs
        import inspect

        import flashinfer

        fn = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        params = inspect.signature(fn).parameters
        extra: dict = {}
        if "multi_ctas_kv_counter_buffer" in params:
            # Same-generation wheels ship these helpers (sglang's
            # _multi_ctas_kv_counter_bytes uses exactly this pair).
            sm_count = flashinfer.utils.get_device_sm_count(
                self.kv_buffer.device
            )
            nbytes = flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
                self.num_requests, self.shard.heads_local * MTP_WIDTH,
                sm_count,
            )
            self._multi_ctas_counter = torch.zeros(
                nbytes, dtype=torch.uint8, device=self.kv_buffer.device
            )
            extra["multi_ctas_kv_counter_buffer"] = self._multi_ctas_counter
        self._decode_extra_kwargs = extra
        return extra

    def decode(self) -> torch.Tensor:
        import flashinfer

        from .kernels.sgl_adapters import mla as sgl_mla
        from .kernels.sgl_adapters.gemm import cutedsl_bf16_gemm

        self._ensure_sgl_ar()
        heads = self.shard.heads_local
        tokens = self.batch
        main = torch.cuda.current_stream()

        hidden = self._attn_res_hidden()  # kernel 1 (fused TMA)

        with torch.profiler.record_function("mla.fused_a_gemm"):
            # Kernel 2. Detached weights: the vendored kernels export via
            # dlpack, which rejects requires_grad tensors.
            qkv_latent = sgl_mla.dsv3_fused_a_gemm(
                hidden, self.fused_qkv_a_proj_with_mqa.weight.detach().T
            )
        # Kernel 3: gate GEMM precomputed on the gate stream
        # (KimiK3MLAAttention._precompute_output_gate).
        self.gate_stream.wait_stream(main)
        with torch.cuda.stream(self.gate_stream):
            with torch.profiler.record_function("mla.g_proj_tgv"):
                gate = cutedsl_bf16_gemm(
                    hidden, self.g_proj.weight.detach()
                )
            gate_ready = self.gate_stream.record_event()

        q_lora = qkv_latent[:, :MLA_Q_LORA_RANK]
        k_nope_raw = qkv_latent[
            :, MLA_Q_LORA_RANK:MLA_Q_LORA_RANK + MLA_KV_LORA_RANK
        ]
        k_rope = qkv_latent[:, MLA_Q_LORA_RANK + MLA_KV_LORA_RANK:]
        # Kernels 4-5: q_a norm on main, kv_a norm on the aux stream
        # (sglang's capture-mode qk-norm overlap). Strided column slices go
        # in directly, as sglang passes them — no copy kernels here.
        self.aux_stream.wait_stream(main)
        with torch.cuda.stream(self.aux_stream):
            with torch.profiler.record_function("mla.kv_a_norm"):
                k_nope = flashinfer.norm.rmsnorm(
                    k_nope_raw,
                    self.kv_a_layernorm_weight.detach(),
                    RMS_EPS,
                )
        with torch.profiler.record_function("mla.q_a_norm"):
            q_lora = flashinfer.norm.rmsnorm(
                q_lora,
                self.q_a_layernorm_weight.detach(),
                RMS_EPS,
            )
        main.wait_stream(self.aux_stream)
        if not torch.cuda.is_current_stream_capturing():
            k_nope.record_stream(main)

        with torch.profiler.record_function("mla.q_b_tgv"):
            q = cutedsl_bf16_gemm(  # kernel 6
                q_lora, self.q_b_proj.weight.detach()
            ).view(tokens, heads, MLA_QK_HEAD_DIM)
        q_nope = q[..., :MLA_QK_NOPE_HEAD_DIM]
        q_pe = q[..., MLA_QK_NOPE_HEAD_DIM:]

        with torch.profiler.record_function("mla.q_absorb_bmm"):
            # Kernel 7: nvjet BMM written into a [T, H, 512] buffer through
            # its transposed view (sglang's forward_absorb layout).
            q_nope_out = torch.empty(
                tokens, heads, MLA_KV_LORA_RANK,
                device=q.device, dtype=torch.bfloat16,
            )
            torch.bmm(
                q_nope.transpose(0, 1),
                self.w_kc.detach(),
                out=q_nope_out.transpose(0, 1),
            )

        with torch.profiler.record_function("mla.set_kv_concat_q_fp8"):
            # Kernel 8: ONE fused launch — fp8 quantize of [k_nope|k_rope],
            # scatter into the paged pool at the step's fixed rows, fp8
            # [q_nope|q_pe] 576-dim concat.
            query = sgl_mla.set_mla_kv_concat_q_fp8(
                self.kv_buffer,
                self.cache_locs,
                k_nope,
                k_rope,
                q_nope_out,
                q_pe,
            )

        with torch.profiler.record_function("mla.trtllm_gen_fmha"):
            # Kernel 9: trtllm-gen fp8 MLA decode (sglang's
            # _run_decode_kernel wiring; q_scale = k_scale = 1).
            raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query.view(
                    self.num_requests, MTP_WIDTH, heads, MLA_ABSORB_Q_DIM
                ),
                kv_cache=self.kv_buffer.view(
                    -1, TOKENS_PER_BLOCK, MLA_ABSORB_Q_DIM
                ).unsqueeze(1),
                workspace_buffer=self.fmha_workspace,
                qk_nope_head_dim=MLA_QK_NOPE_HEAD_DIM,
                kv_lora_rank=MLA_KV_LORA_RANK,
                qk_rope_head_dim=MLA_QK_ROPE_HEAD_DIM,
                block_tables=self.block_tables,
                seq_lens=self.seq_lens,
                max_seq_len=self.kv_len,
                bmm1_scale=MLA_QK_HEAD_DIM ** -0.5,
                **self._fmha_extra_kwargs(),
            )

        with torch.profiler.record_function("mla.v_absorb_bmm"):
            # Kernel 10: nvjet BMM, transposed-view output as above.
            attn_bmm = torch.empty(
                tokens, heads, MLA_V_HEAD_DIM,
                device=q.device, dtype=torch.bfloat16,
            )
            torch.bmm(
                raw_out.view(
                    tokens, heads, MLA_KV_LORA_RANK
                ).transpose(0, 1),
                self.w_vc.detach(),
                out=attn_bmm.transpose(0, 1),
            )

        main.wait_event(gate_ready)
        if not torch.cuda.is_current_stream_capturing():
            gate.record_stream(main)
        with torch.profiler.record_function("mla.output_gate"):
            gated = sgl_mla.kimi_k3_mla_output_gate(  # kernel 11
                attn_bmm.view(tokens, heads * MLA_V_HEAD_DIM), gate
            )

        with torch.profiler.record_function("mla.o_proj_tgv"):
            partial = cutedsl_bf16_gemm(  # kernel 12
                gated, self.o_proj.weight.detach()
            )

        if self._sgl_world > 1:
            # Kernel 13, genuine: 1shot push AR with the prefix/residual
            # fold, in place on the o_proj partial.
            from .kernels.sgl_adapters.all_reduce import all_reduce_push_res

            with torch.profiler.record_function("mla.ar_push_res"):
                return all_reduce_push_res(
                    self._sgl_world, partial, self.prefix_sum
                )
        reduced = self._reduce(partial)
        # Kernel 13 stand-in (no communicator attached): production folds
        # this add into all_reduce_push_res_kernel (see class docstring).
        with torch.profiler.record_function("mla.ar_residual_fold"):
            torch.add(self.prefix_sum, reduced, out=self.next_prefix)
        return self.next_prefix


__all__ = [
    "DEFAULT_CONTEXT_LEN",
    "SglKimiK3MlaBlock",
    "TOKENS_PER_BLOCK",
    "TrtKimiK3MlaBlock",
]
