"""Kimi-K3 MoE, b10-optimized: overrides on the TRT-LLM port.

``KimiK3MoEB10(KimiK3MoE)`` keeps the baseline weights and semantics
(moe_trtllm_kimi_k3.py) and swaps the decode forward for an optimized
pipeline. THE DEFAULTS ARE THE SHIPPED CONFIG (no env vars needed;
validated Aug 5, see moe_optimization.md): every switch below exists
so ``bench_moe_kimi_k3.py --ablate`` can knock optimizations out one
at a time, not because callers should tune them.

The decode pipeline the defaults produce (B <= OPT_MAX_TOKENS):

  ONE wide input GEMM [gate | fc1-shard | shared g/u]  (merge3)
    -> Triton routing top-16, run_moe format          (routing=ours)
    -> one-shot AG rebuilding the latent WITH the MXFP8 quantize
       fused on its write-out                         (fc1_shard<=16)
    -> production trtllm-gen expert kernels (untouched)
    -> shared SiTU+down chain + its AR on an aux stream (overlap)
    tail, B < 16:  fi fused AR+RMSNorm -> per-rank fc2 column slice
                   addmm'd into the shared partial -> fi AR (comm=fi)
    tail, B >= 16: fused finalize+AR+RMSNorm (ONE flashinfer kernel,
                   experts run unfinalized)           (ref_tail+fin_fuse)
                   -> FULL fc2 via the CuTeDSLGen tail GEMM
                   (tail_gemm_cutedsl, B10_TAIL_KERNEL=0 disables);
                   no output collective needed.

Ablation switches (set_opt_flags) and their measured verdicts:
  merged_gemm / merge3   on; one wide GEMM (narrow-N cuBLAS runs at
                         ~2 TB/s, wide at ~8 - fold everything in)
  routing                "ours" (Triton) | "noaux" (TRT op)
  fc1_shard              on to B<=16 (B10_FC1_SHARD_MAX_TOKENS)
  fc2_shard              the B<16 tail's column shard (<=64)
  comm                   "fi" (default; won at every size once merge3
                         freed the aux stream) | "custom" (Lamport
                         RS+norm - kept for re-tuning on other nodes;
                         the fused AG is NOT gated by this)
  overlap_shared         on to B<=64
  ref_tail + fin_fuse    the B>=16 tail (B10_REF_TAIL_MIN_TOKENS)
  prefill_opt            tokens >= 192: sharded fc1/fc2 + CE all-gather
  skip_shared            TIMING PROBE only (output wrong)
  rs_defer               retired (loses at decode sizes; default off)

Node warning: every crossover above was re-measured on the
model-performance box and several DIFFER from the previous node -
re-run --ablate after moving hardware. Falls back to the exact
baseline forward between OPT_MAX_TOKENS and PREFILL_MIN_TOKENS or
before init_opt.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType

from .config import (
    MOE_LATENT,
    NUM_EXPERTS,
    RMS_EPS,
    ROUTED_SCALING,
    TOP_K,
)
from .moe_trtllm_kimi_k3 import KimiK3MoE
from .quant_slice import quant_slice_mxfp8
from .routing_kimi_k3 import (
    route_for_fused_moe,
    route_for_trtllm_gen,
    route_pack,
)

# fc1 column-shard pays a ~flat ~6 us of saved weight read against an
# AG that grows ~linearly in tokens (7 KB/token) - decode regime only.
# The B=8 trace pair (results/..._b8_opt_fc1{off,on}_graph_annotated)
# shows the shard win is NOT (only) the weight read: in the merged
# pipeline the shard's AG rebuilds the latent contiguous, and with the
# MXFP8 quantize FUSED on the AG's write-out (all_gather_mxfp8) the
# whole exposed contiguous-copy + quantize stage between routing and
# permute disappears (B=8: 111.6 -> 104.2 us). Forced-always sweep
# with the fused AG: wins ~4 us at B=16, wash at 32, -3 at 64 - gate
# at 16. Re-measure with --ablate after comm/kernel changes.
FC1_SHARD_MAX_TOKENS = int(os.environ.get("B10_FC1_SHARD_MAX_TOKENS", "16"))

# fc2 column shard + RS crossover (overnight sweep, two runs): the
# sharded tail wins +13..21 us at B<=32 and +5 at B=64, but at B=128
# the baseline-style ONE fused [latent|shared] AR tail is ~6 us
# FASTER - at that size the RS's fixed cost plus the extra rendezvous
# outweigh the fc2 weight-read cut, and the fat AR amortizes.
FC2_SHARD_MAX_TOKENS = int(os.environ.get("B10_FC2_SHARD_MAX_TOKENS", "64"))

# shared-expert overlap crossover (see _forward_opt comment);
# env-overridable so the crossover can be re-measured after routing /
# kernel changes without editing code
OVERLAP_SHARED_MAX_TOKENS = int(
    os.environ.get("B10_OVERLAP_MAX_TOKENS", "64"))

# latent RS+norm strategy: ALWAYS the fused in-kernel norm. At the
# decode sizes we care about (B<=16) fused wins outright - its
# peer-scalar wait is small and a second launch costs more than it
# saves. Deferred (push partials + scale kernel after the shared tail
# GEMM) only pays off at B>=32 (fused standalone degrades 10.4 ->
# 12.2 -> 14.4 us at B=16/32/64 vs ~10.5-11.0 for push+scale), which
# is out of scope; lower this env to re-enable for experiments
# (tmp_rs_defer_launch.py compares all three methods in-situ).
RS_DEFER_MIN_TOKENS = int(
    os.environ.get("B10_RS_DEFER_MIN_TOKENS", "1000000"))

# REF-overlap tail crossover: from here up, drop the fc2 shard and use
# finalize+AR+norm (fin_fuse) -> FULL fc2 (output identical on every
# rank -> NO output collective) with the shared-expert AR(hidden) on
# the aux stream. NODE-DEPENDENT: the old node's probe put this at
# B>=4; on the model-performance box the shard tail wins through B=8
# under multi-input averaging (c8 ablation) -> default 16.
REF_TAIL_MIN_TOKENS = int(os.environ.get("B10_REF_TAIL_MIN_TOKENS", "16"))

# trtllm-gen input-stage crossover: below this, ONE merged [gate|fc1]
# GEMM (single weight pass, shared chain forks immediately) wins; from
# here up, split gate->routing->fc1 wins (routing in a clean window,
# no pad copy; the delayed shared fork still hides under the longer
# expert window). Measured B=8: 164 vs 171; B=128: 386 vs 372.
ROUTED_SPLIT_MIN_TOKENS = int(
    os.environ.get("B10_ROUTED_SPLIT_MIN_TOKENS", "64"))

# prefill path floor: measured +3% at 192, +6% at 256, +12% at 512,
# +22% at 2048-4096 tokens (bench_moe_kimi_k3, TP8). Below ~192 the
# CE AG's ~50 us latency floor cannot hide under the gate/shared
# compute and the sharded-GEMM FLOP cut is too small to pay for it.
PREFILL_MIN_TOKENS = 192


@triton.jit
def _split_gf_kernel(
    gf_ptr, latent_ptr, act_ptr,
    gf_stride,
    E: tl.constexpr, L_SRC: tl.constexpr, L_OUT: tl.constexpr,
    I: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    ACT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One pass over the merged input-GEMM output gf = [logits|latent|g|u]:
    copy the latent (L_OUT columns; 0 when fc1 is column-sharded - the
    AG kernel then reads the strided gf slice directly, L_SRC keeps the
    g/u offset right) and compute the shared-expert SwiGLU silu(g)*u
    (logits are consumed in place by the routing kernel)."""
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = gf_ptr + row * gf_stride

    mm = offs < L_OUT
    y = tl.load(src + E + offs, mask=mm, other=0.0)
    tl.store(latent_ptr + row * LATENT_STRIDE + offs, y, mask=mm)

    ao = offs - L_OUT
    am = (ao >= 0) & (ao < I)
    g = tl.load(src + E + L_SRC + ao, mask=am, other=0.0).to(tl.float32)
    u = tl.load(src + E + L_SRC + I + ao, mask=am, other=0.0).to(
        tl.float32)
    a = g * tl.sigmoid(g) * u
    tl.store(act_ptr + row * ACT_STRIDE + ao, a.to(tl.bfloat16), mask=am)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Single fused RMSNorm kernel (what TRT's RMSNorm module runs)."""
    from flashinfer.norm import rmsnorm

    return rmsnorm(x.contiguous(), weight, RMS_EPS)


class KimiK3MoEB10(KimiK3MoE):
    """Optimized decode forward; identical weights and semantics.

    Call :meth:`init_opt` AFTER weights are loaded/initialized, then
    optionally :meth:`set_opt_flags` between (re)captures to ablate.
    """

    # Decode regime cap: beyond this the layer falls back to the exact
    # baseline forward. Measured TP8 crossover (bench_moe_kimi_k3
    # --ablate): +9% at 128 tokens, -2% at 256 - the Lamport RS/AR wire
    # cost and the one-CTA-per-token routing grow with tokens while the
    # opt savings stay flat (see README section 3).
    OPT_MAX_TOKENS = 128

    def init_opt(
        self,
        *,
        max_batch: int = 64,
        fi_ar=None,
        fi_ar_shared=None,
        ag=None,
        rs_comm=None,
        ag_ce=None,
    ) -> None:
        world = self.mapping.tp_size
        rank = self.mapping.tp_rank
        self._world, self._rank = world, rank
        self._fi_ar = fi_ar
        # second AR instance (own workspace) for the ref_tail shared
        # AR - it runs CONCURRENTLY with fi_ar's latent norm_reduce
        self._fi_ar_sh = fi_ar_shared
        self._ag = ag
        self._rs_comm = rs_comm
        self._ag_ce = ag_ce
        self._max_batch = max_batch

        gate_w = self.gate.weight.data  # [896, 7168] bf16
        self._gate_w = gate_w
        self._gate_bias_f32 = (
            self.gate.e_score_correction_bias.data.float().contiguous()
        )
        self._gate_bias_bf16 = self._gate_bias_f32.to(torch.bfloat16)
        fc1_w = self.fc1_latent_proj.weight.data  # [3584, 7168]
        fc2_w = self.fc2_latent_proj.weight.data  # [7168, 3584]
        su_w = self.shared_experts.gate_up_proj.weight.data  # [2i_l, 7168]
        sd_w = self.shared_experts.down_proj.weight.data  # [7168, i_l]
        self._shared_gate_up = su_w
        self._shared_i = sd_w.shape[1]

        # fc2 column shard + tail pieces
        width = MOE_LATENT // world
        self._latent_width = width
        cols = slice(rank * width, (rank + 1) * width)
        self._fc1_w = fc1_w
        self._fc1_rows = fc1_w[cols].contiguous()
        self._fc2_local = fc2_w[:, cols].contiguous()
        self._fc2_full = fc2_w  # for the -fc2shard ablation tail
        self._tail_w_shared = sd_w.contiguous()
        self._norm_w = self.latent_norm.weight.data
        self._norm_w_slice = self._norm_w[cols].contiguous()

        # merged input GEMM weights. With overlap_shared (default) the
        # shared expert runs on an aux stream, so the merged GEMM is
        # only [gate | fc1(:full/slice)]; the -overlap ablation keeps
        # the old three-way merge [gate | fc1 | shared g/u] + split_gf.
        self._fused_gf_w = torch.cat([gate_w, fc1_w]).contiguous()
        self._fused_gf_w_sh = (
            torch.cat([gate_w, fc1_w[cols]]).contiguous()
            if world > 1
            else None
        )
        self._fused_in_w = torch.cat([gate_w, fc1_w, su_w]).contiguous()
        self._fused_in_w_sh = (
            torch.cat([gate_w, fc1_w[cols], su_w]).contiguous()
            if world > 1
            else None
        )

        # aux stream for the shared-expert chain (independent of the
        # routed path's AG/routing/experts/RS - hide it under them)
        self._aux_stream = torch.cuda.Stream(device=gate_w.device)
        self._evt_fork = torch.cuda.Event()
        self._evt_join = torch.cuda.Event()
        # prefill AG stream: HIGH priority so the CE arrival-sync
        # kernel gets SM slots while the gate/shared GEMMs saturate the
        # device (equal priority = it waits for the GEMM drain)
        self._pf_stream = torch.cuda.Stream(
            device=gate_w.device, priority=-1)
        self._pf_fork = torch.cuda.Event()
        self._pf_join = torch.cuda.Event()

        # Expert backend dispatch. CUTLASS: drive the same in-tree
        # torch.ops.trtllm.fused_moe kernels as the baseline, with our
        # precomputed routing. TRTLLM (production trtllm-gen, packed
        # MXFP4): with routing "ours" we call the backend's run_moe with
        # pre-computed top-k (ids+scales from our Triton kernel), which
        # makes the runner skip its in-kernel scores+top-k stage (a
        # 46.8 us single-cluster kernel at B=64 with 896 experts) and
        # run only the cheap permute pipeline; the -routing ablation
        # falls back to handing raw logits to forward_impl.
        backend = getattr(self.experts, "backend", self.experts)
        self._trtllm_gen = type(backend).__name__ == "TRTLLMGenFusedMoE"
        if self._trtllm_gen:
            self._gen_backend = backend
            # run_moe bypasses forward_impl's quantize_input, so the
            # handoff must replicate it (_quant_latent): production
            # w4a8_mxfp4_mxfp8 = dynamic MXFP8 activation quantize;
            # w4a16_mxfp4 fallback = F.pad to the packed weight width
            self._gen_mxfp8 = bool(
                getattr(backend, "has_w4a8_mxfp4_mxfp8", False))
            self._gen_pad = (
                backend.w3_w1_weight.shape[-1] * 2 - MOE_LATENT)
            # the baseline runs the production op backend (flashinfer,
            # see moe_trtllm_kimi_k3); its ROUTED dispatch packs
            # (ids<<16 | scale) with two extra elementwise kernels and
            # is +1..8 us slower for the precomputed-routing handoff.
            # The opt path is free to pick the fastest op, so its
            # run_moe/quantize calls pin the NATIVE trtllm op backend.
            from tensorrt_llm._torch.modules.fused_moe.moe_op_backend \
                import get_op_backend
            self._native_op_backend = get_op_backend("trtllm")
            # fin_fuse direct routed call: our routing kernel already
            # emits the (id<<16)|bf16(w) packed tensor the flashinfer
            # routed op consumes, so calling it DIRECTLY skips the op
            # wrapper's two elementwise pack kernels (~4 us at B=16).
            # Arg marshaling mirrors run_moe's w4a8_mxfp4_mxfp8 branch
            # + the wrapper's routed dispatch, frozen once here.
            rp = backend._extract_routing_params()
            fi_ob = backend.op_backend  # production flashinfer op
            self._fi_moe = fi_ob._fused_moe
            self._fi_routed_args = (
                backend.w3_w1_weight,
                backend.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                backend.w3_w1_bias if backend.bias else None,
                backend.swiglu_alpha,
                backend.swiglu_beta,
                backend.swiglu_limit,
                backend.w2_weight,
                backend.w2_weight_scale.view(torch.float8_e4m3fn),
                backend.w2_bias if backend.bias else None,
                backend._get_data_or_none("fc31_scale_c"),
                backend._get_data_or_none("fc31_alpha"),
                backend._get_data_or_none("fc2_alpha"),
                backend.num_slots,
                TOP_K,
                rp.n_group,
                rp.topk_group,
                backend.w3_w1_weight.shape[-2] // 2,  # padded inter
                backend.slot_start,
                backend.expert_size_per_partition,
                rp.routed_scaling_factor,
                fi_ob.cvt_routing_method_type(
                    backend.routing_method.routing_method_type),
            )
            self._fi_act_type = fi_ob.cvt_activation_type(
                backend._to_trtllm_gen_activation_type(
                    backend.activation_type))
            self._fi_tune_max = backend.max_num_tokens
        else:
            self._w31, self._w2 = self._expert_weights()
        # shared activation must match the baseline module's: SiTU in
        # production (trtllm) mode, SwiGLU in the cutlass bench mode
        self._shared_act_mod = (
            self.shared_experts.activation if self._trtllm_gen else None)

        # CuTeDSLGen tail GEMM (D = C + A @ fc2.T at ~6.3 TB/s vs
        # cuBLAS ~4.6 on this shape; vendored, graph-safe). Env-gated
        # so the ablation can isolate it: B10_TAIL_KERNEL=0 disables.
        self._tail_kernel = None
        if os.environ.get("B10_TAIL_KERNEL", "1") != "0":
            try:
                from .tail_gemm_cutedsl import moe_tail_gemm
                self._tail_kernel = moe_tail_gemm
            except Exception:  # noqa: BLE001 - cutedsl unavailable
                self._tail_kernel = None

        self.register_buffer(
            "_tail_in",
            torch.zeros(max_batch, self._shared_i,
                        device=gate_w.device, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "_zero_residual",
            torch.zeros(max_batch, MOE_LATENT,
                        device=gate_w.device, dtype=torch.bfloat16),
        )
        self.set_opt_flags()
        self._opt_ready = True

    def set_opt_flags(
        self,
        *,
        merged_gemm: bool = True,
        routing: str = "ours",
        fc1_shard: bool = True,
        fc2_shard: bool = True,
        comm: str = "fi",
        overlap_shared: bool = True,
        prefill_opt: bool = True,
        skip_shared: bool = False,
        rs_defer: bool = True,
        ref_tail: bool = True,
        fin_fuse: bool = True,
        merge3: bool = True,
    ) -> None:
        """Flip individual optimizations (between graph captures).

        skip_shared is a TIMING PROBE, not an optimization: it drops
        the shared-expert chain entirely (output is wrong by the
        shared contribution). If the shared chain is perfectly hidden
        under the routed critical path, skipping it changes e2e time
        by ~0; any delta is scheduling headroom.

        rs_defer: split the latent RS+norm into push (sumsq partials
        sent, slice returned unscaled) + a scale kernel issued AFTER
        the shared-expert tail GEMM, so the wait for peer scalars -
        the structural ~3.4 us a standalone fused RS+norm pays - is
        hidden under independent work instead of the critical path.

        ref_tail: from REF_TAIL_MIN_TOKENS tokens, replace the
        fc2-shard tail (RS+norm -> 1/world fc2 -> AR(hidden)) with
        AR+norm(latent) -> FULL fc2 - identical output on every rank,
        so NO output collective - while the shared partial's
        AR(hidden) runs on the aux stream, overlapped.

        fin_fuse (needs ref_tail + trtllm-gen + routing "ours"): run
        the experts UNFINALIZED (do_finalize=False via the flashinfer
        routed op) and replace finalizeKernel + AR + RMSNorm with ONE
        flashinfer trtllm_moe_finalize_allreduce_fusion kernel - the
        gather/scale of the expert partials rides the latent AR."""
        assert routing in ("ours", "noaux") and comm in ("custom", "fi")
        self._flag_merged = merged_gemm
        self._flag_routing = routing
        self._flag_fc1_shard = fc1_shard
        self._flag_fc2_shard = fc2_shard
        self._flag_comm = comm
        self._flag_overlap = overlap_shared
        self._flag_prefill = prefill_opt
        self._flag_skip_shared = skip_shared
        self._flag_rs_defer = rs_defer
        self._flag_ref_tail = ref_tail
        self._flag_fin_fuse = fin_fuse
        self._flag_merge3 = merge3

    def _expert_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Local [E_l, 2I, latent] gated fc1 + [E_l, latent, I] fc2 from
        whichever backend create_moe picked."""
        mod = self.experts
        for attr in ("w3_w1_weight", "w2_weight"):
            if hasattr(mod, attr):
                return mod.w3_w1_weight.data, mod.w2_weight.data
        backend = getattr(mod, "moe_backend", None) or getattr(
            mod, "backend", None)
        if backend is not None and hasattr(backend, "w3_w1_weight"):
            return backend.w3_w1_weight.data, backend.w2_weight.data
        names = [n for n, _ in mod.named_parameters()]
        raise AttributeError(
            f"cannot locate expert weights on {type(mod).__name__}; "
            f"params: {names[:8]}"
        )

    # -- optimized pieces --------------------------------------------

    def _shared_act(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Gated activation of the shared expert, matching the baseline
        module exactly (SiTUAndMul in production mode, SwiGLU in the
        cutlass comparison mode)."""
        if self._shared_act_mod is not None:
            return self._shared_act_mod(gate_up)
        from flashinfer.activation import silu_and_mul

        return silu_and_mul(gate_up)

    def _run_moe_unfinalized(self, x, packed, x_sf):
        """Expert bmms UNFINALIZED (do_finalize=False, no finalize
        kernel) via the flashinfer routed op called DIRECTLY with our
        routing kernel's packed (id<<16)|bf16(w) tensor - skipping the
        op wrapper's two elementwise pack kernels. (The native mxfp4
        runner has no do_finalize knob, hence flashinfer here.)
        Returns [gemm2_output, expert_weights,
        expanded_idx_to_permuted_idx] for moe_finalize_norm_reduce."""
        if x_sf is not None:
            if x_sf.dim() == 1:
                x_sf = x_sf.view(x.shape[0], -1)
            x_sf = x_sf.view(torch.float8_e4m3fn)
        out = self._fi_moe.trtllm_fp4_block_scale_routed_moe(
            packed, None, x, x_sf,
            *self._fi_routed_args,
            do_finalize=False,
            activation_type=self._fi_act_type,
            tune_max_num_tokens=self._fi_tune_max,
        )
        if out[2].dim() != 2:
            out[2] = out[2].view(-1, TOP_K)
        return out

    def _experts_trtllm_gen(
        self, latent: torch.Tensor, logits: torch.Tensor, batch: int,
        after_routing=None, unfinalized: bool = False,
    ) -> torch.Tensor:
        """Production trtllm-gen experts. With routing "ours": our
        Triton kernel reads the bf16 strided logits slice in place and
        emits (int32 ids, bf16 scales) for run_moe, so the runner skips
        its in-kernel scores+top-k (46.8 us at B=64) AND the
        logits.float() cast disappears. Ablation (-routing): raw logits
        through forward_impl -> in-kernel routing.

        after_routing: small-batch fork point for the shared aux
        chain (see SHARED_FORK_EXPERTS_MIN_TOKENS)."""
        if self._flag_routing == "ours":
            with torch.profiler.record_function("moe.routing"):
                # unfinalized: the routed op takes our packed
                # (id<<16)|bf16(w) tensor directly - no pack kernels
                ids, scales = (
                    route_pack(logits, self._gate_bias_f32)
                    if unfinalized
                    else route_for_trtllm_gen(logits, self._gate_bias_f32))
            if after_routing is not None:
                after_routing()
            x, x_sf = self._quant_latent(latent)
            with torch.profiler.record_function("moe.experts"):
                if unfinalized:
                    return self._run_moe_unfinalized(x, ids, x_sf)
                return self._run_moe_native(x, ids, scales, x_sf)
        if after_routing is not None:
            after_routing()  # in-kernel routing: fork before experts
        if self._gen_mxfp8 and not latent.is_contiguous():
            # the installed mxfp8_quantize op rejects strided input (a
            # merged-GEMM slice; production fc1 output is contiguous)
            with torch.profiler.record_function("moe.latent_copy"):
                latent = latent.contiguous()
        with torch.profiler.record_function("moe.experts"):
            return self.experts(
                latent, logits.float(),
                all_rank_num_tokens=[batch],
                use_dp_padding=False,
            )

    def _run_moe_native(self, x, ids, scales, x_sf):
        """run_moe with the NATIVE trtllm op backend pinned (the
        module-level op backend is the production flashinfer one; its
        routed dispatch packs (ids<<16 | scale) with two extra
        elementwise kernels - measured +1..8 us end-to-end). Host-side
        attribute swap only, CUDA-graph safe."""
        ob = self._gen_backend.op_backend
        self._gen_backend.op_backend = self._native_op_backend
        try:
            return self._gen_backend.run_moe(x, ids, scales, x_sf=x_sf)
        finally:
            self._gen_backend.op_backend = ob

    def _quant_latent(self, latent):
        """Replicate forward_impl's quantize_input (run_moe bypasses
        it). Production w4a8_mxfp4_mxfp8: dynamic MXFP8 activation
        quantize -> (fp8 x, ue8m0 x_sf); the baseline pays the same
        kernel inside forward_impl. Already a tuple when the fused
        AG+quantize produced it (fc1-shard path) - pass through.
        w4a16_mxfp4 fallback: F.pad to the packed weight width - the
        pad also clones, so skip it entirely when it would be a pure
        no-op copy of a contiguous latent."""
        if isinstance(latent, tuple):
            return latent
        if self._gen_mxfp8:
            with torch.profiler.record_function("moe.latent_quant"):
                ob = self._gen_backend.op_backend
                self._gen_backend.op_backend = self._native_op_backend
                try:
                    return self._gen_backend.quantize_input(
                        latent.contiguous(), post_quant_comm=False)
                finally:
                    self._gen_backend.op_backend = ob
        if self._gen_pad == 0 and latent.is_contiguous():
            return latent, None
        with torch.profiler.record_function("moe.latent_pad"):
            return (torch.nn.functional.pad(latent, (0, self._gen_pad)),
                    None)

    def _routing_opt(self, logits: torch.Tensor):
        """(ids int32, scales fp32) for fused_moe, by flag."""
        if self._flag_routing == "ours":
            return route_for_fused_moe(logits, self._gate_bias_f32)
        values, indices = torch.ops.trtllm.noaux_tc_op(
            logits, self._gate_bias_bf16, 1, 1, TOP_K, ROUTED_SCALING
        )
        return indices.to(torch.int32), values.float()

    def _split_gf_opt(self, gf: torch.Tensor, fc1_sharded: bool):
        batch = gf.shape[0]
        if fc1_sharded:
            lat_src, lat_out = self._latent_width, 0
            dst = self._tail_in  # unused: latent stores all masked out
        else:
            lat_src = lat_out = MOE_LATENT
            dst = torch.empty(batch, lat_out, device=gf.device,
                              dtype=torch.bfloat16)
        block = 1024
        tail_in = self._tail_in
        grid = (batch, triton.cdiv(lat_out + self._shared_i, block))
        _split_gf_kernel[grid](
            gf, dst, tail_in,
            gf.stride(0),
            E=NUM_EXPERTS, L_SRC=lat_src, L_OUT=lat_out,
            I=self._shared_i,
            LATENT_STRIDE=dst.stride(0),
            ACT_STRIDE=tail_in.stride(0),
            BLOCK=block,
        )
        return dst if not fc1_sharded else None

    def _experts_opt(
        self,
        latent: torch.Tensor,
        selected: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """The exact op the CUTLASS backend calls, with our routing.
        Returns the finalized [B, latent] partial sum (tokens routed to
        non-local experts are dropped; TP reduce happens downstream)."""
        result = torch.ops.trtllm.fused_moe(
            latent,
            selected,
            scales,
            self._w31,
            None,
            self._w2,
            None,
            torch.bfloat16,
            quant_scales=[],
            tp_size=1,
            tp_rank=0,
            ep_size=self._world,
            ep_rank=self._rank,
            cluster_size=1,
            cluster_rank=0,
            enable_alltoall=False,
            min_latency_mode=False,
            use_fused_finalize=True,
            activation_type=ActivationType.Swiglu,
        )
        return result[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        batch = hidden_states.view(-1, self.hidden_dim).shape[0]
        if getattr(self, "_opt_ready", False):
            if batch <= min(self.OPT_MAX_TOKENS, self._max_batch):
                return self._forward_opt(hidden_states)
            if (
                self._flag_prefill
                and self._world > 1
                and self._ag_ce is not None
                and batch >= PREFILL_MIN_TOKENS
                and batch * MOE_LATENT * 2 <= self._ag_ce.slot_bytes
            ):
                return self._forward_prefill(hidden_states)
        return super().forward(hidden_states, attn_metadata, **kwargs)

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill-scale path: sharded fc1/fc2 GEMMs (1/world of the
        baseline's replicated FLOPs on both latent projections) with the
        rebuild all-gather on the COPY ENGINES.

        At prefill sizes the latent GEMMs are compute-bound, so the
        column shard is a real 1/world FLOP cut (~105 us/GEMM at
        B=4096), not just a weight-read cut like at decode. The price -
        the [B, latent] AG - is DMA-engine traffic (zero SMs, ~+13 us
        of measured interference) issued on a high-priority side stream
        and hidden under the gate GEMM + routing + shared-expert chain,
        none of which depend on it. Total wire is unchanged vs baseline
        (AG latent + AR latent + AR hidden = the baseline's fused
        [B, latent+hidden] AR + the AG's extra latent, all NVLink).
        The only exposed sync is the CE arrival kernel (~6 us).

        Comm choices follow the baseline at this scale: TRT AUTO
        allreduce (NCCL) + unfused RMSNorm - the Lamport kernels'
        sentinel clears price them out past ~1k tokens.
        """
        h = hidden_states.view(-1, self.hidden_dim)
        cur = torch.cuda.current_stream()

        # ONE input GEMM for gate + this rank's fc1 column slice
        with torch.profiler.record_function("moe.pf_in_gemm"):
            gf = h @ self._fused_gf_w_sh.T
        logits = gf[:, :NUM_EXPERTS]

        # CE all-gather of the latent shard on the high-priority side
        # stream: pure DMA + one arrival kernel, overlaps everything
        # below by construction
        self._pf_fork.record(cur)
        self._pf_fork.wait(self._pf_stream)
        with torch.cuda.stream(self._pf_stream):
            with torch.profiler.record_function("moe.pf_ce_allgather"):
                latent = self._ag_ce.all_gather(
                    gf[:, NUM_EXPERTS:NUM_EXPERTS + self._latent_width])
        self._pf_join.record(self._pf_stream)

        # collective/compute pairing (both shard collectives hidden):
        # the SHARED-EXPERT chain runs on the aux stream, forked HERE
        # so it can slide into whichever stall window exists - the CE
        # all-gather wait at small S (gate+routing alone don't cover
        # the AG's ~50 us floor below ~1k tokens) or the latent
        # allreduce's wire time at large S. The main stream never
        # waits on it until the fc2 addmm.
        self._evt_fork.record(cur)
        self._evt_fork.wait(self._aux_stream)
        shared_ref = {}
        with torch.cuda.stream(self._aux_stream):
            with torch.profiler.record_function("moe.pf_shared_aux"):
                act = self._shared_act(h @ self._shared_gate_up.T)
                shared_ref["out"] = act @ self._tail_w_shared.T
        self._evt_join.record(self._aux_stream)

        # independent compute while the DMA flies
        selected = scales = None
        if not self._trtllm_gen:
            with torch.profiler.record_function("moe.pf_routing"):
                selected, scales = self._routing_opt(logits)

        self._pf_join.wait(cur)
        with torch.profiler.record_function("moe.pf_experts"):
            if self._trtllm_gen:
                routed = self._experts_trtllm_gen(
                    latent, logits, h.shape[0])
            else:
                routed = self._experts_opt(latent, selected, scales)

        # latent TP reduce + norm (baseline comm at this scale), then
        # the fc2 COLUMN slice accumulated into the shared partial -
        # the final AR reduces both at once, so fc2 is 1/world FLOPs
        # for the same total wire as the baseline
        with torch.profiler.record_function("moe.pf_ar_latent"):
            reduced = self.allreduce(routed)
        reduced = _rmsnorm(reduced, self._norm_w)
        self._evt_join.wait(cur)
        shared_out = shared_ref["out"]
        width = self._latent_width
        cols = slice(self._rank * width, (self._rank + 1) * width)
        with torch.profiler.record_function("moe.pf_tail_fc2"):
            shared_out.addmm_(reduced[:, cols], self._fc2_local.T)
        with torch.profiler.record_function("moe.pf_ar_out"):
            out = self.allreduce(shared_out)
        return out.view(hidden_states.shape)

    def _forward_opt(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = hidden_states.view(-1, self.hidden_dim)
        batch = h.shape[0]
        world = self._world
        fc1_sharded = (
            self._flag_fc1_shard
            and world > 1
            and batch <= FC1_SHARD_MAX_TOKENS
            and self._ag is not None
        )
        # fc1-shard AG with the MXFP8 quantize fused on the write-out:
        # only when the run_moe handoff consumes the result (production
        # backend + our routing); the -routing fallback needs bf16
        fuse_ag8 = (fc1_sharded and self._trtllm_gen
                    and self._gen_mxfp8 and self._flag_routing == "ours")
        use_custom = self._flag_comm == "custom" and world > 1
        # shared-expert overlap pays while the routed path leaves SMs
        # idle. With balanced routing (~60 active experts/rank) the
        # grouped GEMM is weight-read-bound and the ablation shows the
        # overlap winning through B=128 (B=64: 334 us on vs 350 off;
        # B=128: 412 vs 415); with artificially hot routing it lost
        # past ~32 - re-measure via B10_OVERLAP_MAX_TOKENS if routing
        # or kernels change
        overlap = self._flag_overlap and batch <= OVERLAP_SHARED_MAX_TOKENS
        # timing probe: drop the shared chain entirely (see
        # set_opt_flags docstring)
        skip_shared = getattr(self, "_flag_skip_shared", False)
        if skip_shared:
            overlap = False
        # _split_gf fuses SwiGLU - cutlass mode only; production (SiTU)
        # computes the shared activation via the baseline's module
        merge_shared = (not overlap and not self._trtllm_gen
                        and not skip_shared)

        # REF-overlap tail gate (see set_opt_flags): needs BOTH fi
        # instances - the shared AR runs concurrently with the latent
        # norm_reduce, so they cannot share a workspace
        ref_tail = (
            self._flag_ref_tail
            and world > 1
            and not skip_shared
            and self._fi_ar is not None
            and self._fi_ar_sh is not None
            and batch >= REF_TAIL_MIN_TOKENS
        )
        # expert-finalize fusion: run the experts UNFINALIZED and fold
        # the gather+scale into the tail collective - the flashinfer
        # finalize+AR+norm kernel on the ref tail, or our RS+norm
        # kernel's push stage on the fc2-shard tail. Needs our
        # precomputed routing (run_moe(do_finalize=False) handoff).
        # NOTE: a shard-tail variant (finalize gathered inside the RS
        # push, reduce_scatter_cols_finalize) measured ~6 us SLOWER at
        # B<=8 - the 16-way gather serializes on the one-CTA-per-token
        # push loop - so fin_fuse stays ref-tail-only.
        fin_fuse = (ref_tail and self._flag_fin_fuse
                    and self._trtllm_gen
                    and self._flag_routing == "ours")

        # shared-expert chain on the aux stream: it depends only on h,
        # so its two GEMMs + activation hide under the routed path's
        # AG/routing/experts/RS (same fork/join pattern as the
        # baseline's maybe_execute_in_parallel)
        shared_out = None
        cur = torch.cuda.current_stream()
        routed_pipeline = (self._trtllm_gen
                           and self._flag_routing == "ours"
                           and self._flag_merged
                           and batch >= ROUTED_SPLIT_MIN_TOKENS)

        # merge3: fold the shared gate/up rows into the ONE input GEMM
        # ([gate | fc1-shard | g | u], ~41 MB) - cuBLAS runs the wide-N
        # read at 7-8 TB/s where the narrow [gate|fc1-shard] merge only
        # reaches ~2 TB/s, and the aux chain shrinks to SiTU + down
        # GEMM (+AR), so the shared weight read stops competing with
        # the routed path for DRAM. The aux chain reads the g/u slice
        # of gf (SiTUAndMul makes it contiguous itself).
        merge3 = (self._flag_merge3 and overlap and self._trtllm_gen
                  and self._flag_merged and not routed_pipeline
                  and world > 1)

        def _fork_shared(gu=None):
            nonlocal shared_out
            self._evt_fork.record(cur)
            self._evt_fork.wait(self._aux_stream)
            with torch.cuda.stream(self._aux_stream):
                with torch.profiler.record_function("moe.shared_aux"):
                    act = self._shared_act(
                        h @ self._shared_gate_up.T if gu is None else gu)
                    shared_out = act @ self._tail_w_shared.T
                if ref_tail:
                    # chain the shared AR right here on the aux
                    # stream: every rank pushes at the same logical
                    # point (right after its shared GEMMs), so the
                    # collective completes early, hidden under the
                    # expert bmms
                    with torch.profiler.record_function(
                            "moe.shared_ar_aux"):
                        shared_out = self._fi_ar_sh(shared_out)
            self._evt_join.record(self._aux_stream)

        # Fork placement: ALL trtllm-gen paths fork AFTER routing. An
        # early fork parks the shared gate-up (13.7 us, ~100 MB weight
        # read) on top of the fused input GEMM (8.8 -> 10.8 us) and
        # routing (+1.7 us); after routing it hides under the expert
        # bmms + RS. A third option (fork after the expert kernels so
        # the gate-up sits in the RS spin-wait's DRAM-idle window) was
        # measured and DROPPED: noise at B<=16, ~6 us at B=32 only,
        # and it costs a batch gate + a different tail join - the
        # exposure that remains (-shared probe: ~7-13 us at B=8..32)
        # is the shared weight read competing for DRAM, which no fork
        # point removes. Non-gen paths fork first thing.
        late_fork = overlap and self._trtllm_gen and not routed_pipeline
        if overlap and not routed_pipeline and not late_fork:
            _fork_shared()

        ids = scales = None  # set by the routed trtllm-gen pipeline
        gu3 = None  # merge3's shared g/u slice (merged branch only)
        if routed_pipeline:
            # Routed input pipeline (production experts): small gate
            # GEMM, then our routing kernel in a CLEAN window, then
            # fc1 - all serial on the main stream. The routing kernel
            # is 16 dependent reductions and inflates 3-6x when GEMMs
            # hold the SMs (measured 6 -> 20-40 us overlapped with fc1
            # or the shared chain), so overlapping it always LOSES:
            # clean it costs ~6 us, and the shared chain it displaces
            # still hides fully under the ~180 us expert bmms. fc1's
            # own GEMM output is contiguous, so run_moe takes it with
            # no pad copy (the fused [gate|fc1] GEMM's strided slice
            # cost a 6.3 us pad-clone).
            with torch.profiler.record_function("moe.gate_gemm"):
                logits = h @ self._gate_w.T
            with torch.profiler.record_function("moe.routing"):
                ids, scales = (
                    route_pack(logits, self._gate_bias_f32)
                    if fin_fuse
                    else route_for_trtllm_gen(logits, self._gate_bias_f32))
            if overlap:
                _fork_shared()
            with torch.profiler.record_function("moe.fc1_gemm"):
                if fc1_sharded:
                    shard = h @ self._fc1_rows.T
                    latent = self._ag.all_gather_mxfp8(shard) \
                        if fuse_ag8 else self._ag(shard)
                else:
                    latent = h @ self._fc1_w.T
        elif self._flag_merged:
            # ONE input GEMM for gate + fc1 (+shard); the shared g/u
            # rows are only merged in when _split_gf handles them
            if merge_shared or merge3:
                w = self._fused_in_w_sh if fc1_sharded else self._fused_in_w
            else:
                w = self._fused_gf_w_sh if fc1_sharded else self._fused_gf_w
            with torch.profiler.record_function("moe.fused_in_gemm"):
                gf = h @ w.T
            logits = gf[:, :NUM_EXPERTS]
            lat_w = self._latent_width if fc1_sharded else MOE_LATENT
            gu3 = gf[:, NUM_EXPERTS + lat_w:] if merge3 else None
            if fc1_sharded:
                with torch.profiler.record_function("moe.fc1_allgather"):
                    e0 = NUM_EXPERTS
                    shard = gf[:, e0:e0 + self._latent_width]
                    latent = self._ag.all_gather_mxfp8(shard) \
                        if fuse_ag8 else self._ag(shard)
                if merge_shared:
                    with torch.profiler.record_function("moe.split_gf"):
                        self._split_gf_opt(gf, True)  # shared SwiGLU only
            elif merge_shared:
                with torch.profiler.record_function("moe.split_gf"):
                    latent = self._split_gf_opt(gf, False)
            elif self._trtllm_gen:
                if (self._gen_mxfp8 and self._flag_routing == "ours"
                        and os.environ.get("B10_QUANT_SLICE",
                                           "1") != "0"):
                    # fused strided-slice -> MXFP8 (ONE Triton pass,
                    # bit-exact vs mxfp8_quantize): kills the
                    # contiguous copy AND the quantize kernel with NO
                    # collective - the same fusion the fc1-shard AG
                    # carried, now available shard-free (bs 32..80)
                    with torch.profiler.record_function(
                            "moe.quant_slice"):
                        latent = quant_slice_mxfp8(
                            gf[:, NUM_EXPERTS:NUM_EXPERTS + lat_w])
                else:
                    # the trtllm-gen wrapper's pre-quant F.pad clones
                    # its input unconditionally (even at pad 0), which
                    # also makes it contiguous
                    latent = gf[:, NUM_EXPERTS:NUM_EXPERTS + lat_w]
            else:
                with torch.profiler.record_function("moe.latent_copy"):
                    latent = gf[:, NUM_EXPERTS:].contiguous()
        else:
            # unmerged ablation: separate gate / fc1 (/ shared) GEMMs
            with torch.profiler.record_function("moe.gate_gemm"):
                logits = h @ self._gate_w.T
            with torch.profiler.record_function("moe.fc1_gemm"):
                if fc1_sharded:
                    shard = h @ self._fc1_rows.T
                    latent = self._ag.all_gather_mxfp8(shard) \
                        if fuse_ag8 else self._ag(shard)
                else:
                    latent = h @ self._fc1_w.T
        shared_in_split_gf = merge_shared and self._flag_merged
        tail_act = None
        if not overlap and not shared_in_split_gf and not skip_shared:
            with torch.profiler.record_function("moe.shared_gemm"):
                # feed the tail GEMM the fresh activation directly -
                # staging it through _tail_in would be a pure copy
                # (_tail_in is only written in-place by _split_gf)
                tail_act = self._shared_act(h @ self._shared_gate_up.T)

        if self._trtllm_gen:
            if ids is not None:  # routed pipeline: ids/scales ready,
                # latent contiguous from its own GEMM
                x, x_sf = self._quant_latent(latent)
                with torch.profiler.record_function("moe.experts"):
                    routed = (
                        self._run_moe_unfinalized(x, ids, x_sf)
                        if fin_fuse
                        else self._run_moe_native(x, ids, scales, x_sf))
            else:
                routed = self._experts_trtllm_gen(
                    latent, logits, batch,
                    after_routing=(lambda: _fork_shared(gu3))
                    if late_fork else None,
                    unfinalized=fin_fuse)
        else:
            with torch.profiler.record_function("moe.routing"):
                selected, scales = self._routing_opt(logits)
            with torch.profiler.record_function("moe.experts"):
                routed = self._experts_opt(latent, selected, scales)

        # REF-overlap tail: after AR+norm every rank holds the SAME
        # full normed latent, so the FULL-weight fc2 output is
        # identical everywhere and needs NO output collective. Only
        # the shared partial must be reduced, and that AR(hidden) is
        # independent of the whole latent chain -> it runs on the aux
        # stream, hidden under AR+norm(latent) + full fc2. vs the
        # fc2-shard tail this trades a serial RS(latent)+AR(hidden)
        # for AR(latent) + an OVERLAPPED AR(hidden), paying the full
        # fc2 weight read back; it also drops the split-K rounding of
        # the sharded tail (partials never round through bf16).
        if ref_tail:
            if not overlap:
                with torch.profiler.record_function(
                        "moe.tail_gemm_shared"):
                    src = tail_act if tail_act is not None \
                        else self._tail_in[:batch]
                    shared_out = src @ self._tail_w_shared.T
                self._evt_fork.record(cur)
                self._evt_fork.wait(self._aux_stream)
                with torch.cuda.stream(self._aux_stream):
                    with torch.profiler.record_function(
                            "moe.shared_ar_aux"):
                        shared_out = self._fi_ar_sh(shared_out)
                self._evt_join.record(self._aux_stream)
            if fin_fuse:
                # ONE kernel: expert finalize (gather+scale of the
                # unfinalized gemm2 partials) + oneshot AR + RMSNorm
                g2, exp_w, exp_idx = routed
                with torch.profiler.record_function(
                        "moe.finalize_ar_norm"):
                    reduced = self._fi_ar.moe_finalize_norm_reduce(
                        g2, exp_idx, exp_w, self._norm_w,
                        self._zero_residual[:batch], RMS_EPS,
                    )
            else:
                with torch.profiler.record_function(
                        "moe.ar_norm_latent"):
                    reduced = self._fi_ar.norm_reduce(
                        routed, self._norm_w,
                        self._zero_residual[:batch], RMS_EPS,
                    )
            # shared AR lands while norm_reduce runs; fold the add
            # into the fc2 GEMM epilogue (saves an elementwise pass)
            self._evt_join.wait(cur)
            with torch.profiler.record_function(
                    "moe.tail_gemm_fc2_full"):
                out = (self._tail_kernel(reduced, self._fc2_full,
                                         shared_out)
                       if self._tail_kernel is not None
                       else torch.addmm(shared_out, reduced,
                                        self._fc2_full.T))
            return out.view(hidden_states.shape)

        # -fc2shard ablation: baseline-style tail. Assemble the shared
        # partial, pack [latent | shared] into ONE fused AR (the
        # baseline's own collective), norm, FULL fc2, add. Everything
        # upstream (routing, input pipeline, overlap) stays optimized,
        # so the ablation isolates exactly the fc2-shard trade:
        # (RS + 1/world fc2 + AR-hidden) vs (fat AR + full fc2).
        fc2_sharded = (self._flag_fc2_shard
                       and batch <= getattr(self, "_fc2_shard_max",
                                            FC2_SHARD_MAX_TOKENS))
        if world > 1 and not fc2_sharded:
            if skip_shared:
                reduced = self.allreduce(routed)
                normed = _rmsnorm(reduced, self._norm_w)
                with torch.profiler.record_function("moe.tail_gemm_fc2_full"):
                    out = normed @ self._fc2_full.T
                return out.view(hidden_states.shape)
            if overlap:
                self._evt_join.wait(cur)
            else:
                with torch.profiler.record_function("moe.tail_gemm_shared"):
                    src = tail_act if tail_act is not None \
                        else self._tail_in[:batch]
                    shared_out = src @ self._tail_w_shared.T
            with torch.profiler.record_function("moe.tail_fused_ar"):
                packed = torch.cat((routed, shared_out), dim=-1)
                packed = self.allreduce(packed)
            normed = _rmsnorm(packed[:, :MOE_LATENT], self._norm_w)
            with torch.profiler.record_function("moe.tail_gemm_fc2_full"):
                out = torch.addmm(
                    packed[:, MOE_LATENT:], normed, self._fc2_full.T)
            return out.view(hidden_states.shape)

        # latent reduce + norm + rank slice -> tail
        reduced_slice = None
        rs_deferred = False
        if world == 1:
            reduced = _rmsnorm(routed, self._norm_w)
        elif use_custom and self._rs_comm is not None:
            # column reduce-scatter with the RMSNorm fused: with
            # rs_defer the kernel only PUSHES its sumsq partial (no
            # wait) and the scale runs after the shared tail GEMM,
            # once the peer scalars have long landed. Fused wins below
            # RS_DEFER_MIN_TOKENS; skip_shared has no independent work
            # to hide the wait, so it always stays fused.
            rs_deferred = (self._flag_rs_defer and not skip_shared
                           and batch >= RS_DEFER_MIN_TOKENS)
            # (a finalize-fused variant of this RS exists -
            # reduce_scatter_cols_finalize - but measured ~6 us SLOWER
            # at B<=8: the 16-way gather serializes the push loop. See
            # moe_optimization.md "negative results".)
            with torch.profiler.record_function("moe.rs_latent"):
                reduced_slice = self._rs_comm.reduce_scatter_cols(
                    routed, norm_w=self._norm_w_slice, eps=RMS_EPS,
                    defer_scale=rs_deferred,
                )
        elif self._fi_ar is not None:
            # fused oneshot AR + zero-residual + RMSNorm (one kernel)
            with torch.profiler.record_function("moe.ar_norm_latent"):
                reduced = self._fi_ar.norm_reduce(
                    routed, self._norm_w,
                    self._zero_residual[:batch], RMS_EPS,
                )
        else:
            with torch.profiler.record_function("moe.ar_latent"):
                reduced = self.allreduce(routed)
            reduced = _rmsnorm(reduced, self._norm_w)

        if reduced_slice is None:
            width = self._latent_width
            cols = slice(self._rank * width, (self._rank + 1) * width)
            reduced_slice = reduced[:, cols]

        # tail: shared_down + sharded fc2 partials via beta=1 accumulation
        if skip_shared:
            with torch.profiler.record_function("moe.tail_gemm_fc2"):
                out = reduced_slice @ self._fc2_local.T
        elif overlap:
            self._evt_join.wait(cur)
            if rs_deferred:
                with torch.profiler.record_function("moe.rs_scale"):
                    reduced_slice = self._rs_comm.scale_deferred(
                        reduced_slice, self._norm_w_slice, eps=RMS_EPS)
            out = shared_out
            with torch.profiler.record_function("moe.tail_gemm_fc2"):
                out.addmm_(reduced_slice, self._fc2_local.T)
        else:
            with torch.profiler.record_function("moe.tail_gemm_shared"):
                src = tail_act if tail_act is not None \
                    else self._tail_in[:batch]
                out = src @ self._tail_w_shared.T
            if rs_deferred:
                with torch.profiler.record_function("moe.rs_scale"):
                    reduced_slice = self._rs_comm.scale_deferred(
                        reduced_slice, self._norm_w_slice, eps=RMS_EPS)
            with torch.profiler.record_function("moe.tail_gemm_fc2"):
                out.addmm_(reduced_slice, self._fc2_local.T)
        if world > 1:
            with torch.profiler.record_function("moe.allreduce_7168"):
                out = (
                    self._fi_ar(out)
                    if self._fi_ar is not None
                    else self.allreduce(out)
                )
        return out.view(hidden_states.shape)
