"""CuTeDSL KDA spec-verify replay_ssm_fused kernels (KDA.md schema 2.2) — black-box, importable.

Private shared implementation for the public ``b10_kda_replay_ssm_*_cutedsl``
modules. One parametrized kernel generator (``make_launcher``: compile-time
flags ``gated`` / ``prenorm``) is JIT-cached per flag combination + launch
shape. The bare and gated champions use genuinely different decompositions
(bare: 2 CTAs x 4 warps per (b,h); gated: 1 CTA x 8 warps so the RMSNorm is an
intra-CTA reduction), so the flag selects between the two verbatim kernel
bodies — each entry point compiles to exactly its champion's code path.

IDENTICAL TO (same end-to-end function, just faster):
  bare (``gated=False``):
  - TRT-LLM Triton ``fused_recurrent_gated_delta_rule_cached_replay_update``
    (tensorrt_llm/_torch/modules/fla/cached_replay.py, the
    ``use_replay_state_update`` path) — the ``verify_replay_ssm_fused`` row in
    ``bench_kda_spec_verify.py`` / KDA.md ``replay_ssm_fused``
  gated (``gated=True``): the FUSION of these two Triton kernels into ONE launch:
  - TRT-LLM ``fused_recurrent_gated_delta_rule_cached_replay_update`` (as above)
  - SGLang ``FusedRMSNormGated`` (fla/fused_norm_gate.py, sigmoid)
Same contract: raw gate/beta in (safe gate fused), checkpoint +
(old_u, old_k, old_G) rings, pnat/buf_idx, in-kernel replay, fold-on-overflow.
Gated output is POST-norm.
NOT the SGLang save_ssm path — that is ``b10_kda_save_ssm_cutedsl``.

DROP-IN API? YES (bare) — import the TRT name from the public bare module:
    from kda.b10.b10_kda_replay_ssm_cutedsl import (
        fused_recurrent_gated_delta_rule_cached_replay_update,
    )
  Same keyword/positional contract as TRT-LLM's Triton entry
  (tensorrt_llm/_torch/modules/fla/cached_replay.py). Unused KDA-irrelevant
  kwargs (``old_beta``, tuning flags, …) are accepted and ignored.
  ``state_indices``: identity ``0..B-1`` into a B-slot pool is zero-copy;
  non-identity indices gather/scatter around the kernel (correct, not free).
ALMOST (gated) — TRT replay_ssm_fused with fused norm knobs vs trailing z,w.
  Triton: same cached-replay call with output_gate + norm_weight (bf16) kwargs
           fuse_gated_rmsnorm; still needs old_beta / state_indices / history_size.
  This:   kda_replay_ssm_fused_gated(..., pnat, buf_idx, z, w) — w fp32 [128];
           post-norm o.

Merged from the CuTeDSLGen champions gen_kda_replay_ssm_fused_claude_0729_0238/best
(bare) and gen_kda_replay_ssm_fused_gated_claude_0729_1535/best (gated), scaffolding
removed. WINS: bare matches TensorRT-LLM trt_cached_replay output and beats it
1.7-3.3x at every shape (steady + overflow regimes); gated matches trt_cached_replay
+ FusedRMSNormGated, primary 30.28 us, beats trt+norm 2.6-6.3x and [this kernel
unfused + a separate norm] 1.62x — the fused gated-RMSNorm epilogue is nearly free.

Public API:
    # drop-in (TRT signature) — one-line swap
    o = fused_recurrent_gated_delta_rule_cached_replay_update(
            q, k, v, g, beta, ssm_states, state_indices, old_u, old_k, old_G,
            old_beta, cache_buf_idx, prev_num_accepted_tokens, history_size,
            A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=True)
    # or native (bare)
    o = kda_replay_ssm_fused(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                        S, old_u, old_k, old_G, pnat, buf_idx)
    # gated (fused gated-RMSNorm epilogue; z [B,T,H,128] bf16, w [128] fp32)
    o = kda_replay_ssm_fused_gated(q, k, v, g_raw, beta_raw, A_log, dt_bias,
                              lower_bound, S, old_u, old_k, old_G, pnat, buf_idx,
                              z, w)
    kda_replay_ssm_fused_prenorm(...) is the epilogue-off ablation (returns PRE-norm
    r on the gated 1-CTA body) for the "our kernel unfused + a separate norm"
    timing baseline.
    Cached-update ring contract (see kda_specdec_spec.md / KDA.md 4.4). B200
    sm_100a; JIT on first call.
"""
# ============================ kernel module ============================
import math

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
F16 = cutlass.Float16
I32 = cutlass.Int32

HIST = 16
LOG_HIST = 4
VEC = 4                  # fp32 elements per 128-bit access
LOG_VEC = 2
VECH = 8                 # bf16 elements per 128-bit access
LOG_VECH = 3
K_EPS = 1e-6
NORM_EPS = 1e-5
CONV_W = 4               # causal conv width (KERNEL_WIDTH in SGLang Triton)
CSL = CONV_W - 1         # conv_state history slots (last 3 raw tokens)
NTI = 3                  # packed tensors in the conv stream (q | k | v)
# The gated body's (LK, NCH) state tiling is chosen per T -- see
# _make_onecta_launcher.  T <= T_CHUNK_MAX uses LK=16/NCH=2 (32-register tile,
# 3 CTAs/SM); larger T uses LK=8/NCH=1 (64-register tile, 2 CTAs/SM).
T_CHUNK_MAX = 5


@cute.jit
def _sigmoid(x: FP32) -> FP32:
    """1/(1+exp(-x)) via ex2.approx + rcp.approx.ftz (max rel err ~2^-22)."""
    return cute.arch.rcp_approx(
        FP32(1.0) + FP32(cute.math.exp(FP32(0.0) - x, fastmath=True)))


def _conv_gather(mXv, brow, jtok, cg, xf):
    """TRACE-TIME MACRO: issue the CONV_W-tap raw window of local token `jtok`
    for this lane's VEC consecutive channels into ``xf[:, jj]`` -- ALL loads
    first, each one 64-bit.

    The row index is CLAMPED (``max(t, 0)``) rather than predicated, so every
    load is unconditional and ``xf`` is never written under a branch: a
    fragment defined inside an ``scf.if`` is demoted by ptxas to local memory
    (this module measured 276K/282K local ld/st sectors in that form).  A
    clamped re-load of row 0 that is then discarded costs nothing extra -- the
    warps of a token share their window rows through L1 anyway."""
    for jj in range(CONV_W):
        tc = cutlass.max(jtok - (CONV_W - 1) + jj, I32(0))
        cute.autovec_copy(mXv[brow + tc, cg, None], xf[None, jj])


@cute.jit
def _conv_fold(sCSs: cute.Tensor, sCWs: cute.Tensor, lg: I32, jtok: I32,
               xf: cute.Tensor, csf: cute.Tensor, wfj: cute.Tensor,
               out: cute.Tensor, wf12: cute.Tensor, mCWin: cute.Tensor,
               xrow: I32, cg: I32, wr: I32):
    """Width-4 depthwise causal conv + SiLU for one lane's VEC consecutive
    channels of one tensor: fold the CONV_W taps, substituting the SMEM-staged
    ``conv_state`` slot for the taps that predate the launch.

    fp32 accumulate, fp32 SiLU, THEN a bf16 round into ``out`` so the
    recurrence sees exactly what the separate conv launch would have handed
    it.  When ``wr != 0`` the post-token raw window (slots jtok-2..jtok) of
    the lane's VEC channels is written as ONE vectorized 24 B store through
    the BLOCKED conv_win view ``mCWin[ntok][CDIM/VEC][VEC*CSL]`` (channel
    fastest-by-CSL, exactly conv_win's memory order).  Writing the window as
    per-slot scalar stores instead touched every 32 B sector of the plane up
    to 12 times: ncu measured 1.44M global store sectors against 98K for the
    flag-off build, and the conv build ran ~25% slower on nothing but that.

    * the state tap is ONE 64-bit LDS out of the transposed sCSs[CSL][3K]
      plane (a dynamic SMEM index is just address arithmetic); ``csf`` is
      loaded UNCONDITIONALLY with a clamped slot, so it is never a fragment
      defined inside an ``scf.if`` -- only plain bf16 scalars are reassigned
      under the ``t < 0`` arm, which the AST preprocessor yields through the
      branch as registers.
    * the tap weights are ONE conflict-free LDS.128 out of the fp32 tap-major
      sCWs[CONV_W][3K] plane, staged once per CTA instead of converted
      bf16->fp32 redundantly by every warp of the prologue group."""
    a0 = FP32(0.0)
    a1 = FP32(0.0)
    a2 = FP32(0.0)
    a3 = FP32(0.0)
    for jj in cutlass.range_constexpr(CONV_W):
        t = jtok - (CONV_W - 1) + jj
        slc = cutlass.min(jtok + jj, I32(CSL - 1))
        cute.autovec_copy(sCSs[slc, lg, None], csf)
        cute.autovec_copy(sCWs[jj, lg, None], wfj)
        v0 = xf[0, jj]
        v1 = xf[1, jj]
        v2 = xf[2, jj]
        v3 = xf[3, jj]
        if t < 0:
            v0 = csf[0]
            v1 = csf[1]
            v2 = csf[2]
            v3 = csf[3]
        if cutlass.const_expr(jj >= 1):
            wf12[0 * CSL + jj - 1] = v0
            wf12[1 * CSL + jj - 1] = v1
            wf12[2 * CSL + jj - 1] = v2
            wf12[3 * CSL + jj - 1] = v3
        a0 = a0 + wfj[0] * v0.to(FP32)
        a1 = a1 + wfj[1] * v1.to(FP32)
        a2 = a2 + wfj[2] * v2.to(FP32)
        a3 = a3 + wfj[3] * v3.to(FP32)
    if wr != 0:
        cute.autovec_copy(wf12, mCWin[xrow, cg, None])
    a0 = a0 * cute.arch.rcp_approx(FP32(1.0) + FP32(
        cute.math.exp(FP32(0.0) - a0, fastmath=True)))
    a1 = a1 * cute.arch.rcp_approx(FP32(1.0) + FP32(
        cute.math.exp(FP32(0.0) - a1, fastmath=True)))
    a2 = a2 * cute.arch.rcp_approx(FP32(1.0) + FP32(
        cute.math.exp(FP32(0.0) - a2, fastmath=True)))
    a3 = a3 * cute.arch.rcp_approx(FP32(1.0) + FP32(
        cute.math.exp(FP32(0.0) - a3, fastmath=True)))
    out[0] = a0.to(BF16)
    out[1] = a1.to(BF16)
    out[2] = a2.to(BF16)
    out[3] = a3.to(BF16)


def make_launcher(H: int, K: int, V: int, T: int,
                  gated: bool = False, prenorm: bool = False,
                  fuse_conv: bool = False):
    """Build a compiled-once launcher closure for a given (H, K, V, T, flags).

    ``gated=False, prenorm=False`` -> the bare champion body (2 CTAs x 4 warps
    per (b,h), no epilogue) -- bit- and latency-identical to the pre-merge
    this module's ``make_launcher``.
    ``gated=True`` -> the 1-CTA x 8-warp body with the gated-RMSNorm epilogue
    compiled IN (the production gated path).
    ``prenorm=True`` -> the same 1-CTA body with the epilogue compiled OUT
    (returns raw pre-norm r); exists only so the bench can price the fused
    epilogue against "this kernel + a separate norm launch".
    ``fuse_conv=True`` -> the width-4 causal-conv+SiLU input stage is compiled
    into the prologue: the kernel consumes RAW packed mixed_qkv plus a
    read-only conv_state (rollback semantics) and writes per-token raw window
    snapshots, with numerics identical to a separate SGLang
    ``causal_conv1d_update(activation="silu")`` launch (fp32 accumulate, fp32
    SiLU, then a bf16 round before the recurrence).
    """
    assert not (gated and prenorm)
    if gated or prenorm:
        return _make_onecta_launcher(H, K, V, T, gated=gated,
                                     fuse_conv=fuse_conv)
    return _make_bare_launcher(H, K, V, T, fuse_conv=fuse_conv)


def _make_bare_launcher(H: int, K: int, V: int, T: int,
                        fuse_conv: bool = False):
    """The bare champion body: grid (VSPLITS, H, B), 4 warps per CTA."""
    assert K == 128 and V == 128, "spec fixes K = V = 128"
    assert 1 <= T <= 8
    NWARP = 4
    LK = 8                   # lanes over K: all K partials of a column live in ONE warp
    LOG_LK = 3
    LV = 32 // LK            # 4 lanes over V
    LOG_LV = 2
    CPT = 4                  # V columns per thread == VEC
    THREADS = NWARP * 32     # 128
    WVS = LV * CPT           # 16 V columns per warp
    VS = NWARP * WVS         # 64 V columns per CTA
    KPT = K // LK                    # 16 key channels per thread
    NKG = KPT // VEC                 # 4 K-groups of VEC per thread
    assert KPT * CPT == 64           # 64 fp32 state registers per thread
    VSPLITS = V // VS                # 2
    assert VSPLITS * VS == V
    TPW = (T + NWARP - 1) // NWARP   # prologue tokens per warp
    RSQRT_K = 1.0 / math.sqrt(K)
    KV4 = K // VEC                   # 32 == warp size: one 128-bit group per lane
    assert KV4 == 32
    HV4 = HIST // VEC                # 4
    KV8 = K // VECH                  # 16 ring-k chunks of 8 bf16
    LOG_KV8 = 4
    VS8 = VS // VECH                 # 8 ring-u chunks of 8 bf16
    LOG_VS8 = 3
    NIT_K = (HIST * KV8 + THREADS - 1) // THREADS     # 2
    NIT_U = (HIST * VS8 + THREADS - 1) // THREADS     # 1
    ROWSG = HIST * H                 # rows per (slot, half) in the u/k rings
    KD = H * K                       # packed-qkv channel-block size (fuse_conv)
    LOG_K = 7

    @cute.kernel
    def blackwell_bf16_kda_replay_ssm_fused_kernel(
        mQ: cute.Tensor,      # [B*T*H, K/4, 4]       bf16
        mKt: cute.Tensor,     # [B*T*H, K/4, 4]       bf16
        mV2: cute.Tensor,     # [B*T*H, V/2, 2]       bf16
        mGr: cute.Tensor,     # [B*T*H, K/4, 4]       fp32  raw gate
        mBr: cute.Tensor,     # [B*T*H]               fp32  raw beta
        mAl: cute.Tensor,     # [H]                   fp32  A_log
        mDb: cute.Tensor,     # [H, K/4, 4]           fp32  dt_bias
        mS: cute.Tensor,      # [B*H, V, K/4, 4]      fp32  checkpoint (in / cond out)
        mO: cute.Tensor,      # [B*T*H, V/4, 4]       bf16
        mOu4: cute.Tensor,    # [B*2*HIST*H, V/4, 4]  bf16  ring u (store view)
        mOu8: cute.Tensor,    # [B*2*HIST*H, V/8, 8]  bf16  ring u (load view)
        mOk4: cute.Tensor,    # [B*2*HIST*H, K/4, 4]  bf16  ring k (store view)
        mOk8: cute.Tensor,    # [B*2*HIST*H, K/8, 8]  bf16  ring k (load view)
        mOgV: cute.Tensor,    # [B*2*H, K, HIST/4, 4] fp32  ring G (vector view)
        mOgS: cute.Tensor,    # [B*2*H, K, HIST]      fp32  ring G (scalar view)
        mPn: cute.Tensor,     # [B]                   int32 pnat
        mBi: cute.Tensor,     # [B]                   int32 buf_idx
        mX: cute.Tensor,      # [B*T, 3*H*K]          bf16  raw packed qkv (conv)
        mCS: cute.Tensor,     # [B, 3*H*K, 3]         bf16  conv history (RO)
        mCWt: cute.Tensor,    # [3*H*K, 4]            bf16  conv taps
        mCWin: cute.Tensor,   # [B*T, 3*H*K, 3]       bf16  window snapshots (out)
        lower_bound: FP32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        vsi, h, b = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        w = cute.arch.warp_idx()
        lv = lane & (LV - 1)
        lk = lane >> LOG_LV

        bh = b * H + h
        vbase = vsi * VS
        # Warp w owns V columns [vbase + w*WVS, +WVS) and ALL K channels of them,
        # so the per-token K-reduction is a 3-step lane butterfly -- no SMEM, no
        # barrier.  lk indexes the K partials (8 lanes), lv the V sub-group.
        kg0 = lk                             # + jj*LK, in VEC units
        vr0 = vbase + w * WVS + lv * VEC     # + cw
        vg0 = vr0 >> LOG_VEC                 # same, in VEC units

        # ---- P0: issue the state loads first; HBM latency overlaps P1a/P2 -----
        # one instruction = LV rows x (LK*16 B) = 4 rows x 128 B, all sectors full
        st = cute.make_fragment((VEC, NKG, CPT), FP32)
        for jj in cutlass.range_constexpr(NKG):
            for c in cutlass.range_constexpr(CPT):
                cute.autovec_copy(mS[bh, vr0 + c, kg0 + jj * LK, None],
                                  st[None, jj, c])

        # ---- shared memory (explicit row-major strides -- see module docstring)
        smem = SmemAllocator()
        sD = smem.allocate_tensor(FP32, cute.make_layout((KV4, VEC),
                                                         stride=(VEC, 1)), 16)
        sC = smem.allocate_tensor(FP32, cute.make_layout(
            (HIST, KV4, VEC), stride=(K, VEC, 1)), 16)
        sUo = smem.allocate_tensor(FP32, cute.make_layout(
            (HIST, VS // VEC, VEC), stride=(VS, VEC, 1)), 16)
        sDG = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sGL = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        # kn/qn are staged in **fp32**, not fp16.  fp16 halved these two planes'
        # SMEM words but put an HADD2_F32 widening on the LDS -> FFMA2 critical
        # path of every channel of every token: 128 HADD2 per warp, 6.8% of all
        # instructions, for a stream that measurement showed is not
        # bandwidth-limited (l1tex 67% of peak-active, FMA 28%).  fp32 removes the
        # conversions, shortens the dependency chain, and is strictly more
        # accurate.  The *persisted* ring copy of kn is still bf16 of this same
        # fp32 value, written straight out of the prologue registers.
        sKN = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sQN = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sV = smem.allocate_tensor(FP32, cute.make_layout(
            (T, VS // VEC, VEC), stride=(VS, VEC, 1)), 16)
        sSC = smem.allocate_tensor(FP32, cute.make_layout((T, 2),
                                                          stride=(2, 1)), 16)
        # fused-conv planes (see the onecta body / _conv_fold for the whole
        # story): conv_state transposed bf16 + fp32 tap-major taps, once/CTA
        if cutlass.const_expr(fuse_conv):
            sCSs = smem.allocate_tensor(BF16, cute.make_layout(
                (CSL, NTI * K // VEC, VEC), stride=(NTI * K, VEC, 1)), 16)
            sCWs = smem.allocate_tensor(FP32, cute.make_layout(
                (CONV_W, NTI * K // VEC, VEC), stride=(NTI * K, VEC, 1)), 16)

        # ---- ring bookkeeping (all branch-free; every value is CTA-uniform) ---
        pn = mPn[b]
        half = mBi[b] & 1
        # overflow iff pn + T > HIST; pn in [0,16], T in [1,8] -> pn+T-1 in [0,23]
        ovi = (pn + T - 1) >> LOG_HIST                 # 0 or 1
        half_out = half ^ ovi
        pos0 = pn - pn * ovi                           # pn on append, 0 on fold
        rg = (b * 2 + half) * H + h                    # ring G row (read half)
        rg_o = (b * 2 + half_out) * H + h              # ring G row (write half)
        ru = (b * 2 + half) * ROWSG + h                # ring u/k base (read half)
        ru_o = (b * 2 + half_out) * ROWSG + h          # ring u/k base (write half)
        pm1 = cutlass.max(pn - 1, I32(0))

        # ---- PS: stage conv_state (transposed) + fp32 taps, then barrier 0 ---
        # (identical scheme to the onecta body; flag-off builds compile none
        # of this, so the bare flag-off path keeps its barrier count.)
        if cutlass.const_expr(fuse_conv):
            NCC = (NTI * K + THREADS - 1) // THREADS
            csv = cute.make_fragment((CSL, NCC), BF16)
            cwv = cute.make_fragment((CONV_W, NCC), BF16)
            for it in cutlass.range_constexpr(NCC):
                lcc = cutlass.min(tid + it * THREADS, I32(NTI * K - 1))
                gcc = (lcc >> LOG_K) * KD + h * K + (lcc & (K - 1))
                for m_ in cutlass.range_constexpr(CSL):
                    csv[m_, it] = mCS[b, gcc, m_]
                cute.autovec_copy(mCWt[gcc, None], cwv[None, it])
            for it in cutlass.range_constexpr(NCC):
                lcs = tid + it * THREADS
                if lcs < NTI * K:
                    for m_ in cutlass.range_constexpr(CSL):
                        sCSs[m_, lcs >> LOG_VEC, lcs & (VEC - 1)] = csv[m_, it]
                    for jj in cutlass.range_constexpr(CONV_W):
                        sCWs[jj, lcs >> LOG_VEC, lcs & (VEC - 1)] = (
                            cwv[jj, it].to(FP32))
            cute.arch.sync_threads()

        # ---- P1a: stage the ring records, 8 elements per lane ----------------
        # old_k is contiguous in K and old_u in V, so both are read as 128-bit
        # chunks of 8 bf16 and widened into the fp32 staging buffers.  sC holds
        # the raw k here; P1b multiplies the decay coefficient in place.
        rh8 = cute.make_fragment((VECH,), BF16)
        rf8 = cute.make_fragment((VEC, 2), FP32)
        for it in cutlass.range_constexpr(NIT_K):
            i = tid + it * THREADS
            s = i >> LOG_KV8
            ch = i & (KV8 - 1)
            if s < pn:
                cute.autovec_copy(mOk8[ru + s * H, ch, None], rh8)
                for e in cutlass.range_constexpr(VECH):
                    rf8[e & (VEC - 1), e >> LOG_VEC] = rh8[e].to(FP32)
                for g2 in cutlass.range_constexpr(2):
                    cute.autovec_copy(rf8[None, g2],
                                      sC[s, ch * (VECH // VEC) + g2, None])
        for it in cutlass.range_constexpr(NIT_U):
            i = tid + it * THREADS
            s = i >> LOG_VS8
            ch = i & (VS8 - 1)
            if s < pn:
                cute.autovec_copy(mOu8[ru + s * H, (vbase >> LOG_VECH) + ch, None],
                                  rh8)
                for e in cutlass.range_constexpr(VECH):
                    rf8[e & (VEC - 1), e >> LOG_VEC] = rh8[e].to(FP32)
                for g2 in cutlass.range_constexpr(2):
                    cute.autovec_copy(rf8[None, g2],
                                      sUo[s, ch * (VECH // VEC) + g2, None])

        # ---- P2: new-token prologue (lane owns K channels 4*lane .. 4*lane+3) -
        # KV4 == 32 == warp size, so one 128-bit group per lane covers all of K
        # with a single instruction per tensor per token.
        alpha = FP32(cute.math.exp(mAl[h], fastmath=True))
        dtb = cute.make_fragment((VEC,), FP32)
        cute.autovec_copy(mDb[h, lane, None], dtb)     # head constant, hoisted
        qh = cute.make_fragment((VEC,), BF16)
        kh = cute.make_fragment((VEC,), BF16)
        gr = cute.make_fragment((VEC,), FP32)
        gv = cute.make_fragment((VEC,), FP32)
        dgv = cute.make_fragment((VEC,), FP32)
        knw = cute.make_fragment((VEC,), FP32)
        qnw = cute.make_fragment((VEC,), FP32)
        knb = cute.make_fragment((VEC,), BF16)
        vh2 = cute.make_fragment((2,), BF16)
        # fused-conv scratch (see _conv_gather/_conv_fold); rf2 is the clamped
        # raw window of this lane's 2 v channels, loads-first for the same
        # no-fragment-under-a-branch reason
        if cutlass.const_expr(fuse_conv):
            xf = cute.make_fragment((VEC, CONV_W), BF16)
            csf = cute.make_fragment((VEC,), BF16)
            wfj = cute.make_fragment((VEC,), FP32)
            wf12 = cute.make_fragment((VEC * CSL,), BF16)
            rf2 = cute.make_fragment((CONV_W, 2), BF16)
        for rr in cutlass.range_constexpr(TPW):
            j = w + rr * NWARP
            if j < T:
                row = (b * T + j) * H + h
                if cutlass.const_expr(fuse_conv):
                    # ---- width-4 causal conv + SiLU for this lane's 4 q and
                    # 4 k channels (clamped unconditional window loads,
                    # SMEM-staged state + taps -- see _conv_fold).  q/k window
                    # snapshots are written once (vsi == 0); each CTA writes
                    # its own v slice below.
                    xrow = b * T + j
                    wr = I32(1) - vsi          # q/k snapshots written once
                    _conv_gather(mX, b * T, j, h * KV4 + lane, xf)
                    _conv_fold(sCSs, sCWs, lane, j, xf, csf, wfj, qh,
                               wf12, mCWin, xrow, h * KV4 + lane, wr)
                    _conv_gather(mX, b * T, j,
                                 (KD >> LOG_VEC) + h * KV4 + lane, xf)
                    _conv_fold(sCSs, sCWs, KV4 + lane, j, xf, csf, wfj, kh,
                               wf12, mCWin, xrow,
                               (KD >> LOG_VEC) + h * KV4 + lane, wr)
                else:
                    cute.autovec_copy(mQ[row, lane, None], qh)
                    cute.autovec_copy(mKt[row, lane, None], kh)
                cute.autovec_copy(mGr[row, lane, None], gr)
                nq = FP32(0.0)
                nk = FP32(0.0)
                kq = FP32(0.0)
                for e in cutlass.range_constexpr(VEC):
                    qe = qh[e].to(FP32)
                    ke = kh[e].to(FP32)
                    nq = nq + qe * qe
                    nk = nk + ke * ke
                    kq = kq + qe * ke
                    gv[e] = lower_bound * _sigmoid(alpha * (gr[e] + dtb[e]))
                # one warp owns all K channels of a token -> pure warp reduction
                for off in cutlass.range_constexpr(5):
                    nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                    nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                    kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
                rk = FP32(cute.math.rsqrt(nk + FP32(K_EPS), fastmath=True))
                rqs = FP32(cute.math.rsqrt(nq, fastmath=True)) * FP32(RSQRT_K)
                for e in cutlass.range_constexpr(VEC):
                    dgv[e] = FP32(cute.math.exp(gv[e], fastmath=True))
                    kv = kh[e].to(FP32) * rk
                    knw[e] = kv
                    knb[e] = kv.to(BF16)
                    qnw[e] = qh[e].to(FP32) * rqs
                cute.autovec_copy(dgv, sDG[j, lane, None])
                cute.autovec_copy(gv, sGL[j, lane, None])
                cute.autovec_copy(knw, sKN[j, lane, None])
                cute.autovec_copy(qnw, sQN[j, lane, None])
                # the persisted ring k: bf16 of the fp32 normalised value.  The
                # VSPLITS CTAs split the tokens so each record is written once.
                if (j & (VSPLITS - 1)) == vsi:
                    cute.autovec_copy(knb, mOk4[ru_o + (pos0 + j) * H, lane, None])
                if cutlass.const_expr(fuse_conv):
                    # conv for this lane's 2 v channels (this CTA's own slice:
                    # window snapshots are written unconditionally).  Clamped
                    # unconditional window loads into rf2 first, then the
                    # fold; state taps come from the SMEM-staged plane.
                    xrow2 = b * T + j
                    for i in cutlass.range_constexpr(CONV_W):
                        rowc = b * T + cutlass.max(
                            j - (CONV_W - 1) + i, I32(0))
                        for e in cutlass.range_constexpr(2):
                            cch = 2 * KD + h * K + vbase + 2 * lane + e
                            rf2[i, e] = mX[rowc, cch >> LOG_VEC,
                                           cch & (VEC - 1)]
                    for e in cutlass.range_constexpr(2):
                        vch = vbase + 2 * lane + e
                        cch = 2 * KD + h * K + vch
                        lc = 2 * K + vch              # local channel in sCSs
                        # blocked conv_win view: [row][cch >> LOG_VEC] holds
                        # the VEC x CSL window block, channel-major
                        wsub = (cch & (VEC - 1)) * CSL
                        acc = FP32(0.0)
                        for i in cutlass.range_constexpr(CONV_W):
                            xb = rf2[i, e]
                            if cutlass.const_expr(i < CONV_W - 1):
                                if j - (CONV_W - 1) + i < 0:
                                    xb = sCSs[j + i, lc >> LOG_VEC,
                                              lc & (VEC - 1)]
                                if cutlass.const_expr(i >= 1):
                                    mCWin[xrow2, cch >> LOG_VEC,
                                          wsub + i - 1] = xb
                            else:
                                mCWin[xrow2, cch >> LOG_VEC,
                                      wsub + CONV_W - 2] = xb
                            acc = acc + sCWs[i, lc >> LOG_VEC,
                                             lc & (VEC - 1)] * xb.to(FP32)
                        acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                            cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                        vh2[e] = acc.to(BF16)
                else:
                    cute.autovec_copy(mV2[row, (vbase >> 1) + lane, None], vh2)
                sV[j, (2 * lane) >> LOG_VEC, (2 * lane) & (VEC - 1)] = (
                    vh2[0].to(FP32))
                sV[j, (2 * lane + 1) >> LOG_VEC, (2 * lane + 1) & (VEC - 1)] = (
                    vh2[1].to(FP32))
                if lane == 0:
                    sSC[j, 0] = _sigmoid(mBr[row])
                    sSC[j, 1] = kq * (rk * rqs)

        cute.arch.sync_threads()

        # ---- P1b: fold the decay coefficient into sC -------------------------
        # c_s[k] = exp(g_start[k] - G_s[k]) * k_s[k].  G is a monotone cumsum of
        # g <= 0, so g_start - G_s <= 0 and the exp can never overflow.
        # THREADS == K, so thread tid owns exactly channel tid.
        grow = cute.make_fragment((VEC, HV4), FP32)
        kc = tid
        # guarded (not masked) so an empty ring never even touches old_G --
        # unused ring slots may legitimately hold garbage / NaN
        gs = FP32(0.0)
        if pn > 0:
            gs = mOgS[rg, kc, pm1]
        sD[kc >> LOG_VEC, kc & (VEC - 1)] = FP32(cute.math.exp(gs, fastmath=True))
        for grp in cutlass.range_constexpr(HV4):
            if grp * VEC < pn:
                cute.autovec_copy(mOgV[rg, kc, grp, None], grow[None, grp])
                for ii in cutlass.range_constexpr(VEC):
                    s = grp * VEC + ii
                    if s < pn:
                        sC[s, kc >> LOG_VEC, kc & (VEC - 1)] = (
                            sC[s, kc >> LOG_VEC, kc & (VEC - 1)]
                            * FP32(cute.math.exp(gs - grow[ii, grp],
                                                 fastmath=True)))

        # ---- P7: G cumsum + ring store.  g_start is already in this thread's
        # register (same channel), so no sGS round trip is needed.  The VSPLITS
        # CTAs split the token positions so each record is written once.
        acc = gs * FP32(1 - ovi)           # checkpoint absorbs g_start on fold
        for t in cutlass.range_constexpr(T):
            acc = acc + sGL[t, kc >> LOG_VEC, kc & (VEC - 1)]
            if cutlass.const_expr((t & (VSPLITS - 1)) == 0):
                if vsi == 0:
                    mOgS[rg_o, kc, pos0 + t] = acc
            else:
                if vsi != 0:
                    mOgS[rg_o, kc, pos0 + t] = acc

        cute.arch.sync_threads()

        # ---- P4: st *= exp(g_start)  (fold the checkpoint decay into the tile)
        dsf = cute.make_fragment((VEC,), FP32)
        for jj in cutlass.range_constexpr(NKG):
            cute.autovec_copy(sD[kg0 + jj * LK, None], dsf)
            for kk in cutlass.range_constexpr(VEC):
                dv = dsf[kk]
                for cwp in cutlass.range_constexpr(CPT // 2):
                    c0 = 2 * cwp
                    c1 = c0 + 1
                    st[kk, jj, c0], st[kk, jj, c1] = cute.arch.mul_packed_f32x2(
                        (st[kk, jj, c0], st[kk, jj, c1]), (dv, dv))

        # ---- P5: rank-pnat replay -> st == S_logical --------------------------
        cf = cute.make_fragment((VEC,), FP32)
        uf = cute.make_fragment((CPT,), FP32)
        for s in cutlass.range(pn, unroll=2):
            cute.autovec_copy(sUo[s, vg0 - (vbase >> LOG_VEC), None], uf)
            for jj in cutlass.range_constexpr(NKG):
                cute.autovec_copy(sC[s, kg0 + jj * LK, None], cf)
                for kk in cutlass.range_constexpr(VEC):
                    cv = cf[kk]
                    for cwp in cutlass.range_constexpr(CPT // 2):
                        c0 = 2 * cwp
                        c1 = c0 + 1
                        st[kk, jj, c0], st[kk, jj, c1] = (
                            cute.arch.fma_packed_f32x2(
                                (cv, cv), (uf[c0], uf[c1]),
                                (st[kk, jj, c0], st[kk, jj, c1])))

        # ---- P6: fold -- the ONLY S write, skipped on the common append path --
        if ovi != 0:
            for jj in cutlass.range_constexpr(NKG):
                for c in cutlass.range_constexpr(CPT):
                    cute.autovec_copy(st[None, jj, c],
                                      mS[bh, vr0 + c, kg0 + jj * LK, None])

        # ---- P8: T-step recurrence over the register-resident tile ------------
        # The whole K reduction is a 3-step lane butterfly inside the warp, so all
        # LK lanes end up holding the *same* full dot products.  Each lane can then
        # form u and o for its own CPT columns locally: no SMEM publish, no
        # barrier, and the T iterations of the 4 warps slide freely past each
        # other.  Only the lk == 0 lanes issue the (32 B contiguous) global stores.
        dg4 = cute.make_fragment((VEC,), FP32)
        knf = cute.make_fragment((VEC,), FP32)
        qnf = cute.make_fragment((VEC,), FP32)
        pk = cute.make_fragment((CPT,), FP32)
        pq = cute.make_fragment((CPT,), FP32)
        uu = cute.make_fragment((CPT,), FP32)
        vv = cute.make_fragment((CPT,), FP32)
        sc2 = cute.make_fragment((2,), FP32)
        ob = cute.make_fragment((CPT,), BF16)
        ub = cute.make_fragment((CPT,), BF16)

        for j in cutlass.range(T, unroll=1):
            for c in cutlass.range_constexpr(CPT):
                pk[c] = FP32(0.0)
                pq[c] = FP32(0.0)
            # pass 1: decay + BOTH K-partials in one sweep.  Blackwell packed
            # f32x2: every multiplier here is per-CHANNEL, hence invariant across
            # the column pair, so each op pairs perfectly (bit-identical to scalar).
            for jj in cutlass.range_constexpr(NKG):
                cute.autovec_copy(sDG[j, kg0 + jj * LK, None], dg4)
                cute.autovec_copy(sKN[j, kg0 + jj * LK, None], knf)
                cute.autovec_copy(sQN[j, kg0 + jj * LK, None], qnf)
                for kk in cutlass.range_constexpr(VEC):
                    dgv2 = dg4[kk]
                    knv = knf[kk]
                    qnv = qnf[kk]
                    for cwp in cutlass.range_constexpr(CPT // 2):
                        c0 = 2 * cwp
                        c1 = c0 + 1
                        s0, s1 = cute.arch.mul_packed_f32x2(
                            (st[kk, jj, c0], st[kk, jj, c1]), (dgv2, dgv2))
                        st[kk, jj, c0] = s0
                        st[kk, jj, c1] = s1
                        pk[c0], pk[c1] = cute.arch.fma_packed_f32x2(
                            (s0, s1), (knv, knv), (pk[c0], pk[c1]))
                        pq[c0], pq[c1] = cute.arch.fma_packed_f32x2(
                            (s0, s1), (qnv, qnv), (pq[c0], pq[c1]))
            # LK-way butterfly over the lk lanes (offsets LV, 2LV, 4LV)
            for off in cutlass.range_constexpr(LOG_LK):
                m = LV << off
                for c in cutlass.range_constexpr(CPT):
                    pk[c] = pk[c] + cute.arch.shuffle_sync_bfly(pk[c], m)
                    pq[c] = pq[c] + cute.arch.shuffle_sync_bfly(pq[c], m)
            cute.autovec_copy(sV[j, vg0 - (vbase >> LOG_VEC), None], vv)
            cute.autovec_copy(sSC[j, None], sc2)
            for c in cutlass.range_constexpr(CPT):
                uu[c] = sc2[0] * (vv[c] - pk[c])
            if lk == 0:
                for c in cutlass.range_constexpr(CPT):
                    ob[c] = (pq[c] + uu[c] * sc2[1]).to(BF16)
                    ub[c] = uu[c].to(BF16)
                cute.autovec_copy(ob, mO[(b * T + j) * H + h, vg0, None])
                cute.autovec_copy(ub, mOu4[ru_o + (pos0 + j) * H, vg0, None])
            # pass 2: rank-1 update
            for jj in cutlass.range_constexpr(NKG):
                # kn is re-read rather than carried in a (VEC, NKG) fragment:
                # holding it in fp32 would cost 16 registers and 128 regs/thread
                # is exactly the 4-blocks-per-SM cliff (65536 / (128*128) == 4).
                cute.autovec_copy(sKN[j, kg0 + jj * LK, None], knf)
                for kk in cutlass.range_constexpr(VEC):
                    knv = knf[kk]
                    for cwp in cutlass.range_constexpr(CPT // 2):
                        c0 = 2 * cwp
                        c1 = c0 + 1
                        st[kk, jj, c0], st[kk, jj, c1] = (
                            cute.arch.fma_packed_f32x2(
                                (knv, knv), (uu[c0], uu[c1]),
                                (st[kk, jj, c0], st[kk, jj, c1])))
        # S is deliberately NOT written here: the verify step must not commit.

    # Shared traced launch body; per-variant @cute.jit signatures below expose
    # ONLY the pointers that variant reads (unused tensor views are aliased
    # from live pointers of the same dtype -- the compiled kernel never reads
    # them, their loads are compiled out under const_expr).
    def _launch_body(pQ, pK, pV, pGr, pBr, pAl, pDb, pS, pO, pOu, pOk, pOg,
                     pPn, pBi, pX, pCS, pCWt, pCWin, lower_bound, nb):
        rows = nb * (T * H)
        nbh = nb * H
        nrg = nb * (2 * H)
        nru = nb * (2 * HIST * H)
        lay4 = cute.make_layout((rows, K // VEC, VEC), stride=(K, VEC, 1))
        mQ = cute.make_tensor(pQ, lay4)
        mKt = cute.make_tensor(pK, lay4)
        mGr = cute.make_tensor(pGr, lay4)
        mV2 = cute.make_tensor(pV, cute.make_layout(
            (rows, V // 2, 2), stride=(V, 2, 1)))
        mO = cute.make_tensor(pO, cute.make_layout(
            (rows, V // VEC, VEC), stride=(V, VEC, 1)))
        mBr = cute.make_tensor(pBr, cute.make_layout((rows,), stride=(1,)))
        mAl = cute.make_tensor(pAl, cute.make_layout((H,), stride=(1,)))
        mDb = cute.make_tensor(pDb, cute.make_layout(
            (H, K // VEC, VEC), stride=(K, VEC, 1)))
        mS = cute.make_tensor(pS, cute.make_layout(
            (nbh, V, K // VEC, VEC), stride=(V * K, K, VEC, 1)))
        mOu4 = cute.make_tensor(pOu, cute.make_layout(
            (nru, V // VEC, VEC), stride=(V, VEC, 1)))
        mOu8 = cute.make_tensor(pOu, cute.make_layout(
            (nru, V // VECH, VECH), stride=(V, VECH, 1)))
        mOk4 = cute.make_tensor(pOk, cute.make_layout(
            (nru, K // VEC, VEC), stride=(K, VEC, 1)))
        mOk8 = cute.make_tensor(pOk, cute.make_layout(
            (nru, K // VECH, VECH), stride=(K, VECH, 1)))
        mOgV = cute.make_tensor(pOg, cute.make_layout(
            (nrg, K, HIST // VEC, VEC), stride=(K * HIST, HIST, VEC, 1)))
        mOgS = cute.make_tensor(pOg, cute.make_layout(
            (nrg, K, HIST), stride=(K * HIST, HIST, 1)))
        mPn = cute.make_tensor(pPn, cute.make_layout((nb,), stride=(1,)))
        mBi = cute.make_tensor(pBi, cute.make_layout((nb,), stride=(1,)))
        CDIM = 3 * H * K
        ntok = nb * T
        mX = cute.make_tensor(pX, cute.make_layout(
            (ntok, CDIM // VEC, VEC), stride=(CDIM, VEC, 1)))
        mCS = cute.make_tensor(pCS, cute.make_layout(
            (nb, CDIM, CONV_W - 1),
            stride=(CDIM * (CONV_W - 1), CONV_W - 1, 1)))
        mCWt = cute.make_tensor(pCWt, cute.make_layout(
            (CDIM, CONV_W), stride=(CONV_W, 1)))
        # BLOCKED conv_win view: one lane's VEC channels x CSL slots are 12
        # contiguous bf16, written by _conv_fold as one vectorized copy
        mCWin = cute.make_tensor(pCWin, cute.make_layout(
            (ntok, CDIM // VEC, VEC * (CONV_W - 1)),
            stride=(CDIM * (CONV_W - 1), VEC * (CONV_W - 1), 1)))
        blackwell_bf16_kda_replay_ssm_fused_kernel(
            mQ, mKt, mV2, mGr, mBr, mAl, mDb, mS, mO, mOu4, mOu8, mOk4, mOk8,
            mOgV, mOgS, mPn, mBi, mX, mCS, mCWt, mCWin, lower_bound,
        ).launch(grid=[VSPLITS, H, nb], block=[THREADS, 1, 1])

    if fuse_conv:
        @cute.jit
        def launch(
            pX: cute.Pointer,
            pGr: cute.Pointer,
            pBr: cute.Pointer,
            pAl: cute.Pointer,
            pDb: cute.Pointer,
            pS: cute.Pointer,
            pO: cute.Pointer,
            pOu: cute.Pointer,
            pOk: cute.Pointer,
            pOg: cute.Pointer,
            pPn: cute.Pointer,
            pBi: cute.Pointer,
            pCS: cute.Pointer,
            pCWt: cute.Pointer,
            pCWin: cute.Pointer,
            lower_bound: FP32,
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pX, pX, pX, pGr, pBr, pAl, pDb, pS, pO, pOu, pOk,
                         pOg, pPn, pBi, pX, pCS, pCWt, pCWin, lower_bound, nb)
    else:
        @cute.jit
        def launch(
            pQ: cute.Pointer,
            pK: cute.Pointer,
            pV: cute.Pointer,
            pGr: cute.Pointer,
            pBr: cute.Pointer,
            pAl: cute.Pointer,
            pDb: cute.Pointer,
            pS: cute.Pointer,
            pO: cute.Pointer,
            pOu: cute.Pointer,
            pOk: cute.Pointer,
            pOg: cute.Pointer,
            pPn: cute.Pointer,
            pBi: cute.Pointer,
            lower_bound: FP32,
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pQ, pK, pV, pGr, pBr, pAl, pDb, pS, pO, pOu, pOk,
                         pOg, pPn, pBi, pQ, pQ, pQ, pQ, lower_bound, nb)

    return launch


def _make_onecta_launcher(H: int, K: int, V: int, T: int, gated: bool,
                          fuse_conv: bool = False):
    """The gated champion body: grid (H, B), ONE CTA of 8 warps per (b,h).

    One launch produces the (post-norm when ``gated``) `o` for the T = 1+gamma
    draft tokens of every request WITHOUT committing the recurrent state, and
    appends T per-token records to a small ring so a later launch can resume
    from exactly the accepted prefix.  The checkpoint state S is read every
    launch and written ONLY when the ring would overflow.  The epilogue

        o = (r * rsqrt(mean_v(r^2) + 1e-5)) * w * sigmoid(z)

    (r = the pre-norm recurrence output, mean over the 128 head channels) is
    applied in registers/SMEM before anything reaches HBM -- r never
    round-trips.  ``gated=False`` compiles the epilogue OUT and stores the raw
    pre-norm r (the "unfused + separate norm" ablation).

    Decomposition
    -------------
        grid  = (H, B)                   block = (256,) = 8 warps
        CTA (h,b)  owns the WHOLE S[b, h, :, :]        -> 64 fp32 regs / thread
        warp w     owns V columns [16w, 16w+16) and ALL 128 K channels
        lane = (lk, lv) = (lane>>2, lane&3)

    Because a warp owns every K channel of its columns, the per-token K
    reduction is a 3-step lane butterfly: the T-loop contains no barrier and no
    SMEM reduction traffic at all, and every lane derives u/r for its own
    columns locally. The phase ordering is documented inline below.
    """
    assert K == 128 and V == 128, "spec fixes K = V = 128"
    assert 1 <= T <= 8
    NWARP = 8                # one CTA owns the whole [K,V] tile
    THREADS = NWARP * 32     # 256
    # ---- state tiling, dispatched on T (measured crossover; see optimization.md)
    # Chunking halves the register tile and buys a 3rd CTA per SM (24 of 64 warps
    # instead of 16, long_scoreboard -26%), but costs a duplicated per-token
    # butterfly + sV/sSC read.  The occupancy gain is FIXED per launch while the
    # duplication scales with T, so it wins below the crossover and loses above.
    if T <= T_CHUNK_MAX:
        LK, LOG_LK, NCH, MBPM = 16, 4, 2, 3
    else:
        LK, LOG_LK, NCH, MBPM = 8, 3, 1, 0
    LV = 32 // LK                    # lanes over V
    LOG_LV = 5 - LOG_LK
    CPT = VEC                        # V columns per thread per chunk
    WVS = LV * CPT                   # V columns per warp per chunk
    VSC = NWARP * WVS                # V columns per chunk
    VS = V
    KPT = K // LK                    # key channels per thread
    NKG = KPT // VEC                 # K-groups of VEC per thread
    assert KPT * CPT * NCH == 64     # K*V/THREADS elements per thread, always
    assert NCH * VSC == V            # the chunks tile V exactly
    TPW = (T + NWARP - 1) // NWARP   # prologue / epilogue tokens per warp
    # at T <= NWARP/2 the prologue can run its two independent halves on two
    # disjoint warp groups instead of leaving half the CTA idle (see P2)
    assert TPW == 1                  # T <= NWARP, so one prologue token per warp
    WLO = T if 2 * T <= NWARP else 0
    RSQRT_K = 1.0 / math.sqrt(K)
    RCP_V = 1.0 / V                  # mean over V: a multiply, never a divide
    KV4 = K // VEC                   # 32 == warp size: one 128-bit group per lane
    assert KV4 == 32
    HV4 = HIST // VEC                # 4
    KV8 = K // VECH                  # 16 ring-k chunks of 8 bf16
    LOG_KV8 = 4
    VS8 = VS // VECH                 # 16 ring-u chunks of 8 bf16
    LOG_VS8 = 4
    NIT_K = (HIST * KV8 + THREADS - 1) // THREADS     # 1
    NIT_U = (HIST * VS8 + THREADS - 1) // THREADS     # 1
    ROWSG = HIST * H                 # rows per (slot, half) in the u/k rings
    # THREADS == 2*K lets P1b split the HIST groups over two thread halves.
    assert THREADS == 2 * K
    assert HV4 == 4
    LOG_K = 7
    # widest aligned vector for the ring-G cumsum store (HIST is contiguous)
    GW = 4 if T % 4 == 0 else (2 if T % 2 == 0 else 1)
    NGT = T // GW
    KD = H * K                       # packed-qkv channel-block size (fuse_conv)

    @cute.kernel
    def blackwell_bf16_kda_replay_ssm_fused_gated_kernel(
        mQ: cute.Tensor,      # [B*T*H, K/4, 4]       bf16
        mKt: cute.Tensor,     # [B*T*H, K/4, 4]       bf16
        mV4: cute.Tensor,     # [B*T*H, V/4, 4]       bf16
        mGr: cute.Tensor,     # [B*T*H, K/4, 4]       fp32  raw gate
        mBr: cute.Tensor,     # [B*T*H]               fp32  raw beta
        mAl: cute.Tensor,     # [H]                   fp32  A_log
        mDb: cute.Tensor,     # [H, K/4, 4]           fp32  dt_bias
        mS: cute.Tensor,      # [B*H, V, K/4, 4]      fp32  checkpoint (in / cond out)
        mZ: cute.Tensor,      # [B*T*H, V/4, 4]       bf16  output gate
        mW: cute.Tensor,      # [V/4, 4]              fp32  RMSNorm weight
        mO: cute.Tensor,      # [B*T*H, V/4, 4]       bf16
        mOu4: cute.Tensor,    # [B*2*HIST*H, V/4, 4]  bf16  ring u (store view)
        mOu8: cute.Tensor,    # [B*2*HIST*H, V/8, 8]  bf16  ring u (load view)
        mOk4: cute.Tensor,    # [B*2*HIST*H, K/4, 4]  bf16  ring k (store view)
        mOk8: cute.Tensor,    # [B*2*HIST*H, K/8, 8]  bf16  ring k (load view)
        mOgV: cute.Tensor,    # [B*2*H, K, HIST/4, 4]   fp32  ring G (read view)
        mOgW: cute.Tensor,    # [B*2*H, K, HIST/GW, GW] fp32  ring G (store view)
        mOgS: cute.Tensor,    # [B*2*H, K, HIST]        fp32  ring G (scalar view)
        mPn: cute.Tensor,     # [B]                   int32 pnat
        mBi: cute.Tensor,     # [B]                   int32 buf_idx
        mX: cute.Tensor,      # [B*T, 3*H*K]          bf16  raw packed qkv (conv)
        mCS: cute.Tensor,     # [B, 3*H*K, 3]         bf16  conv history (RO)
        mCWt: cute.Tensor,    # [3*H*K, 4]            bf16  conv taps
        mCWin: cute.Tensor,   # [B*T, 3*H*K, 3]       bf16  window snapshots (out)
        lower_bound: FP32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        h, b, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        w = cute.arch.warp_idx()
        lv = lane & (LV - 1)
        lk = lane >> LOG_LV

        bh = b * H + h
        # Warp w owns V columns [w*WVS, +WVS) and ALL K channels of them, so the
        # per-token K-reduction is a 3-step lane butterfly -- no SMEM, no barrier.
        # lk indexes the K partials (8 lanes), lv the V sub-group.
        kg0 = lk                             # + jj*LK, in VEC units
        vrb = w * WVS + lv * VEC             # + ch*VSC + cw

        # ---- P0: issue CHUNK 0's state loads first; their HBM latency overlaps
        # P1a/P2 and both barriers.  One instruction = LV rows x (LK*16 B) =
        # 2 rows x 256 B, still 4 fully-used 128 B lines -- the same wavefront
        # count per instruction as the LK=8 map it replaces, and the same 16
        # instructions per thread in total across the two chunks.
        st = cute.make_fragment((VEC, NKG, CPT), FP32)
        for jj in cutlass.range_constexpr(NKG):
            for c in cutlass.range_constexpr(CPT):
                cute.autovec_copy(mS[bh, vrb + c, kg0 + jj * LK, None],
                                  st[None, jj, c])

        # ---- shared memory (explicit row-major strides -- see module docstring)
        smem = SmemAllocator()
        sD = smem.allocate_tensor(FP32, cute.make_layout((KV4, VEC),
                                                         stride=(VEC, 1)), 16)
        sC = smem.allocate_tensor(FP32, cute.make_layout(
            (HIST, KV4, VEC), stride=(K, VEC, 1)), 16)
        sUo = smem.allocate_tensor(FP32, cute.make_layout(
            (HIST, VS // VEC, VEC), stride=(VS, VEC, 1)), 16)
        # sA holds the scaled-recurrence pass-1 k-coefficient exp(Gr_t) * kn_t,
        # built by P1c.  It occupies the buffer that used to hold exp(g_t) -- that
        # plane no longer exists, so this costs nothing.
        sA = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sGL = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        # kn/qn are staged in **fp32**, not fp16.  fp16 halved these two planes'
        # SMEM words but put an HADD2_F32 widening on the LDS -> FFMA2 critical
        # path of every channel of every token: 128 HADD2 per warp, 6.8% of all
        # instructions, for a stream that measurement showed is not
        # bandwidth-limited (l1tex 67% of peak-active, FMA 28%).  fp32 removes the
        # conversions, shortens the dependency chain, and is strictly more
        # accurate.  The *persisted* ring copy of kn is still bf16 of this same
        # fp32 value, written straight out of the prologue registers.
        sKN = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sQN = smem.allocate_tensor(FP32, cute.make_layout(
            (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        sV = smem.allocate_tensor(FP32, cute.make_layout(
            (T, VS // VEC, VEC), stride=(VS, VEC, 1)), 16)
        sSC = smem.allocate_tensor(FP32, cute.make_layout((T, 2),
                                                          stride=(2, 1)), 16)
        # pre-norm r, staged fp32 so the RMS reduction and the rescale both see
        # full precision (bf16 here would put a 2^-9 error *inside* the norm on
        # top of the final bf16 store, against a 5e-3 tolerance).
        sR = smem.allocate_tensor(FP32, cute.make_layout(
            (T, VS // VEC, VEC), stride=(VS, VEC, 1)), 16)
        # the epilogue's whole per-element scale, w[v] * sigmoid(z[t,v]), built in
        # the PROLOGUE.  z is a global load that the epilogue consumes right after
        # a CTA barrier, so issuing it there exposes ~600 cycles of HBM latency
        # with nothing to hide it behind -- measured +8.4 us at B=256.  P2 already
        # walks the identical (warp -> token, lane -> 4 consecutive channels) map,
        # so the load is free there and the epilogue becomes one LDS.128.
        if cutlass.const_expr(gated):
            sZG = smem.allocate_tensor(FP32, cute.make_layout(
                (T, KV4, VEC), stride=(K, VEC, 1)), 16)
        # ---- fused-conv planes, staged ONCE per CTA (conv builds only) ------
        # sCSs: conv_state TRANSPOSED to [CSL][3K] bf16 (2304 B) so a lane
        # reads its 4 channels of a dynamic slot with one 64-bit LDS; the
        # global [B, D, 3] layout is channel-strided (6 L1TEX wavefronts per
        # scalar tap read, measured ~90% of the conv's cost in the gen A/B).
        # sCWs: the taps, TAP-MAJOR fp32 [CONV_W][3K] (6144 B) -- one
        # conflict-free LDS.128 per (tensor, tap), converted bf16->fp32 once
        # per CTA instead of by every warp of the prologue group.
        if cutlass.const_expr(fuse_conv):
            sCSs = smem.allocate_tensor(BF16, cute.make_layout(
                (CSL, NTI * K // VEC, VEC), stride=(NTI * K, VEC, 1)), 16)
            sCWs = smem.allocate_tensor(FP32, cute.make_layout(
                (CONV_W, NTI * K // VEC, VEC), stride=(NTI * K, VEC, 1)), 16)
            # post-SiLU conv outputs for all T tokens x 3K local channels,
            # built by the whole CTA in PC below.  P2 then reads q/k/v out of
            # THIS plane with the flag-off code's own LDS.64, so the conv adds
            # ZERO register pressure to the prologue warps: folding the conv
            # into P2 in-place pushed the min_blocks_per_mp=3 (<= 85 register)
            # build over its cap and ptxas spilled ~250K local sectors.
            sXC = smem.allocate_tensor(BF16, cute.make_layout(
                (T, NTI * K // VEC, VEC), stride=(NTI * K, VEC, 1)), 16)
            # raw window snapshots staged in conv_win's own memory order
            # ([channel][slot], channel blocks of VEC*CSL), then streamed out
            # AFTER the PC barrier as 16 B chunks.  Written straight to global
            # out of the fold, a lane's 24 B window only autovectorizes to
            # scalar stores (24 B stride proves 8 B alignment at best) and
            # every 32 B sector of the plane is touched up to 12x: ncu
            # measured 1.44M global store sectors vs 98K flag-off, ~25% of
            # kernel time on an output the gen package does not even produce.
            sCW2 = smem.allocate_tensor(BF16, cute.make_layout(
                (T, NTI * K // VEC, VEC * CSL),
                stride=(NTI * K * CSL, VEC * CSL, 1)), 16)

        # ---- ring bookkeeping (all branch-free; every value is CTA-uniform) ---
        pn = mPn[b]
        half = mBi[b] & 1
        # overflow iff pn + T > HIST; pn in [0,16], T in [1,8] -> pn+T-1 in [0,23]
        ovi = (pn + T - 1) >> LOG_HIST                 # 0 or 1
        half_out = half ^ ovi
        pos0 = pn - pn * ovi                           # pn on append, 0 on fold
        rg = (b * 2 + half) * H + h                    # ring G row (read half)
        rg_o = (b * 2 + half_out) * H + h              # ring G row (write half)
        ru = (b * 2 + half) * ROWSG + h                # ring u/k base (read half)
        ru_o = (b * 2 + half_out) * ROWSG + h          # ring u/k base (write half)
        pm1 = cutlass.max(pn - 1, I32(0))

        # g_start for this thread's channel, issued HERE rather than in P1b.  It
        # is a 4 B load at a 64 B stride -> 16 L1TEX wavefronts per warp, and in
        # P1b it sat directly on the critical path between two CTA barriers.
        # Hoisted above P1a/P2 it costs one register and its latency is free.
        # guarded (not masked) so an empty ring never even touches old_G --
        # unused ring slots may legitimately hold garbage / NaN.
        kc = tid
        gs = FP32(0.0)
        if tid < K:
            if pn > 0:
                gs = mOgS[rg, kc, pm1]

        # ---- PS: stage conv_state (transposed) + fp32 taps, then barrier 0 ---
        # One thread per local channel: its CSL history taps are CSL contiguous
        # bf16 (2 wavefronts per warp instead of 6 per scalar tap read) and its
        # CONV_W taps are one 64-bit read.  Landing fragments are written
        # UNCONDITIONALLY (index clamped, only the SMEM store guarded): a
        # fragment defined inside an scf.if is demoted to local memory.  The
        # extra barrier gates a shorter critical path than the scattered
        # per-token conv_state read it replaces (the gen A/B measured the
        # barrier-stall going DOWN); flag-off builds compile none of this.
        if cutlass.const_expr(fuse_conv):
            NCC = (NTI * K + THREADS - 1) // THREADS
            csv = cute.make_fragment((CSL, NCC), BF16)
            cwv = cute.make_fragment((CONV_W, NCC), BF16)
            for it in cutlass.range_constexpr(NCC):
                lcc = cutlass.min(tid + it * THREADS, I32(NTI * K - 1))
                gcc = (lcc >> LOG_K) * KD + h * K + (lcc & (K - 1))
                for m_ in cutlass.range_constexpr(CSL):
                    csv[m_, it] = mCS[b, gcc, m_]
                cute.autovec_copy(mCWt[gcc, None], cwv[None, it])
            for it in cutlass.range_constexpr(NCC):
                lcs = tid + it * THREADS
                if lcs < NTI * K:
                    for m_ in cutlass.range_constexpr(CSL):
                        sCSs[m_, lcs >> LOG_VEC, lcs & (VEC - 1)] = csv[m_, it]
                    for jj in cutlass.range_constexpr(CONV_W):
                        sCWs[jj, lcs >> LOG_VEC, lcs & (VEC - 1)] = (
                            cwv[jj, it].to(FP32))
            cute.arch.sync_threads()

            # ---- PC: whole-CTA conv, T tokens x 3K channels -> sXC + conv_win
            # Every thread folds VEC channels of one (token, tensor) group per
            # iteration: 8 warps instead of the 2T prologue warps, and the
            # scratch (xfp/csfp/wfjp/obf) lives in a phase where the P2
            # working set is not yet live, so nothing spills under the
            # register cap.  Indices are CLAMPED, never guarded: the trailing
            # threads of an iteration redo group NTI*KV4-1 of their token and
            # write the same bytes again.
            xfp = cute.make_fragment((VEC, CONV_W), BF16)
            csfp = cute.make_fragment((VEC,), BF16)
            wfjp = cute.make_fragment((VEC,), FP32)
            obf = cute.make_fragment((VEC,), BF16)
            wpc = cute.make_fragment((VEC * CSL,), BF16)
            NPC = (T * 4 * KV4 + THREADS - 1) // THREADS
            for it in cutlass.range_constexpr(NPC):
                gpc = cutlass.min(tid + it * THREADS, I32(T * 4 * KV4 - 1))
                tkc = gpc >> 7                       # 4*KV4 == 128 slots/token
                lgc = cutlass.min(gpc & (4 * KV4 - 1), I32(NTI * KV4 - 1))
                tsr = lgc >> 5                       # KV4 == 32 groups/tensor
                gof = lgc & (KV4 - 1)
                cgc = tsr * (KD >> LOG_VEC) + h * KV4 + gof
                _conv_gather(mX, b * T, tkc, cgc, xfp)
                _conv_fold(sCSs, sCWs, lgc, tkc, xfp, csfp, wfjp, obf,
                           wpc, sCW2, tkc, lgc, I32(1))
                cute.autovec_copy(obf, sXC[tkc, lgc, None])
            cute.arch.sync_threads()

        # ---- P1a: stage the ring records, 8 elements per lane ----------------
        # old_k is contiguous in K and old_u in V, so both are read as 128-bit
        # chunks of 8 bf16 and widened into the fp32 staging buffers.  sC holds
        # the raw k here; P1b multiplies the decay coefficient in place.
        rh8 = cute.make_fragment((VECH,), BF16)
        rf8 = cute.make_fragment((VEC, 2), FP32)
        for it in cutlass.range_constexpr(NIT_K):
            i = tid + it * THREADS
            s = i >> LOG_KV8
            ch = i & (KV8 - 1)
            if s < pn:
                cute.autovec_copy(mOk8[ru + s * H, ch, None], rh8)
                for e in cutlass.range_constexpr(VECH):
                    rf8[e & (VEC - 1), e >> LOG_VEC] = rh8[e].to(FP32)
                for g2 in cutlass.range_constexpr(2):
                    cute.autovec_copy(rf8[None, g2],
                                      sC[s, ch * (VECH // VEC) + g2, None])
        for it in cutlass.range_constexpr(NIT_U):
            i = tid + it * THREADS
            s = i >> LOG_VS8
            ch = i & (VS8 - 1)
            if s < pn:
                cute.autovec_copy(mOu8[ru + s * H, ch, None], rh8)
                for e in cutlass.range_constexpr(VECH):
                    rf8[e & (VEC - 1), e >> LOG_VEC] = rh8[e].to(FP32)
                for g2 in cutlass.range_constexpr(2):
                    cute.autovec_copy(rf8[None, g2],
                                      sUo[s, ch * (VECH // VEC) + g2, None])

        # ---- P2: new-token prologue (lane owns K channels 4*lane .. 4*lane+3) -
        # KV4 == 32 == warp size, so one 128-bit group per lane covers all of K
        # with a single instruction per tensor per token.  With NWARP == 8 and
        # T <= 8 this is at most one token per warp -- which means that for the
        # production T <= 4 shapes HALF the CTA was idle here.  When 2T <= NWARP
        # the phase is therefore split into two warp groups working the same
        # tokens concurrently:
        #   warps [0, T)   : q/k/g_raw -> gate, both L2 norms, <kn,qn>, ring k
        #   warps [T, 2T)  : v -> sV, z -> sZG (w * sigmoid(z)), beta -> sSC[.,0]
        # The second group's work is exactly the part with no cross-lane
        # reduction in it, so it needs nothing from the first group and the
        # prologue's critical path drops to the norm chain alone.
        alpha = FP32(cute.math.exp(mAl[h], fastmath=True))
        dtb = cute.make_fragment((VEC,), FP32)
        cute.autovec_copy(mDb[h, lane, None], dtb)     # head constant, hoisted
        qh = cute.make_fragment((VEC,), BF16)
        kh = cute.make_fragment((VEC,), BF16)
        gr = cute.make_fragment((VEC,), FP32)
        gv = cute.make_fragment((VEC,), FP32)
        knw = cute.make_fragment((VEC,), FP32)
        qnw = cute.make_fragment((VEC,), FP32)
        knb = cute.make_fragment((VEC,), BF16)
        vh4 = cute.make_fragment((VEC,), BF16)
        vf4 = cute.make_fragment((VEC,), FP32)
        zh = cute.make_fragment((VEC,), BF16)
        zg = cute.make_fragment((VEC,), FP32)
        wf = cute.make_fragment((VEC,), FP32)
        if cutlass.const_expr(gated):
            cute.autovec_copy(mW[lane, None], wf)      # head-independent, hoisted
        # group 1 -- q/k/g_raw -> gate, L2 norms, <kn,qn>, ring-k store.  This is
        # the half with the cross-lane reduction in it, hence the critical path.
        if w < T:
            j = w
            row = (b * T + j) * H + h
            if cutlass.const_expr(fuse_conv):
                # conv already folded by PC into sXC: q/k are one LDS.64 each,
                # exactly the flag-off code's register profile
                cute.autovec_copy(sXC[j, lane, None], qh)
                cute.autovec_copy(sXC[j, KV4 + lane, None], kh)
            else:
                cute.autovec_copy(mQ[row, lane, None], qh)
                cute.autovec_copy(mKt[row, lane, None], kh)
            cute.autovec_copy(mGr[row, lane, None], gr)
            nq = FP32(0.0)
            nk = FP32(0.0)
            kq = FP32(0.0)
            for e in cutlass.range_constexpr(VEC):
                qe = qh[e].to(FP32)
                ke = kh[e].to(FP32)
                nq = nq + qe * qe
                nk = nk + ke * ke
                kq = kq + qe * ke
                gv[e] = lower_bound * _sigmoid(alpha * (gr[e] + dtb[e]))
            # one warp owns all K channels of a token -> pure warp reduction
            for off in cutlass.range_constexpr(5):
                nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
            rk = FP32(cute.math.rsqrt(nk + FP32(K_EPS), fastmath=True))
            rqs = FP32(cute.math.rsqrt(nq, fastmath=True)) * FP32(RSQRT_K)
            for e in cutlass.range_constexpr(VEC):
                kv = kh[e].to(FP32) * rk
                knw[e] = kv
                knb[e] = kv.to(BF16)
                qnw[e] = qh[e].to(FP32) * rqs
            # NOTE: no exp(g_t) plane is produced here any more.  The scaled
            # recurrence (P1c) needs exp(cumsum(g)_t), not exp(g_t), so this
            # phase -- which is on barrier 1's critical path -- is 4 `exp` and
            # one STS.128 per token lighter than before.
            cute.autovec_copy(gv, sGL[j, lane, None])
            cute.autovec_copy(knw, sKN[j, lane, None])
            cute.autovec_copy(qnw, sQN[j, lane, None])
            # the persisted ring k: bf16 of the fp32 normalised value
            cute.autovec_copy(knb, mOk4[ru_o + (pos0 + j) * H, lane, None])
            if lane == 0:
                sSC[j, 1] = kq * (rk * rqs)

        # group 2 -- v/z/beta: pure elementwise, no cross-lane dependency, so it
        # needs nothing from group 1 and can run on the warps that group 1 leaves
        # idle.  WLO is 0 (same warps, sequentially) when 2T > NWARP.
        if w >= WLO:
            if w < WLO + T:
                jv = w - WLO
                rowv = (b * T + jv) * H + h
                if cutlass.const_expr(fuse_conv):
                    # conv already folded by PC into sXC
                    cute.autovec_copy(sXC[jv, 2 * KV4 + lane, None], vh4)
                else:
                    cute.autovec_copy(mV4[rowv, lane, None], vh4)
                if cutlass.const_expr(gated):
                    cute.autovec_copy(mZ[rowv, lane, None], zh)
                for e in cutlass.range_constexpr(VEC):
                    vf4[e] = vh4[e].to(FP32)
                cute.autovec_copy(vf4, sV[jv, lane, None])
                if cutlass.const_expr(gated):
                    for e in cutlass.range_constexpr(VEC):
                        zg[e] = wf[e] * _sigmoid(zh[e].to(FP32))
                    cute.autovec_copy(zg, sZG[jv, lane, None])
                if lane == 0:
                    sSC[jv, 0] = _sigmoid(mBr[rowv])

        cute.arch.sync_threads()

        # ---- P1b: fold the decay coefficient into sC -------------------------
        # c_s[k] = exp(g_start[k] - G_s[k]) * k_s[k].  G is a monotone cumsum of
        # g <= 0, so g_start - G_s <= 0 and the exp can never overflow.
        # Thread tid owns exactly channel tid (only the first K threads work): the
        # obvious alternative -- splitting the HIST groups over both halves of the
        # CTA -- was measured a loss, because it doubles the scattered g_start
        # load (16 wavefronts/warp) while saving at most one group of ALU, and at
        # the steady-state pnat = 8 both live groups land on the same half anyway.
        grow = cute.make_fragment((VEC, HV4), FP32)
        if tid < K:
            # every live old_G group load is ISSUED before any of them is used:
            # written as load-then-use inside one loop, group g+1's load is only
            # reached after group g's exp chain, so the two ~600-cycle HBM
            # latencies serialize -- and this sits between two CTA barriers where
            # the whole CTA pays.  `grow` is sized (VEC, HV4) either way, so
            # hoisting the loads costs no extra registers.
            for grp in cutlass.range_constexpr(HV4):
                if grp * VEC < pn:
                    cute.autovec_copy(mOgV[rg, kc, grp, None], grow[None, grp])
            sD[kc >> LOG_VEC, kc & (VEC - 1)] = FP32(
                cute.math.exp(gs, fastmath=True))
            for grp in cutlass.range_constexpr(HV4):
                if grp * VEC < pn:
                    for ii in cutlass.range_constexpr(VEC):
                        s = grp * VEC + ii
                        if s < pn:
                            sC[s, kc >> LOG_VEC, kc & (VEC - 1)] = (
                                sC[s, kc >> LOG_VEC, kc & (VEC - 1)]
                                * FP32(cute.math.exp(gs - grow[ii, grp],
                                                     fastmath=True)))

        # ---- P1c: build the scaled-recurrence coefficient planes -------------
        # The T-loop below never multiplies the state tile by exp(g_t).  Instead
        # the register tile holds  hat_t = diag(exp(-Gr_t)) B_t  with
        # Gr_t = cumsum(g)_t, and the decay is folded into three per-channel
        # planes (design.md sec 4b):
        #     a_t = exp( Gr_t) * kn_t     (pass-1 k-partial)
        #     b_t = exp( Gr_t) * qn_t     (pass-1 q-partial)
        #     c_t = exp(-Gr_t) * kn_t     (pass-2 rank-1 update)
        # That is 3 packed ops per state element per token instead of 4, and one
        # fewer LDS per K-group per token.
        #
        # This runs on the OTHER half of the CTA, concurrently with P1b, and that
        # placement is the whole reason it can pay off here.  The ancestor built
        # these planes on the SAME 128 threads that do P1b, serially between the
        # two barriers, and measured +4.3% at T=2.  The plane build depends only
        # on sGL/sKN/sQN -- it needs NOTHING from g_start -- so unlike the
        # reverted P1b group split (attempt_000 cycle 4) there is no shared value
        # to pay for, and the barrier-1..barrier-2 window costs
        # max(P1b, P1c) instead of their sum.
        #
        # a/b/c are written into the buffers of the planes they replace: sA reuses
        # the dead exp(g_t) buffer, and sB/sCt overwrite sQN/sKN IN PLACE (each
        # thread owns one channel exclusively, and P1b never touches them), so the
        # reformulation costs zero extra SMEM.
        if tid >= K:
            kcp = tid - K
            acc0 = FP32(0.0)
            for t in cutlass.range_constexpr(T):
                acc0 = acc0 + sGL[t, kcp >> LOG_VEC, kcp & (VEC - 1)]
                eg = FP32(cute.math.exp(acc0, fastmath=True))     # <= 1
                ieg = cute.arch.rcp_approx(eg)
                knv0 = sKN[t, kcp >> LOG_VEC, kcp & (VEC - 1)]
                qnv0 = sQN[t, kcp >> LOG_VEC, kcp & (VEC - 1)]
                sA[t, kcp >> LOG_VEC, kcp & (VEC - 1)] = eg * knv0
                sQN[t, kcp >> LOG_VEC, kcp & (VEC - 1)] = eg * qnv0
                sKN[t, kcp >> LOG_VEC, kcp & (VEC - 1)] = ieg * knv0

        cute.arch.sync_threads()

        # ---- PW: stream the staged conv windows to conv_win -----------------
        # sCW2 already sits in conv_win's memory order, so per (token, tensor)
        # the destination is one 768 B contiguous block; both sides are recast
        # to Int32 x4 rows so each move is one LDS.128 / STG.128 with
        # consecutive lanes on consecutive addresses.  autovec_copy on the raw
        # bf16 views scalarized to 2 B stores (the 24 B fold rows only prove
        # element alignment; ncu measured 8x the ideal store sectors) -- note
        # conv_win's pointer spec must declare 16 B alignment or the Int32
        # recast legally degrades to a 0-wide vector.  Placed AFTER barrier 2
        # for the same reason as P7 below: its only consumer is HBM, nothing
        # in the CTA reads it back, so between the PC and PS barriers it sat
        # on every warp's wait path (the two largest bench shapes measured
        # ~7% slower) while here it interleaves with P4/P5 compute.
        if cutlass.const_expr(fuse_conv):
            CQ = K * CSL // 8               # 16 B chunks per (token, tensor)
            sCW2F = cute.make_tensor(
                cute.recast_ptr(sCW2.iterator, dtype=I32),
                cute.make_layout((T * NTI * CQ, 4), stride=(4, 1)))
            # shape is only metadata here (indices are always in range); a
            # static bound avoids dragging the dynamic ntok into the layout
            mCWinV = cute.make_tensor(
                cute.recast_ptr(mCWin.iterator, dtype=I32),
                cute.make_layout((1 << 28, 4), stride=(4, 1)))
            wq4 = cute.make_fragment((4,), I32)
            NWC = (T * NTI * CQ + THREADS - 1) // THREADS
            for it in cutlass.range_constexpr(NWC):
                cidx = cutlass.min(tid + it * THREADS, I32(T * NTI * CQ - 1))
                wtk = cidx // (NTI * CQ)
                wrr = cidx - wtk * (NTI * CQ)
                wti = wrr // CQ
                wof = wrr - wti * CQ
                wgc = ((b * T + wtk) * (NTI * KD * CSL // 8)
                       + wti * (KD * CSL // 8) + h * CQ + wof)
                cute.autovec_copy(sCW2F[cidx, None], wq4)
                cute.autovec_copy(wq4, mCWinV[wgc, None])

        # ---- P7: G cumsum + ring store.  g_start is already in this thread's
        # register (same channel), so no sGS round trip is needed.  HIST is the
        # contiguous mode of old_G, so the T records of one channel are adjacent:
        # the cumsum is accumulated into a fragment first and leaves as GW-wide
        # vector stores whenever pos0 is GW-aligned (GW = 4 -> 4x fewer L1TEX
        # wavefronts than the scalar store this replaced).
        # Placed AFTER the barrier on purpose: its only consumer is HBM, nothing
        # later in the CTA reads it, so keeping it in front of the barrier put a
        # 4-warp serial cumsum plus a strided store on all 8 warps' wait path.
        # Here it interleaves with P4/P5 instead.
        if tid < K:
            acc = gs * FP32(1 - ovi)       # checkpoint absorbs g_start on fold
            gac = cute.make_fragment((GW, NGT), FP32)      # LayoutLeft: GW dense
            for tg in cutlass.range_constexpr(NGT):
                for ti in cutlass.range_constexpr(GW):
                    acc = acc + sGL[tg * GW + ti, kc >> LOG_VEC,
                                    kc & (VEC - 1)]
                    gac[ti, tg] = acc
            if cutlass.const_expr(GW > 1):
                if (pos0 & (GW - 1)) == 0:
                    for tg in cutlass.range_constexpr(NGT):
                        cute.autovec_copy(
                            gac[None, tg],
                            mOgW[rg_o, kc, (pos0 // GW) + tg, None])
                else:
                    for tg in cutlass.range_constexpr(NGT):
                        for ti in cutlass.range_constexpr(GW):
                            mOgS[rg_o, kc, pos0 + tg * GW + ti] = gac[ti, tg]
            else:
                for tg in cutlass.range_constexpr(NGT):
                    mOgS[rg_o, kc, pos0 + tg] = gac[0, tg]

        dsf = cute.make_fragment((VEC,), FP32)
        cf = cute.make_fragment((VEC,), FP32)
        uf = cute.make_fragment((CPT,), FP32)
        dg4 = cute.make_fragment((VEC,), FP32)
        knf = cute.make_fragment((VEC,), FP32)
        qnf = cute.make_fragment((VEC,), FP32)
        pk = cute.make_fragment((CPT,), FP32)
        pq = cute.make_fragment((CPT,), FP32)
        uu = cute.make_fragment((CPT,), FP32)
        vv = cute.make_fragment((CPT,), FP32)
        sc2 = cute.make_fragment((2,), FP32)
        rf = cute.make_fragment((CPT,), FP32)
        ub = cute.make_fragment((CPT,), BF16)

        # ================= COLUMN CHUNKS ==================================
        # The state tile is processed as NCH disjoint 64-column chunks.  Columns
        # are fully independent through P4/P5/P6/P8 (u_t[v] and r_t[v] depend only
        # on column v of the tile; only the RMSNorm needs all 128, and that is
        # already behind the final barrier), so this is exact.
        #
        # The point is REGISTERS: with LK=16 the tile is KPT*CPT = 8*4 = 32 fp32
        # per thread instead of 64, which is what lets 3 CTAs fit per SM
        # (24 of 64 warps instead of 16) -- the only occupancy step available on a
        # kernel whose top stall is long_scoreboard.  Chunk 0's loads were issued
        # at P0 and are hidden by the prologue; chunk 1's are issued here, and the
        # extra warps are what hides them.
        #
        # Unlike the ancestor's rejected (K-group, record) nest inversion, chunking
        # by COLUMN re-reads nothing redundantly per chunk that is column-indexed:
        # each chunk reads its OWN disjoint u/v/r columns.  Only the K-indexed
        # planes are re-read, and NKG halved from 4 to 2, so pass-1/pass-2 plane
        # LDS per token is IDENTICAL to before (2 chunks x 2 K-groups == 4).
        for ch in cutlass.range_constexpr(NCH):
            vrc = vrb + ch * VSC
            vgc = vrc >> LOG_VEC
            if cutlass.const_expr(ch > 0):
                # chunk 0's tile is dead now, so these land in the same registers
                for jj in cutlass.range_constexpr(NKG):
                    for c in cutlass.range_constexpr(CPT):
                        cute.autovec_copy(mS[bh, vrc + c, kg0 + jj * LK, None],
                                          st[None, jj, c])
            # ---- P4: st *= exp(g_start)  (fold the checkpoint decay into the tile)
            for jj in cutlass.range_constexpr(NKG):
                cute.autovec_copy(sD[kg0 + jj * LK, None], dsf)
                for kk in cutlass.range_constexpr(VEC):
                    dv = dsf[kk]
                    for cwp in cutlass.range_constexpr(CPT // 2):
                        c0 = 2 * cwp
                        c1 = c0 + 1
                        st[kk, jj, c0], st[kk, jj, c1] = cute.arch.mul_packed_f32x2(
                            (st[kk, jj, c0], st[kk, jj, c1]), (dv, dv))

            # ---- P5: rank-pnat replay -> st == S_logical --------------------------
            for s in cutlass.range(pn, unroll=2):
                cute.autovec_copy(sUo[s, vgc, None], uf)
                for jj in cutlass.range_constexpr(NKG):
                    cute.autovec_copy(sC[s, kg0 + jj * LK, None], cf)
                    for kk in cutlass.range_constexpr(VEC):
                        cv = cf[kk]
                        for cwp in cutlass.range_constexpr(CPT // 2):
                            c0 = 2 * cwp
                            c1 = c0 + 1
                            st[kk, jj, c0], st[kk, jj, c1] = (
                                cute.arch.fma_packed_f32x2(
                                    (cv, cv), (uf[c0], uf[c1]),
                                    (st[kk, jj, c0], st[kk, jj, c1])))

            # ---- P6: fold -- the ONLY S write, skipped on the common append path --
            if ovi != 0:
                for jj in cutlass.range_constexpr(NKG):
                    for c in cutlass.range_constexpr(CPT):
                        cute.autovec_copy(st[None, jj, c],
                                          mS[bh, vrc + c, kg0 + jj * LK, None])

            # ---- P8: T-step recurrence over the register-resident tile ------------
            # The whole K reduction is a 3-step lane butterfly inside the warp, so all
            # LK lanes end up holding the *same* full dot products.  Each lane can then
            # form u and r for its own columns locally: no SMEM publish, no
            # barrier, and the T iterations of the 8 warps slide freely past each
            # other.  Only the lk == 0 lanes publish (32 B contiguous each).

            for j in cutlass.range(T, unroll=1):
                for c in cutlass.range_constexpr(CPT):
                    pk[c] = FP32(0.0)
                    pq[c] = FP32(0.0)
                # pass 1: BOTH K-partials against the tile as-is.  There is NO decay
                # multiply -- the tile holds hat_t and the decay lives in the a/b
                # coefficients P1c built, so this sweep is 2 packed ops per state
                # element instead of 3, and 2 LDS per K-group instead of 3.
                # Blackwell packed f32x2: every multiplier here is per-CHANNEL, hence
                # invariant across the column pair, so each op pairs perfectly
                # (bit-identical to the scalar form).
                for jj in cutlass.range_constexpr(NKG):
                    cute.autovec_copy(sA[j, kg0 + jj * LK, None], dg4)
                    cute.autovec_copy(sQN[j, kg0 + jj * LK, None], qnf)
                    for kk in cutlass.range_constexpr(VEC):
                        av = dg4[kk]
                        bv = qnf[kk]
                        for cwp in cutlass.range_constexpr(CPT // 2):
                            c0 = 2 * cwp
                            c1 = c0 + 1
                            s0 = st[kk, jj, c0]
                            s1 = st[kk, jj, c1]
                            pk[c0], pk[c1] = cute.arch.fma_packed_f32x2(
                                (s0, s1), (av, av), (pk[c0], pk[c1]))
                            pq[c0], pq[c1] = cute.arch.fma_packed_f32x2(
                                (s0, s1), (bv, bv), (pq[c0], pq[c1]))
                # LK-way butterfly over the lk lanes (offsets LV, 2LV, 4LV)
                for off in cutlass.range_constexpr(LOG_LK):
                    m = LV << off
                    for c in cutlass.range_constexpr(CPT):
                        pk[c] = pk[c] + cute.arch.shuffle_sync_bfly(pk[c], m)
                        pq[c] = pq[c] + cute.arch.shuffle_sync_bfly(pq[c], m)
                cute.autovec_copy(sV[j, vgc, None], vv)
                cute.autovec_copy(sSC[j, None], sc2)
                for c in cutlass.range_constexpr(CPT):
                    uu[c] = sc2[0] * (vv[c] - pk[c])
                if lk == 0:
                    for c in cutlass.range_constexpr(CPT):
                        rf[c] = pq[c] + uu[c] * sc2[1]
                        ub[c] = uu[c].to(BF16)
                    # r is published to SMEM, not HBM: the gated RMSNorm below needs
                    # all 128 columns of the token, which live in 8 different warps.
                    cute.autovec_copy(rf, sR[j, vgc, None])
                    cute.autovec_copy(ub, mOu4[ru_o + (pos0 + j) * H, vgc, None])
                # pass 2: rank-1 update, hat += c_t u^T  (c_t = exp(-Gr_t) * kn_t,
                # built in place over sKN by P1c).  The coefficient is re-read rather
                # than carried in a (VEC, NKG) fragment: holding it in fp32 would cost
                # 16 registers and 128 regs/thread is exactly the 2-blocks-per-SM
                # cliff (65536 / (256*128) == 2).
                for jj in cutlass.range_constexpr(NKG):
                    cute.autovec_copy(sKN[j, kg0 + jj * LK, None], knf)
                    for kk in cutlass.range_constexpr(VEC):
                        knv = knf[kk]
                        for cwp in cutlass.range_constexpr(CPT // 2):
                            c0 = 2 * cwp
                            c1 = c0 + 1
                            st[kk, jj, c0], st[kk, jj, c1] = (
                                cute.arch.fma_packed_f32x2(
                                    (knv, knv), (uu[c0], uu[c1]),
                                    (st[kk, jj, c0], st[kk, jj, c1])))
            # S is deliberately NOT written here: the verify step must not commit.

        cute.arch.sync_threads()

        # ---- P9: fused gated-RMSNorm epilogue --------------------------------
        #   o = (r * rsqrt(mean_v(r^2) + 1e-5)) * (w * sigmoid(z))
        # One warp per token: V/VEC == 32 == warp size, so lane owns the 4
        # consecutive columns 4*lane..4*lane+3 and a single 5-step butterfly
        # closes the 128-wide reduction.  The mean is a multiply by 1/V.
        # Everything here is SMEM-resident (sR from P8, sZG from P2), so the tail
        # after the barrier is 2 LDS.128 + 4 FMA + 5 SHFL + rsqrt + 1 STG.
        rn = cute.make_fragment((VEC,), FP32)
        ob = cute.make_fragment((VEC,), BF16)
        for rr in cutlass.range_constexpr(TPW):
            j = w + rr * NWARP
            if j < T:
                cute.autovec_copy(sR[j, lane, None], rn)
                ss = FP32(0.0)
                for e in cutlass.range_constexpr(VEC):
                    ss = ss + rn[e] * rn[e]
                for off in cutlass.range_constexpr(5):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << off)
                rs = FP32(cute.math.rsqrt(ss * FP32(RCP_V) + FP32(NORM_EPS),
                                          fastmath=True))
                if cutlass.const_expr(gated):
                    cute.autovec_copy(sZG[j, lane, None], zg)
                    for e in cutlass.range_constexpr(VEC):
                        ob[e] = ((rn[e] * rs) * zg[e]).to(BF16)
                else:
                    for e in cutlass.range_constexpr(VEC):
                        ob[e] = rn[e].to(BF16)
                cute.autovec_copy(ob, mO[(b * T + j) * H + h, lane, None])

    # Shared traced launch body; per-variant @cute.jit signatures below expose
    # ONLY the pointers that variant reads (unused tensor views are aliased
    # from live pointers of the same dtype).
    def _launch_body(pQ, pK, pV, pGr, pBr, pAl, pDb, pS, pZ, pW, pO, pOu,
                     pOk, pOg, pPn, pBi, pX, pCS, pCWt, pCWin,
                     lower_bound, nb):
        rows = nb * (T * H)
        nbh = nb * H
        nrg = nb * (2 * H)
        nru = nb * (2 * HIST * H)
        lay4 = cute.make_layout((rows, K // VEC, VEC), stride=(K, VEC, 1))
        layv = cute.make_layout((rows, V // VEC, VEC), stride=(V, VEC, 1))
        mQ = cute.make_tensor(pQ, lay4)
        mKt = cute.make_tensor(pK, lay4)
        mGr = cute.make_tensor(pGr, lay4)
        mV4 = cute.make_tensor(pV, layv)
        mZ = cute.make_tensor(pZ, layv)
        mO = cute.make_tensor(pO, layv)
        mW = cute.make_tensor(pW, cute.make_layout(
            (V // VEC, VEC), stride=(VEC, 1)))
        mBr = cute.make_tensor(pBr, cute.make_layout((rows,), stride=(1,)))
        mAl = cute.make_tensor(pAl, cute.make_layout((H,), stride=(1,)))
        mDb = cute.make_tensor(pDb, cute.make_layout(
            (H, K // VEC, VEC), stride=(K, VEC, 1)))
        mS = cute.make_tensor(pS, cute.make_layout(
            (nbh, V, K // VEC, VEC), stride=(V * K, K, VEC, 1)))
        mOu4 = cute.make_tensor(pOu, cute.make_layout(
            (nru, V // VEC, VEC), stride=(V, VEC, 1)))
        mOu8 = cute.make_tensor(pOu, cute.make_layout(
            (nru, V // VECH, VECH), stride=(V, VECH, 1)))
        mOk4 = cute.make_tensor(pOk, cute.make_layout(
            (nru, K // VEC, VEC), stride=(K, VEC, 1)))
        mOk8 = cute.make_tensor(pOk, cute.make_layout(
            (nru, K // VECH, VECH), stride=(K, VECH, 1)))
        mOgV = cute.make_tensor(pOg, cute.make_layout(
            (nrg, K, HIST // VEC, VEC), stride=(K * HIST, HIST, VEC, 1)))
        mOgW = cute.make_tensor(pOg, cute.make_layout(
            (nrg, K, HIST // GW, GW), stride=(K * HIST, HIST, GW, 1)))
        mOgS = cute.make_tensor(pOg, cute.make_layout(
            (nrg, K, HIST), stride=(K * HIST, HIST, 1)))
        mPn = cute.make_tensor(pPn, cute.make_layout((nb,), stride=(1,)))
        mBi = cute.make_tensor(pBi, cute.make_layout((nb,), stride=(1,)))
        CDIM = 3 * H * K
        ntok = nb * T
        mX = cute.make_tensor(pX, cute.make_layout(
            (ntok, CDIM // VEC, VEC), stride=(CDIM, VEC, 1)))
        mCS = cute.make_tensor(pCS, cute.make_layout(
            (nb, CDIM, CONV_W - 1),
            stride=(CDIM * (CONV_W - 1), CONV_W - 1, 1)))
        mCWt = cute.make_tensor(pCWt, cute.make_layout(
            (CDIM, CONV_W), stride=(CONV_W, 1)))
        # BLOCKED conv_win view: one lane's VEC channels x CSL slots are 12
        # contiguous bf16, written by _conv_fold as one vectorized copy
        mCWin = cute.make_tensor(pCWin, cute.make_layout(
            (ntok, CDIM // VEC, VEC * (CONV_W - 1)),
            stride=(CDIM * (CONV_W - 1), VEC * (CONV_W - 1), 1)))
        blackwell_bf16_kda_replay_ssm_fused_gated_kernel(
            mQ, mKt, mV4, mGr, mBr, mAl, mDb, mS, mZ, mW, mO, mOu4, mOu8,
            mOk4, mOk8, mOgV, mOgW, mOgS, mPn, mBi, mX, mCS, mCWt, mCWin,
            lower_bound,
        ).launch(grid=[H, nb, 1], block=[THREADS, 1, 1],
                 min_blocks_per_mp=MBPM)

    if fuse_conv:
        @cute.jit
        def launch(
            pX: cute.Pointer,
            pGr: cute.Pointer,
            pBr: cute.Pointer,
            pAl: cute.Pointer,
            pDb: cute.Pointer,
            pS: cute.Pointer,
            pZ: cute.Pointer,
            pW: cute.Pointer,
            pO: cute.Pointer,
            pOu: cute.Pointer,
            pOk: cute.Pointer,
            pOg: cute.Pointer,
            pPn: cute.Pointer,
            pBi: cute.Pointer,
            pCS: cute.Pointer,
            pCWt: cute.Pointer,
            pCWin: cute.Pointer,
            lower_bound: FP32,
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pX, pX, pX, pGr, pBr, pAl, pDb, pS, pZ, pW, pO,
                         pOu, pOk, pOg, pPn, pBi, pX, pCS, pCWt, pCWin,
                         lower_bound, nb)
    else:
        @cute.jit
        def launch(
            pQ: cute.Pointer,
            pK: cute.Pointer,
            pV: cute.Pointer,
            pGr: cute.Pointer,
            pBr: cute.Pointer,
            pAl: cute.Pointer,
            pDb: cute.Pointer,
            pS: cute.Pointer,
            pZ: cute.Pointer,
            pW: cute.Pointer,
            pO: cute.Pointer,
            pOu: cute.Pointer,
            pOk: cute.Pointer,
            pOg: cute.Pointer,
            pPn: cute.Pointer,
            pBi: cute.Pointer,
            lower_bound: FP32,
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pQ, pK, pV, pGr, pBr, pAl, pDb, pS, pZ, pW, pO,
                         pOu, pOk, pOg, pPn, pBi, pQ, pQ, pQ, pQ,
                         lower_bound, nb)

    return launch

# ==================== launch engine + importable API ====================
import torch
from cutlass.cute.runtime import make_ptr
FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
GMEM = cute.AddressSpace.gmem

PEAK_GBPS = 8000.0                      # B200 HBM3e ~8 TB/s
K_EPS = 1e-6
LOWER_BOUND = -4.0
# (B, T, pnat)
BENCH_SHAPES = [(4, 4, 8), (64, 2, 8), (64, 4, 8), (64, 8, 8), (256, 4, 8),
                (64, 4, 13)]
PRIMARY = (64, 4, 8)
# Measured on THIS B200 with identical inputs (from the kernel spec), us/iter.
SPEC_BAR = {
    (4, 4, 8): {"trt_cached_replay": 56.7, "sglang_snapshot": 21.1},
    (64, 2, 8): {"trt_cached_replay": 65.9, "sglang_snapshot": 54.4},
    (64, 4, 8): {"trt_cached_replay": 68.0, "sglang_snapshot": 87.7},
    (64, 8, 8): {"trt_cached_replay": 96.8, "sglang_snapshot": 151.3},
    (256, 4, 8): {"trt_cached_replay": 267.7, "sglang_snapshot": 285.4},
    (64, 4, 13): {"trt_cached_replay": 95.4},
}


# ---------------------------------------------------------------------------
# Engine: compiled launcher cached per (H, K, V, T, variant).  Nothing
# input-dependent is cached -- every call rebinds the live device pointers of
# the tensors it is given, so the kernel always reads current data (see the
# same-object-mutation audit in run_check(), and
# guide/CuTeDSL_Clean_Kernel_Guide.md).
# ---------------------------------------------------------------------------
# bare:  q     k     v     g_raw  beta  A_log dt_b  S     o     ou    ok    og    pn    bi
_PTR_SPEC_BARE = ((BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16), (FP32, 4),
                  (FP32, 4), (FP32, 16), (FP32, 16), (BF16, 16), (BF16, 16),
                  (BF16, 16), (FP32, 16), (I32, 4), (I32, 4))
# gated/prenorm (1-CTA body):
#        q     k     v     g_raw  beta  A_log dt_b  S     z     w
#        o     ou    ok    og    pn    bi
_PTR_SPEC_GATED = ((BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16), (FP32, 4),
                   (FP32, 4), (FP32, 16), (FP32, 16), (BF16, 16), (FP32, 16),
                   (BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16), (I32, 4),
                   (I32, 4))
# conv (bare body, fuse_conv):
#        x     g_raw  beta  A_log dt_b  S     o     ou    ok    og
#        pn    bi    conv_state conv_weight conv_win
_PTR_SPEC_CONV = ((BF16, 16), (FP32, 16), (FP32, 4), (FP32, 4), (FP32, 16),
                  (FP32, 16), (BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16),
                  (I32, 4), (I32, 4), (BF16, 2), (BF16, 8), (BF16, 16))
# conv_gated (1-CTA body, fuse_conv):
#        x     g_raw  beta  A_log dt_b  S     z     w     o     ou
#        ok    og    pn    bi    conv_state conv_weight conv_win
_PTR_SPEC_CONV_GATED = ((BF16, 16), (FP32, 16), (FP32, 4), (FP32, 4),
                        (FP32, 16), (FP32, 16), (BF16, 16), (FP32, 16),
                        (BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16),
                        (I32, 4), (I32, 4), (BF16, 2), (BF16, 8), (BF16, 16))

_SPECS = {"bare": _PTR_SPEC_BARE, "gated": _PTR_SPEC_GATED,
          "prenorm": _PTR_SPEC_GATED, "conv": _PTR_SPEC_CONV,
          "conv_gated": _PTR_SPEC_CONV_GATED}
_FLAGS = {"bare": (False, False, False), "gated": (True, False, False),
          "prenorm": (False, True, False), "conv": (False, False, True),
          "conv_gated": (True, False, True)}


class _Engine:
    def __init__(self):
        self._compiled = {}
        self._args = {}

    def _get_compiled(self, H, K, V, T, variant):
        c = self._compiled.get((H, K, V, T, variant))
        if c is None:
            spec = _SPECS[variant]
            gated, prenorm, fuse_conv = _FLAGS[variant]
            c = cute.compile(
                make_launcher(H, K, V, T, gated=gated, prenorm=prenorm,
                              fuse_conv=fuse_conv),
                *[make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in spec],
                FP32(0.0), I32(0),
            )
            self._compiled[(H, K, V, T, variant)] = c
        return c

    def prepare(self, H, K, V, T, B, variant, tensors):
        key = (H, K, V, T, B, variant)
        e = self._args.get(key)
        if e is None:
            for t in tensors:
                if not t.is_contiguous():
                    raise ValueError("kda_replay_ssm_fused requires contiguous inputs")
            spec = _SPECS[variant]
            e = (self._get_compiled(H, K, V, T, variant),
                 [make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in spec],
                 I32(B))
            self._args[key] = e
        return e


_ENGINE = _Engine()


def kda_replay_ssm_fused(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                    S, old_u, old_k, old_G, pnat, buf_idx):
    """KDA MTP verify step (rollback semantics). Native (short) entry.

    q, k, v   : [B, T, H, 128] bf16
    g_raw     : [B, T, H, 128] fp32   raw gate pre-activation
    beta_raw  : [B, T, H]      fp32   raw beta logits
    A_log     : [H]            fp32
    dt_bias   : [H*128]        fp32
    lower_bound : float in [-5, 0)
    S         : [B, H, 128, 128] fp32   checkpoint, layout [slot, head, V, K];
                                        written ONLY on ring overflow
    old_u     : [B, 2, HIST, H, 128] bf16
    old_k     : [B, 2, HIST, H, 128] bf16
    old_G     : [B, 2, H, 128, HIST] fp32
    pnat      : [B] int32   accepted tokens in the active ring half (read-only)
    buf_idx   : [B] int32   active ring half (read-only)

    returns o : [B, T, H, 128] bf16
    """
    B, T, H, K = q.shape
    V = S.shape[3]
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=q.device)
    compiled, p, nb = _ENGINE.prepare(
        H, K, V, T, B, "bare",
        (q, k, v, g_raw, beta_raw, A_log, dt_bias, S, old_u, old_k, old_G,
         pnat, buf_idx))
    for ptr, t in zip(p, (q, k, v, g_raw, beta_raw, A_log, dt_bias, S, o,
                          old_u, old_k, old_G, pnat, buf_idx)):
        ptr._desc.value = t.data_ptr()
    compiled(*p, FP32(float(lower_bound)), nb)
    return o


def _run_onecta(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                S, old_u, old_k, old_G, pnat, buf_idx, z, w, variant):
    B, T, H, K = q.shape
    V = S.shape[3]
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=q.device)
    compiled, p, nb = _ENGINE.prepare(
        H, K, V, T, B, variant,
        (q, k, v, g_raw, beta_raw, A_log, dt_bias, S, z, w, old_u, old_k,
         old_G, pnat, buf_idx))
    for ptr, t in zip(p, (q, k, v, g_raw, beta_raw, A_log, dt_bias, S, z, w, o,
                          old_u, old_k, old_G, pnat, buf_idx)):
        ptr._desc.value = t.data_ptr()
    compiled(*p, FP32(float(lower_bound)), nb)
    return o


def kda_replay_ssm_fused_gated(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                          S, old_u, old_k, old_G, pnat, buf_idx, z, w):
    """KDA MTP verify step (rollback semantics) + fused gated RMSNorm epilogue.
    Returns o [B,T,H,V] bf16 POST gated-RMSNorm; S written only on ring overflow."""
    return _run_onecta(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound, S,
                       old_u, old_k, old_G, pnat, buf_idx, z, w, "gated")


def kda_replay_ssm_fused_prenorm(q, k, v, g_raw, beta_raw, A_log, dt_bias,
                            lower_bound, S, old_u, old_k, old_G, pnat, buf_idx,
                            z, w):
    """Ablation: the 1-CTA (gated-champion) body with the epilogue compiled OUT
    (returns PRE-norm r).  Timing stand-in for 'our kernel unfused' + a separate
    gated-RMSNorm launch."""
    return _run_onecta(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound, S,
                       old_u, old_k, old_G, pnat, buf_idx, z, w, "prenorm")


def _check_conv_args(x, g_raw, S, conv_state, conv_weight, conv_win):
    B, T, H, K = g_raw.shape
    V = S.shape[3]
    CDIM = 3 * H * K
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    assert x.shape == (B * T, CDIM) and x.dtype == torch.bfloat16
    assert conv_state.shape == (B, CDIM, CONV_W - 1)
    assert conv_state.dtype == torch.bfloat16
    assert conv_weight.shape == (CDIM, CONV_W)
    assert conv_weight.dtype == torch.bfloat16
    assert conv_win.shape == (B * T, CDIM, CONV_W - 1)
    assert conv_win.dtype == torch.bfloat16
    return B, T, H, K, V


def kda_replay_ssm_fused_conv(x, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                              S, old_u, old_k, old_G, pnat, buf_idx,
                              conv_state, conv_weight, conv_win):
    """:func:`kda_replay_ssm_fused` with the width-4 causal-conv+SiLU input
    stage fused into the kernel.  Instead of post-conv q/k/v it consumes:

    x           : [B*T, 3*H*128] bf16    RAW packed mixed_qkv (row = b*T + t)
    conv_state  : [B, 3*H*128, 3] bf16   raw pre-conv history, READ-ONLY
                                         (rollback semantics: host commits
                                         after sampling)
    conv_weight : [3*H*128, 4] bf16      conv taps (col 3 hits the new token)
    conv_win    : [B*T, 3*H*128, 3] bf16 per-token raw window snapshots, WRITTEN

    Everything else is identical to :func:`kda_replay_ssm_fused`.
    Returns o [B,T,H,128] bf16.
    """
    B, T, H, K, V = _check_conv_args(x, g_raw, S, conv_state, conv_weight,
                                     conv_win)
    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=x.device)
    compiled, p, nb = _ENGINE.prepare(
        H, K, V, T, B, "conv",
        (x, g_raw, beta_raw, A_log, dt_bias, S, old_u, old_k, old_G,
         pnat, buf_idx, conv_state, conv_weight, conv_win))
    for ptr, t in zip(p, (x, g_raw, beta_raw, A_log, dt_bias, S, o, old_u,
                          old_k, old_G, pnat, buf_idx, conv_state,
                          conv_weight, conv_win)):
        ptr._desc.value = t.data_ptr()
    compiled(*p, FP32(float(lower_bound)), nb)
    return o


def kda_replay_ssm_fused_conv_gated(x, g_raw, beta_raw, A_log, dt_bias,
                                    lower_bound, S, old_u, old_k, old_G,
                                    pnat, buf_idx, conv_state, conv_weight,
                                    conv_win, z, w):
    """:func:`kda_replay_ssm_fused_conv` + the fused gated-RMSNorm epilogue
    (POST-norm o).  z [B,T,H,128] bf16 output gate; w [128] fp32 weight."""
    B, T, H, K, V = _check_conv_args(x, g_raw, S, conv_state, conv_weight,
                                     conv_win)
    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=x.device)
    compiled, p, nb = _ENGINE.prepare(
        H, K, V, T, B, "conv_gated",
        (x, g_raw, beta_raw, A_log, dt_bias, S, z, w, old_u, old_k, old_G,
         pnat, buf_idx, conv_state, conv_weight, conv_win))
    for ptr, t in zip(p, (x, g_raw, beta_raw, A_log, dt_bias, S, z, w, o,
                          old_u, old_k, old_G, pnat, buf_idx, conv_state,
                          conv_weight, conv_win)):
        ptr._desc.value = t.data_ptr()
    compiled(*p, FP32(float(lower_bound)), nb)
    return o


def fused_recurrent_gated_delta_rule_cached_replay_update(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    state_indices,
    old_u,
    old_k,
    old_G,
    old_beta,  # noqa: ARG001 — TRT GDN scratch / API compat; unused for KDA
    cache_buf_idx,
    prev_num_accepted_tokens,
    history_size,  # noqa: ARG001 — baked as HIST=16 in this kernel
    replay_indices=None,  # noqa: ARG001
    scale=None,  # noqa: ARG001
    use_qk_l2norm_in_kernel=False,  # noqa: ARG001 — always on in this kernel
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    **kwargs,  # noqa: ARG001 — accept TRT tuning/PDL/norm kwargs, ignore
):
    """Drop-in replacement for TRT-LLM's Triton replay_ssm_fused verify entry.

    Same call signature as
    ``tensorrt_llm._torch.modules.fla.cached_replay.fused_recurrent_gated_delta_rule_cached_replay_update``.
    Maps names onto :func:`kda_replay_ssm_fused` and returns ``o``.
    """
    import torch

    if A_log is None or dt_bias is None or lower_bound is None:
        raise ValueError(
            "b10 drop-in requires A_log, dt_bias, and lower_bound "
            "(same as the TRT KDA path)"
        )
    B = q.shape[0]
    idx = state_indices
    # Zero-copy when the pool is already the dense B-slot layout with
    # identity indices; otherwise gather (and scatter back after, so an
    # overflow fold lands in the caller's pool).
    identity = (
        ssm_states.shape[0] == B
        and idx.dtype in (torch.int32, torch.int64)
        and idx.numel() == B
        and bool(torch.equal(idx, torch.arange(B, device=idx.device, dtype=idx.dtype)))
    )
    if identity:
        S = ssm_states
        o = kda_replay_ssm_fused(
            q, k, v, g, beta, A_log, dt_bias, float(lower_bound),
            S, old_u, old_k, old_G, prev_num_accepted_tokens, cache_buf_idx,
        )
        return o
    S = ssm_states.index_select(0, idx.long()).contiguous()
    o = kda_replay_ssm_fused(
        q, k, v, g, beta, A_log, dt_bias, float(lower_bound),
        S, old_u, old_k, old_G, prev_num_accepted_tokens, cache_buf_idx,
    )
    ssm_states.index_copy_(0, idx.long(), S)
    return o
