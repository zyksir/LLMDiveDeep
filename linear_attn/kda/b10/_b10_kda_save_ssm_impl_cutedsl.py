"""CuTeDSL KDA spec-verify save_ssm kernel family (KDA.md schema 1) — black-box, importable.

Private shared implementation for the public ``b10_kda_save_ssm_*_cutedsl``
modules. One parametrized kernel generator (``make_launcher``: compile-time
flags ``gated`` and ``fuse_conv``) is JIT-cached per flag combination + launch
shape. The former bare and gated sources were near-duplicates;
the bare entry compiles to exactly the bare champion's code path (epilogue
compiled out, same CFG_V2 tile), the gated entry to the gated champion's
(CFG_W8 tile, epilogue compiled in).

=============================== kda_save_ssm ===============================
IDENTICAL TO (same end-to-end function, just faster):
  - SGLang Triton ``fused_sigmoid_gating_delta_rule_update``
    (sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py)
    in save_ssm-verify mode: ``intermediate_states_buffer`` + ``cache_steps``,
    ``disable_state_update=True`` — the ``verify_save_ssm`` row in
    ``bench_kda_spec_verify.py`` / KDA.md ``save_ssm``
Same recurrence, outputs, and per-token full-state snapshots. Difference:
this kernel takes PRE-ACTIVATED fp32 ``g`` (SGLang fuses softplus gate).
NOT the TRT replay-SSM path — that is ``b10_kda_replay_ssm_cutedsl``.

DROP-IN API? NO — not a one-line replace for SGLang save_ssm-verify.
  Triton: fused_sigmoid_gating_delta_rule_update(..., disable_state_update=True,
           intermediate_states_buffer=snap [slots,steps,H,V,K],
           intermediate_state_indices, cache_steps) with raw gate.
  This:   kda_save_ssm(q, k, v, g, beta, S0, cu_seqlens, snapshot) — packed
           [TT,H,*], pre-activated g, S0 read-only [B,H,K,V],
           snapshot [TT,H,K,V] written.
  To swap: use save_ssm_closure in kda_verify_register.py.

From the CuTeDSLGen champion gen_kda_save_ssm_claude_0729_0235/best. WINS:
matches SGLang verify_save_ssm output and beats it 1.32-1.70x at every shape
(H=12/96), ~80% roofline.

============================ kda_save_ssm_gated ============================
IDENTICAL TO (same end-to-end function, just faster):
  the FUSION of these two Triton kernels into ONE launch:
  - SGLang ``fused_sigmoid_gating_delta_rule_update`` in save_ssm-verify mode
    (``intermediate_states_buffer`` + ``cache_steps``) — ``save_ssm``
  - SGLang ``FusedRMSNormGated`` (fla/fused_norm_gate.py, sigmoid)
Same as running ``verify_save_ssm`` then a separate gated RMSNorm, fused.
Takes PRE-ACTIVATED fp32 ``g``; output is POST-norm.

DROP-IN API? NO — no single Triton twin (save_ssm + norm fused here).
  Triton path: verify_save_ssm then separate FusedRMSNormGated.
  This:   kda_save_ssm_gated(..., snapshot, z, w) — same packing/layout
           gaps as kda_save_ssm, plus trailing z/w; post-norm o.

From the CuTeDSLGen champion gen_kda_save_ssm_gated_claude_0729_1535/best.
WINS: matches SGLang verify_save_ssm + FusedRMSNormGated output; the fused
gated-RMSNorm epilogue is nearly free. ~78.7% roofline.

======================= kda_save_ssm_conv[(_gated)] ========================
IDENTICAL TO (same end-to-end function, one launch instead of two/three):
  - SGLang Triton ``causal_conv1d_update(activation="silu")``
    (sglang/srt/layers/attention/mamba/causal_conv1d_triton.py) on the packed
    raw qkv projection, INCLUDING its spec-decode per-step
    ``intermediate_conv_window`` snapshots, followed by
  - ``verify_save_ssm`` (and, for the _gated entry, ``FusedRMSNormGated``).
The width-4 causal conv + SiLU runs in-kernel on RAW pre-conv packed qkv;
the per-request conv window is seeded from ``conv_state`` (raw pre-conv
history) and per-step window snapshots are written so a later commit can
restore the window as it stood after exactly ``accept_len`` tokens —
the conv analogue of the per-token recurrent-state snapshots.
Conv numerics match SGLang/TRT: fp32 accumulate, fp32 SiLU, then the
activated value is rounded to bf16 (matching the separate conv launch's
bf16 output) before entering the recurrence.

DROP-IN API? NO — the Triton path is two/three separate launches.
  This: kda_save_ssm_conv(x, g, beta, S0, cu_seqlens, snapshot,
                          conv_state, conv_weight, conv_win)
        kda_save_ssm_conv_gated(..., z, w)
  x          [TT, 3*H*128] bf16   raw pre-conv packed qkv (q | k | v blocks)
  conv_state [B, 3*H*128, 3] bf16 raw pre-conv history (col 0 oldest), READ-ONLY
  conv_weight[3*H*128, 4] bf16    width-4 conv taps (col 3 hits the new token)
  conv_win   [TT, 3*H*128, 3] bf16 WRITTEN: per-token window snapshot
             (raw [x_{t-2}, x_{t-1}, x_t]; commit = copy row of the accepted
             token into conv_state, same layout)

Public APIs are split by fused stage across
``b10_kda_save_ssm[_conv][_gated]_cutedsl`` modules.
    q,k,v [TT,H,128] bf16; g [TT,H,128] fp32 log-decay(<=0); beta [TT,H] fp32;
    S0 [B,H,128,128] fp32 (read-only); cu_seqlens [B+1] int32; snapshot
    [TT,H,128,128] fp32 written per-token (post-state, [K,V] layout);
    z [TT,H,128] bf16 gate; w [128] fp32 norm weight; o [TT,H,128] bf16
    (POST gated-RMSNorm for the _gated entries). B200 sm_100a; JIT on first
    call. kda_save_ssm_prenorm(...) is the epilogue-off ablation (writes
    PRE-norm r) for the "our kernel unfused + separate norm" timing baseline.

Self-test (conv variants): run this file directly —
    CUDA_VISIBLE_DEVICES=6 python b10_kda_save_ssm_cutedsl.py
"""
# ============================ kernel module ============================
# SPDX-License-Identifier: Apache-2.0
"""
blackwell_bf16_kda_save_ssm — KDA (Kimi Delta Attention) speculative-decode
MTP-verify step in the **save_ssm** scheme (KDA.md schema 1), optionally with
the gated-RMSNorm output epilogue and/or the width-4 causal-conv+SiLU input
stage fused into the same kernel, for B200 (sm_100a), CuTeDSL 4.5.2.

Per (request n, head h), sequentially over the request's T draft tokens:

    q[t],k[t],v[t] = SiLU(conv4(raw qkv window))       (only if fuse_conv)
    qn = q[t]/||q[t]||          kn = k[t]/||k[t]||     (L2 norms fused)
    S  = diag(exp(g[t])) @ S                           (decay)
    u  = beta[t] * (v[t] - S^T kn)
    S  = S + kn u^T                                    (rank-1 update)
    r  = S^T qn / sqrt(K)                              (pre-norm output)
    o[t]        = r * rsqrt(mean(r^2) + 1e-5) * w * sigmoid(z[t])  (if gated)
    snapshot[t] = S                                    (FULL post-token state)
    conv_win[t] = raw window after token t             (only if fuse_conv)

`S0` is read-only; the FULL post-token state is emitted to `snapshot[t]` after
**every** token, which is what makes this kernel a pure HBM-store problem:

    HBM per state element:  4 (read S0)  +  4*T (snapshots)

so the design goal is to stream that store at HBM speed with the recurrence,
the epilogue AND the conv hidden underneath it.

=============================================================================
Decomposition — CFG_V2 (bare) vs CFG_W8 (gated)
=============================================================================
    grid = (V_SPLITS, H, B)   block = (NWARP*32, 1, 1)
    CTA      owns S[b, h, :, vsi*VS : vsi*VS+VS]
    warp w   owns key channels [KPT*w, KPT*(w+1))
    lane     owns CPT V columns
    -> KPT (channel) x CPT (column) fp32 register tile per thread

The un-gated kernel's fastest tile is `CFG_V2` = (NWARP=4, CPT=2), which
splits V in half across **two CTAs** (`VSPLITS = 2`). The RMS reduction
`sum_c r_c^2` spans all 128 output channels, so a V-split CTA holds only half
of it and would need a cross-CTA (cluster/DSMEM) reduction every token. The
gated variant therefore uses `CFG_W8` = (NWARP=8, CPT=4), which keeps the same
64 state registers per thread and the same 16-warps-per-SM occupancy but gives
**one CTA the whole 128-wide output row**. In the un-gated kernel that config
measured 0.3174 ms vs 0.3160 ms for CFG_V2 — a 0.6% tile tax, paid once, in
exchange for an epilogue with **zero extra CTA barriers**.

The epilogue itself rides on a property of the existing reduction: with
`SPT == 1`, warps 0-3 produce `u` and warps 4-7 produce the `q`-partial sums,
and then **warp 0 alone** already stores all 128 output columns (`r == 0` is
exactly one 32-lane group covering `CPT * 32 = 128` columns). So warp 0 holds
the entire pre-norm row `r` in registers at the point of the store: `sum r^2`
is one 5-step warp butterfly, and the scale is one `rsqrt`. No barrier, no
SMEM round-trip, no second kernel.

`w * sigmoid(z[t])` is precomputed into SMEM by the **prologue** (which is
already NWARP-parallel over the chunk's tokens) rather than loaded in the
epilogue: the epilogue is the last ~20 instructions before the next token's
dependent work, and a late L2 round-trip there cannot be hidden. `w` itself is
loaded into registers at the very top of the kernel, next to the `S0` loads.

The fused conv (fuse_conv=True) lives entirely in the PROLOGUE: the conv
window over the T verify tokens is a pure function of the RAW inputs (the
window shifts in raw values, not conv outputs), so every token's conv is
computed independently by the warp that stages that token — no serialization
is added to the recurrence. Positions p-3..p-1 come from the raw input stream
when p-i >= 0 and from `conv_state[b, c, (p-i)+3]` otherwise; per-token window
snapshots (`[x_{p-2}, x_{p-1}, x_p]`, raw bf16) are written in the same phase.
Under a V-split (CFG_V2) both CTAs compute the q/k conv (both need q/k) but
only vsi == 0 writes the q/k window snapshots; each CTA writes its own
disjoint v-slice snapshots.

Three things made the un-gated ancestor fast and are preserved verbatim:

1. **All 32 lanes of a store must land in ONE 128 B line.** The L1TEX data pipe
   retires one wavefront per *distinct 128 B line* per warp instruction, not per
   byte. `LK = 1` (whole warp on one key-channel row) is what keeps the snapshot
   store at one wavefront per instruction; it also makes every per-channel
   scalar load warp-uniform.
2. **The state must stay in registers.** `autovec_copy` against the state
   fragment de-promotes it to local memory (measured 6x slower), so `VECST`
   (128-bit snapshot stores) is a config switch that was measured and rejected.
3. **Every SMEM layout gets an EXPLICIT row-major stride.** `cute.make_layout(
   shape)` is LayoutLeft, so the default puts the FIRST mode contiguous and the
   lane-varying index lands on strided words (8-way bank conflict).
"""

import math
import os

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
F16 = cutlass.Float16
I32 = cutlass.Int32

VEC = 4                  # fp32 elements per 128-bit access
VECH = 8                 # fp16 elements per 128-bit access
TMAX = 8                 # tokens staged in SMEM per chunk (power of two)
LOG_TMAX = 3
EPS = 1.0e-5             # RMSNorm epsilon (matches FusedRMSNormGated default)
CONV_W = 4               # causal conv width (KERNEL_WIDTH in SGLang Triton)
CSL = CONV_W - 1         # conv_state history slots (last 3 raw tokens)
NTI = 3                  # packed tensors in the conv stream (q | k | v)

# Where the 64-per-thread `S0` load burst is issued relative to the first
# chunk's prologue. False (shipped) = at the very top. MEASURED AND REJECTED:
# True is 5.9% SLOWER at B=64 T=2 (220.8 vs 208.3 us, reproduced twice) and a
# wash elsewhere. Kept as a switch for the record.
S0_AFTER_PROLOGUE = os.environ.get("KDA_S0_LATE", "0") == "1"

# Tile configuration `(NWARP, LK, CPT, VECST)`; everything else is derived:
#   LV  = 32/LK lanes over V,  VS = LV*CPT columns per CTA,  VSPLITS = V/VS,
#   KPT = K/(NWARP*LK) key channels per thread, SR = KPT*CPT state registers,
#   R   = NWARP*LK partials per column.
# LK=1 (whole warp on one key-channel row) is the load-bearing choice; see the
# module docstring. VECST switches the state load/store between 128-bit vector
# copies (4 contiguous columns per lane) and coalesced scalar copies (columns
# strided by 32 so that one instruction covers one 128 B line).
# The fused epilogue (gated=True) requires VSPLITS == 1 (the RMS reduction
# spans all V) and SPT == 1 (warp 0 alone owns the whole output row); CFG_W8
# is the only tile that satisfies both at LK == 1.
CFG_ROW = (4, 1, 4, False)   # NWARP=4, KPT=32, CPT=4, SR=128, VS=128, R=4
CFG_VEC = (4, 1, 4, True)    # ... same tile, 128-bit state load/store
CFG_W8 = (8, 1, 4, False)    # NWARP=8, KPT=16, CPT=4, SR=64,  VS=128, R=8
CFG_W8V = (8, 1, 4, True)    # ... same tile, 128-bit state load/store (VECST)
CFG_V2 = (4, 1, 2, False)    # NWARP=4, KPT=32, CPT=2, SR=64,  VS=64,  R=4
CFG_WIDE = (2, 4, 8, False)  # NWARP=2, KPT=16, CPT=8, SR=128, VS=64,  R=8
CFG_W8K2 = (8, 2, 8, False)  # NWARP=8, KPT=8,  CPT=8, SR=64,  VS=128, R=16


def make_launcher(H: int, K: int, V: int, cfg=CFG_V2, gated: bool = False,
                  fuse_conv: bool = False):
    """Build a compiled-once launcher closure for (H, K, V, cfg, flags).

    gated=True compiles the fused gated-RMSNorm epilogue in (POST-norm o);
    gated=False writes the PRE-norm output r — exactly the bare champion.
    fuse_conv=True compiles the width-4 causal-conv+SiLU input stage in
    (q/k/v are then read from the raw packed-qkv stream, not mQ/mK/mV).
    """
    NWARP, LK, CPT, VECST = cfg
    THREADS = NWARP * 32
    LV = 32 // LK
    VS = LV * CPT
    KPT = K // (NWARP * LK)          # key channels per thread
    assert KPT * NWARP * LK == K
    assert KPT % VECH == 0
    assert V % VS == 0 and VS % VEC == 0
    assert CPT == VEC or not VECST   # 128-bit copies need exactly VEC columns
    assert LV % CPT == 0 or VECST
    VSPLITS = V // VS
    CPL = K // 32                    # prologue channels per lane (4)
    TPW = TMAX // NWARP if TMAX >= NWARP else 1
    RSQRT_K = 1.0 / math.sqrt(K)
    RMEAN = 1.0 / V                  # mean(r^2) is a multiply, never a divide
    R = NWARP * LK                   # partials per column
    # sR row stride == LV (mod 32) words: for LK>1 the 32 lanes of one scatter
    # differ by LV words in lk and 1 word in lv, so they cover all 32 banks
    # exactly once. At LK=1 the lanes are already 32 consecutive words.
    RSTRIDE = 2 * VS + (LV if LK > 1 else 0)
    LOG_LV = int(math.log2(LV))
    LOG_VS = int(math.log2(VS))
    LOG_CPT = int(math.log2(CPT))
    SPT = (2 * VS) // THREADS        # reduction slots per thread
    assert SPT * THREADS == 2 * VS
    ZPL = VS // 32                   # z / w elements per lane in the prologue
    assert ZPL * 32 == VS
    KD = H * K                       # one qkv block (q | k | v) in channels
    LOG_K = int(math.log2(K))
    if gated:
        # The fused gated RMSNorm reduces r^2 over ALL V output channels of a
        # (token, head), and warp 0 must own that whole row.
        assert VSPLITS == 1, "fused gated RMSNorm needs the full V row in one CTA"
        assert SPT == 1, "fused gated RMSNorm needs the warp-0 store path"
        assert VS == V and CPT * LV == V
    if fuse_conv:
        assert LK == 1, "the fused conv's q/k channel map assumes LK == 1"

    @cute.kernel
    def blackwell_bf16_kda_save_ssm_kernel(
        mQ: cute.Tensor,      # [TT*H, K]          bf16  (row = t*H + h)
        mK: cute.Tensor,      # [TT*H, K]          bf16
        mV: cute.Tensor,      # [TT*H, V]          bf16
        mG: cute.Tensor,      # [TT*H, K]          fp32  (log decay, <= 0)
        mBeta: cute.Tensor,   # [TT*H]             fp32
        mS0: cute.Tensor,     # [B*H, K, V/4, 4]   fp32  (READ-ONLY in-state)
        mSnap: cute.Tensor,   # [TT*H, K, V/4, 4]  fp32  (post-token state, out)
        mO: cute.Tensor,      # [TT*H, V]          bf16  (output; POST-norm if gated)
        mCu: cute.Tensor,     # [B+1]              int32
        mZ: cute.Tensor,      # [TT*H, V]          bf16  (output gate; gated only)
        mW: cute.Tensor,      # [V]                fp32  (RMSNorm weight; gated only)
        mX: cute.Tensor,      # [TT, 3*H*K]        bf16  (raw packed qkv; conv only)
        mCS: cute.Tensor,     # [B, 3*H*K, 3]      bf16  (raw conv history; conv only)
        mCWt: cute.Tensor,    # [3*H*K, 4]         bf16  (conv taps; conv only)
        mCWin: cute.Tensor,   # [TT, 3*H*K, 3]     bf16  (window snapshots out; conv only)
    ):
        tid, _, _ = cute.arch.thread_idx()
        vsi, h, b = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        w = cute.arch.warp_idx()
        lv = lane & (LV - 1)
        lk = lane >> LOG_LV

        kbase = w * (KPT * LK) + lk * KPT      # first key channel of this thread
        kb4 = kbase >> 2                       # ... in 128-bit fp32 units
        kb8 = kbase >> 3                       # ... in 128-bit fp16 units
        bh = b * H + h
        r = w * LK + lk                        # this thread's partial slot
        # This thread's CPT columns, in 128-bit-group / element-in-group form.
        #   VECST : col(cc) = vsi*VS + CPT*lane + cc      (group cgrp, elem cc)
        #   else  : col(cc) = vsi*VS + lv + LV*cc         (group cg0+dcg*cc, elem cm0)
        cgrp = (vsi * VS >> LOG_CPT) + lane
        cg0 = (vsi * VS + lv) >> LOG_CPT
        cm0 = lv & (CPT - 1)
        dcg = LV >> LOG_CPT

        # ---- the incoming state tile. Issued at the very top: HBM latency
        # overlaps the prologue (see S0_AFTER_PROLOGUE for the rejected
        # alternative). NB: the load is written out inline in both positions
        # rather than through a helper -- CuTeDSL rejects a closure that
        # captures constexpr values when it is called from inside dynamic
        # control flow. 128-bit staging register for the VECST path: the vector
        # copy is done against the 4-element `tv` fragment and the values are
        # moved to/from `st` scalar-wise, so `st` itself is only ever touched
        # by scalar accesses and keeps its register promotion.
        st = cute.make_fragment((CPT, KPT), FP32)
        tv = cute.make_fragment((CPT,), FP32)

        if cutlass.const_expr(not S0_AFTER_PROLOGUE):
            if cutlass.const_expr(VECST):
                for i in cutlass.range_constexpr(KPT):
                    cute.autovec_copy(mS0[bh, kbase + i, cgrp, None], tv)
                    for cc in cutlass.range_constexpr(CPT):
                        st[cc, i] = tv[cc]
            else:
                for i in cutlass.range_constexpr(KPT):
                    for cc in cutlass.range_constexpr(CPT):
                        st[cc, i] = mS0[bh, kbase + i, cg0 + dcg * cc, cm0]

        # ---- the RMSNorm weight, hoisted to the very top. It is consumed in
        # the last handful of instructions of every token, where there is
        # nothing left to hide an L2 round-trip behind; issuing it here costs
        # ZPL registers and is amortised over every token of the request.
        fw = cute.make_fragment((ZPL,), FP32)
        if cutlass.const_expr(gated):
            for e in cutlass.range_constexpr(ZPL):
                fw[e] = mW[vsi * VS + lane + 32 * e]

        t0 = mCu[b]
        tn = mCu[b + 1] - t0

        # ---- shared memory staging -----------------------------------------
        # EVERY layout here is given an EXPLICIT row-major stride: `cute.make_
        # layout(shape)` defaults to LayoutLeft (FIRST mode contiguous), which
        # would put the lane-varying index on strided words -- only
        # 32/gcd(TMAX,32) = 4 distinct banks, an 8-way conflict.
        smem = SmemAllocator()
        # per-token channel vectors; at LK=1 these reads are warp-uniform
        sDG = smem.allocate_tensor(FP32, cute.make_layout(
            (TMAX, K // VEC, VEC), stride=(K, VEC, 1)), 16)
        # kn/qn are staged in fp16: they enter the recurrence only through the
        # per-token reductions, where a 4.9e-4 relative error is bounded and
        # self-correcting (the delta rule S <- (I - beta kn kn^T) S + ... is a
        # contraction). dg deliberately stays fp32: it multiplies the WHOLE state
        # every token, so a bias there compounds geometrically over a serving
        # loop that feeds the committed snapshot back in as S0.
        sKN = smem.allocate_tensor(F16, cute.make_layout(
            (TMAX, K // VECH, VECH), stride=(K, VECH, 1)), 16)
        sQN = smem.allocate_tensor(F16, cute.make_layout(
            (TMAX, K // VECH, VECH), stride=(K, VECH, 1)), 16)
        sV = smem.allocate_tensor(FP32, cute.make_layout(
            (TMAX, VS), stride=(VS, 1)))                                # v slice
        sSC = smem.allocate_tensor(FP32, cute.make_layout(
            (TMAX, 2), stride=(2, 1)))                       # beta, <kn,qn>
        sR = smem.allocate_tensor(FP32, cute.make_layout(
            (R, RSTRIDE), stride=(RSTRIDE, 1)), 16)
        NU = 1 if SPT == 2 else 2
        sU = smem.allocate_tensor(FP32, cute.make_layout(
            (NU, VS), stride=(VS, 1)))                                  # u (, dq)
        # w[c] * sigmoid(z[t, c]), materialised by the prologue so the epilogue
        # is a pure SMEM read (gated only).
        if cutlass.const_expr(gated):
            sZG = smem.allocate_tensor(FP32, cute.make_layout(
                (TMAX, VS), stride=(VS, 1)))
        # ---- fused-conv planes, hoisted into SMEM ONCE per CTA (conv only) --
        # sCW: this head's 3*4*128 taps, tap-major so the prologue's per-tap
        # read has the lane-varying channel contiguous (conflict-free), and
        # bf16 NOT fp32: SMEM is a hard occupancy cliff here -- 2 CTAs/SM need
        # shared_mem_per_block_allocated <= 32768 in the 64 KB carveout.
        # sCS: this request's 3 RAW history rows for this head, staged next to
        # the S0 burst so the scattered [B, D, 3] channel-strided read (6
        # sectors per request where 2 would do) leaves the per-token dependent
        # chain entirely.  2304 B; together with sCW's 3072 B the allocation
        # stays under the 32768 B 2-CTA cliff.
        if cutlass.const_expr(fuse_conv):
            sCW = smem.allocate_tensor(BF16, cute.make_layout(
                (NTI, CONV_W, K), stride=(CONV_W * K, K, 1)), 16)
            sCS = smem.allocate_tensor(BF16, cute.make_layout(
                (NTI, CSL, K), stride=(CSL * K, K, 1)), 16)

        qf = cute.make_fragment((CPL,), FP32)
        kf = cute.make_fragment((CPL,), FP32)
        gf = cute.make_fragment((CPL,), FP32)
        dg4 = cute.make_fragment((VEC,), FP32)
        qnf = cute.make_fragment((VECH, KPT // VECH), F16)
        knf = cute.make_fragment((VECH, KPT // VECH), F16)
        pk = cute.make_fragment((CPT,), FP32)
        pq = cute.make_fragment((CPT,), FP32)
        uu = cute.make_fragment((CPT,), FP32)
        rv = cute.make_fragment((CPT,), FP32)
        # fused-conv scratch: the CLAMPED raw window of this lane's CPL
        # channels, ALL loads issued unconditionally before any is consumed (a
        # fragment written under an `scf.if` is demoted to local memory by
        # ptxas -- measured 24K/49K local sectors before this form).
        if cutlass.const_expr(fuse_conv):
            rfg = cute.make_fragment((CONV_W, CPL), BF16)

        # Reduction storage: the 2*VS slots are dealt out to the THREADS threads,
        # SPT each; flat slot f = s*THREADS + tid -> which = f>>LOG_VS,
        # pos = f & (VS-1). A thread's own slots are pos(cc) = cc*LV + lv, which
        # is conflict-free for the scatter and consecutive for the gather. The V
        # column that slot `pos` belongs to is `rcol` below.
        red_pos = tid & (VS - 1)
        if cutlass.const_expr(VECST):
            rcol = ((red_pos & (LV - 1)) << LOG_CPT) + (red_pos >> LOG_LV)
        else:
            rcol = red_pos

        # ---- stage conv_state + conv taps into SMEM, once per CTA ----------
        # One thread per local channel: the channel's CSL history taps are CSL
        # contiguous bf16 (the warp covers 192 B with no gaps) and its CONV_W
        # taps are one 64-bit read.  The landing fragments are written
        # UNCONDITIONALLY (index clamped, only the SMEM store guarded) because
        # a fragment defined inside an scf.if is demoted to local memory.
        # Issued next to the S0 burst, where the in-flight LDGs hide it.
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
                    pti = lcs >> LOG_K
                    pd = lcs & (K - 1)
                    for m_ in cutlass.range_constexpr(CSL):
                        sCS[pti, m_, pd] = csv[m_, it]
                    for jj in cutlass.range_constexpr(CONV_W):
                        sCW[pti, jj, pd] = cwv[jj, it]
            cute.arch.sync_threads()

        # ---- conv_win plumbing (conv, TPW == 1 only) ------------------------
        # conv_win is written as DENSE 16 B stores at the kernel tail, never
        # as the naive 3-scalars-per-channel scatter (which touched 3x the
        # sectors the payload needs).  An 8-channel unit's 24 bf16 window
        # values are one CONTIGUOUS 48 B block of conv_win ([tok, ch, 3],
        # slot innermost) starting on a 48 B boundary, and the gather
        # fragment is laid out in exactly that order -- so each thread can
        # stream its unit straight to global as three 16 B stores, no SMEM
        # bounce needed.  mCW12 is the 12 x i32 (48 B) row view.
        if cutlass.const_expr(fuse_conv and TPW == 1):
            NCH8 = (3 * K) >> 3            # 8-channel units per token
            XG8 = (3 * KD) >> 3            # 8-channel units per packed row
            mCW12 = cute.make_tensor(
                cute.recast_ptr(mCWin.iterator, dtype=I32),
                cute.make_layout((1 << 24, 12), stride=(12, 1)))
            mXV8 = cute.make_tensor(mX.iterator, cute.make_layout(
                (1 << 26, 8), stride=(8, 1)))

        nchunk = (tn + (TMAX - 1)) >> LOG_TMAX
        for c in cutlass.range(nchunk):
            cbase = t0 + c * TMAX
            ntok = cutlass.min(tn - c * TMAX, TMAX)

            # ---- SPLIT prologue (conv only): TWO warps per token whenever the
            # chunk leaves warps idle. `sp` is 1 exactly when 2*ntok <= NWARP
            # (min/max: no branch, no divide); warp w stages token w >> sp.
            # The EVEN warp of a pair does q and k -- which MUST share a warp,
            # because ||q||, ||k|| and <q,k> are warp butterflies -- while the
            # ODD warp does v and the gate scale, neither of which needs any
            # cross-lane reduction. That shortens the latency-bound conv chain
            # on the warp the whole CTA waits for at the barrier from 3
            # tensors to 2, at zero extra L1TEX wavefronts (same loads, twice
            # the warps). When 2*ntok > NWARP, sp == 0 == hlf, both halves run
            # on one warp, and the schedule is the pre-split one unchanged.
            # The window taps are PREDICATED on lidx (warp-uniform, so both
            # arms are uniform branches and `rfg` keeps its register
            # promotion), and the conv_win snapshot stores are issued AFTER
            # each tensor's SiLU, so they never sit in the dependent chain.
            if cutlass.const_expr(fuse_conv and TPW == 1):
                sp = cutlass.min(
                    cutlass.max(I32(NWARP) - ntok * 2 + I32(1), I32(0)),
                    I32(1))
                j = w >> sp
                hlf = w & sp
                if j < ntok:
                    row = (cbase + j) * H + h
                    tok = cbase + j        # packed token row
                    pos = c * TMAX + j     # position within the request
                    if hlf == 0:
                        nq = FP32(0.0)
                        nk = FP32(0.0)
                        kq = FP32(0.0)
                        # ---- q: window, conv, SiLU, deferred snapshots -----
                        for i in cutlass.range_constexpr(CONV_W):
                            lidx = pos - (CONV_W - 1) + i
                            if lidx >= 0:
                                for e in cutlass.range_constexpr(CPL):
                                    rfg[i, e] = mX[
                                        t0 + lidx, h * K + lane + 32 * e]
                            else:
                                for e in cutlass.range_constexpr(CPL):
                                    rfg[i, e] = sCS[0, pos + i, lane + 32 * e]
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                acc = acc + sCW[0, i, d].to(FP32) * rfg[
                                    i, e].to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            qf[e] = FP32(acc.to(BF16).to(FP32))
                        # ---- k: same, channels KD + h*K + d ----------------
                        for i in cutlass.range_constexpr(CONV_W):
                            lidx = pos - (CONV_W - 1) + i
                            if lidx >= 0:
                                for e in cutlass.range_constexpr(CPL):
                                    rfg[i, e] = mX[
                                        t0 + lidx, KD + h * K + lane + 32 * e]
                            else:
                                for e in cutlass.range_constexpr(CPL):
                                    rfg[i, e] = sCS[1, pos + i, lane + 32 * e]
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                acc = acc + sCW[1, i, d].to(FP32) * rfg[
                                    i, e].to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            kf[e] = FP32(acc.to(BF16).to(FP32))
                        # ---- g, the three warp reductions, and the publish -
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            gf[e] = mG[row, d]
                            nq = nq + qf[e] * qf[e]
                            nk = nk + kf[e] * kf[e]
                            kq = kq + qf[e] * kf[e]
                        for off in cutlass.range_constexpr(5):
                            nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                            nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                            kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
                        rq = FP32(cute.math.rsqrt(nq, fastmath=True))
                        rk = FP32(cute.math.rsqrt(nk, fastmath=True))
                        rqs = rq * FP32(RSQRT_K)
                        # same index decomposition as the unsplit path below,
                        # for the same measured reason -- do NOT "simplify" it
                        ir = lane & (VEC - 1)
                        for e in cutlass.range_constexpr(CPL):
                            sDG[j, (lane >> 2) + (32 // VEC) * e, ir] = FP32(
                                cute.math.exp(gf[e], fastmath=True))
                            sKN[j, (lane >> 3) + (32 // VECH) * e,
                                lane & (VECH - 1)] = (kf[e] * rk).to(F16)
                            sQN[j, (lane >> 3) + (32 // VECH) * e,
                                lane & (VECH - 1)] = (qf[e] * rqs).to(F16)
                        if lane == 0:
                            sSC[j, 0] = mBeta[row]
                            sSC[j, 1] = kq * (rk * rqs)
                    if hlf == sp:
                        # ---- v: conv straight into sV, plus the gate scale -
                        for i in cutlass.range_constexpr(CONV_W):
                            lidx = pos - (CONV_W - 1) + i
                            if lidx >= 0:
                                for e in cutlass.range_constexpr(VS // 32):
                                    rfg[i, e] = mX[
                                        t0 + lidx,
                                        2 * KD + h * K + vsi * VS
                                        + lane + 32 * e]
                            else:
                                for e in cutlass.range_constexpr(VS // 32):
                                    rfg[i, e] = sCS[
                                        2, pos + i, vsi * VS + lane + 32 * e]
                        for e in cutlass.range_constexpr(VS // 32):
                            cidx = lane + 32 * e
                            vd = vsi * VS + cidx
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                acc = acc + sCW[2, i, vd].to(FP32) * rfg[
                                    i, e].to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            sV[j, cidx] = FP32(acc.to(BF16).to(FP32))
                        if cutlass.const_expr(gated):
                            for e in cutlass.range_constexpr(ZPL):
                                cidx = lane + 32 * e
                                zf = mZ[row, vsi * VS + cidx].to(FP32)
                                sZG[j, cidx] = fw[e] * cute.arch.rcp_approx(
                                    FP32(1.0) + cute.math.exp(
                                        FP32(0.0) - zf, fastmath=True))

            # ================= prologue: warp w stages token j = w + rr*NWARP ====
            for rr in cutlass.range_constexpr(
                    0 if (fuse_conv and TPW == 1) else TPW):
                j = w + rr * NWARP
                if j < ntok:
                    row = (cbase + j) * H + h
                    if cutlass.const_expr(fuse_conv):
                        tok = cbase + j        # packed token row
                        pos = c * TMAX + j     # position within the request
                    nq = FP32(0.0)
                    nk = FP32(0.0)
                    kq = FP32(0.0)
                    if cutlass.const_expr(fuse_conv):
                        # ---- width-4 causal conv + SiLU, q then k -----------
                        # The window over the verify tokens is a pure function
                        # of the RAW stream: positions pos-3..pos-1 come from
                        # mX when >= 0, else from conv_state col (p+3).  fp32
                        # accumulate, fp32 SiLU, THEN a bf16 round so the
                        # recurrence sees exactly what the separate conv
                        # launch would have handed it.
                        # The window row is CLAMPED (max(p,0)), not predicated:
                        # every global load is unconditional and issues in one
                        # burst, and `rfg` is never written under a branch --
                        # only plain scalars are selected inside the `p < 0`
                        # arm, from the SMEM-staged history (a dynamic SMEM
                        # index is just address arithmetic).
                        for i in cutlass.range_constexpr(CONV_W):
                            rowc = t0 + cutlass.max(
                                pos - (CONV_W - 1) + i, I32(0))
                            for e in cutlass.range_constexpr(CPL):
                                rfg[i, e] = mX[rowc, h * K + lane + 32 * e]
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            cch = h * K + d                       # q channel
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                xb = rfg[i, e]
                                if cutlass.const_expr(i < CONV_W - 1):
                                    if pos - (CONV_W - 1) + i < 0:
                                        xb = sCS[0, pos + i, d]
                                    if cutlass.const_expr(i >= 1):
                                        if vsi == 0:
                                            mCWin[tok, cch, i - 1] = xb
                                else:
                                    if vsi == 0:
                                        mCWin[tok, cch, CONV_W - 2] = xb
                                acc = acc + sCW[0, i, d].to(FP32) * xb.to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            qf[e] = FP32(acc.to(BF16).to(FP32))
                        for i in cutlass.range_constexpr(CONV_W):
                            rowc = t0 + cutlass.max(
                                pos - (CONV_W - 1) + i, I32(0))
                            for e in cutlass.range_constexpr(CPL):
                                rfg[i, e] = mX[rowc, KD + h * K + lane + 32 * e]
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            cch = KD + h * K + d                  # k channel
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                xb = rfg[i, e]
                                if cutlass.const_expr(i < CONV_W - 1):
                                    if pos - (CONV_W - 1) + i < 0:
                                        xb = sCS[1, pos + i, d]
                                    if cutlass.const_expr(i >= 1):
                                        if vsi == 0:
                                            mCWin[tok, cch, i - 1] = xb
                                else:
                                    if vsi == 0:
                                        mCWin[tok, cch, CONV_W - 2] = xb
                                acc = acc + sCW[1, i, d].to(FP32) * xb.to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            kf[e] = FP32(acc.to(BF16).to(FP32))
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            gf[e] = mG[row, d]
                            nq = nq + qf[e] * qf[e]
                            nk = nk + kf[e] * kf[e]
                            kq = kq + qf[e] * kf[e]
                    else:
                        for e in cutlass.range_constexpr(CPL):
                            d = lane + 32 * e
                            qe = mQ[row, d].to(FP32)
                            ke = mK[row, d].to(FP32)
                            qf[e] = qe
                            kf[e] = ke
                            gf[e] = mG[row, d]
                            nq = nq + qe * qe
                            nk = nk + ke * ke
                            kq = kq + qe * ke
                    # one warp owns all K channels of a token -> pure warp reduction
                    for off in cutlass.range_constexpr(5):
                        nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                        nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                        kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
                    rq = FP32(cute.math.rsqrt(nq, fastmath=True))
                    rk = FP32(cute.math.rsqrt(nk, fastmath=True))
                    rqs = rq * FP32(RSQRT_K)
                    ir = lane & (VEC - 1)
                    for e in cutlass.range_constexpr(CPL):
                        i4 = (lane >> 2) + (32 // VEC) * e
                        sDG[j, i4, ir] = FP32(cute.math.exp(gf[e], fastmath=True))
                        sKN[j, (lane >> 3) + (32 // VECH) * e,
                            lane & (VECH - 1)] = (kf[e] * rk).to(F16)
                        sQN[j, (lane >> 3) + (32 // VECH) * e,
                            lane & (VECH - 1)] = (qf[e] * rqs).to(F16)
                    # v slice for this CTA's VS columns, 32 lanes x (VS/32)
                    if cutlass.const_expr(fuse_conv):
                        # v conv, same clamped-window form; each CTA owns a
                        # disjoint v slice so its snapshots are written always
                        for i in cutlass.range_constexpr(CONV_W):
                            rowc = t0 + cutlass.max(
                                pos - (CONV_W - 1) + i, I32(0))
                            for e in cutlass.range_constexpr(VS // 32):
                                rfg[i, e] = mX[
                                    rowc,
                                    2 * KD + h * K + vsi * VS + lane + 32 * e]
                        for e in cutlass.range_constexpr(VS // 32):
                            cidx = lane + 32 * e
                            vd = vsi * VS + cidx
                            cch = 2 * KD + h * K + vd             # v channel
                            acc = FP32(0.0)
                            for i in cutlass.range_constexpr(CONV_W):
                                xb = rfg[i, e]
                                if cutlass.const_expr(i < CONV_W - 1):
                                    if pos - (CONV_W - 1) + i < 0:
                                        xb = sCS[2, pos + i, vd]
                                    if cutlass.const_expr(i >= 1):
                                        mCWin[tok, cch, i - 1] = xb
                                else:
                                    mCWin[tok, cch, CONV_W - 2] = xb
                                acc = acc + sCW[2, i, vd].to(FP32) * xb.to(FP32)
                            acc = acc * cute.arch.rcp_approx(FP32(1.0) + FP32(
                                cute.math.exp(FP32(0.0) - acc, fastmath=True)))
                            sV[j, cidx] = FP32(acc.to(BF16).to(FP32))
                    else:
                        for e in cutlass.range_constexpr(VS // 32):
                            cidx = lane + 32 * e
                            sV[j, cidx] = mV[row, vsi * VS + cidx].to(FP32)
                    # the epilogue's per-column gate scale, staged here so the
                    # epilogue never touches global memory
                    if cutlass.const_expr(gated):
                        for e in cutlass.range_constexpr(ZPL):
                            cidx = lane + 32 * e
                            zf = mZ[row, vsi * VS + cidx].to(FP32)
                            # sigmoid(z) = 1/(1+exp(-z)); rcp.approx.ftz.f32 is
                            # exact to 2^-23 and 1+exp(-z) >= 1 so ftz never bites
                            sZG[j, cidx] = fw[e] * cute.arch.rcp_approx(
                                FP32(1.0) + cute.math.exp(-zf, fastmath=True))
                    if lane == 0:
                        sSC[j, 0] = mBeta[row]
                        sSC[j, 1] = kq * (rk * rqs)
            if cutlass.const_expr(S0_AFTER_PROLOGUE):
                if c == 0:
                    if cutlass.const_expr(VECST):
                        for i in cutlass.range_constexpr(KPT):
                            cute.autovec_copy(mS0[bh, kbase + i, cgrp, None], tv)
                            for cc in cutlass.range_constexpr(CPT):
                                st[cc, i] = tv[cc]
                    else:
                        for i in cutlass.range_constexpr(KPT):
                            for cc in cutlass.range_constexpr(CPT):
                                st[cc, i] = mS0[bh, kbase + i,
                                                cg0 + dcg * cc, cm0]
            cute.arch.sync_threads()

            # ================= sequential recurrence over this chunk's tokens ====
            for j in cutlass.range(ntok):
                for cc in cutlass.range_constexpr(CPT):
                    pk[cc] = FP32(0.0)
                    pq[cc] = FP32(0.0)
                # pass 1: decay + both K-partials, from one pass over the tile
                for e in cutlass.range_constexpr(KPT // VECH):
                    cute.autovec_copy(sKN[j, kb8 + e, None], knf[None, e])
                    cute.autovec_copy(sQN[j, kb8 + e, None], qnf[None, e])
                for e in cutlass.range_constexpr(KPT // VEC):
                    cute.autovec_copy(sDG[j, kb4 + e, None], dg4)
                    for ee in cutlass.range_constexpr(VEC):
                        i = VEC * e + ee
                        dgv = dg4[ee]
                        knv = knf[i % VECH, i // VECH].to(FP32)
                        qnv = qnf[i % VECH, i // VECH].to(FP32)
                        # Blackwell packed f32x2: one FFMA2/FMUL2 per COLUMN PAIR
                        # instead of two scalar ops. The three per-element ops of
                        # the recurrence (decay, and the two dot-product FMAs) all
                        # have a column-pair-invariant multiplier (dgv/knv/qnv are
                        # per-CHANNEL), so each pairs perfectly. Bit-identical to
                        # the scalar form -- same IEEE fp32 ops, rnd=rn.
                        for cwp in cutlass.range_constexpr(CPT // 2):
                            c0 = 2 * cwp
                            c1 = c0 + 1
                            s0, s1 = cute.arch.mul_packed_f32x2(
                                (st[c0, i], st[c1, i]), (dgv, dgv))
                            st[c0, i] = s0
                            st[c1, i] = s1
                            pk[c0], pk[c1] = cute.arch.fma_packed_f32x2(
                                (s0, s1), (knv, knv), (pk[c0], pk[c1]))
                            pq[c0], pq[c1] = cute.arch.fma_packed_f32x2(
                                (s0, s1), (qnv, qnv), (pq[c0], pq[c1]))
                # R-way column reduction (bank-conflict-free scatter and gather)
                for cc in cutlass.range_constexpr(CPT):
                    sR[r, cc * LV + lv] = pk[cc]
                    sR[r, VS + cc * LV + lv] = pq[cc]
                cute.arch.sync_threads()
                srow = (cbase + j) * H + h
                if cutlass.const_expr(SPT == 2):
                    # THREADS == VS, so ONE thread gathers both slots of column
                    # `rcol`: dk (slot 0) and dq (slot 1). It can therefore form
                    # o = dq + u*<kn,qn> itself and store it straight to global --
                    # no dq publish and no dq readback. (gated=False only -- the
                    # RMS reduction cannot be done from one column per thread
                    # without an extra CTA barrier.)
                    ak = sR[0, red_pos]
                    aq = sR[0, VS + red_pos]
                    for x in cutlass.range_constexpr(1, R):
                        ak = ak + sR[x, red_pos]
                        aq = aq + sR[x, VS + red_pos]
                    uc = sSC[j, 0] * (sV[j, rcol] - ak)
                    sU[0, red_pos] = uc
                    mO[srow, vsi * VS + rcol] = (aq + uc * sSC[j, 1]).to(BF16)
                else:
                    which = tid >> LOG_VS
                    off = (which << LOG_VS) + red_pos
                    acc = sR[0, off]
                    for x in cutlass.range_constexpr(1, R):
                        acc = acc + sR[x, off]
                    if which == 0:
                        sU[0, red_pos] = sSC[j, 0] * (sV[j, rcol] - acc)
                    else:
                        sU[1, red_pos] = acc
                cute.arch.sync_threads()
                for cc in cutlass.range_constexpr(CPT):
                    uu[cc] = sU[0, cc * LV + lv]
                if cutlass.const_expr(SPT != 2):
                    # r == 0 (w == 0 and lk == 0) is exactly one group of lanes
                    # covering all VS columns -- no duplicate stores. With
                    # VSPLITS == 1 that is the FULL 128-wide output row, so this
                    # one warp can do the RMS reduction by itself: CPT values per
                    # lane, then a 5-step butterfly. No extra CTA barrier, which
                    # is the whole reason the fused epilogue is ~free.
                    if r == 0:
                        knqn = sSC[j, 1]
                        # slot index `cc*LV+lv` addresses sR/sU; the V column that
                        # slot belongs to is `ocol` (they differ under VECST, where
                        # a thread owns CPT *contiguous* columns).
                        if cutlass.const_expr(gated):
                            ss = FP32(0.0)
                            for cc in cutlass.range_constexpr(CPT):
                                rc = sU[1, cc * LV + lv] + uu[cc] * knqn
                                rv[cc] = rc
                                ss = ss + rc * rc
                            for eo in cutlass.range_constexpr(5):
                                ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << eo)
                            # mean(r^2) is a multiply by 1/V, not a divide
                            sc = FP32(cute.math.rsqrt(
                                ss * FP32(RMEAN) + FP32(EPS), fastmath=True))
                            for cc in cutlass.range_constexpr(CPT):
                                ocol = ((lv << LOG_CPT) + cc if VECST
                                        else cc * LV + lv)
                                mO[srow, vsi * VS + ocol] = (
                                    rv[cc] * sc * sZG[j, ocol]).to(BF16)
                        else:
                            for cc in cutlass.range_constexpr(CPT):
                                ocol = ((lv << LOG_CPT) + cc if VECST
                                        else cc * LV + lv)
                                mO[srow, vsi * VS + ocol] = (
                                    sU[1, cc * LV + lv] + uu[cc] * knqn).to(BF16)
                # pass 2: rank-1 update, and stream the post-token state straight
                # out of the registers into snapshot[t]. The stores are
                # interleaved with the FMAs so the LSU starts draining while the
                # tail of the tile is still being updated; the WAR hazard against
                # the next token's pass 1 is handled in hardware (a store reads
                # its operand register at issue).
                for i in cutlass.range_constexpr(KPT):
                    knv = knf[i % VECH, i // VECH].to(FP32)
                    for cwp in cutlass.range_constexpr(CPT // 2):
                        c0 = 2 * cwp
                        c1 = c0 + 1
                        st[c0, i], st[c1, i] = cute.arch.fma_packed_f32x2(
                            (knv, knv), (uu[c0], uu[c1]), (st[c0, i], st[c1, i]))
                    if cutlass.const_expr(VECST):
                        for cc in cutlass.range_constexpr(CPT):
                            tv[cc] = st[cc, i]
                        cute.autovec_copy(tv,
                                          mSnap[srow, kbase + i, cgrp, None])
                    else:
                        for cc in cutlass.range_constexpr(CPT):
                            mSnap[srow, kbase + i, cg0 + dcg * cc, cm0] = st[cc, i]
            cute.arch.sync_threads()

        # ---- conv_win, a pure coalesced gather/scatter at the kernel tail ---
        # The window snapshot is NOT a conv result: slot i of token pos is the
        # RAW value at request row pos-2+i (or conv_state history when that is
        # negative).  A thread re-reads each of its 8 channels' window rows as
        # one 16 B load (L2-hot -- the conv already read them) into a
        # fragment already in conv_win's element order, then streams the 48 B
        # unit straight out as three 16 B stores (consecutive threads own
        # consecutive units, so warps write dense 1x-sector runs).  Running
        # at the KERNEL TAIL, after the state tile and the reduction buffers
        # die, keeps the phase off the 128-register 2-CTA cliff (measured:
        # inside the chunk loop, or staged from the prologue's window
        # registers, it spilled 49K local sectors).
        if cutlass.const_expr(fuse_conv and TPW == 1):
            wcv = cute.make_fragment((8 * CSL,), BF16)
            wcvI = cute.make_tensor(
                cute.recast_ptr(wcv.iterator, dtype=I32),
                cute.make_layout((4 * CSL,), stride=(1,)))
            wtv = cute.make_fragment((8,), BF16)
            for p in cutlass.range((tn + (TMAX - 1)) >> LOG_TMAX):
                tbase = p * TMAX
                ntk = cutlass.min(tn - tbase, I32(TMAX))
                # one thread = one (token, 8-channel) unit -------------------
                for it in cutlass.range_constexpr(
                        (TMAX * NCH8 + THREADS - 1) // THREADS):
                    flat = tid + it * THREADS
                    if flat < ntk * NCH8:
                        # flat // 48 == (flat >> 4) // 3, magic exact here
                        jl = ((flat >> 4) * 683) >> 11
                        u8 = flat - jl * NCH8
                        ti4 = u8 >> (LOG_K - 3)
                        d8 = (u8 - (ti4 << (LOG_K - 3))) << 3
                        gg = ti4 * (KD >> 3) + ((h * K + d8) >> 3)
                        jj = tbase + jl
                        for i in cutlass.range_constexpr(CSL):
                            # slot i is window TAP i+1: raw row pos-2+i
                            lidx = jj - (CSL - 1) + i
                            if lidx >= 0:
                                cute.autovec_copy(
                                    mXV8[(t0 + lidx) * XG8 + gg, None], wtv)
                                for m in cutlass.range_constexpr(8):
                                    wcv[m * CSL + i] = wtv[m]
                            else:
                                for m in cutlass.range_constexpr(8):
                                    wcv[m * CSL + i] = sCS[
                                        ti4, jj + i + 1, d8 + m]
                        cute.autovec_copy(
                            wcvI,
                            mCW12[(t0 + jj) * (3 * (KD >> 3)) + gg, None])

    # The traced launch body is shared; the per-variant @cute.jit signatures
    # below expose ONLY the pointers that variant reads.  This keeps the
    # per-call host path of the bare entry identical to the pre-merge champion
    # (9 pointer rebinds, not 15): at the launch-bound bench shapes the extra
    # ctypes stores were a measured +10-15%.  Unused tensor views are aliased
    # from live pointers of the same dtype; the compiled kernel never reads
    # them (their loads are compiled out under const_expr).
    def _launch_body(pQ, pK, pV, pG, pBeta, pS0, pSnap, pO, pCu, pZ, pW,
                     pX, pCS, pCWt, pCWin, rows, nbh, nb):
        tokk = cute.make_layout((rows, K), stride=(K, 1))
        tokv = cute.make_layout((rows, V), stride=(V, 1))
        mQ = cute.make_tensor(pQ, tokk)
        mK = cute.make_tensor(pK, tokk)
        mG = cute.make_tensor(pG, tokk)
        mV = cute.make_tensor(pV, tokv)
        mO = cute.make_tensor(pO, tokv)
        mZ = cute.make_tensor(pZ, tokv)
        mW = cute.make_tensor(pW, cute.make_layout((V,), stride=(1,)))
        mBeta = cute.make_tensor(pBeta, cute.make_layout((rows,), stride=(1,)))
        mS0 = cute.make_tensor(pS0, cute.make_layout(
            (nbh, K, V // CPT, CPT), stride=(K * V, V, CPT, 1)))
        mSnap = cute.make_tensor(pSnap, cute.make_layout(
            (rows, K, V // CPT, CPT), stride=(K * V, V, CPT, 1)))
        mCu = cute.make_tensor(pCu, cute.make_layout((nb + 1,), stride=(1,)))
        # conv tensors; the first-mode extents only shape the offset math (row
        # indices used are < TT <= rows), so `rows` is a safe over-extent
        CDIM = 3 * H * K
        mX = cute.make_tensor(pX, cute.make_layout(
            (rows, CDIM), stride=(CDIM, 1)))
        mCS = cute.make_tensor(pCS, cute.make_layout(
            (nb, CDIM, CONV_W - 1), stride=(CDIM * (CONV_W - 1), CONV_W - 1, 1)))
        mCWt = cute.make_tensor(pCWt, cute.make_layout(
            (CDIM, CONV_W), stride=(CONV_W, 1)))
        mCWin = cute.make_tensor(pCWin, cute.make_layout(
            (rows, CDIM, CONV_W - 1), stride=(CDIM * (CONV_W - 1), CONV_W - 1, 1)))
        if fuse_conv:
            # Pin the 2-CTA occupancy: the conv variant sits exactly on the
            # 128-register cliff, and without this ptxas is free to take 130
            # registers and silently halve occupancy (measured: 41 -> 50 us).
            blackwell_bf16_kda_save_ssm_kernel(
                mQ, mK, mV, mG, mBeta, mS0, mSnap, mO, mCu, mZ, mW,
                mX, mCS, mCWt, mCWin,
            ).launch(grid=[VSPLITS, H, nb], block=[THREADS, 1, 1],
                     min_blocks_per_mp=2)
        else:
            blackwell_bf16_kda_save_ssm_kernel(
                mQ, mK, mV, mG, mBeta, mS0, mSnap, mO, mCu, mZ, mW,
                mX, mCS, mCWt, mCWin,
            ).launch(grid=[VSPLITS, H, nb], block=[THREADS, 1, 1])

    if fuse_conv and gated:
        @cute.jit
        def launch(
            pX: cute.Pointer,
            pG: cute.Pointer,
            pBeta: cute.Pointer,
            pS0: cute.Pointer,
            pSnap: cute.Pointer,
            pO: cute.Pointer,
            pCu: cute.Pointer,
            pCS: cute.Pointer,
            pCWt: cute.Pointer,
            pCWin: cute.Pointer,
            pZ: cute.Pointer,
            pW: cute.Pointer,
            rows: cutlass.Int32,   # TT*H
            nbh: cutlass.Int32,    # B*H
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pX, pX, pX, pG, pBeta, pS0, pSnap, pO, pCu, pZ, pW,
                         pX, pCS, pCWt, pCWin, rows, nbh, nb)
    elif fuse_conv:
        @cute.jit
        def launch(
            pX: cute.Pointer,
            pG: cute.Pointer,
            pBeta: cute.Pointer,
            pS0: cute.Pointer,
            pSnap: cute.Pointer,
            pO: cute.Pointer,
            pCu: cute.Pointer,
            pCS: cute.Pointer,
            pCWt: cute.Pointer,
            pCWin: cute.Pointer,
            rows: cutlass.Int32,   # TT*H
            nbh: cutlass.Int32,    # B*H
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pX, pX, pX, pG, pBeta, pS0, pSnap, pO, pCu, pO, pG,
                         pX, pCS, pCWt, pCWin, rows, nbh, nb)
    elif gated:
        @cute.jit
        def launch(
            pQ: cute.Pointer,
            pK: cute.Pointer,
            pV: cute.Pointer,
            pG: cute.Pointer,
            pBeta: cute.Pointer,
            pS0: cute.Pointer,
            pSnap: cute.Pointer,
            pO: cute.Pointer,
            pCu: cute.Pointer,
            pZ: cute.Pointer,
            pW: cute.Pointer,
            rows: cutlass.Int32,   # TT*H
            nbh: cutlass.Int32,    # B*H
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pQ, pK, pV, pG, pBeta, pS0, pSnap, pO, pCu, pZ, pW,
                         pQ, pQ, pQ, pQ, rows, nbh, nb)
    else:
        @cute.jit
        def launch(
            pQ: cute.Pointer,
            pK: cute.Pointer,
            pV: cute.Pointer,
            pG: cute.Pointer,
            pBeta: cute.Pointer,
            pS0: cute.Pointer,
            pSnap: cute.Pointer,
            pO: cute.Pointer,
            pCu: cute.Pointer,
            rows: cutlass.Int32,   # TT*H
            nbh: cutlass.Int32,    # B*H
            nb: cutlass.Int32,     # B
        ):
            _launch_body(pQ, pK, pV, pG, pBeta, pS0, pSnap, pO, pCu, pO, pG,
                         pQ, pQ, pQ, pQ, rows, nbh, nb)

    return launch


# ==================== launch engine + importable API ====================
# The compiled launcher is cached per (H,K,V,gated,fuse_conv); the only
# per-call host work is rebinding the live device pointers, so every call
# reads the current tensors.
import torch
from cutlass.cute.runtime import make_ptr

GMEM = cute.AddressSpace.gmem

PEAK_GBPS = 8000.0          # B200 HBM3e ~8 TB/s
BENCH_H = 96
BENCH_SHAPES = [(4, 4), (64, 2), (64, 4), (64, 8), (256, 4)]
PRIMARY_SHAPE = (64, 4)
# SGLang `verify_save_ssm` (fused_sigmoid_gating_delta_rule_update,
# disable_state_update=True + intermediate_states_buffer) measured on THIS B200
# with identical inputs -- the target-to-beat from the kernel spec. us/iter.
SPEC_BAR_H96 = {
    (4, 4): 43.8,
    (64, 2): 262.8,
    (64, 4): 424.7,
    (64, 8): 761.5,
    (256, 4): 1632.7,
}
SPEC_BAR_H12 = {(64, 4): 220.8}

_CFGS = {
    "row": CFG_ROW,
    "vec": CFG_VEC,
    "w8": CFG_W8,
    "v2": CFG_V2,
    "wide": CFG_WIDE,
}
# bare (gated=False) tile: measured-fastest CFG_V2, env-overridable as before
_CFG = _CFGS[os.environ.get("KDA_CFG", "v2")]
# gated tile: the only LK==1 tile with VSPLITS==1 and SPT==1 (see docstring)
_CFG_GATED = CFG_W8

#  dtype, assumed alignment for the pointer args, PER VARIANT (the launch
#  signature only exposes the pointers that variant reads -- see make_launcher)
#  bare:        q k v g beta S0 snap o cu
_PTR_SPEC_BARE = ((BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16),
                  (FP32, 4), (FP32, 16), (FP32, 16), (BF16, 16), (I32, 4))
#  gated:       bare + z w
_PTR_SPEC_GATED = _PTR_SPEC_BARE + ((BF16, 16), (FP32, 16))
#  conv:        x g beta S0 snap o cu conv_state conv_weight conv_win
_PTR_SPEC_CONV = ((BF16, 16), (FP32, 16), (FP32, 4), (FP32, 16), (FP32, 16),
                  (BF16, 16), (I32, 4), (BF16, 2), (BF16, 8), (BF16, 16))
#  conv gated:  conv + z w
_PTR_SPEC_CONV_GATED = _PTR_SPEC_CONV + ((BF16, 16), (FP32, 16))


def _ptr_spec(gated, fuse_conv):
    if fuse_conv:
        return _PTR_SPEC_CONV_GATED if gated else _PTR_SPEC_CONV
    return _PTR_SPEC_GATED if gated else _PTR_SPEC_BARE


class _Engine:
    """Caches the compiled launcher and the argument-marshalling objects.

    Nothing input-dependent is cached: every call rebinds the live device
    pointers of the tensors it was given (a CuTeDSL `_Pointer` marshals through
    a ctypes descriptor whose address is what the launcher reads, so rebinding
    `_desc.value` is exactly equivalent to rebuilding the object, minus ~7 us of
    per-call Python). The kernel therefore always reads current data.
    """

    def __init__(self):
        self._compiled = {}
        self._args = {}

    def _get_compiled(self, H, K, V, gated, fuse_conv):
        key = (H, K, V, gated, fuse_conv)
        c = self._compiled.get(key)
        if c is None:
            cfg = _CFG_GATED if gated else _CFG
            spec = _ptr_spec(gated, fuse_conv)
            c = cute.compile(
                make_launcher(H, K, V, cfg=cfg, gated=gated,
                              fuse_conv=fuse_conv),
                *[make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in spec],
                I32(0), I32(0), I32(0),
            )
            self._compiled[key] = c
        return c

    def prepare(self, H, K, V, TT, B, tensors, gated, fuse_conv=False):
        key = (H, K, V, TT, B, gated, fuse_conv)
        e = self._args.get(key)
        if e is None:
            for t in tensors:
                if not t.is_contiguous():
                    raise ValueError(
                        "kda_save_ssm requires contiguous inputs "
                        f"(got a non-contiguous {tuple(t.shape)} tensor)")
                if t.data_ptr() % 16:
                    raise ValueError("inputs must be 16-byte aligned")
            spec = _ptr_spec(gated, fuse_conv)
            e = (self._get_compiled(H, K, V, gated, fuse_conv),
                 [make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in spec],
                 (I32(TT * H), I32(B * H), I32(B)))
            self._args[key] = e
        return e


_ENGINE = _Engine()


def _bind_and_launch(compiled, p, scal, tensors):
    for ptr, t in zip(p, tensors):
        ptr._desc.value = t.data_ptr()
    compiled(*p, scal[0], scal[1], scal[2])


def kda_save_ssm(q, k, v, g, beta, S0, cu_seqlens, snapshot):
    """KDA multi-token MTP-verify step, save_ssm scheme.

    q, k, v    : [TT, H, K] bf16      (varlen token packing, contiguous)
    g          : [TT, H, K] fp32      log decay, <= 0
    beta       : [TT, H]    fp32
    S0         : [B, H, K, V] fp32    incoming state, READ-ONLY
    cu_seqlens : [B+1] int32 on device
    snapshot   : [TT, H, K, V] fp32   post-token states, WRITTEN IN PLACE

    returns o : [TT, H, V] bf16
    """
    TT, H, K = q.shape
    B, _, _, V = S0.shape
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    o = torch.empty((TT, H, V), dtype=q.dtype, device=q.device)
    compiled, p, scal = _ENGINE.prepare(
        H, K, V, TT, B, (q, k, v, g, beta, S0, snapshot, cu_seqlens),
        gated=False)
    # rebind the live pointers (hot path is launch-only)
    p[0]._desc.value = q.data_ptr()
    p[1]._desc.value = k.data_ptr()
    p[2]._desc.value = v.data_ptr()
    p[3]._desc.value = g.data_ptr()
    p[4]._desc.value = beta.data_ptr()
    p[5]._desc.value = S0.data_ptr()
    p[6]._desc.value = snapshot.data_ptr()
    p[7]._desc.value = o.data_ptr()
    p[8]._desc.value = cu_seqlens.data_ptr()
    compiled(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
             scal[0], scal[1], scal[2])
    return o


def _launch_gated(q, k, v, g, beta, S0, cu_seqlens, snapshot, z, w, gated):
    TT, H, K = q.shape
    B, _, _, V = S0.shape
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    o = torch.empty((TT, H, V), dtype=q.dtype, device=q.device)
    if not gated:
        # epilogue compiled out == the bare kernel; z/w are never read
        compiled, p, scal = _ENGINE.prepare(
            H, K, V, TT, B, (q, k, v, g, beta, S0, snapshot, cu_seqlens),
            gated=False)
        _bind_and_launch(compiled, p, scal,
                         (q, k, v, g, beta, S0, snapshot, o, cu_seqlens))
        return o
    compiled, p, scal = _ENGINE.prepare(
        H, K, V, TT, B, (q, k, v, g, beta, S0, snapshot, cu_seqlens, z, w),
        gated=True)
    # rebind the live pointers (hot path is launch-only)
    p[0]._desc.value = q.data_ptr()
    p[1]._desc.value = k.data_ptr()
    p[2]._desc.value = v.data_ptr()
    p[3]._desc.value = g.data_ptr()
    p[4]._desc.value = beta.data_ptr()
    p[5]._desc.value = S0.data_ptr()
    p[6]._desc.value = snapshot.data_ptr()
    p[7]._desc.value = o.data_ptr()
    p[8]._desc.value = cu_seqlens.data_ptr()
    p[9]._desc.value = z.data_ptr()
    p[10]._desc.value = w.data_ptr()
    compiled(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
             scal[0], scal[1], scal[2])
    return o


def kda_save_ssm_gated(q, k, v, g, beta, S0, cu_seqlens, snapshot, z, w):
    """KDA multi-token MTP-verify step, save_ssm scheme, fused gated RMSNorm.
    Returns o [TT,H,V] bf16 POST gated-RMSNorm; snapshot written in place."""
    return _launch_gated(q, k, v, g, beta, S0, cu_seqlens, snapshot, z, w, True)


def kda_save_ssm_prenorm(q, k, v, g, beta, S0, cu_seqlens, snapshot, z, w):
    """Ablation: same kernel, epilogue removed (writes PRE-norm r). Timing stand-in
    for 'our kernel un-fused', to be followed by a separate gated-RMSNorm launch."""
    return _launch_gated(q, k, v, g, beta, S0, cu_seqlens, snapshot, z, w, False)


def _check_conv_args(x, g, S0, conv_state, conv_weight, conv_win):
    TT, H, K = g.shape
    B, _, _, V = S0.shape
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    CDIM = 3 * H * K
    assert x.shape == (TT, CDIM) and x.dtype == torch.bfloat16, (
        f"x must be [TT, 3*H*128] bf16 raw packed qkv, got {tuple(x.shape)}")
    assert conv_state.shape == (B, CDIM, CONV_W - 1), (
        f"conv_state must be [B, 3*H*128, 3], got {tuple(conv_state.shape)}")
    assert conv_state.dtype == torch.bfloat16
    assert conv_weight.shape == (CDIM, CONV_W), (
        f"conv_weight must be [3*H*128, 4], got {tuple(conv_weight.shape)}")
    assert conv_weight.dtype == torch.bfloat16
    assert conv_win.shape == (TT, CDIM, CONV_W - 1), (
        f"conv_win must be [TT, 3*H*128, 3], got {tuple(conv_win.shape)}")
    assert conv_win.dtype == torch.bfloat16
    return TT, H, K, B, V


def kda_save_ssm_conv(x, g, beta, S0, cu_seqlens, snapshot,
                      conv_state, conv_weight, conv_win):
    """save_ssm verify with the width-4 causal conv + SiLU fused in.

    x           : [TT, 3*H*128] bf16  RAW pre-conv packed qkv (q | k | v)
    g           : [TT, H, 128] fp32   pre-activated log decay, <= 0
    beta        : [TT, H] fp32
    S0          : [B, H, 128, 128] fp32  incoming state, READ-ONLY
    cu_seqlens  : [B+1] int32 on device
    snapshot    : [TT, H, 128, 128] fp32  post-token states, WRITTEN
    conv_state  : [B, 3*H*128, 3] bf16   raw pre-conv history, READ-ONLY
    conv_weight : [3*H*128, 4] bf16      conv taps (col 3 hits the new token)
    conv_win    : [TT, 3*H*128, 3] bf16  per-token raw window snapshots, WRITTEN

    returns o : [TT, H, 128] bf16 (PRE-norm)
    """
    TT, H, K, B, V = _check_conv_args(x, g, S0, conv_state, conv_weight,
                                      conv_win)
    o = torch.empty((TT, H, V), dtype=torch.bfloat16, device=x.device)
    compiled, p, scal = _ENGINE.prepare(
        H, K, V, TT, B,
        (x, g, beta, S0, snapshot, cu_seqlens, conv_state, conv_weight,
         conv_win),
        gated=False, fuse_conv=True)
    _bind_and_launch(compiled, p, scal,
                     (x, g, beta, S0, snapshot, o, cu_seqlens,
                      conv_state, conv_weight, conv_win))
    return o


def kda_save_ssm_conv_gated(x, g, beta, S0, cu_seqlens, snapshot,
                            conv_state, conv_weight, conv_win, z, w):
    """kda_save_ssm_conv + the fused gated-RMSNorm epilogue (POST-norm o).
    z [TT,H,128] bf16 output gate; w [128] fp32 RMSNorm weight."""
    TT, H, K, B, V = _check_conv_args(x, g, S0, conv_state, conv_weight,
                                      conv_win)
    o = torch.empty((TT, H, V), dtype=torch.bfloat16, device=x.device)
    compiled, p, scal = _ENGINE.prepare(
        H, K, V, TT, B,
        (x, g, beta, S0, snapshot, cu_seqlens, conv_state, conv_weight,
         conv_win, z, w),
        gated=True, fuse_conv=True)
    _bind_and_launch(compiled, p, scal,
                     (x, g, beta, S0, snapshot, o, cu_seqlens,
                      conv_state, conv_weight, conv_win, z, w))
    return o


# ============================ self-test (__main__) ============================
# Conv-fused variants vs the composed reference:
#   SGLang causal_conv1d_update(activation="silu") semantics per token step
#   (torch fp32 conv + fp32 SiLU -> bf16), then THE SAME unfused kernels in
#   this file (kda_save_ssm / kda_save_ssm_gated) on the conv outputs.
# Checks o (max-abs <= 2e-2 bf16, cosine >= 0.999), the cached intermediate
# states, and exact conv-window snapshots; then prices the fused kernel
# against [separate SGLang Triton conv launch + unfused kernel].


def _ref_conv_silu(x, conv_state, weight, cu):
    """Composed conv reference: per request, per position, fp32 conv+SiLU.

    Returns (y [TT, CDIM] bf16, win [TT, CDIM, 3] bf16) — y is what the
    separate conv launch hands the verify kernel; win is the raw window
    after each token.
    """
    TT, CDIM = x.shape
    B = cu.numel() - 1
    y = torch.empty_like(x)
    win = torch.empty(TT, CDIM, CONV_W - 1, dtype=x.dtype, device=x.device)
    wf = weight.float()
    for bi in range(B):
        t0, t1 = int(cu[bi]), int(cu[bi + 1])
        # positions -3..-1 from conv_state (col 0 oldest), then the request
        hist = conv_state[bi].transpose(0, 1).float()       # [3, CDIM]
        xs = torch.cat([hist, x[t0:t1].float()], dim=0)     # [3+T, CDIM]
        for p in range(t1 - t0):
            wnd = xs[p:p + CONV_W]                          # [4, CDIM]
            acc = (wnd * wf.transpose(0, 1)).sum(dim=0)     # fp32
            y[t0 + p] = (acc * torch.sigmoid(acc)).to(torch.bfloat16)
            win[t0 + p] = xs[p + 1:p + CONV_W].transpose(0, 1).to(
                torch.bfloat16)
    return y, win


def _sglang_conv_closure(x, conv_state, weight, B, T):
    """The separate-launch baseline: ONE SGLang Triton causal_conv1d_update
    call over the T-token verify window with per-step window snapshots on
    (the production spec-decode configuration this kernel replaces)."""
    import sys
    sys.path.insert(0, "/workspace/model-performance/yikai/diffusion_inference/"
                       "sglang/python")
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )
    TT, CDIM = x.shape
    x3 = x.reshape(B, T, CDIM).transpose(1, 2).contiguous()   # [B, CDIM, T]
    state_len = (CONV_W - 1) + (T - 1)
    cs = torch.zeros(B, CDIM, state_len, dtype=x.dtype, device=x.device)
    cs[:, :, :CONV_W - 1] = conv_state
    idx = torch.arange(B, dtype=torch.int32, device=x.device)
    nacc = torch.ones(B, dtype=torch.int32, device=x.device)
    inter = torch.empty(B, T, CDIM, CONV_W - 1, dtype=x.dtype, device=x.device)
    cs_run = cs.clone()

    def run():
        cs_run.copy_(cs)  # the kernel shifts conv_state in place; reset
        return causal_conv1d_update(
            x3, cs_run, weight, activation="silu",
            conv_state_indices=idx, num_accept_tokens=nacc,
            intermediate_conv_window=inter, intermediate_state_indices=idx,
        )

    return run, inter


def _time_fn(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def _selftest():
    torch.manual_seed(0)
    dev = "cuda"
    failures = []
    for (H, B, T) in [(12, 4, 4), (12, 1, 5), (12, 3, 2), (96, 4, 4),
                      (96, 1, 4)]:
        K = V = 128
        CDIM = 3 * H * K
        TT = B * T
        gen = torch.Generator(device=dev).manual_seed(100 + H + B + T)
        x = (0.5 * torch.randn(TT, CDIM, device=dev, generator=gen)).to(
            torch.bfloat16).contiguous()
        conv_state = (0.5 * torch.randn(B, CDIM, CONV_W - 1, device=dev,
                                        generator=gen)).to(
            torch.bfloat16).contiguous()
        conv_weight = (0.5 * torch.randn(CDIM, CONV_W, device=dev,
                                         generator=gen)).to(
            torch.bfloat16).contiguous()
        g = (-0.5 * torch.rand(TT, H, K, device=dev, generator=gen) - 0.05
             ).contiguous()
        beta = torch.sigmoid(torch.randn(TT, H, device=dev, generator=gen)
                             ).contiguous()
        S0 = (0.1 * torch.randn(B, H, K, V, device=dev, generator=gen)
              ).contiguous()
        cu = torch.arange(0, TT + 1, T, dtype=torch.int32, device=dev)
        z = (0.1 * torch.randn(TT, H, V, device=dev, generator=gen)).to(
            torch.bfloat16).contiguous()
        w = torch.randn(V, device=dev, generator=gen).contiguous()

        # composed reference: conv (SGLang semantics) then THIS FILE's
        # unfused kernels on the conv outputs
        y_ref, win_ref = _ref_conv_silu(x, conv_state, conv_weight, cu)
        qr = y_ref[:, :H * K].reshape(TT, H, K).contiguous()
        kr = y_ref[:, H * K:2 * H * K].reshape(TT, H, K).contiguous()
        vr = y_ref[:, 2 * H * K:].reshape(TT, H, V).contiguous()
        snap_ref = torch.empty(TT, H, K, V, device=dev)
        o_ref = kda_save_ssm(qr, kr, vr, g, beta, S0, cu, snap_ref)
        snap_ref_g = torch.empty(TT, H, K, V, device=dev)
        o_ref_g = kda_save_ssm_gated(qr, kr, vr, g, beta, S0, cu, snap_ref_g,
                                     z, w)

        # fused kernels
        snap = torch.empty(TT, H, K, V, device=dev)
        conv_win = torch.empty(TT, CDIM, CONV_W - 1, dtype=torch.bfloat16,
                               device=dev)
        o = kda_save_ssm_conv(x, g, beta, S0, cu, snap, conv_state,
                              conv_weight, conv_win)
        snap_g = torch.empty(TT, H, K, V, device=dev)
        conv_win_g = torch.empty(TT, CDIM, CONV_W - 1, dtype=torch.bfloat16,
                                 device=dev)
        o_g = kda_save_ssm_conv_gated(x, g, beta, S0, cu, snap_g, conv_state,
                                      conv_weight, conv_win_g, z, w)

        def report(name, out, ref, atol, cos_min=0.999, exact=False):
            d = (out.float() - ref.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.float().flatten(), dim=0).item()
            ok = (d == 0.0) if exact else (d <= atol and cos >= cos_min)
            tag = "PASS" if ok else "FAIL"
            print(f"  {tag} H={H} B={B} T={T} {name:<14s} "
                  f"max_abs={d:.3e} cosine={cos:.6f}")
            if not ok:
                failures.append((H, B, T, name, d, cos))

        report("o", o, o_ref, 2e-2)
        report("snapshot", snap, snap_ref, 2e-2)
        report("conv_win", conv_win, win_ref, 0.0, exact=True)
        report("o_gated", o_g, o_ref_g, 2e-2)
        report("snapshot_g", snap_g, snap_ref_g, 2e-2)
        report("conv_win_g", conv_win_g, win_ref, 0.0, exact=True)

        # cross-check the conv reference itself against the real SGLang Triton
        # kernel (the semantic reference named in the task), when importable
        try:
            run_conv, inter = _sglang_conv_closure(x, conv_state, conv_weight,
                                                   B, T)
            y_tri = run_conv().transpose(1, 2).reshape(TT, CDIM)
            report("sglang conv y", y_tri, y_ref, 2e-2)
            report("sglang conv win",
                   inter.reshape(TT, CDIM, CONV_W - 1), win_ref, 0.0,
                   exact=True)
        except Exception as exc:  # sglang unavailable: torch ref stands alone
            print(f"  (sglang Triton conv cross-check skipped: "
                  f"{type(exc).__name__}: {exc})")
            run_conv = None

        # ---- latency: fused vs [separate conv launch + unfused kernel] -----
        def run_fused():
            return kda_save_ssm_conv(x, g, beta, S0, cu, snap, conv_state,
                                     conv_weight, conv_win)

        def run_unfused():
            return kda_save_ssm(qr, kr, vr, g, beta, S0, cu, snap_ref)

        us_fused = _time_fn(run_fused)
        us_unfused = _time_fn(run_unfused)
        line = (f"  latency H={H} B={B} T={T}: fused={us_fused:.2f}us  "
                f"unfused_verify={us_unfused:.2f}us")
        if run_conv is not None:
            us_conv = _time_fn(run_conv)

            def run_sep():
                run_conv()
                run_unfused()

            us_sep = _time_fn(run_sep)
            line += (f"  sglang_conv={us_conv:.2f}us  "
                     f"conv+verify={us_sep:.2f}us  "
                     f"fused_saves={us_sep - us_fused:.2f}us "
                     f"({us_sep / us_fused:.2f}x)")
        print(line)

    if failures:
        print(f"\nSELF-TEST FAILED: {failures}")
        raise SystemExit(1)
    print("\nSELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
