"""CuTeDSL fused KDA multi-token recurrence — SMALL-T ONLY (short prefill).

IDENTICAL TO (same end-to-end function, just faster — SMALL-T only):
  - SGLang Triton ``fused_sigmoid_gating_delta_rule_update`` run with T>1
    (sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py)
  - vLLM Triton ``fused_recurrent_kda``
    (vllm/model_executor/layers/fla/ops/kda.py, T-loop + cu_seqlens)
Same sequential per-token recurrence and varlen packing. Difference: this
kernel takes PRE-ACTIVATED fp32 ``g``. NOT a chunk-prefill / FlashKDA
replacement — seq<=1024 only; for long T use ``b10_kda_chunk_prefill_cutedsl``.

DROP-IN API? NO — packing and state layout differ from SGLang/vLLM recurrent.
  Triton: fused_sigmoid_...(raw gate, [1,TT,H,D], state [slots,H,V,K], indices)
           or vLLM fused_recurrent_kda(..., cu_seqlens, ssm_state_indices)
  This:   kda_spec_decode(q, k, v, g, beta, S, cu_seqlens) — [TT,H,D] packed,
           pre-activated g, S [B,H,K,V] in place.
  To swap: use the adapter in kda_prefill_register._b10_kda_recurrent.


ONLY use this for tiny sequences (registered in kda/kda_prefill_register.py as
the seq<=1024-gated `b10_kda_recurrent` prefill backend). It processes tokens
strictly sequentially with a register-resident state, so it has no chunk-level
parallelism: measured on B200 it only ties FlashKDA at S=32 and loses from
S=64 on (results/bench_kda_short_prefill.csv). For real prefill lengths use
b10_kda_prefill_triton.py / FlashKDA.

History: merged verbatim from the CuTeDSLGen champion package
  gen_kda_specdec_claude_0728_1735/best/  (kernel module kda_spec_decode.py +
  the run.py launch engine), with CLI/reference/benchmark scaffolding removed.
It was generated for a "spec decode" contract that is now RETIRED — it commits
the state in place and stores nothing for rollback, so it cannot serve MTP
verify (see KDA.md section 2; the real target is kda_specdec_spec.md's
kda_replay_ssm_fused). It also remains agent reference material for the
verify-kernel generation (same state-resident layout and recurrence loop).

Public API — the ONLY thing callers need; treat everything below as opaque:

    from kda.b10.b10_kda_prefill_cutedsl import kda_spec_decode
    o = kda_spec_decode(q, k, v, g, beta, S, cu_seqlens)

    q,k,v : [TT,H,128] bf16     varlen-packed tokens (request n owns
                                cu_seqlens[n]:cu_seqlens[n+1]; per-req len 1..8)
    g     : [TT,H,128] fp32     log-decay (g<=0)
    beta  : [TT,H]     fp32     delta-rule gate in (0,1)
    S     : [B,H,128,128] fp32  recurrent state, UPDATED IN PLACE to the FINAL
                                (after-all-tokens) state, written ONCE
    cu_seqlens : [B+1] int32 on device
    returns o : [TT,H,128] bf16 per-token outputs; q/k L2-norm fused in-kernel.

STATE MODE: this is the CACHE-REPLAY / final-state variant — it writes S once
(traffic ~2*B*H*K*V*4, independent of T). It does NOT store per-step intermediate
states (SGLang target_verify's intermediate_states_buffer); partial-acceptance
rollback is handled by a separate replay of the accepted prefix, not by this call.

B200 sm_100a, CuTeDSL 4.5.2. JIT-compiles on first call; launcher cached per (H,K,V).
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

CPT = 8                  # V columns per thread
VEC = 4                  # fp32 elements per 128-bit access
VECH = 8                 # fp16 elements per 128-bit access
TMAX = 8                 # tokens staged in SMEM per chunk (power of two)
LOG_TMAX = 3

# Tile configuration. (NWARP, LK) fixes everything else:
#   KPT = K/(NWARP*LK) key channels per thread, SR = KPT*CPT state registers,
#   LV  = 32/LK lanes over V, VS = LV*CPT columns per CTA, R = NWARP*LK partials.
# The per-token L1TEX traffic per state element is
#   12/CPT (channel scalars, warp-uniform)  +  20*CPT/SR (reduction, lane-varying)
# and the measured T-scan attributes ~75% of the per-token cost to the second
# term, so doubling SR (KPT 8 -> 16) is the lever. NWARP=2 keeps VS=64 (two V
# splits) so small-B SM fill is unchanged.
CFG_WIDE = (2, 4)        # NWARP=2, LK=4 -> KPT=16, SR=128, VS=64, R=8
CFG_NARROW = (4, 4)      # NWARP=4, LK=4 -> KPT=8,  SR=64,  VS=64, R=16


def make_launcher(H: int, K: int, V: int, cfg=CFG_WIDE):
    """Build a compiled-once launcher closure for a given (H, K, V, cfg)."""
    NWARP, LK = cfg
    THREADS = NWARP * 32
    LV = 32 // LK
    VS = LV * CPT
    KPT = K // (NWARP * LK)          # key channels per thread
    assert KPT * NWARP * LK == K
    assert KPT % VEC == 0 and CPT % VEC == 0
    assert V % VS == 0
    VSPLITS = V // VS
    CPL = K // 32                    # prologue channels per lane (4)
    TPW = TMAX // NWARP              # tokens staged per warp per chunk
    assert TPW * NWARP == TMAX
    RSQRT_K = 1.0 / math.sqrt(K)
    R = NWARP * LK                   # partials per column
    # sR row stride == LV (mod 32) words: the 32 lanes of one store differ by
    # LV words in lk and 1 word in lv, so they cover all 32 banks exactly once.
    RSTRIDE = 2 * VS + LV
    NH = CPT // VEC                  # 128-bit groups of a thread's columns
    LOG_LV = int(math.log2(LV))
    LOG_VS = int(math.log2(VS))
    LOG_VEC = int(math.log2(VEC))
    SPT = (2 * VS) // THREADS        # reduction slots per thread
    assert SPT * THREADS == 2 * VS

    @cute.kernel
    def blackwell_bf16_kda_spec_decode_kernel(
        mQ: cute.Tensor,      # [TT*H, K]        bf16  (row = t*H + h)
        mK: cute.Tensor,      # [TT*H, K]        bf16
        mV: cute.Tensor,      # [TT*H, V/4, 4]   bf16
        mG: cute.Tensor,      # [TT*H, K]        fp32  (log decay, <= 0)
        mBeta: cute.Tensor,   # [TT*H]           fp32
        mS: cute.Tensor,      # [B*H, K, V/4, 4] fp32  (in/out)
        mO: cute.Tensor,      # [TT*H, V/4, 4]   bf16
        mCu: cute.Tensor,     # [B+1]            int32
    ):
        tid, _, _ = cute.arch.thread_idx()
        vsi, h, b = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        w = cute.arch.warp_idx()
        lv = lane & (LV - 1)
        lk = lane >> LOG_LV

        kbase = w * (KPT * LK) + lk * KPT      # first key channel of this thread
        kb4 = kbase >> 2                       # ... in 128-bit units
        # This thread's CPT columns are NH groups of VEC, one per "hh", spread
        # LV groups apart: group(hh) = hh*LV + lv. Spreading them (rather than
        # taking 2*VEC contiguous columns) makes every single 128-bit access
        # fully sector-aligned: the LV lanes of one instruction cover
        # LV*VEC*4 = 128 B contiguous. With contiguous per-thread columns each
        # lane would read 16 B out of every 32 B sector, doubling L1 sector
        # requests -- measured as a 72 us (vs ~17 us roofline) state copy.
        cvec = vsi * (VS // VEC) + lv          # + hh*LV
        bh = b * H + h
        r = w * LK + lk                        # this thread's partial slot

        # ---- issue the state loads first: HBM latency overlaps the prologue
        st = cute.make_fragment((KPT, NH, VEC), FP32)
        for i in cutlass.range_constexpr(KPT):
            for hh in cutlass.range_constexpr(NH):
                cute.autovec_copy(mS[bh, kbase + i, cvec + hh * LV, None],
                                  st[i, hh, None])

        t0 = mCu[b]
        tn = mCu[b + 1] - t0

        # ---- shared memory staging -----------------------------------------
        smem = SmemAllocator()
        # per-token channel vectors, read as 128-bit warp-uniform loads
        sDG = smem.allocate_tensor(FP32, cute.make_layout((TMAX, K // VEC, VEC)), 16)
        # kn/qn are staged in fp16: they enter the recurrence only through the
        # per-token reductions, where a 4.9e-4 relative error is bounded and
        # self-correcting (the delta rule S <- (I - beta kn kn^T) S + ... is a
        # contraction). dg deliberately stays fp32: it multiplies the WHOLE state
        # every token, so a bias there compounds geometrically over a serving
        # loop that carries S across calls -- measured and rejected (see
        # optimization.md cycle 1).
        sKN = smem.allocate_tensor(F16, cute.make_layout((TMAX, K // VECH, VECH)), 16)
        sQN = smem.allocate_tensor(F16, cute.make_layout((TMAX, K // VECH, VECH)), 16)
        sV = smem.allocate_tensor(FP32, cute.make_layout((TMAX, VS)))   # v slice
        sSC = smem.allocate_tensor(FP32, cute.make_layout((TMAX, 2)))   # beta, <kn,qn>
        sR = smem.allocate_tensor(FP32, cute.make_layout((R, RSTRIDE)), 16)
        sU = smem.allocate_tensor(FP32, cute.make_layout(
            (1 if SPT == 2 else 2, VS)))                                # u (, dq)

        qf = cute.make_fragment((CPL,), FP32)
        kf = cute.make_fragment((CPL,), FP32)
        gf = cute.make_fragment((CPL,), FP32)
        dg4 = cute.make_fragment((VEC,), FP32)
        qnf = cute.make_fragment((KPT // VECH, VECH), F16)
        knf = cute.make_fragment((KPT // VECH, VECH), F16)
        pk = cute.make_fragment((CPT,), FP32)
        pq = cute.make_fragment((CPT,), FP32)
        uu = cute.make_fragment((CPT,), FP32)
        ob = cute.make_fragment((NH, VEC), BF16)

        # Reduction step: the 2*VS storage slots are dealt out to the THREADS
        # threads, SPT slots each: flat slot f = s*THREADS + tid, which = f>>LOG_VS,
        # pos = f & (VS-1). pos = cc*LV + lv with cc = hh*VEC + cw, so the
        # column-in-CTA of a slot is (hh*LV + lv)*VEC + cw. Shifts only.
        red_pos = tid & (VS - 1)
        red_cc = red_pos >> LOG_LV
        red_col = (((red_cc >> LOG_VEC) * LV) + (red_pos & (LV - 1))) * VEC + (
            red_cc & (VEC - 1))

        nchunk = (tn + (TMAX - 1)) >> LOG_TMAX
        for c in cutlass.range(nchunk):
            cbase = t0 + c * TMAX
            ntok = cutlass.min(tn - c * TMAX, TMAX)

            # ================= prologue: warp w stages token j = w + rr*NWARP ====
            for rr in cutlass.range_constexpr(TPW):
                j = w + rr * NWARP
                if j < ntok:
                    row = (cbase + j) * H + h
                    nq = FP32(0.0)
                    nk = FP32(0.0)
                    kq = FP32(0.0)
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
                    # v slice for this CTA's VS columns, 32 lanes x 2
                    for e in cutlass.range_constexpr(VS // 32):
                        cidx = lane + 32 * e
                        sV[j, cidx] = mV[row, vsi * (VS // VEC) + (cidx >> 2),
                                         cidx & (VEC - 1)].to(FP32)
                    if lane == 0:
                        sSC[j, 0] = mBeta[row]
                        sSC[j, 1] = kq * (rk * rqs)
            cute.arch.sync_threads()

            # ================= sequential recurrence over this chunk's tokens ====
            for j in cutlass.range(ntok):
                for cc in cutlass.range_constexpr(CPT):
                    pk[cc] = FP32(0.0)
                    pq[cc] = FP32(0.0)
                # pass 1: decay + both K-partials; 8 columns per loaded scalar
                for e in cutlass.range_constexpr(KPT // VECH):
                    cute.autovec_copy(sKN[j, (kbase >> 3) + e, None], knf[e, None])
                    cute.autovec_copy(sQN[j, (kbase >> 3) + e, None], qnf[e, None])
                for e in cutlass.range_constexpr(KPT // VEC):
                    cute.autovec_copy(sDG[j, kb4 + e, None], dg4)
                    for ee in cutlass.range_constexpr(VEC):
                        i = VEC * e + ee
                        dgv = dg4[ee]
                        knv = knf[i // VECH, i % VECH].to(FP32)
                        qnv = qnf[i // VECH, i % VECH].to(FP32)
                        # Blackwell packed f32x2: one FFMA2/FMUL2 per COLUMN PAIR
                        # instead of two scalar ops. The three per-element ops of
                        # the recurrence (decay, and the two dot-product FMAs) all
                        # have a column-pair-invariant multiplier (dgv/knv/qnv are
                        # per-CHANNEL), so each pairs perfectly. Bit-identical to
                        # the scalar form -- same IEEE fp32 ops, rnd=rn.
                        for hh in cutlass.range_constexpr(NH):
                            for cwp in cutlass.range_constexpr(VEC // 2):
                                c0 = 2 * cwp
                                c1 = c0 + 1
                                d0 = hh * VEC + c0
                                d1 = hh * VEC + c1
                                s0, s1 = cute.arch.mul_packed_f32x2(
                                    (st[i, hh, c0], st[i, hh, c1]), (dgv, dgv))
                                st[i, hh, c0] = s0
                                st[i, hh, c1] = s1
                                pk[d0], pk[d1] = cute.arch.fma_packed_f32x2(
                                    (s0, s1), (knv, knv), (pk[d0], pk[d1]))
                                pq[d0], pq[d1] = cute.arch.fma_packed_f32x2(
                                    (s0, s1), (qnv, qnv), (pq[d0], pq[d1]))
                # 16-way column reduction (bank-conflict-free scatter and gather)
                for cc in cutlass.range_constexpr(CPT):
                    sR[r, cc * LV + lv] = pk[cc]
                    sR[r, VS + cc * LV + lv] = pq[cc]
                cute.arch.sync_threads()
                if cutlass.const_expr(SPT == 2):
                    # THREADS == VS, so ONE thread gathers both slots of column
                    # `red_col`: dk (slot 0) and dq (slot 1). It can therefore form
                    # o = dq + u*<kn,qn> itself and store it straight to global --
                    # no dq publish, no dq readback, and the store is still fully
                    # coalesced (a warp's red_col values are a permutation of 32
                    # consecutive columns, i.e. 64 B).
                    ak = sR[0, red_pos]
                    aq = sR[0, VS + red_pos]
                    for x in cutlass.range_constexpr(1, R):
                        ak = ak + sR[x, red_pos]
                        aq = aq + sR[x, VS + red_pos]
                    uc = sSC[j, 0] * (sV[j, red_col] - ak)
                    sU[0, red_pos] = uc
                    mO[(cbase + j) * H + h,
                       (vsi * VS + red_col) >> LOG_VEC,
                       red_col & (VEC - 1)] = (aq + uc * sSC[j, 1]).to(BF16)
                else:
                    which = tid >> LOG_VS
                    off = (which << LOG_VS) + red_pos
                    acc = sR[0, off]
                    for x in cutlass.range_constexpr(1, R):
                        acc = acc + sR[x, off]
                    if which == 0:
                        sU[0, red_pos] = sSC[j, 0] * (sV[j, red_col] - acc)
                    else:
                        sU[1, red_pos] = acc
                cute.arch.sync_threads()
                for cc in cutlass.range_constexpr(CPT):
                    uu[cc] = sU[0, cc * LV + lv]
                if cutlass.const_expr(SPT != 2):
                    if lk == 0:
                        knqn = sSC[j, 1]
                        for cc in cutlass.range_constexpr(CPT):
                            ob[cc // VEC, cc % VEC] = (
                                sU[1, cc * LV + lv] + uu[cc] * knqn).to(BF16)
                        for hh in cutlass.range_constexpr(NH):
                            cute.autovec_copy(
                                ob[hh, None],
                                mO[(cbase + j) * H + h, cvec + hh * LV, None])
                # pass 2: rank-1 update
                for i in cutlass.range_constexpr(KPT):
                    knv = knf[i // VECH, i % VECH].to(FP32)
                    for hh in cutlass.range_constexpr(NH):
                        for cwp in cutlass.range_constexpr(VEC // 2):
                            c0 = 2 * cwp
                            c1 = c0 + 1
                            st[i, hh, c0], st[i, hh, c1] = (
                                cute.arch.fma_packed_f32x2(
                                    (knv, knv),
                                    (uu[hh * VEC + c0], uu[hh * VEC + c1]),
                                    (st[i, hh, c0], st[i, hh, c1])))
            cute.arch.sync_threads()

        # ---- write the state tile back once
        for i in cutlass.range_constexpr(KPT):
            for hh in cutlass.range_constexpr(NH):
                cute.autovec_copy(st[i, hh, None],
                                  mS[bh, kbase + i, cvec + hh * LV, None])

    @cute.jit
    def launch(
        pQ: cute.Pointer,
        pK: cute.Pointer,
        pV: cute.Pointer,
        pG: cute.Pointer,
        pBeta: cute.Pointer,
        pS: cute.Pointer,
        pO: cute.Pointer,
        pCu: cute.Pointer,
        rows: cutlass.Int32,   # TT*H
        nbh: cutlass.Int32,    # B*H
        nb: cutlass.Int32,     # B
    ):
        tok = cute.make_layout((rows, K), stride=(K, 1))
        tok4 = cute.make_layout((rows, V // VEC, VEC), stride=(V, VEC, 1))
        mQ = cute.make_tensor(pQ, tok)
        mK = cute.make_tensor(pK, tok)
        mG = cute.make_tensor(pG, tok)
        mV = cute.make_tensor(pV, tok4)
        mO = cute.make_tensor(pO, tok4)
        mBeta = cute.make_tensor(pBeta, cute.make_layout((rows,), stride=(1,)))
        mS = cute.make_tensor(pS, cute.make_layout(
            (nbh, K, V // VEC, VEC), stride=(K * V, V, VEC, 1)))
        mCu = cute.make_tensor(pCu, cute.make_layout((nb + 1,), stride=(1,)))
        blackwell_bf16_kda_spec_decode_kernel(
            mQ, mK, mV, mG, mBeta, mS, mO, mCu
        ).launch(grid=[VSPLITS, H, nb], block=[THREADS, 1, 1])

    return launch

# ==================== launch engine + importable API ====================
import torch
from cutlass.cute.runtime import make_ptr
GMEM = cute.AddressSpace.gmem

PEAK_GBPS = 8000.0          # B200 HBM3e ~8 TB/s
BENCH_SHAPES = [(4, 4), (64, 2), (64, 4), (64, 8), (256, 4)]
PRIMARY_SHAPE = (64, 4)
# Measured on this B200 with identical inputs (see the kernel spec): us/iter.
SPEC_BAR = {
    (4, 4): {"sglang_split": 19.7, "vllm": 25.0, "sglang_chunk": 118.8},
    (64, 2): {"sglang_split": 35.2, "vllm": 53.7, "sglang_chunk": 161.8},
    (64, 4): {"sglang_split": 53.6, "vllm": 80.7, "sglang_chunk": 163.4},
    (64, 8): {"sglang_split": 90.7, "vllm": 139.5, "sglang_chunk": 168.8},
    (256, 4): {"sglang_split": 214.4, "vllm": 314.6, "sglang_chunk": 515.7},
}


# ---------------------------------------------------------------------------
# Engine: compiled launcher cached per (H, K, V). The only per-call host work is
# the output allocation plus pointer marshalling -- nothing input-dependent is
# cached, so every call reads the live tensors (clean-kernel guide).
# ---------------------------------------------------------------------------
#  dtype, assumed alignment for the 8 pointer arguments (q k v g beta S o cu)
_PTR_SPEC = ((BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16),
             (FP32, 4), (FP32, 16), (BF16, 16), (I32, 4))


class _Engine:
    """Caches the compiled launcher and the argument-marshalling objects.

    Nothing input-dependent is cached: every call rebinds the live device
    pointers of the tensors it was given (a CuTeDSL `_Pointer` marshals through
    a ctypes descriptor whose address is what the launcher reads, so rebinding
    `_desc.value` is exactly equivalent to rebuilding the object, minus ~6 us of
    per-call Python). The kernel therefore always reads current data — see the
    same-object-mutation audit in run_check().
    """

    def __init__(self):
        self._compiled = {}
        self._args = {}

    def _get_compiled(self, H, K, V):
        c = self._compiled.get((H, K, V))
        if c is None:
            c = cute.compile(
                make_launcher(H, K, V),
                *[make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in _PTR_SPEC],
                I32(0), I32(0), I32(0),
            )
            self._compiled[(H, K, V)] = c
        return c

    def prepare(self, H, K, V, TT, B, tensors):
        key = (H, K, V, TT, B)
        e = self._args.get(key)
        if e is None:
            for t in tensors:
                if not t.is_contiguous():
                    raise ValueError(
                        "kda_spec_decode requires contiguous inputs "
                        f"(got a non-contiguous {tuple(t.shape)} tensor)")
                if t.data_ptr() % 16:
                    raise ValueError("inputs must be 16-byte aligned")
            e = (self._get_compiled(H, K, V),
                 [make_ptr(dt, 0, GMEM, assumed_align=al) for dt, al in _PTR_SPEC],
                 (I32(TT * H), I32(B * H), I32(B)))
            self._args[key] = e
        return e


_ENGINE = _Engine()


def kda_spec_decode(q, k, v, g, beta, S, cu_seqlens):
    """KDA multi-token (speculative-decode / MTP-verify) recurrent step.

    q, k, v : [TT, H, K] bf16      (varlen token packing, contiguous)
    g       : [TT, H, K] fp32      log decay, <= 0
    beta    : [TT, H]    fp32
    S       : [B, H, K, V] fp32    recurrent state, UPDATED IN PLACE
    cu_seqlens : [B+1] int32 on device

    returns o : [TT, H, V] bf16
    """
    TT, H, K = q.shape
    B, _, _, V = S.shape
    assert K == 128 and V == 128, (
        f"this CuTeDSL kernel is compiled for head_dim K=V=128, got K={K} V={V}"
    )
    o = torch.empty((TT, H, V), dtype=q.dtype, device=q.device)
    compiled, p, scal = _ENGINE.prepare(H, K, V, TT, B,
                                        (q, k, v, g, beta, S, cu_seqlens))
    # rebind the live pointers (hot path is launch-only)
    p[0]._desc.value = q.data_ptr()
    p[1]._desc.value = k.data_ptr()
    p[2]._desc.value = v.data_ptr()
    p[3]._desc.value = g.data_ptr()
    p[4]._desc.value = beta.data_ptr()
    p[5]._desc.value = S.data_ptr()
    p[6]._desc.value = o.data_ptr()
    p[7]._desc.value = cu_seqlens.data_ptr()
    compiled(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
             scal[0], scal[1], scal[2])
    return o
