"""Fully fused KDA speculative-decode replay-SSM kernel, one CuTeDSL launch.

    causal depthwise conv4 + SiLU  ->  chunk/WY replay-SSM
                                   ->  gated RMSNorm

B200 / sm_100a, CuTeDSL 4.5.2.  See ``design.md`` for the full derivation.

The replay core is the chunk / WY form: every K x V sized operation is a
tensor-core GEMM and the only cross-token dependency is a ``T x T`` triangular
system solved by the exactly-terminating Neumann series.  There is no sequential
``for t`` pass over the state.

Contract: identical ring / checkpoint semantics to TRT-LLM's
``fused_recurrent_gated_delta_rule_cached_replay_update``, with the width-4
causal convolution folded into the prologue (``conv_state`` is READ-ONLY) and
the gated RMSNorm folded into the epilogue.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as _cuda


KDIM = 128
VDIM = 128
HIST = 16
CONVW = 4           # depthwise conv width
CSSEG = 3 * 128     # conv_state elements per q/k/v region of one head
CSVW = 4            # bf16 per lane of the coalesced conv_state staging load
MTILE = 16          # MMA M tile: rows 0..T-1 = q side, rows 8..8+T-1 = k side
MHALF = 8
SPAD = 136          # bf16 smem row stride (conflict-free for the A/B fragments)
SPADV = 136         # f32 row stride of the conv-v transpose tile (8 mod 32)
NITER = VDIM // 32  # N iterations of the (16,32,16) CTA mma tile
KITER = KDIM // 16
KHALF = 64          # k columns staged per S sub-chunk (sweepable)
KSTG = KHALF // 16  # mma k-steps per staged sub-chunk
NKSP = KDIM // KHALF  # k-splits of the mainloop
FPAD = 132          # f32 row stride of the fold staging tile (16*132*4 <= arena)
HOISTJ = 2          # k-steps of sub-chunk 0 hoisted above the serial prologue
EPS = 1e-6          # l2-norm epsilon
NEPS = 1e-5         # rms-norm epsilon


# ----------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------
@cute.kernel
def _kda_replay_ssm_conv_gated_wychunk_kernel(
    mQ: cute.Tensor,      # (B,T,H,K)     bf16   raw q projection (strided OK)
    mK: cute.Tensor,      # (B,T,H,K)     bf16   raw k projection (strided OK)
    mV: cute.Tensor,      # (B,T,H,V)     bf16   raw v projection (strided OK)
    mCw: cute.Tensor,     # (3*H*K, 4)    bf16|f32  depthwise conv weights
    mCs: cute.Tensor,     # (B,3*H*K,3)   bf16   3 previous raw tokens (read-only)
    mGr: cute.Tensor,     # (B,T,H,K) f32
    mBr: cute.Tensor,     # (B,T,H)   f32
    mAlog: cute.Tensor,   # (H,)      f32
    mDtb: cute.Tensor,    # (H,K)     f32
    lower_bound: Float32,
    mS: cute.Tensor,      # (B,H,V,K)        f32
    mOU: cute.Tensor,     # (B,2,HIST,H,V)   bf16
    mOK: cute.Tensor,     # (B,2,HIST,H,K)   bf16
    mOG: cute.Tensor,     # (B,2,H,K,HIST)   f32
    mPnat: cute.Tensor,   # (B,) i32
    mBidx: cute.Tensor,   # (B,) i32
    mZ: cute.Tensor,      # (B,T,H,V) bf16   output gate
    mW: cute.Tensor,      # (V,)      f32    rmsnorm weight
    mO: cute.Tensor,      # (B,T,H,V) bf16   post-norm output
    tiled_mma: cute.TiledMma,   # atom layout (1,4,1): CTA tile (16,32,16)
    tms: cute.TiledMma,         # atom layout (1,1,1): warp tile (16,8,16)
    T: cutlass.Constexpr,
    HH: cutlass.Constexpr,
    STUB: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    h, b, _ = cute.arch.block_idx()

    lane = tidx % 32
    warp = tidx // 32

    # ---------------- shared memory -------------------------------------
    smem = cutlass_utils.SmemAllocator()
    lay_tile = cute.make_layout((MTILE, KDIM), stride=(SPAD, 1))
    sAgs = smem.allocate_tensor(BFloat16, lay_tile, byte_alignment=16)
    # sAng / sKI die at the operator-GEMM barrier, so the S staging tile is
    # aliased on top of them instead of costing another 4.6 KB of occupancy.
    sArena = smem.allocate_tensor(
        BFloat16, cute.make_layout((2 * MTILE * SPAD,), stride=(1,)), byte_alignment=16
    )
    sAng = cute.make_tensor(sArena.iterator, lay_tile)
    sKI = cute.make_tensor((sArena.iterator + MTILE * SPAD).align(16), lay_tile)
    sRKD = smem.allocate_tensor(BFloat16, lay_tile, byte_alignment=16)
    sU = smem.allocate_tensor(BFloat16, lay_tile, byte_alignment=16)
    # sAgs dies with GEMM-1; sD is not written until after GEMM-2.
    sD = cute.make_tensor(sAgs.iterator, lay_tile)
    lay_op = cute.make_layout((MTILE, MTILE), stride=(24, 1))
    sRC = smem.allocate_tensor(BFloat16, lay_op, byte_alignment=16)
    sMC = smem.allocate_tensor(BFloat16, lay_op, byte_alignment=16)
    lay_f = cute.make_layout((MTILE, MTILE), stride=(17, 1))
    sAB = smem.allocate_tensor(Float32, lay_f, byte_alignment=16)
    sGs = smem.allocate_tensor(Float32, cute.make_layout((KDIM,), stride=(1,)), byte_alignment=16)
    sBeta = smem.allocate_tensor(Float32, cute.make_layout((MHALF,), stride=(1,)), byte_alignment=16)
    sRed = smem.allocate_tensor(Float32, cute.make_layout((64,), stride=(1,)), byte_alignment=16)
    # convolved v, token-major and in **fp32**.  The prologue owns one *channel*
    # per thread but the d = V - W stage owns one *token* per lane quad, so v has
    # to be transposed; doing it in smem costs 4352 B and rides an already-present
    # barrier, versus a 4x redundant global re-read of the raw window per token.
    # Unlike the unfused pipeline (whose conv kernel writes a bf16 qkv tensor to
    # HBM) this v never leaves the CTA, so it is kept at full precision -- it was
    # the single largest bf16 term in the post-norm output.  The fp32 row stride
    # 136 is 8 mod 32, which makes each 16-lane half of the LDS.64 tile the 32
    # banks exactly.
    lay_v = cute.make_layout((MHALF, KDIM), stride=(SPADV, 1))
    sVc = smem.allocate_tensor(Float32, lay_v, byte_alignment=16)
    # conv_state staging tile, 3 x 384 bf16 = 2304 B, ALIASED onto sAgs: sAgs is
    # not written until the stacked operand tiles, which is two CTA barriers
    # after the last read here, so the staging costs no shared memory at all.
    sCs = cute.make_tensor(sAgs.iterator, cute.make_layout((3 * CSSEG,), stride=(1,)))

    ch = tidx  # channel owned by this thread

    # ---------------- prefetch the first checkpoint sub-chunk -----------
    # Nothing in the serial prologue feeds this load, so issuing it first keeps
    # the memory pipe busy through the gates / norms / operator GEMMs / solve
    # instead of leaving a bubble every time a wave of CTAs restarts together.
    gS = mS[b, h, None, None]                     # (V, K) f32

    # Sub-chunk st = kh * 4 + ni covers rows [32ni + 8w, +8), cols [64kh, +64).
    # This lane owns row (lane >> 2) -- the same row its MMA B fragment reads --
    # and, for each of the KSTG k-steps, the 4 consecutive columns
    # 16j + 4*(lane & 3) .. +3.  One LDG.128 per k-step, 16 B aligned.
    def _gchunk(st):
        return cute.make_tensor(
            (gS.iterator
             + (32 * (st % NITER) + 8 * warp + (lane >> 2)) * KDIM
             + KHALF * (st // NITER) + 4 * (lane & 3)).align(16),
            cute.make_layout((KSTG, 4), stride=(16, 1)),
        )

    def _gchunk_j(st, j0, nj):
        # the j0 .. j0+nj k-steps of sub-chunk st
        return cute.make_tensor(
            (gS.iterator
             + (32 * (st % NITER) + 8 * warp + (lane >> 2)) * KDIM
             + KHALF * (st // NITER) + 16 * j0 + 4 * (lane & 3)).align(16),
            cute.make_layout((nj, 4), stride=(16, 1)),
        )

    rSf = cute.make_fragment(cute.make_layout((KSTG, 4), stride=(4, 1)), Float32)
    # Only HALF of the first sub-chunk is hoisted above the prologue.  The full
    # 16-register fragment has by far the longest live range in the kernel, so
    # holding 8 instead relieves the phase that sets the register allocation and
    # the other half is issued at the top of the mainloop.
    cute.autovec_copy(
        _gchunk_j(0, 0, HOISTJ),
        cute.make_tensor(rSf.iterator, cute.make_layout((HOISTJ, 4), stride=(4, 1))),
    )

    # ---------------- ring bookkeeping ----------------------------------
    pnat = mPnat[b]
    bidx = mBidx[b]
    overflow = pnat + T > HIST
    half = bidx
    woff = pnat
    if overflow:
        half = 1 - bidx
        woff = Int32(0)

    # ---------------- causal depthwise conv4 + SiLU ---------------------
    # This thread owns channel `ch` of head `h`. q/k/v may be non-contiguous
    # views returned by ``packed.split(..., dim=-1)``; their CuTe tensor layouts
    # preserve the parent row stride. The causal window is `conv_state[.., 0:3]`
    # (tokens -3..-1) followed by this draft's own tokens.
    # The taps are a per-channel function only, so the whole conv is thread-local
    # -- no cross-lane traffic, no smem, and the boundary is a compile-time
    # unrolled index, not a runtime predicate.
    chq = h * KDIM + ch
    OFFK = HH * KDIM
    ALW = 16 if cutlass.const_expr(mCw.element_type is Float32) else 8

    def _taps(cg):
        rW = cute.make_fragment(cute.make_layout((CONVW,), stride=(1,)), mCw.element_type)
        cute.autovec_copy(
            cute.make_tensor(mCw[cg, None].iterator.align(ALW),
                             cute.make_layout((CONVW,), stride=(1,))),
            rW,
        )
        wv = []
        for j in cutlass.range_constexpr(CONVW):
            wv.append(rW[j].to(Float32))
        return wv

    def _silu(x):
        return x * (Float32(1.0)
                    / (Float32(1.0) + cute.math.exp(-x, fastmath=True)))

    alog = cute.math.exp(mAlog[h], fastmath=True)
    dtb = mDtb[h, ch]
    cgq = chq
    cgk = OFFK + chq
    cgv = 2 * OFFK + chq

    # The q / k / v convolutions and the gate are FUSED into one pass.  Run
    # separately they cost three back-to-back global round trips (their loads
    # are mutually independent but a sequential `for t` per tensor gives the
    # scheduler no reason to overlap them), and this kernel is memory-*latency*
    # bound in the prologue: the profile puts `long_scoreboard` at 9.2 warps per
    # issue-active cycle.  Interleaved, the 3 tap vectors + 9 conv_state taps +
    # 3T projections + T gates form ONE batch of independent loads.
    qraw = []
    kraw = []
    gcum = []
    run = Float32(0.0)
    if cutlass.const_expr(STUB & 1):
        # diagnostic only (wrong values, valid timing): raw projections straight
        # through -- no weight load, no conv_state load, no FIR, no SiLU.
        for t in cutlass.range_constexpr(T):
            qraw.append(mQ[b, t, h, ch].to(Float32))
            kraw.append(mK[b, t, h, ch].to(Float32))
            sVc[t, ch] = mV[b, t, h, ch].to(Float32)
            x = alog * (mGr[b, t, h, ch] + dtb)
            run = run + lower_bound * (
                Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=True)))
            gcum.append(run)
    else:
        # -- conv_state is staged through shared memory (design-001).  It is
        # [B, D, 3] with the 3 taps CONTIGUOUS, so the natural channel-indexed
        # read puts lanes 6 B apart: a warp asks for 64 B of useful data and
        # touches 192 B / 6 sectors, nine times per thread.  The CTA's three
        # 384-element q/k/v regions are each contiguous and 16 B aligned, so 96
        # threads stage one region with a single 8 B load (4 sectors per warp,
        # the minimum), and the taps come back out of shared memory.
        rCs = cute.make_fragment(
            cute.make_layout((3, CSVW), stride=(CSVW, 1)), BFloat16)
        stg = tidx < (CSSEG // CSVW)
        if stg:
            for r in cutlass.range_constexpr(3):
                cute.autovec_copy(
                    cute.make_tensor(
                        (mCs[b, r * OFFK + h * KDIM, None].iterator
                         + CSVW * tidx).align(8),
                        cute.make_layout((CSVW,), stride=(1,))),
                    rCs[r, None],
                )
        # every global load of the prologue is issued BEFORE the staging barrier
        # so that the barrier sits inside their shadow rather than in front of
        # them -- the batching is what made this stage cheap in the first place.
        wq = _taps(cgq)
        wk = _taps(cgk)
        wv = _taps(cgv)
        xq = []
        xk = []
        xv = []
        grr = []
        for t in cutlass.range_constexpr(T):
            xq.append(mQ[b, t, h, ch].to(Float32))
            xk.append(mK[b, t, h, ch].to(Float32))
            xv.append(mV[b, t, h, ch].to(Float32))
            grr.append(mGr[b, t, h, ch])
        if stg:
            for r in cutlass.range_constexpr(3):
                cute.autovec_copy(
                    rCs[r, None],
                    cute.make_tensor(
                        (sCs.iterator + r * CSSEG + CSVW * tidx).align(8),
                        cute.make_layout((CSVW,), stride=(1,))),
                )
        cute.arch.barrier()
        hq = []
        hk = []
        hv = []
        for j in cutlass.range_constexpr(CONVW - 1):
            hq.append(sCs[3 * ch + j].to(Float32))
            hk.append(sCs[CSSEG + 3 * ch + j].to(Float32))
            hv.append(sCs[2 * CSSEG + 3 * ch + j].to(Float32))
        for t in cutlass.range_constexpr(T):
            hq.append(xq[t])
            hk.append(xk[t])
            hv.append(xv[t])
            aq = wq[0] * hq[t]
            ak = wk[0] * hk[t]
            av = wv[0] * hv[t]
            for j in cutlass.range_constexpr(1, CONVW):
                aq = aq + wq[j] * hq[t + j]
                ak = ak + wk[j] * hk[t + j]
                av = av + wv[j] * hv[t + j]
            qraw.append(_silu(aq))
            kraw.append(_silu(ak))
            # v leaves the register file immediately, straight into the
            # transpose tile, so it never overlaps the q / k live ranges
            sVc[t, ch] = _silu(av)
            x = alog * (grr[t] + dtb)
            run = run + lower_bound * (
                Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=True)))
            gcum.append(run)

    # ---------------- ring replay operands ------------------------------
    gOG = mOG[b, bidx, h, None, None]        # (K, HIST) f32
    gOK = mOK[b, bidx, None, h, None]        # (HIST, K) bf16
    gOU = mOU[b, bidx, None, h, None]        # (HIST, V) bf16

    # old_G is [K][HIST] with HIST contiguous: read the whole 16-slot row as
    # 4 x 128-bit instead of 16 strided 4-byte loads.
    rG16 = cute.make_fragment(cute.make_layout((HIST,), stride=(1,)), Float32)
    rG16.fill(0.0)
    gGr = cute.make_tensor(gOG[ch, None].iterator.align(16),
                           cute.make_layout((4, 4), stride=(4, 1)))
    rG4 = cute.make_fragment(cute.make_layout((4, 4), stride=(4, 1)), Float32)
    for q4 in cutlass.range_constexpr(HIST // 4):
        if 4 * q4 < pnat:
            cute.autovec_copy(gGr[q4, None], rG4[q4, None])
            for j in cutlass.range_constexpr(4):
                rG16[4 * q4 + j] = rG4[q4, j]

    # g_start = G[pnat-1] comes out of the row we just loaded (a select chain),
    # not a second 4-byte strided load across all 128 channels.
    gstart = Float32(0.0)
    for s in cutlass.range_constexpr(HIST):
        if s == pnat - 1:
            gstart = rG16[s]
    egs = cute.math.exp(gstart, fastmath=True)
    sGs[ch] = egs

    for s in cutlass.range_constexpr(HIST):
        val = Float32(0.0)
        if s < pnat:
            val = gOK[s, ch].to(Float32) * cute.math.exp(
                gstart - rG16[s], fastmath=True
            )
        sRKD[s, ch] = val.to(BFloat16)
    # old_u is pure staging (no per-channel math), so copy it 8 halves at a
    # time: 2 LDG.128 + 2 STS.128 per thread instead of 16 + 16 two-byte ops.
    rU8 = cute.make_fragment(cute.make_layout((8,), stride=(1,)), BFloat16)
    for i in cutlass.range_constexpr(2):
        s_ = (tidx >> 4) + 8 * i
        v_ = 8 * (tidx & 15)
        cute.autovec_copy(
            cute.make_tensor((gOU[s_, None].iterator + v_).align(16),
                             cute.make_layout((8,), stride=(1,))),
            rU8,
        )
        cute.autovec_copy(
            rU8,
            cute.make_tensor((sU[s_, None].iterator + v_).align(16),
                             cute.make_layout((8,), stride=(1,))),
        )

    for t in cutlass.range_constexpr(T):
        sq = cute.arch.warp_reduction_sum(qraw[t] * qraw[t])
        sk = cute.arch.warp_reduction_sum(kraw[t] * kraw[t])
        if lane == 0:
            sRed[warp * (2 * T) + 2 * t] = sq
            sRed[warp * (2 * T) + 2 * t + 1] = sk

    # beta (one scalar per token)
    if tidx < MHALF:
        bv = Float32(0.0)
        if tidx < T:
            bv = Float32(1.0) / (Float32(1.0) + cute.math.exp(-mBr[b, tidx, h], fastmath=True))
        sBeta[tidx] = bv

    cute.arch.barrier()

    # one thread per (token, q/k) folds the four warp partials and publishes the
    # finished rsqrt, so every other thread reads 2T scalars instead of 8T.
    if tidx < 2 * T:
        tot = sRed[tidx] + sRed[2 * T + tidx] + sRed[4 * T + tidx] + sRed[6 * T + tidx]
        sRed[32 + tidx] = cute.math.rsqrt(tot + Float32(EPS), fastmath=True)
    cute.arch.barrier()

    qn = []
    kn = []
    for t in cutlass.range_constexpr(T):
        qn.append(qraw[t] * sRed[32 + 2 * t] * Float32(KDIM ** -0.5))
        # k~ is rounded to bf16 BEFORE the operators are built, not only on its
        # way into the ring.  Every later launch replays this window from the
        # bf16 record in old_k, so keeping the fp32 value here would make this
        # launch inconsistent with its own history -- and it is a measurable
        # accuracy term, since k~ is the operand that drives W, u and o.
        kb = (kraw[t] * sRed[32 + 2 * t + 1]).to(BFloat16)
        kn.append(kb.to(Float32))

    # ---------------- stacked operand tiles -----------------------------
    # Only the *B*-side tiles need zero padding: an unused A row pollutes only
    # the matching (unused) C row, but an unused B row of sKI / sD contributes
    # to every output.  sD aliases sAgs, so it is zeroed after GEMM-1.
    zero_bf = BFloat16(0.0)
    if tidx < MTILE:
        for j in cutlass.range_constexpr(MTILE):
            sMC[tidx, j] = zero_bf

    for t in cutlass.range_constexpr(T):
        lm = cute.math.exp(gcum[t], fastmath=True)
        il = cute.math.exp(-gcum[t], fastmath=True)
        ql = qn[t] * lm
        kl = kn[t] * lm
        sAng[t, ch] = ql.to(BFloat16)
        sAng[MHALF + t, ch] = kl.to(BFloat16)
        sAgs[t, ch] = (ql * egs).to(BFloat16)
        sAgs[MHALF + t, ch] = (kl * egs).to(BFloat16)
        sKI[t, ch] = (kn[t] * il).to(BFloat16)

    cute.arch.barrier()

    # ---------------- ring append of the V-independent records ----------
    gsr = gstart
    if overflow:
        gsr = Float32(0.0)
    for t in cutlass.range_constexpr(T):
        mOK[b, half, woff + t, h, ch] = kn[t].to(BFloat16)
    gGw = mOG[b, half, h, ch, None]
    if cutlass.const_expr(T % 4 == 0):
        # G rows are HIST-contiguous: one 128-bit store per 4 tokens when the
        # write offset is 4-aligned, scalar otherwise.
        if (woff & 3) == 0:
            rGw = cute.make_fragment(cute.make_layout((T,), stride=(1,)), Float32)
            for t in cutlass.range_constexpr(T):
                rGw[t] = gcum[t] + gsr
            cute.autovec_copy(
                rGw,
                cute.make_tensor((gGw.iterator + woff).align(16),
                                 cute.make_layout((T,), stride=(1,))),
            )
        else:
            for t in cutlass.range_constexpr(T):
                gGw[woff + t] = gcum[t] + gsr
    else:
        for t in cutlass.range_constexpr(T):
            gGw[woff + t] = gcum[t] + gsr

    # ---------------- operator GEMMs (warp 0: RC, warp 1: AB) -----------
    thr_s = tms.get_slice(lane)
    if warp == 0:
        pA = thr_s.partition_A(sAng)
        pB = thr_s.partition_B(sRKD)
        rA = tms.make_fragment_A(pA)
        rB = tms.make_fragment_B(pB)
        cute.autovec_copy(pA, rA)
        cute.autovec_copy(pB, rB)
        accRC = cute.make_fragment(tms.partition_shape_C((MTILE, MTILE)), Float32)
        accRC.fill(0.0)
        cute.gemm(tms, accRC, rA, rB, accRC)
        rRCb = cute.make_fragment_like(accRC, BFloat16)
        rRCb.store(accRC.load().to(BFloat16))
        cute.autovec_copy(rRCb, thr_s.partition_C(sRC))
    elif warp == 1:
        pA = thr_s.partition_A(sAng)
        pB = thr_s.partition_B(sKI)
        rA = tms.make_fragment_A(pA)
        rB = tms.make_fragment_B(pB)
        cute.autovec_copy(pA, rA)
        cute.autovec_copy(pB, rB)
        accAB = cute.make_fragment(tms.partition_shape_C((MTILE, MTILE)), Float32)
        accAB.fill(0.0)
        cute.gemm(tms, accAB, rA, rB, accAB)
        cute.autovec_copy(accAB, thr_s.partition_C(sAB))

    cute.arch.barrier()

    # ---------------- Lemma 2: Neumann solve on the T x T system --------
    # Warp 0 alone owns the solve.  Lane l holds the two cells (t = l>>2,
    # s = 2*(l&3)) and (t, s+1); the P / Mt *columns* it contracts against live
    # in lane (r<<2) + (l&3), so every cross-cell exchange is a shuffle inside
    # one warp.  Nothing is published here -- the barrier that already guards sD
    # in the epilogue also publishes sMC, so warps 1-3 walk straight into the S
    # stream and the solve runs in their shadow.
    lt = lane >> 2
    lq = lane & 3
    ls = 2 * lq
    nrow = cute.make_fragment(cute.make_layout((T,), stride=(1,)), Float32)
    if warp == 0:
        nrow.fill(0.0)
        bt = sBeta[lt]
        # N = tril_{-1}(diag(beta) A); N[lt, r] is structurally zero for r >= lt
        # so the contraction only ever needs r < T.
        if lt < T:
            for r in cutlass.range_constexpr(T):
                if r < lt:
                    nrow[r] = sAB[MHALF + lt, r] * bt
        p0 = Float32(0.0)
        p1 = Float32(0.0)
        if lt == ls:
            p0 = Float32(1.0)
        if lt == ls + 1:
            p1 = Float32(1.0)
        m0 = p0
        m1 = p1
        for _j in cutlass.range_constexpr(T - 1):
            a0 = Float32(0.0)
            a1 = Float32(0.0)
            for r in cutlass.range_constexpr(T):
                src = (r << 2) + lq
                a0 = a0 - nrow[r] * cute.arch.shuffle_sync_op(p0, src)
                a1 = a1 - nrow[r] * cute.arch.shuffle_sync_op(p1, src)
            p0 = a0
            p1 = a1
            m0 = m0 + a0
            m1 = m1 + a1
        mt0 = m0 * sBeta[ls]
        mt1 = m1 * sBeta[ls + 1]
        # C = tril_0(B) Mt, contracted the same way
        c0 = Float32(0.0)
        c1 = Float32(0.0)
        for r in cutlass.range_constexpr(T):
            src = (r << 2) + lq
            br = Float32(0.0)
            if lt < T:
                if r <= lt:
                    br = sAB[lt, r]
            c0 = c0 + br * cute.arch.shuffle_sync_op(mt0, src)
            c1 = c1 + br * cute.arch.shuffle_sync_op(mt1, src)
        if lt < T:
            if ls < T:
                sMC[lt, ls] = mt0.to(BFloat16)
                sMC[MHALF + lt, ls] = c0.to(BFloat16)
            if ls + 1 < T:
                sMC[lt, ls + 1] = mt1.to(BFloat16)
                sMC[MHALF + lt, ls + 1] = c1.to(BFloat16)

    # ---------------- GEMM-1: stream the fp32 checkpoint ----------------
    thr = tiled_mma.get_slice(tidx)
    accs = []
    for ni in cutlass.range_constexpr(NITER):
        a = cute.make_fragment(tiled_mma.partition_shape_C((MTILE, 32)), Float32)
        a.fill(0.0)
        accs.append(a)

    # The B fragment is assembled from registers by a 4-lane butterfly instead
    # of a shared-memory round trip: 4 shfl + 2 selects per MMA, and no STS /
    # LDS / __syncwarp anywhere in the mainloop.
    srcA = (lane & 28) + ((lane & 3) >> 1)
    srcB = srcA + 2
    podd = (lane & 1) == 1

    cute.autovec_copy(
        _gchunk_j(0, HOISTJ, KSTG - HOISTJ),
        cute.make_tensor(rSf.iterator + 4 * HOISTJ,
                         cute.make_layout((KSTG - HOISTJ, 4), stride=(4, 1))),
    )
    rSb = cute.make_fragment_like(rSf, BFloat16)
    rP = cute.recast_tensor(rSb, Float32)          # (KSTG, 2) bf16x2 words
    rBb = cute.make_fragment(tiled_mma.partition_shape_B((32, 16)), BFloat16)
    rBp = cute.recast_tensor(rBb, Float32)

    # (k-half outermost keeps only half the A fragment live: 16 registers)
    for kh in cutlass.range_constexpr(NKSP):
        pAgs = thr.partition_A(
            cute.make_tensor(
                (sAgs.iterator + KHALF * kh).align(16),
                cute.make_layout((MTILE, KHALF), stride=(SPAD, 1)),
            )
        )
        rAgs = tiled_mma.make_fragment_A(pAgs)
        cute.autovec_copy(pAgs, rAgs)
        for ni in cutlass.range_constexpr(NITER):
            rSb.store(rSf.load().to(BFloat16))
            if cutlass.const_expr(kh * NITER + ni + 1 < NKSP * NITER):
                cute.autovec_copy(_gchunk(kh * NITER + ni + 1), rSf)
            for j in cutlass.range_constexpr(KSTG):
                a0 = cute.arch.shuffle_sync_op(rP[j, 0], srcA)
                a1 = cute.arch.shuffle_sync_op(rP[j, 1], srcA)
                b0 = cute.arch.shuffle_sync_op(rP[j, 0], srcB)
                b1 = cute.arch.shuffle_sync_op(rP[j, 1], srcB)
                wlo = a0
                whi = b0
                if podd:
                    wlo = a1
                    whi = b1
                rBp[0] = wlo
                rBp[1] = whi
                cute.gemm(
                    tiled_mma, accs[ni],
                    rAgs[None, None, j], rBb[None, None, 0], accs[ni],
                )

    # ---------------- gate factors, prefetched ahead of the epilogue -----
    # w * sigmoid(z) depends on nothing this kernel computes, so fetching it
    # where it is *used* -- at the very end, behind the norm's barrier -- leaves
    # its full global latency exposed on a kernel whose profile already shows
    # long_scoreboard as the dominant stall.  Issued here it has GEMM-2, the two
    # d = V - W barriers, GEMM-3 and the ssq butterfly to hide in, and it costs
    # 8 registers (the sigmoid is folded in immediately so z and w themselves do
    # not stay live).
    rZW = cute.make_fragment(cute.make_layout((NITER, 2), stride=(2, 1)), Float32)
    trow = lane >> 2
    v0 = 8 * warp + 2 * (lane & 3)
    if cutlass.const_expr(not (STUB & 2)):
        if trow < T:
            gZrow = mZ[b, trow, h, None]
            rZ2 = cute.make_fragment(cute.make_layout((2,), stride=(1,)), BFloat16)
            rW2 = cute.make_fragment(cute.make_layout((2,), stride=(1,)), Float32)
            for ni in cutlass.range_constexpr(NITER):
                vv = 32 * ni + v0
                cute.autovec_copy(
                    cute.make_tensor((gZrow.iterator + vv).align(4),
                                     cute.make_layout((2,), stride=(1,))),
                    rZ2,
                )
                cute.autovec_copy(
                    cute.make_tensor((mW.iterator + vv).align(8),
                                     cute.make_layout((2,), stride=(1,))),
                    rW2,
                )
                for p in cutlass.range_constexpr(2):
                    zz = rZ2[p].to(Float32)
                    rZW[ni, p] = rW2[p] * (
                        Float32(1.0)
                        / (Float32(1.0) + cute.math.exp(-zz, fastmath=True)))

    # ---------------- GEMM-2: ring replay correction --------------------
    pRC = thr.partition_A(sRC)
    rRC = tiled_mma.make_fragment_A(pRC)
    cute.autovec_copy(pRC, rRC)
    for ni in cutlass.range_constexpr(NITER):
        sUv = cute.make_tensor(
            sU.iterator + 32 * ni, cute.make_layout((32, MTILE), stride=(1, SPAD))
        )
        pUv = thr.partition_B(sUv)
        rUv = tiled_mma.make_fragment_B(pUv)
        cute.autovec_copy(pUv, rUv)
        cute.gemm(tiled_mma, accs[ni], rRC[None, None, 0], rUv[None, None, 0], accs[ni])

    # ---------------- d = V - W  (v comes from the conv transpose tile) --
    cute.arch.barrier()
    for m in cutlass.range_constexpr(T, MTILE):
        sD[m, ch] = BFloat16(0.0)
    if trow < T:
        rV2 = cute.make_fragment(cute.make_layout((2,), stride=(1,)), Float32)
        for ni in cutlass.range_constexpr(NITER):
            vv = 32 * ni + v0
            cute.autovec_copy(
                cute.make_tensor((sVc[trow, None].iterator + vv).align(8),
                                 cute.make_layout((2,), stride=(1,))),
                rV2,
            )
            for p in cutlass.range_constexpr(2):
                sD[trow, vv + p] = (rV2[p] - accs[ni][2 + p]).to(BFloat16)
    cute.arch.barrier()

    # ---------------- GEMM-3: the stacked solve -------------------------
    pMC = thr.partition_A(sMC)
    rMC = tiled_mma.make_fragment_A(pMC)
    cute.autovec_copy(pMC, rMC)
    acc2 = cute.make_fragment(tiled_mma.partition_shape_C((MTILE, 32)), Float32)
    for ni in cutlass.range_constexpr(NITER):
        sDv = cute.make_tensor(
            sD.iterator + 32 * ni, cute.make_layout((32, MTILE), stride=(1, SPAD))
        )
        pDv = thr.partition_B(sDv)
        rDv = tiled_mma.make_fragment_B(pDv)
        cute.autovec_copy(pDv, rDv)
        acc2.fill(0.0)
        cute.gemm(tiled_mma, acc2, rMC[None, None, 0], rDv[None, None, 0], acc2)
        # the pre-norm output lands back in the O_S half of the accumulator, so
        # holding it across the norm reduction costs no extra registers
        for p in cutlass.range_constexpr(2):
            accs[ni][p] = accs[ni][p] + acc2[2 + p]
        # ------------- the u ring append (acc2 dies next iteration) -----
        if trow < T:
            vv = 32 * ni + v0
            rU2 = cute.make_fragment(cute.make_layout((2,), stride=(1,)), BFloat16)
            for p in cutlass.range_constexpr(2):
                rU2[p] = acc2[p].to(BFloat16)
            cute.autovec_copy(
                rU2,
                cute.make_tensor(
                    (mOU[b, half, woff + trow, h, None].iterator + vv).align(4),
                    cute.make_layout((2,), stride=(1,)),
                ),
            )

    # ---------------- gated RMSNorm epilogue ----------------------------
    # Token `trow` owns all 128 v-channels, spread over the 4 lanes of its quad
    # in each of the 4 warps.  The quad folds by butterfly (no smem), the 4 warp
    # partials go through sRed -- 32 live slots, and sRed's prologue use died at
    # the second barrier.
    rms = Float32(1.0)
    if cutlass.const_expr(not (STUB & 2)):
        ssq = Float32(0.0)
        for ni in cutlass.range_constexpr(NITER):
            for p in cutlass.range_constexpr(2):
                ssq = ssq + accs[ni][p] * accs[ni][p]
        ssq = cute.arch.warp_reduction_sum(ssq, threads_in_group=4)
        if cutlass.const_expr(STUB & 4):
            # diagnostic only (wrong values, valid timing): the quad sum alone,
            # i.e. the cross-warp fold and its CTA barrier deleted.
            rms = cute.math.rsqrt(ssq * Float32(1.0 / VDIM) + Float32(NEPS),
                                  fastmath=True)
        else:
            if lq == 0:
                sRed[4 * trow + warp] = ssq
            cute.arch.barrier()
            tot = (sRed[4 * trow] + sRed[4 * trow + 1]
                   + sRed[4 * trow + 2] + sRed[4 * trow + 3])
            rms = cute.math.rsqrt(tot * Float32(1.0 / VDIM) + Float32(NEPS),
                                  fastmath=True)

    if trow < T:
        gOrow = mO[b, trow, h, None]
        rOb = cute.make_fragment(cute.make_layout((2,), stride=(1,)), BFloat16)
        for ni in cutlass.range_constexpr(NITER):
            vv = 32 * ni + v0
            if cutlass.const_expr(STUB & 2):
                for p in cutlass.range_constexpr(2):
                    rOb[p] = accs[ni][p].to(BFloat16)
            else:
                for p in cutlass.range_constexpr(2):
                    rOb[p] = (accs[ni][p] * rms * rZW[ni, p]).to(BFloat16)
            cute.autovec_copy(
                rOb,
                cute.make_tensor((gOrow.iterator + vv).align(4),
                                 cute.make_layout((2,), stride=(1,))),
            )

    # ---------------- rare fold: commit S_logical -----------------------
    # The GEMM produces the correction in the C-fragment layout, whose (m, n)
    # map makes a direct read-modify-write of S touch 8 rows (512 B apart) per
    # instruction and use 32 B of each, so the correction is staged into smem in
    # fragment order and S is then rewritten in row order, 512 B contiguous per
    # warp instruction.  The staging tile is aliased on top of sArena.
    if overflow:
        cute.arch.barrier()
        sF = cute.make_tensor(
            cute.recast_ptr(sArena.iterator, dtype=Float32),
            cute.make_layout((MTILE, FPAD), stride=(FPAD, 1)),
        )
        # exp(g_start) for this lane's 4 columns -- identical for every row
        rEg = cute.make_fragment(cute.make_layout((4,), stride=(1,)), Float32)
        cute.autovec_copy(
            cute.make_tensor((sGs.iterator + 4 * lane).align(16),
                             cute.make_layout((4,), stride=(1,))),
            rEg,
        )
        accF = cute.make_fragment(tiled_mma.partition_shape_C((MTILE, 32)), Float32)
        rF2 = cute.make_fragment(cute.make_layout((2,), stride=(1,)), Float32)
        rSg = cute.make_fragment(cute.make_layout((4,), stride=(1,)), Float32)
        rFf = cute.make_fragment(cute.make_layout((4,), stride=(1,)), Float32)
        for vb in cutlass.range_constexpr(0, VDIM, MTILE):
            sUsub = cute.make_tensor(
                sU.iterator + vb, cute.make_layout((MTILE, MTILE), stride=(1, SPAD))
            )
            pAu = thr.partition_A(sUsub)
            rAu = tiled_mma.make_fragment_A(pAu)
            cute.autovec_copy(pAu, rAu)
            # --- GEMM, then stage the correction in fragment order
            for ni in cutlass.range_constexpr(NITER):
                sRKDv = cute.make_tensor(
                    sRKD.iterator + 32 * ni,
                    cute.make_layout((32, MTILE), stride=(1, SPAD)),
                )
                pBk = thr.partition_B(sRKDv)
                rBk = tiled_mma.make_fragment_B(pBk)
                cute.autovec_copy(pBk, rBk)
                accF.fill(0.0)
                cute.gemm(tiled_mma, accF, rAu[None, None, 0], rBk[None, None, 0], accF)
                for hm in cutlass.range_constexpr(2):
                    for p in cutlass.range_constexpr(2):
                        rF2[p] = accF[2 * hm + p]
                    cute.autovec_copy(
                        rF2,
                        cute.make_tensor(
                            (sF[trow + MHALF * hm, None].iterator + 32 * ni + v0).align(8),
                            cute.make_layout((2,), stride=(1,)),
                        ),
                    )
            cute.arch.barrier()
            # --- rewrite S in row order: warp w owns rows w, w+4, w+8, w+12 and
            #     each of its 32 lanes takes 4 consecutive columns, so one
            #     instruction is 512 B contiguous in both smem and global.
            for i in cutlass.range_constexpr(4):
                m = warp + 4 * i
                gSr = cute.make_tensor((gS[vb + m, None].iterator + 4 * lane).align(16),
                                       cute.make_layout((4,), stride=(1,)))
                cute.autovec_copy(gSr, rSg)
                cute.autovec_copy(
                    cute.make_tensor((sF[m, None].iterator + 4 * lane).align(16),
                                     cute.make_layout((4,), stride=(1,))),
                    rFf,
                )
                for p in cutlass.range_constexpr(4):
                    rSg[p] = rSg[p] * rEg[p] + rFf[p]
                cute.autovec_copy(rSg, gSr)
            cute.arch.barrier()


# ----------------------------------------------------------------------------
# host-side jit entry
# ----------------------------------------------------------------------------
@cute.jit
def _launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mCw: cute.Tensor,
    mCs: cute.Tensor,
    mGr: cute.Tensor,
    mBr: cute.Tensor,
    mAlog: cute.Tensor,
    mDtb: cute.Tensor,
    lower_bound: Float32,
    mS: cute.Tensor,
    mOU: cute.Tensor,
    mOK: cute.Tensor,
    mOG: cute.Tensor,
    mPnat: cute.Tensor,
    mBidx: cute.Tensor,
    mZ: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    T: cutlass.Constexpr,
    B: cutlass.Constexpr,
    H: cutlass.Constexpr,
    MBP: cutlass.Constexpr,
    STUB: cutlass.Constexpr,
    stream,
):
    op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout((1, 4, 1)))
    tms = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)))
    _kda_replay_ssm_conv_gated_wychunk_kernel(
        mQ, mK, mV, mCw, mCs, mGr, mBr, mAlog, mDtb, lower_bound,
        mS, mOU, mOK, mOG, mPnat, mBidx, mZ, mW, mO,
        tiled_mma, tms, T, H, STUB,
    ).launch(grid=[H, B, 1], block=[128, 1, 1], stream=stream, min_blocks_per_mp=MBP)


# ----------------------------------------------------------------------------
# python API
# ----------------------------------------------------------------------------
_compiled: dict = {}


def _t(x: torch.Tensor):
    return from_dlpack(x, assumed_align=16)


def kda_replay_ssm_conv_gated_wychunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    S: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    pnat: torch.Tensor,
    buf_idx: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    MBP: int = 6,
    STUB: int = 0,
) -> torch.Tensor:
    """One fused conv + WY-chunk replay-SSM + gated-RMSNorm launch.

    Returns post-norm ``o [B, T, H, V]`` **bf16**.  Appends the T new ring
    records in place and writes ``S`` only when the ring overflows.
    ``conv_state`` / ``pnat`` / ``buf_idx`` are never written.

    ``q``, ``k``, and ``v`` are raw pre-convolution ``[B,T,H,128]`` views.
    Their layouts need not be contiguous; views produced by
    ``packed_qkv.split(H * 128, dim=-1)`` are consumed without a copy.
    """
    B, T, H, Kd = q.shape
    assert tuple(k.shape) == (B, T, H, Kd)
    assert tuple(v.shape) == (B, T, H, Kd)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.device == k.device == v.device
    H = g_raw.shape[2]
    assert tuple(g_raw.shape) == (B, T, H, Kd)
    Vd = old_u.shape[-1]
    assert Kd == KDIM and Vd == VDIM, "kernel is specialised for K = V = 128"
    assert old_u.shape[2] == HIST, "kernel is specialised for HIST = 16"
    assert 2 <= T <= 8, "chunk replay-SSM supports T in 2..8"
    assert tuple(conv_weight.shape) == (3 * H * Kd, CONVW)
    assert tuple(conv_state.shape) == (B, 3 * H * Kd, CONVW - 1)
    if out is None:
        out = torch.empty(B, T, H, Vd, device=q.device, dtype=torch.bfloat16)

    dtb = dt_bias.view(H, Kd)
    key = (
        B, T, H, Kd, Vd, MBP, STUB, conv_weight.dtype, q.device.index,
        tuple(q.stride()), tuple(k.stride()), tuple(v.stride()),
    )
    fn = _compiled.get(key)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if fn is None:
        fn = cute.compile(
            _launch,
            _t(q), _t(k), _t(v), _t(conv_weight), _t(conv_state),
            _t(g_raw), _t(beta_raw), _t(A_log), _t(dtb),
            Float32(lower_bound),
            _t(S), _t(old_u), _t(old_k), _t(old_G), _t(pnat), _t(buf_idx),
            _t(z), _t(w), _t(out),
            T, B, H, MBP, STUB, stream,
        )
        _compiled[key] = fn
    fn(
        _t(q), _t(k), _t(v), _t(conv_weight), _t(conv_state),
        _t(g_raw), _t(beta_raw), _t(A_log), _t(dtb),
        Float32(lower_bound),
        _t(S), _t(old_u), _t(old_k), _t(old_G), _t(pnat), _t(buf_idx),
        _t(z), _t(w), _t(out),
        stream,
    )
    return out


def bytes_moved(B: int, T: int, H: int, hist: int = HIST, cw_bytes: int = 4) -> int:
    """HBM bytes for the steady-state (non-overflow) fused launch.

    ``conv_weight`` and ``w`` are counted once (they are independent of B and
    are L2-resident across the grid); everything else is counted per (b, h).
    """
    K = KDIM
    V = VDIM
    return (
        B * H * K * V * 4          # S checkpoint read
        + B * hist * H * V * 2     # old_u read
        + B * hist * H * K * 2     # old_k read
        + B * H * K * hist * 4     # old_G read
        + 3 * B * T * H * K * 2    # mixed_qkv (q, k, v projections)
        + 3 * B * H * K * 3 * 2    # conv_state (3 taps x 3 tensors)
        + 3 * H * K * CONVW * cw_bytes   # conv_weight (once)
        + B * T * H * K * 4        # g_raw
        + B * T * H * 4            # beta_raw
        + B * T * H * V * 2        # z read
        + V * 4                    # rmsnorm weight (once)
        + B * T * H * V * 2        # o write (bf16)
        + 2 * B * T * H * K * 2    # ring append u, k
        + B * H * K * T * 4        # ring append G
    )
