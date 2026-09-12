"""CuTeDSL fused KDA single-token decode — ONE parametrized source, black-box.

One kernel generator (`make_launcher`), two compile-time flags:

    gated     : bool — compile the sigmoid-gated RMSNorm epilogue in (post-norm
                output) instead of the bare pre-norm recurrence output.
    fuse_conv : bool — compile the width-4 causal conv + SiLU on q/k/v in
                (packed pre-conv `mixed_qkv` input, conv_state advanced in
                place) in front of the recurrence. g and beta are NOT convolved.

One specialized kernel is JIT-compiled and cached per flag combination plus
launch shape, exactly like the pre-merge files cached per (B*H).

IDENTICAL TO (same end-to-end function, just faster):
  gated=False, fuse_conv=False (kda_decode_step):
  - SGLang Triton ``fused_sigmoid_gating_delta_rule_update``
    (sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py)
    at T=1 — the ``sglang_kda_split`` decode row
  - upstream FLA Triton ``fused_recurrent_kda_fwd``
    (fla/ops/kda/fused_recurrent.py) — the ``fla_kda_recurrent`` row
  - TRT-LLM Triton ``fused_sigmoid_gating_delta_rule_update`` /
    ``fused_kda_packed_decode`` — the ``trtllm_kda_split`` /
    ``trtllm_kda_packed`` rows
  gated=True, fuse_conv=False (kda_decode_gated):
  the FUSION of these two Triton kernels into ONE launch:
  - SGLang/FLA Triton decode above at T=1
  - SGLang Triton ``FusedRMSNormGated`` / ``layer_norm_fwd_kernel``
    (sglang/srt/layers/attention/fla/fused_norm_gate.py, sigmoid)
  fuse_conv=True (kda_decode_conv_step / kda_decode_conv_gated):
  additionally fuses SGLang's packed Triton ``causal_conv1d_update``
  (sglang/srt/layers/attention/mamba/causal_conv1d_triton.py, width 4,
  activation="silu") in front of the recurrence. gated=True + fuse_conv=True
  covers the same layer slice as TRT-LLM ``fused_kda_decode``
  (modules/mamba/fused_kda_decode.py) — the ``trtllm_kda_fused_decode`` row.
Same recurrence + in-place S update + o. Difference: every variant here takes
PRE-ACTIVATED fp32 log-decay ``g`` (activation cooked untimed in
``kda_decode_register.py``); the Triton rows fuse raw-gate activation.

DROP-IN API? NO — none is a one-line replace for the Triton rows.
  Triton: fused_sigmoid_gating_delta_rule_update(A_log, a, dt_bias, ..., q, k,
           v, b, initial_state_source, initial_state_indices, ...) with raw
           gate and state pool [slots,H,V,K]; TRT fused_kda_decode also takes
           the raw output gate and activates it in-kernel.
  This:   kda_decode_step(q, k, v, g, beta, S) — pre-activated g, post-sigmoid
           beta, dense S [B,H,K,V] updated in place.
          kda_decode_gated(q, k, v, g, beta, S, z, w) — plus PRE-ACTIVATION
           sigmoid gate input z and fp32 RMSNorm weight w; POST-norm output.
          kda_decode_conv_step(mixed_qkv, conv_weight, conv_state, g, beta, S)
          kda_decode_conv_gated(mixed_qkv, conv_weight, conv_state, g, beta,
           S, z, w) — RAW pre-conv packed [q|k|v] input, bf16 conv_state
           [B,3*H*128,3] advanced in place (raw last-3-token window).
  To swap: keep the adapters in kda_decode_register (activate gate, reshape,
           transpose state).


Extracted verbatim (kernels + launch engine + importable API only) from the
CuTeDSLGen champion packages
  CuTeDSLGen/generation/workspaces/gen_kda_decode_claude_r2_0728_0228/best/run.py
  CuTeDSLGen/generation/workspaces/gen_kda_decode_gated_claude_r2_0729_0701/best/run.py
  CuTeDSLGen/generation/workspaces/gen_kda_decode_conv_gated_claude_0730_0014/best/run.py
    (the conv-fused prologue, its issue order, the grid-form dispatch and the
    state-band cache policy of the fuse_conv=True paths)
with the CLI / correctness / benchmarking / torch-reference scaffolding
removed, then merged into one source. The bare decode beats SGLang/vLLM/FLA/
TRT-LLM decode at every batch on B200 (99% of HBM roofline); the gated variant
beats TRT-LLM's fully-fused decode 1.4-5.8x.

Public APIs are split by fused stage across
``b10_kda_decode[_conv][_gated]_cutedsl`` modules. Treat this shared
implementation as private.

    q, k, v : [B, H, 128] bf16       one-token per-head query / key / value
    g       : [B, H, 128] fp32       per-channel log-decay (g <= 0)
    beta    : [B, H]      fp32        delta-rule gate in (0, 1)
    S       : [B, H, 128, 128] fp32   recurrent state, UPDATED IN PLACE
    z       : [B, H, 128] bf16       output-gate logits (sigmoid applied here)
    w       : [128]       fp32        RMSNorm weight
    mixed_qkv   : [B, 3*H*128] bf16   RAW pre-conv packed [q | k | v]
    conv_weight : [3*H*128, 4] bf16   depthwise conv weights, no bias
    conv_state  : [B, 3*H*128, 3] bf16 last 3 RAW tokens, UPDATED IN PLACE

    kda_decode_step       -> o [B,H,128] bf16 = S^T qn / sqrt(K); q/k L2-norm
                             fused in-kernel
    kda_decode_gated      -> POST-norm o = (r*rsqrt(mean(r^2)+1e-5))*w*sigmoid(z)
    kda_decode_conv_*     -> same, after conv output = SiLU(w0*s0+w1*s1+w2*s2
                             +w3*x) per channel (fp32 accumulate + fp32 SiLU;
                             the post-conv q/k/v stay fp32 on-chip, never
                             rounded through bf16 -- strictly MORE accurate
                             than the separate Triton conv kernel, which
                             rounds its output tensor to bf16),
                             new state = [s1, s2, x] raw (bit-exact vs the
                             Triton kernel's in-place update).

B200 sm_100a, CuTeDSL 4.5.2. Needs a live CUDA context; JIT-compiles on first
call and caches the compiled launcher per flag combo + launch shape.
"""
from __future__ import annotations

import ctypes
import math
import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu as cute_nv
from cutlass.cute.runtime import make_ptr
from cutlass.utils import SmemAllocator
import cuda.bindings.driver as cuda

K = 128           # key dim (state rows) == head dim of the RMSNorm
V = 128           # value dim (state cols) == block size (one thread per column)
NWARP = V // 32   # 4
RSQRT_K = 1.0 / math.sqrt(K)
RMEAN = 1.0 / V   # mean over the head dim = multiply by a constant, never divide
EPS = 1.0e-5      # RMSNorm epsilon (matches FusedRMSNormGated default)
UNROLL = int(os.environ.get("KDA_UNROLL", "16"))
# Where the decayed state band lives between the reduction pass and the
# write-back pass. 2 = registers (exact fp32, no staged SMEM tile) is the
# default: it is both the most accurate and, measured, the fastest overall.
#   0: re-read S from HBM (3-unit traffic)   1: SMEM tile (see KDA_SDT)
STAGE = int(os.environ.get("KDA_STAGE", "2"))
VEC = int(os.environ.get("KDA_VEC", "1"))       # 1: float4 row-split kernel; 0: scalar column kernel
SVEC = int(os.environ.get("KDA_SVEC", "1"))     # 1: vectorized SMEM staging access
# Precision of the STAGE=1 SMEM tile. fp16 halves the tile but rounds the
# staged value to ~5e-4 relative, which loses elementwise accuracy on large
# states where the rank-1 update cancels; STAGE=2 avoids the question entirely.
SDT = int(os.environ.get("KDA_SDT", "16"))
STAGE_T = cutlass.Float16 if SDT == 16 else cutlass.Float32
SPAD = int(os.environ.get("KDA_SPAD", "0"))     # sDs row padding, in fp16 elements
# 1: compute u once per column and broadcast it through SMEM (fewer shared
# loads, but adds a CTA barrier). Measured slower and unstable at large grids
# (B=256: 90.1 -> 94-152 us), so the redundant per-row-group combine wins.
UBCAST = int(os.environ.get("KDA_UBCAST", "0"))
RVEC = int(os.environ.get("KDA_RVEC", "1"))      # 1: vector read of the norm partials
# 1: store the cross-warp partials column-major (Vc, NW) so the phase-2 combine
# reads NW contiguous floats per column as vector loads instead of NW scalar
# shared loads. PADW pads the row stride so the vector reads stay
# bank-conflict-free; PCH is the vector width used to walk the NW partials.
# Measured: saves exactly 1 register (64 -> 63, still 4 CTAs/SM) and is 2%
# slower at B=64 -- the phase-2 combine is not the critical path. Default off.
PVEC = int(os.environ.get("KDA_PVEC", "0"))
PADW = int(os.environ.get("KDA_PADW", "2"))
# 1: hold q/k/v/g in registers across the norm barrier; 0: re-read them from
# (L1-hot) global in the publish pass, trading 4*NEL registers for 4 loads.
NHOLD = int(os.environ.get("KDA_NHOLD", "1"))
# 1: issue the whole state band's global loads BEFORE the norm prologue, so the
# prologue's two CTA barriers + shuffle chain overlap with the load latency
# instead of being serialized in front of it. Free in registers (the loads land
# in the band the kernel already holds). STAGE=2 only.
PREFETCH = int(os.environ.get("KDA_PREFETCH", "1"))
# 1: within the prefetch block, issue the q/k/v/g vector loads (the norm
# critical path) BEFORE the deep state prefetch, so they do not queue behind
# NIT state loads. Measured neutral (B=64) to marginally worse (B=256:
# 84.28 -> 84.52) -- ptxas already hoists the four small vector loads. Off.
PFORDER = int(os.environ.get("KDA_PFORDER", "0"))
# 1: contiguous per-thread row band; 0: interleave rows across threads.
ROWMAP = int(os.environ.get("KDA_ROWMAP", "1"))


# ===========================================================================
# SECTION 1 — the bare-decode champion (gated=False, fuse_conv=False):
# column-split (V-split) kernel family + its host fast path. Verbatim from the
# pre-merge b10_kda_decode_cutedsl.py; the plain path must not move.
# ===========================================================================


@cute.kernel
def kda_decode_kernel(
    mQ: cute.Tensor,      # (BH, 128) bf16
    mK: cute.Tensor,      # (BH, 128) bf16
    mV: cute.Tensor,      # (BH, 128) bf16
    mG: cute.Tensor,      # (BH, 128) fp32
    mBeta: cute.Tensor,   # (BH,)     fp32
    mS: cute.Tensor,      # (BH, 128, 128) fp32  (in place)
    mO: cute.Tensor,      # (BH, 128) bf16
):
    tidx, _, _ = cute.arch.thread_idx()
    bh, _, _ = cute.arch.block_idx()

    lane = tidx % 32
    wid = tidx // 32

    smem = SmemAllocator()
    # Staged decayed state column tile, row-major (fixed-i is contiguous -> the
    # per-i column stores across threads are bank-conflict-free). Staged in fp16
    # (on-chip only; HBM traffic stays fp32) to halve SMEM -> ~2x occupancy.
    # fp16 rel-error ~5e-4 << 5e-3 tolerance.
    sDs = smem.allocate_tensor(cutlass.Float16, cute.make_layout((K, V), stride=(V, 1)))
    sEg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sQn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sKn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NWARP, 3)))

    # --- phase 0: load vectors + fused L2 norms + P = sum kn*qn --------------
    qt = mQ[bh, tidx].to(cutlass.Float32)
    kt = mK[bh, tidx].to(cutlass.Float32)
    vt = mV[bh, tidx].to(cutlass.Float32)
    gt = mG[bh, tidx]
    beta = mBeta[bh]

    p_qq = qt * qt
    p_kk = kt * kt
    p_kq = kt * qt
    for off in cutlass.range_constexpr(5):        # warp-shuffle butterfly sum
        sh = 1 << (4 - off)
        p_qq = p_qq + cute.arch.shuffle_sync_bfly(p_qq, sh)
        p_kk = p_kk + cute.arch.shuffle_sync_bfly(p_kk, sh)
        p_kq = p_kq + cute.arch.shuffle_sync_bfly(p_kq, sh)
    if lane == 0:
        sRed[wid, 0] = p_qq
        sRed[wid, 1] = p_kk
        sRed[wid, 2] = p_kq
    cute.arch.sync_threads()

    sum_qq = cutlass.Float32(0.0)
    sum_kk = cutlass.Float32(0.0)
    sum_kq = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(NWARP):
        sum_qq = sum_qq + sRed[w, 0]
        sum_kk = sum_kk + sRed[w, 1]
        sum_kq = sum_kq + sRed[w, 2]

    rq = cute.math.rsqrt(sum_qq, fastmath=True)
    rk = cute.math.rsqrt(sum_kk, fastmath=True)
    P = rq * rk * sum_kq

    sQn[tidx] = qt * rq
    sKn[tidx] = kt * rk
    sEg[tidx] = cute.math.exp(gt, fastmath=True)
    cute.arch.sync_threads()

    # --- phase 1: read S once, decay, stage in SMEM, reduce A_j and B_j ------
    A = cutlass.Float32(0.0)
    B = cutlass.Float32(0.0)
    for i in cutlass.range(K, unroll=UNROLL):
        ds = sEg[i] * mS[bh, i, tidx]
        if cutlass.const_expr(STAGE):
            sDs[i, tidx] = ds.to(cutlass.Float16)
        A = A + ds * sQn[i]
        B = B + ds * sKn[i]

    # --- phase 2: per-column scalars, write o -------------------------------
    u = beta * (vt - B)
    o = (A + u * P) * cutlass.Float32(RSQRT_K)
    mO[bh, tidx] = o.to(cutlass.BFloat16)

    # --- phase 3: rank-1 update, write S once -------------------------------
    for i in cutlass.range(K, unroll=UNROLL):
        if cutlass.const_expr(STAGE):
            ds = sDs[i, tidx].to(cutlass.Float32)
        else:
            ds = sEg[i] * mS[bh, i, tidx]     # re-read S from (L2-resident) HBM
        mS[bh, i, tidx] = ds + sKn[i] * u


# Warps per CTA = the row-split factor. This kernel is per-thread
# memory-level-parallelism bound, not occupancy bound, so FEWER, fatter warps
# win: each holds a longer contiguous row band (NIT = K*LPR/NT) and issues that
# many independent LDG.64s back to back. NW=2 gives NIT=64 (a 128-register
# band, 178 regs/thread total, ~10 warps/SM) and is fastest at every shape once
# PREFETCH hoists those loads ahead of the prologue -- before PREFETCH, NW=2
# overshot and NW=4 was the optimum. NW=8 is clearly worse either way.
# 178 registers do not spill (local_op_ld/st sectors are 0); note the headroom
# to the 255 cap is small, so re-check spills if the band ever grows.
NW = int(os.environ.get("KDA_NW", "2"))
RPW = K // NW           # rows per warp band
# Columns per thread. VW=2 (LDG.64) beats VW=4 (LDG.128) here: the kernel is
# latency-bound, not request-bound, so what pays is the *number* of independent
# loads a thread can have in flight, not the width of each one.
VW = int(os.environ.get("KDA_VW", "2"))
PCH = min(4, NW)        # floats per vector read of the cross-warp partials
NCH = NW // PCH         # vector reads needed to cover the NW partials


@cute.kernel
def kda_decode_kernel_vec(
    mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor,
    mBeta: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor,
):
    # Row-split vectorized variant: NW warps. Warp `wid` owns the RPW-row band
    # [wid*RPW, wid*RPW+RPW); lane `l` owns the 4 contiguous columns [4l, 4l+4)
    # => float4 (128-bit) coalesced global loads/stores. Partial column
    # reductions over the band are combined across the NW warps in SMEM. The
    # fused L2 norms only need the 128 head-dim lanes (warps 0..3).
    tidx, _, _ = cute.arch.thread_idx()
    bh, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wid = tidx // 32
    r0 = wid * RPW          # first row of this warp's band
    c0 = lane * VW          # first column this thread owns

    smem = SmemAllocator()
    sDs = smem.allocate_tensor(cutlass.Float16, cute.make_layout((K, V), stride=(V, 1)))
    sEg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sQn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sKn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(V))
    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NWARP, 3)))
    # cross-warp partials: pA[wid, col], pB[wid, col]
    pA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NW, V), stride=(V, 1)))
    pB = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NW, V), stride=(V, 1)))

    # --- phase 0: fused L2 norms. Only the 128 head-dim lanes (warps 0..NWARP-1)
    # participate; extra warps (NW>NWARP) load a wrapped index and are dropped. --
    beta = mBeta[bh]
    di = tidx % K                       # valid head-dim index for every thread
    qt = mQ[bh, di].to(cutlass.Float32)
    kt = mK[bh, di].to(cutlass.Float32)
    vt = mV[bh, di].to(cutlass.Float32)
    gt = mG[bh, di]
    p_qq = qt * qt
    p_kk = kt * kt
    p_kq = kt * qt
    for off in cutlass.range_constexpr(5):
        sh = 1 << (4 - off)
        p_qq = p_qq + cute.arch.shuffle_sync_bfly(p_qq, sh)
        p_kk = p_kk + cute.arch.shuffle_sync_bfly(p_kk, sh)
        p_kq = p_kq + cute.arch.shuffle_sync_bfly(p_kq, sh)
    if wid < NWARP:
        if lane == 0:
            sRed[wid, 0] = p_qq
            sRed[wid, 1] = p_kk
            sRed[wid, 2] = p_kq
    cute.arch.sync_threads()

    sum_qq = cutlass.Float32(0.0)
    sum_kk = cutlass.Float32(0.0)
    sum_kq = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(NWARP):
        sum_qq = sum_qq + sRed[w, 0]
        sum_kk = sum_kk + sRed[w, 1]
        sum_kq = sum_kq + sRed[w, 2]
    rq = cute.math.rsqrt(sum_qq, fastmath=True)
    rk = cute.math.rsqrt(sum_kk, fastmath=True)
    P = rq * rk * sum_kq

    if wid < NWARP:
        sQn[di] = qt * rq
        sKn[di] = kt * rk
        sEg[di] = cute.math.exp(gt, fastmath=True)
        sV[di] = vt
    cute.arch.sync_threads()

    # --- phase 1: float4 load over this warp's row band, decay, stage, reduce -
    fA = cute.make_fragment(VW, cutlass.Float32)
    fB = cute.make_fragment(VW, cutlass.Float32)
    frg = cute.make_fragment(VW, cutlass.Float32)
    for k in cutlass.range_constexpr(VW):
        fA[k] = cutlass.Float32(0.0)
        fB[k] = cutlass.Float32(0.0)
    for ii in cutlass.range(RPW, unroll=UNROLL):
        i = r0 + ii
        row = mS[bh, i, None]
        cute.autovec_copy(cute.local_tile(row, (VW,), (lane,)), frg)   # 128-bit load
        eg = sEg[i]
        qn = sQn[i]
        kn = sKn[i]
        for k in cutlass.range_constexpr(VW):
            ds = eg * frg[k]
            sDs[i, c0 + k] = ds.to(cutlass.Float16)
            fA[k] = fA[k] + ds * qn
            fB[k] = fB[k] + ds * kn

    for k in cutlass.range_constexpr(VW):
        pA[wid, c0 + k] = fA[k]
        pB[wid, c0 + k] = fB[k]
    cute.arch.sync_threads()

    # --- phase 2: combine 4 warps' partials for this thread's columns ---------
    uf = cute.make_fragment(VW, cutlass.Float32)
    for k in cutlass.range_constexpr(VW):
        col = c0 + k
        a = cutlass.Float32(0.0)
        b = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(NW):
            a = a + pA[w, col]
            b = b + pB[w, col]
        u = beta * (sV[col] - b)
        uf[k] = u
        if wid == 0:
            mO[bh, col] = ((a + u * P) * cutlass.Float32(RSQRT_K)).to(cutlass.BFloat16)

    # --- phase 3: rank-1 update + write S once (float4 store) ----------------
    fo = cute.make_fragment(VW, cutlass.Float32)
    for ii in cutlass.range(RPW, unroll=UNROLL):
        i = r0 + ii
        kn = sKn[i]
        for k in cutlass.range_constexpr(VW):
            fo[k] = sDs[i, c0 + k].to(cutlass.Float32) + kn * uf[k]
        row = mS[bh, i, None]
        cute.autovec_copy(fo, cute.local_tile(row, (VW,), (lane,)))    # 128-bit store


def make_vsplit_launcher(CS):
    """Build the column-split (V-split) launcher for `CS` column splits.

    A CTA owns the full K=128 rows but only `Vc = V/CS` columns, so the grid is
    `(CS, B*H)` -- `CS x` more CTAs, each with `1/CS` of the staged-tile SMEM.
    That fills the SMs at small B *and* raises resident CTAs/SM at the target
    shape (sDs shrinks 32KB -> 32KB/CS).

    Thread map inside the CTA (NT = NW*32 threads):
      LPR = Vc/VW      lanes covering one row's Vc columns (float4 each)
      clane = t % LPR  this thread's column group -> cols c0 = cs*Vc + clane*VW
      crow  = t / LPR  this thread's row group; rows crow, crow+RPI, ...
      RPI = NT/LPR     rows advanced per iteration, NIT = K/RPI iterations
    All of LPR/RPI/NIT are compile-time powers of two, so `%` and `/` become
    shift/and -- no runtime division. The grid is 2D so cs/bh need no division
    either.

    Partial-reduction cost is kept constant in CS: the (32/LPR) row groups that
    share a warp are first combined with an intra-warp butterfly over the
    row-group dimension, so only NW partials per column reach SMEM (same as the
    CS=1 kernel) instead of RPI.
    """
    Vc = V // CS
    LPR = Vc // VW                  # lanes per row (<= 32)
    NT = NW * 32
    RPI = NT // LPR                 # rows covered per iteration
    NIT = K // RPI                  # row iterations per thread
    GPW = 32 // LPR                 # row groups sharing one warp
    NSHF = GPW.bit_length() - 1     # butterfly steps to fold them
    # Phase 0 (the fused L2 norms) has to cover all K head-dim elements with NT
    # threads. When NT >= K the first K threads take one element each and the
    # rest are dropped; when NT < K every thread takes NEL = K/NT elements.
    # Getting this wrong is silent and severe -- NW=2 with the NT>=K-only form
    # covered just half the head dim and produced NaNs -- so it is asserted.
    NTK = min(NT, K)                # threads that carry a head-dim element
    NEL = K // NTK                  # head-dim elements per such thread
    NWP = min(NW, NWARP)            # warps holding norm partials
    assert Vc >= VW and 32 % LPR == 0 and NT % LPR == 0 and K % RPI == 0, (
        f"bad V-split config CS={CS} Vc={Vc} LPR={LPR} RPI={RPI}")
    assert K % NTK == 0 and NTK == NWP * 32, (
        f"bad norm-phase config NW={NW} NT={NT} NTK={NTK} NWP={NWP}")

    @cute.kernel
    def kda_decode_kernel_vsplit(
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor,
        mBeta: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cs, bh, _ = cute.arch.block_idx()
        lane = tidx % 32
        wid = tidx // 32
        clane = tidx % LPR              # column group within the CTA's tile
        crow = tidx // LPR              # row group
        cbase = cs * Vc                 # first global column of this CTA
        ctile = cs * LPR + clane        # float4 tile index within the row (no div)

        smem = SmemAllocator()
        # staged decayed tile: only this CTA's Vc columns (K x Vc, fp16).
        # Row stride is padded by SPAD halves so that the LPR lanes covering one
        # row and the next row group in the same warp do not alias onto the same
        # SMEM banks (a row of Vc=64 halves is exactly 128 B = all 32 banks).
        if cutlass.const_expr(STAGE != 2):
            sDs = smem.allocate_tensor(
                STAGE_T, cute.make_layout((K, Vc), stride=(Vc + SPAD, 1)))
        # Per-row broadcast coefficients. Kept as three separate K-length arrays
        # on purpose: because a thread walks a *contiguous* row band, ptxas
        # merges four consecutive rows of each array into one LDS.128. Packing
        # them into an interleaved (K,2) array and reading each row explicitly
        # blocks that merge and measured 18% slower (see optimization.md).
        sEg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sQn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sKn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(V))
        # (3, NWARP): the NWARP partials of one quantity are contiguous, so the
        # cross-warp norm combine is 3 vector reads instead of 3*NWARP scalars.
        sRed = smem.allocate_tensor(cutlass.Float32,
                                    cute.make_layout((3, NWP), stride=(NWP, 1)))
        if cutlass.const_expr(UBCAST):
            # u broadcast: phase 2 computed once per column (see UBCAST).
            sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout(Vc))
        # Cross-warp partials. PVEC stores them COLUMN-major -- `(Vc, NW)`, i.e.
        # the NW partials of one column are contiguous -- so the phase-2 combine
        # reads them as NW/4 `LDS.128` per column instead of NW scalar `LDS.32`.
        # The row stride is padded to `NW + PADW` floats so the two rows a thread
        # touches (`lcol = clane*VW + k`) land on disjoint bank groups: at
        # `PADW=2` the eight lanes of an `LDS.128` phase start at banks
        # 0,20,8,28,16,4,24,12 and their 4-bank spans tile all 32 banks exactly.
        if cutlass.const_expr(PVEC):
            pA = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((Vc, NW), stride=(NW + PADW, 1)))
            pB = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((Vc, NW), stride=(NW + PADW, 1)))
        else:
            pA = smem.allocate_tensor(cutlass.Float32,
                                      cute.make_layout((NW, Vc), stride=(Vc, 1)))
            pB = smem.allocate_tensor(cutlass.Float32,
                                      cute.make_layout((NW, Vc), stride=(Vc, 1)))

        # --- phase -1: PREFETCH the state band -------------------------------
        # None of the K*Vc state loads depend on the norms, so issue them all
        # BEFORE phase 0. The NIT independent LDG.64s are then in flight while
        # the prologue runs its two CTA barriers, the vector loads and the
        # 15-step shuffle chain, instead of the prologue latency and the load
        # latency being paid back to back. This is free in registers: the loads
        # land directly in the band the kernel already holds (`rb[m]` below is
        # the same storage the decayed value is written back into).
        beta = mBeta[bh]
        dib = tidx % NTK                # this thread's first head-dim index
        HOLDV = cutlass.const_expr(NHOLD or PFORDER)
        if cutlass.const_expr(HOLDV):
            fqv = cute.make_fragment(NEL, cutlass.Float32)
            fkv = cute.make_fragment(NEL, cutlass.Float32)
            fvv = cute.make_fragment(NEL, cutlass.Float32)
            fgv = cute.make_fragment(NEL, cutlass.Float32)

        def _issue_vecs():
            # The q/k/v/g loads sit on the norm critical path -- everything in
            # the CTA waits on them -- so with PFORDER they are issued ahead of
            # the deep state prefetch rather than queueing behind it.
            for t in cutlass.range_constexpr(NEL):
                di = dib + t * NTK
                fqv[t] = mQ[bh, di].to(cutlass.Float32)
                fkv[t] = mK[bh, di].to(cutlass.Float32)
                fvv[t] = mV[bh, di].to(cutlass.Float32)
                fgv[t] = mG[bh, di]

        def _issue_state():
            for m in cutlass.range_constexpr(NIT):
                i = crow * NIT + m if ROWMAP else crow + m * RPI
                cute.autovec_copy(
                    cute.local_tile(mS[bh, i, None], (VW,), (ctile,)), rb[m])

        rb = None
        if cutlass.const_expr(STAGE == 2 and PREFETCH):
            rb = [cute.make_fragment(VW, cutlass.Float32)
                  for _ in range(NIT)]
            if cutlass.const_expr(PFORDER):
                _issue_vecs()
                _issue_state()
            else:
                _issue_state()

        # --- phase 0: fused L2 norms (needs the full 128-element q/k vectors) --
        # Each of the NTK carrier threads owns NEL head-dim elements
        # (di = dib + t*NTK), so the whole K vector is covered for any NW.
        p_qq = cutlass.Float32(0.0)
        p_kk = cutlass.Float32(0.0)
        p_kq = cutlass.Float32(0.0)
        if cutlass.const_expr(rb is not None and PFORDER):
            for t in cutlass.range_constexpr(NEL):      # loads already issued
                # `di` is also bound here so it exists at kernel scope before
                # the dynamic `if wid < NWP` publish pass rebinds it (the 4.5.2
                # tracer rejects first-binding a value inside a dynamic if).
                di = dib + t * NTK
                p_qq = p_qq + fqv[t] * fqv[t]
                p_kk = p_kk + fkv[t] * fkv[t]
                p_kq = p_kq + fkv[t] * fqv[t]
        else:
            for t in cutlass.range_constexpr(NEL):
                di = dib + t * NTK
                qt = mQ[bh, di].to(cutlass.Float32)
                kt = mK[bh, di].to(cutlass.Float32)
                if cutlass.const_expr(HOLDV):
                    fqv[t] = qt
                    fkv[t] = kt
                    fvv[t] = mV[bh, di].to(cutlass.Float32)
                    fgv[t] = mG[bh, di]
                p_qq = p_qq + qt * qt
                p_kk = p_kk + kt * kt
                p_kq = p_kq + kt * qt
        for off in cutlass.range_constexpr(5):
            sh = 1 << (4 - off)
            p_qq = p_qq + cute.arch.shuffle_sync_bfly(p_qq, sh)
            p_kk = p_kk + cute.arch.shuffle_sync_bfly(p_kk, sh)
            p_kq = p_kq + cute.arch.shuffle_sync_bfly(p_kq, sh)
        if wid < NWP:
            if lane == 0:
                sRed[0, wid] = p_qq
                sRed[1, wid] = p_kk
                sRed[2, wid] = p_kq
        cute.arch.sync_threads()

        sum_qq = cutlass.Float32(0.0)
        sum_kk = cutlass.Float32(0.0)
        sum_kq = cutlass.Float32(0.0)
        if cutlass.const_expr(RVEC):
            rf = cute.make_fragment(NWP, cutlass.Float32)
            cute.autovec_copy(sRed[0, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_qq = sum_qq + rf[w]
            cute.autovec_copy(sRed[1, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_kk = sum_kk + rf[w]
            cute.autovec_copy(sRed[2, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_kq = sum_kq + rf[w]
        else:
            for w in cutlass.range_constexpr(NWP):
                sum_qq = sum_qq + sRed[0, w]
                sum_kk = sum_kk + sRed[1, w]
                sum_kq = sum_kq + sRed[2, w]
        rq = cute.math.rsqrt(sum_qq, fastmath=True)
        rk = cute.math.rsqrt(sum_kk, fastmath=True)
        P = rq * rk * sum_kq

        if wid < NWP:
            for t in cutlass.range_constexpr(NEL):
                di = dib + t * NTK
                if cutlass.const_expr(HOLDV):
                    sQn[di] = fqv[t] * rq
                    sKn[di] = fkv[t] * rk
                    sEg[di] = cute.math.exp(fgv[t], fastmath=True)
                    sV[di] = fvv[t]
                else:
                    # Re-read instead of holding 4*NEL fragments across the
                    # barrier: these lines are L1-hot and the registers they
                    # free matter when NEL > 1 pushes the CTA past an
                    # occupancy step.
                    sQn[di] = mQ[bh, di].to(cutlass.Float32) * rq
                    sKn[di] = mK[bh, di].to(cutlass.Float32) * rk
                    sEg[di] = cute.math.exp(mG[bh, di], fastmath=True)
                    sV[di] = mV[bh, di].to(cutlass.Float32)
        cute.arch.sync_threads()

        # --- phase 1: float4 load of this thread's rows, decay, stage, reduce --
        fA = cute.make_fragment(VW, cutlass.Float32)
        fB = cute.make_fragment(VW, cutlass.Float32)
        frg = cute.make_fragment(VW, cutlass.Float32)
        # staging fragment: the VW halves a thread owns in row i are contiguous
        # in sDs, so they move as ONE VW*2-byte SMEM access instead of VW
        # separate 2-byte ones (the 2-byte form measured 3.5-way load / 2.5-way
        # store bank conflicts and was the top L1TEX bottleneck).
        frg16 = cute.make_fragment(VW, STAGE_T)
        if cutlass.const_expr(STAGE == 2):
            # STAGE=2: keep the whole decayed band in registers (NIT*VW fp32).
            # Exact, and it removes the staged SMEM tile entirely -- but it costs
            # NIT*VW registers per thread, which is the binding occupancy limit.
            # With PREFETCH the band is `rb`, already loaded above.
            if cutlass.const_expr(not PREFETCH):
                rb = [cute.make_fragment(VW, cutlass.Float32)
                      for _ in range(NIT)]
        for k in cutlass.range_constexpr(VW):
            fA[k] = cutlass.Float32(0.0)
            fB[k] = cutlass.Float32(0.0)
        for m in cutlass.range_constexpr(NIT):
            i = crow * NIT + m if ROWMAP else crow + m * RPI
            row = mS[bh, i, None]
            if cutlass.const_expr(STAGE == 2 and PREFETCH):
                pass                     # already in rb[m] from phase -1
            elif cutlass.const_expr(STAGE == 2):
                cute.autovec_copy(cute.local_tile(row, (VW,), (ctile,)), rb[m])
            else:
                cute.autovec_copy(cute.local_tile(row, (VW,), (ctile,)), frg)
            eg = sEg[i]
            qn = sQn[i]
            kn = sKn[i]
            for k in cutlass.range_constexpr(VW):
                if cutlass.const_expr(STAGE == 2):
                    ds = eg * rb[m][k]
                    rb[m][k] = ds        # decay in place; band reused by phase 3
                else:
                    ds = eg * frg[k]
                    frg16[k] = ds.to(STAGE_T)
                fA[k] = fA[k] + ds * qn
                fB[k] = fB[k] + ds * kn
            if cutlass.const_expr(STAGE == 2):
                pass
            elif cutlass.const_expr(SVEC):
                cute.autovec_copy(
                    frg16, cute.local_tile(sDs[i, None], (VW,), (clane,)))
            else:
                for k in cutlass.range_constexpr(VW):
                    sDs[i, clane * VW + k] = frg16[k]

        # fold the GPW row groups that share this warp (constant SMEM partials)
        for s in cutlass.range_constexpr(NSHF):
            off = LPR << s
            for k in cutlass.range_constexpr(VW):
                fA[k] = fA[k] + cute.arch.shuffle_sync_bfly(fA[k], off)
                fB[k] = fB[k] + cute.arch.shuffle_sync_bfly(fB[k], off)
        if lane < LPR:
            for k in cutlass.range_constexpr(VW):
                if cutlass.const_expr(PVEC):
                    pA[clane * VW + k, wid] = fA[k]
                    pB[clane * VW + k, wid] = fB[k]
                else:
                    pA[wid, clane * VW + k] = fA[k]
                    pB[wid, clane * VW + k] = fB[k]
        cute.arch.sync_threads()

        # --- phase 2: combine the NW warp partials, scalars, write o ----------
        # Only Vc of the NT threads do the combine -- one column each -- and
        # publish u through SMEM. Previously every one of the NT/LPR row groups
        # re-reduced the same NW partials for the same columns, which cost
        # NW*VW*2 scalar shared loads per thread and dominated the kernel's
        # shared-load traffic.
        uf = cute.make_fragment(VW, cutlass.Float32)
        if cutlass.const_expr(UBCAST):
            if tidx < Vc:
                a = cutlass.Float32(0.0)
                b = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    if cutlass.const_expr(PVEC):
                        a = a + pA[tidx, w]
                        b = b + pB[tidx, w]
                    else:
                        a = a + pA[w, tidx]
                        b = b + pB[w, tidx]
                u = beta * (sV[cbase + tidx] - b)
                sU[tidx] = u
                mO[bh, cbase + tidx] = (
                    (a + u * P) * cutlass.Float32(RSQRT_K)).to(cutlass.BFloat16)
            cute.arch.sync_threads()
            cute.autovec_copy(cute.local_tile(sU, (VW,), (clane,)), uf)
        elif cutlass.const_expr(PVEC):
            # NW partials of one column are contiguous -> NW/PCH vector reads
            # each. `pf` is reused across both arrays and both columns, so peak
            # register pressure grows by PCH, not by 2*NW.
            pf = cute.make_fragment(PCH, cutlass.Float32)
            for k in cutlass.range_constexpr(VW):
                lcol = clane * VW + k
                a = cutlass.Float32(0.0)
                b = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(NCH):
                    cute.autovec_copy(
                        cute.local_tile(pA[lcol, None], (PCH,), (c,)), pf)
                    for w in cutlass.range_constexpr(PCH):
                        a = a + pf[w]
                for c in cutlass.range_constexpr(NCH):
                    cute.autovec_copy(
                        cute.local_tile(pB[lcol, None], (PCH,), (c,)), pf)
                    for w in cutlass.range_constexpr(PCH):
                        b = b + pf[w]
                u = beta * (sV[cbase + lcol] - b)
                uf[k] = u
                if wid == 0:
                    if lane < LPR:
                        mO[bh, cbase + lcol] = (
                            (a + u * P) * cutlass.Float32(RSQRT_K)).to(cutlass.BFloat16)
        else:
            for k in cutlass.range_constexpr(VW):
                lcol = clane * VW + k
                a = cutlass.Float32(0.0)
                b = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    a = a + pA[w, lcol]
                    b = b + pB[w, lcol]
                u = beta * (sV[cbase + lcol] - b)
                uf[k] = u
                if wid == 0:
                    if lane < LPR:
                        mO[bh, cbase + lcol] = (
                            (a + u * P) * cutlass.Float32(RSQRT_K)).to(cutlass.BFloat16)

        # --- phase 3: rank-1 update + float4 write-back (writes S once) -------
        fo = cute.make_fragment(VW, cutlass.Float32)
        for m in cutlass.range_constexpr(NIT):
            i = crow * NIT + m if ROWMAP else crow + m * RPI
            kn = sKn[i]
            if cutlass.const_expr(STAGE == 2):
                for k in cutlass.range_constexpr(VW):
                    fo[k] = rb[m][k] + kn * uf[k]
            else:
                if cutlass.const_expr(SVEC):
                    cute.autovec_copy(
                        cute.local_tile(sDs[i, None], (VW,), (clane,)), frg16)
                else:
                    for k in cutlass.range_constexpr(VW):
                        frg16[k] = sDs[i, clane * VW + k]
                for k in cutlass.range_constexpr(VW):
                    fo[k] = frg16[k].to(cutlass.Float32) + kn * uf[k]
            row = mS[bh, i, None]
            cute.autovec_copy(fo, cute.local_tile(row, (VW,), (ctile,)))

    @cute.jit
    def launch(
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor,
        mBeta: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor, stream,
    ):
        bh = cute.size(mS, mode=[0])
        kda_decode_kernel_vsplit(mQ, mK, mV, mG, mBeta, mS, mO).launch(
            grid=[CS, bh, 1], block=[NW * 32, 1, 1], stream=stream,
        )

    return launch


def make_ptr_launcher(CS, BH):
    """Pointer-argument JIT entry for the V-split kernel (host fast path).

    Identical device code to `make_vsplit_launcher(CS)`, but the JIT entry takes
    seven raw `cute.Pointer`s instead of `cute.Tensor`s and rebuilds the (fully
    static) layouts inside the traced function. That matters purely on the host:
    a `cute.Tensor` argument has to go through `from_dlpack` + a memref
    descriptor build on every call (~2.9 us each, ~20 us for seven), while a
    `cute.Pointer` argument is a single `ctypes.c_void_p` whose address is
    stable -- so the launch-argument pack can be built once and re-pointed at
    the caller's current tensors with seven stores. See `KdaDecode.__call__`.

    Baking `BH` in also makes the grid extent a compile-time constant.
    """
    inner = make_vsplit_launcher(CS)

    @cute.jit
    def launch(pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer,
               pG: cute.Pointer, pBeta: cute.Pointer, pS: cute.Pointer,
               pO: cute.Pointer, stream):
        mQ = cute.make_tensor(pQ, cute.make_layout((BH, K), stride=(K, 1)))
        mK = cute.make_tensor(pK, cute.make_layout((BH, K), stride=(K, 1)))
        mV = cute.make_tensor(pV, cute.make_layout((BH, K), stride=(K, 1)))
        mG = cute.make_tensor(pG, cute.make_layout((BH, K), stride=(K, 1)))
        mBeta = cute.make_tensor(pBeta, cute.make_layout(BH))
        mS = cute.make_tensor(pS, cute.make_layout((BH, K, V), stride=(K * V, V, 1)))
        mO = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
        inner(mQ, mK, mV, mG, mBeta, mS, mO, stream)

    return launch


@cute.jit
def kda_decode_launch(
    mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor,
    mBeta: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor, stream,
):
    bh = cute.size(mS, mode=[0])
    if cutlass.const_expr(VEC == 2):
        kda_decode_kernel_vec(mQ, mK, mV, mG, mBeta, mS, mO).launch(
            grid=[bh, 1, 1], block=[NW * 32, 1, 1], stream=stream,
        )
    else:
        kda_decode_kernel(mQ, mK, mV, mG, mBeta, mS, mO).launch(
            grid=[bh, 1, 1], block=[V, 1, 1], stream=stream,
        )


# Column-split (V-split) selection. Measured on B200 at STAGE=2, NW=8, VW=2
# (raw us): CS=2 wins or ties at every shape, so the pick is constant.
#   B (H=16):    1     2     8     16     32     64    128    256
#   CS=2:      4.12  4.12  6.17  8.22  10.64  18.49  49.49  92.2
#   CS=4:      4.12  4.12  6.17  8.22  12.32  21.78  57.31  108.4
# CS=2 is also the floor for VW=2 (LPR = Vc/VW must fit in a warp).
CS_ENV = os.environ.get("KDA_CS", "auto")


def _pick_cs(BH):
    """Pick the column-split factor from the grid size alone.

    Shape-only specialization (static metadata: how many (b,h) CTAs the launch
    has) -- never a function of tensor values, seeds, or call history, so it is
    a legitimate clean-kernel lever.
    """
    if CS_ENV != "auto":
        return int(CS_ENV)
    cs_min = max(1, (V // VW) // 32)      # LPR = Vc/VW must fit in a warp
    return max(cs_min, 2)


# ===========================================================================
# SECTION 2 — the gated-body kernel generator: gated=True (fused sigmoid-gated
# RMSNorm epilogue) and/or fuse_conv=True (width-4 causal conv + SiLU
# prologue). One CTA per (b, h). Body from the pre-merge
# b10_kda_decode_gated_cutedsl.py, parametrized by the two flags.
# ===========================================================================

# ---------------------------------------------------------------------------
# Compile-time levers for the gated body. Defaults are the measured optimum
# (see the source package's optimization.md).
# ---------------------------------------------------------------------------
# Warps per CTA = the row-split factor. This kernel is per-thread
# memory-level-parallelism bound, not occupancy bound: FEWER, fatter warps win,
# because each holds a longer contiguous row band and issues that many
# independent state loads back to back. GNW=2 is not available (the band would
# be 256 registers, over the 255 cap) because this kernel gives one whole
# (b,h) to one CTA -- the price of fusing the epilogue.
GNW = int(os.environ.get("KDA_NW", "4"))
# Columns per thread. GVW=4 (LDG.128, NIT=32 row iterations) rather than GVW=2
# (LDG.64, NIT=64). Both hold the same 128-register state band and move the same
# bytes, so on raw memory behaviour they tie -- but GVW=4 needs half the loop
# bookkeeping and address registers, and that is what decides this kernel.
#
# GVW=2 only wins when it is *forced* under the launch bound (MINCTA=3), and that
# operating point is not portable: it depends on ptxas fitting 168 registers with
# zero spill, which the dev toolchain manages and the torch-2.11 eval toolchain
# does NOT (it spills 2.7M local-load sectors and runs 2.6x slower). GVW=4 reaches
# a good allocation on BOTH toolchains without being forced.
#
# Neither is hard-coded: AUTOCFG (below) picks between them by *asking the
# compiled binary* whether it spilled. These two remain the defaults used when
# the probe is disabled or the config is pinned by hand.
_VW_ENV = os.environ.get("KDA_VW")
GVW = int(_VW_ENV) if _VW_ENV is not None else 4
# Epilogue placement. 1 (default): warp 0 runs the gated-norm epilogue BEFORE the
# state write-back, so its o store is issued while the bulk STGs are still being
# generated. 0: after the write-back. Measured 0.01939 -> 0.01902 (B=64) and
# 0.08624 -> 0.08569 (B=256) for 1.
EPIORD = int(os.environ.get("KDA_EPIORD", "1"))
# 1: warp 0 issues the z/w loads at the very top of the kernel (they are consumed
# last, so this gives them the longest shadow). 0: load them in the epilogue.
# This is NOT a small effect -- 0 costs 35% at B=64 and 34% at B=256.
ZPF = int(os.environ.get("KDA_ZPF", "1"))
# 1: register-lean epilogue (column-outer, scalar SMEM reads -> ~EW+2 live
# registers). 0 (default): vector-read epilogue (NPART LDS.128 per array).
# Under MINCTA=3 the vector form wins (B=64 0.01765 -> 0.01666) -- ptxas fits it
# in the 168-register budget with zero spill anyway, so the cheaper instruction
# count is free. Without the launch bound the lean form was the faster of the two.
ESLIM = int(os.environ.get("KDA_ESLIM", "0"))
# `__launch_bounds__(NT, MINCTA)`. 0 (default) = let ptxas pick.
#
# Forcing the cap is a trap here. At GVW=2, MINCTA=3 caps registers at
# 65536/(3*128) = 170; the dev toolchain lands on 168 with zero spill and is 12%
# faster, but the torch-2.11 eval toolchain cannot make 168 work and spills
# (2.7M local-load sectors, 43 us vs 18 us at B=64). A launch bound is a
# *promise* about occupancy that ptxas must keep even when it has to spill to
# keep it, so it turns a toolchain difference into a 2.6x cliff.
#
# At GVW=4 both toolchains pick a good allocation unforced, so the portable
# configuration is MINCTA=0 -- and it costs only 0.4% against the forced
# dev-venv optimum. 4 caps at 128, below the 128-register band itself, and
# spills catastrophically everywhere.
_MINCTA_ENV = os.environ.get("KDA_MINCTA")
GMINCTA = int(_MINCTA_ENV) if _MINCTA_ENV is not None else 0

# 1 (default): TOOLCHAIN-ADAPTIVE CONFIG. Rather than hard-coding a (VW, MINCTA)
# that is safe on every toolchain, compile the candidates below in preference
# order and keep the first one whose *compiled binary reports zero spill*.
#
# The whole (VW, MINCTA) question in this kernel reduces to one bit -- did ptxas
# have to spill to meet the launch bound -- and that bit is readable directly off
# the compiled kernel via cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).
# It is a compile-target property, exact, and free at runtime. Measured, it is a
# perfect predictor of which configs are fast on which toolchain:
#
#   config          dev venv (2.9)        eval venv (2.11)
#   VW=2 MINCTA=3     0 B -> 16.43 us      600 B -> 43.0 us
#   VW=4 MINCTA=0     0 B -> 16.49 us        0 B -> 18.45 us
#   VW=2 MINCTA=0     0 B -> 18.46 us      192 B -> 32.7 us
#   VW=4 MINCTA=3    24 B -> 18.44 us      304 B -> 25.2 us
#
# Every spilling config is slow and every clean config is fast, on both. So the
# probe lets the kernel take the forced-launch-bound optimum on toolchains that
# can honour it and fall back to the unforced config on those that cannot,
# instead of giving up the former everywhere to stay safe on the latter.
#
# 0: use the (GVW, GMINCTA) above verbatim. Setting KDA_VW or KDA_MINCTA in the
# environment also pins the config and skips the probe, so every sweep recorded
# in optimization.md stays reproducible.
AUTOCFG = int(os.environ.get("KDA_AUTOCFG", "1"))
_PINNED = _VW_ENV is not None or _MINCTA_ENV is not None
# Candidate ladder, most-preferred first. Both entries are measured optima on a
# toolchain that can compile them without spilling; the last entry is the
# portable fallback and has never spilled on either toolchain.
CANDIDATES = ((2, 3), (4, 0))
# 0 (default, measured): DEFERRED NORMALIZATION -- publish RAW q/k to shared
# memory and apply the 1/||q||, 1/||k|| scalars afterwards, on the two reduced
# VW-vectors fA/fB instead of on the 128 broadcast values. Correct,
# algebraically free, and 12% slower here: deferring the scale forces rq (and
# rk, further still) to stay live *through the phase-1 band loop*, where 128 of
# the 168 available registers are already the state band, and two extra
# long-lived registers there is enough to spill. Kept as a lever because it is
# the right transform on any variant with register headroom. Incompatible with
# fuse_conv (the conv output must be held in registers anyway).
ONEBAR = int(os.environ.get("KDA_ONEBAR", "0"))
# 1: issue the small q/k/v/g loads BEFORE the 64-deep state prefetch instead of
# after it. Same instructions, different order: phase 0 is the first thing that
# blocks (its butterfly feeds the CTA barrier that gates phase 1), yet its
# operands are currently queued behind 64 state LDG.64s in the LSU queue --
# which is what `lg_throttle` measures. 0: the original order.
QKPF = int(os.environ.get("KDA_QKPF", "1"))
# 1: SHFL-BROADCAST the per-row coefficients instead of re-reading them from
# shared memory. MEASURED AND REJECTED (see the source package's
# optimization.md round 5): SHFL issues on the same MIO pipe it was meant to
# relieve, and it holds 3*NIT/32 coefficients live across the band loop.
SHFLB = int(os.environ.get("KDA_SHFLB", "0"))
# Debug/ablation only: 1 = skip the fused epilogue and write the PRE-norm r
# even when gated=True. Never a correctness path.
NOEPI = int(os.environ.get("KDA_NOEPI", "0"))

# ---------------------------------------------------------------------------
# fuse_conv-only levers, ported from the conv-fused champion package
# (gen_kda_decode_conv_gated_claude_0730_0014). They touch NOTHING unless
# fuse_conv=True: every use below sits under a const_expr(FCONV) branch, so
# the flag-off kernels compile to the identical bodies they had before.
# ---------------------------------------------------------------------------
# Conv-prologue issue order.
#   1 (default): issue ALL conv loads (weights, history, tokens, g), then the
#     NIT state-band loads, THEN do the conv arithmetic. The conv result gates
#     phase 0's butterfly, which gates the CTA barrier that gates phase 1 for
#     the whole CTA, so its loads must not queue behind the band's 64 LDGs --
#     the QKPF argument with a longer dependent chain. Costs ~15 registers
#     live across the prefetch.
#   0: complete the whole conv (load + FMA + SiLU + conv_state roll) BEFORE
#     the state prefetch. Shortest live range, but serialises one round trip.
CVPF = int(os.environ.get("KDA_CVPF", "1"))
# 1: issue the conv_state roll stores as soon as the values are in registers
# (they feed nothing downstream, so they retire in the shadow of the band
# loads). 0: defer them until after the conv arithmetic pass.
CVST = int(os.environ.get("KDA_CVST", "1"))
# CTA-index form for the conv-fused launch (the conv needs h and b SEPARATELY:
# packed channel index j*H*K + h*K + di, state index b*H + h -- and a runtime
# `bh / H` is a division, which this kernel bans).
#   1: 1-D grid of B*H CTAs, (h, b) recovered with a mask/shift (needs H a
#      power of two); 0: 2-D (H, B) grid, which hands the kernel h and b free.
# Both enumerate CTAs in the same linear order, yet which wins is a WINDOW
# (measured in the gen package): 2-D wins below ~2 CTAs/SM (the 1-D form packs
# CTAs onto a subset of SMs) and above ~L2-sized state (where co-residency
# stops buying L2 hits); in between the 1-D packing is worth up to 24%.
# -1 (default) resolves the window from the shape at compile time.
GRID1 = int(os.environ.get("KDA_GRID1", "-1"))
GRID1_MIN_CTA_PER_SM = float(os.environ.get("KDA_GRID1_MINOCC", "2.0"))
GRID1_BYTES = float(os.environ.get("KDA_GRID1_BYTES", "1.95e8"))
# State-band cache policy for the conv-fused launch. The S band is 98% of the
# kernel's bytes, touched exactly twice per launch (one read into the register
# band, one write-back) and never re-read on-chip; everything else (conv
# weights, w, the conv-state/token/gate rows) is small and wants to stay
# cached. While the state is around L2 the *inter-launch* reuse (same tile,
# next token) is real and worth protecting -> EVICT_NORMAL; far past L2 there
# is no reuse left and every allocated line is pure displacement ->
# EVICT_FIRST on both the band load and the band store. A hint only: no value
# loaded or stored changes. -1 (default) resolves from the shape; the split
# halves only make sense as a pair (first/normal measured worse than nothing).
_EPMAP = {
    "normal": "EVICT_NORMAL", "first": "EVICT_FIRST", "last": "EVICT_LAST",
    "unchanged": "EVICT_UNCHANGED", "noalloc": "NO_ALLOCATE",
}
BANDEPL = os.environ.get("KDA_BANDEPL", "-1")     # band load policy
BANDEPS = os.environ.get("KDA_BANDEPS", "-1")     # band store policy
# State read+write bytes above which the band is hinted evict-first. Same
# physical edge as GRID1_BYTES (is the state L2-co-resident across launches).
BANDEP_BYTES = float(os.environ.get("KDA_BANDEP_BYTES", "1.95e8"))


def _ep(name):
    """`CacheEvictionPriority` for a lever string (validated at build time)."""
    if name not in _EPMAP:
        raise ValueError(f"unknown cache eviction priority {name!r}; "
                         f"expected one of {sorted(_EPMAP)}")
    return getattr(cute_nv.CacheEvictionPriority, _EPMAP[name])


_NSM = None


def _num_sms():
    """SM count of the current device (cached); only resolves the grid-form
    window at compile time. Falls back to the B200's 148 if torch cannot say."""
    global _NSM
    if _NSM is None:
        try:
            _NSM = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        except Exception:                              # pragma: no cover
            _NSM = 148
    return _NSM


def make_launcher(BH, vw=None, mincta=None, *, gated=True, fuse_conv=False,
                  B=None, H=None, cs=None):
    """THE kernel generator: build the (fully static) launcher for a
    `B*H = BH` decode step, specialized by two compile-time flags.

    gated=False, fuse_conv=False returns the SECTION-1 V-split champion
    (`make_ptr_launcher`); every other combination compiles the single-CTA
    gated-body kernel below with the corresponding const_expr branches:

      gated     — compile the sigmoid-gated RMSNorm epilogue in (post-norm
                  output through mZ/mW); off = write the pre-norm r.
      fuse_conv — replace the q/k/v global loads with a width-4 causal conv +
                  SiLU over the packed pre-conv `mixed_qkv`, advancing the raw
                  bf16 `conv_state` window in place. Needs `B` and `H` baked
                  (packed-channel indexing); the grid becomes [H, B], or a
                  1-D [B*H] inside the GRID1 window when H is a power of two.

    `vw` / `mincta` override the module-level `GVW` / `GMINCTA` so the
    autoconfig probe can build several candidates from one source. Everything
    derived from `vw` below (LPR, RPI, NIT, NPART, ...) is recomputed per call,
    so the candidates are genuinely distinct compilations, not a shared body.

    Thread map inside the CTA (NT = GNW*32 threads); every quantity is a
    compile-time power of two, so all `%` and `/` below are shift/mask and the
    kernel contains no runtime division at all:

      LPR   = V/VW        threads covering one row of the state tile
      clane = t % LPR     this thread's column group -> cols clane*VW .. +VW-1
      crow  = t / LPR     this thread's row group
      RPI   = NT/LPR      rows advanced per iteration (= number of row groups)
      NIT   = K/RPI       row iterations per thread   (band = NIT*VW registers)
      MLPR  = max(LPR,32) partial-publish granularity
      NPART = NT/MLPR     distinct (A,B) partials per column
      EW    = V/32        epilogue columns per lane of warp 0

    `LPR > 32` (the default: V=128, VW=2 -> LPR=64) means a row group spans
    several warps, so no intra-warp fold is needed and the partial index is
    `crow`; `LPR < 32` means `32/LPR` row groups share a warp and are folded
    with a butterfly first, leaving the partial index `wid`. Both are
    `pidx = t / MLPR` with `NPART = NT / MLPR`, so one code path covers every
    (GNW, VW).
    """
    if not gated and not fuse_conv:
        return make_ptr_launcher(cs if cs is not None else _pick_cs(BH), BH)

    VW = GVW if vw is None else vw
    MINCTA = GMINCTA if mincta is None else mincta
    GATED = bool(gated)
    FCONV = bool(fuse_conv)
    EPI = GATED and not NOEPI
    if FCONV:
        assert B is not None and H is not None and B * H == BH, (
            f"fuse_conv needs B and H baked, got B={B} H={H} BH={BH}")
        assert not ONEBAR, "fuse_conv requires the two-barrier prologue"
        HK = H * K
        PACK = 3 * HK
        # Resolve the CTA-index form for THIS shape (see the GRID1 comment):
        # 1-D only inside its measured window AND when H is a power of two
        # (the mask/shift recovery of (h, b) needs it).
        G1 = GRID1
        if G1 < 0:
            G1 = 1 if (BH >= GRID1_MIN_CTA_PER_SM * _num_sms()
                       and (2.0 * BH * K * V * 4) <= GRID1_BYTES) else 0
        G1 = bool(G1) and (H & (H - 1)) == 0
        # Resolve the state-band cache policy for THIS shape (see the BANDEPL
        # comment). NB: named BEP*, not EPS -- EPS is the RMSNorm epsilon.
        _big = (2.0 * BH * K * V * 4) > BANDEP_BYTES
        BEPL = _ep(BANDEPL if BANDEPL != "-1" else
                   ("first" if _big else "normal"))
        BEPS = _ep(BANDEPS if BANDEPS != "-1" else
                   ("first" if _big else "normal"))
    else:
        G1 = False
    LPR = V // VW
    NT = GNW * 32
    RPI = NT // LPR
    NIT = K // RPI
    MLPR = max(LPR, 32)
    NPART = NT // MLPR
    GPW = max(1, 32 // LPR)          # row groups sharing one warp
    NSHF = GPW.bit_length() - 1      # butterfly steps to fold them
    PUBL = min(LPR, 32)              # lanes of a warp that publish a partial
    # Phase 0 (the fused L2 norms) has to cover all K head-dim elements with NT
    # threads. When NT >= K the first K threads take one element each and the
    # rest are dropped; when NT < K every thread takes NEL = K/NT elements.
    # Getting this wrong is silent and severe, so it is asserted.
    NTK = min(NT, K)                 # threads that carry a head-dim element
    NEL = K // NTK                   # head-dim elements per such thread
    NWP = min(GNW, NWARP)            # warps holding norm partials
    EW = V // 32                     # epilogue columns per warp-0 lane
    # fuse_conv forces the register-held q/k/v/g path: the conv output exists
    # only in registers (there is no post-conv global tensor to re-read).
    QKPF1 = (QKPF and NHOLD and not ONEBAR) or FCONV
    # SHFL broadcast needs (a) a contiguous per-thread row band, so the
    # distributed load is one conflict-free 32-lane read per slot, and (b) the
    # band to be a whole number of 32-row slots, so `m % 32` / `m // 32` are
    # compile-time constants. Both hold for every shipped (GNW, VW).
    SHFLB1 = SHFLB and ROWMAP and NIT % 32 == 0 and LPR >= 32
    NSLOT = NIT // 32 if SHFLB1 else 0
    assert V % VW == 0 and NT % LPR == 0 and K % RPI == 0, (
        f"bad shape NW={GNW} VW={VW} LPR={LPR} RPI={RPI}")
    assert (LPR % 32 == 0) or (32 % LPR == 0), f"bad LPR={LPR}"
    assert K % NTK == 0 and NTK == NWP * 32, (
        f"bad norm-phase config NW={GNW} NT={NT} NTK={NTK} NWP={NWP}")
    assert NIT * VW <= 200, (
        f"register band NIT*VW={NIT * VW} too large for NW={GNW} VW={VW}")
    assert V == 32 * EW, "the epilogue assumes one warp covers V columns"

    @cute.kernel
    def kda_decode_gated_kernel(
        # Meaning of the first three tensors depends on FCONV:
        #   FCONV=0: q / k / v, each (BH, 128) bf16
        #   FCONV=1: mixed_qkv (B, 3*H*128) bf16 RAW pre-conv packed [q|k|v],
        #            conv_weight (3*H*128, 4) bf16,
        #            conv_state (B, 3*H*128*3) bf16 FLAT view of the (B, C, 3)
        #            raw window buffer (in place; element = ch*3 + slot)
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mG: cute.Tensor,      # (BH, 128) fp32   log-decay
        mBeta: cute.Tensor,   # (BH,)     fp32
        mS: cute.Tensor,      # (BH, 128, 128) fp32   (in place)
        mO: cute.Tensor,      # (BH, 128) bf16   output (post-norm iff GATED)
        mZ: cute.Tensor,      # (BH, 128) bf16   output gate (GATED only)
        mW: cute.Tensor,      # (128,)    fp32   RMSNorm weight (GATED only)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(FCONV):
            if cutlass.const_expr(G1):
                # 1-D grid: H is a power of two here (the dispatch guarantees
                # it), so recovering (h, b) is an AND and a SHR, no division.
                bh, _, _ = cute.arch.block_idx()
                hh = bh % H
                bb = bh // H
            else:
                # 2D grid [H, B]: b/h available without a runtime division, and
                # bh = b*H + h keeps every non-conv tensor's indexing unchanged.
                hh, bb, _ = cute.arch.block_idx()
                bh = bb * H + hh
        else:
            bh, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        wid = tidx // 32
        clane = tidx % LPR               # column group within the row
        crow = tidx // LPR               # row group
        pidx = tidx // MLPR              # partial slot this thread contributes to

        smem = SmemAllocator()
        # Per-row broadcast coefficients. Kept as separate K-length arrays on
        # purpose: because a thread walks a *contiguous* row band, ptxas merges
        # four consecutive rows of each array into one LDS.128. Packing them
        # into an interleaved array blocks that merge and measured 18% slower in
        # the source package.
        sEg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sQn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sKn = smem.allocate_tensor(cutlass.Float32, cute.make_layout(K))
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(V))
        # (3, NWP): the NWP partials of one quantity are contiguous, so the
        # cross-warp norm combine is 3 vector reads instead of 3*NWP scalars.
        sRed = smem.allocate_tensor(cutlass.Float32,
                                    cute.make_layout((3, NWP), stride=(NWP, 1)))
        # Cross-row-group partials, row-major (NPART, V): both the phase-2
        # combine (VW contiguous floats) and the epilogue (EW=4 contiguous
        # floats per lane, 32 lanes x 16 B = 4 conflict-free bank cycles) read
        # them as vectors.
        pA = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((NPART, V), stride=(V, 1)))
        pB = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((NPART, V), stride=(V, 1)))

        beta = mBeta[bh]
        dib = tidx % NTK                 # this thread's first head-dim index
        # Epilogue operands. Allocated at kernel scope (the 4.5.2 tracer rejects
        # a value whose first binding is inside a dynamic `if`) and filled by
        # warp 0 only.
        fw = cute.make_fragment(EW, cutlass.Float32)
        fzb = cute.make_fragment(EW, cutlass.BFloat16)
        # NHOLD staging for the two-barrier prologue only: with ONEBAR the
        # publish happens before the reduction, so nothing has to be held.
        if cutlass.const_expr(not ONEBAR):
            fqv = cute.make_fragment(NEL, cutlass.Float32)
            fkv = cute.make_fragment(NEL, cutlass.Float32)
            fvv = cute.make_fragment(NEL, cutlass.Float32)
            fgv = cute.make_fragment(NEL, cutlass.Float32)
        # The register band: also the destination of the state prefetch, and the
        # storage the decayed value is written back into.
        rb = [cute.make_fragment(VW, cutlass.Float32) for _ in range(NIT)]

        if cutlass.const_expr(FCONV):
            # Conv operands, staged in REGISTER fragments so that every conv
            # LOAD can be issued before the state prefetch while the conv
            # ARITHMETIC happens after it (CVPF=1). Direct strided access, no
            # smem staging: a channel's 3-half history is 6 contiguous bytes
            # at a 6-byte stride, so the 3 LDG.U16 per block together fetch
            # exactly the lines the warp needs (no wasted line traffic, only
            # instruction count) -- and the gen package's KDA_ABL ablation
            # bounds ANY staging scheme at ~3% where the state is L2-resident
            # and NEGATIVE where it is not (the extra in-flight loads supply
            # memory-level parallelism at 2 CTAs/SM).
            # Flat unit index u = t*3 + j: head-dim element t, packed block j.
            fcw = cute.make_fragment(
                cute.make_layout((3 * NEL, 4), stride=(4, 1)),
                cutlass.BFloat16)
            fcs = cute.make_fragment(
                cute.make_layout((3 * NEL, 3), stride=(3, 1)),
                cutlass.BFloat16)
            fcx = cute.make_fragment(3 * NEL, cutlass.BFloat16)

            # Everything the helpers touch is bound as a default argument:
            # the 4.5.2 tracer rejects closures that capture tensors or
            # fragments when the call sits inside a dynamic `if`.
            def conv_load(HK=HK, NTK=NTK, K=K, bb=bb, hh=hh, bh=bh, dib=dib,
                          mQ=mQ, mK=mK, mV=mV, mG=mG,
                          fgv=fgv, fcw=fcw, fcs=fcs, fcx=fcx):
                """Issue every conv load (all independent -> one round trip):
                per unit one 8-B weight vector, three 2-B history halves and
                one 2-B token; plus the (not convolved) g element."""
                for t in cutlass.range_constexpr(NEL):
                    dq = dib + t * NTK
                    fgv[t] = mG[bh, dq]
                    for j in cutlass.range_constexpr(3):
                        ch = j * HK + hh * K + dq      # packed channel
                        u = t * 3 + j
                        cute.autovec_copy(mK[ch, None], fcw[u, None])
                        for sp in cutlass.range_constexpr(3):
                            fcs[u, sp] = mV[bb, ch * 3 + sp]
                        fcx[u] = mQ[bb, ch]

            def conv_compute(HK=HK, NTK=NTK, K=K, bb=bb, hh=hh, dib=dib,
                             mV=mV, fcw=fcw, fcs=fcs, fcx=fcx,
                             fqv=fqv, fkv=fkv, fvv=fvv):
                """4-tap depthwise conv + SiLU, fp32 accumulate. The post-conv
                q/k/v STAY fp32 (never rounded through bf16: they were going
                to be fp32 on-chip anyway, and this is strictly more accurate
                than the Triton conv kernel, which rounds its output tensor).
                SiLU(a) = a * rcp(1 + exp(-a)); rcp.approx.ftz.f32 is exact to
                2^-23 and 1 + exp(-a) >= 1, so ftz never fires. CVST=1 also
                rolls the conv_state window ([s1, s2, x] RAW, bit-exact vs the
                Triton kernel) straight back to global here -- the stores feed
                nothing downstream, so they retire in the band-load shadow."""
                for t in cutlass.range_constexpr(NEL):
                    for j in cutlass.range_constexpr(3):
                        u = t * 3 + j
                        xn = fcx[u]
                        a = fcw[u, 3].to(cutlass.Float32) * xn.to(cutlass.Float32)
                        for sp in cutlass.range_constexpr(3):
                            a = a + (fcw[u, sp].to(cutlass.Float32)
                                     * fcs[u, sp].to(cutlass.Float32))
                        xp = a * cute.arch.rcp_approx(
                            cutlass.Float32(1.0)
                            + cute.math.exp(-a, fastmath=True))
                        # tuple index, not an `if`: `j` is a Python constant
                        # here, and a real `if` would be rewritten by the DSL
                        # into a traced block, inside which an assignment to a
                        # closure fragment does not resolve.
                        (fqv, fkv, fvv)[j][t] = xp
                        if cutlass.const_expr(CVST):
                            ch = j * HK + hh * K + (dib + t * NTK)
                            mV[bb, ch * 3] = fcs[u, 1]
                            mV[bb, ch * 3 + 1] = fcs[u, 2]
                            mV[bb, ch * 3 + 2] = xn

            def conv_store(HK=HK, NTK=NTK, K=K, bb=bb, hh=hh, dib=dib,
                           mV=mV, fcs=fcs, fcx=fcx):
                """CVST=0 only: the deferred conv_state roll stores."""
                for t in cutlass.range_constexpr(NEL):
                    for j in cutlass.range_constexpr(3):
                        u = t * 3 + j
                        ch = j * HK + hh * K + (dib + t * NTK)
                        mV[bb, ch * 3] = fcs[u, 1]
                        mV[bb, ch * 3 + 1] = fcs[u, 2]
                        mV[bb, ch * 3 + 2] = fcx[u]

        # --- phase -1: prefetch ------------------------------------------------
        # z/w are consumed LAST (the epilogue) so warp 0 issues them FIRST: they
        # get the longest possible shadow and cost nothing on the critical path.
        # Then every thread issues all NIT state loads (VW columns each). None
        # depend on the norms, so they fly while the prologue runs its two CTA
        # barriers, four vector loads and 15-step shuffle chain, instead of the
        # prologue latency and the load latency being paid back to back.
        if cutlass.const_expr(ZPF and EPI):
            if wid == 0:
                cute.autovec_copy(cute.local_tile(mW, (EW,), (lane,)), fw)
                cute.autovec_copy(
                    cute.local_tile(mZ[bh, None], (EW,), (lane,)), fzb)
        # ...but q/k/v/g go out BEFORE the state prefetch under QKPF. They are the
        # first thing consumed (phase 0's butterfly feeds the CTA barrier that
        # gates phase 1 for the whole CTA), so queueing them behind 64 state
        # LDG.64s makes every warp wait a full LSU-queue drain for four 2-byte
        # loads. Reordering costs nothing: it is the same four loads into the
        # same four NHOLD registers, just issued first. Under FCONV the loads
        # become the conv4+SiLU prologue -- still ahead of the state prefetch,
        # for the same reason (it feeds the norm critical path).
        if cutlass.const_expr(FCONV):
            # The conv LOADS go out before the state prefetch (CVPF=1: the
            # conv result gates phase 0's butterfly, which gates the CTA
            # barrier that gates phase 1 for the whole CTA, so queueing them
            # behind NIT state LDGs makes every warp wait a full LSU-queue
            # drain); the conv ARITHMETIC runs after the prefetch, so the
            # band loads fly during the conv round trip. Carrier warps only
            # (wid < NWP <=> tidx < NTK, asserted above): a second warp
            # re-running a channel would double-store the roll.
            if cutlass.const_expr(NTK < NT):
                # Non-carrier warps must not touch conv memory and must not
                # read uninitialised registers, so they run the butterfly on
                # zeros; `wid < NWP` discards their partials at the publish.
                for t in cutlass.range_constexpr(NEL):
                    fqv[t] = cutlass.Float32(0.0)
                    fkv[t] = cutlass.Float32(0.0)
                    fvv[t] = cutlass.Float32(0.0)
                    fgv[t] = cutlass.Float32(0.0)
                if wid < NWP:
                    conv_load()
                    if cutlass.const_expr(not CVPF):
                        conv_compute()
            else:
                conv_load()
                if cutlass.const_expr(not CVPF):
                    conv_compute()
        elif cutlass.const_expr(QKPF1):
            for t in cutlass.range_constexpr(NEL):
                dq = dib + t * NTK
                fqv[t] = mQ[bh, dq].to(cutlass.Float32)
                fkv[t] = mK[bh, dq].to(cutlass.Float32)
                fvv[t] = mV[bh, dq].to(cutlass.Float32)
                fgv[t] = mG[bh, dq]
        if cutlass.const_expr(PREFETCH):
            for m in cutlass.range_constexpr(NIT):
                i = crow * NIT + m if ROWMAP else crow + m * RPI
                if cutlass.const_expr(FCONV):
                    # The band load carries the shape-resolved eviction hint
                    # (see BANDEPL); the non-conv path is byte-identical to
                    # the pre-merge kernels and stays unhinted.
                    cute.autovec_copy(
                        cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                        rb[m], l1c_evict_priority=BEPL)
                else:
                    cute.autovec_copy(
                        cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                        rb[m])
        if cutlass.const_expr(FCONV and CVPF):
            if cutlass.const_expr(NTK < NT):
                if wid < NWP:
                    conv_compute()
            else:
                conv_compute()
        if cutlass.const_expr(FCONV and not CVST):
            if cutlass.const_expr(NTK < NT):
                if wid < NWP:
                    conv_store()
            else:
                conv_store()
        # --- phase 0: fused L2 norms (needs the full 128-element q/k vectors) --
        # Each of the NTK carrier threads owns NEL head-dim elements
        # (di = dib + t*NTK), so the whole K vector is covered for any NW.
        p_qq = cutlass.Float32(0.0)
        p_kk = cutlass.Float32(0.0)
        p_kq = cutlass.Float32(0.0)
        di = dib
        for t in cutlass.range_constexpr(NEL):
            di = dib + t * NTK
            if cutlass.const_expr(QKPF1):
                qt = fqv[t]          # already in flight since before the prefetch
                kt = fkv[t]
            else:
                qt = mQ[bh, di].to(cutlass.Float32)
                kt = mK[bh, di].to(cutlass.Float32)
            if cutlass.const_expr(ONEBAR):
                # Publish RAW q/k right now -- nothing here depends on the norms,
                # so this store does not have to wait for the sRed exchange and
                # the second CTA barrier disappears. Nothing is held in registers
                # across the barrier either.
                if wid < NWP:
                    sQn[di] = qt
                    sKn[di] = kt
                    sEg[di] = cute.math.exp(mG[bh, di], fastmath=True)
                    sV[di] = mV[bh, di].to(cutlass.Float32)
            elif cutlass.const_expr(NHOLD and not QKPF1):
                fqv[t] = qt
                fkv[t] = kt
                fvv[t] = mV[bh, di].to(cutlass.Float32)
                fgv[t] = mG[bh, di]
            p_qq = p_qq + qt * qt
            p_kk = p_kk + kt * kt
            p_kq = p_kq + kt * qt
        for off in cutlass.range_constexpr(5):        # warp butterfly sum
            sh = 1 << (4 - off)
            p_qq = p_qq + cute.arch.shuffle_sync_bfly(p_qq, sh)
            p_kk = p_kk + cute.arch.shuffle_sync_bfly(p_kk, sh)
            p_kq = p_kq + cute.arch.shuffle_sync_bfly(p_kq, sh)
        if wid < NWP:
            if lane == 0:
                sRed[0, wid] = p_qq
                sRed[1, wid] = p_kk
                sRed[2, wid] = p_kq
        cute.arch.sync_threads()

        sum_qq = cutlass.Float32(0.0)
        sum_kk = cutlass.Float32(0.0)
        sum_kq = cutlass.Float32(0.0)
        if cutlass.const_expr(RVEC):
            rf = cute.make_fragment(NWP, cutlass.Float32)
            cute.autovec_copy(sRed[0, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_qq = sum_qq + rf[w]
            cute.autovec_copy(sRed[1, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_kk = sum_kk + rf[w]
            cute.autovec_copy(sRed[2, None], rf)
            for w in cutlass.range_constexpr(NWP):
                sum_kq = sum_kq + rf[w]
        else:
            for w in cutlass.range_constexpr(NWP):
                sum_qq = sum_qq + sRed[0, w]
                sum_kk = sum_kk + sRed[1, w]
                sum_kq = sum_kq + sRed[2, w]
        rq = cute.math.rsqrt(sum_qq, fastmath=True)
        rk = cute.math.rsqrt(sum_kk, fastmath=True)
        P = rq * rk * sum_kq             # kn . qn, the rank-1 output coupling

        if cutlass.const_expr(not ONEBAR):
            # scale-before-publish: needs a SECOND barrier, because these stores
            # depend on rq/rk which only exist after the first one.
            if wid < NWP:
                for t in cutlass.range_constexpr(NEL):
                    di = dib + t * NTK
                    if cutlass.const_expr(NHOLD or FCONV):
                        sQn[di] = fqv[t] * rq
                        sKn[di] = fkv[t] * rk
                        sEg[di] = cute.math.exp(fgv[t], fastmath=True)
                        sV[di] = fvv[t]
                    else:
                        sQn[di] = mQ[bh, di].to(cutlass.Float32) * rq
                        sKn[di] = mK[bh, di].to(cutlass.Float32) * rk
                        sEg[di] = cute.math.exp(mG[bh, di], fastmath=True)
                        sV[di] = mV[bh, di].to(cutlass.Float32)
            cute.arch.sync_threads()

        # SHFL broadcast: pull this warp's NIT rows of each coefficient into
        # registers, one row per lane per slot. Each of these is a contiguous
        # 32-lane read (conflict-free, one wavefront), so the whole distributed
        # copy costs 3*NSLOT LDS instead of the 3*NIT/4 merged LDS.128 the
        # broadcast form pays -- the trade is that every *use* then costs a
        # shuffle. Row `crow*NIT + s*32 + lane` lives in lane `s`-slot.
        if cutlass.const_expr(SHFLB1):
            fdEg = cute.make_fragment(NSLOT, cutlass.Float32)
            fdQn = cute.make_fragment(NSLOT, cutlass.Float32)
            fdKn = cute.make_fragment(NSLOT, cutlass.Float32)
            for s in cutlass.range_constexpr(NSLOT):
                rr = crow * NIT + s * 32 + lane
                fdEg[s] = sEg[rr]
                fdQn[s] = sQn[rr]
                fdKn[s] = sKn[rr]

        # --- phase 1: decay this thread's row band in registers, reduce A/B ----
        fA = cute.make_fragment(VW, cutlass.Float32)
        fB = cute.make_fragment(VW, cutlass.Float32)
        for k in cutlass.range_constexpr(VW):
            fA[k] = cutlass.Float32(0.0)
            fB[k] = cutlass.Float32(0.0)
        for m in cutlass.range_constexpr(NIT):
            i = crow * NIT + m if ROWMAP else crow + m * RPI
            if cutlass.const_expr(not PREFETCH):
                if cutlass.const_expr(FCONV):
                    cute.autovec_copy(
                        cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                        rb[m], l1c_evict_priority=BEPL)
                else:
                    cute.autovec_copy(
                        cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                        rb[m])
            if cutlass.const_expr(SHFLB1):
                # m is a constexpr, so both the slot and the source lane fold to
                # immediates: `shfl.idx.b32 d, r<slot>, <m%32>, 0x1f`.
                eg = cute.arch.shuffle_sync(fdEg[m // 32], m % 32)
                qn = cute.arch.shuffle_sync(fdQn[m // 32], m % 32)
                kn = cute.arch.shuffle_sync(fdKn[m // 32], m % 32)
            else:
                eg = sEg[i]
                qn = sQn[i]
                kn = sKn[i]
            for k in cutlass.range_constexpr(VW):
                ds = eg * rb[m][k]
                rb[m][k] = ds            # decay in place; band reused by phase 3
                fA[k] = fA[k] + ds * qn
                fB[k] = fB[k] + ds * kn

        # Deferred normalization: sQn/sKn held the RAW q/k, so the 1/||q||,
        # 1/||k|| scalars are applied here -- 2*VW multiplies on the reduced
        # vectors instead of 2*K on the broadcast values, and, more importantly,
        # off the critical path that used to force a second prologue barrier.
        # pA/pB therefore carry exactly the same quantities as the ONEBAR=0 path,
        # so phase 2 and the epilogue are unchanged.
        if cutlass.const_expr(ONEBAR):
            for k in cutlass.range_constexpr(VW):
                fA[k] = fA[k] * rq
                fB[k] = fB[k] * rk

        # fold the GPW row groups that share this warp (no-op when LPR >= 32),
        # so the number of SMEM partials per column stays NPART
        for s in cutlass.range_constexpr(NSHF):
            off = LPR << s
            for k in cutlass.range_constexpr(VW):
                fA[k] = fA[k] + cute.arch.shuffle_sync_bfly(fA[k], off)
                fB[k] = fB[k] + cute.arch.shuffle_sync_bfly(fB[k], off)
        if lane < PUBL:
            for k in cutlass.range_constexpr(VW):
                pA[pidx, clane * VW + k] = fA[k]
                pB[pidx, clane * VW + k] = fB[k]
        cute.arch.sync_threads()

        # `uf` (phase 2) is computed inside writeback() rather than here: the
        # epilogue does not need it, and keeping it out of the epilogue's live
        # range costs the register peak nothing.
        uf = cute.make_fragment(VW, cutlass.Float32)

        # --- the fused gated-RMSNorm epilogue (warp 0 only) --------------------
        # Warp 0 re-derives ALL V columns of the pre-norm output r from the SMEM
        # partials (EW per lane, vector reads), reduces sum(r^2) with one warp
        # butterfly, and writes the post-norm output. This costs no extra CTA
        # barrier -- which is why the redundant recompute is worth it.
        def epilogue():
            if cutlass.const_expr(EPI):
                if wid == 0:
                    if cutlass.const_expr(not ZPF):
                        cute.autovec_copy(cute.local_tile(mW, (EW,), (lane,)), fw)
                        cute.autovec_copy(
                            cute.local_tile(mZ[bh, None], (EW,), (lane,)), fzb)
                    fr = cute.make_fragment(EW, cutlass.Float32)
                    ss = cutlass.Float32(0.0)
                    if cutlass.const_expr(ESLIM):
                        # Column-outer form: only `fr` survives across columns,
                        # so the epilogue's live set is EW+2 registers instead of
                        # the ~5*EW that the vector-read form needs. The band is
                        # 128 registers, so that difference is what decides
                        # whether a third CTA fits on the SM.
                        for j in cutlass.range_constexpr(EW):
                            c = lane * EW + j
                            a = cutlass.Float32(0.0)
                            b = cutlass.Float32(0.0)
                            for p in cutlass.range_constexpr(NPART):
                                a = a + pA[p, c]
                                b = b + pB[p, c]
                            u = beta * (sV[c] - b)
                            r = (a + u * P) * cutlass.Float32(RSQRT_K)
                            fr[j] = r
                            ss = ss + r * r
                    else:
                        # Vector-read form: NPART LDS.128 per array, one for sV.
                        pf = cute.make_fragment(EW, cutlass.Float32)
                        fsv = cute.make_fragment(EW, cutlass.Float32)
                        ea = cute.make_fragment(EW, cutlass.Float32)
                        eb = cute.make_fragment(EW, cutlass.Float32)
                        for j in cutlass.range_constexpr(EW):
                            ea[j] = cutlass.Float32(0.0)
                            eb[j] = cutlass.Float32(0.0)
                        for p in cutlass.range_constexpr(NPART):
                            cute.autovec_copy(
                                cute.local_tile(pA[p, None], (EW,), (lane,)), pf)
                            for j in cutlass.range_constexpr(EW):
                                ea[j] = ea[j] + pf[j]
                            cute.autovec_copy(
                                cute.local_tile(pB[p, None], (EW,), (lane,)), pf)
                            for j in cutlass.range_constexpr(EW):
                                eb[j] = eb[j] + pf[j]
                        cute.autovec_copy(cute.local_tile(sV, (EW,), (lane,)), fsv)
                        for j in cutlass.range_constexpr(EW):
                            u = beta * (fsv[j] - eb[j])
                            r = (ea[j] + u * P) * cutlass.Float32(RSQRT_K)
                            fr[j] = r
                            ss = ss + r * r
                    for eo in cutlass.range_constexpr(5):
                        ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << (4 - eo))
                    # mean(r^2) is a multiply by the constant 1/V, not a divide
                    sc = cute.math.rsqrt(
                        ss * cutlass.Float32(RMEAN) + cutlass.Float32(EPS),
                        fastmath=True)
                    fo = cute.make_fragment(EW, cutlass.BFloat16)
                    for j in cutlass.range_constexpr(EW):
                        zf = fzb[j].to(cutlass.Float32)
                        # sigmoid(z) = 1/(1+exp(-z)); rcp.approx.ftz.f32 is exact
                        # to 2^-23 and 1+exp(-z) >= 1 so ftz is irrelevant.
                        sg = cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-zf, fastmath=True))
                        fo[j] = (fr[j] * sc * fw[j] * sg).to(cutlass.BFloat16)
                    cute.autovec_copy(
                        fo, cute.local_tile(mO[bh, None], (EW,), (lane,)))
            else:
                # gated=False (or NOEPI ablation): write the PRE-norm r -- what
                # the bare-recurrence kernel emits.
                if wid == 0:
                    pf = cute.make_fragment(EW, cutlass.Float32)
                    fsv = cute.make_fragment(EW, cutlass.Float32)
                    ea = cute.make_fragment(EW, cutlass.Float32)
                    eb = cute.make_fragment(EW, cutlass.Float32)
                    for j in cutlass.range_constexpr(EW):
                        ea[j] = cutlass.Float32(0.0)
                        eb[j] = cutlass.Float32(0.0)
                    for p in cutlass.range_constexpr(NPART):
                        cute.autovec_copy(
                            cute.local_tile(pA[p, None], (EW,), (lane,)), pf)
                        for j in cutlass.range_constexpr(EW):
                            ea[j] = ea[j] + pf[j]
                        cute.autovec_copy(
                            cute.local_tile(pB[p, None], (EW,), (lane,)), pf)
                        for j in cutlass.range_constexpr(EW):
                            eb[j] = eb[j] + pf[j]
                    cute.autovec_copy(cute.local_tile(sV, (EW,), (lane,)), fsv)
                    fo = cute.make_fragment(EW, cutlass.BFloat16)
                    for j in cutlass.range_constexpr(EW):
                        u = beta * (fsv[j] - eb[j])
                        fo[j] = ((ea[j] + u * P)
                                 * cutlass.Float32(RSQRT_K)).to(cutlass.BFloat16)
                    cute.autovec_copy(
                        fo, cute.local_tile(mO[bh, None], (EW,), (lane,)))

        def writeback():
            # --- phase 2: combine the NPART partials for this thread's columns -
            for k in cutlass.range_constexpr(VW):
                lcol = clane * VW + k
                b = cutlass.Float32(0.0)
                for p in cutlass.range_constexpr(NPART):
                    b = b + pB[p, lcol]
                # phase 3 reads the RAW k out of sKn under ONEBAR, so fold the
                # 1/||k|| into u once here (VW multiplies) rather than into all
                # K broadcast values in the prologue. `u` itself is unchanged --
                # the epilogue still sees the true u, because it recomputes it
                # from the (already rk-scaled) pB partials.
                uf[k] = beta * (sV[lcol] - b)
                if cutlass.const_expr(ONEBAR):
                    uf[k] = uf[k] * rk

            # --- phase 3: rank-1 update + write S once (one STG per row) ------
            fo = cute.make_fragment(VW, cutlass.Float32)
            for m in cutlass.range_constexpr(NIT):
                i = crow * NIT + m if ROWMAP else crow + m * RPI
                if cutlass.const_expr(SHFLB1):
                    kn = cute.arch.shuffle_sync(fdKn[m // 32], m % 32)
                else:
                    kn = sKn[i]
                for k in cutlass.range_constexpr(VW):
                    fo[k] = rb[m][k] + kn * uf[k]
                if cutlass.const_expr(FCONV):
                    # The band store carries the same shape-resolved eviction
                    # hint as the load (the halves only work as a pair; the
                    # STORE side carries most of the measured win).
                    cute.autovec_copy(
                        fo, cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                        l1c_evict_priority=BEPS)
                else:
                    cute.autovec_copy(
                        fo, cute.local_tile(mS[bh, i, None], (VW,), (clane,)))

        if cutlass.const_expr(EPIORD):
            epilogue()
            writeback()
        else:
            writeback()
            epilogue()

    if FCONV:
        if GATED:
            @cute.jit
            def launch(pX: cute.Pointer, pCW: cute.Pointer, pCS: cute.Pointer,
                       pG: cute.Pointer, pBeta: cute.Pointer, pS: cute.Pointer,
                       pO: cute.Pointer, pZ: cute.Pointer, pW: cute.Pointer,
                       stream):
                mQ = cute.make_tensor(
                    pX, cute.make_layout((B, PACK), stride=(PACK, 1)))
                mK = cute.make_tensor(
                    pCW, cute.make_layout((PACK, 4), stride=(4, 1)))
                # conv_state as a FLAT (B, PACK*3) view: the kernel addresses
                # window elements as ch*3 + slot, so no runtime division ever
                # touches the packed index.
                mV = cute.make_tensor(
                    pCS, cute.make_layout((B, PACK * 3), stride=(3 * PACK, 1)))
                mG = cute.make_tensor(pG, cute.make_layout((BH, K), stride=(K, 1)))
                mBeta = cute.make_tensor(pBeta, cute.make_layout(BH))
                mS = cute.make_tensor(
                    pS, cute.make_layout((BH, K, V), stride=(K * V, V, 1)))
                mO = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
                mZ = cute.make_tensor(pZ, cute.make_layout((BH, V), stride=(V, 1)))
                mW = cute.make_tensor(pW, cute.make_layout(V))
                kda_decode_gated_kernel(
                    mQ, mK, mV, mG, mBeta, mS, mO, mZ, mW).launch(
                    grid=([BH, 1, 1] if G1 else [H, B, 1]),
                    block=[NT, 1, 1], stream=stream,
                    min_blocks_per_mp=MINCTA,
                )
        else:
            @cute.jit
            def launch(pX: cute.Pointer, pCW: cute.Pointer, pCS: cute.Pointer,
                       pG: cute.Pointer, pBeta: cute.Pointer, pS: cute.Pointer,
                       pO: cute.Pointer, stream):
                mQ = cute.make_tensor(
                    pX, cute.make_layout((B, PACK), stride=(PACK, 1)))
                mK = cute.make_tensor(
                    pCW, cute.make_layout((PACK, 4), stride=(4, 1)))
                # Flat conv_state view -- see the gated launcher above.
                mV = cute.make_tensor(
                    pCS, cute.make_layout((B, PACK * 3), stride=(3 * PACK, 1)))
                mG = cute.make_tensor(pG, cute.make_layout((BH, K), stride=(K, 1)))
                mBeta = cute.make_tensor(pBeta, cute.make_layout(BH))
                mS = cute.make_tensor(
                    pS, cute.make_layout((BH, K, V), stride=(K * V, V, 1)))
                mO = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
                # gated=False compiles the epilogue's mZ/mW reads out entirely;
                # these dummies only satisfy the kernel signature.
                mZ = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
                mW = cute.make_tensor(pBeta, cute.make_layout(V))
                kda_decode_gated_kernel(
                    mQ, mK, mV, mG, mBeta, mS, mO, mZ, mW).launch(
                    grid=([BH, 1, 1] if G1 else [H, B, 1]),
                    block=[NT, 1, 1], stream=stream,
                    min_blocks_per_mp=MINCTA,
                )
    else:
        @cute.jit
        def launch(pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer,
                   pG: cute.Pointer, pBeta: cute.Pointer, pS: cute.Pointer,
                   pO: cute.Pointer, pZ: cute.Pointer, pW: cute.Pointer, stream):
            mQ = cute.make_tensor(pQ, cute.make_layout((BH, K), stride=(K, 1)))
            mK = cute.make_tensor(pK, cute.make_layout((BH, K), stride=(K, 1)))
            mV = cute.make_tensor(pV, cute.make_layout((BH, K), stride=(K, 1)))
            mG = cute.make_tensor(pG, cute.make_layout((BH, K), stride=(K, 1)))
            mBeta = cute.make_tensor(pBeta, cute.make_layout(BH))
            mS = cute.make_tensor(pS, cute.make_layout((BH, K, V), stride=(K * V, V, 1)))
            mO = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
            mZ = cute.make_tensor(pZ, cute.make_layout((BH, V), stride=(V, 1)))
            mW = cute.make_tensor(pW, cute.make_layout(V))
            kda_decode_gated_kernel(mQ, mK, mV, mG, mBeta, mS, mO, mZ, mW).launch(
                grid=[BH, 1, 1], block=[NT, 1, 1], stream=stream,
                min_blocks_per_mp=MINCTA,
            )

    return launch


# ---------------------------------------------------------------------------
# Host drivers: compile once per flag combo + launch shape, dispatch with a
# few microseconds of host work.
# ---------------------------------------------------------------------------
# `torch._C._cuda_getCurrentRawStream` is the cheap stream query (0.05 us);
# `torch.cuda.current_stream()` costs 2.0 us because it builds a Stream object.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream          # type: ignore[attr-defined]
except AttributeError:                                        # pragma: no cover
    def _raw_stream(dev):
        return torch.cuda.current_stream(dev).cuda_stream


_PTR_DTYPES = (cutlass.BFloat16,   # q
               cutlass.BFloat16,   # k
               cutlass.BFloat16,   # v
               cutlass.Float32,    # g
               cutlass.Float32,    # beta
               cutlass.Float32,    # S
               cutlass.BFloat16)   # o

# Gated-body pointer packs. The first three slots are q/k/v OR
# mixed_qkv/conv_weight/conv_state -- same dtypes either way. o sits at slot 6;
# gated combos append z and w.
_PTR_DTYPES_GATED = _PTR_DTYPES + (cutlass.BFloat16,   # z
                                   cutlass.Float32)    # w


def spill_bytes(comp):
    """Per-thread local-memory (spill) bytes of a compiled program, or None.

    This is the whole basis of the autoconfig: `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`
    on the loaded kernel is exactly "did ptxas have to spill", read straight off
    the binary this toolchain produced. It costs one driver query at compile time
    and nothing at all at runtime, and unlike a timing-based autotune it is
    deterministic and immune to a noisy GPU.

    Reaching the kernel handle uses DSL internals (`comp.to(...)` ->
    `jit_module.cuda_library` -> `cuLibraryGetKernel`), so every step is guarded:
    any failure returns None and the caller falls back to the portable config,
    which is exactly the behaviour of the package before this probe existed.
    """
    try:
        import cuda.bindings.driver as _drv
        from cutlass.base_dsl.runtime import cuda as _ch

        ex = comp.to(None)
        libs = ex.jit_module.cuda_library
        syms = list(comp.kernel_info.keys())
        if not libs or not syms:
            return None
        worst = -1
        for sym in syms:
            for lib in libs:
                try:
                    fn = _ch.get_function_from_kernel(
                        _ch.get_library_kernel(lib, sym))
                except Exception:
                    continue        # symbol lives in a different library
                err, val = _drv.cuFuncGetAttribute(
                    _drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    fn)
                if int(err) != 0:
                    return None
                worst = max(worst, int(val))
        return None if worst < 0 else worst
    except Exception:                # pragma: no cover - probe is best-effort
        return None


# The autoconfig decision is a property of the *toolchain*, not of the shape or
# the flag combo, so it is resolved once per process and reused. Each later
# compilation still verifies that the chosen config did not spill for it, which
# costs one driver query.
_AUTO_CHOICE = None      # (vw, mincta) once resolved
_AUTO_LOG = []           # [(vw, mincta, spill_bytes, verdict)] for reporting


def chosen_config():
    """(vw, mincta, how) actually in use for the gated body."""
    if _PINNED:
        return (GVW, GMINCTA, "pinned by environment")
    if not AUTOCFG:
        return (GVW, GMINCTA, "autoconfig disabled")
    if _AUTO_CHOICE is None:
        return (GVW, GMINCTA, "not yet compiled")
    return (_AUTO_CHOICE[0], _AUTO_CHOICE[1], "autoconfig (zero-spill probe)")


class _Launch:
    """A launch site: compiled program + a pre-built C argument pack.

    The pack holds the *addresses* of the `ctypes.c_void_p` pointer slots.
    Those addresses never change, so a call only has to store the current
    tensors' device pointers into the slots and invoke the compiled entry --
    no dlpack, no descriptor build, no argument marshalling. Every launch
    therefore reads whatever the caller's live tensors currently hold; nothing
    about the data is cached.
    """

    __slots__ = ("comp", "ptrs", "descs", "stream", "adapted", "exe_args",
                 "packed", "capi", "cures", "raw_stream", "B", "H", "dev", "fast")

    def __init__(self, comp, ptrs, stream, adapted, exe_args, raw_stream, B, H, dev):
        self.comp = comp
        self.ptrs = tuple(ptrs)
        self.descs = tuple(p._desc for p in ptrs)
        self.stream = stream            # keep alive: exe_args references it
        self.adapted = adapted          # keep alive: adapter-owned buffers
        self.exe_args = exe_args
        self.raw_stream = raw_stream
        self.B, self.H, self.dev = B, H, dev
        # Private-API fast path: reuse one packed ctypes array and call the
        # compiled C entry directly. Falls back to the public call if the
        # DSL internals ever move.
        self.fast = False
        try:
            ex = comp.to(None)
            src = ex._get_invoke_packed_args(exe_args)
            packed = (ctypes.c_void_p * len(src))()
            for i in range(len(src)):
                packed[i] = src[i]
            self.packed = packed
            self.capi = ex.capi_func
            self.cures = ex.cuda_result
            self.comp = ex               # hold the executor alive
            self.fast = self.cures is not None
        except Exception:                # pragma: no cover
            self.packed = self.capi = self.cures = None

    def fire(self):
        if self.fast:
            self.capi(self.packed)
            err = self.cures.value
            if err:
                raise RuntimeError(f"CUDA error {err} launching kda decode kernel")
        else:                            # pragma: no cover
            self.comp.run_compiled_program(self.exe_args)


class KdaDecode:
    """Host driver for the plain path (gated=False, fuse_conv=False). Compiles
    once per (BH, device) and launches with ~2.8 us of host work, which is what
    the importable `kda_decode_step` API costs."""

    def __init__(self):
        self._comp = {}      # (BH, cs, dev) -> compiled program
        self._sites = {}     # (B, H, dev, raw_stream) -> _Launch
        self._last = None    # single-entry fast lookup

    def _compiled(self, BH, cs, dev, args, stream):
        key = (BH, cs, dev, VEC)
        comp = self._comp.get(key)
        if comp is None:
            if VEC == 1:
                comp = cute.compile(
                    make_launcher(BH, gated=False, fuse_conv=False, cs=cs),
                    *args, stream)
            else:
                # VEC=2: legacy single-CTA-per-(b,h) float4 kernel (== CS=1)
                # VEC=0: scalar one-thread-per-column kernel
                comp = cute.compile(kda_decode_launch, *args, stream)
            self._comp[key] = comp
        return comp

    def _site(self, B, H, dev, rs):
        site = self._sites.get((B, H, dev, rs))
        if site is None:
            BH = B * H
            cs = _pick_cs(BH)
            ptrs = [make_ptr(dt, 0, cute.AddressSpace.gmem, assumed_align=16)
                    for dt in _PTR_DTYPES]
            stream = cuda.CUstream(rs)
            comp = self._compiled(BH, cs, dev, ptrs, stream)
            exe_args, adapted = comp.generate_execution_args(*ptrs, stream)
            site = _Launch(comp, ptrs, stream, adapted, exe_args, rs, B, H, dev)
            self._sites[(B, H, dev, rs)] = site
        self._last = site
        return site

    def prepare(self, q, k, v, g, beta, S):
        """Return a callable that launches on these exact tensors with no Python
        argument work at all, plus the output tensor it writes. Used for
        pure-kernel timing; the kernel still reads live tensor contents."""
        B, H, D = q.shape
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=q.device)
        dev = q.device.index
        site = self._site(B, H, dev, _raw_stream(dev))
        d = site.descs
        for slot, t in zip(d, (q, k, v, g, beta, S, o)):
            slot.value = t.data_ptr()
        return site.fire, o

    def __call__(self, q, k, v, g, beta, S):
        B = q.shape[0]
        H = q.shape[1]
        dev = q.device.index
        rs = _raw_stream(dev)
        site = self._last
        if (site is None or site.B != B or site.H != H or site.dev != dev
                or site.raw_stream != rs):
            assert q.shape[2] == K and S.shape[-1] == V and S.shape[-2] == K
            site = self._site(B, H, dev, rs)
        # Contiguity is required: the kernel indexes the caller's buffers
        # directly with static row-major layouts (and updates S in place).
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
                and g.is_contiguous() and beta.is_contiguous()
                and S.is_contiguous()):
            raise ValueError("kda_decode_step requires contiguous inputs")
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=q.device)
        d = site.descs
        d[0].value = q.data_ptr()
        d[1].value = k.data_ptr()
        d[2].value = v.data_ptr()
        d[3].value = g.data_ptr()
        d[4].value = beta.data_ptr()
        d[5].value = S.data_ptr()
        d[6].value = o.data_ptr()
        site.fire()
        return o


class KdaDecodeGatedBody:
    """Host driver for the gated-body combos: (gated, fuse_conv) in
    {(1,0), (0,1), (1,1)}. Compiles once per launch shape and device and
    launches with ~3 us of host work, which is what the importable
    `kda_decode_gated` / `kda_decode_conv_*` APIs cost."""

    def __init__(self, gated: bool, fuse_conv: bool):
        assert gated or fuse_conv, "plain path is served by KdaDecode"
        self.gated = gated
        self.fuse_conv = fuse_conv
        self.dtypes = _PTR_DTYPES_GATED if gated else _PTR_DTYPES
        self._comp = {}      # shape key + dev -> compiled program
        self._sites = {}     # (B, H, dev, raw_stream) -> _Launch
        self._last = None    # single-entry fast lookup

    def _make(self, B, H, vw, mincta):
        return make_launcher(B * H, vw, mincta, gated=self.gated,
                             fuse_conv=self.fuse_conv, B=B, H=H)

    def _compiled(self, B, H, dev, args, stream):
        # fuse_conv bakes (B, H) into the packed-channel indexing; the
        # non-conv body only depends on BH (same key as pre-merge).
        key = ((B, H) if self.fuse_conv else B * H, dev)
        comp = self._comp.get(key)
        if comp is None:
            if _PINNED or not AUTOCFG:
                comp = cute.compile(
                    self._make(B, H, GVW, GMINCTA), *args, stream)
            else:
                comp = self._autocompile(B, H, args, stream)
            self._comp[key] = comp
        return comp

    def _autocompile(self, B, H, args, stream):
        """Compile candidates in preference order; keep the first that does not
        spill. See `spill_bytes` and the `AUTOCFG` comment for why zero spill is
        the right acceptance test."""
        global _AUTO_CHOICE
        # Once the toolchain has been characterised, go straight to the winner --
        # but still verify it for this shape, since the body is re-specialised
        # per shape. If it spilled here after all, resume walking the ladder.
        order = list(CANDIDATES)
        if _AUTO_CHOICE is not None and _AUTO_CHOICE in order:
            order = order[order.index(_AUTO_CHOICE):]
        comp = None
        for i, (vw, mc) in enumerate(order):
            comp = cute.compile(self._make(B, H, vw, mc), *args, stream)
            sp = spill_bytes(comp)
            last = i == len(order) - 1
            if sp == 0:
                verdict = "accepted (zero spill)"
            elif last:
                # The ladder ends on the portable config, which has never spilled
                # on any toolchain measured; take it whatever the probe said.
                verdict = "accepted (end of ladder)"
            else:
                # A positive spill count rejects. So does an unavailable probe:
                # without evidence, never gamble on the forced launch bound.
                _AUTO_LOG.append((vw, mc, sp, "rejected: spills" if sp
                                  else "rejected: probe unavailable"))
                continue
            _AUTO_LOG.append((vw, mc, sp, verdict))
            _AUTO_CHOICE = (vw, mc)
            return comp
        return comp

    def _site(self, B, H, dev, rs):
        site = self._sites.get((B, H, dev, rs))
        if site is None:
            ptrs = [make_ptr(dt, 0, cute.AddressSpace.gmem, assumed_align=16)
                    for dt in self.dtypes]
            stream = cuda.CUstream(rs)
            comp = self._compiled(B, H, dev, ptrs, stream)
            exe_args, adapted = comp.generate_execution_args(*ptrs, stream)
            site = _Launch(comp, ptrs, stream, adapted, exe_args, rs, B, H, dev)
            self._sites[(B, H, dev, rs)] = site
        self._last = site
        return site

    def _check_shapes(self, tensors, B, H):
        """Full shape/dtype validation, run only on a site miss."""
        if self.fuse_conv:
            x, cw, cst, g, beta, S = tensors[:6]
            assert x.shape == (B, 3 * H * K) and x.dtype == torch.bfloat16
            assert cw.shape == (3 * H * K, 4) and cw.dtype == torch.bfloat16
            assert cst.shape == (B, 3 * H * K, 3) and cst.dtype == torch.bfloat16
        else:
            q, k, v, g, beta, S = tensors[:6]
            assert q.shape[2] == K and k.shape == q.shape and v.shape == q.shape
        assert S.shape == (B, H, K, V) and g.shape == (B, H, K)
        if self.gated:
            z, w = tensors[6], tensors[7]
            assert z.shape == (B, H, V) and w.shape == (V,)

    def prepare(self, *tensors):
        """Return a callable that launches on these exact tensors with no Python
        argument work at all, plus the output tensor it writes. `tensors` is
        the input tuple of the public API (o is allocated here at slot 6)."""
        if self.fuse_conv:
            B, H = tensors[5].shape[0], tensors[5].shape[1]
        else:
            B, H = tensors[0].shape[0], tensors[0].shape[1]
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=tensors[5].device)
        dev = tensors[5].device.index
        site = self._site(B, H, dev, _raw_stream(dev))
        args = tensors[:6] + (o,) + tensors[6:]
        for slot, t in zip(site.descs, args):
            slot.value = t.data_ptr()
        return site.fire, o

    def __call__(self, *tensors):
        S = tensors[5]
        B, H = S.shape[0], S.shape[1]
        dev = S.device.index
        rs = _raw_stream(dev)
        site = self._last
        if (site is None or site.B != B or site.H != H or site.dev != dev
                or site.raw_stream != rs):
            self._check_shapes(tensors, B, H)
            site = self._site(B, H, dev, rs)
        # Contiguity is required: the kernel indexes the caller's buffers
        # directly with static row-major layouts (and updates S -- and, under
        # fuse_conv, conv_state -- in place).
        for t in tensors:
            if not t.is_contiguous():
                raise ValueError("b10 kda decode kernels require contiguous inputs")
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=S.device)
        d = site.descs
        d[0].value = tensors[0].data_ptr()
        d[1].value = tensors[1].data_ptr()
        d[2].value = tensors[2].data_ptr()
        d[3].value = tensors[3].data_ptr()
        d[4].value = tensors[4].data_ptr()
        d[5].value = S.data_ptr()
        d[6].value = o.data_ptr()
        if self.gated:
            d[7].value = tensors[6].data_ptr()
            d[8].value = tensors[7].data_ptr()
        site.fire()
        return o

    # Unrolled hot paths for the conv combos (named args, single chained
    # contiguity test, no per-call asserts -- shape/dtype validation runs in
    # `_check_shapes` on a site miss). Statement-for-statement the host path
    # of the conv-fused gen package, which is ~0.5 us/call cheaper than the
    # generic `__call__` above; the flag-off combos keep using `__call__`
    # so their host path is untouched.
    def call_conv(self, mixed_qkv, conv_weight, conv_state, g, beta, S):
        B = mixed_qkv.shape[0]
        H = g.shape[1]
        dev = g.device.index
        rs = _raw_stream(dev)
        site = self._last
        if (site is None or site.B != B or site.H != H or site.dev != dev
                or site.raw_stream != rs):
            self._check_shapes(
                (mixed_qkv, conv_weight, conv_state, g, beta, S), B, H)
            site = self._site(B, H, dev, rs)
        if not (mixed_qkv.is_contiguous() and conv_weight.is_contiguous()
                and conv_state.is_contiguous() and g.is_contiguous()
                and beta.is_contiguous() and S.is_contiguous()):
            raise ValueError("kda_decode_conv_step requires contiguous inputs")
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=g.device)
        d = site.descs
        d[0].value = mixed_qkv.data_ptr()
        d[1].value = conv_weight.data_ptr()
        d[2].value = conv_state.data_ptr()
        d[3].value = g.data_ptr()
        d[4].value = beta.data_ptr()
        d[5].value = S.data_ptr()
        d[6].value = o.data_ptr()
        site.fire()
        return o

    def call_conv_gated(self, mixed_qkv, conv_weight, conv_state, g, beta,
                        S, z, w):
        B = mixed_qkv.shape[0]
        H = g.shape[1]
        dev = g.device.index
        rs = _raw_stream(dev)
        site = self._last
        if (site is None or site.B != B or site.H != H or site.dev != dev
                or site.raw_stream != rs):
            self._check_shapes(
                (mixed_qkv, conv_weight, conv_state, g, beta, S, z, w), B, H)
            site = self._site(B, H, dev, rs)
        if not (mixed_qkv.is_contiguous() and conv_weight.is_contiguous()
                and conv_state.is_contiguous() and g.is_contiguous()
                and beta.is_contiguous() and S.is_contiguous()
                and z.is_contiguous() and w.is_contiguous()):
            raise ValueError("kda_decode_conv_gated requires contiguous inputs")
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=g.device)
        d = site.descs
        d[0].value = mixed_qkv.data_ptr()
        d[1].value = conv_weight.data_ptr()
        d[2].value = conv_state.data_ptr()
        d[3].value = g.data_ptr()
        d[4].value = beta.data_ptr()
        d[5].value = S.data_ptr()
        d[6].value = o.data_ptr()
        d[7].value = z.data_ptr()
        d[8].value = w.data_ptr()
        site.fire()
        return o


_ENGINE = KdaDecode()
_ENGINE_GATED = KdaDecodeGatedBody(gated=True, fuse_conv=False)
_ENGINE_CONV = KdaDecodeGatedBody(gated=False, fuse_conv=True)
_ENGINE_CONV_GATED = KdaDecodeGatedBody(gated=True, fuse_conv=True)


# ---------------------------------------------------------------------------
# MANDATORY importable API: updates S (and conv_state) in place, returns o.
# ---------------------------------------------------------------------------
def kda_decode_step(q, k, v, g, beta, S):
    """Fused KDA decode step. Updates S in place, returns o [B,H,128] bf16."""
    assert q.shape[-1] == 128 and v.shape[-1] == 128 and S.shape[-1] == 128, (
        "this CuTeDSL kernel is compiled for head_dim K=V=128, got "
        f"K={q.shape[-1]} V={v.shape[-1]}"
    )
    return _ENGINE(q, k, v, g, beta, S)


def kda_decode_gated(q, k, v, g, beta, S, z, w):
    """Fused KDA decode step + gated RMSNorm.

    q, k, v, z : [B,H,128] bf16     g : [B,H,128] fp32 log-decay (<= 0)
    beta       : [B,H]     fp32     S : [B,H,128,128] fp32, updated IN PLACE
    w          : [128]     fp32     -> o : [B,H,128] bf16 (post-norm)
    """
    assert q.shape[-1] == 128 and v.shape[-1] == 128 and w.numel() == 128, (
        "this CuTeDSL kernel is compiled for head_dim K=V=128, got "
        f"K={q.shape[-1]} V={v.shape[-1]}"
    )
    return _ENGINE_GATED(q, k, v, g, beta, S, z, w)


def kda_decode_conv_step(mixed_qkv, conv_weight, conv_state, g, beta, S):
    """Width-4 causal conv + SiLU on packed q/k/v, then the fused decode step.

    mixed_qkv   : [B, 3*H*128] bf16   RAW pre-conv packed [q | k | v]
    conv_weight : [3*H*128, 4] bf16   depthwise conv weights, no bias
    conv_state  : [B, 3*H*128, 3] bf16 last 3 RAW tokens, updated IN PLACE
    g           : [B,H,128] fp32 log-decay (NOT convolved)
    beta        : [B,H] fp32          S : [B,H,128,128] fp32, updated IN PLACE
    -> o : [B,H,128] bf16 (pre-norm recurrence output)

    Shapes/dtypes (incl. the K=V=128 bake) are validated on a launch-site
    miss; the hot path adds no per-call checks beyond contiguity.
    """
    return _ENGINE_CONV.call_conv(mixed_qkv, conv_weight, conv_state,
                                  g, beta, S)


def kda_decode_conv_gated(mixed_qkv, conv_weight, conv_state, g, beta, S, z, w):
    """Width-4 causal conv + SiLU + fused decode + gated RMSNorm, one launch.

    Same conv contract as `kda_decode_conv_step`, same gated-norm contract as
    `kda_decode_gated` (z [B,H,128] bf16, w [128] fp32) -> POST-norm o bf16.
    The layer-step counterpart of TRT-LLM ``fused_kda_decode`` (which instead
    takes the raw gate/beta and activates them in-kernel).

    Shapes/dtypes (incl. the K=V=128 bake) are validated on a launch-site
    miss; the hot path adds no per-call checks beyond contiguity.
    """
    return _ENGINE_CONV_GATED.call_conv_gated(mixed_qkv, conv_weight,
                                              conv_state, g, beta, S, z, w)


# ---------------------------------------------------------------------------
# Self-test: all four (gated, fuse_conv) combos.
#   fuse_conv=False vs the pre-merge kernels' math (torch fp32), <=1e-3.
#   fuse_conv=True vs SGLang causal_conv1d_update(silu) + the fuse_conv=False
#   kernels on the conv output, <=2e-2 max-abs / cosine >=0.999, plus the
#   in-place conv_state update (BIT-exact: both roll the raw window).
#   NB the composed reference rounds the post-conv q/k/v to bf16 (the Triton
#   conv kernel's output dtype) while the fused path keeps them fp32 on-chip,
#   so a small bf16-rounding-level gap on o/S is expected and correct -- the
#   fused path is the strictly more accurate of the two.
# ---------------------------------------------------------------------------
def _ref_decode(q, k, v, g, beta, S):
    """The pre-merge kda_decode_step math in torch fp32 (S updated in place)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True))
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True))
    Sd = S * torch.exp(g).unsqueeze(-1)
    A = torch.einsum("bhk,bhkv->bhv", qn, Sd)
    Bv = torch.einsum("bhk,bhkv->bhv", kn, Sd)
    u = beta.unsqueeze(-1) * (vf - Bv)
    P = (kn * qn).sum(-1)
    o = (A + u * P.unsqueeze(-1)) * (K ** -0.5)
    S.copy_(Sd + kn.unsqueeze(-1) * u.unsqueeze(-2))
    return o


def _ref_gated_norm(r, z, w):
    """The pre-merge gated-RMSNorm epilogue math in torch fp32."""
    rf = r.float()
    sc = torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + EPS)
    return rf * sc * w.float() * torch.sigmoid(z.float())


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _self_test():
    torch.manual_seed(0)
    dev = "cuda"
    failures = []

    def check(name, got, want, atol, cos_min=None):
        err = (got.float() - want.float()).abs().max().item()
        cos = _cos(got, want)
        ok = err <= atol and (cos_min is None or cos >= cos_min)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: max-abs {err:.3e} (tol {atol:g}) "
              f"cosine {cos:.6f}")
        if not ok:
            failures.append(name)

    for B, H in ((3, 12), (2, 5)):
        print(f"-- self-test B={B} H={H} --")
        q = (0.1 * torch.randn(B, H, K, device=dev)).bfloat16()
        k = (0.1 * torch.randn(B, H, K, device=dev)).bfloat16()
        v = (0.1 * torch.randn(B, H, K, device=dev)).bfloat16()
        g = (-0.5 * torch.rand(B, H, K, device=dev)).float().contiguous()
        beta = torch.sigmoid(torch.randn(B, H, device=dev)).float().contiguous()
        S0 = (0.5 * torch.randn(B, H, K, V, device=dev)).float().contiguous()
        z = (0.1 * torch.randn(B, H, V, device=dev)).bfloat16()
        # w kept small so post-norm outputs stay < 0.25: at that magnitude one
        # bf16 ulp is <= 2^-11, which keeps the <=1e-3 gate meaningful even if
        # a value straddles a rounding boundary.
        w = (0.05 + 0.05 * torch.rand(V, device=dev)).float().contiguous()

        # combo 1: gated=False, fuse_conv=False vs the old kernel's math
        S1 = S0.clone()
        o1 = kda_decode_step(q, k, v, g, beta, S1)
        Sr = S0.clone()
        orf = _ref_decode(q, k, v, g, beta, Sr)
        check("gated=0 conv=0  o", o1, orf.bfloat16(), 1e-3)
        check("gated=0 conv=0  S", S1, Sr, 1e-3)

        # combo 2: gated=True, fuse_conv=False vs old decode + gated-norm math
        S2 = S0.clone()
        o2 = kda_decode_gated(q, k, v, g, beta, S2, z, w)
        Sr2 = S0.clone()
        og = _ref_gated_norm(_ref_decode(q, k, v, g, beta, Sr2), z, w)
        check("gated=1 conv=0  o", o2, og.bfloat16(), 1e-3)
        check("gated=1 conv=0  S", S2, Sr2, 1e-3)

        # conv combos: composed reference = SGLang causal_conv1d_update(silu)
        # then the fuse_conv=False kernels on its output.
        try:
            from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
                causal_conv1d_update,
            )
        except ImportError as exc:  # pragma: no cover
            print(f"  [SKIP] conv combos: sglang unavailable ({exc})")
            continue

        HK = H * K
        mixed = (0.2 * torch.randn(B, 3 * HK, device=dev)).bfloat16().contiguous()
        cw = (0.1 * torch.randn(3 * HK, 4, device=dev)).bfloat16().contiguous()
        cst0 = (0.1 * torch.randn(B, 3 * HK, 3, device=dev)).bfloat16().contiguous()

        def conv_ref():
            cst_r = cst0.clone()
            out = causal_conv1d_update(
                mixed.clone(), cst_r, cw, bias=None, activation="silu")
            if out.dim() == 3:
                out = out[..., 0]
            qc = out[:, :HK].reshape(B, H, K).contiguous()
            kc = out[:, HK:2 * HK].reshape(B, H, K).contiguous()
            vc = out[:, 2 * HK:].reshape(B, H, K).contiguous()
            return qc, kc, vc, cst_r

        # combo 3: gated=False, fuse_conv=True
        qc, kc, vc, cst_r = conv_ref()
        S3 = S0.clone()
        cst3 = cst0.clone()
        o3 = kda_decode_conv_step(mixed, cw, cst3, g, beta, S3)
        S3r = S0.clone()
        o3r = kda_decode_step(qc, kc, vc, g, beta, S3r)
        check("gated=0 conv=1  o", o3, o3r, 2e-2, cos_min=0.999)
        check("gated=0 conv=1  S", S3, S3r, 2e-2)
        check("gated=0 conv=1  conv_state", cst3, cst_r, 0.0)

        # combo 4: gated=True, fuse_conv=True
        S4 = S0.clone()
        cst4 = cst0.clone()
        o4 = kda_decode_conv_gated(mixed, cw, cst4, g, beta, S4, z, w)
        S4r = S0.clone()
        o4r = kda_decode_gated(qc, kc, vc, g, beta, S4r, z, w)
        check("gated=1 conv=1  o", o4, o4r, 2e-2, cos_min=0.999)
        check("gated=1 conv=1  S", S4, S4r, 2e-2)
        check("gated=1 conv=1  conv_state", cst4, cst_r, 0.0)

    if failures:
        raise SystemExit(f"self-test FAILED: {failures}")
    print("self-test: all four (gated, fuse_conv) combos PASS")


if __name__ == "__main__":
    _self_test()
