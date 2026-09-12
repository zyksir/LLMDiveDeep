# SPDX-License-Identifier: Apache-2.0
"""
blackwell_bf16_kda_save_ssm_conv_gated — KDA (Kimi Delta Attention)
speculative-decode MTP-verify step in the **SAVE-SSM** scheme (KDA.md "1a")
with **BOTH** the width-4 depthwise causal-conv1d + SiLU input stage **and** the
gated-RMSNorm output epilogue fused into the same kernel, for B200 (sm_100a),
CuTeDSL 4.5.2.

ONE launch replaces the production three-launch chain
`causal_conv1d_update -> fused_sigmoid_gating_delta_rule_update(snapshot)
 -> FusedRMSNormGated`.

Per (request n, head h), sequentially over the request's T draft tokens, with
`raw` = the packed pre-conv projections `mixed_qkv [TT, 3*H*K]` and the request's
3 committed history tokens `conv_state[n]`:

    x[t,c] = SiLU( sum_{j<4} conv_weight[c,j] * raw(t-3+j, c) )   (FUSED conv)
    q[t], k[t], v[t] = unpack(x[t])                (c = ti*H*K + h*K + d)
    qn = q[t]/||q[t]||          kn = k[t]/||k[t]||        (L2 norms fused)
    S  = diag(exp(g[t])) @ S                              (decay)
    u  = beta[t] * (v[t] - S^T kn)
    S  = S + kn u^T                                       (rank-1 update)
    r  = S^T qn / sqrt(K)                                 (pre-norm output)
    o[t]        = r * rsqrt(mean(r^2) + 1e-5) * w * sigmoid(z[t])   (FUSED)
    snapshot[t] = S                                       (FULL post-token state)

`S0` **and** `conv_state` are read-only (rollback semantics: acceptance is
unknown at kernel time, so the host re-derives both after sampling — `S0` from
`snapshot`, the conv window from the raw tokens it already has).

The FULL post-token state is emitted to `snapshot[t]` after
**every** token, which is what makes this kernel a pure HBM-store problem:

    HBM per state element:  4 (read S0)  +  4*T (snapshots)

so the design goal is to stream that store at HBM speed with the recurrence AND
the epilogue hidden underneath it.

=============================================================================
Decomposition — `CFG_W8`, and why the *gated* variant cannot use `CFG_V2`
=============================================================================
    grid = (V_SPLITS, H, B)   block = (256, 1, 1) = 8 warps
    CTA      owns S[b, h, :, 0:128]   (all V columns; VSPLITS == 1)
    warp w   owns key channels [16w, 16w+16)
    lane     owns 4 V columns   lane + 32*(0..3)
    -> 16 (channel) x 4 (column) fp32 register tile per thread (SR = 64)

The un-gated snapshot kernel's fastest tile was `CFG_V2` = (NWARP=4, CPT=2),
which splits V in half across **two CTAs** (`VSPLITS = 2`). The RMS reduction
`sum_c r_c^2` spans all 128 output channels, so a V-split CTA holds only half of
it and would need a cross-CTA (cluster/DSMEM) reduction every token. Instead the
tile is changed to `CFG_W8` = (NWARP=8, CPT=4), which keeps the same 64 state
registers per thread and the same 16-warps-per-SM occupancy but gives **one CTA
the whole 128-wide output row**. In the un-gated kernel that config measured
0.3174 ms vs 0.3160 ms for CFG_V2 — a 0.6% tile tax, paid once, in exchange for
an epilogue with **zero extra CTA barriers**.

The epilogue itself rides on a property of the existing reduction: with
`SPT == 1`, warps 0-3 produce `u` and warps 4-7 produce the `q`-partial sums, and
then **warp 0 alone** already stores all 128 output columns (`r == 0` is exactly
one 32-lane group covering `CPT * 32 = 128` columns). So warp 0 holds the entire
pre-norm row `r` in registers at the point of the store: `sum r^2` is one 5-step
warp butterfly, and the scale is one `rsqrt`. No barrier, no SMEM round-trip, no
second kernel.

`w * sigmoid(z[t])` is precomputed into SMEM by the **prologue** (which is
already 8-warp-parallel over the chunk's 8 tokens) rather than loaded in the
epilogue: the epilogue is the last ~20 instructions before the next token's
dependent work, and a late L2 round-trip there cannot be hidden. `w` itself is
loaded into registers at the very top of the kernel, next to the `S0` loads.

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

=============================================================================
Where the conv goes
=============================================================================
The conv is a *per-channel* width-4 FIR over the token axis, so it fits the
existing prologue exactly: warp `w` already owns **all 128 channels of one
token** (4 per lane), which is what made the L2 norms a pure warp butterfly. The
conv adds, per lane and per tensor `ti in {q,k,v}`, four taps over the same 4
channels:

    raw(local i-3+j)  =  mixed_qkv[t-3+j]   if  i-3+j >= 0
                         conv_state[n][.., i+j]  otherwise   (i = local index)

The tap predicate is a function of the *token*, which is `warp_idx` here, so
every branch is **warp-uniform** — no divergence, and the 16 loads of a tensor's
window are all independent so they issue as one burst.

`conv_weight` is channel-indexed only, so it is hoisted **once per kernel** into
SMEM (`sCW`, one head's 3*4*128 bf16 = 3 KB, tap-major so a per-tap read has the
lane-varying channel contiguous) — putting it in registers instead would cost 48
registers per thread live across the whole kernel, and the state tile already
caps occupancy at 2 CTAs (see 2. above). bf16 rather than fp32 because SMEM is a
hard occupancy cliff here: 2 CTAs/SM need <= 32768 B/block in the 64 KB carveout,
and fp32 weights measured 32896.

`conv_state` is READ-ONLY. The window of the next step is re-derived by the host
after sampling, exactly like `S0`.
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
WIDTH = 4                # causal-conv1d kernel width (depthwise, no bias)
CSL = WIDTH - 1          # conv_state length: the 3 committed history tokens
NTI = 3                  # mixed_qkv is packed as [q | k | v]

# How many conv taps of one tensor are staged in registers at once. This is a
# REGISTER-CLIFF knob, not a style choice: the state tile alone is 64 registers
# per thread and 2 CTAs/SM need `registers_per_thread_allocated <= 128`, so the
# 16-register "stage all 4 taps" version measured 136 allocated -> ncu
# `occupancy_limit_registers = 1` -> HALF the occupancy and 1.72 ms at B=64 T=8.
# The cliff was actually the fp32 `sCW` SMEM (see below); with bf16 weights and
# scalar reads, TG=4 fits in 128 registers too and is the fastest measured
# (361.0 / 363.1 / 372 us at TG = 4 / 2 / 1, H=96 B=64 T=4).
CONV_TAP_GROUP = int(os.environ.get("KDA_CONV_TG", "4"))

# Whether the conv stage walks channel PAIRS (one 32-bit load covers channels
# 2p and 2p+1, so each raw load fills a whole 128 B L1TEX line and the weight
# read becomes one pre-swizzled vector) instead of one channel per load.
# Halves the conv's L1TEX wavefronts and its address arithmetic; costs a 2-way
# bank conflict on the fp32 sDG/sV writes and halves memory-level parallelism.
# Measured both ways -- see optimization.md cycle 4.
CONV_PAIR = os.environ.get("KDA_CONV_PAIR", "0") == "1"

# Whether a token's conv is SPLIT across two warps when the chunk leaves warps
# idle (2*ntok <= NWARP, i.e. every bench shape except T=8): q+k on the even warp
# of the pair, v on the odd one. The whole CTA waits on the slowest prologue warp
# at the barrier, so this shortens the critical chain from 3 conv tensors to 2
# without changing any wavefront count -- a pure latency fix, which is what
# cycles 2 and 4 established is the thing that matters in this prologue.
# `0` compiles the single-warp prologue verbatim (bit-identical to attempt_000).
CONV_WARP_SPLIT = os.environ.get("KDA_CONV_SPLIT", "1") == "1"

# Whether the request's 3 RAW history tokens (`conv_state`) are staged into SMEM
# ONCE per CTA instead of being re-read from global inside the per-token tap loop.
# `conv_state` is [B, D, 3] with the 3 taps CONTIGUOUS, so a channel-indexed read
# has consecutive lanes 6 B apart: one scalar load spans 192 B for 64 B of useful
# data -- 6 sectors per request where a coalesced load takes 2. Measured at
# H=96 B=64 T=2: 60 such requests per CTA burning 360 of the CTA's 2769 global
# load sectors (13%), and the count is FIXED per CTA (6 out-of-range tap-slots per
# request whatever T is), so it lands hardest on exactly the short-T shapes that
# still sit above the store-only floor. Staging it once also lifts the longest
# -latency load in the prologue out of the per-token dependent chain.
CONV_STATE_SMEM = os.environ.get("KDA_CS_SMEM", "1") == "1"

# Where the 64-per-thread `S0` load burst is issued relative to the first chunk's
# prologue. False (shipped) = at the very top; True = after the prologue, on the
# theory that the prologue's dependent chain (load q/k/g -> warp shuffles ->
# exp/rsqrt -> SMEM stores) should not sit behind 64 in-flight LDGs.
# MEASURED AND REJECTED: True is 5.9% SLOWER at B=64 T=2 (220.8 vs 208.3 us,
# reproduced twice) and a wash elsewhere. The S0 burst is the long, bandwidth-
# heavy stream and pass 1 blocks on it, so issuing it late delays every store in
# the CTA by more than the prologue chain costs. Kept as a switch for the record.
S0_AFTER_PROLOGUE = os.environ.get("KDA_S0_LATE", "0") == "1"

# Tile configuration `(NWARP, LK, CPT, VECST)`; everything else is derived:
#   LV  = 32/LK lanes over V,  VS = LV*CPT columns per CTA,  VSPLITS = V/VS,
#   KPT = K/(NWARP*LK) key channels per thread, SR = KPT*CPT state registers,
#   R   = NWARP*LK partials per column.
# The fused epilogue requires VSPLITS == 1 (the RMS reduction spans all V) and
# SPT == 1 (warp 0 alone owns the whole output row). CFG_W8 is the only tile that
# satisfies both at LK == 1; CFG_ROW satisfies VSPLITS == 1 but has SPT == 2, so
# it is only usable with `epi=False` (the un-fused ablation).
CFG_W8 = (8, 1, 4, False)    # NWARP=8, KPT=16, CPT=4, SR=64,  VS=128, R=8  <- shipped
CFG_W8V = (8, 1, 4, True)    # ... same tile, 128-bit state load/store (VECST)
CFG_ROW = (4, 1, 4, False)   # NWARP=4, KPT=32, CPT=4, SR=128, VS=128, R=4
CFG_V2 = (4, 1, 2, False)    # NWARP=4, KPT=32, CPT=2, SR=64,  VS=64,  R=4
CFG_W8K2 = (8, 2, 8, False)  # NWARP=8, KPT=8,  CPT=8, SR=64,  VS=128, R=16


def make_launcher(H: int, K: int, V: int, cfg=CFG_W8, epi: bool = True,
                  conv: bool = True):
    """Build a compiled-once launcher closure for a given (H, K, V, cfg).

    epi=False is the timing ablation: it writes the PRE-norm output r, i.e. what
    an un-fused kernel would emit before a separate gated-RMSNorm launch. It is
    never valid for --check.

    conv=False is the other timing ablation: the kernel reads POST-conv
    `q, k, v` directly (what today's framework path feeds it after a separate
    `causal_conv1d_update` launch) and ignores `mixed_qkv/conv_weight/conv_state`.
    Also never valid for --check.
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
    RSTRIDE = 2 * VS + (LV if LK > 1 else 0)
    LOG_LV = int(math.log2(LV))
    LOG_VS = int(math.log2(VS))
    LOG_CPT = int(math.log2(CPT))
    SPT = (2 * VS) // THREADS        # reduction slots per thread
    assert SPT * THREADS == 2 * VS
    ZPL = VS // 32                   # z / w elements per lane in the prologue
    assert ZPL * 32 == VS
    if epi:
        # The fused gated RMSNorm reduces r^2 over ALL V output channels of a
        # (token, head), and warp 0 must own that whole row.
        assert VSPLITS == 1, "fused gated RMSNorm needs the full V row in one CTA"
        assert SPT == 1, "fused gated RMSNorm needs the warp-0 store path"
        assert VS == V and CPT * LV == V
    if conv:
        # The conv produces the WHOLE post-conv v row of a token in the prologue
        # (the same lane->channel map as q/k), so a V-split CTA would either
        # convolve columns it does not own or duplicate the work.
        assert VSPLITS == 1, "fused causal conv needs the full V row in one CTA"
        assert K == V, "packed mixed_qkv assumes head_k_dim == head_v_dim"
        assert VS // 32 == CPL
    LOG_K = int(math.log2(K))
    CWPT = (NTI * K + THREADS - 1) // THREADS      # sCW staging rounds
    TG = CONV_TAP_GROUP
    assert WIDTH % TG == 0
    NG = WIDTH // TG                 # conv tap groups
    PAIR = CONV_PAIR and conv        # conv walks channel pairs
    # The split path needs the plain (non-PAIR) channel map, one token per warp
    # per chunk (TPW == 1, so `w >> sp` addresses the whole chunk), and at least
    # two warps to split across. It is conv-only: the conv-less ablation keeps the
    # single-warp prologue so the un-fused timing baselines stay unchanged.
    SPLIT = (conv and CONV_WARP_SPLIT and not PAIR and TPW == 1 and NWARP >= 2)
    # PAIR uses a different channel map for the conv_state read and is a rejected
    # variant kept only for reproducibility, so it stays on the global path.
    CSSM = conv and CONV_STATE_SMEM and not PAIR
    CSPT = (NTI * K + THREADS - 1) // THREADS      # sCS staging rounds
    CPP = CPL // 2                   # channel PAIRS per lane
    assert CPP * 2 == CPL

    @cute.kernel
    def blackwell_bf16_kda_save_ssm_conv_gated_kernel(
        mQ: cute.Tensor,      # [TT*H, K]          bf16  (row = t*H + h)
        mK: cute.Tensor,      # [TT*H, K]          bf16
        mV: cute.Tensor,      # [TT*H, V]          bf16
        mG: cute.Tensor,      # [TT*H, K]          fp32  (log decay, <= 0)
        mBeta: cute.Tensor,   # [TT*H]             fp32
        mS0: cute.Tensor,     # [B*H, K, V/4, 4]   fp32  (READ-ONLY in-state)
        mSnap: cute.Tensor,   # [TT*H, K, V/4, 4]  fp32  (post-token state, out)
        mO: cute.Tensor,      # [TT*H, V]          bf16  (POST-norm output)
        mCu: cute.Tensor,     # [B+1]              int32
        mZ: cute.Tensor,      # [TT*H, V]          bf16  (output gate)
        mW: cute.Tensor,      # [V]                fp32  (RMSNorm weight)
        mMix: cute.Tensor,    # [TT, 3, H, K]      bf16  (RAW pre-conv packed qkv)
        mCW: cute.Tensor,     # [3, H, K, 4]       bf16  (depthwise conv weight)
        mCS: cute.Tensor,     # [B, 3, H, K, 3]    bf16  (RAW history, READ-ONLY)
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
        cgrp = (vsi * VS >> LOG_CPT) + lane
        cg0 = (vsi * VS + lv) >> LOG_CPT
        cm0 = lv & (CPT - 1)
        dcg = LV >> LOG_CPT

        # ---- the incoming state tile. `load_s0()` is called either here or after
        # the first chunk's prologue (see S0_AFTER_PROLOGUE): 64 LDGs per thread
        # are enough to push the prologue's own 12 loads to the back of the queue,
        # and the prologue is a long dependent chain that the barrier waits on,
        # whereas `st` is not needed until pass 1.
        st = cute.make_fragment((CPT, KPT), FP32)

        # NB: the load is written out inline in both positions rather than through
        # a helper -- CuTeDSL rejects a closure that captures constexpr values
        # when it is called from inside dynamic control flow ("Function `load_s0`
        # is a closure that captures variable `CPT`").
        # 128-bit staging register for the VECST path (see the VECST comment at the
        # config table): the vector copy is done against THIS 4-element fragment
        # and the values are moved to/from `st` scalar-wise, so `st` itself is only
        # ever touched by scalar accesses and keeps its register promotion.
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

        # ---- the RMSNorm weight, hoisted to the very top. It is consumed in the
        # last handful of instructions of every token, where there is nothing left
        # to hide an L2 round-trip behind; issuing it here costs ZPL registers and
        # is amortised over every token of the request. (Same finding as the
        # decode+gated kernel, where hoisting z/w was worth 35% there.)
        fw = cute.make_fragment((ZPL,), FP32)
        if cutlass.const_expr(epi):
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
        # is a pure SMEM read.
        NZG = TMAX if epi else 1
        sZG = smem.allocate_tensor(FP32, cute.make_layout(
            (NZG, VS), stride=(VS, 1)))
        # conv_weight for THIS head only: [ti, TAP, channel] bf16 -- TAP-major, so
        # a per-tap read has the *channel* (= lane) contiguous and is
        # conflict-free. Deliberately SCALAR bf16 reads, not a pre-swizzled vector:
        # cycle 2 replaced these 48 reads/token with 12 wider ones, cut L1TEX
        # throughput 70.4% -> 66.0%, and was 4% SLOWER -- the prologue is
        # latency-bound, so halving its memory-level parallelism costs more than
        # the wavefronts save (see optimization.md).
        # bf16, not fp32, because SMEM is a HARD occupancy cliff here: 2 CTAs/SM
        # need `shared_mem_per_block_allocated <= 32768` in the 64 KB carveout and
        # the fp32 version measured 32896 -> ncu `occupancy_limit_shared_mem = 1`.
        NCW = NTI if conv else 1
        if cutlass.const_expr(PAIR):
            # [ti, tap-group, channel-PAIR, (channel-in-pair, tap-in-group)]:
            # one 2*TG-wide bf16 ld.shared serves BOTH channels of the pair for
            # the whole tap group.
            sCW = smem.allocate_tensor(BF16, cute.make_layout(
                (NCW, NG, K // 2, 2 * TG),
                stride=(NG * (K // 2) * 2 * TG, (K // 2) * 2 * TG, 2 * TG, 1)),
                16)
        else:
            sCW = smem.allocate_tensor(BF16, cute.make_layout(
                (NCW, WIDTH, K), stride=(WIDTH * K, K, 1)), 16)

        # The request's 3 RAW history tokens for THIS head: [ti][tap][channel],
        # channel contiguous so the prologue's read is bank-conflict-free and the
        # lane-varying term is `lane` alone. 3*3*128*2 = 2304 B, which keeps
        # `shared_mem_per_block_allocated` under the 32768 B 2-CTA cliff (see 5.
        # in the header) -- that cliff is why this stages only the HISTORY rows and
        # not the whole raw window, which would need 8448 B and break it.
        NCS = NTI if CSSM else 1
        sCS = smem.allocate_tensor(BF16, cute.make_layout(
            (NCS, CSL, K), stride=(CSL * K, K, 1)), 16)

        # ---- hoist conv_weight into SMEM once per kernel. It is indexed by
        # channel only (not by token), so this is 3*K*4 values read once instead
        # of re-read from L2 per token; keeping it in registers instead would cost
        # NTI*CPL*WIDTH = 48 registers live across the whole kernel, and the state
        # tile already caps occupancy at 2 CTAs.
        if cutlass.const_expr(conv):
            for idx in cutlass.range_constexpr(CWPT):
                p = tid + THREADS * idx
                if p < NTI * K:
                    pti = p >> LOG_K
                    pd = p & (K - 1)
                    for jj in cutlass.range_constexpr(WIDTH):
                        if cutlass.const_expr(PAIR):
                            sCW[pti, jj // TG, pd >> 1,
                                (pd & 1) * TG + (jj % TG)] = mCW[pti, h, pd, jj]
                        else:
                            sCW[pti, jj, pd] = mCW[pti, h, pd, jj]
            # ---- and the 3 RAW history tokens, also once per kernel. A thread
            # takes ONE channel and reads its CSL contiguous taps, so the warp
            # covers 32*CSL*2 = 192 B with no gaps -- the same 6 sectors that a
            # single channel-indexed tap load costs, but paid 6x fewer times, and
            # paid HERE, next to the S0 burst, where 64 in-flight LDGs per thread
            # already hide it, instead of inside the per-token dependent chain.
            if cutlass.const_expr(CSSM):
                for idx in cutlass.range_constexpr(CSPT):
                    p = tid + THREADS * idx
                    if p < NTI * K:
                        pti = p >> LOG_K
                        pd = p & (K - 1)
                        for m in cutlass.range_constexpr(CSL):
                            sCS[pti, m, pd] = mCS[b, pti, h, pd, m]
            cute.arch.sync_threads()

        qf = cute.make_fragment((CPL,), FP32)
        kf = cute.make_fragment((CPL,), FP32)
        gf = cute.make_fragment((CPL,), FP32)
        dg4 = cute.make_fragment((VEC,), FP32)
        qnf = cute.make_fragment((VECH, KPT // VECH), F16)
        knf = cute.make_fragment((VECH, KPT // VECH), F16)
        # conv staging: TG taps of ONE tensor over this lane's CPL channels, plus
        # the running conv accumulator. Both are dead outside the prologue, so
        # they overlap the recurrence's own live values -- but their PEAK is what
        # sets `registers_per_thread`, hence TG (see CONV_TAP_GROUP).
        NRF = TG if conv else 1
        rf = cute.make_fragment((NRF, CPL), FP32)
        cacc = cute.make_fragment((CPL,), FP32)
        rp = cute.make_fragment((2 if PAIR else 1,), BF16)        # raw pair
        cwg = cute.make_fragment((2 * TG if PAIR else 1,), BF16)  # tap group
        pk = cute.make_fragment((CPT,), FP32)
        pq = cute.make_fragment((CPT,), FP32)
        uu = cute.make_fragment((CPT,), FP32)
        rv = cute.make_fragment((CPT,), FP32)

        # Reduction storage: the 2*VS slots are dealt out to the THREADS threads,
        # SPT each; flat slot f = s*THREADS + tid -> which = f>>LOG_VS,
        # pos = f & (VS-1). A thread's own slots are pos(cc) = cc*LV + lv, which
        # is conflict-free for the scatter and consecutive for the gather.
        red_pos = tid & (VS - 1)
        if cutlass.const_expr(VECST):
            rcol = ((red_pos & (LV - 1)) << LOG_CPT) + (red_pos >> LOG_LV)
        else:
            rcol = red_pos

        nchunk = (tn + (TMAX - 1)) >> LOG_TMAX
        for c in cutlass.range(nchunk):
            cbase = t0 + c * TMAX
            ntok = cutlass.min(tn - c * TMAX, TMAX)

            # ================= prologue: warp w stages token j = w + rr*NWARP ====
            if cutlass.const_expr(SPLIT):
                # ---- TWO warps per token whenever the chunk leaves warps idle --
                # `sp` is 1 exactly when 2*ntok <= NWARP, from min/max so there is
                # no branch and no divide. Warp `w` then stages token `w >> sp`,
                # and in split mode the EVEN warp of a pair does q and k -- which
                # MUST share a warp, because ||q||, ||k|| and <q,k> are warp
                # butterflies -- while the ODD warp does v and the gate scale,
                # neither of which needs any cross-lane reduction. That shortens
                # the conv's serialized per-warp chain from 3 tensors to 2 on the
                # warp the whole CTA waits for at the barrier, and it costs zero
                # extra L1TEX wavefronts (the same loads, spread over 2x the
                # warps) -- which matters because cycles 2 and 4 showed this
                # prologue is latency-bound, not L1TEX-throughput-bound.
                # When 2*ntok > NWARP (T=8) sp is 0, `hlf` is 0, and both halves
                # run on the same warp exactly as before.
                sp = cutlass.min(
                    cutlass.max(I32(NWARP) - ntok * 2 + I32(1), I32(0)), I32(1))
                j = w >> sp
                hlf = w & sp
                if j < ntok:
                    row = (cbase + j) * H + h
                    li = c * TMAX + j
                    if hlf == 0:
                        # ---- q, k: conv, the L2 norms, <q,k>, and the publish --
                        nq = FP32(0.0)
                        nk = FP32(0.0)
                        kq = FP32(0.0)
                        for ti in cutlass.range_constexpr(2):
                            for gi in cutlass.range_constexpr(NG):
                                for jt in cutlass.range_constexpr(TG):
                                    lidx = li - CSL + gi * TG + jt
                                    if lidx >= 0:
                                        grow = cbase + j - CSL + gi * TG + jt
                                        for e in cutlass.range_constexpr(CPL):
                                            rf[jt, e] = mMix[
                                                grow, ti, h, lane + 32 * e].to(FP32)
                                    else:
                                        # history taps: SMEM (see CONV_STATE_SMEM)
                                        if cutlass.const_expr(CSSM):
                                            mm = lidx + CSL
                                            for e in cutlass.range_constexpr(CPL):
                                                rf[jt, e] = sCS[
                                                    ti, mm,
                                                    lane + 32 * e].to(FP32)
                                        else:
                                            for e in cutlass.range_constexpr(CPL):
                                                rf[jt, e] = mCS[
                                                    b, ti, h, lane + 32 * e,
                                                    lidx + CSL].to(FP32)
                                for jt in cutlass.range_constexpr(TG):
                                    jw = gi * TG + jt
                                    for e in cutlass.range_constexpr(CPL):
                                        cwv = sCW[ti, jw, lane + 32 * e].to(FP32)
                                        if cutlass.const_expr(jw == 0):
                                            cacc[e] = cwv * rf[jt, e]
                                        else:
                                            cacc[e] = cacc[e] + cwv * rf[jt, e]
                            for e in cutlass.range_constexpr(CPL):
                                # the ONLY dynamic term is `lane`; the per-e part is a
                                # compile-time constant (see the publish note below)
                                d = lane + 32 * e
                                acc = cacc[e]
                                # SiLU(a) = a / (1 + exp(-a)); rcp.approx.ftz is exact to
                                # 2^-23 and the denominator is >= 1 so ftz never bites; a
                                # very negative -> exp overflows -> rcp(inf) == 0, the
                                # correct SiLU limit.
                                xv = acc * cute.arch.rcp_approx(
                                    FP32(1.0) + cute.math.exp(-acc, fastmath=True))
                                if cutlass.const_expr(ti == 0):
                                    qf[e] = xv
                                    gf[e] = mG[row, d]
                                    nq = nq + xv * xv
                                elif cutlass.const_expr(ti == 1):
                                    kf[e] = xv
                                    nk = nk + xv * xv
                                    kq = kq + qf[e] * xv
                                else:
                                    sV[j, d] = xv
                        # one warp owns all K channels of the token -> a pure warp
                        # reduction, no barrier
                        for off in cutlass.range_constexpr(5):
                            nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                            nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                            kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
                        rq = FP32(cute.math.rsqrt(nq, fastmath=True))
                        rk = FP32(cute.math.rsqrt(nk, fastmath=True))
                        rqs = rq * FP32(RSQRT_K)
                        # Same index decomposition as the single-warp path below,
                        # for the same measured reason -- do NOT "simplify" it.
                        ir = lane & (VEC - 1)
                        irh = lane & (VECH - 1)
                        for e in cutlass.range_constexpr(CPL):
                            sDG[j, (lane >> 2) + (32 // VEC) * e, ir] = FP32(
                                cute.math.exp(gf[e], fastmath=True))
                            sKN[j, (lane >> 3) + (32 // VECH) * e, irh] = (
                                kf[e] * rk).to(F16)
                            sQN[j, (lane >> 3) + (32 // VECH) * e, irh] = (
                                qf[e] * rqs).to(F16)
                        if lane == 0:
                            sSC[j, 0] = mBeta[row]
                            sSC[j, 1] = kq * (rk * rqs)
                    if hlf == sp:
                        # ---- v: conv straight into sV, plus the gate scale -----
                        for ti in cutlass.range_constexpr(2, NTI):
                            for gi in cutlass.range_constexpr(NG):
                                for jt in cutlass.range_constexpr(TG):
                                    lidx = li - CSL + gi * TG + jt
                                    if lidx >= 0:
                                        grow = cbase + j - CSL + gi * TG + jt
                                        for e in cutlass.range_constexpr(CPL):
                                            rf[jt, e] = mMix[
                                                grow, ti, h, lane + 32 * e].to(FP32)
                                    else:
                                        # history taps: SMEM (see CONV_STATE_SMEM)
                                        if cutlass.const_expr(CSSM):
                                            mm = lidx + CSL
                                            for e in cutlass.range_constexpr(CPL):
                                                rf[jt, e] = sCS[
                                                    ti, mm,
                                                    lane + 32 * e].to(FP32)
                                        else:
                                            for e in cutlass.range_constexpr(CPL):
                                                rf[jt, e] = mCS[
                                                    b, ti, h, lane + 32 * e,
                                                    lidx + CSL].to(FP32)
                                for jt in cutlass.range_constexpr(TG):
                                    jw = gi * TG + jt
                                    for e in cutlass.range_constexpr(CPL):
                                        cwv = sCW[ti, jw, lane + 32 * e].to(FP32)
                                        if cutlass.const_expr(jw == 0):
                                            cacc[e] = cwv * rf[jt, e]
                                        else:
                                            cacc[e] = cacc[e] + cwv * rf[jt, e]
                            for e in cutlass.range_constexpr(CPL):
                                # the ONLY dynamic term is `lane`; the per-e part is a
                                # compile-time constant (see the publish note below)
                                d = lane + 32 * e
                                acc = cacc[e]
                                # SiLU(a) = a / (1 + exp(-a)); rcp.approx.ftz is exact to
                                # 2^-23 and the denominator is >= 1 so ftz never bites; a
                                # very negative -> exp overflows -> rcp(inf) == 0, the
                                # correct SiLU limit.
                                xv = acc * cute.arch.rcp_approx(
                                    FP32(1.0) + cute.math.exp(-acc, fastmath=True))
                                if cutlass.const_expr(ti == 0):
                                    qf[e] = xv
                                    gf[e] = mG[row, d]
                                    nq = nq + xv * xv
                                elif cutlass.const_expr(ti == 1):
                                    kf[e] = xv
                                    nk = nk + xv * xv
                                    kq = kq + qf[e] * xv
                                else:
                                    sV[j, d] = xv
                        if cutlass.const_expr(epi):
                            for e in cutlass.range_constexpr(ZPL):
                                cidx = lane + 32 * e
                                zf = mZ[row, vsi * VS + cidx].to(FP32)
                                # sigmoid(z) = 1/(1+exp(-z)); rcp.approx.ftz.f32
                                # is exact to 2^-23 and 1+exp(-z) >= 1
                                sZG[j, cidx] = fw[e] * cute.arch.rcp_approx(
                                    FP32(1.0) + cute.math.exp(-zf,
                                                              fastmath=True))
            else:
                for rr in cutlass.range_constexpr(TPW):
                    j = w + rr * NWARP
                    if j < ntok:
                        row = (cbase + j) * H + h
                        nq = FP32(0.0)
                        nk = FP32(0.0)
                        kq = FP32(0.0)
                        if cutlass.const_expr(conv):
                            # ===== FUSED width-4 depthwise causal conv + SiLU =======
                            # `li` is this token's index WITHIN its request, so taps
                            # li-3..li reach back into conv_state only for li < 3.
                            # li is warp-uniform (j == warp_idx), so every tap
                            # predicate below is a warp-uniform branch, and the 16
                            # loads of a tensor's window issue as one burst.
                            li = c * TMAX + j
                            for ti in cutlass.range_constexpr(NTI):
                                for gi in cutlass.range_constexpr(NG):
                                    for jt in cutlass.range_constexpr(TG):
                                        lidx = li - CSL + gi * TG + jt
                                        if lidx >= 0:
                                            grow = cbase + j - CSL + gi * TG + jt
                                            if cutlass.const_expr(PAIR):
                                                for ep in cutlass.range_constexpr(CPP):
                                                    cute.autovec_copy(
                                                        mMix[grow, ti, h,
                                                             lane + 32 * ep, None],
                                                        rp)
                                                    rf[jt, 2 * ep] = rp[0].to(FP32)
                                                    rf[jt, 2 * ep + 1] = (
                                                        rp[1].to(FP32))
                                            else:
                                                for e in cutlass.range_constexpr(CPL):
                                                    rf[jt, e] = mMix[
                                                        grow, ti, h,
                                                        lane + 32 * e].to(FP32)
                                        else:
                                            # only the first CSL tokens of a request
                                            # reach here, and conv_state is
                                            # channel-strided, so it stays scalar
                                            for e in cutlass.range_constexpr(CPL):
                                                if cutlass.const_expr(CSSM):
                                                    rf[jt, e] = sCS[
                                                        ti, lidx + CSL,
                                                        lane + 32 * e].to(FP32)
                                                else:
                                                    if cutlass.const_expr(PAIR):
                                                        dcs = (2 * lane + (e & 1)
                                                               + 64 * (e >> 1))
                                                    else:
                                                        dcs = lane + 32 * e
                                                    rf[jt, e] = mCS[
                                                        b, ti, h, dcs,
                                                        lidx + CSL].to(FP32)
                                    if cutlass.const_expr(PAIR):
                                        for ep in cutlass.range_constexpr(CPP):
                                            cute.autovec_copy(
                                                sCW[ti, gi, lane + 32 * ep, None],
                                                cwg)
                                            for jt in cutlass.range_constexpr(TG):
                                                jj = gi * TG + jt
                                                for sub in cutlass.range_constexpr(2):
                                                    e = 2 * ep + sub
                                                    cwv = cwg[sub * TG + jt].to(FP32)
                                                    if cutlass.const_expr(jj == 0):
                                                        cacc[e] = cwv * rf[jt, e]
                                                    else:
                                                        cacc[e] = (cacc[e]
                                                                   + cwv * rf[jt, e])
                                    else:
                                        for jt in cutlass.range_constexpr(TG):
                                            jj = gi * TG + jt
                                            for e in cutlass.range_constexpr(CPL):
                                                cwv = sCW[ti, jj,
                                                          lane + 32 * e].to(FP32)
                                                if cutlass.const_expr(jj == 0):
                                                    cacc[e] = cwv * rf[jt, e]
                                                else:
                                                    cacc[e] = (cacc[e]
                                                               + cwv * rf[jt, e])
                                for e in cutlass.range_constexpr(CPL):
                                    # dynamic term is `lane`-only (or `2*lane+sub`),
                                    # the per-e part a compile-time constant -- see
                                    # the publish loop's warning below.
                                    if cutlass.const_expr(PAIR):
                                        d = 2 * lane + (e & 1) + 64 * (e >> 1)
                                    else:
                                        d = lane + 32 * e
                                    acc = cacc[e]
                                    # SiLU(a) = a / (1 + exp(-a)); rcp.approx.ftz is
                                    # exact to 2^-23 and the denominator is >= 1, so
                                    # ftz never bites. a very negative -> exp
                                    # overflows -> rcp(inf) == 0, which is the right
                                    # limit for SiLU.
                                    xv = acc * cute.arch.rcp_approx(
                                        FP32(1.0) + cute.math.exp(-acc,
                                                                  fastmath=True))
                                    if cutlass.const_expr(ti == 0):
                                        qf[e] = xv
                                        gf[e] = mG[row, d]
                                        nq = nq + xv * xv
                                    elif cutlass.const_expr(ti == 1):
                                        kf[e] = xv
                                        nk = nk + xv * xv
                                        kq = kq + qf[e] * xv
                                    else:
                                        sV[j, d] = xv
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
                        # Channel `lane + 32*e`, decomposed so that the ONLY dynamic
                        # term in each SMEM index is a function of `lane` alone and
                        # the per-`e` part is a compile-time constant. Writing the
                        # arithmetically identical `dd = lane + 32*e; sDG[j, dd >> 2,
                        # dd & 3]` instead measured **25% slower** end-to-end
                        # (454.8 vs 363.2 us at H=96 B=64 T=4, same GPU, same lease,
                        # 0.02% run-to-run spread) -- see optimization.md. Do not
                        # "simplify" this.
                        if cutlass.const_expr(PAIR):
                            # channel c = 2*lane + sub + 64*ep; 64*ep is a multiple of
                            # both VEC and VECH, so the whole dynamic part is
                            # `2*lane + sub`, hoisted per (constexpr) sub.
                            for sub in cutlass.range_constexpr(2):
                                base = 2 * lane + sub
                                b4 = base >> 2
                                r4 = base & (VEC - 1)
                                b8 = base >> 3
                                r8 = base & (VECH - 1)
                                for ep in cutlass.range_constexpr(CPP):
                                    e = 2 * ep + sub
                                    sDG[j, b4 + (64 // VEC) * ep, r4] = FP32(
                                        cute.math.exp(gf[e], fastmath=True))
                                    sKN[j, b8 + (64 // VECH) * ep, r8] = (
                                        kf[e] * rk).to(F16)
                                    sQN[j, b8 + (64 // VECH) * ep, r8] = (
                                        qf[e] * rqs).to(F16)
                        else:
                            ir = lane & (VEC - 1)
                            irh = lane & (VECH - 1)
                            for e in cutlass.range_constexpr(CPL):
                                sDG[j, (lane >> 2) + (32 // VEC) * e, ir] = FP32(
                                    cute.math.exp(gf[e], fastmath=True))
                                sKN[j, (lane >> 3) + (32 // VECH) * e, irh] = (
                                    kf[e] * rk).to(F16)
                                sQN[j, (lane >> 3) + (32 // VECH) * e, irh] = (
                                    qf[e] * rqs).to(F16)
                        # v slice for this CTA's VS columns, 32 lanes x (VS/32).
                        # With conv=True the conv stage above already wrote sV.
                        if cutlass.const_expr(not conv):
                            for e in cutlass.range_constexpr(VS // 32):
                                cidx = lane + 32 * e
                                sV[j, cidx] = mV[row, vsi * VS + cidx].to(FP32)
                        # the epilogue's per-column gate scale, staged here so the
                        # epilogue never touches global memory
                        if cutlass.const_expr(epi):
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
                        # have a column-pair-invariant multiplier, so each pairs
                        # perfectly. Bit-identical to the scalar form.
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
                    # `rcol`: dk (slot 0) and dq (slot 1), forms o itself and
                    # stores it straight to global. (epi=False only -- the RMS
                    # reduction cannot be done from one column per thread without
                    # an extra CTA barrier.)
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
                        if cutlass.const_expr(epi):
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
        pMix: cute.Pointer,
        pCW: cute.Pointer,
        pCS: cute.Pointer,
        rows: cutlass.Int32,   # TT*H
        nbh: cutlass.Int32,    # B*H
        nb: cutlass.Int32,     # B
        ntt: cutlass.Int32,    # TT
        stream,
    ):
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
        # mixed_qkv [TT, D] with D = 3*H*K packed as [q | k | v]; conv_weight
        # [D, 4]; conv_state [B, D, 3]. All three are re-modelled as
        # (.., ti, h, d, ..) so a (tensor, head, channel) triple is one index
        # expression with no division.
        if cutlass.const_expr(PAIR):
            # K split into channel PAIRS so the conv can pull two channels with
            # one 32-bit load (one L1TEX wavefront per full 128 B line).
            mMix = cute.make_tensor(pMix, cute.make_layout(
                (ntt, NTI, H, K // 2, 2),
                stride=(NTI * H * K, H * K, K, 2, 1)))
        else:
            mMix = cute.make_tensor(pMix, cute.make_layout(
                (ntt, NTI, H, K), stride=(NTI * H * K, H * K, K, 1)))
        mCW = cute.make_tensor(pCW, cute.make_layout(
            (NTI, H, K, WIDTH), stride=(H * K * WIDTH, K * WIDTH, WIDTH, 1)))
        mCS = cute.make_tensor(pCS, cute.make_layout(
            (nb, NTI, H, K, CSL),
            stride=(NTI * H * K * CSL, H * K * CSL, K * CSL, CSL, 1)))
        blackwell_bf16_kda_save_ssm_conv_gated_kernel(
            mQ, mK, mV, mG, mBeta, mS0, mSnap, mO, mCu, mZ, mW, mMix, mCW, mCS
        ).launch(grid=[VSPLITS, H, nb], block=[THREADS, 1, 1], stream=stream)

    return launch


# ===========================================================================
# Host engine + public API (promoted from the CuTeDSLGen workspace
# gen_kda_save_ssm_conv_gated_claude_0730_0022/best/run.py; only the
# production entry points are kept).
# ===========================================================================
import cuda.bindings.driver as _cuda  # noqa: E402
import torch  # noqa: E402
from cutlass.cute.runtime import make_ptr  # noqa: E402

GMEM = cute.AddressSpace.gmem

# the launch stream is torch's CURRENT stream at call time (cheap raw query),
# so the kernel lands inside CUDA-graph captures and side-stream timing loops
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream  # type: ignore[attr-defined]
except AttributeError:                                 # pragma: no cover
    def _raw_stream(dev):
        return torch.cuda.current_stream(dev).cuda_stream

#  dtype, assumed alignment for the 14 pointer args
#  (q k v g beta S0 snap o cu z w mix cw cs)
_PTR_SPEC = ((BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16),
             (FP32, 4), (FP32, 16), (FP32, 16), (BF16, 16), (I32, 4),
             (BF16, 16), (FP32, 16), (BF16, 16), (BF16, 16), (BF16, 16))
_PTR_TORCH_DTYPE = (torch.bfloat16, torch.bfloat16, torch.bfloat16,
                    torch.float32, torch.float32, torch.float32,
                    torch.float32, torch.bfloat16, torch.int32,
                    torch.bfloat16, torch.float32, torch.bfloat16,
                    torch.bfloat16, torch.bfloat16)
_NPTR = len(_PTR_SPEC)
_DUMMY = {}


def _dummy_ptr(dtype, device):
    """A small live allocation for the pointer slots a variant ignores
    (conv=False never dereferences mix/cw/cs; conv=True never dereferences
    q/k/v), since the compiled launcher's signature is fixed."""
    key = (dtype, str(device))
    t = _DUMMY.get(key)
    if t is None:
        t = torch.zeros(64, dtype=dtype, device=device)
        _DUMMY[key] = t
    return t.data_ptr()


class _Engine:
    """Caches the compiled launcher and argument-marshalling objects per
    (H, K, V, conv). Nothing input-dependent is cached: every call rebinds
    the live device pointers, so the kernel always reads current data."""

    def __init__(self):
        self._compiled = {}
        self._args = {}

    def _get_compiled(self, H, K, V, conv):
        key = (H, K, V, conv)
        c = self._compiled.get(key)
        if c is None:
            c = cute.compile(
                make_launcher(H, K, V, cfg=CFG_W8, epi=True, conv=conv),
                *[make_ptr(dt, 0, GMEM, assumed_align=al)
                  for dt, al in _PTR_SPEC],
                I32(0), I32(0), I32(0), I32(0),
                _cuda.CUstream(_raw_stream(torch.cuda.current_device())),
            )
            self._compiled[key] = c
        return c

    def prepare(self, H, K, V, TT, B, tensors, conv):
        key = (H, K, V, TT, B, conv)
        e = self._args.get(key)
        if e is None:
            for t in tensors:
                if t is None:
                    continue
                if not t.is_contiguous():
                    raise ValueError(
                        "kda_save_ssm_conv_gated requires contiguous "
                        f"inputs (got a non-contiguous {tuple(t.shape)})")
                if t.data_ptr() % 16:
                    raise ValueError("inputs must be 16-byte aligned")
            e = (self._get_compiled(H, K, V, conv),
                 [make_ptr(dt, 0, GMEM, assumed_align=al)
                  for dt, al in _PTR_SPEC],
                 (I32(TT * H), I32(B * H), I32(B), I32(TT)))
            self._args[key] = e
        return e


_ENGINE = _Engine()


def _launch(g, beta, S0, cu_seqlens, snapshot, z, w, conv,
            q=None, k=None, v=None, mixed_qkv=None, conv_weight=None,
            conv_state=None):
    B, H, K, V = S0.shape
    TT = snapshot.shape[0]
    dev = S0.device
    o = torch.empty((TT, H, V), dtype=torch.bfloat16, device=dev)
    ten = (q, k, v, g, beta, S0, snapshot, o, cu_seqlens, z, w,
           mixed_qkv, conv_weight, conv_state)
    compiled, p, scal = _ENGINE.prepare(H, K, V, TT, B, ten, conv)
    for i in range(_NPTR):
        t = ten[i]
        p[i]._desc.value = (t.data_ptr() if t is not None
                            else _dummy_ptr(_PTR_TORCH_DTYPE[i], dev))
    compiled(*p, *scal, _cuda.CUstream(_raw_stream(dev.index)))
    return o


def kda_save_ssm_conv_gated(mixed_qkv, conv_weight, conv_state, g, beta,
                                 S0, cu_seqlens, snapshot, z, w):
    """KDA multi-token MTP-verify step, SAVE-SSM scheme, ONE fused kernel
    (conv4+SiLU -> recurrence with per-token state snapshots -> gated RMSNorm).

    mixed_qkv   : [TT, 3*H*K] bf16   RAW pre-conv packed projections [q|k|v]
    conv_weight : [3*H*K, 4]  bf16   depthwise causal conv weight, NO bias
    conv_state  : [B, 3*H*K, 3] bf16 last 3 RAW committed tokens, READ-ONLY
    g           : [TT, H, K]  fp32   pre-activated log decay <= 0
    beta        : [TT, H]     fp32   in (0, 1)
    S0          : [B, H, K, V] fp32  incoming state, READ-ONLY
    cu_seqlens  : [B+1] int32 on device
    snapshot    : [TT, H, K, V] fp32 post-token states, WRITTEN IN PLACE
    z           : [TT, H, V]  bf16   output gate
    w           : [V]         fp32   RMSNorm weight

    returns o : [TT, H, V] bf16, POST gated-RMSNorm
    """
    return _launch(g, beta, S0, cu_seqlens, snapshot, z, w, True,
                   mixed_qkv=mixed_qkv, conv_weight=conv_weight,
                   conv_state=conv_state)


def kda_save_ssm_gated_noconv(q, k, v, g, beta, S0, cu_seqlens, snapshot,
                                   z, w):
    """Same kernel WITHOUT the fused conv (takes POST-conv bf16 q, k, v);
    the ladder row that must be preceded by a separate conv launch."""
    return _launch(g, beta, S0, cu_seqlens, snapshot, z, w, False,
                   q=q, k=k, v=v)
