# SPDX-License-Identifier: Apache-2.0
"""
blackwell_bf16_kda_replay_ssm_conv_gated -- KDA (Kimi Delta Attention) MTP
*verify* step with BOTH the width-4 depthwise causal-conv1d + SiLU input stage
AND the output gated-RMSNorm epilogue fused in, for B200 (sm_100a), CuTeDSL
4.5.2.  See design.md for the full derivation.

ONE launch replaces three kernels of SGLang's production path
(`causal_conv1d_update` -> the verify kernel -> `FusedRMSNormGated`):

  conv     x[n,i,c] = SiLU(sum_{j<4} conv_weight[c,j] * raw(n, i-3+j, c))
           raw(n,t,c) = mixed_qkv[n,t,c] for t >= 0, else conv_state[n,c,t+3]
           q,k,v = unpack(x)                     # the three D/3-wide blocks
  verify   the T-step KDA recurrence against checkpoint S + the record ring
  epilogue o = (r * rsqrt(mean_v(r^2) + 1e-5)) * w * sigmoid(z)

It produces the post-norm `o` for the T = 1+gamma draft tokens of every request
WITHOUT committing the recurrent state, and appends T per-token records to a
small ring so a later launch can resume from exactly the accepted prefix.  The
checkpoint state S is read every launch and written ONLY when the ring would
overflow.  `conv_state` is READ-ONLY: acceptance is unknown at kernel time, so
the host re-derives the conv window after sampling -- the same rollback rule S
and the ring follow.  Post-conv q/k/v and the pre-norm r never reach HBM.

The conv's per-token math is fused into the existing prologue (P2) because it is
PER-CHANNEL: a lane that already owns 4 consecutive channels can produce its own
token's post-conv values from its own `mixed_qkv` loads with no cross-thread
dependency.  Its two CTA-shared inputs are staged into SMEM first, in a phase PS
behind an extra barrier (design.md sec 4a1 -- this is where the current version's
-5.8% over its predecessor comes from):
  * `conv_state` is [B,D,3] with the 3 slots CONTIGUOUS, hence channel-strided, so
    reading it per-tap costs 12 scalar 2 B loads per lane at a 24 B lane stride
    (6 L1TEX wavefronts for 64 useful bytes, ~432 per CTA) -- ablated at ~90% of
    the conv's whole cost.  Staged TRANSPOSED to sCs[CSL][3K], a lane reads its 4
    channels of a slot with one 64-bit LDS and the DYNAMIC slot index is free,
    because a dynamic index into SMEM is address arithmetic while a dynamic index
    into a register fragment spills.  Two earlier attempts to vectorize this read
    while keeping it in registers needed scf.if dispatch and measured +6% to +39%.
  * the conv taps were converted bf16->fp32 4x redundantly (every warp of a
    prologue group converts the same 16 values), 6144 converts per CTA for 1536
    distinct weights.  Staged fp32 and TAP-MAJOR in sCw[CONVW][3K], which makes a
    lane's 4 channels one conflict-free LDS.128 -- the [3K][CONVW] layout would be
    a 4-way bank conflict.
Counter-intuitively the extra barrier LOWERS the barrier stall (1.33 -> 0.97): P2
stops waiting on scattered global loads, so the phase it gates shrinks by more
than the barrier costs.  Three further traps shape the phase:
  * the raw-window load must be CLAMPED, not predicated -- a fragment defined in
    both arms of an scf.if is demoted to LOCAL memory (+22% at B=64,T=4);
  * `mixed_qkv` is deliberately NOT staged: its redundant per-warp gathers are
    ablated at 0.06 us (same-CTA L1 hits), while staging them costs O(T) to buy a
    fixed-per-launch saving and measured +4.1% at T=8;
  * SMEM is now a CO-LIMITER at 3 CTAs/SM (40736 B), so any new plane must be
    re-checked with ncu rather than assumed free.

Decomposition
-------------
    grid  = (H, B)                   block = (256,) = 8 warps
    CTA (h,b)  owns the WHOLE S[b, h, :, :]        -> 64 fp32 regs / thread
    warp w     owns V columns [16w, 16w+16) and ALL 128 K channels
    lane = (lk, lv) = (lane>>2, lane&3)
      lk -> 4 K-groups of 4:  k(jj,kk) = (jj*8 + lk)*4 + kk           (KPT = 16)
      lv -> 1 V-group  of 4:  v(cw)    = 16w + lv*4 + cw              (CPT = 4)

Because a warp owns every K channel of its columns, the per-token K reduction is
a 3-step lane butterfly: the T-loop contains no barrier and no SMEM reduction
traffic at all, and every lane derives u/r for its own columns locally.

Why ONE CTA per (b,h) and 8 warps (this is the change the fused epilogue forced).
The un-fused ancestor of this kernel split V over VSPLITS = 2 *CTAs* of 4 warps.
The RMSNorm needs sum_v r^2 over all 128 columns, which under a 2-CTA V-split is
a cross-CTA reduction (cluster + DSMEM + an extra barrier) for a value that is
needed T times per launch.  Merging the two CTAs into one 256-thread CTA instead:
  * keeps the state tile at exactly the same 64 fp32 registers per thread
    (SR = K*V/THREADS = 128*128/256), so occupancy is unchanged -- 128
    registers/thread -> 2 blocks x 8 warps = 16 warps/SM, identical to
    4 blocks x 4 warps before;
  * makes the norm a plain intra-CTA reduction behind ONE barrier;
  * and *removes* work: the two half-V CTAs each used to redundantly compute the
    whole gate/L2-norm prologue, the whole exp(g_start - G_s) coefficient plane
    and the whole q/k/g_raw read.  Now each is done once per (b,h).

The state tile is deliberately 64 (not 128) registers per thread: at 216
registers the SM fit only 8 warps (11.5% occupancy) and the compute phases issued
at ~20% of one IPC.

S is laid out [slot, head, V, K] with K CONTIGUOUS, so the LK lanes sharing a V
row must cover *contiguous* K for every 32 B sector to be fully used -- hence the
strided-VEC-group channel map rather than 16 contiguous channels per thread.

EVERY SMEM/GMEM access in this kernel is 64/128-bit vectorized.  Two measured traps
drove that:
  * `cute.make_layout(shape)` is COLUMN-major, so a naive (T, K/4, 4) SMEM layout
    leaves the trailing mode strided and `autovec_copy` silently degrades to 4
    scalar LDS -- 462 vs 213 LDS/warp, L1TEX pinned at 61% of peak.  All SMEM
    layouts below therefore carry explicit row-major strides.
  * the per-token / per-history-record staging loops originally walked one
    element per lane (2 B loads).  Reshaping them so every lane moves 8/16 B cut
    P1+P2 from ~28 us to a few us at B=64,T=4.

Phases
------
 P0  issue the 16 x 128-bit S loads                 (HBM latency starts here)
 PS  stage conv_state -> sCs[CSL][3K] (transposed, bf16) and the conv taps ->
     sCw[CONVW][3K] (tap-major, fp32); one thread per local channel
 --  barrier 0
 P1a stage ring old_k -> sC (raw, fp32), old_u -> sUo   [8 elems/lane]
     (after barrier 0, so its ring-load latency overlaps P2)
 P2  new-token prologue INCLUDING THE CONV, split over two warp groups when
     2T <= NWARP:
       warps [0,T)  : conv(q), conv(k), both L2 norms, <kn,qn>, ring old_k store
       warps [T,2T) : conv(v) -> sV, gate -> sGL, z -> sZG, beta -> sSC[.,0]
 --  barrier
 P1b per channel (threads 0..127): sD = exp(g_start);
                  sC[s,k] *= exp(g_start - G_s)   -> replay coefficients
 P1c per channel (threads 128..255, CONCURRENT with P1b): the scaled-recurrence
                  planes a/b/c = exp(+-cumsum(g)_t) * kn_t/qn_t
 P7  G cumsum + ring old_G store (vectorized when pos0 % VEC == 0)
 --  barrier
 P4  st *= exp(g_start)
 P5  rank-pnat replay             -> st == S_logical
 P6  if overflow: store S         (the only S write)
 P8  T-step register-resident recurrence -> pre-norm r into sR, ring old_u
 --  barrier
 P9  fused gated-RMSNorm epilogue: warp w reduces sum_v r^2 for token w over its
     32 lanes (one LDS.128 each), then scales by w * sigmoid(z) and stores o
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32

HIST = 16
LOG_HIST = 4
CONVW = 4                # depthwise causal conv width
CSL = CONVW - 1          # conv_state length: the 3 last committed raw tokens
VEC = 4                  # fp32 elements per 128-bit access
LOG_VEC = 2
VECH = 8                 # bf16 elements per 128-bit access
LOG_VECH = 3
NWARP = 8                # one CTA owns the whole [K,V] tile (see docstring)
THREADS = NWARP * 32     # 256
# The (LK, NCH) state tiling is chosen per T inside make_launcher -- see the
# "column chunking is a per-T dispatch" note in the docstring.  T <= T_CHUNK_MAX
# uses LK=16/NCH=2 (32-register tile, 3 CTAs/SM); larger T uses LK=8/NCH=1
# (64-register tile, 2 CTAs/SM).
T_CHUNK_MAX = 5
K_EPS = 1e-6
NORM_EPS = 1e-5


@cute.jit
def _sigmoid(x: FP32) -> FP32:
    """1/(1+exp(-x)) via ex2.approx + rcp.approx.ftz (max rel err ~2^-22)."""
    return cute.arch.rcp_approx(
        FP32(1.0) + FP32(cute.math.exp(FP32(0.0) - x, fastmath=True)))


def _conv_gather(mXv, brow, jtok, cg, xf):
    """TRACE-TIME MACRO (plain Python): issue the 4-tap raw window of local token
    `jtok` for this lane's VEC channels into `xf[:, jj]`, ALL FOUR LOADS FIRST.

    The row index is CLAMPED (`max(t, 0)`) rather than predicated, so every load
    is unconditional and `xf` is never defined inside a branch: a fragment
    written in both arms of an `scf.if` and read after the join is demoted by
    ptxas to LOCAL memory (measured 8 `STL` + 16 `LDL` per conv warp, +22%).
    A clamped load of a row that will be discarded is far cheaper than a spill,
    and it lets all 4 taps' HBM latencies overlap because nothing separates them.

    These `mixed_qkv` reads are deliberately NOT staged through SMEM even though
    the `conv_state` reads are.  Tokens `j` and `j+1` share 3 of their 4 window
    rows, so each raw row is fetched by up to 4 warps -- but they are the same
    CTA hitting the same 128 B lines, and ablation prices the 3 redundant gathers
    at **0.06 us**.  Staging them instead costs one LDG + one STS per token per
    staging thread, i.e. a cost that scales with `T` to buy back nothing; the
    measured T=8 penalty for doing so was +4.1% (optimization.md cycle 1).
    """
    for jj in range(CONVW):
        tc = cutlass.max(jtok - (CONVW - 1) + jj, I32(0))
        cute.autovec_copy(mXv[brow + tc, cg, None], xf[None, jj])


@cute.jit
def _conv_acc4(sCs, sCw, lg, jtok, xf, wfj, csf, out):
    """The depthwise conv + SiLU proper: fold the 4 taps, substituting the staged
    `conv_state` window for the taps that predate the launch.

        out[e] = SiLU( sum_{jj<4} conv_weight[cbb+e, jj] * raw(jtok-3+jj, cbb+e) )

    `raw(t, c)` is `xf[e, jj]` (== mixed_qkv) for t >= 0, else `conv_state` slot
    `sl = jtok + jj == t + 3` -- slot 2 being the most recent committed token,
    sglang `causal_conv1d_update`'s roll convention.

    The state tap is now ONE 64-bit `LDS` covering this lane's 4 channels, out of
    the transposed `sCs[CSL][3K]` plane staged in PS, instead of 4 scalar 2 B
    global loads at a 24 B lane stride (6 L1TEX wavefronts each for 64 useful
    bytes).  Two things make that safe here where two earlier attempts failed:

    * the slot index `sl` is dynamic, and indexing a register FRAGMENT dynamically
      spills it -- but indexing SMEM dynamically is just address arithmetic, so
      the `min` clamp below is all it takes.  That is what the earlier attempts
      needed `scf.if` dispatch chains for, at +6% to +39%.
    * `csf` is loaded UNCONDITIONALLY (clamped slot) *before* the branch, so it is
      never a fragment defined inside an `scf.if`; the `t < 0` arm then assigns
      only PLAIN SCALARS, which the AST preprocessor yields through the branch as
      registers.  The unconditional `LDS` for taps that will not use it is 2
      wavefronts -- far cheaper than a local-memory spill or a branch chain.
    """
    a0 = FP32(0.0)
    a1 = FP32(0.0)
    a2 = FP32(0.0)
    a3 = FP32(0.0)
    for jj in cutlass.range_constexpr(CONVW):
        t = jtok - (CONVW - 1) + jj
        slc = cutlass.min(jtok + jj, I32(CSL - 1))
        cute.autovec_copy(sCs[slc, lg, None], csf)
        # tap jj's weight for this lane's 4 channels: ONE conflict-free LDS.128
        # out of the fp32 plane staged in PS.  Read straight from a global bf16
        # [D,4] the same 16 values would be re-converted by EVERY warp of the
        # group -- 4x redundant, 6144 bf16->fp32 converts per CTA for 1536
        # distinct weights.  Staging them fp32 does the conversion once (1536 per
        # CTA) and takes the converts off this pre-barrier critical path.
        cute.autovec_copy(sCw[jj, lg, None], wfj)
        v0 = xf[0, jj].to(FP32)
        v1 = xf[1, jj].to(FP32)
        v2 = xf[2, jj].to(FP32)
        v3 = xf[3, jj].to(FP32)
        if t < 0:
            v0 = csf[0].to(FP32)
            v1 = csf[1].to(FP32)
            v2 = csf[2].to(FP32)
            v3 = csf[3].to(FP32)
        a0 = a0 + wfj[0] * v0
        a1 = a1 + wfj[1] * v1
        a2 = a2 + wfj[2] * v2
        a3 = a3 + wfj[3] * v3
    out[0] = a0 * _sigmoid(a0)
    out[1] = a1 * _sigmoid(a1)
    out[2] = a2 * _sigmoid(a2)
    out[3] = a3 * _sigmoid(a3)


def make_launcher(H: int, K: int, V: int, T: int, gated: bool = True,
                  conv: bool = True):
    """Build a compiled-once launcher closure for a given (H, K, V, T).

    `gated=False` builds the same kernel with the RMSNorm/gate epilogue removed
    (it then stores the raw pre-norm r).  `conv=False` builds it reading
    already-convolved `q/k/v` instead of raw `mixed_qkv`.  Both variants exist
    only so the bench can price the fusions against "this kernel + a separate
    launch"; the production path is `gated=True, conv=True`.
    """
    assert K == 128 and V == 128, "spec fixes K = V = 128"
    assert 1 <= T <= 8
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
    # packed projection width and the per-block channel bases of [q | k | v]
    DW = 3 * H * K
    CB = (0, H * K, 2 * H * K)               # scalar channel base per block
    CG = (0, H * KV4, 2 * H * KV4)           # VEC-group base per block
    # ---- the staged raw conv window (see the PS phase) ----------------------
    # This CTA needs the q|k|v channels of ONE head: 3*K == 384 of the D
    # channels, over CSL committed + T in-launch raw tokens.
    DCTA = 3 * K                             # 384 channels
    NRT = DCTA // VEC                        # 96 VEC-groups per staged row
    NCS = (DCTA + THREADS - 1) // THREADS    # channel rounds for conv_state (2)
    LOG_KV4 = LOG_K - LOG_VEC                # 5

    @cute.kernel
    def blackwell_bf16_kda_replay_ssm_conv_gated_kernel(
        mQ: cute.Tensor,      # [B*T*H, K/4, 4]       bf16  (conv=False only)
        mKt: cute.Tensor,     # [B*T*H, K/4, 4]       bf16  (conv=False only)
        mV4: cute.Tensor,     # [B*T*H, V/4, 4]       bf16  (conv=False only)
        mXv: cute.Tensor,     # [B*T, D/4, 4]         bf16  raw mixed_qkv (conv)
        mCw: cute.Tensor,     # [D, 4]                bf16  conv taps (staged in PS)
        mCs: cute.Tensor,     # [B, D, 3]             bf16  conv state (READ-ONLY)
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
        # conv_state, staged TRANSPOSED to [CSL][3K] so a lane reads its 4
        # channels of a slot with one 64-bit LDS.  bf16, and only 3 rows: the
        # plane is 2304 B and -- unlike a version that also staged the T
        # in-launch tokens -- its size and its staging cost are both INDEPENDENT
        # of T, which is what keeps the extra barrier affordable at T=8.
        if cutlass.const_expr(conv):
            sCs = smem.allocate_tensor(BF16, cute.make_layout(
                (CSL, NRT, VEC), stride=(DCTA, VEC, 1)), 16)
            # conv taps, TAP-MAJOR and fp32: row jj holds all 3K channels' tap jj,
            # so a lane's 4 channels are 4 contiguous fp32 -> one conflict-free
            # LDS.128 (32 lanes x 16 B contiguous).  The transposed-the-other-way
            # [3K][CONVW] layout would make that a 4-way bank conflict.
            sCw = smem.allocate_tensor(FP32, cute.make_layout(
                (CONVW, NRT, VEC), stride=(DCTA, VEC, 1)), 16)

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

        brow = b * T                                   # mixed_qkv token row base

        # ---- PS: transpose conv_state into SMEM, then barrier 0 --------------
        # conv_state is [B, D, 3] with the 3 slots CONTIGUOUS, hence
        # channel-strided.  A per-lane conv must read it one channel and one slot
        # at a time: 12 scalar 2 B loads per lane at a 24 B lane stride, each
        # spanning 768 B across the warp = 6 L1TEX wavefronts for 64 useful
        # bytes, ~432 wavefronts per CTA.  Ablation priced that at 3.97 us of the
        # conv's 4.44 us total (optimization.md).
        #
        # Here each element is read exactly ONCE, by the thread that owns its
        # channel (3 scalars at a 6 B lane stride = 2 wavefronts), and written
        # TRANSPOSED so the consumer reads 4 channels of a slot in one 64-bit
        # LDS with a free dynamic slot index.  ~96 wavefronts per CTA instead of
        # ~432, and both the plane and this phase are independent of T.
        #
        # Local channel lc in [0, 3K) decomposes as (block, c) =
        # (lc >> LOG_K, lc & (K-1)) -- shifts, no division -- mapping to global
        # channel block*(H*K) + h*K + c.
        #
        # The landing fragment is written UNCONDITIONALLY (index clamped, only
        # the SMEM *store* guarded) because a fragment defined inside an scf.if
        # is demoted to local memory -- the +22% trap from attempt_000 cycle 1.
        # The clamped loads made by the out-of-range threads all collapse onto
        # one address, so they cost 1 wavefront each.
        if cutlass.const_expr(conv):
            csv = cute.make_fragment((CSL, NCS), BF16)
            cwv = cute.make_fragment((CONVW, NCS), BF16)
            for it in cutlass.range_constexpr(NCS):
                lcc = cutlass.min(tid + it * THREADS, I32(DCTA - 1))
                gcc = (lcc >> LOG_K) * (H * K) + h * K + (lcc & (K - 1))
                for s in cutlass.range_constexpr(CSL):
                    csv[s, it] = mCs[b, gcc, s]
                # this channel's CONVW taps are CONVW contiguous bf16 == 8 B, and
                # 8*gcc is 8-aligned, so the whole warp's read is 256 B
                # contiguous: one fully-coalesced 64-bit load, 2 wavefronts.
                cute.autovec_copy(mCw[gcc, None], cwv[None, it])
            for it in cutlass.range_constexpr(NCS):
                lcs = tid + it * THREADS
                if lcs < DCTA:
                    for s in cutlass.range_constexpr(CSL):
                        sCs[s, lcs >> LOG_VEC, lcs & (VEC - 1)] = csv[s, it]
                    for jj in cutlass.range_constexpr(CONVW):
                        sCw[jj, lcs >> LOG_VEC, lcs & (VEC - 1)] = (
                            cwv[jj, it].to(FP32))
            cute.arch.sync_threads()

        # ---- P1a: stage the ring records, 8 elements per lane ----------------
        # Placed AFTER barrier 0 on purpose: its ~600-cycle ring loads then
        # overlap P2, which no longer waits on any global load of its own.
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
        qh = cute.make_fragment((VEC,), FP32)          # post-conv, post-SiLU q
        kh = cute.make_fragment((VEC,), FP32)          # post-conv, post-SiLU k
        # conv scratch: xfq/xfk hold the 4-tap raw windows of this lane's VEC
        # channels.  Group 1 gathers q's and k's windows BEFORE consuming either,
        # so all 8 loads' HBM latencies overlap (same issue-before-use argument as
        # sec 4c); group 2 reuses xfq for v.  wgf holds this lane's 16 conv taps
        # as two 128-bit groups -- conv_weight[c, 0..3] is 8 contiguous bytes, so
        # 4 consecutive channels are 32 contiguous bytes and one lane covers them
        # in 2 instructions instead of 4 (4 vs 32 L1TEX wavefronts per warp).
        xfq = cute.make_fragment((VEC, CONVW), BF16)
        xfk = cute.make_fragment((VEC, CONVW), BF16)
        wfj = cute.make_fragment((VEC,), FP32)      # one tap's 4 fp32 weights
        qkh = cute.make_fragment((VEC,), BF16)         # conv=False raw landing
        # global mixed_qkv VEC-group of this lane's 4 channels, per q|k|v block
        cgq = CG[0] + h * KV4 + lane
        cgk = CG[1] + h * KV4 + lane
        cgv = CG[2] + h * KV4 + lane
        # ...and the LOCAL (in-sCs) VEC-group of the same 4 channels
        lgq = 0 * KV4 + lane
        lgk = 1 * KV4 + lane
        lgv = 2 * KV4 + lane
        csf = cute.make_fragment((VEC,), BF16)     # staged conv_state landing
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
            # ---- fused conv stage: q/k for token j come out of the width-4
            # depthwise causal conv + SiLU over the RAW packed projections.
            # The window gather is ONE branch structure covering both tensors:
            # `j` is the warp index so the predicate is warp-UNIFORM (no
            # divergence), and issuing q's and k's raw loads together overlaps
            # their HBM latencies.
            if cutlass.const_expr(conv):
                _conv_gather(mXv, brow, j, cgq, xfq)
                _conv_gather(mXv, brow, j, cgk, xfk)
                _conv_acc4(sCs, sCw, lgq, j, xfq, wfj, csf, qh)
                _conv_acc4(sCs, sCw, lgk, j, xfk, wfj, csf, kh)
            else:
                cute.autovec_copy(mQ[row, lane, None], qkh)
                for e in cutlass.range_constexpr(VEC):
                    qh[e] = qkh[e].to(FP32)
                cute.autovec_copy(mKt[row, lane, None], qkh)
                for e in cutlass.range_constexpr(VEC):
                    kh[e] = qkh[e].to(FP32)
            nq = FP32(0.0)
            nk = FP32(0.0)
            kq = FP32(0.0)
            for e in cutlass.range_constexpr(VEC):
                qe = qh[e]
                ke = kh[e]
                nq = nq + qe * qe
                nk = nk + ke * ke
                kq = kq + qe * ke
            # one warp owns all K channels of a token -> pure warp reduction
            for off in cutlass.range_constexpr(5):
                nq = nq + cute.arch.shuffle_sync_bfly(nq, 1 << off)
                nk = nk + cute.arch.shuffle_sync_bfly(nk, 1 << off)
                kq = kq + cute.arch.shuffle_sync_bfly(kq, 1 << off)
            rk = FP32(cute.math.rsqrt(nk + FP32(K_EPS), fastmath=True))
            rqs = FP32(cute.math.rsqrt(nq, fastmath=True)) * FP32(RSQRT_K)
            for e in cutlass.range_constexpr(VEC):
                kv = kh[e] * rk
                knw[e] = kv
                knb[e] = kv.to(BF16)
                qnw[e] = qh[e] * rqs
            # NOTE: no exp(g_t) plane is produced here any more.  The scaled
            # recurrence (P1c) needs exp(cumsum(g)_t), not exp(g_t), so this
            # phase -- which is on barrier 1's critical path -- is 4 `exp` and
            # one STS.128 per token lighter than before.  The GATE has moved to
            # group 2 as well (see below).
            cute.autovec_copy(knw, sKN[j, lane, None])
            cute.autovec_copy(qnw, sQN[j, lane, None])
            # the persisted ring k: bf16 of the fp32 normalised value
            cute.autovec_copy(knb, mOk4[ru_o + (pos0 + j) * H, lane, None])
            if lane == 0:
                sSC[j, 1] = kq * (rk * rqs)

        # group 2 -- v / GATE / z / beta: pure elementwise, no cross-lane
        # dependency, so it needs nothing from group 1 and can run on the warps
        # group 1 leaves idle.  WLO is 0 (same warps, sequentially) when
        # 2T > NWARP.
        #
        # The GATE lives here rather than in group 1 (where the un-convolved
        # ancestor put it).  Group 1's only consumer of `gv` was the `sGL` store
        # -- nothing in the L2-norm chain reads it -- and fusing the conv made
        # group 1 the clearly heavier half (two convs plus three butterflies vs
        # one conv), so a load + 4 `sigmoid` + one STS.128 come off barrier 1's
        # critical path and land on the lighter side.
        if w >= WLO:
            if w < WLO + T:
                jv = w - WLO
                rowv = (b * T + jv) * H + h
                # v likewise comes out of the conv; z is NOT convolved (it is a
                # separate projection) and keeps its own [B,T,H,V] tensor.
                if cutlass.const_expr(conv):
                    _conv_gather(mXv, brow, jv, cgv, xfq)
                    cute.autovec_copy(mGr[rowv, lane, None], gr)
                    if cutlass.const_expr(gated):
                        cute.autovec_copy(mZ[rowv, lane, None], zh)
                    _conv_acc4(sCs, sCw, lgv, jv, xfq, wfj, csf, vf4)
                else:
                    cute.autovec_copy(mV4[rowv, lane, None], vh4)
                    cute.autovec_copy(mGr[rowv, lane, None], gr)
                    if cutlass.const_expr(gated):
                        cute.autovec_copy(mZ[rowv, lane, None], zh)
                    for e in cutlass.range_constexpr(VEC):
                        vf4[e] = vh4[e].to(FP32)
                cute.autovec_copy(vf4, sV[jv, lane, None])
                for e in cutlass.range_constexpr(VEC):
                    gv[e] = lower_bound * _sigmoid(alpha * (gr[e] + dtb[e]))
                cute.autovec_copy(gv, sGL[jv, lane, None])
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
            # form u and r for its own CPT columns locally: no SMEM publish, no
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

    @cute.jit
    def launch(
        pQ: cute.Pointer,
        pK: cute.Pointer,
        pV: cute.Pointer,
        pXv: cute.Pointer,
        pCw: cute.Pointer,
        pCs: cute.Pointer,
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
        stream,
    ):
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
        # raw packed projections + depthwise conv taps + the read-only conv state
        mXv = cute.make_tensor(pXv, cute.make_layout(
            (nb * T, DW // VEC, VEC), stride=(DW, VEC, 1)))
        # one channel's CONVW taps are contiguous -> a 64-bit group per channel
        mCw = cute.make_tensor(pCw, cute.make_layout(
            (DW, CONVW), stride=(CONVW, 1)))
        mCs = cute.make_tensor(pCs, cute.make_layout(
            (nb, DW, CSL), stride=(DW * CSL, CSL, 1)))
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
        blackwell_bf16_kda_replay_ssm_conv_gated_kernel(
            mQ, mKt, mV4, mXv, mCw, mCs, mGr, mBr, mAl, mDb, mS, mZ, mW, mO,
            mOu4, mOu8, mOk4, mOk8, mOgV, mOgW, mOgS, mPn, mBi, lower_bound,
        ).launch(grid=[H, nb, 1], block=[THREADS, 1, 1],
                 min_blocks_per_mp=MBPM, stream=stream)

    return launch


# ===========================================================================
# Host engine + public API (promoted from the CuTeDSLGen workspace
# gen_kda_replay_ssm_conv_gated_claude_0730_0025/best/run.py; only the
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

#            q     k     v     mixed_qkv  conv_w  conv_state
#            g_raw beta  A_log dt_b  S     z     w
#            o     ou    ok    og    pn    bi
_PTR_SPEC = ((BF16, 16), (BF16, 16), (BF16, 16), (BF16, 16), (BF16, 16),
             (BF16, 16),
             (FP32, 16), (FP32, 4), (FP32, 4), (FP32, 16), (FP32, 16),
             (BF16, 16), (FP32, 16),
             (BF16, 16), (BF16, 16), (BF16, 16), (FP32, 16), (I32, 4),
             (I32, 4))


class _Engine:
    """Caches the compiled launcher and argument-marshalling objects per
    (H, K, V, T, conv). Nothing input-dependent is cached: every call rebinds
    the live device pointers, so the kernel always reads current data."""

    def __init__(self):
        self._compiled = {}
        self._args = {}

    def _get_compiled(self, H, K, V, T, conv):
        c = self._compiled.get((H, K, V, T, conv))
        if c is None:
            c = cute.compile(
                make_launcher(H, K, V, T, gated=True, conv=conv),
                *[make_ptr(dt, 0, GMEM, assumed_align=al)
                  for dt, al in _PTR_SPEC],
                FP32(0.0), I32(0),
                _cuda.CUstream(_raw_stream(torch.cuda.current_device())),
            )
            self._compiled[(H, K, V, T, conv)] = c
        return c

    def prepare(self, H, K, V, T, B, conv, tensors):
        key = (H, K, V, T, B, conv)
        e = self._args.get(key)
        if e is None:
            for t in tensors:
                if not t.is_contiguous():
                    raise ValueError(
                        "kda_replay_ssm_conv_gated requires contiguous inputs")
            e = (self._get_compiled(H, K, V, T, conv),
                 [make_ptr(dt, 0, GMEM, assumed_align=al)
                  for dt, al in _PTR_SPEC],
                 I32(B))
            self._args[key] = e
        return e


_ENGINE = _Engine()


def _run(qkv, g_raw, beta_raw, A_log, dt_bias, lower_bound,
         S, old_u, old_k, old_G, pnat, buf_idx, z, w, conv):
    """`qkv` is (mixed_qkv, conv_weight, conv_state) when conv else (q, k, v)."""
    B, T, H, V = z.shape
    K = S.shape[3]
    if conv:
        mixed_qkv, conv_weight, conv_state = qkv
        q = k = v = mixed_qkv                     # unused pointers
    else:
        q, k, v = qkv
        mixed_qkv = conv_weight = conv_state = q  # unused pointers
    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=z.device)
    live = (q, k, v, mixed_qkv, conv_weight, conv_state, g_raw, beta_raw,
            A_log, dt_bias, S, z, w, o, old_u, old_k, old_G, pnat, buf_idx)
    compiled, p, nb = _ENGINE.prepare(H, K, V, T, B, conv,
                                      [t for t in live if t is not o])
    for ptr, t in zip(p, live):
        ptr._desc.value = t.data_ptr()
    compiled(*p, FP32(float(lower_bound)), nb,
             _cuda.CUstream(_raw_stream(z.device.index)))
    return o


def kda_replay_ssm_conv_gated(mixed_qkv, conv_weight, conv_state, g_raw,
                               beta_raw, A_log, dt_bias, lower_bound, S, old_u,
                               old_k, old_G, pnat, buf_idx, z, w):
    """KDA MTP verify step, REPLAY (cached-update ring) scheme, ONE fused
    kernel (conv4+SiLU -> verify recurrence with ring append -> gated RMSNorm).

    mixed_qkv  : [B, T, D] bf16   RAW pre-conv packed projections, D = 3*H*K
    conv_weight: [D, 4]    bf16   depthwise causal conv taps (no bias)
    conv_state : [B, D, 3] bf16   the last 3 RAW committed tokens; READ-ONLY
    g_raw      : [B, T, H, K] fp32  raw gate pre-activation (not convolved)
    beta_raw   : [B, T, H]    fp32  raw beta logits         (not convolved)
    A_log      : [H]   fp32
    dt_bias    : [H*K] fp32
    lower_bound: float in [-5, 0)  (safe gate)
    S          : [B, H, V, K] fp32  checkpoint; written ONLY on ring overflow
    old_u      : [B, 2, HIST, H, V] bf16 \\
    old_k      : [B, 2, HIST, H, K] bf16  } always receive the T new records
    old_G      : [B, 2, H, K, HIST] fp32 /
    pnat       : [B] int32   accepted tokens in the active ring half (read-only)
    buf_idx    : [B] int32   active ring half (read-only)
    z          : [B, T, H, V] bf16  output gate (not convolved)
    w          : [V]          fp32  RMSNorm weight

    returns o : [B, T, H, V] bf16, POST gated-RMSNorm
    """
    return _run((mixed_qkv, conv_weight, conv_state), g_raw, beta_raw, A_log,
                dt_bias, lower_bound, S, old_u, old_k, old_G, pnat, buf_idx,
                z, w, True)


def kda_replay_ssm_gated_noconv(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                          S, old_u, old_k, old_G, pnat, buf_idx, z, w):
    """The conv-less build: takes ALREADY-convolved bf16 q/k/v [B,T,H,K];
    the ladder row that must be preceded by a separate conv launch."""
    return _run((q, k, v), g_raw, beta_raw, A_log, dt_bias, lower_bound, S,
                old_u, old_k, old_G, pnat, buf_idx, z, w, False)
