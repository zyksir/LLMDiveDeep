#!/usr/bin/env python3
"""blackwell_bf16_kda_decode_conv_gated -- fused single-token KDA decode step with
BOTH the width-4 causal-conv1d + SiLU INPUT stage and the gated-RMSNorm OUTPUT
epilogue fused into the same kernel.

Memory-bound CuTeDSL kernel for Blackwell B200 (sm_100a), nvidia-cutlass-dsl 4.5.2.

Packed channel count D = 3*H*128 (layout [q | k | v], head h owning [h*128, +128)
inside each block). Per packed channel c, then per (batch, head) pair:

    a    = cw[c,0]*cs[c,0] + cw[c,1]*cs[c,1] + cw[c,2]*cs[c,2] + cw[c,3]*x[c]
    xp   = a * sigmoid(a)                             (SiLU, fp32 accumulate)
    cs[c] = [cs[c,1], cs[c,2], x[c]]                  (RAW roll, in place)
    q,k,v = unpack(xp)                                (post-conv, fp32 on-chip)
    qn = q/||q||;  kn = k/||k||                       (L2 norms fused in-kernel)
    S  = diag(exp(g)) @ S                             (decay rows of S)
    u  = beta * (v - S^T kn)
    S  = S + kn @ u^T                                 (rank-1 update, in place)
    r  = S^T qn / sqrt(K)                             (pre-norm decode output)
    o  = (r * rsqrt(mean(r^2) + 1e-5)) * w * sigmoid(z)   (FUSED gated RMSNorm)

Both fusions exist to remove a *launch*, not FLOPs. sglang's Triton
`causal_conv1d_update` is a flat 1.5-6 us on this GPU and a standalone
gated-RMSNorm launch is a flat ~11 us; at B=4 the two of them together dwarf the
recurrence itself. Fused, the conv costs 15 loads + 9 stores + ~18 FLOP per thread
in a CTA that is already resident, and 6.4% of the bytes.

The conv is nearly free structurally because at the default NW=4 the CTA has
exactly NT = 128 = K threads, each already carrying exactly one head-dim element
-- so the thread that needs q[di], k[di], v[di] is the thread that computes them,
and the conv adds *zero* CTA barriers.

Structure (see design.md): one CTA per (b,h) owning the whole 128x128 state tile
and the whole 128-wide output row, so the RMS reduction over the head dim is a
single in-warp shuffle chain with no extra CTA barrier. grid = (H, B) so the conv
gets `h` and `b` separately with no runtime division. The decayed state band
lives in registers between the reduction pass and the write-back pass, so S is
read exactly once and written once -- the optimal 2-unit HBM traffic.

`kda_decode_conv_gated` -- the importable API -- launches through a pre-built C
argument pack (pointer-typed JIT entry), so a call costs ~3 us of host work
instead of the ~50 us nine per-call dlpack conversions would.

See design.md for the full decomposition and optimization.md for what was
measured. Derived from the accepted `blackwell_bf16_kda_decode_gated` package;
written against the CuTeDSL 4.5.2 library API (no example is imported).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu as cute_nv
from cutlass.cute.runtime import make_ptr
from cutlass.utils import SmemAllocator
import cuda.bindings.driver as cuda

K = 128            # key dim (state rows) == head dim of the RMSNorm
V = 128            # value dim (state cols) == output width
NWARP = V // 32    # 4 -- warps needed to carry the K-long norm vectors
CW = 4             # causal-conv1d kernel width (state_len = CW - 1 = 3)
CSL = CW - 1       # conv state length per channel
RSQRT_K = 1.0 / math.sqrt(K)
RMEAN = 1.0 / V    # mean over the head dim = multiply by a constant, never divide
EPS = 1.0e-5       # RMSNorm epsilon (matches FusedRMSNormGated default)

# ---------------------------------------------------------------------------
# Compile-time levers. Defaults are the measured optimum (see optimization.md).
# ---------------------------------------------------------------------------
# Warps per CTA = the row-split factor. This kernel is per-thread
# memory-level-parallelism bound, not occupancy bound: FEWER, fatter warps win,
# because each holds a longer contiguous row band (NIT = K*VW/V * ... see below)
# and issues that many independent state loads back to back. NW=2 is not
# available (the band would be 256 registers, over the 255 cap) because this
# kernel gives one whole (b,h) to one CTA -- the price of fusing the epilogue.
NW = int(os.environ.get("KDA_NW", "4"))
# Columns per thread. VW=4 (LDG.128, NIT=32 row iterations) rather than VW=2
# (LDG.64, NIT=64). Both hold the same 128-register state band and move the same
# bytes, so on raw memory behaviour they tie -- but VW=4 needs half the loop
# bookkeeping and address registers, and that is what decides this kernel.
#
# VW=2 only wins when it is *forced* under the launch bound (MINCTA=3), and that
# operating point is not portable: it depends on ptxas fitting 168 registers with
# zero spill, which the dev toolchain manages and the torch-2.11 eval toolchain
# does NOT (it spills 2.7M local-load sectors and runs 2.6x slower). VW=4 reaches
# a good allocation on BOTH toolchains without being forced. See optimization.md.
#
# As of this attempt neither is hard-coded: AUTOCFG (below) picks between them by
# *asking the compiled binary* whether it spilled. These two remain the defaults
# used when the probe is disabled or the config is pinned by hand.
_VW_ENV = os.environ.get("KDA_VW")
VW = int(_VW_ENV) if _VW_ENV is not None else 4
# 1: contiguous per-thread row band; 0: interleave rows across threads.
ROWMAP = int(os.environ.get("KDA_ROWMAP", "1"))
# 1: issue the whole state band's global loads BEFORE the norm prologue, so the
# prologue's barriers + shuffle chain overlap with the load latency instead of
# being serialized in front of it. Free in registers (the loads land in the band
# the kernel already holds for the write-back).
PREFETCH = int(os.environ.get("KDA_PREFETCH", "1"))
# 1: vector read of the cross-warp norm partials.
RVEC = int(os.environ.get("KDA_RVEC", "1"))
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
# Forcing the cap is a trap here. At VW=2, MINCTA=3 caps registers at
# 65536/(3*128) = 170; the dev toolchain lands on 168 with zero spill and is 12%
# faster, but the torch-2.11 eval toolchain cannot make 168 work and spills
# (2.7M local-load sectors, 43 us vs 18 us at B=64). A launch bound is a
# *promise* about occupancy that ptxas must keep even when it has to spill to
# keep it, so it turns a toolchain difference into a 2.6x cliff.
#
# At VW=4 both toolchains pick a good allocation unforced, so the portable
# configuration is MINCTA=0 -- and it costs only 0.4% against the forced
# dev-venv optimum. 4 caps at 128, below the 128-register band itself, and
# spills catastrophically everywhere. See optimization.md.
_MINCTA_ENV = os.environ.get("KDA_MINCTA")
MINCTA = int(_MINCTA_ENV) if _MINCTA_ENV is not None else 0

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
# 0: use the (VW, MINCTA) above verbatim. Setting KDA_VW or KDA_MINCTA in the
# environment also pins the config and skips the probe, so every sweep recorded
# in optimization.md stays reproducible.
AUTOCFG = int(os.environ.get("KDA_AUTOCFG", "1"))
# Conv-prologue issue order.
#   1 (default): issue ALL 15 conv loads, then the NIT state-band loads, THEN do
#     the conv arithmetic. The conv result gates phase 0, which gates the CTA
#     barrier that gates everything, so its loads must not queue behind the band's
#     64 LDGs -- exactly the QKPF argument, which was worth a measurable win in
#     the source package. The cost is ~15 extra registers live across the
#     prefetch, which at this operating point is the thing that can spill.
#   0: complete the whole conv (load + FMA + SiLU + conv_state roll) BEFORE the
#     state prefetch. Shortest live range, but serialises one round trip.
CVPF = int(os.environ.get("KDA_CVPF", "1"))
# 1: issue the 9 conv_state roll stores as soon as the values are in registers
# (they feed nothing downstream). 0: defer them to after the norm butterfly.
CVST = int(os.environ.get("KDA_CVST", "1"))
# Head-dim elements -- i.e. conv channels per packed block -- carried by ONE
# thread. This is the conv's only structural lever.
#
# A channel's 3 conv-state halves are 6 contiguous bytes at byte offset 6c, which
# is 4-byte aligned only for even c, so at CPT=1 the history is 3 separate LDG.U16
# per block and each of them drags the warp's whole 192-byte span through L1TEX
# (3x line redundancy). Give ONE thread CPT ADJACENT channels and its 3*CPT halves
# become 6*CPT contiguous bytes starting at 6*CPT*cg -- which IS 2*CPT-byte
# aligned -- so they load as 3 vector units of CPT halves each:
#
#   CPT   carrier threads   history unit   weights        tokens     conv warps
#    1        128           3 x LDG.U16    1 x LDG.64     LDG.U16        4
#    2         64           3 x LDG.32     1 x LDG.128    LDG.32         2
#    4         32           3 x LDG.64     2 x LDG.128    LDG.64         1
#
# CPT=4 also collapses the L2-norm reduction into a single warp, which removes one
# of the two prologue CTA barriers. The cost is register pressure and warp-0
# serialisation. Swept in optimization.md.
CPT = int(os.environ.get("KDA_CPT", "1"))
# TIMING-ONLY ABLATIONS (all of them compute the WRONG answer; `--check` refuses
# to run unless ABL == 0). They exist to price each part of the fused conv against
# the un-fused kernel BEFORE building machinery to optimize it.
#   0: the real kernel
#   1: no conv-state history at all (a = cw[3]*x, still SiLU, no roll stores) ->
#      prices the 9 LDG.U16 + 9 STG.U16 history traffic
#   2: history loaded and used, roll stores dropped -> prices the store half
#   3: no conv at all: q/k/v are the RAW packed values -> this is exactly the
#      un-fused decode+gated kernel, i.e. the composed path's second launch
ABL = int(os.environ.get("KDA_ABL", "0"))
# 1: launch a 1-D grid of B*H CTAs and recover (h, b) with a mask/shift (needs H a
# power of two). 0 (default): a 2-D (H, B) grid, which gives h and b for free.
# Both dispatch CTAs in the same linear order; this lever exists to prove that.
# CTA index form:
#   1: a 1-D grid of B*H CTAs; recover (h, b) with a mask/shift (needs H a power of
#      two, which KDA's H=16 is).
#   0: a 2-D (H, B) grid, which hands the kernel h and b for free.
# Both enumerate CTAs in the SAME linear order on paper, so this was expected to be
# a no-op. It is not, and which one wins is a WINDOW, not a threshold. Measured on
# the shipped config (168 registers, 3 CTAs/SM), `api_ms(1-D) / api_ms(2-D)`:
#
#   B       4     8    16    20    24    28    32    40    48    56
#   ratio 1.00  1.228 1.070  .857  .762  .846 1.000  .907  .877  .999
#   B      64    72    80    88    96   128   192   256
#   ratio  .928  .995  .902  .939 1.004 1.031 1.020 1.017
#
# 2-D wins BELOW ~2 CTAs per SM (the 1-D form packs CTAs onto fewer SMs and leaves
# the rest idle: -23% at B=8) and ABOVE ~L2-sized state (where co-residency stops
# buying L2 hits: +1.7-3.1%). In between, the 1-D packing is worth up to 24%.
#
# Total work is invariant under the knob, so this is a per-shape DISPATCH, not a
# constant: `-1` (default) resolves it from the shape at compile time. NOTE that
# the window moved when the register config changed in cycle 2 -- it has to be
# re-swept whenever (VW, MINCTA) changes.
GRID1 = int(os.environ.get("KDA_GRID1", "-1"))
# Lower edge, in CTAs per SM: below this the 1-D form idles SMs.
GRID1_MIN_CTA_PER_SM = float(os.environ.get("KDA_GRID1_MINOCC", "2.0"))
# Upper edge, in state read+write bytes: above this L2 co-residency stops paying.
# The B200's L2 is 126 MB; the measured edge is between B=88 (184 MB) and B=96
# (201 MB).
GRID1_BYTES = float(os.environ.get("KDA_GRID1_BYTES", "1.95e8"))
_PINNED = (_VW_ENV is not None or _MINCTA_ENV is not None
           or os.environ.get("KDA_NW") is not None)

# ---------------------------------------------------------------------------
# CTA SHAPE IS A PER-SHAPE DISPATCH, NOT A CONSTANT (the lever this attempt adds)
#
# One CTA owns one whole (b,h) tile, so the tile's 128*128 fp32 register band is
# 16384 registers per CTA however the CTA is shaped -- a quarter of the SM's
# 65536. What the (NW, VW) pair decides is how that band is *sliced*:
#
#   NW  VW   NT   LPR  NIT  band/thread   measured regs   CTAs/SM   threads/SM
#    4   4  128    32   32     128            255            2         256
#    8   2  256    64   32      64            128            2         512
#    8   4  256    32   16      64            142            1         256
#   16   4  512    32    8      32            128            1         512
#
# `NW=4 VW=4` lets ptxas spend the whole 255-register budget, so only 2 CTAs of
# 128 threads are resident -- 8 warps, 12.5% occupancy. `NW=8 VW=2` lands on
# EXACTLY 128 registers with zero spill, which is the 2-CTA cliff for a 256-thread
# CTA, so the SAME band bytes are held by 16 warps instead of 8.
#
# That matters only where the kernel is LATENCY-bound rather than HBM-bound, and
# the deciding quantity is neither `B` nor bytes but CTAs PER SM = B*HV/148, i.e.
# how many waves of tiles the launch has. Measured (CUDA-graph medians, one
# process, rotated interleaved rounds; each column's better config in bold-ish):
#
#   HV  B   CTAs/SM   wide us   many-warp us   winner
#   12   8    0.65      3.555      3.609       wide
#   12  16    1.30      5.386      5.450       wide
#   12  32    2.59      9.405      8.143       many-warp  (-13.4%)
#   12  64    5.19     13.860     13.596       many-warp  (-1.9%)
#   12  96    7.78     20.546     21.772       wide
#   12 128   10.38     33.851     33.992       wide
#   12 256   20.76     63.955     64.358       wide
#   16  16    1.73      5.568      5.545       many-warp  (-0.4%)
#   16  32    3.46      9.787      9.701       many-warp  (-0.9%)
#   16  64    6.92     17.856     17.604       many-warp  (-1.4%)
#   16 256   27.68     83.957     84.348       wide
#
# One window separates all eleven points: the many-warp shape wins for
# 1.5 <= CTAs/SM <= 7.5 and loses outside it. Below the window the launch is under
# one wave, so nothing overlaps and the wide shape's shorter instruction stream
# wins; above it the memory system is saturated (95% of the measured stream
# ceiling) and extra warps buy nothing while the narrower LDG.64s cost ~0.5%.
# Total work is invariant under (NW, VW), so this is a DISPATCH, not a constant.
# `-1` = resolve per shape; 1 = pin many-warp; 0 = pin wide.
CTASHAPE = int(os.environ.get("KDA_CTASHAPE", "-1"))
CTASHAPE_MIN_CTA_PER_SM = float(os.environ.get("KDA_CTASHAPE_MINOCC", "1.5"))
CTASHAPE_MAX_CTA_PER_SM = float(os.environ.get("KDA_CTASHAPE_MAXOCC", "7.5"))
# Candidate ladders as (NW, VW, MINCTA), most preferred first. Every entry is a
# measured optimum for its regime; the ladder exists because acceptance is
# "compiled with zero spill on THIS toolchain" (see AUTOCFG), so a shape must
# always have a fallback.
CAND_LATENCY = ((8, 2, 0), (4, 4, 0))     # small/mid B: many warps
CAND_STREAM = ((4, 4, 0), (8, 2, 0))      # large B: wide loads, few warps
# Minimum CTAs/SM a candidate must actually achieve to be accepted. BOTH shipped
# configs sit exactly ON their 2-CTA register cliff (NW=8 VW=2 lands on 128 = the
# cliff for a 256-thread CTA; NW=4 VW=4 lands on 255 = the cliff for 128 threads),
# and ONE register over halves the occupancy: the same kernel with `KDA_ZPF=0`
# allocates 129 registers, drops to 1 CTA/SM, and costs +13% at B=32 and +20% at
# B=64. So the acceptance test is not just "did it spill" but "did it keep the
# occupancy the config exists for" -- both read straight off the compiled binary.
MIN_CTA_PER_SM = int(os.environ.get("KDA_MIN_CTA_PER_SM", "2"))


def candidate_ladder(B, H):
    """(NW, VW, MINCTA) candidates for this shape, most preferred first."""
    if _PINNED or not AUTOCFG:
        return ((NW, VW, MINCTA),)
    if CTASHAPE >= 0:
        return CAND_LATENCY if CTASHAPE else CAND_STREAM
    cps = B * H / float(_num_sms())
    inwin = CTASHAPE_MIN_CTA_PER_SM <= cps <= CTASHAPE_MAX_CTA_PER_SM
    return CAND_LATENCY if inwin else CAND_STREAM
# 1: SHFL-BROADCAST the per-row coefficients instead of re-reading them from
# shared memory. The phase-1 `sEg/sQn/sKn` reads and phase-3's `sKn` re-read are
# warp-UNIFORM (every lane of a warp is on the same state row), so instead of
# NIT broadcast LDS per array the warp can hold those rows distributed one-per-
# lane (`NIT/32` registers per array, loaded with one conflict-free contiguous
# LDS each) and broadcast row `m` with `shfl.idx` from lane `m % 32` -- a
# compile-time constant, since the band loop is fully unrolled.
#
# Motivation is real and current (`ncu`, shipped kernel, B=64): shared loads are
# 291,840 instructions = ~71 per warp, only 8% of the instruction count, but
# `mio_throttle` 2.03 + `short_scoreboard` 2.02 = 4.05 is now the LARGEST stall
# term, ahead of `long_scoreboard` 2.49.
#
# MEASURED AND REJECTED -- see optimization.md round 5. It loses for two
# independent architectural reasons that the stall counters do not show:
# (a) SHFL issues on the same MIO pipe that `mio_throttle` measures, so it does
#     not move work off the contended port, it moves *more* work onto it:
#     3 shuffles/row in phase 1 + 1 in phase 3 = 256 MIO instructions per warp
#     against the 71 LDS it replaces, because ptxas merges four consecutive
#     rows of an array into one LDS.128 while a shuffle is inherently 1 row x
#     1 array x 32 bits.
# (b) it holds 3*NIT/32 coefficients live across the whole band loop, which is
#     the exact pattern rounds 2-4 measured at ~+10% per 2 long-lived registers.
SHFLB = int(os.environ.get("KDA_SHFLB", "0"))
# Debug/ablation only: 1 = skip the fused epilogue and write the PRE-norm r.
# Used by `--compare` to price the fused epilogue against the unfused path.
# Never a correctness path: `--check` always runs with the epilogue on.
NOEPI = int(os.environ.get("KDA_NOEPI", "0"))

# ---------------------------------------------------------------------------
# STATE-BAND CACHE POLICY (the lever this attempt adds).
#
# The `S` tile is the whole kernel: 98% of the bytes, touched EXACTLY TWICE per
# launch (one read into the register band, one write-back of the same lines) and
# never re-read on-chip. Everything else the kernel touches is tiny and wants to
# stay cached: `conv_weight` (48 KB, shared by every `b` of a head), `w` (512 B,
# shared by every CTA), and the conv-state/token/gate rows.
#
# So the band is exactly the traffic that should NOT be allowed to displace the
# hot data, and CuTeDSL exposes the hint: `autovec_copy(..., l1c_evict_priority=)`
# builds a specialised G2R/R2G copy atom carrying an eviction-priority hint
# instead of the universal copy.
#
# It is a HINT ONLY -- no value the kernel loads or stores changes, so it cannot
# affect correctness (verified anyway by the full `--check` suite).
#
# BUT the right hint is shape-dependent and the reason is a real property of
# decode, not of the benchmark: in a decode loop the SAME state tile is re-read on
# the NEXT token. When `2*B*H*K*V*4` is around L2 (B=64 -> 134 MB vs the B200's
# 126 MB) that inter-launch reuse is real and worth keeping, and `ncu` shows it --
# at B=64 only 71 MB read + 12 MB write reach DRAM out of 268 MB of touches. When
# the state is far past L2 (B=256 -> 537 MB) there is no reuse to protect and
# every allocated line is pure displacement.
#
# Hence a per-shape dispatch on the same measured quantity `GRID1` already uses.
#   normal / first / last / unchanged / noalloc  (band LOAD and STORE separately)
#   -1 (default) = resolve from the shape at compile time
_EP = {
    "normal": "EVICT_NORMAL", "first": "EVICT_FIRST", "last": "EVICT_LAST",
    "unchanged": "EVICT_UNCHANGED", "noalloc": "NO_ALLOCATE",
}
BANDEPL = os.environ.get("KDA_BANDEPL", "-1")     # band load policy
BANDEPS = os.environ.get("KDA_BANDEPS", "-1")     # band store policy
# State read+write bytes above which the band is hinted evict-first rather than
# left normal: past this there is no inter-token L2 reuse left to protect.
BANDEP_BYTES = float(os.environ.get("KDA_BANDEP_BYTES", "1.95e8"))


def _ep(name):
    """`CacheEvictionPriority` for a lever string (validated at build time)."""
    if name not in _EP:
        raise ValueError(f"unknown cache eviction priority {name!r}; "
                         f"expected one of {sorted(_EP)}")
    return getattr(cute_nv.CacheEvictionPriority, _EP[name])


_NSM = None


def _num_sms():
    """SM count of the current device (cached). Used only to resolve the grid-form
    window at compile time; falls back to the B200's 148 if torch cannot say."""
    global _NSM
    if _NSM is None:
        try:
            _NSM = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        except Exception:                              # pragma: no cover
            _NSM = 148
    return _NSM


def make_launcher(B, H, q_stride, k_stride, v_stride, z_stride,
                  g_stride, beta_stride, raw_gate, abl,
                  vw=None, mincta=None, nw=None):
    """Build the (fully static) launcher for a `(B, H)` fused conv+decode step.

    `nw` / `vw` / `mincta` override the module-level `NW` / `VW` / `MINCTA` so the
    autoconfig probe can build several candidates from one source. Everything
    derived from them below (NT, LPR, RPI, NIT, NPART, ...) is recomputed per
    call, so the candidates are genuinely distinct compilations, not a shared
    body.

    Thread map inside the CTA (NT = NW*32 threads); every quantity is a
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
    (NW, VW).
    """
    VW = globals()["VW"] if vw is None else vw
    MINCTA = globals()["MINCTA"] if mincta is None else mincta
    NW = globals()["NW"] if nw is None else nw
    RAW_GATE = raw_gate
    ABL = globals()["ABL"] if abl is None else abl
    BH = B * H
    # resolve the CTA-index form for THIS shape (see the GRID1 comment)
    G1 = GRID1
    if G1 < 0:
        # RE-SWEPT for this attempt (HV=12 AND HV=16, CUDA-graph medians, one
        # process, rotated interleaved rounds). With the 128-thread wide config the
        # 1-D form now wins or ties at EVERY shape measured -- HV=12 B=8/16/32/64/
        # 96/128/256: -4.2/-1.7/-1.7/-0.5/-0.3/-0.1/-0.03%, HV=16 B=8: -2.4%. It
        # also changes the register allocation (255 -> 199 regs), which is why the
        # source package's occupancy/byte window no longer applies: that window was
        # measured on a different allocation and expired with it.
        #
        # With the 256-thread latency config the sign FLIPS at B >= 32 (HV=12:
        # +2.4/+1.2/+1.8% at B=32/64/128), so the form is tied to the CTA shape.
        G1 = 1 if NW * 32 <= 128 else 0
    # `bh -> (b, h)` without a division. A power-of-two H is a shift/mask; H=12
    # (the production TP8 shard) is not, so use a compile-time MAGIC multiply:
    # b = (bh * MAG) >> MSH exactly for every bh < BH, verified below by
    # enumeration (BH <= a few thousand, so the check is exhaustive and free).
    MAG, MSH = 0, 0
    if G1 and (H & (H - 1)) != 0:
        for sh in range(12, 31):
            m = ((1 << sh) + H - 1) // H
            if m * (BH - 1) < (1 << 31) and all(
                    (n * m) >> sh == n // H for n in range(BH)):
                MAG, MSH = m, sh
                break
        G1 = 1 if MAG else 0              # no exact magic -> keep the 2-D grid
    # Resolve the band cache policy for THIS shape (see the BANDEPL comment).
    _big = (2.0 * BH * K * V * 4) > BANDEP_BYTES
    # NB: named BEP*, not EP*/EPS -- the kernel body is nested in this function and
    # `EPS` is the module-level RMSNorm epsilon, which a local `EPS` would shadow.
    BEPL = _ep(BANDEPL if BANDEPL != "-1" else ("first" if _big else "normal"))
    BEPS = _ep(BANDEPS if BANDEPS != "-1" else ("first" if _big else "normal"))
    HK = H * K                       # channels per packed block (q / k / v)
    D = 3 * HK                       # total packed channel count
    LPR = V // VW
    NT = NW * 32
    RPI = NT // LPR
    NIT = K // RPI
    MLPR = max(LPR, 32)
    NPART = NT // MLPR
    GPW = max(1, 32 // LPR)          # row groups sharing one warp
    NSHF = GPW.bit_length() - 1      # butterfly steps to fold them
    PUBL = min(LPR, 32)              # lanes of a warp that publish a partial
    # Phase 0 (the fused L2 norms) and the fused conv both cover the K head-dim
    # elements. Thread `t < NTK` owns the CPT CONTIGUOUS elements
    # dbase = t*CPT ... +CPT-1, so NTK*CPT == K exactly and every carrier thread's
    # conv channels are adjacent (which is what makes the history vectorise).
    NTK = K // CPT                   # threads that carry head-dim elements
    NWP = NTK // 32                  # warps holding norm partials / doing the conv
    UW = CPT                         # bf16 halves per conv-state vector unit
    NCU = (CSL * CPT) // UW          # units per group == CSL == 3 for every CPT
    EW = V // 32                     # epilogue columns per warp-0 lane
    # SHFL broadcast needs (a) a contiguous per-thread row band, so the
    # distributed load is one conflict-free 32-lane read per slot, and (b) the
    # band to be a whole number of 32-row slots, so `m % 32` / `m // 32` are
    # compile-time constants. Both hold for every shipped (NW, VW).
    SHFLB1 = SHFLB and ROWMAP and NIT % 32 == 0 and LPR >= 32
    NSLOT = NIT // 32 if SHFLB1 else 0
    assert V % VW == 0 and NT % LPR == 0 and K % RPI == 0, (
        f"bad shape NW={NW} VW={VW} LPR={LPR} RPI={RPI}")
    assert (LPR % 32 == 0) or (32 % LPR == 0), f"bad LPR={LPR}"
    assert K % CPT == 0 and NTK == NWP * 32 and NTK <= NT, (
        f"bad norm/conv config NW={NW} NT={NT} CPT={CPT} NTK={NTK} NWP={NWP}")
    assert NCU * UW == CSL * CPT and CSL * CPT % UW == 0
    # the vector units are only legal if every group base is 2*UW-byte aligned
    assert (D * CSL) % UW == 0 and D % CPT == 0 and HK % CPT == 0, (
        f"conv vector unit UW={UW} not aligned for D={D} H={H}")
    assert UW * 2 <= 16, f"conv history unit {UW*2} B exceeds one vector load"
    assert NIT * VW <= 200, (
        f"register band NIT*VW={NIT * VW} too large for NW={NW} VW={VW}")
    assert V == 32 * EW, "the epilogue assumes one warp covers V columns"

    @cute.kernel
    def kda_decode_conv_gated_kernel(
        mQv: cute.Tensor,     # (B, H, K/CPT, CPT)     bf16  RAW q
        mKv: cute.Tensor,     # (B, H, K/CPT, CPT)     bf16  RAW k
        mVv: cute.Tensor,     # (B, H, K/CPT, CPT)     bf16  RAW v
        mCWv: cute.Tensor,    # (D/CPT, CPT*4)         bf16  conv weight (no bias)
        mCSu: cute.Tensor,    # (B, D*3/UW, UW)        bf16  conv state (in place)
        mGv: cute.Tensor,     # (B,H,128/CPT,CPT) raw bf16 or fp32 log-decay
        mBeta: cute.Tensor,   # (B,H) raw bf16 or activated fp32
        mA: cute.Tensor,      # (H,) fp32 A_log, used by raw-gate mode
        mDt: cute.Tensor,     # (H,128) fp32 dt_bias, used by raw-gate mode
        mS: cute.Tensor,      # (BH, 128, 128) fp32   (in place)
        mO: cute.Tensor,      # (BH, 128)  bf16   post-norm output
        mZ: cute.Tensor,      # (BH, 128)  bf16   output gate
        mW: cute.Tensor,      # (128,)     fp32   RMSNorm weight
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # grid = (H, B): the conv needs h and b SEPARATELY (packed channel index
        # is j*H*K + h*K + di, state index is b*H + h), and a 2-D grid supplies
        # both with one IMAD instead of a runtime `bh / H`.
        if cutlass.const_expr(G1):
            bh, _, _ = cute.arch.block_idx()
            if cutlass.const_expr(MAG):
                # non-power-of-two H (the production HV=12): one IMAD + one SHR +
                # one IMAD, exact for every bh < BH by the build-time check above
                bb = (bh * MAG) >> MSH
                hh = bh - bb * H
            else:
                hh = bh % H              # H is a power of two -> AND, not a divide
                bb = bh // H             # -> SHR
        else:
            hh, bb, _ = cute.arch.block_idx()
            bh = bb * H + hh
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

        if cutlass.const_expr(RAW_GATE):
            raw_beta = mBeta[bb, hh].to(cutlass.Float32)
            beta = cute.arch.rcp_approx(
                cutlass.Float32(1.0)
                + cute.math.exp(-raw_beta, fastmath=True)
            )
        else:
            beta = mBeta[bb, hh]
        # This thread's first head-dim element. Carrier threads are tidx < NTK,
        # i.e. warps 0..NWP-1, so the carrier predicate is warp-uniform.
        dbase = tidx * CPT
        # Epilogue operands. Allocated at kernel scope (the 4.5.2 tracer rejects
        # a value whose first binding is inside a dynamic `if`) and filled by
        # warp 0 only.
        fw = cute.make_fragment(EW, cutlass.Float32)
        fzb = cute.make_fragment(EW, cutlass.BFloat16)
        # POST-CONV q/k/v (fp32) and the un-convolved log-decay for this thread's
        # CPT head-dim elements. These are produced by the fused conv below and
        # never re-read from global, which is why the source package's NHOLD /
        # ONEBAR levers (both of which re-read or re-order the raw q/k loads) are
        # gone: there are no raw q/k loads any more.
        fqv = cute.make_fragment(CPT, cutlass.Float32)
        fkv = cute.make_fragment(CPT, cutlass.Float32)
        fvv = cute.make_fragment(CPT, cutlass.Float32)
        fgv = cute.make_fragment(CPT, cutlass.Float32)
        # Conv operands, staged so that every conv LOAD is issued before the state
        # prefetch while the conv ARITHMETIC happens after it (CVPF=1).
        #   fcw[j, m*CW+s] : conv weight, block j, channel m, tap s  (one vector
        #                    copy of CPT*CW halves per block)
        #   fcs[j, u, e]   : conv history as NCU units of UW halves; channel m tap
        #                    s lives at flat half index 3*m+s
        #   fcx[j, m]      : the new RAW token
        fcw = cute.make_fragment(
            cute.make_layout((3, CPT * CW), stride=(CPT * CW, 1)), cutlass.BFloat16)
        fcs = cute.make_fragment(
            cute.make_layout((3, NCU, UW), stride=(NCU * UW, UW, 1)),
            cutlass.BFloat16)
        fcx = cute.make_fragment(
            cute.make_layout((3, CPT), stride=(CPT, 1)), cutlass.BFloat16)
        # The register band: also the destination of the state prefetch, and the
        # storage the decayed value is written back into.
        rb = [cute.make_fragment(VW, cutlass.Float32) for _ in range(NIT)]

        def conv_load(UW=UW, CPT=CPT, NCU=NCU, HK=HK, K=K, ABL=ABL,
                      RAW_GATE=RAW_GATE,
                      bb=bb, hh=hh, bh=bh, tidx=tidx,
                      mGv=mGv, mA=mA, mDt=mDt,
                      mCWv=mCWv, mCSu=mCSu,
                      mQv=mQv, mKv=mKv, mVv=mVv,
                      fgv=fgv, fcw=fcw, fcs=fcs, fcx=fcx):
            """Issue every conv load. All independent -> one round trip.

            Group index `cg` counts CPT-channel groups, so cg = j*(HK/CPT) +
            hh*(K/CPT) + tidx -- no division, and every base offset below is a
            multiple of the vector width it is loaded with.
            """
            if cutlass.const_expr(RAW_GATE):
                rate = cute.math.exp(mA[hh], fastmath=True)
                for m in cutlass.range_constexpr(CPT):
                    raw = mGv[bb, hh, tidx, m].to(cutlass.Float32)
                    dt = mDt[hh, tidx * CPT + m]
                    x = rate * (raw + dt)
                    fgv[m] = cutlass.Float32(-5.0) * cute.arch.rcp_approx(
                        cutlass.Float32(1.0)
                        + cute.math.exp(-x, fastmath=True)
                    )
            else:
                cute.autovec_copy(mGv[bb, hh, tidx, None], fgv)
            for j in cutlass.range_constexpr(3):
                cg = j * (HK // CPT) + hh * (K // CPT) + tidx
                if cutlass.const_expr(ABL != 3):
                    cute.autovec_copy(mCWv[cg, None], fcw[j, None])
                    for u in cutlass.range_constexpr(NCU):
                        if cutlass.const_expr(ABL != 1):
                            cute.autovec_copy(mCSu[bb, cg * NCU + u, None],
                                              fcs[j, u, None])
                cute.autovec_copy(
                    (mQv, mKv, mVv)[j][bb, hh, tidx, None],
                    fcx[j, None],
                )

        def _roll(j, UW=UW, CPT=CPT, NCU=NCU, CSL=CSL, HK=HK, K=K,
                      bb=bb, hh=hh, tidx=tidx, mCSu=mCSu, fcs=fcs, fcx=fcx):
            """[cs1, cs2, x_new] -- RAW bf16, in place. Rewrites `fcs` first (each
            channel reads both of its shifted halves before either is overwritten,
            and the CPT channels touch disjoint slots), then stores the NCU units
            back with the same vector width they were loaded with."""
            for m in cutlass.range_constexpr(CPT):
                h0 = CSL * m
                keep = [fcs[j, (h0 + sp + 1) // UW, (h0 + sp + 1) % UW]
                        for sp in range(CSL - 1)]
                for sp in cutlass.range_constexpr(CSL - 1):
                    fcs[j, (h0 + sp) // UW, (h0 + sp) % UW] = keep[sp]
                fcs[j, (h0 + CSL - 1) // UW, (h0 + CSL - 1) % UW] = fcx[j, m]
            cg = j * (HK // CPT) + hh * (K // CPT) + tidx
            for u in cutlass.range_constexpr(NCU):
                cute.autovec_copy(fcs[j, u, None], mCSu[bb, cg * NCU + u, None])

        def conv_compute(UW=UW, CPT=CPT, NCU=NCU, CSL=CSL, CW=CW, HK=HK, K=K,
                         ABL=ABL, CVST=CVST, bb=bb, hh=hh, tidx=tidx, mCSu=mCSu,
                         fcw=fcw, fcs=fcs, fcx=fcx,
                         fqv=fqv, fkv=fkv, fvv=fvv, _roll=_roll):
            """4-tap depthwise conv + SiLU in fp32, then roll the conv state."""
            for j in cutlass.range_constexpr(3):
                for m in cutlass.range_constexpr(CPT):
                    xn = fcx[j, m]
                    if cutlass.const_expr(ABL == 3):
                        # tuple index, not an `if`: `j` is a Python constant here,
                        # and a real `if` would be rewritten by the DSL into a
                        # traced block, inside which an assignment to a closure
                        # fragment of the enclosing kernel scope does not resolve.
                        (fqv, fkv, fvv)[j][m] = xn.to(cutlass.Float32)
                        continue
                    a = (fcw[j, m * CW + CSL].to(cutlass.Float32)
                         * xn.to(cutlass.Float32))
                    if cutlass.const_expr(ABL != 1):
                        for sp in cutlass.range_constexpr(CSL):
                            hl = CSL * m + sp
                            a = a + (fcw[j, m * CW + sp].to(cutlass.Float32)
                                     * fcs[j, hl // UW, hl % UW].to(cutlass.Float32))
                    # SiLU(a) = a / (1 + exp(-a)). rcp.approx.ftz.f32 is exact to
                    # 2^-23 and 1 + exp(-a) >= 1, so ftz never fires. Saturates
                    # correctly at both tails: exp(-a) -> inf gives 0, exp(-a) -> 0
                    # gives a.
                    xp = a * cute.arch.rcp_approx(
                        cutlass.Float32(1.0)
                        + cute.math.exp(-a, fastmath=True))
                    (fqv, fkv, fvv)[j][m] = xp
                if cutlass.const_expr(CVST and ABL == 0):
                    _roll(j)

        def conv_store(_roll=_roll):
            for j in cutlass.range_constexpr(3):
                _roll(j)

        # --- phase -1: prefetch ------------------------------------------------
        # z/w are consumed LAST (the epilogue) so warp 0 issues them FIRST: they
        # get the longest possible shadow and cost nothing on the critical path.
        # Then the conv loads, then every thread issues all NIT state loads (VW
        # columns each). None of the state loads depend on the conv or the norms,
        # so they fly while the prologue runs its barriers and shuffle chains.
        if cutlass.const_expr(ZPF and not NOEPI):
            if wid == 0:
                cute.autovec_copy(cute.local_tile(mW, (EW,), (lane,)), fw)
                cute.autovec_copy(
                    cute.local_tile(mZ[bb, hh, None], (EW,), (lane,)), fzb)
        # The conv loads go out BEFORE the state prefetch: the conv result gates
        # phase 0's butterfly, which gates the CTA barrier that gates phase 1 for
        # the WHOLE CTA, so queueing them behind NIT state LDGs makes every warp
        # wait a full LSU-queue drain for them (this is exactly the source
        # package's QKPF effect, now with a longer dependent chain).
        if cutlass.const_expr(NWP < NW):
            # Non-carrier warps must not touch conv memory (the roll would be
            # stored several times) and must not read uninitialised registers, so
            # they run the butterfly on zeros; `wid < NWP` discards their partials.
            for m in cutlass.range_constexpr(CPT):
                fqv[m] = cutlass.Float32(0.0)
                fkv[m] = cutlass.Float32(0.0)
                fvv[m] = cutlass.Float32(0.0)
                fgv[m] = cutlass.Float32(0.0)
            if wid < NWP:
                conv_load()
                if cutlass.const_expr(not CVPF):
                    conv_compute()
        else:
            conv_load()
            if cutlass.const_expr(not CVPF):
                conv_compute()
        if cutlass.const_expr(PREFETCH):
            for m in cutlass.range_constexpr(NIT):
                i = crow * NIT + m if ROWMAP else crow + m * RPI
                cute.autovec_copy(
                    cute.local_tile(mS[bh, i, None], (VW,), (clane,)), rb[m],
                    l1c_evict_priority=BEPL)
        if cutlass.const_expr(CVPF):
            if cutlass.const_expr(NWP < NW):
                if wid < NWP:
                    conv_compute()
            else:
                conv_compute()
        if cutlass.const_expr(not CVST):
            if cutlass.const_expr(NWP < NW):
                if wid < NWP:
                    conv_store()
            else:
                conv_store()

        # --- phase 0: fused L2 norms (needs the full 128-element q/k vectors) --
        # Carrier thread `tidx` owns head-dim elements dbase .. dbase+CPT-1, so
        # warps 0..NWP-1 jointly cover the whole K vector.
        p_qq = cutlass.Float32(0.0)
        p_kk = cutlass.Float32(0.0)
        p_kq = cutlass.Float32(0.0)
        for m in cutlass.range_constexpr(CPT):
            qt = fqv[m]              # post-conv, already resident
            kt = fkv[m]
            p_qq = p_qq + qt * qt
            p_kk = p_kk + kt * kt
            p_kq = p_kq + kt * qt
        for off in cutlass.range_constexpr(5):        # warp butterfly sum
            sh = 1 << (4 - off)
            p_qq = p_qq + cute.arch.shuffle_sync_bfly(p_qq, sh)
            p_kk = p_kk + cute.arch.shuffle_sync_bfly(p_kk, sh)
            p_kq = p_kq + cute.arch.shuffle_sync_bfly(p_kq, sh)

        sum_qq = p_qq
        sum_kk = p_kk
        sum_kq = p_kq
        if cutlass.const_expr(NWP > 1):
            # more than one warp holds head-dim elements -> exchange through SMEM
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
                for wq in cutlass.range_constexpr(NWP):
                    sum_qq = sum_qq + rf[wq]
                cute.autovec_copy(sRed[1, None], rf)
                for wq in cutlass.range_constexpr(NWP):
                    sum_kk = sum_kk + rf[wq]
                cute.autovec_copy(sRed[2, None], rf)
                for wq in cutlass.range_constexpr(NWP):
                    sum_kq = sum_kq + rf[wq]
            else:
                for wq in cutlass.range_constexpr(NWP):
                    sum_qq = sum_qq + sRed[0, wq]
                    sum_kk = sum_kk + sRed[1, wq]
                    sum_kq = sum_kq + sRed[2, wq]
        # NWP == 1: the whole reduction already lives in warp 0, so BOTH the SMEM
        # exchange and one of the two prologue CTA barriers disappear.
        rq = cute.math.rsqrt(sum_qq, fastmath=True)
        rk = cute.math.rsqrt(sum_kk, fastmath=True)
        P = rq * rk * sum_kq             # kn . qn, the rank-1 output coupling

        # scale-before-publish: needs a SECOND barrier, because these stores
        # depend on rq/rk which only exist after the first one.
        if wid < NWP:
            for m in cutlass.range_constexpr(CPT):
                di = dbase + m
                sQn[di] = fqv[m] * rq
                sKn[di] = fkv[m] * rk
                sEg[di] = cute.math.exp(fgv[m], fastmath=True)
                sV[di] = fvv[m]
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
                cute.autovec_copy(
                    cute.local_tile(mS[bh, i, None], (VW,), (clane,)), rb[m],
                    l1c_evict_priority=BEPL)
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
            if cutlass.const_expr(not NOEPI):
                if wid == 0:
                    if cutlass.const_expr(not ZPF):
                        cute.autovec_copy(cute.local_tile(mW, (EW,), (lane,)), fw)
                        cute.autovec_copy(
                            cute.local_tile(mZ[bb, hh, None], (EW,), (lane,)), fzb)
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
                # ablation: write the PRE-norm r (what the un-fused kernel emits)
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
                uf[k] = beta * (sV[lcol] - b)

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
                cute.autovec_copy(
                    fo, cute.local_tile(mS[bh, i, None], (VW,), (clane,)),
                    l1c_evict_priority=BEPS)

        if cutlass.const_expr(EPIORD):
            epilogue()
            writeback()
        else:
            writeback()
            epilogue()

    @cute.jit
    def launch(pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer,
               pCW: cute.Pointer, pCS: cute.Pointer,
               pG: cute.Pointer, pBeta: cute.Pointer,
               pA: cute.Pointer, pDt: cute.Pointer, pS: cute.Pointer,
               pO: cute.Pointer, pZ: cute.Pointer, pW: cute.Pointer, stream):
        # Grouped views: every conv access is one vector copy whose base offset is
        # a multiple of its own width (see the CPT comment). CPT=1 degenerates to
        # the scalar form.
        qkv_shape = (B, H, K // CPT, CPT)
        mQv = cute.make_tensor(
            pQ,
            cute.make_layout(
                qkv_shape, stride=(q_stride, K, CPT, 1)
            ),
        )
        mKv = cute.make_tensor(
            pK,
            cute.make_layout(
                qkv_shape, stride=(k_stride, K, CPT, 1)
            ),
        )
        mVv = cute.make_tensor(
            pV,
            cute.make_layout(
                qkv_shape, stride=(v_stride, K, CPT, 1)
            ),
        )
        mCWv = cute.make_tensor(
            pCW, cute.make_layout((D // CPT, CPT * CW), stride=(CPT * CW, 1)))
        mCSu = cute.make_tensor(
            pCS, cute.make_layout((B, (D * CSL) // UW, UW),
                                  stride=(D * CSL, UW, 1)))
        mGv = cute.make_tensor(
            pG,
            cute.make_layout(
                (B, H, K // CPT, CPT),
                stride=(g_stride, K, CPT, 1),
            ),
        )
        mBeta = cute.make_tensor(
            pBeta,
            cute.make_layout((B, H), stride=(beta_stride, 1)),
        )
        mA = cute.make_tensor(pA, cute.make_layout(H))
        mDt = cute.make_tensor(
            pDt, cute.make_layout((H, K), stride=(K, 1))
        )
        mS = cute.make_tensor(pS, cute.make_layout((BH, K, V), stride=(K * V, V, 1)))
        mO = cute.make_tensor(pO, cute.make_layout((BH, V), stride=(V, 1)))
        mZ = cute.make_tensor(
            pZ,
            cute.make_layout((B, H, V), stride=(z_stride, V, 1)),
        )
        mW = cute.make_tensor(pW, cute.make_layout(V))
        kda_decode_conv_gated_kernel(
            mQv, mKv, mVv, mCWv, mCSu, mGv, mBeta, mA, mDt,
            mS, mO, mZ, mW).launch(
            grid=([BH, 1, 1] if G1 else [H, B, 1]),
            block=[NT, 1, 1], stream=stream,
            min_blocks_per_mp=MINCTA,
        )

    return launch


# ---------------------------------------------------------------------------
# Host driver: compiles once per (BH, device) and dispatches with ~3 us of work.
# ---------------------------------------------------------------------------
# `torch._C._cuda_getCurrentRawStream` is the cheap stream query (0.05 us);
# `torch.cuda.current_stream()` costs 2.0 us because it builds a Stream object.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream          # type: ignore[attr-defined]
except AttributeError:                                        # pragma: no cover
    def _raw_stream(dev):
        return torch.cuda.current_stream(dev).cuda_stream


def _ptr_dtypes(raw_gate):
    gate_dtype = cutlass.BFloat16 if raw_gate else cutlass.Float32
    return (
        cutlass.BFloat16,   # q (raw, pre-conv)
        cutlass.BFloat16,   # k (raw, pre-conv)
        cutlass.BFloat16,   # v (raw, pre-conv)
        cutlass.BFloat16,   # conv_weight
        cutlass.BFloat16,   # conv_state
        gate_dtype,         # raw gate or activated log-decay
        gate_dtype,         # raw beta or activated beta
        cutlass.Float32,    # A_log
        cutlass.Float32,    # dt_bias
        cutlass.Float32,    # S
        cutlass.BFloat16,   # o
        cutlass.BFloat16,   # z
        cutlass.Float32,    # w
    )


def func_attrs(comp):
    """`(num_regs, local_bytes, smem_bytes)` of a compiled program, or None.

    Same driver-query trick as `spill_bytes` (which this generalises), but it also
    returns the register count -- and the register count is what actually decides
    this kernel's speed in the latency-bound mid-`B` regime, because
    `CTAs/SM = 65536 // (NT * num_regs)` and the whole kernel is one CTA per
    (b, h) tile. `ncu` on the shipped config reports 255 registers -> 2 CTAs/SM
    -> 12.5% occupancy, and at B=32-128 that occupancy, not bandwidth, is the
    binding constraint (see optimization.md round 1).
    """
    try:
        import cuda.bindings.driver as _drv
        from cutlass.base_dsl.runtime import cuda as _ch

        ex = comp.to(None)
        libs = ex.jit_module.cuda_library
        syms = list(comp.kernel_info.keys())
        if not libs or not syms:
            return None
        A = _drv.CUfunction_attribute
        out = None
        for sym in syms:
            for lib in libs:
                try:
                    fn = _ch.get_function_from_kernel(
                        _ch.get_library_kernel(lib, sym))
                except Exception:
                    continue
                vals = []
                for att in (A.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            A.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES):
                    err, val = _drv.cuFuncGetAttribute(att, fn)
                    if int(err) != 0:
                        return None
                    vals.append(int(val))
                out = tuple(vals) if out is None else tuple(
                    max(a, b) for a, b in zip(out, vals))
        return out
    except Exception:                # pragma: no cover - probe is best-effort
        return None


def ctas_per_sm(regs, nt=None, smem=0):
    """CTAs resident per SM for `regs` registers/thread at `nt` threads/CTA."""
    nt = NW * 32 if nt is None else nt
    lim = 65536 // (nt * max(1, regs)) if regs else 32
    if smem:
        lim = min(lim, 233472 // max(1, smem))
    return max(0, min(lim, 32, 2048 // nt))


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


# The autoconfig decision is a property of the *toolchain*, not of the shape, so
# it is resolved once per process and reused. Each later shape still verifies
# that the chosen config did not spill for it, which costs one driver query.
_AUTO_CHOICE = None      # (nw, vw, mincta) once resolved
_AUTO_LOG = []           # [(cand, regs, spill_bytes, verdict)] for reporting
_AUTO_BAD = set()        # candidates THIS toolchain cannot compile without spill


def chosen_config():
    """(nw, vw, mincta, how) actually in use -- reported by `--bench`."""
    if _PINNED:
        return (NW, VW, MINCTA, "pinned by environment")
    if not AUTOCFG:
        return (NW, VW, MINCTA, "autoconfig disabled")
    if _AUTO_CHOICE is None:
        return (NW, VW, MINCTA, "not yet compiled")
    return (*_AUTO_CHOICE, "per-shape ladder + zero-spill probe")


class _Launch:
    """A launch site: compiled program + a pre-built C argument pack.

    The pack holds the addresses of the `ctypes.c_void_p` pointer slots.
    Those addresses never change, so a call only has to store the current
    tensors' device pointers into the slots and invoke the compiled entry -- no
    dlpack, no descriptor build, no argument marshalling. Every launch therefore
    reads whatever the caller's live tensors currently hold; nothing about the
    data is cached.
    """

    __slots__ = ("comp", "ptrs", "descs", "stream", "adapted", "exe_args",
                 "packed", "capi", "cures", "raw_stream", "B", "H", "dev",
                 "q_stride", "k_stride", "v_stride", "z_stride",
                 "g_stride", "beta_stride", "fast")

    def __init__(self, comp, ptrs, stream, adapted, exe_args, raw_stream,
                 B, H, dev, q_stride, k_stride, v_stride, z_stride,
                 g_stride, beta_stride):
        self.comp = comp
        self.ptrs = tuple(ptrs)
        self.descs = tuple(p._desc for p in ptrs)
        self.stream = stream            # keep alive: exe_args references it
        self.adapted = adapted          # keep alive: adapter-owned buffers
        self.exe_args = exe_args
        self.raw_stream = raw_stream
        self.B, self.H, self.dev = B, H, dev
        (
            self.q_stride,
            self.k_stride,
            self.v_stride,
            self.z_stride,
            self.g_stride,
            self.beta_stride,
        ) = (
            q_stride, k_stride, v_stride, z_stride, g_stride, beta_stride
        )
        # Private-API fast path: reuse one packed ctypes array and call the
        # compiled C entry directly. Falls back to the public call if the DSL
        # internals ever move.
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
                raise RuntimeError(f"CUDA error {err} launching kda gated kernel")
        else:                            # pragma: no cover
            self.comp.run_compiled_program(self.exe_args)


class KdaDecodeConvGated:
    """Host driver. Compiles once per (B, H, device) and launches with ~3 us of
    host work, which is what the importable `kda_decode_conv_gated` API costs."""

    def __init__(self, *, raw_gate=False, abl=None):
        self.raw_gate = raw_gate
        self.abl = abl
        self._comp = {}
        self._cfg = {}
        self._sites = {}
        self._last = None    # single-entry fast lookup

    def _compiled(self, B, H, dev, strides, args, stream):
        key = (B, H, dev, *strides)
        comp = self._comp.get(key)
        if comp is None:
            if _PINNED or not AUTOCFG:
                comp = cute.compile(
                    make_launcher(
                        B, H, *strides, self.raw_gate, self.abl,
                        VW, MINCTA, NW
                    ),
                    *args,
                    stream,
                )
                self._cfg[key] = (NW, VW, MINCTA)
            else:
                comp = self._autocompile(B, H, strides, args, stream)
                self._cfg[key] = _AUTO_CHOICE or (NW, VW, MINCTA)
            self._comp[key] = comp
        return comp

    def config_of(self, B, H, dev=None):
        """(nw, vw, mincta) compiled for this shape -- per shape, unlike
        `chosen_config()`, which reports the most recent compile."""
        if dev is None:
            for (b, h, d, *_), c in self._cfg.items():
                if (b, h) == (B, H):
                    return c
            return None
        for (b, h, d, *_), config in self._cfg.items():
            if (b, h, d) == (B, H, dev):
                return config
        return None

    def _autocompile(self, B, H, strides, args, stream):
        """Compile this shape's candidates in preference order; keep the first
        that does not spill. See `spill_bytes` and the `AUTOCFG` comment for why
        zero spill is the right acceptance test, and `candidate_ladder` for why
        the preference order is per shape."""
        global _AUTO_CHOICE
        order = [c for c in candidate_ladder(B, H) if c not in _AUTO_BAD]
        if not order:                      # every candidate spilled somewhere
            order = list(candidate_ladder(B, H))[-1:]
        comp = None
        for i, (nwc, vw, mc) in enumerate(order):
            comp = cute.compile(
                make_launcher(
                    B, H, *strides, self.raw_gate, self.abl,
                    vw, mc, nwc
                ),
                *args,
                stream,
            )
            at = func_attrs(comp)
            regs, sp = (at[0], at[1]) if at else (None, spill_bytes(comp))
            cps = ctas_per_sm(regs or 0, nwc * 32) if regs else None
            last = i == len(order) - 1
            if sp == 0 and (cps is None or cps >= MIN_CTA_PER_SM):
                verdict = f"accepted (zero spill, {regs} regs, {cps} CTAs/SM)"
            elif last:
                # The ladder ends on a config that has never spilled on any
                # toolchain measured; take it whatever the probe said.
                verdict = "accepted (end of ladder)"
            else:
                # A positive spill count rejects, and so does an unavailable
                # probe: without evidence, never gamble on a tight allocation.
                why = ("rejected: spills" if sp else
                       f"rejected: only {cps} CTAs/SM ({regs} regs)"
                       if cps is not None and cps < MIN_CTA_PER_SM else
                       "rejected: probe unavailable")
                _AUTO_LOG.append(((nwc, vw, mc), regs, sp, why))
                _AUTO_BAD.add((nwc, vw, mc))
                continue
            _AUTO_LOG.append(((nwc, vw, mc), regs, sp, verdict))
            _AUTO_CHOICE = (nwc, vw, mc)
            return comp
        return comp

    def _site(self, B, H, dev, rs, strides):
        key = (B, H, dev, rs, *strides)
        site = self._sites.get(key)
        if site is None:
            ptrs = [
                make_ptr(dt, 0, cute.AddressSpace.gmem, assumed_align=16)
                for dt in _ptr_dtypes(self.raw_gate)
            ]
            stream = cuda.CUstream(rs)
            comp = self._compiled(B, H, dev, strides, ptrs, stream)
            exe_args, adapted = comp.generate_execution_args(*ptrs, stream)
            site = _Launch(
                comp, ptrs, stream, adapted, exe_args, rs,
                B, H, dev, *strides
            )
            self._sites[key] = site
        self._last = site
        return site

    @staticmethod
    def _shape(q, g):
        B = q.shape[0]
        H = g.shape[1]
        return B, H

    def prepare(self, mixed_qkv, conv_weight, conv_state, g, beta, S, z, w):
        """Return a callable that launches on these exact tensors with no Python
        argument work at all, plus the output tensor it writes. Used for
        pure-kernel timing; the kernel still reads live tensor contents."""
        B, H = self._shape(mixed_qkv, g)
        q, k, v = (
            item.view(B, H, K)
            for item in mixed_qkv.split(H * K, dim=-1)
        )
        strides = (
            q.stride(0), k.stride(0), v.stride(0), z.stride(0),
            g.stride(0), beta.stride(0),
        )
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=g.device)
        dev = g.device.index
        site = self._site(B, H, dev, _raw_stream(dev), strides)
        for slot, t in zip(
            site.descs,
            (q, k, v, conv_weight, conv_state, g, beta, g, g, S, o, z, w),
        ):
            slot.value = t.data_ptr()
        return site.fire, o

    def __call__(
        self, q, k, v, conv_weight, conv_state, g, beta, S, z, w,
        A_log=None, dt_bias=None,
    ):
        B, H = self._shape(q, g)
        dev = g.device.index
        rs = _raw_stream(dev)
        strides = (
            q.stride(0), k.stride(0), v.stride(0), z.stride(0),
            g.stride(0), beta.stride(0),
        )
        site = self._last
        if (site is None or site.B != B or site.H != H or site.dev != dev
                or site.raw_stream != rs
                or (
                    site.q_stride,
                    site.k_stride,
                    site.v_stride,
                    site.z_stride,
                    site.g_stride,
                    site.beta_stride,
                ) != strides):
            D = 3 * H * K
            assert q.shape == k.shape == v.shape == (B, H, K)
            if self.abl != 3:
                assert conv_weight.shape == (D, CW), conv_weight.shape
                assert conv_state.shape == (B, D, CSL), conv_state.shape
            assert g.shape == (B, H, K) and beta.shape == (B, H)
            assert S.shape == (B, H, K, V)
            assert z.shape == (B, H, V) and w.shape == (V,)
            site = self._site(B, H, dev, rs, strides)
        qkv_inner_contiguous = all(
            item.stride()[1:] == (K, 1) for item in (q, k, v)
        )
        gate_dtype = torch.bfloat16 if self.raw_gate else torch.float32
        conv_inputs_ok = self.abl == 3 or (
            conv_weight.is_contiguous() and conv_state.is_contiguous()
        )
        if not (qkv_inner_contiguous and conv_inputs_ok
                and g.stride()[1:] == (K, 1)
                and beta.stride(1) == 1
                and g.dtype == gate_dtype and beta.dtype == gate_dtype
                and S.is_contiguous()
                and z.stride()[1:] == (V, 1) and w.is_contiguous()):
            raise ValueError(
                "q/k/v require contiguous [H,K] rows; all other inputs "
                "must be contiguous"
            )
        o = torch.empty((B, H, V), dtype=torch.bfloat16, device=g.device)
        d = site.descs
        A_log = g if A_log is None else A_log
        dt_bias = g if dt_bias is None else dt_bias
        for slot, tensor in zip(
            d,
            (
                q, k, v, conv_weight, conv_state, g, beta,
                A_log, dt_bias, S, o, z, w,
            ),
        ):
            slot.value = tensor.data_ptr()
        site.fire()
        return o


_ENGINE = KdaDecodeConvGated()
_ENGINE_RAW = KdaDecodeConvGated(raw_gate=True)
_ENGINE_RAW_NOCONV = KdaDecodeConvGated(raw_gate=True, abl=3)


# ---------------------------------------------------------------------------
# MANDATORY importable API: updates S in place, returns the POST-norm o.
# ---------------------------------------------------------------------------
def kda_decode_conv_gated_packed(mixed_qkv, conv_weight, conv_state, g, beta, S, z, w):
    """Fused causal-conv1d(+SiLU) -> KDA decode step -> gated RMSNorm.

    mixed_qkv  : [B, 3*H*128] bf16  RAW pre-conv packed projection [q|k|v]
    conv_weight: [3*H*128, 4] bf16  depthwise conv kernel (no bias)
    conv_state : [B, 3*H*128, 3] bf16, updated IN PLACE (RAW token history)
    g          : [B,H,128] fp32 log-decay (<= 0)   beta : [B,H] fp32
    S          : [B,H,128,128] fp32, updated IN PLACE
    z          : [B,H,128] bf16 gate               w    : [128] fp32
    -> o       : [B,H,128] bf16 (post-norm)
    """
    B, H = g.shape[:2]
    q, k, v = (
        item.view(B, H, K)
        for item in mixed_qkv.split(H * K, dim=-1)
    )
    return _ENGINE(
        q, k, v, conv_weight, conv_state, g, beta, S, z, w
    )


def kda_decode_conv_gated(q, k, v, conv_weight, conv_state, g, beta, S, z, w):
    """Decode from non-contiguous Q/K/V views produced by packed-QKV splitting.

    Batch strides may include gaps, as they do when Q/K/V are split from Kimi
    K3's larger fused projection. ``conv_state`` and ``S`` are updated in place.
    """
    return _ENGINE(
        q, k, v, conv_weight, conv_state, g, beta, S, z, w
    )


def kda_decode_conv_gated_raw(
    q,
    k,
    v,
    conv_weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    S,
    z,
    w,
):
    """Fused decode directly from Kimi-K3's BF16 projection views.

    Safe-gate and beta sigmoid activation are performed inside the CuTeDSL
    kernel, so no dtype conversion or contiguous materialization is needed.
    """
    return _ENGINE_RAW(
        q,
        k,
        v,
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        S,
        z,
        w,
        A_log,
        dt_bias,
    )


def kda_decode_gated_raw_strided(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    S,
    z,
    w,
):
    """KDA + gated RMSNorm from strided post-convolution Q/K/V views."""
    return _ENGINE_RAW_NOCONV(
        q,
        k,
        v,
        q,  # unused when ABL=3
        q,  # unused when ABL=3
        raw_gate,
        raw_beta,
        S,
        z,
        w,
        A_log,
        dt_bias,
    )


# ---------------------------------------------------------------------------
# Trusted torch reference (fp32).
# ---------------------------------------------------------------------------
def ref_conv(mixed_qkv, conv_weight, conv_state, H):
    """Reference width-4 depthwise causal conv + SiLU on the packed projection.

    Accumulates in fp32 and applies SiLU in fp32 (exactly what the kernel does),
    rolls `conv_state` IN PLACE with the RAW bf16 token values, and returns the
    post-conv (q, k, v) as fp32 `[B,H,128]` tensors.
    """
    x = mixed_qkv.float()                                   # [B, D]
    cw = conv_weight.float()                                # [D, 4]
    cs = conv_state.float()                                 # [B, D, 3]
    a = (cs[:, :, 0] * cw[:, 0] + cs[:, :, 1] * cw[:, 1]
         + cs[:, :, 2] * cw[:, 2] + x * cw[:, 3])
    xp = torch.nn.functional.silu(a)                         # [B, D] fp32
    # roll: [cs1, cs2, x_new] -- RAW values, built before the in-place write
    rolled = torch.stack((conv_state[:, :, 1], conv_state[:, :, 2], mixed_qkv),
                         dim=-1)
    conv_state.copy_(rolled)
    B = x.shape[0]
    xp = xp.view(B, 3, H, K)
    return xp[:, 0].contiguous(), xp[:, 1].contiguous(), xp[:, 2].contiguous()


def ref_conv_triton_emul(mixed_qkv, conv_weight, conv_state):
    """Bit-level emulation of sglang's Triton `causal_conv1d_update` arithmetic.

    The Triton kernel forms each tap product as `matrix_x * matrix_w` where BOTH
    operands are bf16, so every product is rounded to bf16 before it is added into
    the fp32 accumulator, and the output is rounded to bf16 once more. That is
    strictly less accurate than this package's fp32-product kernel, and it is the
    entire reason a direct fp32-reference-vs-sglang comparison at 5e-3 fails.

    Reproducing it here lets `--check` assert two separate things:
      (a) this emulation matches the real sglang kernel to bf16 ULP  -> proves the
          fused conv implements *sglang's* conv, tap order and all;
      (b) the CuTeDSL kernel matches the fp32 reference at the spec tolerance.
    Returns bf16 `[B, D]`, exactly like `causal_conv1d_update`.
    """
    cw = conv_weight
    acc = (conv_state[:, :, 0] * cw[:, 0]).float()
    acc = acc + (conv_state[:, :, 1] * cw[:, 1]).float()
    acc = acc + (conv_state[:, :, 2] * cw[:, 2]).float()
    acc = acc + (mixed_qkv * cw[:, 3]).float()
    return (acc / (1.0 + torch.exp(-acc))).to(torch.bfloat16)


def ref_pre_norm(q, k, v, g, beta, S):
    """Reference decode recurrence: updates S in place (fp32), returns pre-norm r."""
    qf = q.float(); kf = k.float(); vf = v.float()
    qn = qf / qf.norm(dim=-1, keepdim=True)
    kn = kf / kf.norm(dim=-1, keepdim=True)
    S.mul_(torch.exp(g.float()).unsqueeze(-1))                    # diag(exp(g)) @ S
    Stk = torch.einsum("bhkv,bhk->bhv", S, kn)                    # S^T kn
    u = beta.float().unsqueeze(-1) * (vf - Stk)
    S.add_(torch.einsum("bhk,bhv->bhkv", kn, u))                  # kn u^T
    return torch.einsum("bhkv,bhk->bhv", S, qn) * RSQRT_K


def ref_gate(r, z, w):
    """Reference gated RMSNorm, matching FusedRMSNormGated(activation='sigmoid'):
    rstd = 1/sqrt(mean(x^2)+eps); y = x*rstd*w; y = y*sigmoid(gate)."""
    rf = r.float()
    rstd = torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return rf * rstd * w.float() * torch.sigmoid(z.float())


def ref_step(mixed_qkv, conv_weight, conv_state, g, beta, S, z, w):
    """Full reference: updates conv_state AND S in place, returns
    (post-norm o, pre-norm r)."""
    H = g.shape[1]
    q, k, v = ref_conv(mixed_qkv, conv_weight, conv_state, H)
    r = ref_pre_norm(q, k, v, g, beta, S)
    return ref_gate(r, z, w), r


# ---------------------------------------------------------------------------
# Independent oracles.
#
#   * conv prologue: the REAL sglang Triton `causal_conv1d_update`
#     (sglang/srt/layers/attention/mamba/causal_conv1d_triton.py). It lives in a
#     different venv here, so the leaf source FILE is executed directly with a
#     one-symbol shim for `sglang.jit_kernel.utils.is_arch_support_pdl` (the
#     module's only sglang dependency). The Triton kernel itself is then the real
#     thing, byte for byte.
#   * gated-RMSNorm epilogue: the real `FusedRMSNormGated`, same trick, else a
#     self-contained Triton port of the same kernel body.
# ---------------------------------------------------------------------------
def _venv_src_candidates(rel):
    env = os.environ.get("KDA_SGLANG_ROOT")
    if env:
        yield os.path.join(env, rel)
    import sys as _sys
    for sp in _sys.path:
        if sp.endswith(("site-packages", "dist-packages")):
            yield os.path.join(sp, rel)
    import glob as _glob
    for pat in ("/workspace/*/*/*/*/.venv/lib/python3.*/site-packages/" + rel,
                "/workspace/*/*/*/.venv/lib/python3.*/site-packages/" + rel,
                "/usr/local/lib/python3.*/dist-packages/" + rel):
        for q in _glob.glob(pat):
            yield q


def _load_real_sglang_leaf(rel, modname, shims):
    """Execute a leaf sglang source file directly, with `shims` (name -> module)
    injected into sys.modules so its imports resolve without importing the whole
    sglang package (whose __init__ explodes across venvs)."""
    import importlib.util
    import sys as _sys
    import types
    cached = _sys.modules.get(modname)
    if cached is not None:
        return cached, getattr(cached, "__file__", "?")
    for path in _venv_src_candidates(rel):
        if not (path and os.path.exists(path)):
            continue
        keys = tuple(shims)
        saved = {k: _sys.modules.get(k) for k in keys}
        try:
            for name, mod in shims.items():
                parts = name.split(".")
                for d in range(1, len(parts)):
                    pkg_name = ".".join(parts[:d])
                    if pkg_name not in _sys.modules:
                        pkg = types.ModuleType(pkg_name)
                        pkg.__path__ = []
                        _sys.modules[pkg_name] = pkg
                _sys.modules[name] = mod
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            _sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            return mod, path
        except Exception:
            _sys.modules.pop(modname, None)
            for k, vv in saved.items():
                if vv is None:
                    _sys.modules.pop(k, None)
                else:
                    _sys.modules[k] = vv
    return None, None


_CONV_REL = "sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"


def sglang_conv_update():
    """(callable, name) for the real sglang `causal_conv1d_update`, or (None, why)."""
    try:
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_update)
        return causal_conv1d_update, "sglang causal_conv1d_update (installed)"
    except Exception:
        pass
    import types
    shim = types.ModuleType("sglang.jit_kernel.utils")
    # faithful to sglang's own definition: CUDA arch major >= 9 enables PDL
    shim.is_arch_support_pdl = (
        lambda: torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9)
    mod, where = _load_real_sglang_leaf(
        _CONV_REL, "_real_sglang_causal_conv1d_triton",
        {"sglang.jit_kernel.utils": shim})
    if mod is None:
        return None, "sglang causal_conv1d_triton source not found"
    return (mod.causal_conv1d_update,
            f"sglang causal_conv1d_update (real source: {where})")


_TRITON_NG = None


def _triton_norm_gate_fn():
    global _TRITON_NG
    if _TRITON_NG is not None:
        return _TRITON_NG
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_gated_sigmoid(x, gp, y, w, eps, D: tl.constexpr, BD: tl.constexpr):
        i_t = tl.program_id(0)
        x += i_t * D
        y += i_t * D
        gp += i_t * D
        o_d = tl.arange(0, BD)
        m_d = o_d < D
        b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_var = tl.sum(tl.where(m_d, b_x, 0.0) * tl.where(m_d, b_x, 0.0), axis=0) / D
        b_rstd = 1 / tl.sqrt(b_var + eps)
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_y = b_x * b_rstd * b_w
        b_g = tl.load(gp + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_y = b_y * tl.sigmoid(b_g)
        tl.store(y + o_d, b_y.to(y.dtype.element_ty), mask=m_d)

    def fn(x, z, w, eps=EPS):
        xs = x.reshape(-1, x.shape[-1]).contiguous()
        zs = z.reshape(-1, z.shape[-1]).contiguous()
        out = torch.empty_like(xs, dtype=torch.bfloat16)
        D = xs.shape[-1]
        _rms_norm_gated_sigmoid[(xs.shape[0],)](
            xs, zs, out, w, eps, D=D, BD=triton.next_power_of_2(D))
        return out.view(*x.shape)

    _TRITON_NG = fn
    return fn


_NG_REL = "sglang/srt/layers/attention/fla/fused_norm_gate.py"


def epilogue_oracle():
    """(callable(r, z, w) -> bf16 o, name)."""
    import types
    shim = types.ModuleType("sglang.srt.utils")
    shim.cdiv = lambda a, b: -(a // -b)
    shim.next_power_of_2 = lambda n: 1 << (n - 1).bit_length() if n > 0 else 1
    shim.is_cpu = lambda: False
    shim.is_npu = lambda: False
    shim.cpu_has_amx_support = lambda: False
    try:
        from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
        src = "sglang FusedRMSNormGated"
    except Exception:
        mod, where = _load_real_sglang_leaf(
            _NG_REL, "_real_sglang_fused_norm_gate", {"sglang.srt.utils": shim})
        if mod is not None:
            FusedRMSNormGated = mod.FusedRMSNormGated
            src = f"sglang FusedRMSNormGated (real source: {where})"
        else:
            try:
                from fla.modules.fused_norm_gate import FusedRMSNormGated
                src = "fla FusedRMSNormGated"
            except Exception:
                return (_triton_norm_gate_fn(),
                        "local triton port of sglang rms_norm_gated")

    mods = {}

    def fn(r, z, w, eps=EPS):
        key = (w.device, w.dtype)
        m = mods.get(key)
        if m is None:
            m = FusedRMSNormGated(V, eps=eps, activation="sigmoid",
                                  device=w.device, dtype=torch.float32)
            mods[key] = m
        with torch.no_grad():
            m.weight.copy_(w.float())
            return m(r.reshape(-1, V).float(), z.reshape(-1, V).float()).view(*r.shape)

    return fn, src


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
# HV=12 is the PRODUCTION shard layout (Kimi-K3 TP8: 96 heads / 8 ranks), and it
# is the shape this attempt is tuned for. HV=16 is kept as a regression row
# because the source package was tuned there.
HP = 12               # primary head count
TARGET = (64, HP)     # B, H  -> 201 MB state traffic at HV=12
SMALL = (4, HP)
BIG = (256, HP)
PRIMARY = (256, HP)   # the scored shape
BENCH_B = (1, 8, 32, 64, 128, 256)     # HV=12 sweep required by the spec
REGRESSION = (256, 16)                 # HV=16 regression row


def _make_inputs(B, H, gen, device="cuda", s_scale=1.0, w_scale=0.5):
    D = 3 * H * K
    mixed_qkv = gen((B, D), torch.bfloat16)
    conv_state = gen((B, D, CSL), torch.bfloat16)
    # Real depthwise conv weights are O(0.1-1); keep them in that range so the
    # conv output magnitude is representative rather than artificially large.
    conv_weight = (torch.randn((D, CW), device=device, dtype=torch.float32)
                   * w_scale).to(torch.bfloat16)
    z = gen((B, H, K), torch.bfloat16)
    g = -torch.rand((B, H, K), device=device, dtype=torch.float32)   # log-decay <= 0
    beta = torch.rand((B, H), device=device, dtype=torch.float32) * 0.98 + 0.01
    S = torch.randn((B, H, K, V), device=device, dtype=torch.float32) * s_scale
    w = torch.rand((V,), device=device, dtype=torch.float32) * 1.5 + 0.25
    return mixed_qkv, conv_weight, conv_state, g, beta, S, z, w


def _randn(device="cuda"):
    return lambda shape, dt: torch.randn(shape, device=device, dtype=dt)


def _run_both(inp, atol=5e-3, rtol=5e-3):
    """Run reference and kernel on independent copies of the mutable tensors.
    Returns (ok, o_err, S_err, cs_err)."""
    mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = inp
    cs_ref = conv_state.clone(); S_ref = S.clone()
    o_ref, _ = ref_step(mixed_qkv, conv_weight, cs_ref, g, beta, S_ref, z, w)
    cs_k = conv_state.clone(); S_k = S.clone()
    o_k = kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_k, g, beta, S_k, z, w)
    torch.cuda.synchronize()
    o_ok = torch.allclose(o_k.float(), o_ref, atol=atol, rtol=rtol)
    s_ok = torch.allclose(S_k, S_ref, atol=atol, rtol=rtol)
    c_ok = torch.equal(cs_k, cs_ref)      # a pure RAW copy: must be EXACT
    return (o_ok and s_ok and c_ok,
            (o_k.float() - o_ref).abs().max().item(),
            (S_k - S_ref).abs().max().item(),
            (cs_k.float() - cs_ref.float()).abs().max().item())


def _check_one(B, H, gen, atol=5e-3, rtol=5e-3, s_scale=1.0, w_scale=0.5):
    return _run_both(_make_inputs(B, H, gen, s_scale=s_scale, w_scale=w_scale),
                     atol, rtol)


def run_check():
    # The epilogue ablation is a timing tool only -- never validate with it on.
    assert not NOEPI, "KDA_NOEPI=1 writes the PRE-norm output; --check is invalid"
    assert not ABL, f"KDA_ABL={ABL} is a timing-only ablation; --check is invalid"
    torch.manual_seed(0)
    device = "cuda"

    def gens():
        yield "randn", (lambda shape, dt: torch.randn(shape, device=device, dtype=dt))
        yield "uniform", (lambda shape, dt: (torch.rand(shape, device=device, dtype=dt) * 4 - 2))
        yield "small", (lambda shape, dt: torch.randn(shape, device=device, dtype=dt) * 0.1)
        yield "big", (lambda shape, dt: torch.randn(shape, device=device, dtype=dt) * 6)

    all_ok = True
    # HV=12 (production) AND HV=16 (regression), both required by the spec.
    for (B, H) in (TARGET, SMALL, (64, 16), (4, 16), (3, 12), (5, 16)):
        for name, gen in gens():
            ok, oe, se, ce = _check_one(B, H, gen)
            all_ok = all_ok and ok
            print(f"  shape=(B={B},H={H}) pattern={name:7s} ok={ok} "
                  f"o_err={oe:.3e} S_err={se:.3e} cs_err={ce:.3e}")

    # state-magnitude sweep: the recurrence's cancellation makes the elementwise
    # tolerance on S the sharpest test of the on-chip precision.
    for sc in (0.01, 100.0, 1000.0):
        ok, oe, se, ce = _check_one(*SMALL, _randn(), s_scale=sc)
        all_ok = all_ok and ok
        print(f"  S_scale={sc:<8g} ok={ok} o_err={oe:.3e} S_err={se:.3e}")

    # conv-weight magnitude sweep (the conv accumulates 4 taps in fp32; large
    # weights push the SiLU into its saturating regions)
    for wc in (0.05, 1.0, 4.0):
        ok, oe, se, ce = _check_one(*SMALL, _randn(), w_scale=wc)
        all_ok = all_ok and ok
        print(f"  w_scale={wc:<8g} ok={ok} o_err={oe:.3e} S_err={se:.3e}")

    # ---- the FUSED CONV PROLOGUE vs the real sglang causal_conv1d_update -----
    conv_fn, cname = sglang_conv_update()
    print(f"  conv oracle: {cname}")
    conv_ok = True
    if conv_fn is None:
        conv_ok = False
        print("  conv cross-check UNAVAILABLE -> FAIL (the spec requires it)")
    else:
        for (B, H) in (SMALL, TARGET, (4, 16), (64, 16)):
            mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = _make_inputs(
                B, H, _randn())
            cs_ref = conv_state.clone()
            q_r, k_r, v_r = ref_conv(mixed_qkv, conv_weight, cs_ref, H)
            xp_ref = torch.cat(
                (q_r.reshape(B, -1), k_r.reshape(B, -1), v_r.reshape(B, -1)), dim=1)
            xp_emul = ref_conv_triton_emul(mixed_qkv, conv_weight, conv_state)
            cs_sg = conv_state.clone()
            out_sg = conv_fn(mixed_qkv, cs_sg, conv_weight, None,
                             activation="silu")
            torch.cuda.synchronize()
            # (a) we reproduce sglang's conv to within ONE bf16 ULP once its bf16
            #     tap products are emulated -> the fused tap order and the
            #     tap <-> state-slot mapping are sglang's. The residual is a
            #     handful of elements where Triton's approximate exp() rounds the
            #     bf16 output the other way (6 of 393216 at B=64); 2^-7 is the
            #     largest 1-ULP relative step in bf16, so this is "off by at most
            #     one representable value", not a loose tolerance.
            d = (out_sg.float() - xp_emul.float()).abs()
            em_ok = torch.allclose(out_sg.float(), xp_emul.float(),
                                   atol=1e-6, rtol=2.0 ** -7)
            nbit = int((d > 0).sum().item())
            # (b) sglang vs the fp32 reference: the residual is sglang's own bf16
            #     product rounding, reported so the gap is visible, not hidden.
            sg_err = (out_sg.float() - xp_ref).abs().max().item()
            bnd = (xp_emul.float() - xp_ref).abs().max().item()
            st_ok = torch.equal(cs_sg, cs_ref)
            conv_ok = conv_ok and em_ok and st_ok
            print(f"  conv vs sglang (B={B}): within_1ulp_of_bf16_emul={em_ok} "
                  f"({nbit}/{d.numel()} elems differ at all)  "
                  f"state_exact={st_ok}  "
                  f"|sglang-fp32ref|={sg_err:.3e} (bf16-product bound {bnd:.3e})")
    all_ok = all_ok and conv_ok

    # per-tap unit-weight probe: with conv_weight = e_tap the conv output IS one
    # specific history slot (or the new token), so this pins the tap <-> slot
    # mapping that a plausible-but-wrong kernel would get backwards.
    tap_ok = True
    for tap in range(CW):
        inp = list(_make_inputs(*SMALL, _randn()))
        cwt = torch.zeros_like(inp[1]); cwt[:, tap] = 1.0
        inp[1] = cwt
        ok, oe, se, ce = _run_both(tuple(inp))
        tap_ok = tap_ok and ok
        print(f"  conv_w==e_{tap}  ok={ok} o_err={oe:.3e} S_err={se:.3e}")
    all_ok = all_ok and tap_ok

    # --- the fused epilogue vs an INDEPENDENT gated-RMSNorm implementation ----
    oracle, oname = epilogue_oracle()
    inp = _make_inputs(*TARGET, _randn())
    mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = inp
    cs_ref = conv_state.clone(); S_ref = S.clone()
    q_r, k_r, v_r = ref_conv(mixed_qkv, conv_weight, cs_ref, g.shape[1])
    r_ref = ref_pre_norm(q_r, k_r, v_r, g, beta, S_ref)   # pre-norm r (fp32)
    o_ref_gate = ref_gate(r_ref, z, w)
    # NB: sglang/FLA's FusedRMSNormGated overwrites its `x` argument in place, so
    # the oracle must be handed a clone.
    o_oracle = oracle(r_ref.clone(), z, w).float()
    cs_k = conv_state.clone(); S_k = S.clone()
    o_k = kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_k, g, beta, S_k, z, w)
    torch.cuda.synchronize()
    epi_ok = torch.allclose(o_k.float(), o_oracle, atol=5e-3, rtol=5e-3)
    epi_err = (o_k.float() - o_oracle).abs().max().item()
    all_ok = all_ok and epi_ok
    print(f"  epilogue vs {oname}: ok={epi_ok} err={epi_err:.3e}")
    ref_gate_err = (o_ref_gate - o_oracle).abs().max().item()
    rg_ok = torch.allclose(o_ref_gate, o_oracle, atol=5e-3, rtol=5e-3)
    all_ok = all_ok and rg_ok
    print(f"  torch ref gate vs {oname}: ok={rg_ok} err={ref_gate_err:.3e}")

    # gate-value sweep: sigmoid is evaluated with exp + rcp.approx, so check the
    # saturating tails as well as the linear region.
    for zc in (-20.0, -6.0, 0.0, 6.0, 20.0):
        inp = list(_make_inputs(*SMALL, _randn()))
        inp[6] = torch.full_like(inp[6], zc)
        ok, oe, se, ce = _run_both(tuple(inp))
        all_ok = all_ok and ok
        print(f"  z=={zc:<7g} ok={ok} o_err={oe:.3e}")

    # SiLU tail sweep on the conv input: shift the pre-activation far into each
    # tail so `a * rcp(1+exp(-a))` is exercised where exp() overflows/underflows.
    # A SHIFT, not a constant: a constant-0 conv input makes the post-conv q
    # identically zero, i.e. 0/0 in the L2 normalization, which is ill-posed for
    # the reference too (both sides produce NaN and no tolerance is meaningful).
    for xc in (-60.0, -8.0, 8.0, 60.0):
        inp = list(_make_inputs(*SMALL, _randn()))
        inp[0] = (inp[0].float() + xc).to(torch.bfloat16)
        inp[2] = (inp[2].float() + xc).to(torch.bfloat16)
        ok, oe, se, ce = _run_both(tuple(inp))
        all_ok = all_ok and ok
        print(f"  conv-input+={xc:<7g} ok={ok} o_err={oe:.3e} S_err={se:.3e}")

    # conv-weight patterns (conv_weight is a live input, not a baked constant).
    # 0.0 is excluded for the same reason as above: an all-zero conv weight makes
    # q identically zero.
    for cwname, cwv in (("tiny", 0.01), ("ones", 1.0), ("neg", -1.0)):
        inp = list(_make_inputs(*SMALL, _randn()))
        inp[1] = torch.full_like(inp[1], cwv)
        ok, oe, se, ce = _run_both(tuple(inp))
        all_ok = all_ok and ok
        print(f"  conv_w=={cwname:6s} ok={ok} o_err={oe:.3e} S_err={se:.3e}")

    # RMSNorm weight patterns (w is a live input, not a baked constant)
    for wname, wv in (("ones", 1.0), ("small", 0.01), ("large", 8.0)):
        inp = list(_make_inputs(*SMALL, _randn()))
        inp[7] = torch.full_like(inp[7], wv)
        ok, oe, se, ce = _run_both(tuple(inp))
        all_ok = all_ok and ok
        print(f"  w=={wname:6s} ok={ok} o_err={oe:.3e}")

    # ---- repeated-step check: the conv state ROLLS, so the second step must see
    # the first step's output. This is the test a stale-input shortcut fails.
    inp = _make_inputs(*SMALL, _randn())
    mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = inp
    cs_ref = conv_state.clone(); S_ref = S.clone()
    cs_k = conv_state.clone(); S_k = S.clone()
    roll_ok = True
    for step in range(3):
        xs = torch.randn_like(mixed_qkv)
        o_ref, _ = ref_step(xs, conv_weight, cs_ref, g, beta, S_ref, z, w)
        o_k = kda_decode_conv_gated_packed(xs, conv_weight, cs_k, g, beta, S_k, z, w)
        torch.cuda.synchronize()
        roll_ok = (roll_ok
                   and torch.allclose(o_k.float(), o_ref, atol=5e-3, rtol=5e-3)
                   and torch.allclose(S_k, S_ref, atol=5e-3, rtol=5e-3)
                   and torch.equal(cs_k, cs_ref))
    all_ok = all_ok and roll_ok
    print(f"  3-step rolling conv_state ok={roll_ok}")

    # same-object mutation check (clean-kernel): warm on one set of tensors,
    # mutate every input in place, rerun -> must reflect the current contents.
    inp = _make_inputs(*SMALL, _randn())
    mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = inp
    kda_decode_conv_gated_packed(mixed_qkv, conv_weight, conv_state.clone(), g, beta,
                          S.clone(), z, w)
    mixed_qkv.mul_(0).add_(0.3); conv_weight.mul_(0).add_(0.4)
    conv_state.mul_(0).add_(-0.6); z.mul_(0).add_(1.1)
    S.mul_(0).add_(1.5); w.mul_(0).add_(0.7)
    ok, oe, se, ce = _run_both((mixed_qkv, conv_weight, conv_state, g, beta,
                                S, z, w))
    all_ok = all_ok and ok
    print(f"  same-object-mutation ok={ok} o_err={oe:.3e}")

    # new-object check (the launch site caches an argument pack keyed by shape;
    # this proves it re-points at freshly allocated same-shape tensors instead
    # of replaying the warmed-up ones).
    new_ok = True
    for _ in range(3):
        ok, _, _, _ = _check_one(*SMALL, _randn())
        new_ok = new_ok and ok
    all_ok = all_ok and new_ok
    print(f"  new-object rebind ok={new_ok}")

    # interleaved-shape check: alternating shapes must not reuse the wrong
    # cached launch site.
    inter_ok = True
    sets = [_make_inputs(B_, H_, _randn())
            for (B_, H_) in ((2, 12), (7, 16), (4, 12), (4, 16))]
    for _ in range(2):
        for st in sets:
            ok, _, _, _ = _run_both(st)
            inter_ok = inter_ok and ok
    all_ok = all_ok and inter_ok
    print(f"  interleaved-shape ok={inter_ok}")

    print("CORRECT: PASS" if all_ok else "CORRECT: FAIL")
    print("RESULT: " + json.dumps({
        "correct": bool(all_ok), "kernel_ms": None, "tflops": None,
        "gbps": None, "bound": "memory", "speedup": None,
    }))
    return all_ok


def _time_block(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    return _time_block(fn, iters)


def _bench_many(fns, warmup, iters, rounds=3):
    """Time several callables and return one median ms each.

    The measurements are INTERLEAVED, round-robin, after warming every callable
    first. Timing them one after the other instead biases whichever runs first:
    at B=4 the whole kernel is ~5 us, so the SM/HBM clock ramp during the first
    timed block showed up as the "raw launch" path measuring *slower* than the
    importable API that wraps it -- which is impossible. Interleaved rounds plus a
    median removes both the ramp and the allocator-placement noise.
    """
    for f in fns:
        for _ in range(warmup):
            f()
    torch.cuda.synchronize()
    samples = [[] for _ in fns]
    for _ in range(rounds):
        for i, f in enumerate(fns):
            # Settle on THIS callable before timing it. Without this, whichever
            # callable follows the heavy torch baseline block inherits its clock /
            # power state: at B=4 that made the raw-launch path measure 1.2 us
            # SLOWER than the importable API that wraps it.
            for _ in range(max(8, min(iters, 64))):
                f()
            samples[i].append(_time_block(f, iters))
    return [sorted(s)[len(s) // 2] for s in samples]


# ---------------------------------------------------------------------------
# CUDA-graph replay timing -- the honest number for a few-microsecond decode
# kernel, and what the spec's table is measured with.
# ---------------------------------------------------------------------------
def _graph_of(inp, reps):
    """Capture `reps` back-to-back launches of this kernel into a CUDA graph.

    The launch SITE must be built on the CAPTURE stream: `_Launch` holds a
    `CUstream` captured when the site was created, and a site bound to a
    different stream records NOTHING into the graph (the launch goes to a stream
    the graph is not capturing), so the replay would time an empty graph.
    Hence `prepare()` is called inside `with torch.cuda.stream(s)` and the graph
    is captured on that same `s`.

    Nothing about the data is baked in: the graph replays the same launch against
    the same live buffers, which the kernel reads (and updates in place) fresh on
    every replay -- the state really does evolve across replays.
    """
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        launch, o = _ENGINE.prepare(*inp)
        for _ in range(5):
            launch()
    s.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        for _ in range(reps):
            launch()
    return gr, launch, o


def _time_graph(gr, reps, iters=20, rounds=5):
    """Median per-launch ms of a captured graph (`reps` launches per replay)."""
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(rounds):
        a.record()
        for _ in range(iters):
            gr.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / (iters * reps))
    return sorted(out)[len(out) // 2]


def _graph_reps(B, H):
    """Launches per replay: keep a replay ~0.5-1.5 ms so the event pair and the
    graph-launch overhead are negligible at every B."""
    est_us = max(2.0, 2.0 * B * H * K * V * 4 / 6.0e12 * 1e6)
    return max(4, min(256, int(round(600.0 / est_us))))


def bench_shape(B, H, ref_baseline=False, iters=20, rounds=5, check=True):
    """One shape: CUDA-graph median us + GB/s (+ optional torch-reference ms).

    Returns a dict. Correctness is re-verified here on independent copies so a
    benchmark row can never come from a wrong kernel.
    """
    inp = _make_inputs(B, H, _randn())
    mixed_qkv, conv_weight, conv_state, g, beta, S0, z, w = inp
    correct = None
    if check:
        cs_ref = conv_state.clone(); S_ref = S0.clone()
        o_ref, _ = ref_step(mixed_qkv, conv_weight, cs_ref, g, beta, S_ref, z, w)
        cs_k = conv_state.clone(); S_k = S0.clone()
        o_k = kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_k, g, beta,
                                    S_k, z, w)
        torch.cuda.synchronize()
        correct = bool(torch.allclose(o_k.float(), o_ref, atol=5e-3, rtol=5e-3)
                       and torch.allclose(S_k, S_ref, atol=5e-3, rtol=5e-3)
                       and torch.equal(cs_k, cs_ref))

    # timed on scratch copies of the two in-place buffers (the kernel overwrites
    # them fully every launch, so reuse across replays is legitimate)
    cs_t = conv_state.clone(); S_t = S0.clone()
    reps = _graph_reps(B, H)
    gr, launch, _o = _graph_of((mixed_qkv, conv_weight, cs_t, g, beta, S_t, z, w),
                               reps)
    ms = _time_graph(gr, reps, iters=iters, rounds=rounds)
    # the importable API in a plain launch loop, for the host-overhead view
    cs_a = conv_state.clone(); S_a = S0.clone()
    api_ms = _bench(lambda: kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_a,
                                                  g, beta, S_a, z, w),
                    10, max(50, reps))
    cfg = _ENGINE.config_of(B, H) or chosen_config()[:3]
    row = {"cfg": "NW%d/VW%d/MC%d" % cfg,
           "HV": int(H), "B": int(B), "kernel_us": float(ms * 1e3),
           "gbps": float(2 * B * H * K * V * 4 / (ms * 1e-3) / 1e9),
           "api_us": float(api_ms * 1e3), "correct": correct}
    if ref_baseline:
        cs_b = conv_state.clone(); S_b = S0.clone()

        def baseline():
            q, k, v = ref_conv(mixed_qkv, conv_weight, cs_b, H)
            r = ref_pre_norm(q, k, v, g, beta, S_b)
            return ref_gate(r, z, w)

        row["baseline_ms"] = float(_bench(baseline, 5, 20))
    del gr
    return row


def run_suite(iters=20, rounds=5):
    """The spec's benchmark contract: CUDA-graph medians over the HV=12 sweep
    plus the HV=16 regression row, scored at HV=12 B=256."""
    torch.manual_seed(0)
    rows = []
    for B in BENCH_B:
        rows.append(bench_shape(B, HP, ref_baseline=(B == PRIMARY[0]),
                                iters=iters, rounds=rounds))
    rows.append(bench_shape(*REGRESSION, iters=iters, rounds=rounds))
    prim = next(r for r in rows if r["HV"] == PRIMARY[1] and r["B"] == PRIMARY[0])
    for (cand_, regs_, sp_, verdict_) in _AUTO_LOG:
        print(f"    probe NW={cand_[0]} VW={cand_[1]} MINCTA={cand_[2]}: "
              f"regs={regs_} spill={sp_} B/thread -> {verdict_}")
    print(f"  fixed knobs: CVPF={CVPF} CVST={CVST} CPT={CPT} EPIORD={EPIORD} "
          f"[{chosen_config()[3]}]")
    print("  HV   B   kernel_us    GB/s   %of8TB/s    api_us  correct  cta-shape")
    for r in rows:
        print(f"  {r['HV']:>2} {r['B']:>3}  {r['kernel_us']:9.3f} {r['gbps']:8.1f}"
              f"   {100*r['gbps']/8000:6.1f}%  {r['api_us']:9.3f}  {str(r['correct']):>5}"
              f"  {r['cfg']}")
    b_ms = prim["baseline_ms"]
    speedup = b_ms / (prim["kernel_us"] * 1e-3)
    correct = all(r["correct"] for r in rows)
    print(f"  baseline(torch ref conv+step+gated rmsnorm) @HV=12 B=256"
          f" = {b_ms*1e3:.1f} us")
    print(f"SPEEDUP: {speedup:.4f}")
    print("CORRECT: PASS" if correct else "CORRECT: FAIL")
    print("RESULT: " + json.dumps({
        "correct": bool(correct), "kernel_ms": float(prim["kernel_us"] * 1e-3),
        "tflops": None, "gbps": float(prim["gbps"]), "bound": "memory",
        "speedup": float(speedup),
        "per_shape": [{"HV": r["HV"], "B": r["B"], "kernel_us": r["kernel_us"],
                       "gbps": r["gbps"]} for r in rows],
    }))
    return correct


def run_bench(B=TARGET[0], H=TARGET[1], warmup=10, iters=50, compare=False):
    torch.manual_seed(0)
    inp = _make_inputs(B, H, _randn())
    mixed_qkv, conv_weight, conv_state, g, beta, S0, z, w = inp

    # correctness gate
    cs_ref = conv_state.clone(); S_ref = S0.clone()
    o_ref, r_ref = ref_step(mixed_qkv, conv_weight, cs_ref, g, beta, S_ref, z, w)
    cs_k = conv_state.clone(); S_k = S0.clone()
    o_k = kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_k, g, beta, S_k, z, w)
    torch.cuda.synchronize()
    correct = (torch.allclose(o_k.float(), o_ref, atol=5e-3, rtol=5e-3)
               and torch.allclose(S_k, S_ref, atol=5e-3, rtol=5e-3)
               and torch.equal(cs_k, cs_ref))

    # Baseline = the spec/reference implementation: the fp32 torch conv prologue
    # + recurrence + gated RMSNorm. Both paths mutate persistent state buffers in
    # place across timed iterations (no per-iter refresh copy on either side), so
    # the comparison is pure compute.
    cs_base = conv_state.clone(); S_base = S0.clone()

    def baseline():
        q, k, v = ref_conv(mixed_qkv, conv_weight, cs_base, H)
        r = ref_pre_norm(q, k, v, g, beta, S_base)
        return ref_gate(r, z, w)

    # Generated kernel: pure GPU launch time (compile + arg pack done once in
    # prepare()). The kernel reads the current contents of the scratch buffers on
    # every launch -- no stale-output shortcut. They are persistent buffers the
    # kernel overwrites in place (allowed reuse: fully overwritten before read).
    cs_scratch = conv_state.clone(); S_scratch = S0.clone()
    launch, _o = _ENGINE.prepare(mixed_qkv, conv_weight, cs_scratch, g, beta,
                                 S_scratch, z, w)
    cs_api = conv_state.clone(); S_api = S0.clone()

    # Headline number: the importable API itself (which is what external
    # cross-framework benchmarking times), the raw launch, and the reference
    # baseline -- all three interleaved so no one of them eats the clock ramp.
    raw_ms, api_ms, b_ms = _bench_many(
        [launch,
         lambda: kda_decode_conv_gated_packed(mixed_qkv, conv_weight, cs_api, g, beta,
                                       S_api, z, w),
         baseline],
        warmup, iters)

    bytes_moved = 2 * B * H * K * V * 4       # state read + write
    gbps = bytes_moved / (api_ms * 1e-3) / 1e9
    raw_gbps = bytes_moved / (raw_ms * 1e-3) / 1e9
    speedup = b_ms / api_ms
    peak = 8000.0
    cnw, cvw, cmc, chow = chosen_config()
    print(f"  shape=(B={B},H={H})  state={bytes_moved/1e6:.1f} MB  warmup={warmup} iters={iters}")
    print(f"  config: NW={cnw} VW={cvw} MINCTA={cmc} CVPF={CVPF} CVST={CVST}  [{chow}]")
    _bigb = (2.0 * B * H * K * V * 4) > BANDEP_BYTES
    print(f"  band cache policy: load={BANDEPL if BANDEPL != '-1' else ('first' if _bigb else 'normal')}"
          f" store={BANDEPS if BANDEPS != '-1' else ('first' if _bigb else 'normal')}"
          f"  [{'pinned' if BANDEPL != '-1' or BANDEPS != '-1' else 'per-shape dispatch'}]")
    for (cand_, regs_, sp_, verdict_) in _AUTO_LOG:
        print(f"    probe NW={cand_[0]} VW={cand_[1]} MINCTA={cand_[2]}: "
              f"regs={regs_} spill={sp_} B/thread -> {verdict_}")
    print(f"  kernel_ms={raw_ms:.5f} (raw launch, {raw_gbps:.1f} GB/s)")
    print(f"  api_ms={api_ms:.5f} (kda_decode_conv_gated, the scored path)")
    print(f"  baseline(torch ref conv+step+gated rmsnorm)_ms={b_ms:.5f}")
    print(f"  GB/s={gbps:.1f}  ({100*gbps/peak:.1f}% of {peak:.0f} GB/s peak)")

    if compare:
        # The composed two-launch path the fused kernel has to beat: sglang's
        # Triton causal_conv1d_update, then the un-fused decode+gated kernel.
        conv_fn, cname = sglang_conv_update()
        if conv_fn is not None:
            cs_c = conv_state.clone()
            conv_fn(mixed_qkv, cs_c, conv_weight, None, activation="silu")
            torch.cuda.synchronize()
            cv_ms = _bench(lambda: conv_fn(mixed_qkv, cs_c, conv_weight, None,
                                           activation="silu"), warmup, iters)
            print(f"  [compare] sglang causal_conv1d_update alone: {cv_ms*1e3:.2f} us"
                  f"   ({cname})")
            print(f"  [compare] fused (this kernel):               {api_ms*1e3:.2f} us")
            print(f"  [compare] composed >= conv + this kernel   = "
                  f"{(api_ms+cv_ms)*1e3:.2f} us")
        ng = _triton_norm_gate_fn()
        r_bf = r_ref.to(torch.bfloat16)
        ng(r_bf, z, w)
        torch.cuda.synchronize()
        ng_ms = _bench(lambda: ng(r_bf, z, w), warmup, iters)
        print(f"  [compare] separate gated-RMSNorm launch alone: {ng_ms*1e3:.2f} us")

    print(f"SPEEDUP: {speedup:.4f}")
    print("RESULT: " + json.dumps({
        "correct": bool(correct), "kernel_ms": float(api_ms),
        "raw_kernel_ms": float(raw_ms), "tflops": None,
        "gbps": float(gbps), "bound": "memory", "speedup": float(speedup),
    }))
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="also price the fusion against the composed launches")
    ap.add_argument("--mode", choices=["check", "benchmark", "profile"], default=None)
    ap.add_argument("--warmup_iterations", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--B", type=int, default=None,
                    help="single-shape mode (sweeps / profiling); default = the "
                         "full spec shape suite")
    ap.add_argument("--H", type=int, default=HP)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    mode = args.mode
    if args.check:
        mode = "check"
    if args.bench:
        mode = "benchmark"

    if mode == "profile":
        torch.manual_seed(0)
        mixed_qkv, conv_weight, conv_state, g, beta, S, z, w = _make_inputs(
            args.B if args.B is not None else PRIMARY[0], args.H, _randn())
        for _ in range(max(1, args.warmup_iterations)):
            kda_decode_conv_gated_packed(mixed_qkv, conv_weight, conv_state.clone(), g,
                                  beta, S.clone(), z, w)
        torch.cuda.synchronize()
        kda_decode_conv_gated_packed(mixed_qkv, conv_weight, conv_state.clone(), g,
                              beta, S.clone(), z, w)
        torch.cuda.synchronize()
        return

    if mode == "benchmark":
        if args.B is None:
            run_suite(iters=max(4, args.iterations), rounds=args.rounds)
        elif args.compare:
            run_bench(B=args.B, H=args.H, warmup=args.warmup_iterations,
                      iters=args.iterations, compare=True)
        else:
            # single-shape CUDA-graph point measurement (config sweeps)
            torch.manual_seed(0)
            cnw, cvw, cmc, chow = chosen_config()
            r = bench_shape(args.B, args.H, iters=max(4, args.iterations),
                            rounds=args.rounds)
            print(f"  config: NW={cnw} VW={cvw} MINCTA={cmc} CPT={CPT} "
                  f"[{chow}]")
            print(f"  HV={r['HV']} B={r['B']} kernel_us={r['kernel_us']:.3f} "
                  f"GB/s={r['gbps']:.1f} api_us={r['api_us']:.3f} "
                  f"correct={r['correct']}")
            print("RESULT: " + json.dumps({
                "correct": r["correct"],
                "kernel_ms": r["kernel_us"] * 1e-3, "tflops": None,
                "gbps": r["gbps"], "bound": "memory", "speedup": None,
                "per_shape": [{"HV": r["HV"], "B": r["B"],
                               "kernel_us": r["kernel_us"], "gbps": r["gbps"]}],
            }))
        raise SystemExit(0)

    run_check()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
