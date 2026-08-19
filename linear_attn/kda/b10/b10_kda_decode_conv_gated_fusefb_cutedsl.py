#!/usr/bin/env python3
"""b10_kda_decode_conv_gated_fusefb -- FUSE_FB variant (candidate, 2026-08-19):
the f_b gate projection ([B,128] @ [128,1536] GEMV) is computed INSIDE the
kernel from f_a and the f_b weight, removing the separate cuBLAS GEMV launch
and the raw-gate global round-trip. Derived from
b10_kda_decode_conv_gated_cutedsl.py; raw-gate mode only.

Original header:
blackwell_bf16_kda_decode_conv_gated -- fused single-token KDA decode step with
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
        mFa: cute.Tensor,     # (B, 128/CPT, CPT) bf16 f_a (gate GEMV input)
        mWfb: cute.Tensor,    # (H, 128, 128/CPT, CPT) bf16 f_b weight rows
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
                      mFa=mFa, mWfb=mWfb, mA=mA, mDt=mDt,
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
                # in-kernel f_b GEMV: raw[d] = sum_k f_a[b,k] * Wfb[h*128+d,k]
                # per thread: CPT output dims d = tidx*CPT+m, chunked over k
                # in CPT-wide vector loads; f_a chunks reused across the CPT
                # accumulators so each is loaded once per thread.
                fF = cute.make_fragment(CPT, cutlass.BFloat16)
                fWr = cute.make_fragment(CPT, cutlass.BFloat16)
                faccs = cute.make_fragment(CPT, cutlass.Float32)
                for m in cutlass.range_constexpr(CPT):
                    faccs[m] = cutlass.Float32(0.0)
                for kk in cutlass.range_constexpr(K // CPT):
                    cute.autovec_copy(mFa[bb, kk, None], fF)
                    for m in cutlass.range_constexpr(CPT):
                        cute.autovec_copy(
                            mWfb[hh, tidx * CPT + m, kk, None], fWr)
                        for c in cutlass.range_constexpr(CPT):
                            faccs[m] += (fF[c].to(cutlass.Float32)
                                         * fWr[c].to(cutlass.Float32))
                for m in cutlass.range_constexpr(CPT):
                    raw = faccs[m]
                    dt = mDt[hh, tidx * CPT + m]
                    x = rate * (raw + dt)
                    fgv[m] = cutlass.Float32(-5.0) * cute.arch.rcp_approx(
                        cutlass.Float32(1.0)
                        + cute.math.exp(-x, fastmath=True)
                    )
            else:
                # non-raw mode unsupported in the FUSE_FB variant
                pass
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
               pO: cute.Pointer, pZ: cute.Pointer, pW: cute.Pointer,
               pWfb: cute.Pointer, stream):
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
        # pG carries f_a (B, 128) in the FUSE_FB variant
        mFa = cute.make_tensor(
            pG,
            cute.make_layout(
                (B, K // CPT, CPT),
                stride=(g_stride, CPT, 1),
            ),
        )
        mWfb = cute.make_tensor(
            pWfb,
            cute.make_layout(
                (H, K, K // CPT, CPT),
                stride=(K * K, K, CPT, 1),
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
            mQv, mKv, mVv, mCWv, mCSu, mFa, mWfb, mBeta, mA, mDt,
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
    # FUSE_FB: gate pointer carries f_a (always bf16); +1 trailing Wfb ptr
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
        cutlass.BFloat16,   # W_fb (FUSE_FB)
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
        H = q.shape[1]  # g carries f_a (B, K) in the FUSE_FB variant
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
            (q, k, v, conv_weight, conv_state, g, beta, g, g, S, o, z, w, g),
        ):
            slot.value = t.data_ptr()
        return site.fire, o

    def __call__(
        self, q, k, v, conv_weight, conv_state, g, beta, S, z, w,
        A_log=None, dt_bias=None, w_fb=None,
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
            assert g.shape == (B, K) and beta.shape == (B, H)  # g = f_a
            assert w_fb is not None and w_fb.shape == (H * K, K) \
                and w_fb.is_contiguous() and w_fb.dtype == torch.bfloat16
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
                and g.stride(1) == 1
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
                A_log, dt_bias, S, o, z, w, w_fb,
            ),
        ):
            slot.value = tensor.data_ptr()
        site.fire()
        return o


_ENGINE_RAW_FB = KdaDecodeConvGated(raw_gate=True)


# ---------------------------------------------------------------------------
# MANDATORY importable API: updates S in place, returns the POST-norm o.
# ---------------------------------------------------------------------------
def kda_decode_conv_gated_raw_fusefb(
    q,
    k,
    v,
    conv_weight,
    conv_state,
    f_a,
    w_fb,
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
    return _ENGINE_RAW_FB(
        q,
        k,
        v,
        conv_weight,
        conv_state,
        f_a,
        raw_beta,
        S,
        z,
        w,
        A_log,
        dt_bias,
        w_fb=w_fb,
    )


