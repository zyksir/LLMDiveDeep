"""CuTe DSL single-kernel fused allreduce + residual + RMSNorm (AR+norm).

Computes, in ONE kernel launch on torch symmetric memory (NVLS):

    new_residual = all_reduce_sum(x) + residual   # x [B, H] rank-local
    norm         = rmsnorm(new_residual, eps) * gamma

CuTe DSL port of the shipped CUDA kernel ``mm_ar_norm_kernel``
(multimem_ar_cuda.py) — same proven protocol, two scheduling changes:

  * Grid 128 x 1024, ROLE-SPLIT: the first ``n_prod`` blocks are
    producers (the NVLS reduce path saturates at 8-16 blocks), the rest
    consumers (the local normalize needs full-GPU HBM bandwidth).
  * PRODUCERS, warp-per-owned-row (``row % world == rank``): 4-deep
    batched ``multimem.ld_reduce`` per lane (shallow chains stall ~5us
    per NVLS load), fp32 residual add in registers, ``multimem.st`` of
    the summed row (this multicast IS the new_residual delivery), then
    release the row flag with ``multimem.red.release.sys.max(flag,
    call_id)`` — max, not add, so varying B across calls and CUDA-graph
    replay stay correct.
  * CHANGE 1 — producer-side norm: the owning warp accumulated the
    row's sum-of-squares during the reduce, so it normalizes its OWN
    rows locally right after the broadcast (consumers skip them) —
    1/world of the normalize work leaves the consumer tail.
  * CHANGE 2 — ``n_prod`` swept on-node: 16 producer blocks won every
    shipping bucket (8 ties it at 4096 but loses ~17us at 8192 where
    the extra producer work — residual add + local store + ssq — needs
    the full 16-block NVLS plateau; 12 and 24 lose everywhere).
  * CONSUMERS, warp-per-foreign-row: spin on the local row-flag copy at
    RELAXED GPU scope with nanosleep backoff (tight acquire.sys spins
    from hundreds of warps throttle the producers), one acquire.sys
    confirm, then two local passes (fp32 sum-of-squares + warp
    butterfly + rsqrt, then scale by gamma into the local norm buffer).
  * Entry/exit barriers: multicast release-add flag + local acquire
    spin; the call counter is a per-block DEVICE round (each block
    bumps its own slot) -> CUDA-graph capture/replay safe, at the price
    of a FIXED grid (128 blocks of 1024 threads is co-resident on the
    148-SM B200; larger grids deadlock the exit barrier).

ALIASING contract: :meth:`CuteDslARNorm.allreduce_norm` returns VIEWS
of internal buffers (norm, new_residual), valid until the next call on
this instance. All ranks must call in lockstep with identical shapes;
``residual`` must be rank-identical.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cute.typing import Int32, Float32
from cutlass.cutlass_dsl import dsl_user_op

from .cutedsl_cache import load_or_compile


# ------------------------------------------------------------------ asm ops

@dsl_user_op
def _multimem_st_4xb32(
    mc_ptr: cute.Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """16B ``multimem.st`` accepting DSL Int32 values (the stock helper
    only takes raw MLIR values as produced by the ld_reduce ops)."""
    llvm.inline_asm(
        None,
        [
            mc_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(x).ir_value(loc=loc, ip=ip),
            Int32(y).ir_value(loc=loc, ip=ip),
            Int32(z).ir_value(loc=loc, ip=ip),
            Int32(w).ir_value(loc=loc, ip=ip),
        ],
        "multimem.st.weak.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _multimem_red_max_release_sys(
    mc_ptr: cute.Pointer,
    val: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Per-row flag release on the multicast VA: ``max(flag, call_id)``
    — monotonic AND correct when B varies across calls / graph replay
    (an add would double-count re-flagged rows)."""
    llvm.inline_asm(
        None,
        [
            mc_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "multimem.red.release.sys.global.max.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nanosleep(ns: int, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _spin_ge_acquire_sys(flag_ptr: cute.Pointer, target: Int32):
    """Spin until ``*flag >= target`` with acquire.sys loads (barrier
    flags: one spinning thread per block, contention is bounded)."""
    v = Int32(0)
    while v < target:
        v = cute.arch.load(flag_ptr.llvm_ptr, Int32, sem="acquire",
                           scope="sys")


@cute.jit
def _spin_ge_relaxed_gpu_backoff(flag_ptr: cute.Pointer, target: Int32):
    """Row-flag spin at RELAXED GPU scope with nanosleep backoff — a
    tight acquire.sys spin from hundreds of consumer warps throttles
    the producers on the same SMs (measured on the CUDA reference)."""
    v = cute.arch.load(flag_ptr.llvm_ptr, Int32, sem="relaxed",
                       scope="gpu")
    while v < target:
        _nanosleep(128)
        v = cute.arch.load(flag_ptr.llvm_ptr, Int32, sem="relaxed",
                           scope="gpu")


# ------------------------------------------------------------------ kernel

_N_BLOCKS = 128    # FIXED grid: must stay co-resident (exit barrier)
_N_THREADS = 1024  # 32 warps per block
_MAX_ROWS = 65536  # row-flag capacity


class _ARNormKernel:
    """Role-split fused AR+residual+RMSNorm kernel (see module doc)."""

    def __init__(self, rank: int, world: int, hidden: int,
                 n_prod: int, prod_norm: bool):
        if hidden % 256 != 0:
            raise ValueError(f"H={hidden} must be a multiple of 256")
        self.rank = rank
        self.world = world
        self.hidden = hidden
        self.n_prod = n_prod
        self.prod_norm = prod_norm
        self.n_blocks = _N_BLOCKS
        self.n_threads = _N_THREADS

    @cute.jit
    def __call__(
        self,
        x_mc_ptr: cute.Pointer,
        res_mc_ptr: cute.Pointer,
        mRes: cute.Tensor,        # [max_m, H] local view of symm res
        mNorm: cute.Tensor,       # [max_m, H] local norm output
        mResid: cute.Tensor,      # [B, H] residual input (rank-identical)
        mGamma: cute.Tensor,      # [H]
        row_flags: cute.Tensor,   # [MAX_ROWS] int32 local view
        row_flags_mc_ptr: cute.Pointer,
        bar: cute.Tensor,         # [2] int32 local view (symm)
        bar_mc_ptr: cute.Pointer,
        rounds: cute.Tensor,      # [n_blocks] int32 device rounds
        eps: Float32,
        stream: cuda.CUstream,
    ):
        rows_layout = cute.make_layout(
            (cute.size(mRes, mode=[0]), cute.size(mRes, mode=[1])),
            stride=(cute.size(mRes, mode=[1]), 1),
        )
        x_mc = cute.make_tensor(x_mc_ptr, rows_layout)
        res_mc = cute.make_tensor(res_mc_ptr, rows_layout)
        row_flags_mc = cute.make_tensor(row_flags_mc_ptr, row_flags.layout)
        bar_mc = cute.make_tensor(bar_mc_ptr, bar.layout)
        self.kernel(
            x_mc, res_mc, mRes, mNorm, mResid, mGamma,
            row_flags, row_flags_mc, bar, bar_mc, rounds, eps,
        ).launch(
            grid=[self.n_blocks, 1, 1],
            block=[self.n_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_mc: cute.Tensor,
        res_mc: cute.Tensor,
        mRes: cute.Tensor,
        mNorm: cute.Tensor,
        mResid: cute.Tensor,
        mGamma: cute.Tensor,
        row_flags: cute.Tensor,
        row_flags_mc: cute.Tensor,
        bar: cute.Tensor,
        bar_mc: cute.Tensor,
        rounds: cute.Tensor,
        eps: Float32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        bidx = cute.arch.make_warp_uniform(bidx)
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        world = self.world               # constexpr
        rank = self.rank                 # constexpr
        h = self.hidden                  # constexpr
        h128 = h // 8                    # Int128 chunks per row
        it_cnt = h // 256                # 16B chunks per lane per row
        batch = 4                        # ld_reduce MLP depth
        n_blk = it_cnt // batch
        if cutlass.const_expr(it_cnt % batch != 0):
            raise ValueError("H/256 must be a multiple of 4")
        wpb = self.n_threads // 32       # warps per block

        rows = cute.size(mResid, mode=[0])
        # rows this rank owns under row % world == rank
        own = (rows - Int32(rank) + Int32(world - 1)) // Int32(world)

        # ---- device round + entry barrier -------------------------------
        # thread 0 bumps this block's DEVICE round slot (single writer,
        # graph-replay safe); block 0 releases +1 on every rank's entry
        # flag; everyone waits for all ranks' arrivals of THIS call.
        s_call = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Int32, 1), cute.make_layout(1)
        )
        if tidx == 0:
            r = rounds[bidx] + Int32(1)
            rounds[bidx] = r
            s_call[0] = r
            if bidx == 0:
                utils.distributed.multimem_red_add1(
                    lock_ptr=bar_mc.iterator, scope="sys", order="release"
                )
        cute.arch.sync_threads()
        call_id = s_call[0]
        if tidx == 0:
            # flag counts 2 waves per call (entry + exit)
            _spin_ge_acquire_sys(
                bar.iterator,
                Int32(world) * (Int32(2) * call_id - Int32(1)),
            )
        cute.arch.sync_threads()

        # per-thread scratch: 4x i32 (ld_reduce regs) viewed as 8x bf16,
        # two Int128 packs viewed as 8x bf16 (data / gamma), fp32 acc
        ar32 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
        ar_bf = cute.make_tensor(
            cute.recast_ptr(ar32.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(8),
        )
        pk = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int128)
        pk_bf = cute.make_tensor(
            cute.recast_ptr(pk.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(8),
        )
        pk32 = cute.make_tensor(
            cute.recast_ptr(pk.iterator, dtype=cutlass.Int32),
            cute.make_layout(4),
        )
        pg = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int128)
        pg_bf = cute.make_tensor(
            cute.recast_ptr(pg.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(8),
        )
        acc8 = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)

        # 16B (Int128) base pointers for vectorized local traffic
        res128 = cute.recast_ptr(mRes.iterator, dtype=cutlass.Int128)
        norm128 = cute.recast_ptr(mNorm.iterator, dtype=cutlass.Int128)
        resid128 = cute.recast_ptr(mResid.iterator, dtype=cutlass.Int128)
        gamma128 = cute.recast_ptr(mGamma.iterator, dtype=cutlass.Int128)
        x_mc_base = x_mc.iterator
        res_mc_base = res_mc.iterator

        inv_h = 1.0 / float(h)

        if bidx < self.n_prod:
            # ---- producers: warp per OWNED row --------------------------
            pwarp = bidx * wpb + warp
            npwarps = self.n_prod * wpb
            for i in cutlass.range(pwarp, own, npwarps, unroll=1):
                row = i * Int32(world) + Int32(rank)
                acc8.fill(0.0)
                accv = acc8.load()
                for blk in cutlass.range_constexpr(n_blk):
                    # 4 independent NVLS reduced loads in flight before
                    # any consumption (the deep-MLP lesson)
                    regs = []
                    for u in cutlass.range_constexpr(batch):
                        e128 = lane + (blk * batch + u) * 32
                        regs.append(
                            utils.distributed.multimem_ld_reduce_8xbf16(
                                x_mc_base + (row * h + e128 * 8)
                            )
                        )
                    for u in cutlass.range_constexpr(batch):
                        e128 = lane + (blk * batch + u) * 32
                        off = row * h128 + e128
                        a0, a1, a2, a3 = regs[u]
                        ar32[0] = a0
                        ar32[1] = a1
                        ar32[2] = a2
                        ar32[3] = a3
                        arv = ar_bf.load().to(cutlass.Float32)
                        g_in = cute.make_tensor(
                            resid128 + off, cute.make_layout(1)
                        )
                        pk[0] = g_in[0]
                        rv = pk_bf.load().to(cutlass.Float32)
                        pk_bf.store((arv + rv).to(cutlass.BFloat16))
                        # ssq from the bf16-rounded res (matches ref) and
                        # a plain local copy so the norm pass re-reads it
                        # with same-thread visibility (L2 hit)
                        rf = pk_bf.load().to(cutlass.Float32)
                        accv = accv + rf * rf
                        g_res = cute.make_tensor(
                            res128 + off, cute.make_layout(1)
                        )
                        g_res[0] = pk[0]
                        _multimem_st_4xb32(
                            res_mc_base + (row * h + e128 * 8),
                            pk32[0], pk32[1], pk32[2], pk32[3],
                        )
                # release the row flag FIRST (remote consumers start),
                # then normalize the own row locally
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    _multimem_red_max_release_sys(
                        row_flags_mc.iterator + row, call_id
                    )
                if cutlass.const_expr(self.prod_norm):
                    acc8.store(accv)
                    ssq = acc8[0]
                    for j in cutlass.range_constexpr(1, 8):
                        ssq = ssq + acc8[j]
                    for sh in (16, 8, 4, 2, 1):
                        ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, sh)
                    rstd = cute.math.rsqrt(ssq * Float32(inv_h) + eps)
                    for it in cutlass.range(it_cnt, unroll=4):
                        e128 = lane + it * 32
                        off = row * h128 + e128
                        g_res = cute.make_tensor(
                            res128 + off, cute.make_layout(1)
                        )
                        pk[0] = g_res[0]
                        nv = pk_bf.load().to(cutlass.Float32) * rstd
                        g_g = cute.make_tensor(
                            gamma128 + e128, cute.make_layout(1)
                        )
                        pg[0] = g_g[0]
                        gv = pg_bf.load().to(cutlass.Float32)
                        pk_bf.store((nv * gv).to(cutlass.BFloat16))
                        g_n = cute.make_tensor(
                            norm128 + off, cute.make_layout(1)
                        )
                        g_n[0] = pk[0]
        else:
            # ---- consumers: warp per row, poll + normalize --------------
            cwarp = (bidx - self.n_prod) * wpb + warp
            ncwarps = (self.n_blocks - self.n_prod) * wpb
            wm1 = world - 1
            # with producer-side norm the consumers walk only FOREIGN
            # rows (j -> ascending row skipping row % world == rank)
            span = rows - own if cutlass.const_expr(self.prod_norm) \
                else rows
            for j in cutlass.range(cwarp, span, ncwarps, unroll=1):
                row = j
                if cutlass.const_expr(self.prod_norm):
                    q = j // Int32(wm1)
                    m = j % Int32(wm1)
                    row = q * Int32(world) + m
                    if m >= rank:
                        row = row + Int32(1)
                if lane == 0:
                    _spin_ge_relaxed_gpu_backoff(
                        row_flags.iterator + row, call_id
                    )
                    _spin_ge_acquire_sys(
                        row_flags.iterator + row, call_id
                    )
                cute.arch.sync_warp()
                acc8.fill(0.0)
                accv = acc8.load()
                for it in cutlass.range(it_cnt, unroll=4):
                    e128 = lane + it * 32
                    g_res = cute.make_tensor(
                        res128 + (row * h128 + e128), cute.make_layout(1)
                    )
                    pk[0] = g_res[0]
                    rf = pk_bf.load().to(cutlass.Float32)
                    accv = accv + rf * rf
                acc8.store(accv)
                ssq = acc8[0]
                for jj in cutlass.range_constexpr(1, 8):
                    ssq = ssq + acc8[jj]
                for sh in (16, 8, 4, 2, 1):
                    ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, sh)
                rstd = cute.math.rsqrt(ssq * Float32(inv_h) + eps)
                for it in cutlass.range(it_cnt, unroll=4):
                    e128 = lane + it * 32
                    off = row * h128 + e128
                    g_res = cute.make_tensor(
                        res128 + off, cute.make_layout(1)
                    )
                    pk[0] = g_res[0]
                    nv = pk_bf.load().to(cutlass.Float32) * rstd
                    g_g = cute.make_tensor(
                        gamma128 + e128, cute.make_layout(1)
                    )
                    pg[0] = g_g[0]
                    gv = pg_bf.load().to(cutlass.Float32)
                    pk_bf.store((nv * gv).to(cutlass.BFloat16))
                    g_n = cute.make_tensor(
                        norm128 + off, cute.make_layout(1)
                    )
                    g_n[0] = pk[0]

        # ---- exit barrier: last local block releases; everyone waits for
        # all ranks so the next call's stage-in cannot race peers still
        # reading our x / writing our res.
        cute.arch.sync_threads()
        if tidx == 0:
            old = cute.arch.atomic_add(
                (bar.iterator + 1).llvm_ptr, Int32(1),
                sem="release", scope="gpu",
            )
            # fixed grid => cumulative arrivals incl. this call
            if old == Int32(self.n_blocks) * call_id - Int32(1):
                utils.distributed.multimem_red_add1(
                    lock_ptr=bar_mc.iterator, scope="sys", order="release"
                )
            _spin_ge_acquire_sys(
                bar.iterator, Int32(world * 2) * call_id
            )
            # peer multicast writes into our res are visible to
            # post-kernel readers
            cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()


# =====================================================================
# Host wrapper on torch symmetric memory
# =====================================================================


class CuteDslARNorm:
    """Single-kernel ``rmsnorm(allreduce(x) + residual, gamma, eps)``.

    Collective constructor — all ranks of ``group`` must call together
    with identical arguments. :meth:`allreduce_norm` returns
    ``(norm [B, H], new_residual [B, H])`` as VIEWS of internal
    buffers, valid until the next call on this instance. All ranks must
    call in lockstep with identical shapes; ``residual`` must be
    rank-identical (caller contract). CUDA-GRAPH SAFE: device rounds +
    red.max row flags — no host-side counters in the launch args.
    """

    def __init__(self, group, max_m: int, h: int,
                 dtype: torch.dtype = torch.bfloat16, *,
                 n_prod: Optional[int] = None,
                 prod_norm: Optional[bool] = None):
        if dtype != torch.bfloat16:
            raise ValueError("only bf16 supported")
        if h % 256 != 0 or (h // 256) % 4 != 0:
            raise ValueError(f"H={h} must be a multiple of 1024")
        if max_m > _MAX_ROWS:
            raise ValueError(f"max_m={max_m} exceeds {_MAX_ROWS}")
        if torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError("cutedsl_arnorm needs sm100+")
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        if self.world not in (2, 4, 8):
            raise ValueError(f"TP must be 2/4/8, got {self.world}")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.h = h
        self.max_m = max_m
        self.dtype = dtype
        if n_prod is None:
            n_prod = int(os.environ.get("ARNORM_NPROD", "16"))
        if prod_norm is None:
            prod_norm = os.environ.get("ARNORM_PRODNORM", "1") != "0"
        assert 1 <= n_prod < _N_BLOCKS

        gname = self.group.group_name

        def _symm(numel, tdtype, cdtype):
            buf = symm_mem.empty(numel, dtype=tdtype, device=self.device)
            hdl = symm_mem.rendezvous(buf, gname)
            if not hdl.has_multicast_support:
                raise RuntimeError(
                    "NVLS multicast unsupported on this system; the "
                    "fused AR+norm kernel requires it")
            mc = make_ptr(cdtype, hdl.multicast_ptr,
                          cute.AddressSpace.gmem, assumed_align=16)
            return buf, mc

        # symmetric buffers: x staging (ld_reduce source), res
        # (multicast-written new_residual), per-row flags, entry/exit
        # flags ([0] mc counter, [1] local block arrivals)
        self.x_symm, self.x_mc_ptr = _symm(
            max_m * h, dtype, cutlass.BFloat16)
        res_flat, self.res_mc_ptr = _symm(
            max_m * h, dtype, cutlass.BFloat16)
        self.row_flags, self.row_flags_mc_ptr = _symm(
            _MAX_ROWS, torch.int32, cutlass.Int32)
        self.bar, self.bar_mc_ptr = _symm(
            2, torch.int32, cutlass.Int32)
        self.row_flags.zero_()
        self.bar.zero_()
        # per-block device rounds (graph-replay-safe call counter)
        self.rounds = torch.zeros(_N_BLOCKS, dtype=torch.int32,
                                  device=self.device)
        self.norm_buf = torch.empty(max_m * h, dtype=dtype,
                                    device=self.device)
        self.x_view = self.x_symm.view(max_m, h)
        self.res_view = res_flat.view(max_m, h)
        self.norm_view = self.norm_buf.view(max_m, h)

        # fixed cute descriptors
        self._res_c = self._2d(self.res_view)
        self._norm_c = self._2d(self.norm_view)
        rf = from_dlpack(self.row_flags, assumed_align=16)
        rf.element_type = cutlass.Int32
        self._row_flags_c = rf.mark_layout_dynamic(leading_dim=0)
        bc = from_dlpack(self.bar, assumed_align=16)
        bc.element_type = cutlass.Int32
        self._bar_c = bc.mark_layout_dynamic(leading_dim=0)
        rc = from_dlpack(self.rounds, assumed_align=16)
        rc.element_type = cutlass.Int32
        self._rounds_c = rc.mark_layout_dynamic(leading_dim=0)

        self.kernel = _ARNormKernel(
            rank=self.rank, world=self.world, hidden=h,
            n_prod=n_prod, prod_norm=prod_norm)

        self.compiled = None
        self._desc_cache: dict = {}

        # flags (zeroed) and multicast bindings must be visible on
        # every rank before the first launch
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        g_sample = torch.zeros(h, device=self.device, dtype=dtype)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compile_args = (
            self.x_mc_ptr, self.res_mc_ptr,
            self._res_c, self._norm_c,
            self._2d(self.res_view), self._1d(g_sample),
            self._row_flags_c, self.row_flags_mc_ptr,
            self._bar_c, self.bar_mc_ptr,
            self._rounds_c, Float32(1e-5), stream)
        key = (f"arnorm_tp{self.world}_r{self.rank}_m{self.max_m}"
               f"_h{self.h}_p{self.kernel.n_prod}"
               f"_pn{int(self.kernel.prod_norm)}")
        self.compiled = load_or_compile(self.kernel, compile_args, key)

    # ------------------------------------------------------- descriptors

    @staticmethod
    def _2d(t: torch.Tensor):
        d = from_dlpack(t, assumed_align=16)
        d.element_type = cutlass.BFloat16
        return d.mark_layout_dynamic(leading_dim=1)

    @staticmethod
    def _1d(t: torch.Tensor):
        d = from_dlpack(t, assumed_align=16)
        d.element_type = cutlass.BFloat16
        return d.mark_layout_dynamic(leading_dim=0)

    def _descs(self, b: int, residual: torch.Tensor,
               gamma: torch.Tensor):
        key = (b, residual.data_ptr(), gamma.data_ptr())
        hit = self._desc_cache.get(key)
        if hit is None:
            hit = (self._2d(residual), self._1d(gamma))
            self._desc_cache[key] = hit
        return hit

    # ----------------------------------------------------------------- op

    def allreduce_norm(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One kernel launch. Returns ``(norm, new_residual)`` as views
        of internal buffers, valid until the next call."""
        if x.dtype != self.dtype:
            raise TypeError("bf16 input required")
        b, h = x.shape
        if h != self.h:
            raise ValueError(f"H mismatch: {h} != {self.h}")
        if b > self.max_m:
            raise ValueError(f"B={b} exceeds max_m={self.max_m}")
        if residual.shape != (b, h) or gamma.shape != (h,):
            raise ValueError("residual/gamma shape mismatch")
        if not (x.is_contiguous() and residual.is_contiguous()
                and gamma.is_contiguous()):
            raise ValueError("inputs must be contiguous")

        # stage rank-local x into the symmetric buffer (stream-ordered
        # before the kernel; the in-kernel entry flag orders ranks)
        v = self.x_view[:b]
        if x.data_ptr() != v.data_ptr():
            v.copy_(x)

        resid_c, gamma_c = self._descs(b, residual, gamma)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        runtime_args = (
            self.x_mc_ptr, self.res_mc_ptr,
            self._res_c, self._norm_c,
            resid_c, gamma_c,
            self._row_flags_c, self.row_flags_mc_ptr,
            self._bar_c, self.bar_mc_ptr,
            self._rounds_c, Float32(eps), stream)
        self.compiled(*runtime_args)
        return self.norm_view[:b], self.res_view[:b]


# ---------------------------------------------------------------- selftest
# mpirun -n 8 python3 communication/kernel_benchmarks/bench_cutedsl_arnorm.py \
#     [--smoke] [--no-seq]
# (correctness vs NCCL AR + fp32 norm at H=7168 — two calls per size
# plus CUDA-graph capture/replay with refreshed inputs — then a
# CUDA-event bench against the dispatcher's seq baseline re-measured
# in the same run)

def _selftest() -> None:
    import statistics
    import sys

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29567")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    assert world == 8, f"expected TP8, got {world}"
    group = dist.group.WORLD
    h, eps = 7168, 1e-5
    smoke = "--smoke" in sys.argv
    no_seq = "--no-seq" in sys.argv
    check_sizes = (32, 256) if smoke else (32, 256, 1024, 2048,
                                           4096, 8192)
    bench_sizes = (128, 256, 512, 1024, 2048, 4096, 8192)
    # spec table (B200 TP8, re-measured seq below when available)
    spec_seq = {128: 26.8, 256: 31.0, 512: 44.5, 1024: 66.8,
                2048: 119.7, 4096: 210.2, 8192: 396.8}

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    def time_us(fn, reps=5, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group)
        samples = []
        for _ in range(reps):
            s0 = torch.cuda.Event(enable_timing=True)
            s1 = torch.cuda.Event(enable_timing=True)
            s0.record()
            for _ in range(iters):
                fn()
            s1.record()
            s1.synchronize()
            samples.append(s0.elapsed_time(s1) * 1000.0 / iters)
        t = torch.tensor([statistics.median(samples)], device="cuda",
                         dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        return float(t.item())

    op = CuteDslARNorm(group, max_m=8192, h=h)
    gen = torch.Generator(device="cuda")

    def ref_norm(x, residual, gamma):
        ar = x.clone()
        dist.all_reduce(ar, group=group)
        resf = (ar + residual).float()
        rms = resf.pow(2).mean(-1, keepdim=True).add_(eps).rsqrt_()
        return ((resf * rms) * gamma.float()).to(torch.bfloat16), \
            ar + residual

    gamma_ones = torch.ones(h, device="cuda", dtype=torch.bfloat16)
    gamma_rand = (1.0 + 0.02 * torch.randn(
        h, device="cuda")).to(torch.bfloat16)
    dist.broadcast(gamma_rand, src=0, group=group)

    for b in check_sizes:
        gen.manual_seed(1234 + b * 8 + rank)
        x = (0.02 * torch.randn(b, h, generator=gen,
                                device="cuda")).to(torch.bfloat16)
        gen.manual_seed(777 + b)
        residual = (0.02 * torch.randn(b, h, generator=gen,
                                       device="cuda")).to(torch.bfloat16)
        dist.broadcast(residual, src=0, group=group)
        for gname, gamma, tol in (("ones", gamma_ones, 0.1),
                                  ("rand", gamma_rand, 0.35)):
            ref_n, ref_r = ref_norm(x, residual, gamma)
            for _ in range(2):  # 2nd call exercises flag reuse
                norm, res = op.allreduce_norm(x, gamma, eps, residual)
            torch.cuda.synchronize()
            err_n = (norm.float() - ref_n.float()).abs().max().item()
            err_r = (res.float() - ref_r.float()).abs().max().item()
            passed = err_n < tol and err_r < 0.25
            log(f"correctness B={b:<5} gamma={gname} "
                f"max|dnorm|={err_n:.4f} max|dres|={err_r:.4f} "
                f"{'PASS' if passed else 'FAIL'}")
            assert passed, f"B={b} gamma={gname}"

    # ---- CUDA-graph capture / replay with refreshed inputs -----------
    for b in ((256,) if smoke else (256, 4096)):
        xg = torch.zeros(b, h, device="cuda", dtype=torch.bfloat16)
        rg = torch.zeros(b, h, device="cuda", dtype=torch.bfloat16)
        gen.manual_seed(9000 + b + rank)
        xg.copy_((0.02 * torch.randn(b, h, generator=gen,
                                     device="cuda")).to(torch.bfloat16))
        gen.manual_seed(9500 + b)
        rg.copy_((0.02 * torch.randn(b, h, generator=gen,
                                     device="cuda")).to(torch.bfloat16))
        dist.broadcast(rg, src=0, group=group)
        for _ in range(2):  # warmup outside capture
            op.allreduce_norm(xg, gamma_rand, eps, rg)
        torch.cuda.synchronize()
        dist.barrier(group)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            norm_g, res_g = op.allreduce_norm(xg, gamma_rand, eps, rg)
        for trial in range(2):
            gen.manual_seed(9100 + b * 10 + trial * 100 + rank)
            xg.copy_((0.02 * torch.randn(
                b, h, generator=gen, device="cuda")).to(torch.bfloat16))
            ref_n, ref_r = ref_norm(xg, rg, gamma_rand)
            torch.cuda.synchronize()
            dist.barrier(group)
            g.replay()
            torch.cuda.synchronize()
            err_n = (norm_g.float() - ref_n.float()).abs().max().item()
            err_r = (res_g.float() - ref_r.float()).abs().max().item()
            passed = err_n < 0.35 and err_r < 0.25
            log(f"graph-replay B={b:<5} trial={trial} "
                f"max|dnorm|={err_n:.4f} max|dres|={err_r:.4f} "
                f"{'PASS' if passed else 'FAIL'}")
            assert passed, f"graph B={b} trial={trial}"
        # eager call after graph replays must still be correct
        norm, res = op.allreduce_norm(xg, gamma_rand, eps, rg)
        torch.cuda.synchronize()
        ref_n, ref_r = ref_norm(xg, rg, gamma_rand)
        err_n = (norm.float() - ref_n.float()).abs().max().item()
        assert err_n < 0.35, f"post-graph eager B={b}"
        log(f"graph-interop B={b:<5} post-replay eager PASS")

    if smoke:
        log("PASS cutedsl_arnorm selftest (smoke)")
        dist.barrier(group)
        dist.destroy_process_group()
        return

    # ---- bench vs the dispatcher's seq baseline ----------------------
    comm = None
    if not no_seq:
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", ".."))
        from communication.collective import Collectives
        comm = Collectives(group, max_numel=8192 * h,
                           dtype=torch.bfloat16, max_hidden=h,
                           flashinfer_max_tokens=8192)
    log(f"{'B':>6} {'seq_us':>8} {'fused_us':>9} {'ratio':>6}")
    for b in bench_sizes:
        gen.manual_seed(4321 + b * 8 + rank)
        x = (0.02 * torch.randn(b, h, generator=gen,
                                device="cuda")).to(torch.bfloat16)
        gen.manual_seed(888 + b)
        residual = (0.02 * torch.randn(b, h, generator=gen,
                                       device="cuda")).to(torch.bfloat16)
        dist.broadcast(residual, src=0, group=group)
        us = time_us(lambda: op.allreduce_norm(
            x, gamma_rand, eps, residual))
        if comm is not None:
            seq = time_us(lambda: comm.allreduce_norm(
                x, gamma_rand, eps, residual=residual, impl="seq"))
            src = "meas"
        else:
            seq = spec_seq[b]
            src = "spec"
        log(f"{b:>6} {seq:>8.1f} {us:>9.1f} {seq / us:>6.2f}x  "
            f"(seq={src})")
    log("PASS cutedsl_arnorm selftest")
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    _selftest()
