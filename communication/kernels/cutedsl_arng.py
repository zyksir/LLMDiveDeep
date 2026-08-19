"""CuTe DSL single-kernel fused allreduce + RMSNorm + GEMM (ARNG).

Computes, in ONE kernel launch (FLUX-style fusion, M-SPLIT layout):

    res = all_reduce_sum(x) + residual          # x [B, H] rank-local partial
    y   = rmsnorm(res, gamma, eps) @ w.T        # w [N, H], K of the GEMM = H

Sibling of ``cutedsl_gemm_ar.py`` (epilogue-fused GEMM+AR) and reuses all
its plumbing: torch symm-mem allocation + rendezvous, NVLS multicast VAs
passed as raw ``cute.Pointer``, persistent tcgen05 GEMM structure, and the
per-SM sys-scope exit protocol.

M-SPLIT design (v3): the [B, N] GEMM output is identical on every rank
(same normed A, same w), so computing it redundantly on all ranks wastes
7/8 of the FLOPs at TP8. Instead, 128-row M-tiles are owned round-robin
by rank (tile t -> rank t % world) and each rank:

  * PROLOGUE warps (6-9), own rows only, one warp per row, fused pass:
    ``multimem.ld_reduce`` the staged symmetric x (all ranks reduced by
    the NVLink switch), add the (rank-identical) residual, accumulate the
    row sum-of-squares inline, ``multimem.st``-broadcast the res row to
    every rank's new-residual buffer (plus a plain local copy for the
    same-thread re-read), then normalize * gamma into the LOCAL GEMM A
    buffer and release a LOCAL gpu-scope per-row ``norm_flags`` entry.
  * GEMM warps process ONLY the rank's own M-tiles (the persistent
    scheduler enumerates all tiles; foreign tiles are skipped by every
    role identically). The TMA warp spin-waits (``ld.acquire.gpu``) on
    the local norm flags of its tile's rows, so early GEMM tiles overlap
    later prologue rows.
  * EPILOGUE warps TMA-store each finished y tile into the local
    symmetric C, wait for the store to land, and ``multimem.st`` the
    tile to every rank's C copy (broadcast, not reduce - each tile has
    exactly one producer).
  * Entry protocol: rank-local x is staged into the symmetric x buffer
    by a plain D2D copy just before launch (stream-ordered); block 0 of
    each rank then ``multimem.red.add`` +1 a symmetric entry flag and
    prologue warps wait until it reaches ``world * call_id`` (monotonic,
    never reset) before any ``ld_reduce``.
  * Exit protocol: identical per-SM sys-scope release+spin as the
    sibling. Local kernel completion therefore implies that EVERY rank
    finished its multimem traffic into this rank's buffers (res rows and
    y tiles from peers, peers' reads of our x), so single (un-phased)
    buffers are safe and no mid-kernel cross-rank flags are needed for
    the outputs: the next call's staging copy is stream-ordered after
    kernel completion.

Norm flags are local (producer and consumer are the same GPU) and
monotonic: the producer stores the call counter, consumers wait for
equality - no zeroing between calls, B may vary per call.

ALIASING contract: :meth:`CuteDslARNG.allreduce_norm_gemm` returns VIEWS
of internal buffers (y and new_residual), valid until the next call on
this instance. All ranks must call in lockstep with identical shapes.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cute.typing import Int32, Float32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .cutedsl_cache import load_or_compile


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    c_smem_layout: cute.Layout,
) -> Tuple[int, int, int]:
    """A/B/C stage heuristics (same as the sibling GEMM+AR kernel)."""
    num_acc_stage = 2
    num_c_stage = 2

    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    num_c_stage += (
        smem_capacity
        - occupancy * ab_bytes_per_stage * num_ab_stage
        - occupancy * (mbar_helpers_bytes + c_bytes)
    ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


@cute.jit
def _spin_wait_eq_relaxed_gpu(flag_ptr: cute.Pointer, expected: Int32):
    """Non-destructive spin until ``*flag_ptr == expected`` (gpu relaxed).

    Safe for many concurrent waiters: plain loads, no CAS reset. The flag
    is monotonic (the producer release-stores the per-call counter); the
    caller issues an acquire fence after all its spins complete."""
    result = Int32(0)
    while result != expected:
        result = cute.arch.load(
            flag_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu"
        )


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


class PersistentARNGKernel:
    """Blackwell persistent M-split GEMM with fused AR+RMSNorm prologue.

    Warp roles (one 320-thread CTA, occupancy 1):
      0-3  epilogue: TMEM -> rmem -> smem -> TMA-store y tile (local C),
           then multimem.st-broadcast the tile to every rank
      4    MMA: tcgen05 mainloop into TMEM (own M-tiles only)
      5    TMA: per-tile spin on local norm flags, then A/B loads
      6-9  prologue: multimem.ld_reduce(x) + residual + RMSNorm on OWN
           rows -> res (multicast) + normed (local GEMM A), release the
           local per-row norm flag
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        rank_id: int,
        world_size: int,
        hidden: int,
        skip_prologue: bool = False,
        skip_gemm: bool = False,
    ):
        # debug-only constexpr switches for phase isolation benchmarks
        self.skip_prologue = skip_prologue
        self.skip_gemm = skip_gemm
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.prologue_warp_id = (6, 7, 8, 9)
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
                *self.prologue_warp_id,
            )
        )
        self.epilogue_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.rank_id = rank_id
        self.num_ranks = world_size
        # constexpr shape facts baked into the prologue codegen
        self.hidden = hidden          # H (= K of the GEMM)
        if hidden % 256 != 0:
            raise ValueError(f"H={hidden} must be a multiple of 256")
        it_cnt = hidden // 256
        self.p1_batch = next(
            c for c in (7, 4, 2, 1) if it_cnt % c == 0)
        self.p1_nblk = it_cnt // self.p1_batch

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        c_smem_layout = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            c_smem_layout,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        c_mc_ptr: cute.Pointer,
        x_mc_ptr: cute.Pointer,
        res_mc_ptr: cute.Pointer,
        res_in: cute.Tensor,
        res_out: cute.Tensor,
        gamma: cute.Tensor,
        norm_flags: cute.Tensor,
        symm_flags: cute.Tensor,
        symm_flags_mc_ptr: cute.Pointer,
        rounds: cute.Tensor,
        eps: Float32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the fused prologue+GEMM kernel.

        ``a`` is the LOCAL normed-activation buffer [M_pad, H, 1]: written
        by this rank's prologue warps for its own M-tiles, read (via TMA)
        as the GEMM A operand. ``c`` is the SYMMETRIC y buffer
        [M_pad, N_pad, 1]; own tiles are TMA-stored locally then
        multimem.st-broadcast. ``*_mc_ptr`` are the NVLS multicast VAs of
        the corresponding symmetric buffers.
        """
        # MC view of the symm flags (entry flag + per-SM exit flags)
        symm_flags_mc = cute.make_tensor(symm_flags_mc_ptr, symm_flags.layout)
        # MC view of the symmetric C (same layout as the local view)
        c_mc = cute.make_tensor(c_mc_ptr, c.layout)
        # MC views of staged x / res, [M_pad, H] row-major
        rows_layout = cute.make_layout(
            (cute.size(a, mode=[0]), cute.size(a, mode=[1])),
            stride=(cute.size(a, mode=[1]), 1),
        )
        x_mc = cute.make_tensor(x_mc_ptr, rows_layout)
        res_mc = cute.make_tensor(res_mc_ptr, rows_layout)

        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            a,
            c,
            c_mc,
            x_mc,
            res_mc,
            res_in,
            res_out,
            gamma,
            norm_flags,
            symm_flags,
            symm_flags_mc,
            rounds,
            eps,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mA_plain: cute.Tensor,
        mC_plain: cute.Tensor,
        mC_mc: cute.Tensor,
        x_mc: cute.Tensor,
        res_mc: cute.Tensor,
        mResIn: cute.Tensor,
        mResOut: cute.Tensor,
        mGamma: cute.Tensor,
        norm_flags: cute.Tensor,
        symm_flags: cute.Tensor,
        symm_flags_mc: cute.Tensor,
        rounds: cute.Tensor,
        eps: Float32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        bidx, bidy, bidz = cute.arch.block_idx()
        gdimx, gdimy, gdimz = cute.arch.grid_dim()
        sm_linear = cute.arch.make_warp_uniform(
            bidx + bidy * gdimx + bidz * gdimx * gdimy
        )
        num_ctas = gdimx * gdimy * gdimz
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init mbarriers
        #
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_call = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Int32, 1), cute.make_layout(1)
        )
        if tidx == 0:
            current_round = rounds[sm_linear] + Int32(1)
            rounds[sm_linear] = current_round
            s_call[0] = current_round
        cute.arch.sync_threads()
        call_id = s_call[0]

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensors A/B
        #
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gClc_mnl = cute.local_tile(
            mC_plain, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gCmc_mnl = cute.local_tile(
            mC_mc, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgC_lc = thr_mma.partition_C(gClc_mnl)
        tCgC_mc = thr_mma.partition_C(gCmc_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # debug phase-isolation gates (constexpr; -1/999 disable a role)
        tma_role = self.tma_warp_id if not self.skip_gemm else -1
        mma_role = self.mma_warp_id if not self.skip_gemm else -1
        epi_gate = self.mma_warp_id if not self.skip_gemm else -1
        prologue_gate = (self.prologue_warp_id[0]
                         if not self.skip_prologue else 999)

        #
        # Specialized TMA load warp: own tiles only, gated on local
        # per-row norm flags
        #
        if warp_idx == tma_role:
            lane_tma = cute.arch.lane_idx()
            b_rows_tma = cute.size(mResIn, mode=[0])
            tile_m = self.mma_tiler_mn[0]
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    # raster-swapped scheduler: coord[0]=n, coord[1]=m
                    cur_tile_coord[1] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[0],
                    cur_tile_coord[2],
                )
                is_mine = (
                    mma_tile_coord_mnl[0] % self.num_ranks == self.rank_id
                )

                if is_mine:
                    # Wait for every valid row of this tile: 32 lanes spin
                    # on 4 monotonic per-row LOCAL flags each (rows >= B
                    # are never produced and never consumed). Flags are
                    # release-stored by this rank's own prologue warps.
                    if cutlass.const_expr(not self.skip_prologue):
                        r_base = mma_tile_coord_mnl[0] * tile_m
                        for j in cutlass.range_constexpr(tile_m // 32):
                            r = r_base + lane_tma + j * 32
                            if r < b_rows_tma:
                                _spin_wait_eq_relaxed_gpu(
                                    norm_flags.iterator + r, call_id
                                )
                        cute.arch.sync_warp()
                        cute.arch.fence_acq_rel_gpu()
                        # generic->async proxy: the TMA engine must
                        # observe the prologue's normed stores
                        cute.arch.fence_proxy("async.global")

                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)

                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=b_full_mcast_mask,
                        )

                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        #
        # Specialized MMA warp (own tiles only)
        #
        if warp_idx == mma_role:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                m_tile_mma = (
                    # raster-swapped scheduler: coord[0]=n, coord[1]=m
                    cur_tile_coord[1] // cute.size(tiled_mma.thr_id.shape)
                )
                is_mine = m_tile_mma % self.num_ranks == self.rank_id

                if is_mine:
                    tCtAcc = tCtAcc_base[
                        (None, None, None, acc_producer_state.index)]

                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if is_leader_cta:
                        peek_ab_full_status = ab_consumer.try_wait()

                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in range(k_tile_cnt):
                        if is_leader_cta:
                            handle = ab_consumer.wait_and_advance(
                                peek_ab_full_status)

                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kblk_idx in cutlass.range(
                                    num_kblocks, unroll_full=True):
                                kblk_crd = (None, None, kblk_idx, handle.index)

                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblk_crd],
                                    tCrB[kblk_crd],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            handle.release()

                            peek_ab_full_status = cutlass.Boolean(1)
                            if handle.count + 1 < k_tile_cnt:
                                peek_ab_full_status = ab_consumer.try_wait()

                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        #
        # Specialized epilogue warps: TMA store of own y tiles into the
        # local symmetric C, then multimem.st-broadcast each tile
        #
        if warp_idx < epi_gate:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            self.epilogue_tma_store_broadcast(
                tidx,
                warp_idx,
                acc_pipeline,
                tiled_mma,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                tCgC_lc,
                tCgC_mc,
                epi_tile,
                tile_sched,
            )

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////
        #  Prologue warps: AR (multimem.ld_reduce) + residual + RMSNorm
        #  over this rank's OWN rows only (M-split tile ownership)
        # ///////////////////////////////////////////////////////////////////
        if warp_idx >= prologue_gate:
            lane = cute.arch.lane_idx()
            pw = warp_idx - self.prologue_warp_id[0]
            h = self.hidden                       # constexpr
            h128 = h // 8                         # Int128 elems per row
            it_cnt = h // 256                     # 8 bf16 per lane per iter
            num_pwarps = len(self.prologue_warp_id)
            gwarp = sm_linear * num_pwarps + pw   # global prologue warp id
            total_warps = num_ctas * num_pwarps
            b_rows = cute.size(mResIn, mode=[0])
            tile_m = self.mma_tiler_mn[0]         # constexpr (128)
            world = self.num_ranks                # constexpr

            # own-row index space: i in [0, own_span) maps to
            # row = (i//tile_m * world + rank) * tile_m + i%tile_m
            m_tiles = (b_rows + tile_m - 1) // tile_m
            own_span = (
                (m_tiles - Int32(self.rank_id) + Int32(world - 1))
                // world
            ) * tile_m

            # Entry signal: this rank's staged x is visible (stream order);
            # +1 on every rank's entry flag. Monotonic, never reset.
            if warp_idx == self.prologue_warp_id[0] and sm_linear == 0:
                with cute.arch.elect_one():
                    utils.distributed.multimem_red_add1(
                        lock_ptr=symm_flags_mc.iterator,
                        scope="sys",
                        order="release",
                    )

            if gwarp < own_span:
                # This warp reduces >= 1 row: wait for ALL ranks' x
                # (one spinning lane, then a warp-wide acquire fence)
                with cute.arch.elect_one():
                    utils.distributed.spin_lock_ld_lt_relaxed_wait(
                        lock_ptr=symm_flags.iterator,
                        expected_val=Int32(self.num_ranks) * call_id,
                        scope="sys",
                    )
                cute.arch.sync_warp()
                cute.arch.fence_acq_rel_sys()

            # per-thread scratch: 4x i32 (ld_reduce regs) viewed as 8x bf16,
            # one Int128 pack viewed as 8x bf16, fp32 accumulator vector
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
            # one 16B pack per in-flight batch element, so a whole group
            # of multimem.st ops can issue back-to-back (volatile asm
            # cannot be reordered, so interleaving one st per element
            # serializes every surrounding memory op)
            pk_l, pk_bf_l, pk32_l = [], [], []
            for _u in cutlass.range_constexpr(self.p1_batch):
                t = cute.make_rmem_tensor(
                    cute.make_layout(1), cutlass.Int128)
                pk_l.append(t)
                pk_bf_l.append(cute.make_tensor(
                    cute.recast_ptr(t.iterator, dtype=cutlass.BFloat16),
                    cute.make_layout(8)))
                pk32_l.append(cute.make_tensor(
                    cute.recast_ptr(t.iterator, dtype=cutlass.Int32),
                    cute.make_layout(4)))
            acc8 = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)

            # 16B (Int128) base pointers for vectorized local traffic
            res_in128 = cute.recast_ptr(mResIn.iterator, dtype=cutlass.Int128)
            res_out128 = cute.recast_ptr(mResOut.iterator, dtype=cutlass.Int128)
            gamma128 = cute.recast_ptr(mGamma.iterator, dtype=cutlass.Int128)
            norm128 = cute.recast_ptr(mA_plain.iterator, dtype=cutlass.Int128)
            x_mc_base = x_mc.iterator
            res_mc_base = res_mc.iterator

            # ld_reduce batching: the volatile multimem asm ops cannot be
            # reordered against memory ops, so interleaving one load with
            # its consumption serializes every NVLS access. Issue BATCH
            # loads back-to-back (nothing but address arithmetic between
            # them), and consume batch k while batch k+1 is in flight.
            batch = self.p1_batch     # constexpr
            n_blk = self.p1_nblk      # constexpr

            # ---- fused per-row pass over OWN rows: one warp per row.
            # The 32 lanes cover the whole H row, so the owning warp can
            # do everything for its row in one pass over NVLink:
            #   reduce x -> +residual -> ssq -> rstd -> norm*gamma
            # broadcasting only res (the normed row is written LOCALLY:
            # only this rank's GEMM reads it under the M-split).
            for i in cutlass.range(gwarp, own_span, total_warps, unroll=1):
                kk = i // tile_m
                jj = i % tile_m
                row = (kk * world + Int32(self.rank_id)) * tile_m + jj
                if row < b_rows:
                    acc8.fill(0.0)
                    accv = acc8.load()
                    regs = []
                    for u in cutlass.range_constexpr(batch):
                        e128 = lane + u * 32
                        regs.append(
                            utils.distributed.multimem_ld_reduce_8xbf16(
                                x_mc_base + (row * h + e128 * 8)
                            )
                        )
                    for blk in cutlass.range_constexpr(n_blk):
                        nxt = []
                        if cutlass.const_expr(blk + 1 < n_blk):
                            for u in cutlass.range_constexpr(batch):
                                e128 = lane + ((blk + 1) * batch + u) * 32
                                nxt.append(
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
                                res_in128 + off, cute.make_layout(1)
                            )
                            pk_l[u][0] = g_in[0]
                            rv = pk_bf_l[u].load().to(cutlass.Float32)
                            pk_bf_l[u].store((arv + rv).to(cutlass.BFloat16))
                            # ssq from the bf16-rounded res (matches ref)
                            # and a plain local copy so pass 2 re-reads it
                            # from L2 with same-thread visibility
                            rf = pk_bf_l[u].load().to(cutlass.Float32)
                            accv = accv + rf * rf
                            g_res = cute.make_tensor(
                                res_out128 + off, cute.make_layout(1)
                            )
                            g_res[0] = pk_l[u][0]
                        # broadcast res to every rank's copy, sts issued
                        # back-to-back
                        for u in cutlass.range_constexpr(batch):
                            e128 = lane + (blk * batch + u) * 32
                            _multimem_st_4xb32(
                                res_mc_base + (row * h + e128 * 8),
                                pk32_l[u][0], pk32_l[u][1],
                                pk32_l[u][2], pk32_l[u][3],
                            )
                        regs = nxt
                    # row sum-of-squares -> rstd (8 vector lanes + 32 thr)
                    acc8.store(accv)
                    ssq = acc8[0]
                    for j in cutlass.range_constexpr(1, 8):
                        ssq = ssq + acc8[j]
                    for sh in (16, 8, 4, 2, 1):
                        ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, sh)
                    rstd = cute.math.rsqrt(ssq / Float32(h) + eps)
                    # pass 2 (local L2 hits): normalize, apply gamma, and
                    # write the normed row into the LOCAL GEMM A buffer
                    for it in cutlass.range(it_cnt, unroll=7):
                        e128 = lane + it * 32
                        off = row * h128 + e128
                        g_res = cute.make_tensor(
                            res_out128 + off, cute.make_layout(1)
                        )
                        pk[0] = g_res[0]
                        nv = pk_bf.load().to(cutlass.Float32) * rstd
                        nbf = nv.to(cutlass.BFloat16).to(cutlass.Float32)
                        g_g = cute.make_tensor(
                            gamma128 + e128, cute.make_layout(1)
                        )
                        pk_l[0][0] = g_g[0]
                        gv = pk_bf_l[0].load().to(cutlass.Float32)
                        pk_bf.store((nbf * gv).to(cutlass.BFloat16))
                        g_n = cute.make_tensor(
                            norm128 + off, cute.make_layout(1)
                        )
                        g_n[0] = pk[0]
                    # generic->async proxy visibility for the TMA loads,
                    # warp memory sync, then one lane releases the LOCAL
                    # norm flag
                    cute.arch.fence_proxy("async.global")
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.store(
                            norm_flags.iterator + row,
                            call_id,
                            sem="release",
                            scope="gpu",
                        )

        #
        # Exit protocol (all warps): per-SM sys-scope release + spin so no
        # rank leaves while peers may still multimem.ld_reduce its x or
        # multimem.st into its res / y buffers, and so the next call's
        # staging copy (stream-ordered after this kernel) cannot race any
        # peer's reads.
        #
        cute.arch.sync_threads()
        if warp_idx == self.prologue_warp_id[0]:
            with cute.arch.elect_one():
                utils.distributed.multimem_red_add1(
                    lock_ptr=symm_flags_mc.iterator + 1 + sm_linear,
                    scope="sys",
                    order="release",
                )
                utils.distributed.spin_lock_atom_cas_relaxed_wait(
                    lock_ptr=symm_flags.iterator + 1 + sm_linear,
                    expected_val=Int32(self.num_ranks),
                    reset_val=0,
                    scope="sys",
                )
                # acquire: peer multicast writes into our res/y copies
                # are visible to post-kernel readers
                cute.arch.fence_acq_rel_sys()

    @cute.jit
    def epilogue_tma_store_broadcast(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        tCgC_lc: cute.Tensor,
        tCgC_mc: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
    ) -> None:
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilogue_tmem_copy_and_partition(
            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilogue_smem_copy_and_partition(
            tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilogue_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )

        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilogue_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        # 128-bit broadcast partition of a full (CTA_M, TILE_N) y tile
        # across the 128 epilogue threads (same tv pattern as the sibling
        # kernel's AR warps)
        atom_val = 128 // self.c_dtype.width
        thr_n = self.mma_tiler[1] // atom_val
        thr_m = (32 * len(self.epilogue_warp_id)) // thr_n
        bc_thr_layout = cute.make_layout(
            (thr_m, thr_n), stride=(thr_n, 1)
        )
        bc_val_layout = cute.make_layout((1, atom_val), stride=(atom_val, 1))
        copy_atom_bc = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.c_dtype
        )
        tiled_copy_bc = cute.make_tiled_copy_tv(
            copy_atom_bc, bc_thr_layout, bc_val_layout
        )
        thr_copy_bc = tiled_copy_bc.get_slice(epi_tidx)
        pkb = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int128)
        pkb32 = cute.make_tensor(
            cute.recast_ptr(pkb.iterator, dtype=cutlass.Int32),
            cute.make_layout(4),
        )

        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                # raster-swapped scheduler: coord[0]=n, coord[1]=m
                cur_tile_coord[1] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[0],
                cur_tile_coord[2],
            )
            is_mine = mma_tile_coord_mnl[0] % self.num_ranks == self.rank_id

            if is_mine:
                bSG_gC = bSG_gC_partitioned[
                    (None, None, None, *mma_tile_coord_mnl)]

                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    c_buffer = (
                        num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s, tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilogue_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    epilogue_sync_barrier.arrive_and_wait()

                epilogue_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # drain this tile's TMA stores, then broadcast the tile
                # from the local symmetric C into every rank's copy
                c_pipeline.producer_tail()
                epilogue_sync_barrier.arrive_and_wait()

                tile_lc = tCgC_lc[((None, None), 0, 0, *mma_tile_coord_mnl)]
                tile_mc = tCgC_mc[((None, None), 0, 0, *mma_tile_coord_mnl)]
                frg_lc = thr_copy_bc.partition_S(tile_lc)
                frg_mc = thr_copy_bc.partition_S(tile_mc)
                for bi in cutlass.range_constexpr(
                        cute.size(frg_lc, mode=[1])):
                    for bj in cutlass.range_constexpr(
                            cute.size(frg_lc, mode=[2])):
                        src128 = cute.recast_ptr(
                            frg_lc[(None, bi, bj)].iterator,
                            dtype=cutlass.Int128,
                        )
                        g_src = cute.make_tensor(src128, cute.make_layout(1))
                        pkb[0] = g_src[0]
                        _multimem_st_4xb32(
                            frg_mc[(None, bi, bj)].iterator,
                            pkb32[0], pkb32[1], pkb32[2], pkb32[3],
                        )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        c_pipeline.producer_tail()

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilogue_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        # RASTER SWAP: enumerate N fastest so the first wave of CTAs
        # covers ALL n-tiles of the LOWEST m-tiles — GEMM starts as soon
        # as the first prologue rows land instead of the last.
        num_ctas_nml = (num_ctas_mnl[1], num_ctas_mnl[0], num_ctas_mnl[2])
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_nml, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# =====================================================================
# Host wrapper on torch symmetric memory
# =====================================================================

_TILE = 128    # mma tiler M (M-split ownership granularity), 1-CTA
_TILE_N = 256  # mma tiler N


def _pad(v: int, q: int = _TILE) -> int:
    return int(math.ceil(v / q) * q)


class CuteDslARNG:
    """Single-kernel ``rmsnorm(allreduce(x) + residual, gamma, eps) @ w.T``.

    Collective constructor — all ranks of ``group`` must call together
    with identical arguments. ``allreduce_norm_gemm`` returns
    ``(y [B, N], new_residual [B, H])`` as VIEWS of internal buffers,
    valid until the next call on this instance. All ranks must call in
    lockstep with identical shapes (caller contract).

    N is padded internally to a multiple of 256 (zero-padded w rows);
    the kernel compiles eagerly in the constructor.
    """

    def __init__(self, group, max_m: int, h: int, n: int,
                 dtype: torch.dtype = torch.bfloat16):
        if dtype != torch.bfloat16:
            raise ValueError("only bf16 supported")
        if h % 256 != 0:
            raise ValueError(f"H={h} must be a multiple of 256")
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        if self.world not in (2, 4, 8):
            raise ValueError(f"TP must be 2/4/8, got {self.world}")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.h = h
        self.n = n
        self.n_pad = _pad(n, _TILE_N)
        self.max_m = _pad(max_m)
        self.dtype = dtype

        dev_props = torch.cuda.get_device_properties(self.device)
        num_sms = dev_props.multi_processor_count

        gname = self.group.group_name

        def _symm(numel, tdtype, cdtype):
            buf = symm_mem.empty(numel, dtype=tdtype, device=self.device)
            hdl = symm_mem.rendezvous(buf, gname)
            if not hdl.has_multicast_support:
                raise RuntimeError(
                    "NVLS multicast unsupported on this system; the "
                    "fused ARNG kernel requires it")
            mc = make_ptr(cdtype, hdl.multicast_ptr,
                          cute.AddressSpace.gmem, assumed_align=16)
            return buf, mc

        # symmetric buffers (single-phase: the exit protocol makes the
        # next call's staging copy / flag reuse safe without phases):
        #   x staging, new_residual (multicast-written by row owners),
        #   y output (multicast-written by tile owners),
        #   entry/exit flags ([0] entry monotonic, [1:1+num_sms] exit)
        self.x_symm, self.x_mc_ptr = _symm(
            self.max_m * h, dtype, cutlass.BFloat16)
        res_flat, self.res_mc_ptr = _symm(
            self.max_m * h, dtype, cutlass.BFloat16)
        c_flat, self.c_mc_ptr = _symm(
            self.max_m * self.n_pad, dtype, cutlass.BFloat16)
        self.symm_flags, self.symm_flags_mc_ptr = _symm(
            1 + num_sms, torch.int32, cutlass.Int32)
        self.symm_flags.zero_()
        # local per-row flags: released by the local prologue pass,
        # acquired by the TMA warp (gpu scope; monotonic with call_id)
        self.norm_flags = torch.zeros(
            self.max_m, dtype=torch.int32, device=self.device)
        self.rounds = torch.zeros(
            num_sms, dtype=torch.int32, device=self.device)
        self.x_view = self.x_symm.view(self.max_m, h)
        self.res_buf = res_flat.view(self.max_m, h)
        self.c_buf = c_flat.view(self.max_m, self.n_pad, 1)

        # local buffers: normed activations (GEMM A, written by this
        # rank's prologue for its own rows only) and the padded weight
        self.norm_buf = torch.zeros(
            (self.max_m, h, 1), device=self.device, dtype=dtype)
        self.w_buf = torch.zeros(
            (self.n_pad, h, 1), device=self.device, dtype=dtype)

        sf = from_dlpack(self.symm_flags, assumed_align=16)
        sf.element_type = cutlass.Int32
        self.symm_flags_cute = sf.mark_layout_dynamic(leading_dim=0)
        nf = from_dlpack(self.norm_flags, assumed_align=16)
        nf.element_type = cutlass.Int32
        self.norm_flags_cute = nf.mark_layout_dynamic(leading_dim=0)
        rounds = from_dlpack(self.rounds, assumed_align=16)
        rounds.element_type = cutlass.Int32
        self.rounds_cute = rounds.mark_layout_dynamic(leading_dim=0)
        wb = from_dlpack(self.w_buf, assumed_align=16)
        wb.element_type = cutlass.BFloat16
        self.w_cute = wb.mark_layout_dynamic(leading_dim=1)

        self.kernel = PersistentARNGKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=(_TILE, _TILE_N),
            cluster_shape_mn=(1, 1),
            rank_id=self.rank,
            world_size=self.world,
            hidden=h,
            skip_prologue=os.environ.get("ARNG_SKIP_PROLOGUE") == "1",
            skip_gemm=os.environ.get("ARNG_SKIP_GEMM") == "1",
        )
        self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)

        self._w_key: Optional[tuple] = None
        self._desc_cache: dict = {}

        self.compiled = None

        # flags (zeroed) and multicast bindings must be visible on every
        # rank before the first launch
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        g_sample = torch.zeros(h, device=self.device, dtype=dtype)
        a_c, c_c = self._ac_descs(self.max_m)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compile_args = (
            a_c, self.w_cute, c_c,
            self.c_mc_ptr, self.x_mc_ptr, self.res_mc_ptr,
            self._2d_desc(self.res_buf),
            self._2d_desc(self.res_buf),
            self._1d_desc(g_sample),
            self.norm_flags_cute,
            self.symm_flags_cute, self.symm_flags_mc_ptr,
            self.rounds_cute, Float32(1e-5),
            self.max_active_clusters, stream)
        key = (f"arng_tp{self.world}_r{self.rank}_m{self.max_m}"
               f"_h{self.h}_n{self.n_pad}")
        self.compiled = load_or_compile(self.kernel, compile_args, key)

    # ------------------------------------------------------- descriptors

    @staticmethod
    def _2d_desc(t: torch.Tensor):
        d = from_dlpack(t, assumed_align=16)
        d.element_type = cutlass.BFloat16
        return d.mark_layout_dynamic(leading_dim=1)

    @staticmethod
    def _1d_desc(t: torch.Tensor):
        d = from_dlpack(t, assumed_align=16)
        d.element_type = cutlass.BFloat16
        return d.mark_layout_dynamic(leading_dim=0)

    def _ac_descs(self, m_pad: int):
        a = from_dlpack(self.norm_buf[:m_pad], assumed_align=16)
        a.element_type = cutlass.BFloat16
        a_c = a.mark_layout_dynamic(leading_dim=1)
        c = from_dlpack(self.c_buf[:m_pad], assumed_align=16)
        c.element_type = cutlass.BFloat16
        c_c = c.mark_layout_dynamic(leading_dim=1)
        return a_c, c_c

    def _descs(self, m_pad: int, b: int, residual: torch.Tensor,
               gamma: torch.Tensor):
        key = (m_pad, b, residual.data_ptr(), gamma.data_ptr())
        hit = self._desc_cache.get(key)
        if hit is not None:
            return hit
        a_c, c_c = self._ac_descs(m_pad)
        entry = (
            a_c, c_c,
            self._2d_desc(residual),
            self._2d_desc(self.res_buf[:b]),
            self._1d_desc(gamma),
        )
        self._desc_cache[key] = entry
        return entry

    # ----------------------------------------------------------------- op

    def allreduce_norm_gemm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One kernel launch. Returns ``(y [B, N], new_residual [B, H])``
        as views of internal buffers, valid until the next call."""
        if x.dtype != self.dtype or w.dtype != self.dtype:
            raise TypeError("bf16 inputs required")
        if not (x.is_contiguous() and residual.is_contiguous()
                and gamma.is_contiguous() and w.is_contiguous()):
            raise ValueError("inputs must be contiguous")
        b, h = x.shape
        n, h_w = w.shape
        if h != self.h or h_w != self.h or n != self.n:
            raise ValueError(
                f"shape mismatch: x{tuple(x.shape)} w{tuple(w.shape)} "
                f"vs (H={self.h}, N={self.n})")
        if residual.shape != (b, h) or gamma.shape != (h,):
            raise ValueError("residual/gamma shape mismatch")
        if b > self.max_m:
            raise ValueError(f"B={b} exceeds max_m={self.max_m}")

        m_pad = _pad(b)

        # stage w once per weight pointer (zero-padded rows stay zero)
        w_key = (w.data_ptr(), tuple(w.shape))
        if torch.cuda.is_current_stream_capturing() or w_key != self._w_key:
            self.w_buf[:n, :, 0].copy_(w)
            self._w_key = w_key
        # stage rank-local x into the symmetric buffer (stream-ordered
        # before the kernel; the in-kernel entry flag orders ranks)
        self.x_view[:b].copy_(x)

        a_c, c_c, ri_c, ro_c, g_c = self._descs(m_pad, b, residual, gamma)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        runtime_args = (
            a_c, self.w_cute, c_c,
            self.c_mc_ptr, self.x_mc_ptr, self.res_mc_ptr,
            ri_c, ro_c, g_c,
            self.norm_flags_cute,
            self.symm_flags_cute, self.symm_flags_mc_ptr,
            self.rounds_cute, Float32(eps), stream)
        self.compiled(*runtime_args)
        return self.c_buf[:b, : self.n, 0], self.res_buf[:b]


# ---------------------------------------------------------------- selftest
# mpirun -n 8 python3 communication/kernel_benchmarks/bench_cutedsl_arng.py
# (correctness of rmsnorm(AR(x)+res, gamma) @ w.T and new_residual vs an
# NCCL + fp32-norm reference at H=7168, N=6288, then a CUDA-event bench
# against the two-kernel dispatcher numbers)

def _selftest() -> None:
    import statistics
    import sys

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29563")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    assert world == 8, f"expected TP8, got {world}"
    group = dist.group.WORLD
    h, n_dim, eps = 7168, 6288, 1e-5
    smoke = "--smoke" in sys.argv
    batches = (32, 256) if smoke else (32, 256, 1024, 2048, 4096, 8192)
    seq_us = {32: 33.5, 256: 54.9, 1024: 121.0, 2048: 228.4,
              4096: 453.3, 8192: 897.5}  # B200 TP8 dispatcher reference

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    def time_us(fn, reps=5, iters=10):
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

    op = CuteDslARNG(group, max_m=max(batches), h=h, n=n_dim,
                     dtype=torch.bfloat16)
    gen = torch.Generator(device="cuda")
    w = (0.02 * torch.randn(n_dim, h, device="cuda")).to(torch.bfloat16)
    gamma = (1.0 + 0.02 * torch.randn(h, device="cuda")).to(torch.bfloat16)
    dist.broadcast(w, src=0)
    dist.broadcast(gamma, src=0)
    for b in batches:
        gen.manual_seed(1234 + b * 8 + rank)
        x = (0.02 * torch.randn(b, h, generator=gen,
                                device="cuda")).to(torch.bfloat16)
        gen.manual_seed(777 + b)
        residual = (0.02 * torch.randn(b, h, generator=gen,
                                       device="cuda")).to(torch.bfloat16)
        dist.broadcast(residual, src=0)
        ar = x.clone()
        dist.all_reduce(ar)
        resf = (ar + residual).float()
        rms = resf.pow(2).mean(-1, keepdim=True).add_(eps).rsqrt_()
        ref_y = ((resf * rms).to(torch.bfloat16) * gamma) @ w.T
        ref_res = ar + residual
        for _ in range(2):  # 2nd call exercises the flag/slot reuse
            y, res = op.allreduce_norm_gemm(x, residual, gamma, w, eps=eps)
        torch.cuda.synchronize()
        err_y = (y.float() - ref_y.float()).abs().max().item()
        err_r = (res.float() - ref_res.float()).abs().max().item()
        log(f"correctness B={b:<5} max|dy|={err_y:.4f} "
            f"max|dres|={err_r:.4f} "
            f"{'PASS' if err_y < 0.35 and err_r < 0.25 else 'FAIL'}")
        assert err_y < 0.35 and err_r < 0.25, f"B={b}"
        if not smoke:
            us = time_us(lambda: op.allreduce_norm_gemm(
                x, residual, gamma, w, eps=eps))
            base = seq_us[b]
            log(f"bench       B={b:<5} fused={us:7.1f}us seq="
                f"{base:6.1f}us  {base / us:4.2f}x")
    log("PASS cutedsl_arng selftest" + (" (smoke)" if smoke else ""))
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    _selftest()
