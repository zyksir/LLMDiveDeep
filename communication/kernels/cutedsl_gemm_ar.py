"""CuTe DSL tile-fused GEMM + AllReduce on torch symmetric memory.

Port of /workspace/CuTeDSLGen/generated/blackwell_gemm_all_reduce (prefill
path: persistent tcgen05 GEMM whose epilogue TMA-stores partial tiles into
symmetric C, then AR warps do ``multimem.ld_reduce`` + ``multimem.st`` per
tile) with the NVSHMEM allocator/bootstrap replaced by
``torch.distributed._symmetric_memory``:

  * C and the tile-barrier flags live in ``symm_mem.empty`` buffers
    (rendezvous over the caller's ProcessGroup).
  * The kernel receives the NVLS multicast VA (``hdl.multicast_ptr``) as a
    raw ``cute.Pointer`` and rebuilds the MC tensor with the local tensor's
    layout — torch symm-mem guarantees identical offsets across ranks
    within one rendezvous buffer, so this is exactly the NVSHMEM
    symmetric-address contract the original relied on.
  * Double-buffered (C, flags) phases are kept: consecutive calls alternate
    phases so launches never race on the same tile flags; the kernel
    CAS-resets each flag it consumes (no host zeroing between calls).

Computes ``all_reduce_sum(x @ w.T)`` for x[B, K] / w[N, K] bf16, fp32
accumulation, TP in {2, 4, 8}. M is padded to the 128-row MMA tile.

ALIASING contract: :meth:`CuteDslGemmAR.gemm_allreduce` returns a VIEW of
the phase's symmetric C buffer, valid until the next call on this instance
(the codebase-wide staging contract). All ranks must call in lockstep.
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
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cute.typing import Int32, Float16, BFloat16, Float32, Float8E4M3FN, Float8E5M2
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
    """A/B/C stage heuristics (TMA-store variant of the original)."""
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


class PersistentGemmARKernel:
    """Blackwell persistent GEMM with LDMCxSTMC all-reduce epilogue.

    Identical mechanism to the NVSHMEM original (_kernel_cutlass.py):
      1. TMA warp loads A/B; MMA warp runs tcgen05 MMA into TMEM.
      2. Epilogue warps TMA-store bf16 partial tiles into the local
         symmetric C and release a per-tile flag on every rank via
         ``multimem.red.add`` on the flags MC address.
      3. AR warps spin (CAS-reset) on the tile flag until all ranks have
         stored, then ``multimem.ld_reduce`` + ``multimem.st`` their
         row-slice of the tile.
      4. Exit protocol: per-SM sys-scope release flag + spin so no rank
         leaves while peers still issue multimem traffic.

    Only the tensor plumbing changed: the MC views of C and the flags are
    rebuilt inside the jit entry from raw multicast pointers.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        rank_id: int,
        world_size: int,
    ):
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
        self.all_reduce_warp_id = (6, 7, 8, 9)
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
                *self.all_reduce_warp_id,
            )
        )
        self.epilogue_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.all_reduce_sync_bar_id = 4
        self.all_reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.all_reduce_sync_bar_id,
            num_threads=32 * len(self.all_reduce_warp_id),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.rank_id = rank_id
        self.num_ranks = world_size

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
        barrier_flag: cute.Tensor,
        barrier_flag_mc_ptr: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Launch the fused kernel.

        ``c_mc_ptr`` / ``barrier_flag_mc_ptr`` are the NVLS multicast VAs of
        the symmetric buffers backing ``c`` / ``barrier_flag``; the MC
        tensors are rebuilt here with the local layouts (identical offsets
        across ranks by the symm-mem rendezvous contract).
        """
        c_mc = cute.make_tensor(c_mc_ptr, c.layout)
        barrier_flag_mc = cute.make_tensor(barrier_flag_mc_ptr, barrier_flag.layout)

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
            c_mc,
            barrier_flag,
            barrier_flag_mc,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
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
        c_mc: cute.Tensor,
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
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
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

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

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

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
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
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
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
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
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            self.epilogue_tma_store_release_flag(
                tidx,
                warp_idx,
                acc_pipeline,
                tiled_mma,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                epi_tile,
                tile_sched,
                epilogue_op,
                flag_base=barrier_flag_mc,
                flag_mem_scope="gpu",
            )

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////
        #  Allreduce warps (LDMCxSTMC)
        # ///////////////////////////////////////////////////////////////////
        if warp_idx >= self.all_reduce_warp_id[0]:
            rank_id = self.rank_id
            num_ranks = Int32(self.num_ranks)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            # 128-bit ld/st for best multimem throughput
            atom_val = 128 // c_mc.element_type.width
            atom_thr_n = self.mma_tiler[1] // atom_val
            atom_thr_m = len(self.all_reduce_warp_id) * cute.arch.WARP_SIZE // atom_thr_n
            thr_layout = cute.make_layout(
                (atom_thr_m, atom_thr_n), stride=(atom_thr_n, 1)
            )
            val_layout = cute.make_layout((1, atom_val), stride=(atom_val, 1))

            copy_atom_load = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), c_mc.element_type
            )
            tiled_copy_fake = cute.make_tiled_copy_tv(
                copy_atom_load, thr_layout, val_layout
            )
            thr_copy_fake = tiled_copy_fake.get_slice(
                tidx - self.all_reduce_warp_id[0] * 32
            )
            idC = cute.make_identity_tensor(c_mc.shape)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                tile_id = Int32(
                    tile_sched._current_work_linear_idx
                    * cute.size(self.cluster_shape_mn)
                    + cute.arch.block_idx_in_cluster()
                )
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Wait until every rank's epilogue released this tile
                # (flag CAS-reset to 0 for the next phase reuse).
                if warp_idx == self.all_reduce_warp_id[0]:
                    with cute.arch.elect_one():
                        flag = barrier_flag.iterator + tile_id
                        utils.distributed.spin_lock_atom_cas_relaxed_wait(
                            lock_ptr=flag,
                            expected_val=num_ranks,
                            reset_val=0,
                            scope="gpu",
                        )

                self.all_reduce_sync_barrier.arrive_and_wait()
                gC_mc = cute.local_tile(
                    c_mc,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                cC = cute.local_tile(
                    idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                )

                tCgC_mc = thr_mma.partition_C(gC_mc)
                tCpC = thr_mma.partition_C(cC)
                tCgC_mc_slice = tCgC_mc[((None, None), 0, 0, *mma_tile_coord_mnl)]
                tCpC_slice = tCpC[((None, None), 0, 0, *mma_tile_coord_mnl)]

                # split rows of the tile across ranks
                cta_mma_tile_m = self.mma_tiler[0] // cute.size(
                    tiled_mma.thr_id.shape
                )
                m_local_rank = int(cta_mma_tile_m / self.num_ranks)
                tCgC_mc_slice_partitioned = cute.zipped_divide(
                    tCgC_mc_slice, (m_local_rank, self.mma_tiler[1])
                )
                tCpC_slice_partitioned = cute.zipped_divide(
                    tCpC_slice, (m_local_rank, self.mma_tiler[1])
                )
                tCgC_mc_local_rank = cute.slice_(
                    tCgC_mc_slice_partitioned, ((None, None), (rank_id, 0))
                )
                tCpC_local_rank = cute.slice_(
                    tCpC_slice_partitioned, ((None, None), (rank_id, 0))
                )

                frgC_mc = thr_copy_fake.partition_S(tCgC_mc_local_rank)
                frpC = thr_copy_fake.partition_S(tCpC_local_rank)
                atom, loop_m, loop_n = frgC_mc.shape
                for i in cutlass.range_constexpr(loop_m):
                    for j in cutlass.range_constexpr(loop_n):
                        if cute.elem_less(frpC[0, i, j], c_mc.shape):
                            mc_ptr = frgC_mc[None, i, j].iterator
                            x, y, z, w = 0, 0, 0, 0
                            if cutlass.const_expr(self.c_dtype == Float16):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == Float32):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == BFloat16):
                                x, y, z, w = (
                                    utils.distributed.multimem_ld_reduce_8xbf16(mc_ptr)
                                )
                            elif cutlass.const_expr(self.c_dtype == Float8E4M3FN):
                                x, y, z, w = (
                                    utils.distributed.multimem_ld_reduce_16xe4m3(mc_ptr)
                                )
                            elif cutlass.const_expr(self.c_dtype == Float8E5M2):
                                x, y, z, w = (
                                    utils.distributed.multimem_ld_reduce_16xe5m2(mc_ptr)
                                )
                            utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            self.all_reduce_sync_barrier.arrive_and_wait()

            #
            # Exit protocol: per-SM sys-scope release + spin so
            # 1. no rank exits while peers still issue multimem.ld_reduce
            # 2. each rank's multimem.st is visible system-wide
            #
            if warp_idx == self.all_reduce_warp_id[0]:
                with cute.arch.elect_one():
                    last_tile_id_linear = cute.size(
                        tile_sched.params.problem_layout_ncluster_mnl
                    ) * cute.size(self.cluster_shape_mn)
                    sm_id_linear = (
                        cute.arch.block_idx()[0]
                        + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                        + cute.arch.block_idx()[2]
                        * cute.arch.grid_dim()[0]
                        * cute.arch.grid_dim()[1]
                    )
                    utils.distributed.multimem_red_add1(
                        lock_ptr=barrier_flag_mc.iterator
                        + last_tile_id_linear
                        + sm_id_linear,
                        scope="sys",
                        order="release",
                    )
                    utils.distributed.spin_lock_atom_cas_relaxed_wait(
                        lock_ptr=barrier_flag.iterator
                        + last_tile_id_linear
                        + sm_id_linear,
                        expected_val=num_ranks,
                        reset_val=0,
                        scope="sys",
                    )

    @cute.jit
    def epilogue_tma_store_release_flag(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
        epilogue_op: cutlass.Constexpr,
        flag_base: cute.Tensor,
        flag_mem_scope: str,
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

        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

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
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tRS_rC.store(acc_vec)

                c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
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

            #
            # Per-tile release flag: wait for this tile's TMA store to land,
            # then multimem-increment the tile flag on every rank.
            #
            tile_id_linear = Int32(
                tile_sched._current_work_linear_idx * cute.size(self.cluster_shape_mn)
                + cute.arch.block_idx_in_cluster()
            )
            c_pipeline.producer_tail()
            if warp_idx == self.epilogue_warp_id[0]:
                with cute.arch.elect_one():
                    flag_curr_tile = flag_base.iterator + tile_id_linear
                    utils.distributed.multimem_red_add1(
                        lock_ptr=flag_curr_tile,
                        scope=flag_mem_scope,
                        order="release",
                    )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

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

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
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

NUM_PHASES = 2
_TILE_M = 128  # mma tiler (128, 128), 1-CTA — M pads to 128


def _pad_m(m: int) -> int:
    return int(math.ceil(m / _TILE_M) * _TILE_M)


class _Phase:
    """One (C, flags) symmetric buffer pair plus its multicast pointers."""

    def __init__(self, group_name: str, max_m: int, n: int, num_flags: int,
                 device: torch.device, dtype: torch.dtype):
        self.c = symm_mem.empty(max_m * n, dtype=dtype, device=device)
        c_hdl = symm_mem.rendezvous(self.c, group_name)
        self.flags = symm_mem.empty(num_flags, dtype=torch.int32, device=device)
        f_hdl = symm_mem.rendezvous(self.flags, group_name)
        if not (c_hdl.has_multicast_support and f_hdl.has_multicast_support):
            raise RuntimeError(
                "NVLS multicast unsupported on this system; the LDMCxSTMC "
                "GEMM+AR kernel requires it"
            )
        self.flags.zero_()
        self.c_view = self.c.view(max_m, n)  # [max_m, N] result rows
        self.c3 = self.c.view(max_m, n, 1)  # kernel-facing (M, N, L)
        self.c_mc_ptr = make_ptr(
            cutlass.BFloat16, c_hdl.multicast_ptr,
            cute.AddressSpace.gmem, assumed_align=16,
        )
        self.flags_mc_ptr = make_ptr(
            cutlass.Int32, f_hdl.multicast_ptr,
            cute.AddressSpace.gmem, assumed_align=16,
        )
        flags_cute = from_dlpack(self.flags, assumed_align=16)
        flags_cute.element_type = cutlass.Int32
        self.flags_cute = flags_cute.mark_layout_dynamic(leading_dim=0)
        # (m_pad) -> (a_cute, c_cute) descriptor cache on fixed pointers
        self.cute_cache: dict = {}


class CuteDslGemmAR:
    """Tile-fused GEMM+AllReduce: ``all_reduce_sum(x @ w.T)`` on TP ranks.

    Collective constructor — all ranks of ``group`` must call together.
    ``gemm_allreduce`` returns a [B, N] VIEW of the internal symmetric
    buffer, valid until the next call on this instance. All-rank lockstep
    calling with identical shapes is required (caller contract).

    The CuTe kernel is compiled on the first ``gemm_allreduce`` call (K is
    taken from ``x``); pass ``k=`` to compile eagerly in ``__init__``.
    """

    def __init__(self, group, max_m: int, n: int,
                 dtype: torch.dtype = torch.bfloat16, *, k: Optional[int] = None):
        if dtype != torch.bfloat16:
            raise ValueError("only bf16 supported")
        if n % _TILE_M != 0:
            raise ValueError(f"N={n} must be a multiple of {_TILE_M}")
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        if self.world not in (2, 4, 8):
            raise ValueError(f"TP must be 2/4/8, got {self.world}")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.n = n
        self.max_m = _pad_m(max_m)
        self.dtype = dtype

        num_tiles_max = (self.max_m // _TILE_M) * (n // _TILE_M)
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        num_flags = num_tiles_max + num_sms

        gname = self.group.group_name
        self.phases = [
            _Phase(gname, self.max_m, n, num_flags, self.device, dtype)
            for _ in range(NUM_PHASES)
        ]
        self.phase = 0

        self.gemm = PersistentGemmARKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=(_TILE_M, _TILE_M),
            cluster_shape_mn=(1, 1),
            rank_id=self.rank,
            world_size=self.world,
        )
        self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)

        self.k: Optional[int] = None
        self.a_buf: Optional[torch.Tensor] = None
        self.b_buf: Optional[torch.Tensor] = None
        self.b_cute = None
        self.compiled = None
        if k is not None:
            self._ensure_compiled(k)

        # flags (zeroed) and multicast bindings must be visible on every
        # rank before the first launch
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    # ------------------------------------------------------------- compile

    def _ensure_compiled(self, k: int) -> None:
        if self.compiled is not None:
            if k != self.k:
                raise ValueError(f"K changed: compiled for {self.k}, got {k}")
            return
        if (k * self.dtype.itemsize) % 16 != 0:
            raise ValueError(f"K={k} must give 16B-aligned rows")
        self.k = k
        self.a_buf = torch.zeros(
            (self.max_m, k, 1), device=self.device, dtype=self.dtype)
        self.b_buf = torch.zeros(
            (self.n, k, 1), device=self.device, dtype=self.dtype)
        b_cute = from_dlpack(self.b_buf, assumed_align=16)
        b_cute.element_type = cutlass.BFloat16
        self.b_cute = b_cute.mark_layout_dynamic(leading_dim=1)

        self.compiled = None
        ph0 = self.phases[0]
        a_cute, c_cute = self._descs(ph0, self.max_m)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compile_args = (
            a_cute, self.b_cute, c_cute,
            ph0.c_mc_ptr, ph0.flags_cute, ph0.flags_mc_ptr,
            self.max_active_clusters, stream)
        key = (f"gemm_ar_tp{self.world}_r{self.rank}_m{self.max_m}"
               f"_n{self.n}_k{k}")
        self.compiled = load_or_compile(self.gemm, compile_args, key)
        self.phase = 1 % NUM_PHASES

    def _descs(self, ph: _Phase, m_pad: int):
        hit = ph.cute_cache.get(m_pad)
        if hit is not None:
            return hit
        a_cute = from_dlpack(self.a_buf[:m_pad], assumed_align=16)
        a_cute.element_type = cutlass.BFloat16
        a_cute = a_cute.mark_layout_dynamic(leading_dim=1)
        c_cute = from_dlpack(ph.c3[:m_pad], assumed_align=16)
        c_cute.element_type = cutlass.BFloat16
        c_cute = c_cute.mark_layout_dynamic(leading_dim=1)
        ph.cute_cache[m_pad] = (a_cute, c_cute)
        return ph.cute_cache[m_pad]

    # ----------------------------------------------------------------- op

    def gemm_allreduce(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``all_reduce_sum(x @ w.T)`` as a [B, N] view of the
        current phase's symmetric buffer (valid until the next call)."""
        if x.dtype != self.dtype or w.dtype != self.dtype:
            raise TypeError("bf16 inputs required")
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("inputs must be contiguous")
        b, k = x.shape
        n, k_w = w.shape
        if n != self.n or k_w != k:
            raise ValueError(f"shape mismatch: x{tuple(x.shape)} w{tuple(w.shape)}")
        if b > self.max_m:
            raise ValueError(f"B={b} exceeds max_m={self.max_m}")
        self._ensure_compiled(k)

        m_pad = _pad_m(b)
        ph = self.phases[self.phase]
        self.phase = (self.phase + 1) % NUM_PHASES

        if m_pad != b:
            self.a_buf[b:m_pad].zero_()
        self.a_buf[:b, :, 0].copy_(x)
        self.b_buf[:, :, 0].copy_(w)

        a_cute, c_cute = self._descs(ph, m_pad)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        runtime_args = (
            a_cute, self.b_cute, c_cute, ph.c_mc_ptr,
            ph.flags_cute, ph.flags_mc_ptr, stream)
        self.compiled(*runtime_args)
        return ph.c_view[:b]


# ---------------------------------------------------------------- selftest
# mpirun -n 8 python3 communication/kernel_benchmarks/bench_cutedsl_gemm_ar.py
# (correctness vs torch.mm + dist.all_reduce at K=896, N=7168, then a
# CUDA-event bench against the two-kernel dispatcher numbers)

def _selftest() -> None:
    import statistics

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    assert world == 8, f"expected TP8, got {world}"
    group = dist.group.WORLD
    k, n_dim = 896, 7168
    batches = (32, 256, 2048, 4096, 8192)
    two_kernel_us = {32: 18.8, 256: 29.9, 2048: 103.6, 4096: 190.7,
                     8192: 375.1}  # B200 TP8 dispatcher reference

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

    op = CuteDslGemmAR(group, max_m=max(batches), n=n_dim,
                       dtype=torch.bfloat16, k=k)
    gen = torch.Generator(device="cuda")
    w = (0.02 * torch.randn(n_dim, k, device="cuda")).to(torch.bfloat16)
    dist.broadcast(w, src=0)
    for b in batches:
        gen.manual_seed(1234 + b * 8 + rank)
        x = (0.02 * torch.randn(b, k, generator=gen,
                                device="cuda")).to(torch.bfloat16)
        ref = torch.mm(x, w.T)
        dist.all_reduce(ref)
        for _ in range(2):  # 2nd call exercises phase/flag reuse
            out = op.gemm_allreduce(x, w)
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        log(f"correctness B={b:<5} max_abs_err={err:.4f} "
            f"{'PASS' if err < 0.25 else 'FAIL'}")
        assert err < 0.25, f"B={b} err={err}"
        us = time_us(lambda: op.gemm_allreduce(x, w))
        base = two_kernel_us[b]
        log(f"bench       B={b:<5} fused={us:7.1f}us two-kernel="
            f"{base:6.1f}us  {base / us:4.2f}x")
    log("PASS cutedsl_gemm_ar selftest")
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    _selftest()
