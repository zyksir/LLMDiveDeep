"""B200 CuTe DSL GEMM with bf16 and raw-fp32 outputs.

The split-K reduction is deterministic. Compiled variants cache by shape and
configuration; batch is dynamic and launches use the current CUDA stream.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


# ---------------------------------------------------------------------------
# DSM (distributed shared memory) primitives — plain tensor loads through a
# mapa'd pointer lower to ld.shared::cta, which is illegal for a peer CTA's
# SMEM, so the cluster reduction uses explicit inline PTX.
# ---------------------------------------------------------------------------


@dsl_user_op
def _cp_async_cg_16(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
    """PTX: cp.async.cg.shared.global — one 16 B LDGSTS (small-batch A path,
    bypassing the TMA tensor-descriptor slow path for tiny dim0)."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _createpolicy_evict_last(*, loc=None, ip=None) -> cutlass.Int64:
    """PTX: createpolicy.fractional.L2::evict_last.b64 — encode an L2
    cache policy descriptor (fraction 1.0) marking accessed lines
    EVICT_LAST. The TMA copy trait accepts this raw Int64 through the
    ``cache_policy`` kwarg (the DSL has no enum->Int64 conversion and no
    createpolicy op). Inert unless the device's persisting-L2 set-aside
    is nonzero (see _ensure_l2_carve)."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None) -> Int32:
    """PTX: mapa.shared::cluster.u32 — peer CTA SMEM address (32-bit)."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, Int32(peer_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_cluster_v4_f32(
    mapped_addr, v0, v1, v2, v3, *, loc=None, ip=None
) -> None:
    """PTX: st.shared::cluster.v4.f32 — one 16 B fabric message into a peer
    CTA's SMEM (scalar remote stores are one message per 4 B; vectorizing is
    a 4x cut in message count). Address must be 16 B aligned.

    NOTE: an st.async + mbarrier::complete_tx variant (owner waits for a byte
    total instead of the fence + remote-arrive handshake) was tried and
    measured ~2.5 us SLOWER at every B: per-message tx-accounting serializes
    on the owner's mbarrier."""
    llvm.inline_asm(
        None,
        [
            Int32(mapped_addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

io_dtype = BFloat16
acc_dtype = Float32

_EPI_WARPS = (0, 1, 2, 3)
_MMA_WARP = 4
_TMA_WARP = 5
_THREADS = 192  # 6 warps: 4 epilogue + 1 MMA + 1 TMA

_EPI_BAR_ID = 1
_TMEM_BAR_ID = 2

# A rows fetched per stage on the small-batch cp.async path (SMEM rows above
# the real batch hold garbage that is never reduced or stored)
_SMALLB_ROWS = 16


@dataclass(frozen=True)
class Config:
    """Compile-time kernel variant (one cute.compile per distinct value)."""

    use_2cta: bool = False  # cta_group::2 pairs (split_k == 1 only)
    block_n: int = 96       # N tile (need not divide n1 or n1+n2)
    block_k: int = 64       # K tile (K % block_k == 0 required)
    cluster_n: int = 1      # h TMA multicast width across N-tile groups
    split_k: int = 4        # K splits = reduction-cluster size (1 = direct)
    ab_stages: int = 6      # TMA pipeline depth (SMEM bound)
    acc_stages: int = 2     # TMEM accumulator double buffering
    use_pdl: bool = True    # programmatic dependent launch
    opt0: bool = True       # ptxas --opt-level=0 (template's verified setting)
    # load A via per-stage cp.async instead of tensor TMA: gmem tensors with
    # dim0 <= 9 hit a measured TMA slow path (~2x mainloop), so tiny batches
    # use LDGSTS of _SMALLB_ROWS clamped rows (garbage rows never stored)
    small_batch: bool = False
    # EVICT_LAST L2 policy on the weight (B) TMA loads, so the 39 MB
    # fused front weight survives the expert stage's streaming traffic
    # (isolated probe: 12.6 -> 9.6 us under a 120 MB evictor, bitwise
    # identical). Requires the persisting-L2 set-aside; default on via
    # _EVICT_LAST_ENV (opt out with LLMDD_DUALOUT_EVICT_LAST=0).
    evict_last_b: bool = False
    # dev-probe-only flag (produces WRONG outputs; never set in production)
    dbg_skip_store: bool = False   # skip staging/stores/reduction entirely

    def __post_init__(self):
        if self.small_batch:
            assert self.split_k > 1, "small_batch rides the split-K variant"
            assert self.block_k == 64, "LDGSTS path assumes one SW128 atom"
        if self.split_k > 1:
            # the split-K reduction runs over DSM inside one cluster
            assert not self.use_2cta, "split_k > 1 requires 1-CTA MMA"
            assert self.cluster_n == 1, "split_k > 1 requires cluster_n == 1"
            assert self.split_k <= 8, "cluster size cap (portable) is 8"
            assert 128 % self.split_k == 0, "row ownership needs cta_m % sk"
        assert self.block_n % 32 == 0

    @property
    def mma_m(self) -> int:
        return 256 if self.use_2cta else 128

    @property
    def cta_m(self) -> int:
        return 128  # per-CTA output rows (both cta_group variants)


@lru_cache(maxsize=None)
def _shared_storage_ty(ab_stages: int, acc_stages: int, epi_words: int):
    # NOTE: built via an explicit __annotations__ dict because this module
    # uses `from __future__ import annotations` (which would stringify the
    # member annotations and break cute.struct's inspection).
    cls = type(
        "SharedStorage",
        (),
        {
            "__annotations__": {
                "ab_mbar_ptr": cute.struct.MemRange[
                    cutlass.Int64, ab_stages * 2
                ],
                "acc_mbar_ptr": cute.struct.MemRange[
                    cutlass.Int64, acc_stages * 2
                ],
                # split-K cross-CTA handshake (cluster-scope mbarriers)
                "dsm_ready_mbar": cute.struct.MemRange[cutlass.Int64, 1],
                "dsm_done_mbar": cute.struct.MemRange[cutlass.Int64, 1],
                # small-batch A path: per-stage cp.async completion mbarriers
                "a_cp_mbar": cute.struct.MemRange[cutlass.Int64, ab_stages],
                # padded fp32 staging tile [cta_m, block_n + 4]: split_k == 1
                # stages the CTA's own tile here; split_k > 1 treats it as
                # split_k sender slots of cta_m/split_k rows each, filled by
                # the peers' st.shared::cluster pushes. Row stride block_n+4
                # keeps 16 B alignment while spreading banks.
                "epi_buf": cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, epi_words], 16
                ],
                "tmem_dealloc_mbar": cutlass.Int64,
                "tmem_holding_buf": cutlass.Int32,
            }
        },
    )
    return cute.struct(cls)


_max_active_clusters_cache: dict[int, int] = {}


def _get_max_active_clusters(cluster_size: int) -> int:
    """Plain-int hardware limit (primed before compile; JIT constexpr)."""
    if cluster_size not in _max_active_clusters_cache:
        _max_active_clusters_cache[cluster_size] = int(
            utils.HardwareInfo().get_max_active_clusters(cluster_size)
        )
    return _max_active_clusters_cache[cluster_size]


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


class _DualOutGemm:
    """One compiled variant: fixed (K, n1, n2, Config); batch B is dynamic."""

    def __init__(self, k_dim: int, n1: int, n2: int, cfg: Config):
        assert k_dim % cfg.block_k == 0, (k_dim, cfg.block_k)
        self.k = k_dim
        self.n1 = n1
        self.n2 = n2
        self.ntot = n1 + n2
        self.cfg = cfg
        self.mma_tiler = (cfg.mma_m, cfg.block_n, cfg.block_k)
        # cluster m covers the CTA pair (2cta) OR the split-K reduction group
        self.cluster_m = cfg.split_k if cfg.split_k > 1 else (
            2 if cfg.use_2cta else 1
        )
        self.cluster_shape_mnk = (self.cluster_m, cfg.cluster_n, 1)
        n_tiles = -(-self.ntot // cfg.block_n)
        self.n_tiles_pad = -(-n_tiles // cfg.cluster_n) * cfg.cluster_n
        self.num_k_tiles = k_dim // cfg.block_k
        assert cfg.split_k <= self.num_k_tiles
        self.epi_tile = (cfg.cta_m, 32)
        self.max_active_clusters = _get_max_active_clusters(
            self.cluster_shape_mnk[0] * self.cluster_shape_mnk[1]
        )
        self.shared_storage_ty = _shared_storage_ty(
            cfg.ab_stages, cfg.acc_stages, cfg.cta_m * (cfg.block_n + 4)
        )

    # -- host-side JIT entry -------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mH: cute.Tensor,     # (B, K) bf16, B dynamic
        mW: cute.Tensor,     # (n1+n2, K) bf16
        mBF: cute.Tensor,    # (B, n1) bf16 view of buf, row stride n1+2*n2
        mF32: cute.Tensor,   # (B, n2) fp32 view of buf
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        op = tcgen05.MmaF16BF16Op(
            io_dtype,
            acc_dtype,
            (cfg.mma_m, cfg.block_n, 16),
            tcgen05.CtaGroup.TWO if cfg.use_2cta else tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,  # h  [B, K], K contiguous
            tcgen05.OperandMajorMode.K,  # w  [N, K], K contiguous
        )
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, io_dtype, cfg.ab_stages
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, io_dtype, cfg.ab_stages
        )

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

        op_tma = cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO if cfg.use_2cta else tcgen05.CtaGroup.ONE
        )
        a_smem_layout_one = cute.slice_(a_smem_layout, (None, None, None, 0))
        b_smem_layout_one = cute.slice_(b_smem_layout, (None, None, None, 0))

        # split-K cluster CTAs load DIFFERENT k-slices: the TMA atoms must
        # not partition/multicast operands across the cluster m-mode, so the
        # atoms are built against a degenerate cluster.
        if cutlass.const_expr(cfg.split_k > 1):
            atom_vmnk_shape = (1, 1, 1, 1)
        else:
            atom_vmnk_shape = cta_layout_vmnk.shape

        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma, mH, a_smem_layout_one, self.mma_tiler, tiled_mma,
            atom_vmnk_shape,
        )
        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            op_tma, mW, b_smem_layout_one, self.mma_tiler, tiled_mma,
            atom_vmnk_shape,
        )

        # scheduler m-slot interleaves the cluster-m rank: CTA pair half
        # (2cta) or split index (split-K); mma_m_idx = m // cluster_m
        m_slot = cute.ceil_div(mH.shape[0], cfg.mma_m) * self.cluster_m

        tile_sched_params = utils.PersistentTileSchedulerParams(
            (m_slot, self.n_tiles_pad, 1), self.cluster_shape_mnk
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, self.max_active_clusters
        )

        self.kernel(
            tiled_mma,
            a_tma_atom,
            a_tma_tensor,
            b_tma_atom,
            b_tma_tensor,
            mBF,
            mF32,
            mH,
            a_smem_layout,
            b_smem_layout,
            cta_layout_vmnk,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=(_THREADS, 1, 1),
            cluster=self.cluster_shape_mnk,
            stream=stream,
            use_pdl=cfg.use_pdl,
        )

    # -- device kernel --------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mBF: cute.Tensor,
        mF32: cute.Tensor,
        mH_raw: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        cfg = self.cfg
        cluster_m = cutlass.const_expr(self.cluster_m)

        # PDL: nothing before this touches gmem data (descriptor prefetch is
        # tensormap-only), so every warp gates its first data access here.
        if cutlass.const_expr(cfg.use_pdl):
            cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _ = cute.arch.block_idx()

        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == _TMA_WARP:
            if cutlass.const_expr(not cfg.small_batch):
                cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        if cutlass.const_expr(cfg.split_k > 1):
            # cluster m-mode CTAs hold DIFFERENT k-slices: no operand
            # multicast at all (self-delivery masks via the size-1 k-mode)
            num_mcast_participants = 1
            tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=3
            )
            tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=3
            )
        else:
            num_mcast_participants = (
                cute.size(cta_layout_vmnk, mode=[1])
                + cute.size(cta_layout_vmnk, mode=[2])
                - 1
            )
            tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
            )
            tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
            )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_ty)

        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=_EPI_BAR_ID, num_threads=32 * len(_EPI_WARPS)
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=_TMEM_BAR_ID, num_threads=32 * (1 + len(_EPI_WARPS))
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=_EPI_WARPS[0],
            is_two_cta=cfg.use_2cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        if cutlass.const_expr(cfg.small_batch):
            # A arrives via cp.async (own mbarriers); only B counts TMA tx
            num_tma_copy_bytes = cute.size_in_bytes(
                io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
            )
        else:
            num_tma_copy_bytes = (
                cute.size_in_bytes(
                    io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
                )
                + cute.size_in_bytes(
                    io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
                )
            ) * cute.size(cta_layout_vmnk, mode=[0])

        # for split-K the cluster only scopes the DSM reduction: the TMA/UMMA
        # pipelines must stay strictly per-CTA (degenerate layout)
        if cutlass.const_expr(cfg.split_k > 1):
            pipe_vmnk = cute.tiled_divide(
                cute.make_layout((1, 1, 1)), (tiled_mma.thr_id,)
            )
        else:
            pipe_vmnk = cta_layout_vmnk

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=cfg.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, size=num_mcast_participants
            ),
            tx_count=num_tma_copy_bytes,
            cta_layout_vmnk=pipe_vmnk,
        ).make_participants()

        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=cfg.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=cute.size(cta_layout_vmnk, mode=[0]) * len(_EPI_WARPS),
            ),
            cta_layout_vmnk=pipe_vmnk,
        ).make_participants()

        # split-K cross-CTA handshake barriers (one 'ready' + one 'done' per
        # CTA; each receives split_k arrivals per work unit — one elected
        # thread per cluster rank)
        dsm_ready_ptr = storage.dsm_ready_mbar.data_ptr()
        dsm_done_ptr = storage.dsm_done_mbar.data_ptr()
        a_cp_ptr = storage.a_cp_mbar.data_ptr()
        if cutlass.const_expr(cfg.split_k > 1):
            if warp_idx == _EPI_WARPS[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(dsm_ready_ptr, cfg.split_k)
                    cute.arch.mbarrier_init(dsm_done_ptr, cfg.split_k)
                    if cutlass.const_expr(cfg.small_batch):
                        # one arrival per TMA-warp thread per stage
                        for s in cutlass.range_constexpr(cfg.ab_stages):
                            cute.arch.mbarrier_init(a_cp_ptr + s, 32)
            cute.arch.mbarrier_init_fence()

        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True
        )

        sEpi = cute.make_tensor(
            storage.epi_buf.data_ptr(),
            cute.make_layout(
                (cfg.cta_m, cfg.block_n + 4), stride=(cfg.block_n + 4, 1)
            ),
        )
        sA = smem.allocate_tensor(
            element_type=io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        num_rows = mBF.shape[0]  # dynamic batch B
        m_mma_tiles = cute.ceil_div(num_rows, cfg.mma_m)

        # Coordinate space of the (padded) output: identity tensor partitioned
        # exactly like a C operand -> per-element global (row, col).
        idC = cute.make_identity_tensor(
            (m_mma_tiles * cfg.mma_m, self.n_tiles_pad * cfg.block_n)
        )

        gA = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None)
        )
        gB = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None)
        )
        gC = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(
            cute.select(self.mma_tiler, mode=[0, 1])
        )
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, cfg.acc_stages)
        )

        # partition coords/layouts must match the (possibly degenerate) atom
        # construction: for split-K every CTA owns its full A and B boxes
        if cutlass.const_expr(cfg.split_k > 1):
            a_part_coord = cutlass.Int32(0)
            a_part_layout = cute.make_layout(1)
            b_part_coord = cutlass.Int32(0)
            b_part_layout = cute.make_layout(1)
        else:
            a_part_coord = cta_in_cluster_coord_vmnk[2]
            a_part_layout = cute.make_layout(
                cute.size(cta_layout_vmnk, mode=[2])
            )
            b_part_coord = cta_in_cluster_coord_vmnk[1]
            b_part_layout = cute.make_layout(
                cute.size(cta_layout_vmnk, mode=[1])
            )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_part_coord,
            a_part_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_part_coord,
            b_part_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # split-K k-tile distribution: splits [0, rem) get floor+1 tiles.
        ktps_floor = cutlass.const_expr(self.num_k_tiles // cfg.split_k)
        nkt_rem = cutlass.const_expr(self.num_k_tiles % cfg.split_k)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # -----------------------------------------------------------------
        # TMA warp: stream this work tile's split K-slice of h and w_all
        # -----------------------------------------------------------------
        if warp_idx == _TMA_WARP:
            b_cache_policy = None
            if cutlass.const_expr(cfg.evict_last_b):
                # EVICT_LAST for the weight (B) loads only; A untouched
                b_cache_policy = _createpolicy_evict_last()
            if cutlass.const_expr(cfg.small_batch):
                # LDGSTS addressing for the A tile (K_SW128 atom, one atom
                # per 64-col k-tile): 16 rows x 8 16 B chunks, 4 per lane
                lane = tidx % 32
                a_gmem_base = mH_raw.iterator.toint()
                a_smem_base = sA.iterator.toint()
                a_stage_bytes = cutlass.const_expr(
                    cfg.cta_m * cfg.block_k * 2
                )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_m_idx = cur_tile_coord[0] // cluster_m
                n_idx = cur_tile_coord[1]
                if cutlass.const_expr(cfg.split_k > 1):
                    s = cur_tile_coord[0] % cluster_m
                else:
                    s = cutlass.Int32(0)
                if cutlass.const_expr(nkt_rem == 0):
                    k_base = s * ktps_floor
                    k_cnt = ktps_floor
                else:
                    k_base = s * ktps_floor + nkt_rem
                    k_cnt = cutlass.Int32(ktps_floor)
                    if s < nkt_rem:
                        k_base = s * (ktps_floor + 1)
                        k_cnt = cutlass.Int32(ktps_floor + 1)

                tAgA_slice = tAgA[(None, mma_m_idx, None)]
                tBgB_slice = tBgB[(None, n_idx, None)]

                for k_local in cutlass.range(k_cnt):
                    handle = ab_producer.acquire_and_advance()
                    if cutlass.const_expr(cfg.small_batch):
                        # rows >= B re-read row B-1 (results never stored);
                        # dest swizzle mirrors the K_SW128 atom the MMA
                        # descriptor expects
                        kt = k_base + k_local
                        for j in cutlass.range_constexpr(
                            _SMALLB_ROWS * 8 // 32
                        ):
                            t = lane + 32 * j
                            row = t // 8
                            chunk = t % 8
                            rr = cutlass.min(
                                mma_m_idx * cfg.mma_m + row, num_rows - 1
                            )
                            # SW128 hardware swizzle (what TMA/UMMA actually
                            # apply, fitted via one-hot probe): the 16 B chunk
                            # index is XORed with bits [9:7] of the ABSOLUTE
                            # byte address, i.e. addr ^ (((addr >> 7) & 7) << 4)
                            # (the sA base offset participates; this is NOT the
                            # element-level S<3,4,3> crd2idx formula).
                            pre = (
                                a_smem_base
                                + handle.index * a_stage_bytes
                                + row * 128
                                + chunk * 16
                            )
                            _cp_async_cg_16(
                                pre ^ (((pre >> 7) & 7) << 4),
                                a_gmem_base
                                + 2 * (rr * self.k + kt * cfg.block_k)
                                + 16 * chunk,
                            )
                        cute.arch.cp_async_mbarrier_arrive_noinc(
                            a_cp_ptr + handle.index
                        )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, k_base + k_local)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=tma_mcast_mask_a,
                        )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_base + k_local)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=tma_mcast_mask_b,
                        cache_policy=b_cache_policy,
                    )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # -----------------------------------------------------------------
        # MMA warp: accumulate the split's K-slice into TMEM
        # -----------------------------------------------------------------
        elif warp_idx == _MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            if cutlass.const_expr(cfg.small_batch):
                a_phase = Int32(0)

            while work_tile.is_valid_tile:
                if is_leader_cta:
                    if cutlass.const_expr(cfg.split_k > 1):
                        s = work_tile.tile_idx[0] % cluster_m
                    else:
                        s = cutlass.Int32(0)
                    if cutlass.const_expr(nkt_rem == 0):
                        k_cnt = ktps_floor
                    else:
                        k_cnt = cutlass.Int32(ktps_floor)
                        if s < nkt_rem:
                            k_cnt = cutlass.Int32(ktps_floor + 1)

                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _ in cutlass.range(k_cnt):
                        handle = ab_consumer.wait_and_advance()
                        if cutlass.const_expr(cfg.small_batch):
                            # A stage completion (cp.async mbarrier), then
                            # make the generic-proxy LDGSTS writes visible
                            # to the async proxy the UMMA reads through
                            cute.arch.mbarrier_wait(
                                a_cp_ptr + handle.index, a_phase
                            )
                            cute.arch.fence_proxy(
                                "async.shared", space="cta"
                            )
                            if handle.index == cfg.ab_stages - 1:
                                a_phase = a_phase ^ 1
                        num_k_blocks = cute.size(tCrA, mode=[2])
                        for k_blk in cutlass.range_constexpr(num_k_blocks):
                            k_blk_coord = (None, None, k_blk, handle.index)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[k_blk_coord],
                                tCrB[k_blk_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()

                    acc_empty.commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_producer.tail()

        # -----------------------------------------------------------------
        # Epilogue warps: TMEM -> RMEM -> padded SMEM -> coalesced N-major
        # gmem stores. The SMEM round trip exists because the TMEM load
        # hands each lane a full accumulator ROW: storing that directly
        # scatters every warp store across 32 cache sectors.
        #
        # split_k == 1: stage locally, drain N-major.
        # split_k > 1: PUSH reduce-scatter over the cluster. Output rows are
        # owned strided by cluster rank (owner = row % split_k); every CTA
        # st.shared::cluster-pushes its partial rows into the owner's slot
        # (fire-and-forget stores — no DSM read latency on the critical
        # path), then after the ready-mbarrier each owner sums its split_k
        # slots from LOCAL SMEM in fixed rank order (deterministic) and
        # stores its rows. Rows >= B are neither pushed nor reduced.
        # -----------------------------------------------------------------
        elif warp_idx < _MMA_WARP:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE),
                cutlass.Float32,
            )

            # N-major store mapping: lane -> column, warp -> row group
            st_col = tidx % 32
            st_row0 = tidx // 32
            row_passes = cfg.cta_m // 4
            n_csubs = cfg.block_n // 32
            rows_per = cfg.cta_m // cfg.split_k  # rows owned per rank
            red_passes = rows_per // 4

            if cutlass.const_expr(cfg.split_k > 1):
                epi_phase = Int32(0)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_m_idx = cur_tile_coord[0] // cluster_m
                n_idx = cur_tile_coord[1]
                s_idx = cur_tile_coord[0] % cluster_m

                acc_full = acc_consumer.wait_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

                tAcc_epi = cute.flat_divide(
                    tCtAcc[((None, None), 0, 0)], self.epi_tile
                )
                tCgC_epi = cute.flat_divide(
                    tCgC[((None, None), 0, 0, mma_m_idx, n_idx)], self.epi_tile
                )

                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                tTR_cC = thr_copy_t2r.partition_D(tCgC_epi)
                tTR_rAcc = cute.make_rmem_tensor(
                    tTR_cC[(None, None, None, 0, 0)].shape, cutlass.Float32
                )
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                # per-element LOCAL (row, col) inside one epi subtile — the
                # same tiled-copy partitioning as tTR_cC, so element i of
                # tTR_rAcc lands at local coord tTR_cL[i]
                cEpi_local = cute.make_identity_tensor(self.epi_tile)
                tTR_cL = thr_copy_t2r.partition_D(cEpi_local)

                # tile-base global coords (thread-invariant differences)
                cC0 = tTR_cC[(None, None, None, 0)]
                tile_row0 = cC0[0][0] - tTR_cL[0][0]
                tile_col0 = cC0[0][1] - tTR_cL[0][1]

                # The Ld32x32b partitioning hands each lane one full TMEM
                # row, constant across elements and subtiles.
                row_l = tTR_cL[0][0]
                if cutlass.const_expr(cfg.split_k > 1):
                    # push destination: owner rank's slot for this sender
                    # (byte address in the owner CTA's sEpi via mapa)
                    dst_row = s_idx * rows_per + row_l // cfg.split_k
                    dst_base = _mapa_shared_cluster(
                        sEpi.iterator, row_l % cfg.split_k
                    ) + Int32(4 * (cfg.block_n + 4)) * dst_row
                    row_live = (tile_row0 + row_l) < num_rows

                # Ld32x32b hands warp w rows [32w, 32w+32): a warp whose whole
                # row range is dead (B < 32w) skips the TMEM drain entirely —
                # at tiny B this cuts 3 of 4 TMEM load chains off the epilogue
                # critical path. (warp-uniform predicate: row_l is lane-linear)
                warp_live = (tile_row0 + row_l - tidx % 32) < num_rows

                # ---- drain TMEM; stage the fp32 accumulator row into SMEM:
                # locally (split_k == 1) or pushed to the row-owner CTA
                for sub in cutlass.range(subtile_cnt):
                    if warp_live:
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, sub)],
                            tTR_rAcc,
                        )
                    if cutlass.const_expr(cfg.dbg_skip_store):
                        pass
                    elif cutlass.const_expr(cfg.split_k == 1):
                        if warp_live:
                            for i in cutlass.range_constexpr(
                                cute.size(tTR_rAcc)
                            ):
                                sEpi[row_l, sub * 32 + tTR_cL[i][1]] = (
                                    tTR_rAcc[i]
                                )
                    else:
                        if row_live:
                            # element i <-> subtile column i for this copy
                            # atom, so pack 4 consecutive columns per 16 B
                            # remote store message
                            for i in cutlass.range_constexpr(
                                0, cute.size(tTR_rAcc), 4
                            ):
                                _st_shared_cluster_v4_f32(
                                    dst_base
                                    + Int32(4)
                                    * (sub * 32 + tTR_cL[i][1]),
                                    tTR_rAcc[i],
                                    tTR_rAcc[i + 1],
                                    tTR_rAcc[i + 2],
                                    tTR_rAcc[i + 3],
                                )

                with cute.arch.elect_one():
                    acc_full.release()

                if cutlass.const_expr(cfg.dbg_skip_store):
                    pass
                elif cutlass.const_expr(cfg.split_k == 1):
                    # ---- direct path: coalesced N-major drain (lane = col)
                    epilogue_sync_barrier.arrive_and_wait()
                    for csub in cutlass.range_constexpr(n_csubs):
                        col_l = csub * 32 + st_col
                        col_g = tile_col0 + col_l
                        for p in cutlass.range(row_passes):
                            row_s = st_row0 + p * 4
                            row_g = tile_row0 + row_s
                            if (row_g < num_rows) & (col_g < self.ntot):
                                val = sEpi[row_s, col_l]
                                if col_g < self.n1:
                                    mBF[row_g, col_g] = val.to(io_dtype)
                                else:
                                    mF32[row_g, col_g - self.n1] = val
                    epilogue_sync_barrier.arrive_and_wait()  # sEpi reuse
                else:
                    # ---- split path: all pushes issued -> release fence ->
                    # arrive every peer's ready barrier; once all split_k
                    # senders arrived, sum OWN rows from local slots in
                    # FIXED rank order (deterministic) and store them.
                    epilogue_sync_barrier.arrive_and_wait()
                    if warp_idx == _EPI_WARPS[0]:
                        with cute.arch.elect_one():
                            cute.arch.fence_acq_rel_cluster()
                            for r in cutlass.range_constexpr(cfg.split_k):
                                cute.arch.mbarrier_arrive(
                                    dsm_ready_ptr, Int32(r)
                                )
                    cute.arch.mbarrier_wait(dsm_ready_ptr, epi_phase)
                    cute.arch.fence_acq_rel_cluster()
                    for csub in cutlass.range_constexpr(n_csubs):
                        col_l = csub * 32 + st_col
                        col_g = tile_col0 + col_l
                        for p in cutlass.range_constexpr(red_passes):
                            row_in = st_row0 + p * 4
                            row_g = (
                                tile_row0 + s_idx + row_in * cfg.split_k
                            )
                            if (row_g < num_rows) & (col_g < self.ntot):
                                total = sEpi[row_in, col_l]
                                for r in cutlass.range_constexpr(
                                    1, cfg.split_k
                                ):
                                    total = total + sEpi[
                                        r * rows_per + row_in, col_l
                                    ]
                                if col_g < self.n1:
                                    mBF[row_g, col_g] = total.to(io_dtype)
                                else:
                                    mF32[row_g, col_g - self.n1] = total
                    # signal our reads finished; wait for everyone before
                    # the next unit may overwrite sEpi. This wait is ALSO
                    # the cluster exit barrier: any remote access (even an
                    # mbarrier arrive) to an already-exited CTA's SMEM is
                    # illegal, so nobody may exit before all peers' remote
                    # traffic has landed.
                    epilogue_sync_barrier.arrive_and_wait()
                    if warp_idx == _EPI_WARPS[0]:
                        with cute.arch.elect_one():
                            for r in cutlass.range_constexpr(cfg.split_k):
                                cute.arch.mbarrier_arrive(
                                    dsm_done_ptr, Int32(r)
                                )
                    cute.arch.mbarrier_wait(dsm_done_ptr, epi_phase)
                    epi_phase = epi_phase ^ 1

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if cutlass.const_expr(cfg.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}

# TMA loads from a gmem tensor with dim0 <= 9 hit a measured per-request
# slow path on B200 (~2-3x mainloop time); those batches run the
# small_batch variant, which loads A via cp.async instead of tensor TMA.
_TMA_SLOW_MAX_ROWS = 9

# Per-B-bucket tuned configs for the Kimi shape (offline sweep on B200; fixed
# here — a runtime autotuner could pick different configs per process, which
# would break bitwise run-to-run determinism). Fallback: DEFAULT_CONFIG.
DEFAULT_CONFIG = Config()  # split_k=4, block_n=96, 6 stages, PDL
_SMALLB_CONFIGS = {
    "small8": Config(small_batch=True, split_k=8),
    "small4": Config(small_batch=True, split_k=4),
}
_TUNED: dict[tuple, Config] = {}


def _bucket(batch: int) -> int:
    b = 1
    while b < batch:
        b *= 2
    return min(max(b, 1), 128)


def _pick_config(batch: int, k_dim: int, n1: int, n2: int) -> Config:
    if batch <= _TMA_SLOW_MAX_ROWS:
        # B200 sweep: split_k=8 wins at B<=2, split_k=4 at B in 3..9
        tag = "small8" if batch <= 2 else "small4"
        cfg = _TUNED.get((k_dim, n1, n2, tag), _SMALLB_CONFIGS[tag])
    else:
        cfg = _TUNED.get((k_dim, n1, n2, _bucket(batch)), DEFAULT_CONFIG)
    if _EVICT_LAST_ENV:
        cfg = dataclasses.replace(cfg, evict_last_b=True)
        _ensure_l2_carve()
    return cfg


# Weight TMA loads carry an EVICT_LAST L2 policy AND the device
# persisting-L2 set-aside is carved (the policy is inert without it).
# Default ON: in-layer TP8 A/B (Aug-11) wins 0.9-3.0 us at B=8..64 and
# ties at 128; prefill in the same process is unaffected (carve +
# lingering pinned lines within session noise at 4096/8192). Set
# LLMDD_DUALOUT_EVICT_LAST=0 to opt out. Probe:
# local_debug/l2pin_evictlast_probe.py.
_EVICT_LAST_ENV = os.environ.get("LLMDD_DUALOUT_EVICT_LAST", "1") == "1"
_L2_CARVE_BYTES = 48 << 20
_l2_carve_done = False


def _ensure_l2_carve() -> None:
    """One-time cudaDeviceSetLimit for the persisting-L2 set-aside.
    Called from eager picks only (first call per shape precedes any CUDA
    graph capture via the mandatory warmup)."""
    global _l2_carve_done
    if _l2_carve_done:
        return
    from cuda.bindings import runtime as cudart

    err = cudart.cudaDeviceSetLimit(
        cudart.cudaLimit.cudaLimitPersistingL2CacheSize, _L2_CARVE_BYTES)
    err = err[0] if isinstance(err, tuple) else err
    assert int(err) == 0, err
    _l2_carve_done = True


def _wrap(h, w_all, buf_bf, buf_f32):
    mH = (
        from_dlpack(h, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=64)
    )
    mW = (
        from_dlpack(w_all, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=64)
    )
    mBF = from_dlpack(buf_bf, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    mF32 = from_dlpack(buf_f32, assumed_align=4).mark_layout_dynamic(
        leading_dim=1
    )
    return mH, mW, mBF, mF32


def dual_out_gemm_cutedsl(
    h: torch.Tensor,
    w_all: torch.Tensor,
    n1: int,
    n2: int,
    *,
    config: Config | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """out_bf16 = h @ w_all[:n1].T (bf16), out_fp32 = h @ w_all[n1:].T (raw
    fp32 accumulator bits), both from ONE kernel reading w_all once.

    h [B, K] bf16 contiguous; w_all [n1+n2, K] bf16 contiguous
    (= cat([W1, W2], 0), caller-guaranteed). K % 64 == 0 and n1 even required.
    Returns zero-copy views into one [B, n1 + 2*n2] bf16-typed buffer:
    (out_bf16 [B, n1] bf16, out_fp32 [B, n2] fp32), row stride n1 + 2*n2.
    Deterministic: bitwise identical outputs run-to-run for fixed inputs.
    """
    assert h.dtype == torch.bfloat16 and h.dim() == 2 and h.is_contiguous()
    assert w_all.dtype == torch.bfloat16 and w_all.is_contiguous()
    k_dim = h.shape[1]
    assert w_all.shape == (n1 + n2, k_dim)
    assert n1 % 2 == 0, "n1 must be even for the fp32 view alignment"
    assert k_dim % 64 == 0, "K must be a multiple of 64"
    batch = h.shape[0]

    cfg = config if config is not None else _pick_config(batch, k_dim, n1, n2)
    if cfg.small_batch:
        assert batch <= _SMALLB_ROWS, "small_batch variant loads 16 A rows"

    buf = torch.empty(
        batch, n1 + 2 * n2, device=h.device, dtype=torch.bfloat16
    )
    out_bf = buf[:, :n1]
    out_f32 = buf[:, n1:].view(torch.float32)

    tensors = _wrap(h, w_all, out_bf, out_f32)
    # the launch stream is fetched fresh on EVERY call (graph-capture safe)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (k_dim, n1, n2, cfg)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        op = _DualOutGemm(k_dim, n1, n2, cfg)
        options = "--gpu-arch sm_100a"
        if cfg.opt0:
            options += (
                " --ptxas-options '--opt-level=0"
                " --allow-expensive-optimizations true'"
            )
        compiled = cute.compile(op, *tensors, stream, options=options)
        _KERNEL_CACHE[key] = compiled
    compiled(*tensors, stream)
    return out_bf, out_f32


# ---------------------------------------------------------------------------
# Kimi-K3 decode instantiation (thin wrapper)
# ---------------------------------------------------------------------------

HIDDEN = 7168   # K
N_BF16 = 1984   # fc1 shard (448) + shared gate/up (1536) -> bf16 merged
N_GATE = 896    # router gate -> fp32 logits
N_TOTAL = N_BF16 + N_GATE  # 2880


def fused_input_gemm_cutedsl(
    h: torch.Tensor, w_all: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi-K3 decode input GEMMs, fused. h [B, 7168] bf16; w_all
    [2880, 7168] bf16 = cat([w_fc1(448), w_gu(1536), w_gate(896)], 0).

    Returns (merged [B, 1984] bf16, logits [B, 896] fp32 raw accumulator)
    as zero-copy views into one buffer (row stride 3776).
    """
    assert w_all.shape == (N_TOTAL, HIDDEN)
    return dual_out_gemm_cutedsl(h, w_all, N_BF16, N_GATE)


AVAILABLE = True
dual_out_gemm = dual_out_gemm_cutedsl
fused_front = fused_input_gemm_cutedsl


def build_fused_front_weight(
    front_weight: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Pack front and gate rows for fused_front."""
    assert front_weight.dtype == gate_weight.dtype == torch.bfloat16
    assert front_weight.shape == (N_BF16, HIDDEN)
    assert gate_weight.shape == (N_GATE, HIDDEN)
    return torch.cat((front_weight, gate_weight), dim=0).contiguous()


def bench(
    *,
    batch: int = 16,
    iters: int = 100,
    repeats: int = 5,
) -> dict[str, float]:
    """Check correctness and graph latency against unfused PyTorch GEMMs."""
    from ._bench import (
        check_regression,
        dual_out_reference,
        graph_time_us,
    )

    torch.manual_seed(0)
    h = torch.randn(batch, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N_TOTAL, HIDDEN, device="cuda", dtype=torch.bfloat16)

    expected = dual_out_reference(h, w, N_BF16, N_GATE)
    actual = fused_front(h, w)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.5)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.02, atol=0.5)

    reference_us = graph_time_us(
        lambda: dual_out_reference(h, w, N_BF16, N_GATE),
        iters=iters,
        repeats=repeats,
    )
    kernel_us = graph_time_us(
        lambda: fused_front(h, w), iters=iters, repeats=repeats
    )
    check_regression(reference_us, kernel_us)
    return {"reference_us": reference_us, "kernel_us": kernel_us}


__all__ = [
    "AVAILABLE",
    "Config",
    "bench",
    "build_fused_front_weight",
    "dual_out_gemm",
    "dual_out_gemm_cutedsl",
    "fused_front",
    "fused_input_gemm_cutedsl",
]
