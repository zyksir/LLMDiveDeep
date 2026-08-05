"""Vendored from CuTeDSLGen run gen_tailgemm_b (best/kernel.py),
plus a per-call CUDA stream (CUDA-graph capturable) and the
make_config/moe_tail_gemm glue from its run.py. D = C + A @ W.T,
bf16, fp32 accumulate; generated + verified at [16,7168]x[3584].
"""
# ===========================================================================
# blackwell_fp16_moe_tail_gemm -- MoE tail projection  D = C + A @ W.T
#
#   A: [M, K] bf16 row-major   (M = decode batch, tiny)
#   W: [N, K] bf16 row-major   (used transposed)
#   C: [M, N] bf16 row-major
#   D: [M, N] bf16 row-major, fp32 accumulate
#
# MEMORY BOUND: with M tiny, streaming W (2*N*K bytes) at HBM bandwidth is the
# whole game.  See design.md for the roofline and the role-swap that maps the
# large N dimension onto the tcgen05 MMA-M axis.
#
# Target: Blackwell B200 (sm_100a), nvidia-cutlass-dsl 4.5.2.
# ===========================================================================

from functools import lru_cache

import cuda.bindings.driver as cuda_drv

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

IO_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32

# Warp roles (6 warps / 192 threads).
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP_ID = 4
TMA_WARP_ID = 5
NUM_EPI_THREADS = 32 * len(EPI_WARPS)
THREADS_PER_CTA = 32 * (len(EPI_WARPS) + 2)

_EPI_BAR_ID = 1
_TMEM_BAR_ID = 2

# sm_100 shared memory capacity available to a CTA.
SMEM_CAPACITY = 227 * 1024


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


class Config:
    """Compile-time kernel configuration for one (M, N, K) problem."""

    def __init__(self, m: int, n: int, k: int, tile_nr: int, tile_k: int,
                 ab_stages: int | None = None, num_acc: int = 1):
        assert tile_nr in (64, 128), "tcgen05 MMA-M must be 64 or 128"
        assert n % tile_nr == 0, f"N={n} must be a multiple of TILE_NR={tile_nr}"
        assert k % tile_k == 0, f"K={k} must be a multiple of TILE_K={tile_k}"
        assert 1 <= m <= 256
        self.m = m
        self.n = n
        self.k = k
        self.tile_nr = tile_nr
        self.tile_k = tile_k
        self.mma_n = _round_up(m, 8)
        # bytes of SMEM per pipeline stage: W tile + A tile
        self.stage_bytes = (tile_nr + self.mma_n) * tile_k * 2
        if ab_stages is None:
            budget = SMEM_CAPACITY - 2048  # leave room for mbarriers/alignment
            ab_stages = max(2, min(budget // self.stage_bytes, k // tile_k))
        self.ab_stages = ab_stages
        num_k_blocks = tile_k // 16
        assert num_k_blocks % num_acc == 0, (
            f"TILE_K/16={num_k_blocks} must be a multiple of num_acc={num_acc}")
        self.num_acc = num_acc
        # TMEM allocation granularity: power of two, >= 32 columns.
        cols = num_acc * self.mma_n
        self.tmem_cols = max(32, 1 << (cols - 1).bit_length())
        assert self.tmem_cols <= 512, "accumulators exceed TMEM capacity"

    def key(self):
        return (self.m, self.n, self.k, self.tile_nr, self.tile_k, self.ab_stages,
                self.num_acc)

    def __repr__(self):
        return (f"Config(M={self.m}, N={self.n}, K={self.k}, TILE_NR={self.tile_nr}, "
                f"TILE_K={self.tile_k}, MMA_N={self.mma_n}, stages={self.ab_stages}, "
                f"acc={self.num_acc}, "
                f"smem={self.ab_stages * self.stage_bytes / 1024:.1f}KiB, "
                f"grid={self.n // self.tile_nr})")


def _make_shared_storage(ab_stages: int):
    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_dealloc_mbar: cutlass.Int64
        tmem_holding_buf: cutlass.Int32

    return SharedStorage


@cute.kernel
def blackwell_fp16_moe_tail_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_w: cute.CopyAtom,
    mW: cute.Tensor,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    mCt: cute.Tensor,
    mDt: cute.Tensor,
    w_smem_layout: cute.ComposedLayout,
    a_smem_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
    copy_atom_t2r: cute.CopyAtom,
    storage_ty: cutlass.Constexpr,
    tile_nr: cutlass.Constexpr,
    mma_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    num_k_tiles: cutlass.Constexpr,
    m_size: cutlass.Constexpr,
    num_acc: cutlass.Constexpr,
    tmem_cols: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()

    if warp_idx == TMA_WARP_ID:
        cpasync.prefetch_descriptor(tma_atom_w)
        cpasync.prefetch_descriptor(tma_atom_a)

    # Cluster is (1,1,1): both masks fold to the single self bit, but building
    # them through the helper keeps the atoms' multicast contract satisfied.
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())
    mcast_mask_w = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2
    )
    mcast_mask_a = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_coord_vmnk, mcast_mode=1
    )

    smem = utils.SmemAllocator()
    storage = smem.allocate(storage_ty)

    epi_barrier = pipeline.NamedBarrier(
        barrier_id=_EPI_BAR_ID, num_threads=NUM_EPI_THREADS
    )
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_BAR_ID, num_threads=32 * (1 + len(EPI_WARPS))
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=EPI_WARPS[0],
        is_two_cta=False,
    )

    num_tma_bytes = cute.size_in_bytes(
        IO_DTYPE, cute.select(w_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=cute.size(w_smem_layout, mode=[3]),
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
        tx_count=num_tma_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=len(EPI_WARPS)
        ),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

    sW = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=w_smem_layout.outer,
        byte_alignment=128,
        swizzle=w_smem_layout.inner,
    )
    sA = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )

    mma_tiler = (tile_nr, mma_n, tile_k)
    gW = cute.local_tile(mW, cute.slice_(mma_tiler, (None, 0, None)), (None, None))
    gA = cute.local_tile(mA, cute.slice_(mma_tiler, (0, None, None)), (None, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgW = thr_mma.partition_A(gW)
    tCgA = thr_mma.partition_B(gA)

    tCrW = tiled_mma.make_fragment_A(sW)
    tCrA = tiled_mma.make_fragment_B(sA)

    # `num_acc` INDEPENDENT TMEM accumulators.  With MMA-N = 16 a single
    # tcgen05.mma carries almost no work, so back-to-back MMAs into one
    # accumulator serialise on the accumulator RAW latency (~70 cycles for
    # MMA-M=64).  Round-robining the k-blocks over several accumulators turns
    # that latency into ILP; the partials are summed once in the epilogue.
    acc_shape = tiled_mma.partition_shape_C((tile_nr, mma_n))
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc))

    tWsW, tWgW = cpasync.tma_partition(
        tma_atom_w, 0, cute.make_layout(1),
        cute.group_modes(sW, 0, 3), cute.group_modes(tCgW, 0, 3),
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )

    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # ------------------------------------------------------------------
    # TMA warp: stream the W tile rows + the (tiny) A tile for every k-tile
    # ------------------------------------------------------------------
    if warp_idx == TMA_WARP_ID:
        tWgW_slice = tWgW[(None, bidx, None)]
        tAgA_slice = tAgA[(None, 0, None)]
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            handle = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, k_tile)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=mcast_mask_w,
            )
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, k_tile)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=mcast_mask_a,
            )
        ab_producer.tail()

    # ------------------------------------------------------------------
    # MMA warp
    # ------------------------------------------------------------------
    elif warp_idx == MMA_WARP_ID:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        acc_empty = acc_producer.acquire_and_advance()
        num_k_blocks = cute.size(tCrW, mode=[2])

        # Peel k-tile 0 so the first touch of each accumulator initialises it
        # (ACCUMULATE=False) instead of needing a runtime branch in the loop.
        handle = ab_consumer.wait_and_advance()
        for kb in cutlass.range_constexpr(num_k_blocks):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kb >= num_acc)
            acc_j = tCtAcc[(None, None, None, kb % num_acc)]
            coord = (None, None, kb, handle.index)
            cute.gemm(tiled_mma, acc_j, tCrW[coord], tCrA[coord], acc_j)
        handle.release()

        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_tile in cutlass.range(num_k_tiles - 1, unroll=1):
            handle = ab_consumer.wait_and_advance()
            for kb in cutlass.range_constexpr(num_k_blocks):
                acc_j = tCtAcc[(None, None, None, kb % num_acc)]
                coord = (None, None, kb, handle.index)
                cute.gemm(tiled_mma, acc_j, tCrW[coord], tCrA[coord], acc_j)
            handle.release()
        acc_empty.commit()
        acc_producer.tail()

    # ------------------------------------------------------------------
    # Epilogue warps: TMEM -> RMEM, add C, convert, store D
    # ------------------------------------------------------------------
    elif warp_idx < MMA_WARP_ID:
        tmem.allocate(tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tAcc = tCtAcc[((None, None), 0, 0, 0)]

        gCt = cute.local_tile(mCt, (tile_nr, mma_n), (bidx, 0))
        gDt = cute.local_tile(mDt, (tile_nr, mma_n), (bidx, 0))

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
        thr_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_gC = thr_t2r.partition_D(gCt)
        tTR_gD = thr_t2r.partition_D(gDt)

        acc_full = acc_consumer.wait_and_advance()

        # Sum the independent accumulator chains.
        rAcc = cute.make_rmem_tensor(tTR_gD.shape, ACC_DTYPE)
        cute.copy(tiled_copy_t2r, thr_t2r.partition_S(tAcc), rAcc)
        for j in cutlass.range_constexpr(1, num_acc):
            rPart = cute.make_rmem_tensor(tTR_gD.shape, ACC_DTYPE)
            cute.copy(
                tiled_copy_t2r,
                thr_t2r.partition_S(tCtAcc[((None, None), 0, 0, j)]),
                rPart,
            )
            rAcc.store(rAcc.load() + rPart.load())

        rC = cute.make_rmem_tensor(tTR_gD.shape, IO_DTYPE)
        if cutlass.const_expr(m_size == mma_n):
            cute.autovec_copy(tTR_gC, rC)
            rD = cute.make_rmem_tensor(tTR_gD.shape, IO_DTYPE)
            rD.store((rAcc.load() + rC.load().to(ACC_DTYPE)).to(IO_DTYPE))
            cute.autovec_copy(rD, tTR_gD)
        else:
            # M is not a multiple of 8: the MMA tile over-hangs C/D in the
            # column (batch) direction, so predicate on the real M.
            cId = cute.make_identity_tensor((tile_nr, mma_n))
            tTR_cId = thr_t2r.partition_D(cId)
            for i in cutlass.range_constexpr(cute.size(rAcc)):
                if tTR_cId[i][1] < m_size:
                    tTR_gD[i] = (
                        rAcc[i] + tTR_gC[i].to(ACC_DTYPE)
                    ).to(IO_DTYPE)

        with cute.arch.elect_one():
            acc_full.release()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def _host(
    mW: cute.Tensor,
    mA: cute.Tensor,
    mCt: cute.Tensor,
    mDt: cute.Tensor,
    tile_nr: cutlass.Constexpr,
    mma_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    ab_stages: cutlass.Constexpr,
    num_k_tiles: cutlass.Constexpr,
    m_size: cutlass.Constexpr,
    grid_n: cutlass.Constexpr,
    storage_ty: cutlass.Constexpr,
    num_acc: cutlass.Constexpr,
    tmem_cols: cutlass.Constexpr,
    stream: cuda_drv.CUstream,
):
    mma_tiler = (tile_nr, mma_n, tile_k)
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        (tile_nr, mma_n, 16),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    w_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler, IO_DTYPE, ab_stages
    )
    a_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, IO_DTYPE, ab_stages
    )

    cta_layout_mnk = cute.make_layout((1, 1, 1))
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    copy_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
        copy_op,
        mW,
        cute.slice_(w_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_B(
        copy_op,
        mA,
        cute.slice_(a_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        (tile_nr, mma_n, tile_k),
        utils.LayoutEnum.from_tensor(mDt),
        IO_DTYPE,
        ACC_DTYPE,
        (tile_nr, mma_n),
        False,
    )

    blackwell_fp16_moe_tail_gemm_kernel(
        tiled_mma,
        tma_atom_w,
        tma_tensor_w,
        tma_atom_a,
        tma_tensor_a,
        mCt,
        mDt,
        w_smem_layout,
        a_smem_layout,
        cta_layout_vmnk,
        copy_atom_t2r,
        storage_ty,
        tile_nr,
        mma_n,
        tile_k,
        num_k_tiles,
        m_size,
        num_acc,
        tmem_cols,
    ).launch(
        grid=(grid_n, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Host-side glue
# ---------------------------------------------------------------------------


def wrap_tensors(w, a, c, d):
    """Wrap the four live torch tensors as CuTe tensors (zero copy)."""
    mW = from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mA = from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    # C/D are [M, N] row-major; the kernel wants the [N, M] transposed view,
    # which is a pure reinterpretation (leading dim 0 is contiguous).
    mCt = from_dlpack(c.t(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    mDt = from_dlpack(d.t(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    return mW, mA, mCt, mDt


@lru_cache(maxsize=32)
def _compile(cfg_key, m, n, k, tile_nr, tile_k, ab_stages, num_acc):
    import torch

    cfg = Config(m, n, k, tile_nr, tile_k, ab_stages, num_acc)
    dev = "cuda"
    w = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    a = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    c = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    d = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    mW, mA, mCt, mDt = wrap_tensors(w, a, c, d)

    import torch as _t
    _specimen = cuda_drv.CUstream(
        _t.cuda.current_stream().cuda_stream)
    return cute.compile(
        _host,
        mW,
        mA,
        mCt,
        mDt,
        cfg.tile_nr,
        cfg.mma_n,
        cfg.tile_k,
        cfg.ab_stages,
        cfg.k // cfg.tile_k,
        cfg.m,
        cfg.n // cfg.tile_nr,
        _make_shared_storage(cfg.ab_stages),
        cfg.num_acc,
        cfg.tmem_cols,
        _specimen,
        options="--gpu-arch sm_100a --ptxas-options '--opt-level=3'",
    )


def get_compiled(cfg: Config):
    return _compile(cfg.key(), cfg.m, cfg.n, cfg.k, cfg.tile_nr, cfg.tile_k,
                    cfg.ab_stages, cfg.num_acc)


DEFAULT_TILE_NR = 64
DEFAULT_TILE_K = 128
DEFAULT_STAGES = 6


def make_config(m, n, k, tile_nr=None, tile_k=None, stages=None):
    return Config(
        m, n, k,
        tile_nr=DEFAULT_TILE_NR if tile_nr is None else tile_nr,
        tile_k=DEFAULT_TILE_K if tile_k is None else tile_k,
        ab_stages=DEFAULT_STAGES if stages is None else stages,
    )



_raw_stream = None


def moe_tail_gemm(a, w, c, out=None, cfg=None):
    """D = C + A @ W.T on the current torch stream (graph-safe)."""
    global _raw_stream
    if _raw_stream is None:
        import torch as _t
        _raw_stream = _t._C._cuda_getCurrentRawStream
    m, k = a.shape
    n = w.shape[0]
    if cfg is None:
        cfg = make_config(m, n, k)
    if out is None:
        import torch as _t
        out = _t.empty((m, n), device=a.device, dtype=a.dtype)
    compiled = get_compiled(cfg)
    mW, mA, mCt, mDt = wrap_tensors(w, a, c, out)
    compiled(mW, mA, mCt, mDt,
             cuda_drv.CUstream(_raw_stream(a.device.index)))
    return out
