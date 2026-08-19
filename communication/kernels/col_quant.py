"""Column all-gather engines (Lamport one-shot / copy-engine / NCCL)
with an optional MXFP8-quantized write-out — the transport behind
``Collectives.all_gather_col_quant``. Moved here from
kimi_k3_layer/comm.py so the communication package is self-contained;
the FlashInferAllReduce class that used to live alongside is now in
``communication.backends.flashinfer_backend``.

One-shot single-kernel collectives over torch symmetric memory.

Small-message (decode) collectives for NVLink-connected single-node TP.
NCCL launches ring/tree pipelines that cost 10-20 us regardless of
size; flashinfer's Lamport oneshot AR shows a single SM kernel does the
same job in ~5 us. This module provides the two collectives flashinfer
does NOT ship - all-gather and reduce-scatter - built the same way, so
a shard -> AG (or RS -> shard) pipeline never pays NCCL latency or has
to fake the AG as a zero-padded allreduce. Full reductions go through
flashinfer's oneshot AR (``FlashInferAllReduce``): our own AR measured
6-8x behind it, so only the primitives that BEAT the alternatives are
kept.

All device code is CUDA (comm_cuda.py, JIT-built): the latency-
critical piece of a Lamport collective is the volatile sentinel poll,
which needs 128-bit vectorized loads (``ld.volatile.global.v4.u32`` + 8
bf16 sentinel checks per instruction). Triton prototypes of these
kernels existed but compiled the poll to scalar 16-bit loads and were
deleted in favor of the CUDA build; NCCL (torch.distributed) is the
correctness reference and the large-message fallback.

Protocol (flashinfer's Lamport scheme):

  * No barrier anywhere: consumers spin on the payload itself. bf16
    ``-0.0`` (0x8000) is the "not yet written" sentinel; producers
    canonicalize ``-0.0`` to ``+0.0`` so real payload never matches.
  * THREE rotating data slots per rank. A call pushes into slot
    ``n mod 3`` of every rank, re-sentinels slot ``(n-1) mod 3``
    (consumed by the previous call), then polls its own slot
    ``n mod 3``. Safety: rank A starts call n only after its call n-1
    poll saw every rank's n-1 data, so every rank already finished
    call n-2 - the slot being cleared can no longer be read, and the
    slot being written was re-sentineled one call ago everywhere.
  * Round counters (slot selection) are per-CTA device counters bumped
    inside the kernel, so CUDA-graph capture/replay of any call count
    stays correct with no host-side state.
  * ``reduce_scatter_cols`` keeps its own [flag, counter, clear_bytes]
    metadata (FlashInfer's layout, variable launch grid): an instance
    used for it must be DEDICATED to it - the rotation state is not
    shared with all_gather/reduce_scatter. (This used to be crammed
    into ``rounds[0:3]``, which corrupted per-CTA round counters of
    grid>=3 instances: CTA 2 rotated to a slot no peer wrote and its
    poll span forever - the "tune() hang" in the trt-dev image.)

All ranks must issue the same sequence of calls on a given instance.
World size 1 degrades to plain local ops.

Two implementation families live here:

  * ``OneShotComm`` - single-kernel Lamport collectives (above); the
    latency winners at decode sizes.
  * ``CeComm`` - copy-engine (DMA) collectives: cudaMemcpy2DAsync peer
    copies + ONE tiny arrival-sync kernel. ZERO SMs on the data path,
    no sentinel clear, so it takes over at large messages and overlaps
    concurrent compute for free (graph mode; see class docstring).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class OneShotComm:
    """Symmetric-memory workspace + one-shot bf16 collectives.

    ``max_bytes`` is the largest gathered/reduced message; ``grid`` is
    the CTA count of the AG/RS launches (per-CTA round counters, so it
    is fixed per instance - tune across instances). All methods need
    16B-aligned rows: ``cols`` and the row stride multiples of 8 bf16
    elements (CUDA build does 128-bit pushes/polls).
    """

    def __init__(self, rank: int, world: int, *,
                 max_bytes: int = 8 << 20, grid: int = 8,
                 group=None):
        self.rank = rank
        self.world = world
        self.grid = grid
        if world == 1:
            return
        self.data_off = 256
        self.slot_bytes = (max_bytes + 255) // 256 * 256
        self.buf = symm_mem.empty(
            self.data_off + 3 * self.slot_bytes, dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        group = group if group is not None else dist.group.WORLD
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.buf[:self.data_off].zero_()
        # whole data region starts sentineled (bf16 -0.0)
        self.buf[self.data_off:].view(torch.int16).fill_(-32768)
        self.buf_ptrs = torch.tensor(
            self.hdl.buffer_ptrs, dtype=torch.int64, device="cuda")
        # per-CTA round counters (slot rotation for AG/RS)
        self.rounds = torch.zeros(grid, dtype=torch.int32, device="cuda")
        # per-CTA x per-slot record of vectors written (so the
        # re-sentinel pass clears the previous MESSAGE, not the whole
        # slot capacity); slots start sentineled -> nothing to clear
        self.clear_hist = torch.zeros(grid, 3, dtype=torch.int32,
                                      device="cuda")
        # rs_cols FI-style metadata [flag, counter, clear_bytes]
        self.meta = torch.zeros(3, dtype=torch.int32, device="cuda")
        self.meta[2] = self.slot_bytes
        torch.cuda.synchronize()
        dist.barrier()

    def all_gather(self, x: torch.Tensor, *,
                   block: int = 256) -> torch.Tensor:
        """Gather [rows, cols] shards along COLUMNS -> a regular
        [rows, world*cols] tensor (the feature-dim gather NCCL AG would
        need a transpose for; rows=1 recovers the flat dim0 concat).
        ``x`` may be row-strided."""
        rows, cols = x.shape
        if self.world == 1:
            return x
        assert rows * cols * self.world * 2 <= self.slot_bytes
        from .col_quant_cuda import get_module

        return get_module().ag_lamport(
            self.buf_ptrs, self.rounds, self.clear_hist, x,
            self.data_off, self.slot_bytes, self.rank, self.world,
            block,
        )

    def all_gather_mxfp8(self, x: torch.Tensor, *,
                         block: int = 256) -> tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Column AG with the MXFP8 activation quantize fused on the
        write-out: [rows, cols] bf16 shards -> ([rows, world*cols]
        e4m3, [rows, world*cols/32] ue8m0 scales), bit-exact with
        ``torch.ops.trtllm.mxfp8_quantize(ag(x), False)``. Transport
        is unchanged bf16 - the fusion kills the separate quantize
        kernel that would otherwise sit on the critical path."""
        rows, cols = x.shape
        assert self.world > 1, "no transport at TP1; quantize locally"
        assert rows * cols * self.world * 2 <= self.slot_bytes
        from .col_quant_cuda import get_module

        return get_module().ag_mxfp8(
            self.buf_ptrs, self.rounds, self.clear_hist, x,
            self.data_off, self.slot_bytes, self.rank, self.world,
            block,
        )

    def reduce_scatter(self, x: torch.Tensor, *,
                       block: int = 256) -> torch.Tensor:
        """Sum over ranks, return my flat dim0 1/world chunk. The
        sentinel poll IS the synchronization: each rank polls the WORLD
        rows of its chunk and accumulates fp32 as they land."""
        x = x.reshape(-1)
        if self.world == 1:
            return x
        assert x.numel() * 2 <= self.slot_bytes
        from .col_quant_cuda import get_module

        return get_module().rs_lamport(
            self.buf_ptrs, self.rounds, self.clear_hist, x,
            self.data_off, self.slot_bytes, self.rank, self.world,
            block,
        )

    def reduce_scatter_cols(self, x: torch.Tensor) -> torch.Tensor:
        """Column reduce-scatter: [B, C] partial sums on every rank ->
        my [B, C/world] column slice of the total (world x less wire
        than an AR). Forked from FlashInfer's oneshot Lamport AR (same
        push/clear/batched-poll structure, one CTA per token).
        REQUIRES a dedicated instance (own rotation metadata).

        (Fused-RMSNorm / deferred-scale / fused-finalize variants were
        REMOVED - all lost to the flashinfer fused AR+norm tail; see
        moe_optimization.md Appendix A.)"""
        rows, cols = x.shape
        if self.world == 1:
            raise ValueError("col-RS is a no-op at TP1; slice locally")
        assert rows * cols * 2 <= self.slot_bytes, "instance too small"
        from .col_quant_cuda import get_module

        return get_module().rs_cols(
            self.buf_ptrs, self.meta, x,
            self.data_off, self.slot_bytes, self.rank, self.world,
        )


class CeComm:
    """Copy-engine (DMA) collectives over symmetric memory.

    The data movement is host-issued ``cudaMemcpy2DAsync`` peer copies
    (push orientation: every rank only SENDS): the DMA engines move the
    bytes, ZERO SMs, so the transfer overlaps concurrent compute by
    construction and there is no Lamport re-sentinel cost (which is
    what makes the one-shot kernels lose past ~1k tokens).

    SM cost per collective: ONE tiny sync kernel AFTER the copies
    (publish "my pushes are done" - stream order makes that true -
    then wait for every peer's publication). There is no pre-copy
    barrier: my round r+1 copies are stream-ordered after my round-r
    consumer, and all ranks run the same pipeline, so nobody can push
    round r+1 into my slot before my consumer read round r. This
    REQUIRES (a) consumers of the returned view to run on the caller's
    current stream before the next call on the instance, and (b) all
    ranks issuing the same sequence of calls with comparable work in
    between (true for graph-replayed TP pipelines).

    Copies serialize on one P2P engine (~530 GB/s; measured: extra
    streams do NOT parallelize them, so ``streams`` is tuning-only),
    and each copy has ~2 us of engine setup - at tiny messages the
    single-kernel Lamport path is faster. Use CUDA graphs: eager-mode
    ``cudaMemcpy2DAsync`` host calls cost ~10 us each with peer
    pointers; as graph memcpy nodes they are free.

    No alignment requirements (memcpy is byte-granular).
    """

    def __init__(self, rank: int, world: int, *,
                 max_bytes: int = 8 << 20, streams: int = 0,
                 sync_threads: int = 32, group=None):
        self.rank = rank
        self.world = world
        self.nstreams = streams
        self.sync_threads = sync_threads
        if world == 1:
            return
        self.data_off = 256
        self.slot_bytes = (max_bytes + 255) // 256 * 256
        self.buf = symm_mem.empty(
            self.data_off + self.slot_bytes, dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        group = group if group is not None else dist.group.WORLD
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.buf[:self.data_off].zero_()  # arrival flags start at 0
        self.ptrs = [int(p) for p in self.hdl.buffer_ptrs]
        self.buf_ptrs = torch.tensor(
            self.hdl.buffer_ptrs, dtype=torch.int64, device="cuda")
        # host-side pointer table + C++ whole-collective fast path
        # (B10_CE_CPP=0 falls back to the Python reference loops)
        self._ptrs_cpu = torch.tensor(self.ptrs, dtype=torch.int64)
        self._cpp = os.environ.get("B10_CE_CPP", "1") == "1"
        # monotonic round (device-side -> graph replay safe); starts at
        # 1 so the first wait cannot pass on the zeroed flags
        self.round_buf = torch.ones(1, dtype=torch.int32, device="cuda")
        self.streams = [torch.cuda.Stream(priority=-1)
                        for _ in range(streams)]
        self._fork = torch.cuda.Event()
        self._joins = [torch.cuda.Event() for _ in self.streams]
        torch.cuda.synchronize()
        dist.barrier()

    def _sync(self):
        from .col_quant_cuda import get_module

        get_module().ce_sync(
            self.buf_ptrs, self.round_buf, 0, self.rank, self.world,
            self.sync_threads)

    def _copies(self, jobs):
        """Issue (dst, dpitch, src, spitch, width, height) copies,
        fanned out over the side streams when configured. Copies whose
        destination is THIS rank's buffer go through a small SM copy
        kernel: a same-device cudaMemcpy2DAsync runs on the driver's SM
        copy path and does not co-schedule with a saturating kernel
        (it queues until concurrent compute drains, stalling the
        arrival sync), while a regular kernel slots in as blocks
        retire. Peer copies stay on the DMA engines (zero SMs)."""
        from .col_quant_cuda import get_module

        mod = get_module()
        lo = self.ptrs[self.rank]
        hi = lo + self.data_off + self.slot_bytes

        def issue(j):
            (mod.ce_local2d if lo <= j[0] < hi else mod.ce_copy2d)(*j)

        if not self.streams:
            # one pybind crossing for the whole copy set (eager host
            # cost of the per-peer loop is ~4us/call; see comm_cuda)
            mod.ce_copy2d_batch(
                torch.tensor([v for j in jobs for v in j],
                             dtype=torch.int64), lo, hi)
            return
        cur = torch.cuda.current_stream()
        self._fork.record(cur)
        for s in self.streams:
            s.wait_event(self._fork)
        lanes = [cur] + self.streams
        for i, j in enumerate(jobs):
            with torch.cuda.stream(lanes[i % len(lanes)]):
                issue(j)
        for s, evt in zip(self.streams, self._joins):
            evt.record(s)
            cur.wait_event(evt)

    def _slot_view(self, nbytes: int) -> torch.Tensor:
        return self.buf[self.data_off:self.data_off + nbytes].view(
            torch.bfloat16)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Column AG: [rows, cols] shards -> [rows, world*cols]. The
        result is a VIEW of this instance's slot (stable address across
        graph replays); it is valid until the next call on the
        instance. ``x`` may be row-strided."""
        rows, cols = x.shape
        if self.world == 1:
            return x
        out_bytes = rows * self.world * cols * 2
        assert out_bytes <= self.slot_bytes, "instance too small"
        assert x.stride(1) == 1
        if self._cpp and not self.streams:
            from .col_quant_cuda import get_module
            return get_module().ce_ag_cols(
                self._ptrs_cpu, self.buf_ptrs, self.round_buf, x,
                self.data_off, self.slot_bytes, self.rank, self.world,
                self.sync_threads)
        row_bytes = cols * 2
        dpitch = self.world * row_bytes
        jobs = []
        for i in range(self.world):
            r = (self.rank + 1 + i) % self.world  # self last
            jobs.append((
                self.ptrs[r] + self.data_off + self.rank * row_bytes,
                dpitch, x.data_ptr(), x.stride(0) * 2, row_bytes, rows,
            ))
        self._copies(jobs)
        self._sync()
        return self._slot_view(out_bytes).view(rows, self.world * cols)

    def reduce_scatter_cols(self, x: torch.Tensor) -> torch.Tensor:
        """Column RS: [rows, world*C] partial sums on every rank -> my
        [rows, C] column slice of the total. Copies push column block r
        to rank r as a contiguous slab; the only SM work is the final
        slab sum (fp32 accumulation). No fused norm (use the Lamport
        ``reduce_scatter_cols`` when the norm must be fused)."""
        rows, cols = x.shape
        if self.world == 1:
            raise ValueError("col-RS is a no-op at TP1; slice locally")
        assert cols % self.world == 0 and x.stride(1) == 1
        chunk = cols // self.world
        slab_bytes = rows * chunk * 2
        assert self.world * slab_bytes <= self.slot_bytes
        jobs = []
        for i in range(self.world):
            r = (self.rank + 1 + i) % self.world
            jobs.append((
                self.ptrs[r] + self.data_off + self.rank * slab_bytes,
                chunk * 2, x.data_ptr() + r * chunk * 2,
                x.stride(0) * 2, chunk * 2, rows,
            ))
        self._copies(jobs)
        self._sync()
        slabs = self._slot_view(self.world * slab_bytes).view(
            self.world, rows, chunk)
        return slabs.sum(0)  # opmath fp32 accumulation, bf16 out

    # -- row (token/sequence-parallel) collectives ---------------------
    # Row slabs of a contiguous [B, W] tensor are contiguous, so each
    # peer push is ONE flat copy - even cheaper than the cols variants
    # (no pitch interleave). Requires B % world == 0 (pad tokens at the
    # layer entry).

    def push_rows(self, x: torch.Tensor):
        """All-to-all half of a row RS: push row-block r of the [B, W]
        partial to rank r's slot at offset rank*slab. Split from the
        reduce so the copies can be issued early (e.g. on a side
        stream, overlapping compute) and reduced later. Every rank must
        call ``reduce_rows`` (which syncs) before reusing the slot."""
        rows, cols = x.shape
        assert rows % self.world == 0 and x.stride(1) == 1
        chunk = rows // self.world
        slab_bytes = chunk * cols * 2
        assert self.world * slab_bytes <= self.slot_bytes
        if self._cpp and not self.streams:
            from .col_quant_cuda import get_module
            get_module().ce_push_rows(
                self._ptrs_cpu, x, self.data_off, self.slot_bytes,
                self.rank, self.world)
            return chunk, cols, slab_bytes
        jobs = []
        for i in range(self.world):
            r = (self.rank + 1 + i) % self.world
            jobs.append((
                self.ptrs[r] + self.data_off + self.rank * slab_bytes,
                slab_bytes, x.data_ptr() + r * slab_bytes,
                slab_bytes, slab_bytes, 1,
            ))
        self._copies(jobs)
        return chunk, cols, slab_bytes

    def reduce_rows(self, chunk: int, cols: int,
                    slab_bytes: int) -> torch.Tensor:
        """Sync + local fp32-accumulated sum of the world slabs pushed
        by ``push_rows``: my reduced [B/world, W] row slice."""
        if self._cpp and not self.streams:
            from .col_quant_cuda import get_module
            return get_module().ce_reduce_rows(
                self._ptrs_cpu, self.buf_ptrs, self.round_buf,
                self.data_off, self.rank, self.world, self.sync_threads,
                chunk, cols)
        self._sync()
        slabs = self._slot_view(self.world * slab_bytes).view(
            self.world, chunk, cols)
        return slabs.sum(0)

    def reduce_scatter_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Row RS: [B, W] partial sums on every rank -> my reduced
        [B/world, W] row slice."""
        if self.world == 1:
            return x
        return self.reduce_rows(*self.push_rows(x))

    def all_to_all_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Row A2A: chunk d of my [B, W] goes to rank d -> my received
        [B, W] (source-major). push_rows already IS the A2A data movement
        (slab d of x lands in rank d's slot at position [my rank]); this
        just adds the sync and views the received slabs. The result is a
        view of this instance's slot, valid until the next call."""
        if self.world == 1:
            return x
        chunk, cols, _slab = self.push_rows(x)
        self._sync()
        return self._slot_view(x.shape[0] * cols * 2).view(x.shape[0], cols)

    def all_gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Row AG: my [B/world, W] row slice -> full [B, W]. The result
        is a view of this instance's slot, valid until the next call on
        the instance."""
        if self.world == 1:
            return x
        rows, cols = x.shape
        slab_bytes = rows * cols * 2
        out_bytes = self.world * slab_bytes
        assert out_bytes <= self.slot_bytes
        assert x.stride(1) == 1
        if self._cpp and not self.streams:
            from .col_quant_cuda import get_module
            return get_module().ce_ag_rows(
                self._ptrs_cpu, self.buf_ptrs, self.round_buf, x,
                self.data_off, self.slot_bytes, self.rank, self.world,
                self.sync_threads)
        jobs = []
        for i in range(self.world):
            r = (self.rank + 1 + i) % self.world  # self last
            jobs.append((
                self.ptrs[r] + self.data_off + self.rank * slab_bytes,
                slab_bytes, x.data_ptr(), slab_bytes, slab_bytes, 1,
            ))
        self._copies(jobs)
        self._sync()
        return self._slot_view(out_bytes).view(
            self.world * rows, cols)


def nccl_all_gather_cols(x: torch.Tensor) -> torch.Tensor:
    """NCCL reference for the column AG (correctness baseline and
    large-message fallback): dim0 gather + transpose-copy."""
    world = dist.get_world_size()
    rows, cols = x.shape
    out = torch.empty(world, rows, cols, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out.view(world * rows, cols),
                                x.contiguous())
    return out.permute(1, 0, 2).reshape(rows, world * cols)


class TunedAllGather:
    """Size-keyed autotune over one-shot instances (grid sizes x launch
    blocks) plus the NCCL fallback. Tune once per shape OUTSIDE graph
    capture; replays then always dispatch the winning launch."""

    def __init__(self, comms: list[OneShotComm],
                 ces: list[CeComm] | None = None):
        self.comms = comms
        self.ces = ces or []
        self.world = comms[0].world
        self._plans: dict[tuple, tuple] = {}

    def _candidates(self, numel_out: int):
        for ci, comm in enumerate(self.comms):
            if numel_out > comm.slot_bytes // 2:
                continue
            for blk in (128, 256, 512):
                yield ("oneshot", ci, blk)
        for ci, ce in enumerate(self.ces):
            if numel_out * 2 <= ce.slot_bytes:
                yield ("ce", ci, 0)
        yield ("nccl", 0, 0)

    def _run(self, plan, x):
        kind, ci, blk = plan
        if kind == "nccl":
            return nccl_all_gather_cols(x)
        if kind == "ce":
            return self.ces[ci].all_gather(x)
        return self.comms[ci].all_gather(x, block=blk)

    def all_gather_mxfp8(self, x: torch.Tensor) -> tuple[torch.Tensor,
                                                         torch.Tensor]:
        """AG with the MXFP8 quantize fused on the write-out (decode
        path); tuned across one-shot instances like plain AG (the
        instance GRID matters: at [64, 3584] grid 32 runs the fused AG
        in 7.6 us where grid 8 took 12.5 - the poll+quantize pass is
        grid-starved on a small instance)."""
        key = ("q",) + tuple(x.shape)
        plan = self._plans.get(key)
        if plan is None:
            plans = [("oneshot", ci, blk)
                     for ci in range(len(self.comms))
                     for blk in (256, 512)]
            plan = self._tune_plans(key, plans, lambda p: self._run_q(p, x))
        return self._run_q(plan, x)

    def _run_q(self, plan, x):
        _, ci, blk = plan
        return self.comms[ci].all_gather_mxfp8(x, block=blk)

    def tune(self, x: torch.Tensor, *, iters: int = 30) -> tuple:
        key = tuple(x.shape)
        if key in self._plans or self.world == 1:
            return self._plans.get(key, ("local", 0, 0))
        numel_out = x.numel() * self.world
        return self._tune_plans(
            key, list(self._candidates(numel_out)),
            lambda p: self._run(p, x), iters=iters)

    def _tune_plans(self, key, plans, run, *, iters: int = 30) -> tuple:
        best, best_t = None, float("inf")
        for plan in plans:
            for _ in range(3):
                run(plan)
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                run(plan)
            end.record()
            end.synchronize()
            t = start.elapsed_time(end) / iters
            t_all = torch.tensor([t], device="cuda", dtype=torch.float64)
            dist.all_reduce(t_all, op=dist.ReduceOp.MAX)
            t = t_all.item()
            if t < best_t:
                best, best_t = plan, t
        self._plans[key] = best
        return best

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return x
        plan = self._plans.get(tuple(x.shape))
        if plan is None:
            plan = self.tune(x)
        return self._run(plan, x)


def build_tuned_all_gather(
    rank: int,
    world: int,
    max_rows: int,
    max_output_columns: int,
) -> TunedAllGather:
    """Build launch variants for a bounded column all-gather."""
    max_bytes = max_rows * max_output_columns * 2
    comms = [OneShotComm(rank, world, max_bytes=max_bytes, grid=8)]
    if max_rows >= 8:
        # bigger grid for the poll/copy(/quantize) drain: at [64, 3584]
        # grid 32 beats grid 8 by ~7 us (plain) / ~5 us (fused mxfp8)
        comms.append(
            OneShotComm(rank, world, max_bytes=max_bytes, grid=32)
        )
    if max_rows >= 32:
        comms.append(
            OneShotComm(rank, world, max_bytes=max_bytes, grid=64)
        )
    # copy-engine candidate (SM-free; wins past the Lamport clear-cost
    # crossover and whenever the caller wants compute overlap)
    ces = [CeComm(rank, world, max_bytes=max_bytes)]
    return TunedAllGather(comms, ces)
