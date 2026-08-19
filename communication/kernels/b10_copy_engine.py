"""``b10_copy_engine`` backend: our LowContentionComm movers.

Rotated-schedule unicast collectives over a shared symm buffer, in two
mover flavors:

    b10_copy_engine      DMA copy engines (zero SMs on the data path —
                         never contends with concurrent compute)
    b10_copy_engine:sm   the same schedule with an SM copy-kernel
                         mover (wins large all_to_all)

Each mover pre-allocates two ``max_numel`` symm buffers, so they are
built LAZILY on first use (collective — every rank must reach the
first call together, true for explicit ``impl=`` requests issued in
TP lockstep).

The transport below (``LowContentionComm``, formerly
communication/low_contention_comm.py) and the dispatcher-facing
wrapper (``B10CopyEngineBackend``) live in this one file.

Low-contention symmetric-memory collectives: all_gather / all_to_all
(push- and pull-based), reduce_scatter as all_to_all + local reduce, and
the composed 2-shot all_reduce.

Same idea as the copy-engine family in ``kimi_k3_layer/comm.py`` — a
rotated peer schedule over symmetric memory — with a SELECTABLE data
mover and a C++ host path (``LOW_CONTENTION_CPP=0`` falls back to the
Python per-peer ``copy_`` loop):

* ``mover="dma"`` (default): one C++ call queues cudaMemcpyAsync peer
  copies — ZERO SMs, so the collective never contends with compute (the
  other half of "low contention"); single-engine cap ~530 GB/s.
* ``mover="sm"``: one fused copy kernel issues every peer copy, looping
  peers in the rotated order so all blocks sweep destinations in loose
  lockstep (concurrent-destination variants measured 20-50% slower);
  ~630 GB/s at large messages, costs ~SM-copy occupancy.

Either C++ path removes the per-peer ``get_buffer`` + ``copy_`` Python
cost (~15-25 us/call) that made CeComm win at small messages.

The rotated schedule (the load-bearing idea):
* push: at step ``i`` every rank WRITES to ``(rank + 1 + i) % world`` —
  no receiver's inbound link ever has two writers in a step;
* pull: at step ``i`` every rank READS from ``(rank + 1 + i) % world`` —
  no source's outbound link ever has two readers in a step.

reduce_scatter is NOT a separate implementation: the data movement of a
row-RS is exactly an all_to_all (chunk d of my partials -> rank d), so it
calls ``all_to_all`` and reduces the received slabs (fp32 accumulation).
all_reduce = reduce_scatter + all_gather (ring-shaped wire).

For host-cost-free timing/production, capture calls in a CUDA graph: all
paths (copies, barriers) are capture-safe; the remaining eager host cost
is the torch symm-mem dispatch itself.

Correctness contract: all ranks issue the same sequence of calls on one
instance; consumers of a returned SYMM view (push all_gather/all_to_all)
must run before the instance's next call, or call ``wait()``. Pull ops
and reduce_scatter/all_reduce return regular local tensors.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import rendezvous

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

struct CopySet {
    int64_t dst[8];
    int64_t src[8];
    int32_t n;          // number of copies (<= 8)
};

__global__ void fused_rotated_copies(CopySet cs, int64_t n_vec) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < cs.n; ++j) {              // rotated peer order
        const uint4 *s = reinterpret_cast<const uint4 *>(cs.src[j]);
        uint4 *d = reinterpret_cast<uint4 *>(cs.dst[j]);
        for (int64_t i = i0; i < n_vec; i += stride) d[i] = s[i];
    }
}

void async_copies(torch::Tensor dst_ptrs, torch::Tensor src_ptrs,
                  int64_t bytes) {
    // copy-engine mover: zero SMs; the engine executes the queue in
    // order, so the rotated schedule is preserved. NOTE eager-mode
    // cudaMemcpyAsync with peer pointers costs ~10 us host each; under
    // CUDA graphs they become memcpy nodes and the host cost vanishes.
    TORCH_CHECK(dst_ptrs.numel() == src_ptrs.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *dp = dst_ptrs.data_ptr<int64_t>();
    auto *sp = src_ptrs.data_ptr<int64_t>();
    for (int j = 0; j < dst_ptrs.numel(); ++j) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(dp[j]),
            reinterpret_cast<const void *>(sp[j]),
            (size_t)bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void fused_copies(torch::Tensor dst_ptrs, torch::Tensor src_ptrs,
                  int64_t bytes) {
    TORCH_CHECK(dst_ptrs.numel() == src_ptrs.numel());
    TORCH_CHECK(dst_ptrs.numel() <= 8, "up to 8 copies per launch");
    TORCH_CHECK(bytes % 16 == 0, "16B-aligned payloads only");
    CopySet cs;
    cs.n = (int32_t)dst_ptrs.numel();
    auto *dp = dst_ptrs.data_ptr<int64_t>();
    auto *sp = src_ptrs.data_ptr<int64_t>();
    for (int j = 0; j < cs.n; ++j) { cs.dst[j] = dp[j]; cs.src[j] = sp[j]; }
    const int64_t n_vec = bytes / 16;
    const int threads = 512;
    const int blocks = (int)std::min<int64_t>(
        (n_vec + threads - 1) / threads, 432);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_rotated_copies<<<blocks, threads, 0, stream>>>(cs, n_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_ext = None

def _fused_ext():
    """JIT-build the fused-copy extension once (cached across runs)."""
    global _ext
    if _ext is None and os.environ.get("LOW_CONTENTION_CPP", "1") == "1":
        from torch.utils.cpp_extension import load_inline
        _ext = load_inline(
            name="low_contention_fused_copy",
            cpp_sources="void fused_copies(torch::Tensor, torch::Tensor,"
                        " int64_t);\n"
                        "void async_copies(torch::Tensor,"
                        " torch::Tensor, int64_t);",
            cuda_sources=_CPP_SRC,
            functions=["fused_copies", "async_copies"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


class LowContentionComm:
    """Rotated-schedule push/pull collectives over one symmetric buffer."""

    def __init__(
        self,
        group,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        num_streams: int = 1,
        launch_barrier: bool = True,
        streams: list[torch.cuda.Stream] | None = None,
        mover: str = "dma",
        buffer: torch.Tensor | None = None,
    ) -> None:
        # mover="dma": cudaMemcpyAsync peer copies — ZERO SMs (the point
        #   of low contention: never fights compute), ~530 GB/s engine cap.
        # mover="sm": one fused rotated copy kernel — ~630 GB/s at large
        #   messages, costs SMs. Same schedule either way.
        assert mover in ("dma", "sm")
        self.mover = mover
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group=group)
        self.dtype = dtype
        self.device = device
        self.group_name = group.group_name
        self.num_streams = num_streams
        if streams is None:
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        else:
            assert len(streams) == num_streams
            self.streams = streams
        if buffer is not None:
            # Shared IPC pool: reuse an existing symm allocation (e.g.
            # the torch_symm output staging) instead of a private one.
            # rendezvous() on an already-rendezvous'd tensor returns
            # the cached handle, so this adds no collective. ALIASING:
            # results living in the shared buffer (torch_symm AG
            # output views) are invalid once this mover runs, and vice
            # versa — the same one-collective-at-a-time contract the
            # symm staging already carries.
            assert buffer.dtype == dtype and buffer.numel() >= max_numel
            self.buffer = buffer.view(-1)[:max_numel]
        else:
            self.buffer = _SymmetricMemory.empty_strided_p2p(
                (max_numel,), (1,), dtype=dtype, device=device,
                group_name=self.group_name,
            )
        # rendezvous once — the per-call lookup is measurable host cost
        self._symm = rendezvous(self.buffer, self.group_name)
        self._x_ref = None
        self.launch_barrier = launch_barrier
        self._fused = _fused_ext()
        # scratch for the fused path's pointer lists (CPU, reused)
        self._dst_ptrs = torch.empty(8, dtype=torch.int64)
        self._src_ptrs = torch.empty(8, dtype=torch.int64)

    # ------------------------------------------------------------------ util

    def _fan_out(self) -> None:
        for stream in self.streams:
            stream.wait_stream(torch.cuda.current_stream())

    def _join(self) -> None:
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)

    def wait(self) -> None:
        """Drop the input reference after the current-stream join."""
        self._x_ref = None

    def _rot(self, i: int) -> int:
        return (self.rank + 1 + i) % self.world

    def _issue(self, pairs) -> None:
        """Issue the rotated copies. pairs: [(dst_view, src_view), ...] in
        rotated order, all equal-sized, on this instance's buffer/peers."""
        bytes_each = pairs[0][1].numel() * pairs[0][1].element_size()
        if self._fused is not None and bytes_each % 16 == 0 \
                and len(pairs) <= 8:
            for j, (d, s) in enumerate(pairs):
                self._dst_ptrs[j] = d.data_ptr()
                self._src_ptrs[j] = s.data_ptr()
            issue = (self._fused.async_copies if self.mover == "dma"
                     else self._fused.fused_copies)
            issue(self._dst_ptrs[: len(pairs)],
                  self._src_ptrs[: len(pairs)], bytes_each)
            return
        self._fan_out()
        for j, (d, s) in enumerate(pairs):
            with torch.cuda.stream(self.streams[j % self.num_streams]):
                d.copy_(s)
        self._join()

    # ------------------------------------------------------------ all_gather

    def all_gather(self, x: torch.Tensor, mode: str = "push",
                   out: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, D] (my shard) -> [world*B, D] rank-major."""
        if self.world == 1:
            return x
        symm = self._symm
        rows, cols = x.shape
        shard_numel = x.numel()
        if mode == "push":
            self._x_ref = x
            result = symm.get_buffer(
                self.rank, (self.world * rows, cols), x.dtype)
            symm.barrier()   # slot-reuse guard: after everyone's consumer
            pairs = [(symm.get_buffer(self._rot(i), x.shape, x.dtype,
                                      shard_numel * self.rank), x)
                     for i in range(self.world - 1)]
            pairs.append((symm.get_buffer(self.rank, x.shape, x.dtype,
                                          shard_numel * self.rank), x))
            self._issue(pairs)
            symm.barrier()
            return result                 # symm view; call wait() before reuse
        # pull
        if out is None:
            out = torch.empty(self.world * rows, cols, device=x.device,
                              dtype=x.dtype)
        symm.get_buffer(self.rank, x.shape, x.dtype).copy_(x)   # publish
        symm.barrier()
        pairs = [(out[self._rot(i) * rows:(self._rot(i) + 1) * rows],
                  symm.get_buffer(self._rot(i), x.shape, x.dtype))
                 for i in range(self.world - 1)]
        pairs.append((out[self.rank * rows:(self.rank + 1) * rows], x))
        self._issue(pairs)
        symm.barrier()   # my slot is reusable only after every peer read
        return out

    # ------------------------------------------------------------ all_to_all

    def all_to_all(self, x: torch.Tensor, mode: str = "push",
                   out: torch.Tensor | None = None) -> torch.Tensor:
        """x [world*B, D]; chunk d goes to rank d -> [world*B, D]
        (received, source-major)."""
        if self.world == 1:
            return x
        assert x.shape[0] % self.world == 0
        symm = self._symm
        chunk, cols = x.shape[0] // self.world, x.shape[1]
        shard_shape, shard_numel = (chunk, cols), chunk * cols

        def my_chunk(r):
            return x[r * chunk:(r + 1) * chunk]

        if mode == "push":
            self._x_ref = x
            result = symm.get_buffer(self.rank, x.shape, x.dtype)
            symm.barrier()   # slot-reuse guard
            pairs = [(symm.get_buffer(self._rot(i), shard_shape, x.dtype,
                                      shard_numel * self.rank),
                      my_chunk(self._rot(i)))
                     for i in range(self.world - 1)]
            pairs.append((symm.get_buffer(self.rank, shard_shape, x.dtype,
                                          shard_numel * self.rank),
                          my_chunk(self.rank)))
            self._issue(pairs)
            symm.barrier()
            return result                 # symm view; call wait() before reuse
        # pull
        if out is None:
            out = torch.empty_like(x)
        symm.get_buffer(self.rank, x.shape, x.dtype).copy_(x)   # publish ALL
        symm.barrier()
        pairs = [(out[self._rot(i) * chunk:(self._rot(i) + 1) * chunk],
                  symm.get_buffer(self._rot(i), shard_shape, x.dtype,
                                  shard_numel * self.rank))
                 for i in range(self.world - 1)]
        pairs.append((out[self.rank * chunk:(self.rank + 1) * chunk],
                      my_chunk(self.rank)))
        self._issue(pairs)
        symm.barrier()
        return out

    # -------------------------------------------------------- reduce_scatter

    def reduce_scatter(self, x: torch.Tensor,
                       mode: str = "push") -> torch.Tensor:
        """x [world*B, D] partial sums -> my reduced [B, D].

        Not a separate implementation: the data movement IS an all_to_all
        (chunk d of my partials -> rank d); this reduces the received
        source-major slabs with fp32 accumulation.
        """
        if self.world == 1:
            return x
        chunk = x.shape[0] // self.world
        received = self.all_to_all(x, mode=mode)
        if mode == "push":
            self.wait()   # the reduce is the consumer of the symm view
        slabs = received.view(self.world, chunk, x.shape[1])
        return slabs.sum(dim=0, dtype=torch.float32).to(x.dtype)

    # ------------------------------------------------------------ all_reduce

    def all_reduce(self, x: torch.Tensor,
                   mode: str = "push") -> torch.Tensor:
        """2-shot AR: reduce_scatter -> all_gather (ring-shaped wire).

        push mode returns a SYMM VIEW (same contract as push all_gather:
        valid until the next call on this instance — ``.clone()`` it
        yourself if you need it to outlive that). No hidden copy here.
        pull mode returns a local tensor.
        """
        if self.world == 1:
            return x
        reduced = self.reduce_scatter(x, mode=mode)
        gathered = self.all_gather(reduced, mode=mode)
        if mode == "push":
            # join FIRST: the view is complete only after the AG barrier
            self.wait()
        return gathered


