"""``b10_multimem`` backend: our NVLS multimem AllReduce kernel.

Same ``multimem.ld_reduce`` / ``multimem.st`` instructions as torch's
``multimem_all_reduce_``, with the two changes measured to matter on
B200 (local_result/why_backends.md):

* deep per-thread memory-level parallelism (4 independent reduced
  loads in flight before the stores) — torch's shallower loop leaves
  1.45x on the table at the SAME grid size;
* the grid size is an ARGUMENT (default 16 blocks — the sweep showed
  the NVLS reduce path saturates at 8-16 blocks and gets SLOWER with
  more, so wide grids are pointless for the pure AR).

Wire ceiling measured: ~540 GB/s per direction (~72% of practical
NVLink-5 payload rate) — the NVSwitch reduction path limit; this
kernel sits on it (117 MB in ~247 us vs torch's 358 us).

Correctness protocol: rank r reduces slice r via the multicast VA and
multicast-stores the result, so every rank's out buffer receives all
slices. In-kernel entry barrier (release-add on a multicast flag,
acquire-spin locally) orders every rank's stage-in copy before any
``ld_reduce``; an exit barrier (last local block arrives, one
release-add, all spin) keeps the NEXT call's stage-in from racing
peers still reading. Flags are monotonic counters — no reset.

CUDA-GRAPH SAFETY: the barrier variant (``all_reduce``) keeps its
call counter in DEVICE memory — a per-block ``rounds`` slot bumped
inside the kernel (the ``OneShotComm`` pattern from col_quant.py) —
so the barrier targets advance correctly under graph replay; the grid
is fixed per instance as a consequence. The ``:lamport`` variant
still rotates slots on the HOST and remains capture-unsafe (it
hard-asserts; also registered in ``collectives._CAPTURE_UNSAFE``).
Validation: local_result/prefill_graph_smallbatch_analysis.md §3 +
local_debug/pfl_ar_graph_probe.py document the original host-counter bug.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed._symmetric_memory as symm_mem_mod

from ..context import Ctx

_CPP = ("void mm_ar(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "int64_t,int64_t,int64_t);"
        "void mm_ar_norm(int64_t,int64_t,int64_t,int64_t,int64_t,"
        "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "int64_t,double,int64_t,int64_t,int64_t);"
        "void mm_ar_lp(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);")

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda_bf16.h>

__device__ __forceinline__ uint4 ldreduce(const void* p) {
  uint4 v;
  asm volatile(
    "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
    : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void mst(void* p, uint4 v) {
  asm volatile(
    "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
    :: "l"(p), "f"(__int_as_float(v.x)), "f"(__int_as_float(v.y)),
       "f"(__int_as_float(v.z)), "f"(__int_as_float(v.w)) : "memory");
}
__device__ __forceinline__ void flag_add_release(unsigned* mc_flag) {
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;"
               :: "l"(mc_flag) : "memory");
}
__device__ __forceinline__ unsigned flag_ld_acquire(unsigned* flag) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v) : "l"(flag) : "memory");
  return v;
}
__device__ __forceinline__ unsigned flag_ld_relaxed_gpu(unsigned* flag) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
               : "=r"(v) : "l"(flag) : "memory");
  return v;
}
// bf16 -0.0 (0x8000) is the "not yet written" sentinel; canonicalize
// real payload so it never matches
__device__ __forceinline__ unsigned canon(unsigned u) {
  if ((u & 0xffffu) == 0x8000u) u &= 0xffff0000u;
  if ((u >> 16) == 0x8000u) u &= 0x0000ffffu;
  return u;
}
__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
    : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool has_sentinel(uint4 v) {
  return ((v.x & 0xffffu) == 0x8000u) || ((v.x >> 16) == 0x8000u) ||
         ((v.y & 0xffffu) == 0x8000u) || ((v.y >> 16) == 0x8000u) ||
         ((v.z & 0xffffu) == 0x8000u) || ((v.z >> 16) == 0x8000u) ||
         ((v.w & 0xffffu) == 0x8000u) || ((v.w >> 16) == 0x8000u);
}

__device__ __forceinline__ void flag_max_release(unsigned* mc_flag,
                                                  unsigned v) {
  asm volatile("multimem.red.release.sys.global.max.u32 [%0], %1;"
               :: "l"(mc_flag), "r"(v) : "memory");
}

// bf16x8 chunk helpers (uint4 <-> 8 bf16), fp32 math
__device__ __forceinline__ float2 _up(unsigned u) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u));
}
__device__ __forceinline__ unsigned _dn(float2 f) {
  __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<unsigned*>(&b);
}

// flags layout (uint32): [0] entry/exit counter (multicast),
// [1] local block-arrival counter (unicast, monotonic).
// ``rounds`` is a per-block device counter (one slot per block of the
// FIXED grid): each block bumps its own slot at launch, so the call
// number lives on the DEVICE and CUDA-graph replays advance it
// naturally — no host-baked call_id (the graph-safety fix).
__global__ void mm_ar_kernel(const char* mc_in, char* mc_out,
                             unsigned* mc_flag, unsigned* loc_flag,
                             unsigned* rounds,
                             int64_t slice_bytes, int64_t base,
                             int world) {
  __shared__ unsigned s_call;
  if (threadIdx.x == 0) {
    s_call = ++rounds[blockIdx.x];  // single writer per slot
    if (blockIdx.x == 0) flag_add_release(mc_flag);
  }
  __syncthreads();
  const unsigned call_id = s_call;
  // entry barrier: my stage-in is stream-ordered before this kernel;
  // wait until every rank has signalled arrival for this call
  unsigned entry_target = world * (2 * call_id - 1);
  if (threadIdx.x == 0)
    while (flag_ld_acquire(loc_flag) < entry_target) {}
  __syncthreads();

  int64_t stride = (int64_t)gridDim.x * blockDim.x * 16;
  int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  for (; i + 3 * stride < slice_bytes; i += 4 * stride) {
    uint4 a = ldreduce(mc_in + base + i);
    uint4 b = ldreduce(mc_in + base + i + stride);
    uint4 c = ldreduce(mc_in + base + i + 2 * stride);
    uint4 d = ldreduce(mc_in + base + i + 3 * stride);
    mst(mc_out + base + i, a);
    mst(mc_out + base + i + stride, b);
    mst(mc_out + base + i + 2 * stride, c);
    mst(mc_out + base + i + 3 * stride, d);
  }
  for (; i < slice_bytes; i += stride)
    mst(mc_out + base + i, ldreduce(mc_in + base + i));

  // exit barrier: last local block signals; everyone waits for all
  // ranks so the next call's stage-in cannot race remote readers
  __syncthreads();
  if (threadIdx.x == 0) {
    // fixed grid => cumulative arrivals incl. this call = grid * call
    if (atomicAdd(loc_flag + 1, 1u) == gridDim.x * call_id - 1)
      flag_add_release(mc_flag);
    while (flag_ld_acquire(loc_flag) < (unsigned)(world * 2 * call_id)) {}
  }
  __syncthreads();
}

// Lamport variant: ONE entry flag wave, NO exit barrier. Rank r
// reduces slice r and multicast-stores it (canonicalized); every rank
// then polls its LOCAL out-slot copy until no 16B chunk holds the
// sentinel — the poll is both the result gate and the per-copy
// delivery confirmation (so the next call may clear this slot safely).
// Also re-sentinels the OTHER out slot (used two calls ago) for reuse.
__global__ void mm_ar_lamport(const char* mc_in, char* mc_out,
                              const char* out_local, char* clear_ptr,
                              unsigned* mc_flag, unsigned* loc_flag,
                              int64_t slice_bytes, int64_t base,
                              int64_t total_bytes, int64_t clear_bytes,
                              unsigned call_id, int world) {
  if (blockIdx.x == 0 && threadIdx.x == 0) flag_add_release(mc_flag);
  unsigned target = (unsigned)world * call_id;
  if (threadIdx.x == 0)
    while (flag_ld_acquire(loc_flag) < target) {}
  __syncthreads();

  int64_t stride = (int64_t)gridDim.x * blockDim.x * 16;
  int64_t tid0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  for (int64_t i = tid0; i < slice_bytes; i += stride) {
    uint4 v = ldreduce(mc_in + base + i);
    v.x = canon(v.x); v.y = canon(v.y); v.z = canon(v.z); v.w = canon(v.w);
    mst(mc_out + base + i, v);
  }
  // re-sentinel the other slot (its consumer poll finished last call,
  // so no in-flight remote stores can land in our copy anymore)
  const uint4 s16 = {0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u};
  for (int64_t i = tid0; i < clear_bytes; i += stride)
    *reinterpret_cast<uint4*>(clear_ptr + i) = s16;
  // consume: poll own copy until every chunk is real payload
  for (int64_t i = tid0; i < total_bytes; i += stride)
    while (has_sentinel(ld_volatile16(out_local + i))) {}
}

void mm_ar_lp(int64_t mc_in, int64_t mc_out, int64_t out_local,
              int64_t clear_ptr, int64_t mc_flag, int64_t loc_flag,
              int64_t nbytes, int64_t clear_bytes, int64_t rank,
              int64_t world, int64_t call_id, int64_t blocks) {
  int64_t slice = nbytes / world;
  mm_ar_lamport<<<(unsigned)blocks, 1024, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      (const char*)mc_in, (char*)mc_out, (const char*)out_local,
      (char*)clear_ptr, (unsigned*)mc_flag, (unsigned*)loc_flag,
      slice, rank * slice, nbytes, clear_bytes, (unsigned)call_id,
      (int)world);
}

// Fused allreduce + residual + rmsnorm: rank r ld_reduces its rows
// (round-robin row % world == rank, so ANY B works), adds the (rank-
// identical) residual in fp32, multicast-stores the summed row (res =
// the next residual stream), then releases a PER-ROW flag with
// multimem.red.max(call) — monotonic AND correct for varying row
// counts across calls, so CUDA-graph replay is safe. Every rank
// (owners included) then polls rows as they arrive and computes
// norm = res * rsqrt(mean(res^2)+eps) * gamma LOCALLY — the norm
// broadcast costs nothing on the wire, and early rows normalize while
// late rows are still reducing. Entry/exit barriers and the device
// rounds are shared with mm_ar_kernel (same grid, same flags).
__global__ void mm_ar_norm_kernel(
    const char* mc_in, char* mc_res, const char* res_local,
    char* norm_local, const char* residual_in, const char* gamma,
    unsigned* mc_flag, unsigned* loc_flag, unsigned* rounds,
    unsigned* row_flags_mc, unsigned* row_flags_local,
    int rows, int row_bytes, float eps, int rank, int world) {
  __shared__ unsigned s_call;
  if (threadIdx.x == 0) {
    s_call = ++rounds[blockIdx.x];
    if (blockIdx.x == 0) flag_add_release(mc_flag);
  }
  __syncthreads();
  const unsigned call_id = s_call;
  if (threadIdx.x == 0)
    while (flag_ld_acquire(loc_flag) < world * (2 * call_id - 1)) {}
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5;
  const int chunks = row_bytes / 16;
  const float inv_h = 1.0f / (float)(row_bytes / 2);
  // Role split: the NVLS reduce path saturates at ~16 blocks, but the
  // local normalize pass needs FULL-GPU bandwidth — so the first
  // N_PROD blocks produce (phase 1) and the rest consume (phase 2),
  // overlapped row-by-row through the flags.
  const int N_PROD = 16;
  const bool producer = blockIdx.x < N_PROD;
  const int gwarp = producer
      ? blockIdx.x * wpb + (threadIdx.x >> 5)
      : (blockIdx.x - N_PROD) * wpb + (threadIdx.x >> 5);
  const int nwarps = (producer ? N_PROD : (gridDim.x - N_PROD)) * wpb;

  // phase 1 (producer blocks): reduce + residual-add + broadcast MY
  // rows, flag each.
  // NVLS loads are issued in 4-deep independent batches per lane —
  // a shallow serial chain stalls ~5us per reduced load (the same
  // MLP lesson as mm_ar_kernel / why_backends.md).
  if (producer)
  for (int i = gwarp; i * world + rank < rows; i += nwarps) {
    const int row = i * world + rank;
    const int64_t off = (int64_t)row * row_bytes;
    for (int c0 = lane; c0 < chunks; c0 += 32 * 4) {
      uint4 vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = c0 + j * 32;
        if (c < chunks) vv[j] = ldreduce(mc_in + off + (int64_t)c * 16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = c0 + j * 32;
        if (c >= chunks) break;
        uint4 v = vv[j];
        const uint4 r = *reinterpret_cast<const uint4*>(
            residual_in + off + (int64_t)c * 16);
        float2 a;
        a = _up(v.x); { float2 b = _up(r.x); a.x += b.x; a.y += b.y; } v.x = _dn(a);
        a = _up(v.y); { float2 b = _up(r.y); a.x += b.x; a.y += b.y; } v.y = _dn(a);
        a = _up(v.z); { float2 b = _up(r.z); a.x += b.x; a.y += b.y; } v.z = _dn(a);
        a = _up(v.w); { float2 b = _up(r.w); a.x += b.x; a.y += b.y; } v.w = _dn(a);
        mst(mc_res + off + (int64_t)c * 16, v);
      }
    }
    __syncwarp();
    if (lane == 0) flag_max_release(row_flags_mc + row, call_id);
  }

  // phase 2: poll each row's flag, then normalize it locally.
  // Spin at RELAXED GPU scope with backoff — a tight acquire.sys spin
  // from hundreds of waiting warps throttles the phase-1 producers on
  // the same SMs; one acquire.sys load confirms with the right
  // ordering once the relaxed poll sees the value.
  if (!producer)
  for (int row = gwarp; row < rows; row += nwarps) {
    if (lane == 0) {
      while (flag_ld_relaxed_gpu(row_flags_local + row) < call_id)
        __nanosleep(128);
      while (flag_ld_acquire(row_flags_local + row) < call_id) {}
    }
    __syncwarp();
    const int64_t off = (int64_t)row * row_bytes;
    float ssq = 0.f;
    for (int c = lane; c < chunks; c += 32) {
      const uint4 r = *reinterpret_cast<const uint4*>(
          res_local + off + (int64_t)c * 16);
      float2 f;
      f = _up(r.x); ssq += f.x * f.x + f.y * f.y;
      f = _up(r.y); ssq += f.x * f.x + f.y * f.y;
      f = _up(r.z); ssq += f.x * f.x + f.y * f.y;
      f = _up(r.w); ssq += f.x * f.x + f.y * f.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
    const float rs = rsqrtf(ssq * inv_h + eps);
    for (int c = lane; c < chunks; c += 32) {
      const uint4 r = *reinterpret_cast<const uint4*>(
          res_local + off + (int64_t)c * 16);
      const uint4 g = *reinterpret_cast<const uint4*>(
          gamma + (int64_t)c * 16);
      uint4 n;
      float2 fr, fg;
      fr = _up(r.x); fg = _up(g.x); n.x = _dn({fr.x * rs * fg.x, fr.y * rs * fg.y});
      fr = _up(r.y); fg = _up(g.y); n.y = _dn({fr.x * rs * fg.x, fr.y * rs * fg.y});
      fr = _up(r.z); fg = _up(g.z); n.z = _dn({fr.x * rs * fg.x, fr.y * rs * fg.y});
      fr = _up(r.w); fg = _up(g.w); n.w = _dn({fr.x * rs * fg.x, fr.y * rs * fg.y});
      *reinterpret_cast<uint4*>(norm_local + off + (int64_t)c * 16) = n;
    }
  }

  // exit barrier: protects the x staging and res reuse (same as mm_ar)
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(loc_flag + 1, 1u) == gridDim.x * call_id - 1)
      flag_add_release(mc_flag);
    while (flag_ld_acquire(loc_flag) < (unsigned)(world * 2 * call_id)) {}
  }
  __syncthreads();
}

void mm_ar_norm(int64_t mc_in, int64_t mc_res, int64_t res_local,
                int64_t norm_local, int64_t residual_in, int64_t gamma,
                int64_t mc_flag, int64_t loc_flag, int64_t rounds,
                int64_t row_flags_mc, int64_t row_flags_local,
                int64_t rows, int64_t row_bytes, double eps,
                int64_t rank, int64_t world, int64_t blocks) {
  mm_ar_norm_kernel<<<(unsigned)blocks, 1024, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      (const char*)mc_in, (char*)mc_res, (const char*)res_local,
      (char*)norm_local, (const char*)residual_in, (const char*)gamma,
      (unsigned*)mc_flag, (unsigned*)loc_flag, (unsigned*)rounds,
      (unsigned*)row_flags_mc, (unsigned*)row_flags_local,
      (int)rows, (int)row_bytes, (float)eps, (int)rank, (int)world);
}

void mm_ar(int64_t mc_in, int64_t mc_out, int64_t mc_flag,
           int64_t loc_flag, int64_t rounds, int64_t nbytes,
           int64_t rank, int64_t world, int64_t blocks) {
  int64_t slice = nbytes / world;
  mm_ar_kernel<<<(unsigned)blocks, 1024, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      (const char*)mc_in, (char*)mc_out, (unsigned*)mc_flag,
      (unsigned*)loc_flag, (unsigned*)rounds, slice, rank * slice,
      (int)world);
}
"""

_mod = None


def _jit():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault(
            "TORCH_EXTENSIONS_DIR",
            f"/tmp/torchext_r{int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('RANK', 0)))}")
        _mod = load_inline(
            name="b10_multimem_ar", cpp_sources=_CPP, cuda_sources=_CUDA,
            functions=["mm_ar", "mm_ar_lp", "mm_ar_norm"], verbose=False,
            extra_cuda_cflags=["-gencode=arch=compute_100a,code=sm_100a"])
    return _mod


def _assert_not_capturing() -> None:
    """The LAMPORT variant's slot rotation + call counter live on the
    HOST — a replayed graph re-runs stale slots/targets -> WRONG
    outputs. AUTO never dispatches it (not a candidate); an explicit
    ``impl=`` request under capture must fail loudly."""
    assert not torch.cuda.is_current_stream_capturing(), (
        "b10_multimem:lamport is NOT CUDA-graph capture safe "
        "(host-side slot rotation); use b10_multimem (graph-safe) or "
        "a torch_symm impl inside captures.")


class MultimemARKernel:
    """Collective constructor (symm rendezvous + JIT build)."""

    def __init__(self, ctx: Ctx, blocks: int = 16) -> None:
        if torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError("b10_multimem needs sm100+")
        self.ctx = ctx
        self.blocks = blocks
        self.mod = _jit()
        gname = ctx.group.group_name
        self._in = symm_mem_mod.empty(ctx.max_numel, dtype=ctx.dtype,
                                      device=ctx.device)
        self._out = symm_mem_mod.empty(ctx.max_numel, dtype=ctx.dtype,
                                       device=ctx.device)
        self._flags = symm_mem_mod.empty(4, dtype=torch.int32,
                                         device=ctx.device)
        self._h_in = symm_mem_mod.rendezvous(self._in, gname)
        self._h_out = symm_mem_mod.rendezvous(self._out, gname)
        self._h_flags = symm_mem_mod.rendezvous(self._flags, gname)
        if not self._h_in.multicast_ptr:
            raise RuntimeError("no NVLS multicast support")
        self._flags.zero_()
        # per-block device rounds: the barrier kernel's call counter
        # lives on the device (graph-replay safe); grid is FIXED per
        # instance as a consequence
        self._rounds = torch.zeros(blocks, dtype=torch.int32,
                                   device=ctx.device)
        self._lp = None  # lazy lamport slot state
        self._norm = None  # lazy fused AR+norm state
        torch.cuda.synchronize()
        import torch.distributed as dist
        dist.barrier(ctx.group)

    def symm_input(self, shape, offset: int = 0) -> torch.Tensor:
        """A view of the in staging buffer to produce into (skips the
        stage-in copy of the next :meth:`all_reduce`). ``offset`` in
        elements — must keep 16-byte alignment; lets a second AR's
        operand live above the first AR's [0, n) staging region."""
        n = int(math.prod(shape))
        assert offset + n <= self.ctx.max_numel, "symm staging overflow"
        assert (offset * self._in.element_size()) % 16 == 0
        return self._in[offset : offset + n].view(*shape)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a VIEW of the out staging buffer, valid until the
        next call. CUDA-GRAPH SAFE: the call counter is a per-block
        DEVICE round (each block bumps its own ``rounds`` slot at
        launch), so capture/replay of any call count advances the
        barrier targets correctly — the grid is FIXED per instance for
        the same reason. The world-size slice split needs
        ``numel*2 % (world*16) == 0`` (true for [B, 7168] bf16).

        ZERO-COPY: an operand already resident in the in buffer (any
        16B-aligned offset view from :meth:`symm_input`) skips the
        stage-in copy; the kernel runs at that offset (the multicast
        VAs are linear, so base+offset is valid on every rank) and the
        result view sits at the SAME offset of the out buffer."""
        n = x.numel()
        el = self._in.element_size()
        nbytes = n * el
        assert nbytes % (self.ctx.world * 16) == 0
        base = self._in.data_ptr()
        if x.is_contiguous() and base <= x.data_ptr() < base + \
                self.ctx.max_numel * el:
            off = (x.data_ptr() - base) // el
            assert (off * el) % 16 == 0 and off + n <= self.ctx.max_numel
        else:
            off = 0
            self._in[:n].view_as(x).copy_(x)
        off_bytes = off * el
        self.mod.mm_ar(
            self._h_in.multicast_ptr + off_bytes,
            self._h_out.multicast_ptr + off_bytes,
            self._h_flags.multicast_ptr, self._flags.data_ptr(),
            self._rounds.data_ptr(), nbytes, self.ctx.rank,
            self.ctx.world, self.blocks)
        return self._out[off : off + n].view_as(x)

    # ------------------------------------------- fused AR+norm variant

    _MAX_ROWS = 65536  # row-flag capacity
    # 16 producers + 112 consumers. HARD CAP < 148 (B200 SM count):
    # at 1024 threads/block only ~1 block/SM is resident, and the exit
    # barrier needs EVERY block of the grid to arrive — a grid larger
    # than what fits co-resident deadlocks (192 hung the node).
    _NORM_BLOCKS = 128

    def _norm_state(self):
        """res symm buffer + local norm buffer + per-row flag array,
        built on first use (COLLECTIVE: rendezvous)."""
        if self._norm is None:
            gname = self.ctx.group.group_name
            res = symm_mem_mod.empty(self.ctx.max_numel,
                                     dtype=self.ctx.dtype,
                                     device=self.ctx.device)
            h_res = symm_mem_mod.rendezvous(res, gname)
            flags = symm_mem_mod.empty(self._MAX_ROWS,
                                       dtype=torch.int32,
                                       device=self.ctx.device)
            h_flags = symm_mem_mod.rendezvous(flags, gname)
            flags.zero_()
            norm = torch.empty(self.ctx.max_numel, dtype=self.ctx.dtype,
                               device=self.ctx.device)
            # kernel-private entry/exit flags + per-block rounds: the
            # norm kernel's grid differs from mm_ar's, so the shared
            # counters would desync across kernel types
            bar = symm_mem_mod.empty(2, dtype=torch.int32,
                                     device=self.ctx.device)
            h_bar = symm_mem_mod.rendezvous(bar, self.ctx.group.group_name)
            bar.zero_()
            rounds = torch.zeros(self._NORM_BLOCKS, dtype=torch.int32,
                                 device=self.ctx.device)
            torch.cuda.synchronize()
            import torch.distributed as dist
            dist.barrier(self.ctx.group)
            self._norm = {"res": res, "h_res": h_res, "norm": norm,
                          "flags": flags, "h_flags": h_flags,
                          "bar": bar, "h_bar": h_bar, "rounds": rounds}
        return self._norm

    def allreduce_norm(self, x, gamma, eps, residual):
        """Fused ``rmsnorm(allreduce(x) + residual, gamma, eps)`` in
        ONE kernel: my rows are switch-reduced + residual-added +
        multicast as the new residual stream; every rank normalizes
        rows locally as they arrive (zero extra wire for the norm).
        Returns ``(norm, new_residual)`` — views of internal buffers,
        valid until the next call. CUDA-GRAPH SAFE (device rounds +
        red.max per-row flags). ``residual`` must be rank-identical
        and is REQUIRED here (the dispatcher passes zeros for the
        residual-free form)."""
        b, h = x.shape
        nbytes = b * h * 2
        assert h * 2 % 16 == 0 and b <= self._MAX_ROWS
        assert residual.is_contiguous() and gamma.is_contiguous()
        st = self._norm_state()
        v = self._in[: b * h].view(b, h)
        if x.data_ptr() != v.data_ptr():
            v.copy_(x)
        self.mod.mm_ar_norm(
            self._h_in.multicast_ptr, st["h_res"].multicast_ptr,
            st["res"].data_ptr(), st["norm"].data_ptr(),
            residual.data_ptr(), gamma.data_ptr(),
            st["h_bar"].multicast_ptr, st["bar"].data_ptr(),
            st["rounds"].data_ptr(), st["h_flags"].multicast_ptr,
            st["flags"].data_ptr(), b, h * 2, float(eps),
            self.ctx.rank, self.ctx.world, self._NORM_BLOCKS)
        return (st["norm"][: b * h].view(b, h),
                st["res"][: b * h].view(b, h))

    # ------------------------------------------------ lamport variant

    def _lp_state(self):
        """Two rotating (in, out) symm slot pairs + own entry flag.
        COLLECTIVE on first call (rendezvous). Out slots start fully
        sentineled (bf16 -0.0)."""
        if self._lp is None:
            gname = self.ctx.group.group_name
            slots = []
            for _ in range(2):
                i_t = symm_mem_mod.empty(self.ctx.max_numel,
                                         dtype=self.ctx.dtype,
                                         device=self.ctx.device)
                o_t = symm_mem_mod.empty(self.ctx.max_numel,
                                         dtype=self.ctx.dtype,
                                         device=self.ctx.device)
                h_i = symm_mem_mod.rendezvous(i_t, gname)
                h_o = symm_mem_mod.rendezvous(o_t, gname)
                o_t.view(torch.int16).fill_(-32768)  # sentinel
                slots.append((i_t, o_t, h_i, h_o))
            torch.cuda.synchronize()
            import torch.distributed as dist
            dist.barrier(self.ctx.group)
            self._lp = {"slots": slots, "call": 0,
                        "last_bytes": [0, 0]}
        return self._lp

    def all_reduce_lamport(self, x: torch.Tensor,
                           blocks: int | None = None) -> torch.Tensor:
        """Exit-barrier-free multimem AR (decode band): one entry flag
        wave, slot rotation, sentinel-poll consumption. Returns a VIEW
        of the slot's out buffer, valid until the NEXT call (which
        re-sentinels the previous slot). NOT capture-safe (host-side
        call id + host slot rotation) — hard-asserts."""
        _assert_not_capturing()
        lp = self._lp_state()
        n = x.numel()
        nbytes = n * 2
        assert nbytes % (self.ctx.world * 16) == 0
        lp["call"] += 1
        s = lp["call"] % 2
        i_t, o_t, h_i, h_o = lp["slots"][s]
        _, o_prev, _, _ = lp["slots"][1 - s]
        v = i_t[:n].view_as(x)
        if x.data_ptr() != v.data_ptr():
            v.copy_(x)
        clear_bytes = lp["last_bytes"][1 - s]
        lp["last_bytes"][s] = nbytes
        # lamport entry flag lives at element 2 of the shared flags
        self.mod.mm_ar_lp(
            h_i.multicast_ptr, h_o.multicast_ptr, o_t.data_ptr(),
            o_prev.data_ptr(), self._h_flags.multicast_ptr + 8,
            self._flags.data_ptr() + 8, nbytes, clear_bytes,
            self.ctx.rank, self.ctx.world, lp["call"],
            blocks or self.blocks)
        return o_t[:n].view_as(x)
