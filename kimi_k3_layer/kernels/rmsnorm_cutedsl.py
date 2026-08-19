"""Column-window RMSNorm (CuTeDSL).

Same semantics as kimi_k3/kernels/rmsnorm_triton.py: the row reduction
covers the FULL row, the normalized output is written for a column
window only, fp32 accumulation/normalization with a single rounding to
the output dtype. The input may be row-strided (inner dim contiguous).

Differences from the Triton kernel that matter for bandwidth:
  * the whole row is loaded ONCE into registers with 128-bit
    vectorized loads and the window columns are normalized straight
    from those registers - no second gmem read of the window;
  * thread/vector geometry divides n_cols exactly, so there are no
    masked lanes in the mainloop.

Variants compile per (n_cols, width, vec, threads-per-row,
rows-per-CTA); ``rows`` and ``col_start`` are runtime parameters, so a
single compiled callable serves every batch size.

Capture-safety: `cute.compile`'s on-disk cache is disabled in this
environment, so compiled callables are cached in a module-level dict.
Callers MUST warm the kernel up (one eager call per distinct
(n_cols, width, stride-kind, rows-bucket) combination) before CUDA
graph capture; after warmup the hot path only allocates the output
tensor and launches.
"""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

__all__ = ["rmsnorm_column_slice_cutedsl"]


@dsl_user_op
def _gmem_ptr_at(x: cute.Tensor, coord, align: int, *, loc=None, ip=None) -> cute.Pointer:
    """Pointer to ``x[coord]`` with an explicit alignment guarantee.

    Pointer arithmetic loses the ``assumed_align`` of the source tensor,
    so (like the routing kernel) we rebuild the pointer with the
    alignment the host wrapper has verified; ``align`` gates whether
    ``cute.autovec_copy`` vectorizes the subsequent copy.
    """
    ptr = x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)
    return cute.make_ptr(
        x.element_type, ptr.toint(loc=loc, ip=ip), cute.AddressSpace.gmem,
        assumed_align=align,
    )


class RmsNormColumnSlice:
    """One compile-time variant of the column-window RMSNorm kernel.

    Geometry: each row is owned by a group of ``tpr`` threads; a CTA
    holds ``rpc`` groups. Thread ``t`` of a group loads vectors
    ``t, t+tpr, ...`` (``nv`` per thread, ``vec`` elements each) of its
    row, i.e. fully coalesced 16B loads. After a group-wide
    sum-of-squares reduction, every in-window vector is normalized from
    registers and stored.
    """

    def __init__(self, n_cols: int, width: int, vec: int, tpr: int, rpc: int):
        assert tpr % 32 == 0, "threads-per-row must be whole warps"
        assert n_cols % (vec * tpr) == 0, (n_cols, vec, tpr)
        assert width % vec == 0, (width, vec)
        self.n_cols = n_cols
        self.width = width
        self.vec = vec
        self.tpr = tpr
        self.rpc = rpc
        self.nv = n_cols // (vec * tpr)      # vectors per thread
        self.wv = width // vec               # vectors in the window
        self.align = vec * 2                 # bf16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,      # (rows, n_cols) bf16, row stride dynamic
        mW: cute.Tensor,      # (n_cols,) bf16 contiguous
        mO: cute.Tensor,      # (rows, width) bf16 contiguous
        col_start: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mW, mO, col_start, eps).launch(
            grid=[cute.ceil_div(mX.shape[0], self.rpc), 1, 1],
            block=[self.tpr * self.rpc, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        col_start: Int32,
        eps: Float32,
    ):
        vec, tpr, rpc, nv = self.vec, self.tpr, self.rpc, self.nv
        wpg = tpr // 32                       # warps per row group

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grp = tidx // tpr
        t = tidx % tpr

        rows = mX.shape[0]
        row = bidx * rpc + grp
        row_ok = row < rows
        # clamp instead of early-exit so CTA-wide barriers stay uniform
        row_c = Int32(cutlass.select_(row_ok, row, rows - 1))

        # ---- load the thread's nv vectors of the row (one gmem pass) ----
        lane_layout = cute.make_layout((vec, nv), stride=(1, vec * tpr))
        gX = cute.make_tensor(
            _gmem_ptr_at(mX, (row_c, vec * t), self.align), lane_layout
        )
        rX = cute.make_rmem_tensor((vec, nv), BFloat16)
        cute.autovec_copy(gX, rX)

        # ---- fp32 sum of squares, group-wide reduction ------------------
        acc = Float32(0.0)
        for i in cutlass.range_constexpr(vec * nv):
            v = rX[i].to(Float32)
            acc += v * v
        acc = cute.arch.warp_reduction_sum(acc)
        if cutlass.const_expr(wpg > 1):
            smem_ptr = cute.arch.alloc_smem(Float32, rpc * wpg)
            smem = cute.make_tensor(
                smem_ptr, cute.make_layout((rpc * wpg,))
            )
            if t % 32 == 0:
                smem[grp * wpg + t // 32] = acc
            cute.arch.sync_threads()
            total = Float32(0.0)
            for w in cutlass.range_constexpr(wpg):
                total += smem[grp * wpg + w]
            acc = total

        rstd = cute.math.rsqrt(acc / self.n_cols + eps, fastmath=True)

        # ---- normalize the window straight from registers ---------------
        c0v = col_start // vec
        for s in cutlass.range_constexpr(nv):
            gv = Int32(s * tpr) + t           # vector index within the row
            rel = gv - c0v
            if row_ok & (rel >= 0) & (rel < self.wv):
                gW = cute.make_tensor(
                    _gmem_ptr_at(mW, (vec * gv,), self.align),
                    cute.make_layout((vec,)),
                )
                rW = cute.make_rmem_tensor((vec,), BFloat16)
                cute.autovec_copy(gW, rW)
                rO = cute.make_rmem_tensor((vec,), BFloat16)
                for i in cutlass.range_constexpr(vec):
                    y = rX[i, s].to(Float32) * rstd * rW[i].to(Float32)
                    rO[i] = y.to(BFloat16)
                gO = cute.make_tensor(
                    _gmem_ptr_at(mO, (row_c, vec * rel), self.align),
                    cute.make_layout((vec,)),
                )
                cute.autovec_copy(rO, gO)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}

# (vec, threads_per_row, rows_per_cta) configurations; keys are picked by
# _pick_config below. Sweepable via LLMDD_RMSNORM_CUTE_CFG="vec,tpr,rpc".
_CONFIGS = {
    "v8_t224_r1": (8, 224, 1),
    "v8_t448_r1": (8, 448, 1),
    "v16_t224_r1": (16, 224, 1),
    "v16_t224_r2": (16, 224, 2),
    "v16_t224_r4": (16, 224, 4),
    "v8_t64_r4": (8, 64, 4),
    "v8_t64_r2": (8, 64, 2),
    "v8_t64_r1": (8, 64, 1),
    "v8_t64_r3": (8, 64, 3),
    "v8_t32_r2": (8, 32, 2),
}


def _pick_config(rows: int, width: int, n_cols: int) -> tuple[int, int, int]:
    import os

    forced = os.environ.get("LLMDD_RMSNORM_CUTE_CFG")
    if forced:
        vec, tpr, rpc = (int(v) for v in forced.split(","))
        return vec, tpr, rpc
    # tuned on B200 (see local_debug/rmsnorm_cols_cute.py --sweep): 224
    # threads (7 warps) per row with 256-bit vectors wins everywhere
    # except the large-batch narrow-window case, where the tiny output
    # write makes a skinny 2-warp CTA per row the best reader.
    if rows > 1024 and width * 4 <= n_cols:
        return _CONFIGS["v8_t64_r1"]
    return _CONFIGS["v16_t224_r1"]


def rmsnorm_column_slice_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    col_start: int = 0,
    width: int | None = None,
) -> torch.Tensor:
    """RMSNorm over the full row of ``x``; output only columns
    [col_start, col_start+width) as a fresh contiguous tensor. ``x``
    may be row-strided (``x.stride(1) == 1``). CuTeDSL port of
    ``rmsnorm_triton.rmsnorm_cols``; warm up per shape before CUDA
    graph capture.
    """
    rows, n_cols = x.shape
    assert x.stride(1) == 1
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    width = n_cols if width is None else width
    out = torch.empty((rows, width), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out

    vec, tpr, rpc = _pick_config(rows, width, n_cols)
    align = vec * 2
    assert x.data_ptr() % align == 0
    assert (x.stride(0) * 2) % align == 0, "row stride must keep vectors aligned"
    assert col_start % vec == 0 and width % vec == 0

    mX = from_dlpack(x, assumed_align=align).mark_layout_dynamic(leading_dim=1)
    mW = from_dlpack(weight, assumed_align=align)
    mO = from_dlpack(out, assumed_align=align).mark_layout_dynamic(leading_dim=1)
    # the launch stream is fetched fresh on EVERY call (graph-capture safe)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (n_cols, width, vec, tpr, rpc)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        op = RmsNormColumnSlice(n_cols, width, vec, tpr, rpc)
        kernel = cute.compile(
            op, mX, mW, mO, Int32(col_start), Float32(eps), stream
        )
        _KERNEL_CACHE[key] = kernel
    kernel(mX, mW, mO, Int32(col_start), Float32(eps), stream)
    return out


AVAILABLE = True
