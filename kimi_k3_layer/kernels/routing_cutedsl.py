"""Kimi-K3 noaux_tc top-16 routing as a CuTe DSL SIMT kernel.

One warp routes each token. Variants cache by dtype, format, and alignment;
batch and row stride are dynamic.
"""

from __future__ import annotations

import math

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

NUM_EXPERTS = 896
TOP_K = 16
ROUTED_SCALING = 1.0

_VEC = 4                       # elements per vectorized load
_NV = NUM_EXPERTS // (_VEC * 32)   # vectors per lane = 7
_TOKENS_PER_CTA = 4            # warps per CTA
_IDX_REV = 1023                # reversed-index tie-break constant (as Triton)
_NCAND = 4                     # per-lane sorted candidate list depth
_LOG2E = math.log2(math.e)


@dsl_user_op
def _gmem_ptr_at(x: cute.Tensor, coord, align: int, *, loc=None, ip=None) -> cute.Pointer:
    """Pointer to ``x[coord]`` with an explicit alignment guarantee.

    Pointer arithmetic loses the ``assumed_align`` of the source tensor, so
    (like QuACK's ``domain_offset_aligned``) we rebuild the pointer with the
    alignment the host wrapper has verified. ``align`` gates whether
    ``cute.autovec_copy`` vectorizes the subsequent copy.
    """
    ptr = x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)
    return cute.make_ptr(
        x.element_type, ptr.toint(loc=loc, ip=ip), cute.AddressSpace.gmem,
        assumed_align=align,
    )


class KimiK3Routing:
    """CuTeDSL noaux_tc top-16 routing kernel (one warp per token)."""

    def __init__(self, in_dtype, out_dtype, x_align: int, bias_align: int,
                 exact_sigmoid: bool = False):
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.x_align = x_align
        self.bias_align = bias_align
        self.exact_sigmoid = exact_sigmoid

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,        # (B, 896) in_dtype, row stride dynamic
        mBias: cute.Tensor,     # (896,) f32 contiguous
        mIds: cute.Tensor,      # (B, 16) int32
        mScales: cute.Tensor,   # (B, 16) out_dtype
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mBias, mIds, mScales).launch(
            grid=[cute.ceil_div(mX.shape[0], _TOKENS_PER_CTA), 1, 1],
            block=[32 * _TOKENS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mBias: cute.Tensor,
        mIds: cute.Tensor,
        mScales: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = bidx * _TOKENS_PER_CTA + tidx // 32

        if row < mX.shape[0]:
            # ---- load the lane's 28 logits + bias, coalesced + vectorized --
            # lane owns experts e = 128*v + 4*lane + t, v in 0..6, t in 0..3
            lane_layout = cute.make_layout((_VEC, _NV), stride=(1, _VEC * 32))
            gX = cute.make_tensor(
                _gmem_ptr_at(mX, (row, _VEC * lane), self.x_align), lane_layout
            )
            gB = cute.make_tensor(
                _gmem_ptr_at(mBias, (_VEC * lane,), self.bias_align), lane_layout
            )
            rX = cute.make_rmem_tensor((_VEC, _NV), self.in_dtype)
            rB = cute.make_rmem_tensor((_VEC, _NV), Float32)
            cute.autovec_copy(gX, rX)
            cute.autovec_copy(gB, rB)

            # ---- fp32 sigmoid, biased score, order-preserving u32 key ------
            # (same monotone map as the Triton kernel's key32). Sigmoid uses
            # ex2.approx + rcp.approx: <= a few fp32 ulp vs torch (spec tol
            # 1e-6), exact at x = 0 (all-tied case: ex2(0)=1, rcp(2)=0.5).
            rS = cute.make_rmem_tensor((_VEC, _NV), Float32)
            rBiased = cute.make_rmem_tensor((_VEC, _NV), Float32)
            for i in cutlass.range_constexpr(_VEC * _NV):
                x = rX[i].to(Float32)
                if cutlass.const_expr(self.exact_sigmoid):
                    # bit-identical to torch fp32 sigmoid (expf + div.rn)
                    s = 1.0 / (1.0 + cute.math.exp(-x, fastmath=False))
                else:
                    e = cute.math.exp2(-x * _LOG2E, fastmath=True)
                    s = cute.arch.rcp_approx(1.0 + e)
                rS[i] = s
                rBiased[i] = s + rB[i]
            rK = cute.recast_tensor(rBiased, Uint32)
            for i in cutlass.range_constexpr(_VEC * _NV):
                bits = rK[i]
                sign = bits >> 31
                rK[i] = bits ^ ((Uint32(0) - sign) | Uint32(0x80000000))

            # ---- build the lane-local sorted top-4 candidate list ---------
            # 4 fused kill+scan passes; ascending-e scan with strict > keeps
            # the smaller expert index on ties. Passes j>0 kill the previous
            # winner's slot in rK (key 0 == strictly below any real key), so
            # rK afterwards holds "everything below candidate 3" and a rare
            # refill is just the same pass again.
            ck0 = Uint32(0); ce0 = Int32(_IDX_REV); cs0 = Float32(0.0)
            ck1 = Uint32(0); ce1 = Int32(_IDX_REV); cs1 = Float32(0.0)
            ck2 = Uint32(0); ce2 = Int32(_IDX_REV); cs2 = Float32(0.0)
            ck3 = Uint32(0); ce3 = Int32(_IDX_REV); cs3 = Float32(0.0)
            prev_e = Int32(-1)  # nothing to kill on the first pass
            for j in cutlass.range_constexpr(_NCAND):
                best_k = Uint32(0)
                best_e = Int32(_IDX_REV)
                best_s = Float32(0.0)
                for v in cutlass.range_constexpr(_NV):
                    for t in cutlass.range_constexpr(_VEC):
                        eid = Int32(128 * v + t) + _VEC * lane
                        k0 = Uint32(
                            cutlass.select_(eid == prev_e, Uint32(0), rK[t, v])
                        )
                        rK[t, v] = k0
                        p = k0 > best_k
                        best_k = Uint32(cutlass.select_(p, k0, best_k))
                        best_e = Int32(cutlass.select_(p, eid, best_e))
                        best_s = Float32(cutlass.select_(p, rS[t, v], best_s))
                if cutlass.const_expr(j == 0):
                    ck0, ce0, cs0 = best_k, best_e, best_s
                elif cutlass.const_expr(j == 1):
                    ck1, ce1, cs1 = best_k, best_e, best_s
                elif cutlass.const_expr(j == 2):
                    ck2, ce2, cs2 = best_k, best_e, best_s
                else:
                    ck3, ce3, cs3 = best_k, best_e, best_s
                prev_e = best_e
            nrem = Int32(_NCAND)

            # ---- 16 selection rounds (O(1) pop per round) ------------------
            acc = Float32(0.0)
            out_e = Int32(0)
            out_s = Float32(0.0)
            for r in cutlass.range(TOP_K):
                # warp max of the candidate heads (redux.sync.umax on sm100)
                m = cute.arch.warp_redux_sync(ck0, "umax")
                # tie-break: smallest expert id among head key == m
                rev = Int32(
                    cutlass.select_(ck0 == m, Int32(_IDX_REV) - ce0, Int32(0))
                )
                rwin = cute.arch.warp_redux_sync(rev, "max")
                e_win = Int32(_IDX_REV) - rwin
                win_lane = (e_win >> 2) & 31
                # winner's unbiased sigmoid from its owner lane's head
                s_win = cute.arch.shuffle_sync(cs0, offset=win_lane)

                acc += s_win  # serial sum in pick order (all lanes identical)
                picked = lane == r
                out_e = Int32(cutlass.select_(picked, e_win, out_e))
                out_s = Float32(cutlass.select_(picked, s_win, out_s))

                # owner lane pops its head; everyone else keeps its list
                owner = lane == win_lane
                ck0 = Uint32(cutlass.select_(owner, ck1, ck0))
                ce0 = Int32(cutlass.select_(owner, ce1, ce0))
                cs0 = Float32(cutlass.select_(owner, cs1, cs0))
                ck1 = Uint32(cutlass.select_(owner, ck2, ck1))
                ce1 = Int32(cutlass.select_(owner, ce2, ce1))
                cs1 = Float32(cutlass.select_(owner, cs2, cs1))
                ck2 = Uint32(cutlass.select_(owner, ck3, ck2))
                ce2 = Int32(cutlass.select_(owner, ce3, ce2))
                cs2 = Float32(cutlass.select_(owner, cs3, cs2))
                ck3 = Uint32(cutlass.select_(owner, Uint32(0), ck3))
                ce3 = Int32(cutlass.select_(owner, Int32(_IDX_REV), ce3))
                cs3 = Float32(cutlass.select_(owner, Float32(0.0), cs3))
                nrem = Int32(cutlass.select_(owner, nrem - 1, nrem))

                # rare refill: only if the owner lane exhausted its list and
                # more rounds remain (never taken for typical inputs; makes
                # the kernel correct for adversarial >4-picks-per-lane rows)
                if owner & (nrem == 0) & (r < TOP_K - 1):
                    fill_prev = e_win
                    for j in cutlass.range_constexpr(_NCAND):
                        best_k = Uint32(0)
                        best_e = Int32(_IDX_REV)
                        best_s = Float32(0.0)
                        for v in cutlass.range_constexpr(_NV):
                            for t in cutlass.range_constexpr(_VEC):
                                eid = Int32(128 * v + t) + _VEC * lane
                                k0 = Uint32(
                                    cutlass.select_(
                                        eid == fill_prev, Uint32(0), rK[t, v]
                                    )
                                )
                                rK[t, v] = k0
                                p = k0 > best_k
                                best_k = Uint32(cutlass.select_(p, k0, best_k))
                                best_e = Int32(cutlass.select_(p, eid, best_e))
                                best_s = Float32(
                                    cutlass.select_(p, rS[t, v], best_s)
                                )
                        if cutlass.const_expr(j == 0):
                            ck0, ce0, cs0 = best_k, best_e, best_s
                        elif cutlass.const_expr(j == 1):
                            ck1, ce1, cs1 = best_k, best_e, best_s
                        elif cutlass.const_expr(j == 2):
                            ck2, ce2, cs2 = best_k, best_e, best_s
                        else:
                            ck3, ce3, cs3 = best_k, best_e, best_s
                        fill_prev = best_e
                    nrem = Int32(_NCAND)

            # ---- epilogue: lane k < 16 stores pick k ----------------------
            if lane < TOP_K:
                w = out_s * (Float32(ROUTED_SCALING) / acc)
                mIds[row, lane] = out_e
                mScales[row, lane] = w.to(self.out_dtype)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float32: Float32}
_KERNEL_CACHE: dict = {}


def _compile_variant(key, mX, mBias, mIds, mScales, stream):
    in_dt, out_dt, x_align, bias_align, exact_sigmoid = key
    op = KimiK3Routing(in_dt, out_dt, x_align, bias_align, exact_sigmoid)
    return cute.compile(op, mX, mBias, mIds, mScales, stream)


def route_for_kimi_k3_cutedsl(
    logits: torch.Tensor,
    bias: torch.Tensor,
    fmt: str = "fused",
    exact_sigmoid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi-K3 noaux_tc routing (CuTeDSL): top-16 of sigmoid(logits)+bias.

    Args:
        logits: ``[B, 896]`` bf16 or fp32 on CUDA; rows may be strided
            (e.g. a ``[:, :896]`` view of a wider buffer).
        bias: ``[896]`` fp32 contiguous (e_score_correction_bias).
        fmt: ``"fused"`` -> ``(ids int32, scales fp32)``;
             ``"trtllm_gen"`` -> ``(ids int32, scales bf16)``.
        exact_sigmoid: default False computes sigmoid with
            ``ex2.approx + rcp.approx`` (a few fp32 ulp vs torch; pick order
            can swap two experts whose oracle biased scores are bitwise
            equal — ~1 row in 10M for random logits). ``True`` uses
            ``expf + div.rn`` (bit-identical selection to the fp32 torch
            oracle, ~1.4 us slower per warp).

    Returns:
        ``(ids [B, 16] int32, scales [B, 16])`` in pick order (descending
        biased score, ties to the smaller expert index).
    """
    assert fmt in ("fused", "trtllm_gen"), fmt
    assert logits.is_cuda and logits.dim() == 2, "logits must be CUDA [B, E]"
    assert logits.shape[1] == NUM_EXPERTS, f"E must be {NUM_EXPERTS}"
    assert logits.stride(1) == 1, "logits rows must be innermost-contiguous"
    assert logits.dtype in _TORCH2CUTE, "logits must be bf16 or fp32"
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.is_contiguous()
    assert bias.shape == (NUM_EXPERTS,)

    batch = logits.shape[0]
    scale_dtype = torch.float32 if fmt == "fused" else torch.bfloat16
    ids = torch.empty(batch, TOP_K, device=logits.device, dtype=torch.int32)
    scales = torch.empty(batch, TOP_K, device=logits.device, dtype=scale_dtype)
    if batch == 0:
        return ids, scales

    elem = logits.element_size()
    vec_bytes = _VEC * elem
    x_aligned = logits.data_ptr() % vec_bytes == 0 and (
        batch == 1 or (logits.stride(0) * elem) % vec_bytes == 0
    )
    x_align = vec_bytes if x_aligned else elem
    bias_align = 16 if bias.data_ptr() % 16 == 0 else 4

    mX = from_dlpack(logits, assumed_align=x_align).mark_layout_dynamic(leading_dim=1)
    mBias = from_dlpack(bias, assumed_align=bias_align)
    mIds = from_dlpack(ids, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    mScales = from_dlpack(scales, assumed_align=2).mark_layout_dynamic(leading_dim=1)
    # the launch stream is fetched fresh on EVERY call (graph-capture safe)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (_TORCH2CUTE[logits.dtype], _TORCH2CUTE[scale_dtype], x_align,
           bias_align, exact_sigmoid)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _compile_variant(key, mX, mBias, mIds, mScales, stream)
        _KERNEL_CACHE[key] = kernel
    kernel(mX, mBias, mIds, mScales, stream)
    return ids, scales


AVAILABLE = True


def route_cutedsl(
    logits: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    fmt: str = "trtllm_gen",
    exact: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route with exact fp32 sigmoid by default."""
    return route_for_kimi_k3_cutedsl(
        logits,
        gate_bias,
        fmt=fmt,
        exact_sigmoid=exact,
    )


def bench(
    *,
    batch: int = 16,
    fmt: str = "trtllm_gen",
    iters: int = 100,
    repeats: int = 5,
) -> dict[str, float]:
    """Check correctness and graph latency against PyTorch routing."""
    from ._bench import check_regression, graph_time_us, routing_reference

    torch.manual_seed(0)
    logits = torch.randn(
        batch, NUM_EXPERTS + 32, device="cuda", dtype=torch.bfloat16
    )[:, :NUM_EXPERTS]
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32)

    expected = routing_reference(
        logits,
        bias,
        top_k=TOP_K,
        scale=ROUTED_SCALING,
        fmt=fmt,
    )
    actual = route_cutedsl(logits, bias, fmt=fmt, exact=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)

    reference_us = graph_time_us(
        lambda: routing_reference(
            logits,
            bias,
            top_k=TOP_K,
            scale=ROUTED_SCALING,
            fmt=fmt,
        ),
        iters=iters,
        repeats=repeats,
    )
    kernel_us = graph_time_us(
        lambda: route_cutedsl(logits, bias, fmt=fmt, exact=True),
        iters=iters,
        repeats=repeats,
    )
    check_regression(reference_us, kernel_us)
    return {"reference_us": reference_us, "kernel_us": kernel_us}


__all__ = [
    "AVAILABLE",
    "KimiK3Routing",
    "bench",
    "route_cutedsl",
    "route_for_kimi_k3_cutedsl",
]
