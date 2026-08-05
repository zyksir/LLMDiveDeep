"""Fused strided-slice -> MXFP8 quantize (one Triton pass).

The merged input GEMM leaves the latent as a STRIDED column slice of
gf; the trtllm-gen quantize op demands contiguous input, so the
non-fc1-shard path historically paid contiguous-copy (~6.5 us at B=8)
+ quantize_with_block_size (~4.3 us). This kernel replaces both with
ONE pass that reads the strided bf16 slice and writes the (e4m3,
ue8m0 per-32 block scales) pair `run_moe` consumes - the same fusion
the fc1-shard AG carried, but WITHOUT any collective (works at TP1,
and at bs=32..80 where the AG's wire cost killed the shard).

Recipe = bit-exact replica of TRT-LLM's ``cvt_warp_fp16_to_mxfp8``
(same as comm_cuda's ag_mxfp8 epilogue): per 32-element block,
sf = e8m0(amax/448) rounded UP (saturating, exponent-only), value =
e4m3_satfinite(x * 2^-(sf-127)); amax == 0 -> sf = 0, values = 0.
Verified against ``torch.ops.trtllm.mxfp8_quantize`` by
``debug/quant_slice_check.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_slice_kernel(
    src_ptr, out_ptr, sf_ptr,
    src_stride,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,  # elements per program (multiple of 32)
):
    row = tl.program_id(0)
    col0 = tl.program_id(1) * BLOCK
    offs = col0 + tl.arange(0, BLOCK)
    x = tl.load(src_ptr + row * src_stride + offs).to(tl.float32)

    # per-32 block amax -> ue8m0 scale, rounded UP (exponent of the
    # smallest power of two >= amax/448; ties/inexact round up)
    g = tl.reshape(x, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(g), axis=1)  # [BLOCK//32]
    r = amax * (1.0 / 448.0)
    bits = r.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    # round toward +inf in exponent space: any mantissa bits bump exp.
    # (subnormal r (exp==0) also lands here -> exp 1 == 2^-126, the
    # smallest e8m0 step, matching cudaRoundPosInf of a nonzero r.)
    exp = tl.where(mant != 0, exp + 1, exp)
    exp = tl.minimum(exp, 254)  # e8m0 satfinite (0xFF = NaN reserved)
    sf = tl.where(amax > 0.0, exp, 0)
    tl.store(sf_ptr + row * (COLS // 32) + col0 // 32
             + tl.arange(0, BLOCK // 32), sf.to(tl.uint8))

    # value scale = 2^-(sf-127) (0 when amax == 0), applied elementwise
    scale_bits = tl.where(amax > 0, (254 - exp) << 23, 0)
    scale = scale_bits.to(tl.float32, bitcast=True)  # [BLOCK//32]
    q = g * scale[:, None]
    q = tl.reshape(q, (BLOCK,))
    tl.store(out_ptr + row * COLS + offs,
             q.to(tl.float8e4nv))


def quant_slice_mxfp8(src: torch.Tensor) -> tuple[torch.Tensor,
                                                  torch.Tensor]:
    """[rows, cols] bf16 (row-strided OK) -> (e4m3 [rows, cols] viewed
    uint8-compatible fp8, ue8m0 scales [rows, cols//32] uint8)."""
    rows, cols = src.shape
    assert cols % 32 == 0 and src.stride(1) == 1
    out = torch.empty(rows, cols, device=src.device,
                      dtype=torch.float8_e4m3fn)
    sf = torch.empty(rows, cols // 32, device=src.device,
                     dtype=torch.uint8)
    # largest power-of-2 divisor of cols, capped (tl.arange needs a
    # power of two; 3584 = 512 * 7 -> block 512, grid 7 per row)
    block = min(cols & -cols, 2048)
    assert cols % block == 0 and block % 32 == 0
    _quant_slice_kernel[(rows, cols // block)](
        src, out, sf, src.stride(0), COLS=cols, BLOCK=block,
        num_warps=4,
    )
    return out, sf
