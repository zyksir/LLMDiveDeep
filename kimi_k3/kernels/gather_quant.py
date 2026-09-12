"""Fused rank-block interleave -> MXFP8 quantize (one Triton pass).

The prefill fc1 gather returns rank-major blocks ``[world*B, width]``
(rank r's shard at rows ``[r*B, (r+1)*B)``); the expert path consumes
row-major ``(e4m3 [B, world*width], ue8m0 sf [B, world*width/32])``.
The unfused path pays an interleave copy (read+write bf16) plus the
trtllm quantize's separate read pass; this kernel does one read of the
blocks and one write of the (half-size) fp8 payload + scales.

Recipe = bit-exact replica of TRT-LLM ``cvt_warp_fp16_to_mxfp8`` (the
same recipe as ``communication/kernels/comm_cuda.py``'s ag_mxfp8
epilogue): per 32-element block, sf = e8m0(amax/448) rounded UP
(saturating, exponent-only); value = e4m3_satfinite(x * 2^-(sf-127));
amax == 0 -> sf = 0, values = 0. Verified against
``torch.ops.trtllm.mxfp8_quantize`` by ``benchmarks/bench_gather_quant.py``.

``width`` must divide by 32 (448 does: 14 blocks/rank), so MXFP8
blocks never straddle rank boundaries. The sf output is the LINEAR
layout ``[B * world*width/32]`` u8 — exactly what ``quantize_input``
produces (it calls ``mxfp8_quantize(is_sf_swizzled_layout=False)``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_quant_kernel(
    src_ptr, out_ptr, sf_ptr,
    B, WORLD: tl.constexpr, WIDTH: tl.constexpr,
    NBLK: tl.constexpr, NBLK_POW2: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // WORLD
    r = pid % WORLD
    blk = tl.arange(0, NBLK_POW2)[:, None]          # sf blocks in chunk
    lane = tl.arange(0, 32)[None, :]                # elems per sf block
    mask = blk < NBLK
    # src: rank-major [world*B, WIDTH]; this program reads row r*B + b
    x = tl.load(
        src_ptr + (r * B + b) * WIDTH + blk * 32 + lane,
        mask=mask, other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)                # [NBLK_POW2]
    ratio = amax / 448.0
    bits = ratio.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sf = tl.minimum(exp + (mant != 0), 255)         # e8m0 round UP
    sf = tl.where(amax == 0, 0, sf)
    scale = tl.exp2(127.0 - sf.to(tl.float32))
    y = tl.where(amax[:, None] == 0, 0.0, x * scale[:, None])
    y = y.to(tl.float8e4nv)
    # out: row-major [B, world*WIDTH]; rank r's chunk at cols r*WIDTH+
    tl.store(
        out_ptr + b * (WORLD * WIDTH) + r * WIDTH + blk * 32 + lane,
        y, mask=mask,
    )
    # LINEAR sf layout [B, world*NBLK] — what quantize_input feeds the
    # expert backends (it calls mxfp8_quantize(is_sf_swizzled=False)).
    sf_idx = tl.arange(0, NBLK_POW2)
    tl.store(
        sf_ptr + b * (WORLD * NBLK) + r * NBLK + sf_idx,
        sf.to(tl.uint8), mask=sf_idx < NBLK,
    )


def gather_quant_mxfp8(
    gathered: torch.Tensor, world: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 [B, world*width], sf u8 [B*world*width/32]) from rank-major
    bf16 blocks ``[world*B, width]``. Zero host state; one launch."""
    rows, width = gathered.shape
    assert rows % world == 0 and width % 32 == 0
    batch = rows // world
    nblk = width // 32
    out = torch.empty(batch, world * width, device=gathered.device,
                      dtype=torch.float8_e4m3fn)
    sf = torch.empty(batch * world * nblk, device=gathered.device,
                      dtype=torch.uint8)
    _gather_quant_kernel[(batch * world,)](
        gathered, out, sf, batch,
        WORLD=world, WIDTH=width, NBLK=nblk,
        NBLK_POW2=triton.next_power_of_2(nblk),
        num_warps=4,
    )
    return out, sf
