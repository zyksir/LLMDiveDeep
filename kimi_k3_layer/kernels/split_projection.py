"""Split the Kimi-K3 merged projection into latent and shared activations."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def split_merged_projection_kernel(
    gf_ptr,
    latent_ptr,
    act_ptr,
    gf_stride,
    E: tl.constexpr,
    L_SRC: tl.constexpr,
    L_OUT: tl.constexpr,
    I: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    ACT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Split latent and shared activation from the merged projection."""
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = gf_ptr + row * gf_stride

    latent_mask = offs < L_OUT
    latent = tl.load(src + E + offs, mask=latent_mask, other=0.0)
    tl.store(
        latent_ptr + row * LATENT_STRIDE + offs,
        latent,
        mask=latent_mask,
    )

    act_offs = offs - L_OUT
    act_mask = (act_offs >= 0) & (act_offs < I)
    gate = tl.load(
        src + E + L_SRC + act_offs,
        mask=act_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        src + E + L_SRC + I + act_offs,
        mask=act_mask,
        other=0.0,
    ).to(tl.float32)
    activation = gate * tl.sigmoid(gate) * up
    tl.store(
        act_ptr + row * ACT_STRIDE + act_offs,
        activation.to(tl.bfloat16),
        mask=act_mask,
    )
