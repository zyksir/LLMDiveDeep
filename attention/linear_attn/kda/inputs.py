"""Synthetic KDA input contracts and builders shared by all backends/benches.

Two contracts:

- :class:`DecodeInputs`  one token per request, packed serving layout
  (``mixed_qkv [B, (2*HQ*K + HV*V)]`` plus raw gate/beta logits and the
  indexed fp32 state pool) — what a serving engine hands its decode kernel.
- :class:`PrefillInputs` varlen packed ``[1, total_tokens, H, D]`` q/k/v with
  ``cu_seqlens``, raw gate, post-sigmoid beta — what the chunk kernels take.

Both carry RAW gate logits (plus ``A_log``/``dt_bias``): gate activation is
part of each kernel's fused work; backends that need pre-activated inputs
cook them untimed in their builders (see ``kda_attention.activate_kda_gate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from linear_attention import Shape


@dataclass
class DecodeInputs:
    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    beta_logit: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor


@dataclass
class ConvNormInputs:
    """Synthetic conv + output-norm stage tensors for the full decode step.

    Every conv/norm-fused decode row AND the bench's composed correctness
    pipelines build these through :func:`make_conv_norm_inputs` with the same
    seed, so all pipelines run the same conv weights / conv state / output
    gate and their post-norm outputs are directly comparable."""

    conv_weight: torch.Tensor  # [3*H*K, 4] bf16
    conv_state: torch.Tensor  # [B, 3*H*K, 3] bf16, channel-major
    z: torch.Tensor  # [B, H, V] bf16 raw output-gate logits
    norm_weight: torch.Tensor  # [V] fp32 ones


def make_conv_norm_inputs(
    batch: int,
    shape: Shape,
    *,
    device: str = "cuda",
    seed: int = 7,
) -> ConvNormInputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *s: torch.randn(
        *s, device=device, dtype=torch.bfloat16, generator=generator
    )
    packed = 3 * shape.value_heads * shape.key_dim
    return ConvNormInputs(
        conv_weight=(0.1 * randn(packed, 4)).contiguous(),
        conv_state=(0.1 * randn(batch, packed, 3)).contiguous(),
        z=(0.1 * randn(batch, shape.value_heads, shape.value_dim)).contiguous(),
        norm_weight=torch.ones(shape.value_dim, device=device, dtype=torch.float32),
    )


@dataclass
class PrefillInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor


def make_decode_inputs(
    batch: int,
    shape: Shape,
    *,
    device: str = "cuda",
    seed: int = 0,
    state_seed: int | None = None,
) -> DecodeInputs:
    """``state_seed=None`` (benching) leaves the state pool zeroed;
    correctness checks pass a seed so the recurrence starts from a nonzero
    ``S0`` (0.1 * randn) and the decay/delta paths are actually exercised.
    The state gets its own generator so the other tensors are identical
    either way."""
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    randn = lambda *s: torch.randn(*s, device=device, dtype=dtype, generator=generator)
    state = torch.zeros(
        batch,
        shape.value_heads,
        shape.value_dim,
        shape.key_dim,
        device=device,
        dtype=shape.torch_state_dtype,
    )
    if state_seed is not None:
        state_generator = torch.Generator(device=device).manual_seed(state_seed)
        state.normal_(generator=state_generator).mul_(0.1)
    return DecodeInputs(
        mixed_qkv=(0.1 * randn(batch, shape.qkv_dim)).contiguous(),
        raw_gate=(0.5 * randn(batch, shape.value_heads * shape.key_dim) - 1).contiguous(),
        beta_logit=(0.5 * randn(batch, shape.value_heads)).contiguous(),
        A_log=(0.2 * torch.randn(shape.value_heads, device=device, generator=generator)).float(),
        dt_bias=(
            0.1
            * torch.randn(
                shape.value_heads * shape.key_dim,
                device=device,
                generator=generator,
            )
        ).float(),
        state=state,
        state_indices=torch.arange(batch, device=device, dtype=torch.int32),
        cu_seqlens=torch.arange(batch + 1, device=device, dtype=torch.int32),
    )


def make_prefill_inputs(
    batch: int,
    seq_len: int,
    shape: Shape,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> PrefillInputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    tokens = batch * seq_len
    randn = lambda *s: torch.randn(*s, device=device, dtype=dtype, generator=generator)
    return PrefillInputs(
        q=(0.1 * randn(1, tokens, shape.qk_heads, shape.key_dim)).contiguous(),
        k=(0.1 * randn(1, tokens, shape.qk_heads, shape.key_dim)).contiguous(),
        v=(0.1 * randn(1, tokens, shape.value_heads, shape.value_dim)).contiguous(),
        raw_gate=(
            0.5 * randn(1, tokens, shape.value_heads, shape.key_dim) - 1
        ).contiguous(),
        beta=torch.sigmoid(randn(1, tokens, shape.value_heads)).contiguous(),
        A_log=(0.2 * torch.randn(shape.value_heads, device=device, generator=generator)).float(),
        dt_bias=(
            0.1
            * torch.randn(
                shape.value_heads * shape.key_dim,
                device=device,
                generator=generator,
            )
        ).float(),
        state=torch.zeros(
            batch,
            shape.value_heads,
            shape.value_dim,
            shape.key_dim,
            device=device,
            dtype=shape.torch_state_dtype,
        ),
        state_indices=torch.arange(batch, device=device, dtype=torch.int32),
        cu_seqlens=torch.arange(
            0,
            tokens + 1,
            seq_len,
            device=device,
            dtype=torch.int32,
        ),
    )


def split_qkv(
    inputs: DecodeInputs, shape: Shape
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack the serving-layout ``mixed_qkv`` into ``[1, B, H, D]`` q/k/v."""
    q, k, v = torch.split(
        inputs.mixed_qkv,
        [
            shape.qk_heads * shape.key_dim,
            shape.qk_heads * shape.key_dim,
            shape.value_heads * shape.value_dim,
        ],
        dim=-1,
    )
    q = q.view(1, -1, shape.qk_heads, shape.key_dim).contiguous()
    k = k.view(1, -1, shape.qk_heads, shape.key_dim).contiguous()
    v = v.view(1, -1, shape.value_heads, shape.value_dim).contiguous()
    return q, k, v


def beta_logit_of(inputs: PrefillInputs) -> torch.Tensor:
    """Exact preimage of the shared post-sigmoid beta (for kernels fusing σ)."""

    return torch.logit(inputs.beta.float().clamp(1e-4, 1 - 1e-4)).to(
        inputs.beta.dtype
    )
