"""KDA (Kimi Delta Attention), the naive PyTorch implementation.

This file is the pedagogical core of the package: it contains ONLY plain
PyTorch — no Triton, no CUTLASS, no framework imports. Every kernel registered
in ``kda_decode_register.py`` / ``kda_prefill_register.py`` /
``kda_verify_register.py`` computes exactly
this math (up to gate parameterization and numerics); the benches use these
functions as the correctness oracle.

The recurrence (one token t, one head):

    S <- diag(exp(g_t)) S            per-KEY-CHANNEL decay (the "delta" of
                                     KDA vs GDN's single scalar per head)
    u_t = beta_t * (v_t - k_t^T S)   delta-rule correction against the
                                     already-decayed state
    S <- S + k_t u_t^T
    o_t = q_t^T S / sqrt(K)

with L2-normalized q/k and state S of shape [K, V] per head. The full math
walkthrough (chunked form, WY representation, spec-decode taxonomy) lives in
../KDA.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# FlashKDA / TRT-LLM "safe gate" bound used across the repo's benches.
SAFE_GATE_LOWER_BOUND = -5.0


def activate_kda_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    *,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Convert raw gate logits to per-channel log-decays ``g <= 0``.

    The original (canonical) KDA parameterization is
    ``g = -exp(A_log) * softplus(raw_gate + dt_bias)``.
    Safe-gate KDA (FlashKDA, TRT-LLM) instead uses the bounded
    ``g = lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``.
    """

    x = raw_gate.float()
    if dt_bias is not None:
        x = x + dt_bias.float().reshape(1, 1, *x.shape[-2:])
    rate = A_log.float().exp().reshape(1, 1, -1, 1)
    if lower_bound is None:
        return -rate * F.softplus(x)
    if lower_bound >= 0:
        raise ValueError("lower_bound must be negative")
    return lower_bound * torch.sigmoid(rate * x)


@torch.no_grad()
def conv4_silu_reference(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """One decode step of the width-4 causal conv + SiLU on the packed
    pre-conv q/k/v row (the input stage of the KDA.md §1.1 decode step).

    ``x`` ``[B, C]`` is the current token, ``conv_state`` ``[B, C, 3]`` the
    previous 3 tokens (channel-major), ``conv_weight`` ``[C, 4]``. Returns
    the convolved row in ``x``'s dtype — the unfused chains materialize it
    in bf16, which is why composed pipelines differ from the fused kernels
    (fp32 on-chip) at bf16-rounding level."""

    window = torch.cat([conv_state.float(), x.float().unsqueeze(-1)], dim=-1)
    out = (window * conv_weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias.float()
    return F.silu(out).to(x.dtype)


@torch.no_grad()
def gated_rmsnorm_reference(
    o: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sigmoid-gated output RMSNorm (the epilogue of the KDA.md §1.1 decode
    step): ``o * rsqrt(mean(o^2) + eps) * weight * sigmoid(z)``. All of the
    fused kernels (b10, SGLang, TRT-LLM) implement exactly this formula with
    eps = 1e-5."""

    of = o.float()
    scale = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    return (of * scale * weight.float() * torch.sigmoid(z.float())).to(o.dtype)


@torch.no_grad()
def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    normalize_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact token recurrence used as the small-shape correctness oracle.

    Inputs are BSHD (``[B, T, H, D]``); ``log_decay`` is the already-activated
    gate (see :func:`activate_kda_gate`); ``beta`` is post-sigmoid. The state
    is ``[B, HV, K, V]`` fp32. KDA first decays each key channel, then applies
    the delta-rule correction against that decayed state:

    ``S <- diag(exp(g)) S``
    ``S <- S + beta * k (v - k^T S)^T``
    ``o <- q^T S / sqrt(K)``
    """

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if HV % H:
        raise ValueError("value_heads must be divisible by qk_heads")
    groups = HV // H

    qf, kf, vf = q.float(), k.float(), v.float()
    if normalize_qk:
        qf = F.normalize(qf, dim=-1)
        kf = F.normalize(kf, dim=-1)
    qf = qf.repeat_interleave(groups, dim=2) * (K**-0.5)
    kf = kf.repeat_interleave(groups, dim=2)
    state = torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        state.copy_(initial_state.float())
    out = torch.empty(B, T, HV, V, device=q.device, dtype=torch.float32)

    for t in range(T):
        kt, qt, vt = kf[:, t], qf[:, t], vf[:, t]
        state.mul_(log_decay[:, t].float().exp().unsqueeze(-1))
        prediction = torch.einsum("bhk,bhkv->bhv", kt, state)
        residual = vt - prediction
        state.add_(
            torch.einsum(
                "bhk,bhv->bhkv",
                kt,
                beta[:, t].float().unsqueeze(-1) * residual,
            )
        )
        out[:, t] = torch.einsum("bhk,bhkv->bhv", qt, state)
    return out.to(v.dtype), state
