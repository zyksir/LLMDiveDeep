"""Linear (sub-quadratic) attention, isolated from the benchmark drivers.

Companion to `DSA.py`. Where `DSA.py` holds the *sparse* (still-softmax) path,
this file holds the *linear* path -- the other escape hatch from O(S^2): drop
the softmax and carry a recurrent state instead. It covers the two
linear-attention families that ship in modern open-weight LLMs:

    GDN  -- Gated Delta Net (Qwen3-Next, Qwen3.5-VL, Jet-Nemotron, InternS2).
            Scalar per-(head) forget gate on a delta-rule recurrence.
    KDA  -- Kimi Delta Attention (Kimi-Linear, Moonshot AI). A GDN refinement
            with a *fine-grained diagonal* forget gate (one decay per key
            channel), implemented in the paper via a Diagonal-Plus-Low-Rank
            (DPLR) chunkwise kernel. Kimi-Linear-48B-A3B interleaves 3 KDA
            layers per 1 full MLA layer (the MLA layers use NoPE).

Both are O(S) in sequence length. This module gives two things, mirroring how
`DSA.py` exposes both a fast kernel and a slow-but-portable reference:

    1. A pure-torch recurrent *reference* (`gated_delta_rule_reference`) that
       runs anywhere -- no sglang, no Triton, no FP8. It is the correctness
       oracle and the fallback when the fast kernel is missing (analogous to
       the FP32 indexer fallback in DSA.py). O(S * D^2) naive time loop: correct
       but slow, meant for small S.
    2. Thin wrappers (`register_gdn_kernel` / `register_kda_kernel`) around
       sglang's Flash-Linear-Attention (FLA) Triton kernels for the fast path,
       returning None when sglang is not importable.

The `LinearAttention` nn.Module ties them together: it dispatches to the fast
FLA kernel when available and falls back to the reference otherwise.

Layout convention matches DSA.py and the bench drivers: q, k, v are
[B, S, H, D] bf16 (BSHD). The gate `g` is the (log-space) forget gate --
[B, S, H] for GDN (one scalar per head) or [B, S, H, D] for KDA (one per key
channel). `beta` is the [B, S, H] delta-rule write strength in [0, 1].

References (see SURVEY.md / README.md for the full annotated list):
  - Gated DeltaNet:  https://arxiv.org/abs/2412.06464
  - Qwen3-Next blog: https://qwenlm.github.io/blog/qwen3-next/
  - Kimi Linear:     https://arxiv.org/abs/2510.26692
  - Kimi-Linear repo: https://github.com/MoonshotAI/Kimi-Linear
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make `common` (one dir up) importable regardless of cwd, same as the drivers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLMDIVEDEEP = os.path.dirname(_HERE)
if _LLMDIVEDEEP not in sys.path:
    sys.path.insert(0, _LLMDIVEDEEP)


# ---------------------------------------------------------------------------
# Pure-torch reference recurrence (portable; the correctness oracle)
# ---------------------------------------------------------------------------

@torch.no_grad()
def gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    variant: str = "gdn",
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Naive O(S) gated-delta-rule recurrence. Returns out [B, S, H, Dv].

    State H_t in R^{Dk x Dv} per (batch, head), updated per token with the
    canonical gated delta rule (the same recurrence FLA's
    `chunk_gated_delta_rule` computes chunk-wise, here done token-by-token):

        kh_t = k_t^T H_{t-1}                          # [Dv]  read state by key
        H_t  = a_t * (H_{t-1} - beta_t * k_t (x) kh_t)  + beta_t * k_t (x) v_t
        o_t  = q_t^T H_t                              # [Dv]  read state by query

    where `(x)` is the outer product and `a_t` is the forget gate:
        GDN -> scalar per head       : a_t = exp(g_t),        g_t : [B,H]
        KDA -> diagonal per key chan : a_t = exp(g_t) on Dk,  g_t : [B,H,Dk]
    `g` is expected to be a log-decay (<= 0) so a_t in (0, 1]; the self-test
    passes g = -softplus(.) to stay in range.

    This is intentionally simple and slow -- it materialises no [S, S] score
    matrix, so it is O(S * Dk * Dv) memory-light but has a Python time loop.
    Use it for correctness and small-S runs; use the FLA kernel for speed.
    """
    if variant not in ("gdn", "kda"):
        raise ValueError(f"variant must be 'gdn' or 'kda', got {variant!r}")
    B, S, Hh, Dk = q.shape
    Dv = v.shape[-1]

    qf = q.float()
    kf = k.float()
    if use_qk_l2norm:
        qf = F.normalize(qf, p=2, dim=-1)
        kf = F.normalize(kf, p=2, dim=-1)
    vf = v.float()
    a_all = g.float().exp()                 # forget gate, applied on Dk (or scalar)
    beta_f = beta.float()

    state = torch.zeros(B, Hh, Dk, Dv, device=q.device, dtype=torch.float32)
    out = torch.empty(B, S, Hh, Dv, device=q.device, dtype=torch.float32)

    for t in range(S):
        kt = kf[:, t]                       # [B, H, Dk]
        vt = vf[:, t]                       # [B, H, Dv]
        qt = qf[:, t]                       # [B, H, Dk]
        bt = beta_f[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        if variant == "gdn":
            a_t = a_all[:, t].view(B, Hh, 1, 1)        # scalar decay
        else:  # kda: one decay per key channel, broadcast across Dv
            a_t = a_all[:, t].unsqueeze(-1)            # [B, H, Dk, 1]

        kh = torch.einsum("bhk,bhkv->bhv", kt, state)  # k_t^T H_{t-1}  -> [B,H,Dv]
        old = torch.einsum("bhk,bhv->bhkv", kt, kh)    # k_t (x) kh
        new = torch.einsum("bhk,bhv->bhkv", kt, vt)    # k_t (x) v_t
        state = a_t * (state - bt * old) + bt * new
        out[:, t] = torch.einsum("bhk,bhkv->bhv", qt, state)

    return out


# ---------------------------------------------------------------------------
# Fast-path wrappers around sglang's Flash-Linear-Attention (FLA) kernels
# ---------------------------------------------------------------------------

def register_gdn_kernel() -> Optional[Callable]:
    """Qwen3-Next Gated Delta Net Triton kernel, or None if sglang is absent.

    Returns a `run(q, k, v, g, beta, scale, use_qk_l2norm)` callable wrapping
    `sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule`.
    """
    try:
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    except Exception:  # noqa: BLE001
        return None

    def run(q, k, v, g, beta, scale, use_qk_l2norm=True):
        B, _, Hh, D = q.shape
        initial_state = torch.zeros(B, Hh, D, D, dtype=torch.float32, device=q.device)
        init_idx = torch.arange(B, dtype=torch.int32, device=q.device)
        return chunk_gated_delta_rule(
            q, k, v, g, beta, scale=scale,
            initial_state=initial_state,
            initial_state_indices=init_idx,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )

    return run


def register_kda_kernel() -> Optional[Callable]:
    """Kimi Delta Attention Triton kernel, or None if sglang is absent.

    Wraps `sglang.srt.layers.attention.fla.kda.chunk_kda`. KDA takes a
    fine-grained (per-key-channel) gate `g` of shape [B, S, H, Dk] rather than
    GDN's per-head scalar. The exact kwargs mirror `chunk_gated_delta_rule`;
    we forward defensively and let the reference be the guaranteed path when
    the signature drifts across sglang releases.
    """
    try:
        from sglang.srt.layers.attention.fla.kda import chunk_kda
    except Exception:  # noqa: BLE001
        return None

    def run(q, k, v, g, beta, scale, use_qk_l2norm=True):
        return chunk_kda(
            q, k, v, g, beta, scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )

    return run


# ---------------------------------------------------------------------------
# nn.Module wrapper: fast kernel when available, reference otherwise
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """Gated-delta linear attention (GDN / KDA), kernel-backed with a fallback.

    forward(q, k, v, g, beta) -> out [B, S, H, Dv].

    Dispatches to the sglang FLA kernel for `variant` when it is importable and
    `prefer_fast=True`; otherwise runs the pure-torch reference recurrence, so
    the module is usable in environments without sglang (the reference is slow
    but correct). `.backend` reports which path a forward will take.
    """

    def __init__(
        self,
        variant: str = "gdn",
        *,
        head_dim: int = 128,
        use_qk_l2norm: bool = True,
        prefer_fast: bool = True,
    ):
        super().__init__()
        if variant not in ("gdn", "kda"):
            raise ValueError(f"variant must be 'gdn' or 'kda', got {variant!r}")
        self.variant = variant
        self.head_dim = head_dim
        self.use_qk_l2norm = use_qk_l2norm
        self.scale = head_dim ** -0.5
        reg = register_kda_kernel if variant == "kda" else register_gdn_kernel
        self._fast = reg() if prefer_fast else None
        self.backend = "fla-kernel" if self._fast is not None else "torch-reference"

    @torch.no_grad()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if self._fast is not None:
            out = self._fast(q, k, v, g, beta, self.scale, self.use_qk_l2norm)
            # FLA kernels return either the output tensor or (output, state).
            return out[0] if isinstance(out, tuple) else out
        return gated_delta_rule_reference(
            q, k, v, g, beta,
            variant=self.variant, use_qk_l2norm=self.use_qk_l2norm,
        )


# ---------------------------------------------------------------------------
# Self-test / smoke bench (independently runnable component check)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Small standalone check: run GDN + KDA reference, and the FLA kernel if
    present, and report shapes / timing. Verifies the component in isolation
    before it is wired into any sweep (per the incremental-dev workflow)."""
    from common.kernel_bench import bench_cuda, print_section

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    B, S, H, D = 1, 512, 4, 128
    print_section("LinearAttention self-test (GDN + KDA)")
    print(f"  device={device}  B={B} S={S} H={H} D={D}  dtype={dtype}")

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device)
    beta = torch.rand(B, S, H, dtype=torch.float32, device=device)
    # Log-decay gates kept <= 0 so exp(g) in (0, 1]: g = -softplus(randn).
    g_scalar = -F.softplus(torch.randn(B, S, H, dtype=torch.float32, device=device))
    g_channel = -F.softplus(torch.randn(B, S, H, D, dtype=torch.float32, device=device))

    for variant, g in (("gdn", g_scalar), ("kda", g_channel)):
        mod = LinearAttention(variant=variant, head_dim=D)
        out = mod(q, k, v, g, beta)
        finite = bool(torch.isfinite(out).all().item())
        norm = out.float().norm().item()
        print(
            f"\n  [{variant}]  backend={mod.backend}  out.shape={tuple(out.shape)}  "
            f"finite={finite}  |out|={norm:.3f}"
        )
        if device.type == "cuda":
            ms = bench_cuda(lambda: mod(q, k, v, g, beta), warmup=1, iters=3) / 1000
            print(f"           forward time: {ms:.3f} ms (O(S) recurrence)")


if __name__ == "__main__":
    _selftest()
