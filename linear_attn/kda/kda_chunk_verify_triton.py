"""Chunk-form (way 2) KDA replay_ssm_fused verify in Triton, THREE launches.

Same contract as TRT-LLM's ``fused_recurrent_gated_delta_rule_cached_replay_update``
(the ``trt_closure`` row): checkpoint + solved-update ring in, outputs + ring
appends out, S written only on ring overflow. The internals are the chunked
WY form of KDA.md §2.1 / Appendix A instead of a sequential T-loop.

The split keeps ALL serial [16,*]-tile work off the checkpoint stream:

  prep kernel  (one program per (b,h), never touches S or old_u):
      gate activation + cumsum, L2 norms, the stacked dot operand
      [q̃λ·e^gs; k̃λ·e^gs] (bf16 scratch), the ring correction
      RC = [q̃λ; k̃λ]·K_dec against the decayed record keys (Lemma 1),
      the T×T solve operator M̃ = (I + tril₋₁(diag(β)A))⁻¹ diag(β) via the
      exactly-terminating Neumann series (Lemma 2) stacked with its output
      twin C = tril₀(B)M̃, and the V-independent ring appends (k̃, G).
  state kernel (one program per (b,h), full V): a GEMM-style mainloop
      streams the fp32 checkpoint through ONE tensor-core dot per K block,
      [O_S; W] += stack_kb·S0_kbᵀ, pipelined by num_stages; the epilogue
      folds the ring replay in ([O_S; W] += RC·U_ring) and applies the
      solve as ONE stacked dot: [U; C(V−W)] = [M̃; C]·(V−W). Appends U.
  fold kernel  no-op unless the ring overflowed; commits S_logical without
      taxing the state kernel's register budget.

Design history (tmp/probe_chunk_*.py, B=64 H=96, B200): a monolithic
kernel sits at ~530us — every V-tile repeats the serial prologue, ~200
live registers kill occupancy, and ncu shows 57% DRAM at 18% occupancy;
per-K-block accumulation of ALL six contractions in one loop spills
catastrophically (~1.5ms); computing the ring correction inside the state
kernel costs +120us. This split measures ~95us prep + ~94us state, with
the state pass near its ~78us streaming floor (SOL for its bytes ~60us).

Assumes the bounded (safe) gate: the factored decay e^{G_t}·e^{−G_s} needs
|G| ≲ 30 to stay in fp32 range; lower_bound=-2, T≤8, HIST=16 satisfies it.
All dots accumulate in fp32. T is padded to 16 rows for tl.dot; padded rows
are zeroed (k̃ = β = v = q̃ = 0) so they contribute nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TP = 16  # padded token-window rows (T <= 8 in practice)
TP = tl.constexpr(_TP)


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2)],
    key=["B", "Hn", "T", "K"],
)
@triton.jit
def _chunk_prep_kernel(
    q_ptr, k_ptr, graw_ptr, braw_ptr, alog_ptr, dtb_ptr,
    ok_ptr, oG_ptr, pnat_ptr, bidx_ptr,
    stack_ptr, rc_ptr, mc_ptr,
    lower_bound,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    offs_k = tl.arange(0, K)
    offs_t = tl.arange(0, TP)
    tmask = offs_t < T

    pnat = tl.load(pnat_ptr + b)
    bi = tl.load(bidx_ptr + b)

    # ---- gates, cumulative decays, L2 norms of the T-token window
    tk_base = (b * T + offs_t[:, None]) * (Hn * K) + h * K + offs_k[None, :]
    q = tl.load(q_ptr + tk_base, mask=tmask[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + tk_base, mask=tmask[:, None], other=0.0).to(tl.float32)
    graw = tl.load(graw_ptr + tk_base, mask=tmask[:, None], other=0.0)
    beta = tl.sigmoid(tl.load(braw_ptr + (b * T + offs_t) * Hn + h,
                              mask=tmask, other=0.0))
    beta = tl.where(tmask, beta, 0.0)
    alog = tl.load(alog_ptr + h)
    dtb = tl.load(dtb_ptr + h * K + offs_k)
    g = lower_bound * tl.sigmoid(tl.exp(alog) * (graw + dtb[None, :]))
    g = tl.where(tmask[:, None], g, 0.0)
    Gc = tl.cumsum(g, axis=0)                                       # [TP, K]
    el = tl.exp(Gc)
    kn = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
    qn = q * tl.rsqrt(tl.sum(q * q, axis=1) + 1e-6)[:, None] * (K ** -0.5)
    kn = tl.where(tmask[:, None], kn, 0.0)
    qn = tl.where(tmask[:, None], qn, 0.0)
    kl = kn * el                                                    # k̃⊙λ
    ql = qn * el                                                    # q̃⊙λ

    # ---- ring history: decayed record keys (Lemma 1) and g_start
    offs_r = tl.arange(0, HIST)
    rmask = offs_r < pnat
    ok_base = ok_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * K) + h * K
    rk = tl.load(ok_base + offs_k[None, :], mask=rmask[:, None],
                 other=0.0).to(tl.float32)
    oG_base = oG_ptr + ((b * 2 + bi) * Hn + h) * (K * HIST)
    rG = tl.load(oG_base + offs_k[:, None] * HIST + offs_r[None, :],
                 mask=rmask[None, :], other=0.0)                    # [K, HIST]
    gs = tl.load(oG_base + offs_k * HIST + tl.maximum(pnat - 1, 0))
    gs = tl.where(pnat > 0, gs, 0.0)
    egs = tl.exp(gs)
    rk_dec = tl.trans(rk) * tl.exp(gs[:, None] - rG)                # [K, HIST]
    rk_dec = tl.where(rmask[None, :], rk_dec, 0.0)

    # ---- scratch for the state kernel: the stacked state-dot operand
    # [q̃λ·e^gs; k̃λ·e^gs] and the ring correction RC = [q̃λ; k̃λ]·K_dec
    st_base = stack_ptr + (b * Hn + h) * (2 * TP * K) + offs_t[:, None] * K
    tl.store(st_base + offs_k[None, :],
             (ql * egs[None, :]).to(stack_ptr.dtype.element_ty))
    tl.store(st_base + TP * K + offs_k[None, :],
             (kl * egs[None, :]).to(stack_ptr.dtype.element_ty))
    rc_base = rc_ptr + (b * Hn + h) * (2 * TP * HIST) + offs_t[:, None] * HIST
    tl.store(rc_base + offs_r[None, :],
             tl.dot(ql, rk_dec).to(rc_ptr.dtype.element_ty))
    tl.store(rc_base + TP * HIST + offs_r[None, :],
             tl.dot(kl, rk_dec).to(rc_ptr.dtype.element_ty))

    # ---- Lemma 2 operators, stacked: [M̃; C] with M̃ = (I+N)⁻¹diag(β) and
    # C = tril₀(B)·M̃ (so the state kernel applies both in ONE dot)
    ki = kn / el                                                    # k̃⊘λ
    Amat = tl.dot(kl, tl.trans(ki))
    N = tl.where(offs_t[:, None] > offs_t[None, :], Amat * beta[:, None], 0.0)
    eye = tl.where(offs_t[:, None] == offs_t[None, :], 1.0, 0.0)
    Minv = eye
    p = eye
    for _ in tl.static_range(T - 1):                                # exact: N nilpotent
        p = -tl.dot(N, p)
        Minv = Minv + p
    Mt = Minv * beta[None, :]
    Bmat = tl.dot(ql, tl.trans(ki))
    Bmat = tl.where(offs_t[:, None] >= offs_t[None, :], Bmat, 0.0)
    Cmat = tl.dot(Bmat, Mt)
    mc_base = mc_ptr + (b * Hn + h) * (2 * TP * TP) + offs_t[:, None] * TP
    tl.store(mc_base + offs_t[None, :], Mt.to(mc_ptr.dtype.element_ty))
    tl.store(mc_base + TP * TP + offs_t[None, :],
             Cmat.to(mc_ptr.dtype.element_ty))

    # ---- ring append of the V-independent records (k̃, G); U comes from
    # the state kernel. Overflow appends to the OTHER half at offset 0.
    overflow = pnat + T > HIST
    half = tl.where(overflow, 1 - bi, bi)
    woff = tl.where(overflow, 0, pnat)
    G_rec = Gc + tl.where(overflow, 0.0, 1.0) * gs[None, :]
    ok_w = ok_ptr + ((b * 2 + half) * HIST + woff + offs_t[:, None]) * (Hn * K) + h * K
    tl.store(ok_w + offs_k[None, :], kn.to(ok_ptr.dtype.element_ty),
             mask=tmask[:, None])
    oG_w = oG_ptr + ((b * 2 + half) * Hn + h) * (K * HIST)
    tl.store(oG_w + offs_k[:, None] * HIST + (woff + offs_t[None, :]),
             tl.trans(G_rec), mask=(offs_t[None, :] < T))


@triton.autotune(
    configs=[
        triton.Config({"BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BK": 32}, num_warps=8, num_stages=4),
    ],
    key=["B", "Hn", "T", "K", "Vd"],
)
@triton.jit
def _chunk_state_kernel(
    v_ptr, S_ptr, ou_ptr, pnat_ptr, bidx_ptr,
    stack_ptr, rc_ptr, mc_ptr, o_ptr,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr, Vd: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    offs_c = tl.arange(0, BK)
    offs_v = tl.arange(0, Vd)
    offs_r = tl.arange(0, HIST)
    offs_t = tl.arange(0, TP)
    offs_2t = tl.arange(0, 2 * TP)
    tmask = offs_t < T

    pnat = tl.load(pnat_ptr + b)
    bi = tl.load(bidx_ptr + b)

    # ---- pipelined mainloop: pure checkpoint stream, nothing else
    st_base = stack_ptr + (b * Hn + h) * (2 * TP * K)
    S_base = S_ptr + (b * Hn + h) * (Vd * K)
    big = tl.zeros((2 * TP, Vd), tl.float32)      # rows: [O_S; W]
    for kb in tl.range(0, K, BK):
        cols = kb + offs_c
        stack = tl.load(st_base + offs_2t[:, None] * K + cols[None, :])
        s0 = tl.load(S_base + offs_v[:, None] * K + cols[None, :])
        big += tl.dot(stack, tl.trans(s0.to(tl.bfloat16)))

    # ---- epilogue: ring-u replay, then ONE stacked operator dot
    rc_base = rc_ptr + (b * Hn + h) * (2 * TP * HIST)
    rc = tl.load(rc_base + offs_2t[:, None] * HIST + offs_r[None, :])
    ou_r = ou_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * Vd) + h * Vd
    ru = tl.load(ou_r + offs_v[None, :], mask=(offs_r < pnat)[:, None],
                 other=0.0)
    big += tl.dot(rc, ru)
    oS, w = tl.split(big.reshape(2, TP, Vd).trans(1, 2, 0))

    tv_base = (b * T + offs_t[:, None]) * (Hn * Vd) + h * Vd + offs_v[None, :]
    v = tl.load(v_ptr + tv_base, mask=tmask[:, None], other=0.0).to(tl.float32)
    d = (v - w).to(tl.bfloat16)
    mc_base = mc_ptr + (b * Hn + h) * (2 * TP * TP)
    mc = tl.load(mc_base + offs_2t[:, None] * TP + offs_t[None, :])
    ud = tl.dot(mc, d)                            # rows: [U; C(V−W)]
    u, cd = tl.split(ud.reshape(2, TP, Vd).trans(1, 2, 0))
    o = oS + cd
    tl.store(o_ptr + tv_base, o, mask=tmask[:, None])

    overflow = pnat + T > HIST
    half = tl.where(overflow, 1 - bi, bi)
    woff = tl.where(overflow, 0, pnat)
    ou_w = ou_ptr + ((b * 2 + half) * HIST + woff + offs_t[:, None]) * (Hn * Vd) + h * Vd
    tl.store(ou_w + offs_v[None, :], u.to(ou_ptr.dtype.element_ty),
             mask=tmask[:, None])


@triton.jit
def _chunk_fold_kernel(
    S_ptr, ou_ptr, ok_ptr, oG_ptr, pnat_ptr, bidx_ptr,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr, Vd: tl.constexpr, BK: tl.constexpr,
):
    """Rare ring-overflow fold, its own launch so its registers never tax
    the hot state kernel: commit S_logical = e^gs⊙S0 + Σ_s (k̃_s e^{gs−G_s})u_sᵀ
    (Lemma 1 over the pnat history rows). No-op unless the ring overflowed."""
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    pnat = tl.load(pnat_ptr + b)
    if pnat + T <= HIST:
        return
    bi = tl.load(bidx_ptr + b)
    offs_c = tl.arange(0, BK)
    offs_v = tl.arange(0, Vd)
    offs_r = tl.arange(0, HIST)
    rmask = offs_r < pnat
    ok_r = ok_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * K) + h * K
    oG_r = oG_ptr + ((b * 2 + bi) * Hn + h) * (K * HIST)
    ou_r = ou_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * Vd) + h * Vd
    ru = tl.load(ou_r + offs_v[None, :], mask=rmask[:, None],
                 other=0.0).to(tl.float32)
    S_base = S_ptr + (b * Hn + h) * (Vd * K)
    for kb in tl.range(0, K, BK):
        cols = kb + offs_c
        rk = tl.load(ok_r + cols[None, :], mask=rmask[:, None],
                     other=0.0).to(tl.float32)
        rG = tl.load(oG_r + cols[:, None] * HIST + offs_r[None, :],
                     mask=rmask[None, :], other=0.0)
        gs = tl.load(oG_r + cols * HIST + tl.maximum(pnat - 1, 0))
        gs = tl.where(pnat > 0, gs, 0.0)
        rk_dec = tl.trans(rk) * tl.exp(gs[:, None] - rG)
        rk_dec = tl.where(rmask[None, :], rk_dec, 0.0)
        s0 = tl.load(S_base + offs_v[:, None] * K + cols[None, :])
        s_log = s0 * tl.exp(gs)[None, :] + tl.dot(tl.trans(ru),
                                                  tl.trans(rk_dec))
        tl.store(S_base + offs_v[:, None] * K + cols[None, :], s_log)


_scratch_cache: dict = {}


def _scratch(B, Hn, K, device):
    key = (B, Hn, K, device)
    if key not in _scratch_cache:
        _scratch_cache[key] = (
            torch.empty(B, Hn, 2 * _TP, K, device=device, dtype=torch.bfloat16),
            torch.empty(B, Hn, 2 * _TP, _TP, device=device, dtype=torch.bfloat16),
            torch.empty(B, Hn, 2 * _TP, _TP, device=device, dtype=torch.bfloat16),
        )
    return _scratch_cache[key]


def kda_chunk_verify(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound,
                     S, old_u, old_k, old_G, pnat, buf_idx, out=None):
    """Chunk-form replay_ssm_fused verify (prep + state + fold kernels). Same
    buffers/semantics as the TRT kernel (``trt_closure``) and the b10 replay
    closure: returns o [B,T,H,V] fp32, appends T ring records, writes S only
    on overflow."""
    B, T, Hn, K = q.shape
    Vd = v.shape[-1]
    HIST = old_u.shape[2]
    assert HIST == _TP, "operator stacking assumes HIST == 16"
    if out is None:
        out = torch.empty(B, T, Hn, Vd, device=q.device, dtype=torch.float32)
    stack, rc, mc = _scratch(B, Hn, K, q.device)
    _chunk_prep_kernel[(B * Hn,)](
        q, k, g_raw, beta_raw, A_log, dt_bias,
        old_k, old_G, pnat, buf_idx, stack, rc, mc,
        lower_bound, B, Hn=Hn, T=T, HIST=HIST, K=K,
    )
    _chunk_state_kernel[(B * Hn,)](
        v, S, old_u, pnat, buf_idx,
        stack, rc, mc, out,
        B, Hn=Hn, T=T, HIST=HIST, K=K, Vd=Vd,
    )
    # the fold no-ops in the steady regime; kept off the state kernel's
    # register budget by living in its own (cheap) launch
    _chunk_fold_kernel[(B * Hn,)](
        S, old_u, old_k, old_G, pnat, buf_idx,
        B, Hn=Hn, T=T, HIST=HIST, K=K, Vd=Vd, BK=32, num_warps=4,
    )
    return out
