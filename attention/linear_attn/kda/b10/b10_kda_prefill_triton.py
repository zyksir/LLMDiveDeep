"""FlashKDA-style KDA chunk prefill in Triton.

IDENTICAL TO (same end-to-end prefill function; ablation of FlashKDA's algorithm):
  - MoonshotAI FlashKDA CUTLASS ``flash_kda.fwd``
    (`_flash_kda_fwd_prepare` + `_flash_kda_fwd_recurrence`)
  - SGLang / FLA / vLLM / TRT-LLM Triton ``chunk_kda`` pipelines (same math
    result; those are multi-kernel, this is two Triton kernels like FlashKDA)
Same math split (prep + carry); workspace contract differs deliberately
(see below). Not a CuTeDSL champion — the CuTeDSL INT21-port is
``b10_kda_chunk_prefill_cutedsl``.

DROP-IN API? NO — same math role as flash_kda.fwd / chunk_kda, different signature.
  This:   kda_chunk_prefill(q, k, v, raw_gate, beta, s0, chunk=..., A_log=...,
           dt_bias=...) — dense [B,T,...], fuses gate in prep.
  To swap: use the adapter in kda_prefill_register._b10_flashkda_triton_*.

A Triton re-implementation of FlashKDA's two-kernel decomposition (see
references/flash-kda/csrc/smxx/fwd_kernel1.cuh / fwd_kernel2.cuh):

- kernel 1 (``_prep_kernel``, fully parallel): everything chunk-local — q/k L2
  norm, (fused) gate activation, segmented gate cumsum, score matrices,
  ``(I + diag(beta) L)^{-1}`` via the exact squaring-Neumann ladder
  (nilpotency: no substitution, no block composition), and the WY tensors W/U.
- kernel 2 (``_carry_kernel``, grid = B*H x V-splits, sequential over chunks):
  only the inter-chunk state recurrence + outputs. Columns of the state are
  independent, so V is split across programs.

One deliberate deviation from FlashKDA's workspace contract: FlashKDA K1
stores (k_decayed, q_decayed, k_restored, g_total, INV, Mqk) and its K2
recomputes W/U from INV, hiding those extra MMAs behind the state chain with
specialized producer warps. Triton has no warp specialization — measured, that
split inflates the sequential kernel ~1.5x — so here K1 multiplies INV through
to W/U and stores those instead. The parallel/sequential split of the *math*
is identical.

Prep is tuned for register liveness — the occupancy limiter on B200 (NCU: 255
regs/thread, 12.5% occupancy, heavy spills in the naive form). The body is
ordered so the k-side tiles are built and freed before the Neumann ladder, v
and q are loaded only at their use sites, every [C,K] MMA operand / workspace
tensor is bf16 (fp32 exponent range; FlashKDA likewise computes in fp16), only
the gate cumsum stays fp32, and k_restored is derived as ``kd * e^{g_last}``
(a [K] row rescale) instead of a third exp tile. The workspace is
chunk-contiguous per (batch, head) so both kernels stream it. Each dot's
result is stored before the next dot is issued so tcgen05 tensor-memory
accumulators can be reused (C=64 overflows TMEM otherwise).

``chunk`` may be any power of two >= 16 (the Neumann ladder adapts), so C=16
vs C=64 is a one-flag experiment.

Numerical caveat: within-chunk decays use the factored form
``e^{G_t} * e^{-G_j}`` with G clamped at -80 — exact for in-chunk decay ranges
up to e^80, not bit-exact in pathological regimes.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _prep_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    W,
    U,
    QG,
    KGL,
    AQK,
    DEC,
    T,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    LOG2C: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    NC: tl.constexpr,  # chunks per program
):
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // C

    o_c = tl.arange(0, C)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    m_lower = o_c[:, None] > o_c[None, :]
    m_incl = o_c[:, None] >= o_c[None, :]

    # NC chunks per program: gives the software pipeliner independent
    # iterations so chunk i+1's loads overlap chunk i's compute.
    i_c0 = tl.program_id(0) * NC
    for i_c in range(i_c0, min(i_c0 + NC, NT)):
        t0 = i_c * C

        # The body is ordered to minimize concurrently-live [C,K] tiles (the
        # occupancy limiter): k-side tiles are built and freed first, v and q
        # are loaded only at their use sites, and all [C,K] MMA operands /
        # stores are bf16 (fp32 exponent range; FlashKDA computes in fp16 too).
        # Only the gate cumsum stays fp32.
        # int64 scalar bases: beyond ~1M tokens the flattened offsets
        # (tokens*H*K) overflow int32
        row0 = (i_b * T + t0).to(tl.int64)
        p_qk = (row0 * H + i_h) * K + o_c[:, None] * H * K + o_k[None, :]
        b_k = tl.load(k + p_qk)
        b_g = tl.load(g + p_qk).to(tl.float32)
        b_beta = tl.load(beta + (i_b * T + t0) * H + i_h + o_c * H).to(tl.float32)

        r_k = 1.0 / tl.sqrt(tl.sum(b_k.to(tl.float32) * b_k.to(tl.float32), 1) + 1e-12)

        if FUSE_GATE:
            # canonical KDA gate: g = -exp(A_log) * softplus(raw + dt_bias)
            b_x = b_g + tl.load(dt_bias + i_h * K + o_k)[None, :]
            b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
            b_g = -tl.exp(tl.load(A_log + i_h)) * b_sp

        # chunk-total decay = last cumsum row; computing it as a plain sum
        # (before the pointwise clamp) frees b_G earlier
        g_last = tl.maximum(tl.sum(b_g, 0), -80.0)
        b_G = tl.maximum(tl.cumsum(b_g, 0), -80.0)
        e_pos = tl.exp(b_G).to(tl.bfloat16)
        e_neg = tl.exp(-b_G).to(tl.bfloat16)  # b_G dead after this
        b_kd = b_k * e_neg * r_k[:, None].to(tl.bfloat16)
        a_k = b_k * e_pos * (r_k * b_beta)[:, None].to(tl.bfloat16)  # b_k dead

        b_L = tl.dot(a_k, tl.trans(b_kd))
        b_L = tl.where(m_lower, b_L, 0.0)

        # (I + L)^{-1} = (I - L)(I + L^2)(I + L^4)... — exact by nilpotency.
        # [C,C] tiles are tiny; keep the ladder in tf32 for precision.
        eye = (o_c[:, None] == o_c[None, :]).to(tl.float32)
        b_M = eye - b_L
        b_Lp = b_L
        for _ in tl.static_range(LOG2C - 1):
            b_Lp = tl.dot(b_Lp, b_Lp)
            b_M = b_M + tl.dot(b_M, b_Lp)
        b_Mb = b_M.to(tl.bfloat16)

        # Workspace is [B, H, NT, C, *]: chunk-contiguous per (batch, head).
        # Store each dot's result before issuing the next so accumulators
        # (TMEM on Blackwell) can be reused — C=64 overflows TMEM otherwise.
        base = (i_bh * NT + i_c).to(tl.int64) * C
        ws_ck = (base + o_c[:, None]) * K + o_k[None, :]
        ws_cv = (base + o_c[:, None]) * V + o_v[None, :]
        ws_cc = (base + o_c[:, None]) * C + o_c[None, :]

        v_off = (row0 * H + i_h) * V
        b_v = tl.load(v + v_off + o_c[:, None] * H * V + o_v[None, :])
        b_vb = (b_v.to(tl.float32) * b_beta[:, None]).to(tl.bfloat16)
        b_u = tl.dot(b_Mb, b_vb)
        tl.store(U + ws_cv, b_u.to(U.dtype.element_ty))
        b_w = tl.dot(b_Mb, a_k)
        tl.store(W + ws_ck, b_w.to(W.dtype.element_ty))

        b_q = tl.load(q + p_qk)
        r_q = 1.0 / tl.sqrt(tl.sum(b_q.to(tl.float32) * b_q.to(tl.float32), 1) + 1e-12)
        a_q = b_q * e_pos * (r_q * scale)[:, None].to(tl.bfloat16)
        b_aqk = tl.dot(a_q, tl.trans(b_kd))
        b_aqk = tl.where(m_incl, b_aqk, 0.0)
        tl.store(AQK + ws_cc, b_aqk.to(AQK.dtype.element_ty))
        tl.store(QG + ws_ck, a_q)
        # k_restored = kd * e^{g_last} — a [K] row rescale, no extra exp tile
        e_gl = tl.exp(g_last)
        tl.store(KGL + ws_ck, b_kd * e_gl[None, :].to(tl.bfloat16))
        tl.store(DEC + (i_bh * NT + i_c).to(tl.int64) * K + o_k, e_gl)


@triton.jit
def _carry_kernel(
    W,
    U,
    QG,
    KGL,
    AQK,
    DEC,
    S0,
    O,
    ST,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // C

    o_c = tl.arange(0, C)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    p_s = (i_bh * K + o_k[:, None]) * V + o_v[None, :]
    b_S = tl.load(S0 + p_s).to(tl.float32)

    for i_c in range(NT):
        # int64 scalar bases: beyond ~1M tokens the flattened offsets
        # (tokens*H*K) overflow int32
        base = (i_bh * NT + i_c).to(tl.int64) * C
        ws_ck = (base + o_c[:, None]) * K + o_k[None, :]
        ws_cv = (base + o_c[:, None]) * V + o_v[None, :]
        ws_cc = (base + o_c[:, None]) * C + o_c[None, :]
        b_w = tl.load(W + ws_ck)
        b_qg = tl.load(QG + ws_ck)
        b_kgl = tl.load(KGL + ws_ck)
        b_u = tl.load(U + ws_cv)
        b_aqk = tl.load(AQK + ws_cc)
        b_dec = tl.load(DEC + (i_bh * NT + i_c).to(tl.int64) * K + o_k)

        # bf16 tensor-core dots against a bf16 snapshot of the state; the state
        # itself accumulates in fp32.
        b_Sb = b_S.to(tl.bfloat16)
        b_uc = b_u.to(tl.float32) - tl.dot(b_w, b_Sb)
        b_ucb = b_uc.to(tl.bfloat16)
        b_o = tl.dot(b_qg, b_Sb) + tl.dot(b_aqk, b_ucb)
        b_S = b_S * b_dec[:, None] + tl.dot(tl.trans(b_kgl), b_ucb)

        o_off = (
            ((i_b * T + i_c * C).to(tl.int64) * H + i_h) * V
            + o_c[:, None] * H * V
            + o_v[None, :]
        )
        tl.store(O + o_off, b_o.to(O.dtype.element_ty))

    tl.store(ST + p_s, b_S)


def kda_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor,
    *,
    chunk: int = 16,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    prep_chunks_per_program: int | None = None,
    prep_num_warps: int | None = None,
    prep_num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTeDSLGen-contract entry point: [B,T,H,D] bf16 q/k/v, fp32 g/beta,
    [B,H,K,V] fp32 state; returns (o bf16, S_T fp32).

    If ``A_log``/``dt_bias`` are given, ``g`` is treated as the RAW gate and the
    canonical activation ``-exp(A_log)*softplus(g + dt_bias)`` is fused in-kernel.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if T % chunk:
        raise ValueError(f"T={T} must be a multiple of chunk={chunk}")
    NT = T // chunk
    f32 = torch.float32
    bf16 = torch.bfloat16
    dev = q.device
    Wt = torch.empty(B, H, NT, chunk, K, device=dev, dtype=bf16)
    QG = torch.empty_like(Wt)
    KGL = torch.empty_like(Wt)
    Ut = torch.empty(B, H, NT, chunk, V, device=dev, dtype=bf16)
    AQK = torch.empty(B, H, NT, chunk, chunk, device=dev, dtype=bf16)
    DEC = torch.empty(B, H, NT, K, device=dev, dtype=f32)
    O = torch.empty(B, T, H, V, device=dev, dtype=bf16)
    ST = torch.empty(B, H, K, V, device=dev, dtype=f32)

    fuse_gate = A_log is not None
    if fuse_gate and dt_bias is None:
        raise ValueError("fused gate needs both A_log and dt_bias")
    # Swept defaults (B200). chunk > 16 must keep one chunk per program and
    # no pipelining: extra in-flight iterations duplicate the [C,C] ladder
    # accumulators and overflow Blackwell tensor memory.
    if chunk <= 16:
        NC = prep_chunks_per_program or 1
        warps = prep_num_warps or 1
        stages = prep_num_stages or 2
    else:
        NC = 1
        warps = prep_num_warps or 2
        stages = prep_num_stages or 1
    _prep_kernel[(triton.cdiv(NT, NC), B * H)](
        q, k, v, g if fuse_gate else g.float(), beta.float(),
        A_log if fuse_gate else q, dt_bias if fuse_gate else q,
        Wt, Ut, QG, KGL, AQK, DEC,
        T, K**-0.5,
        H=H, K=K, V=V, C=chunk, LOG2C=int(math.log2(chunk)),
        FUSE_GATE=fuse_gate, NC=NC,
        num_warps=warps, num_stages=stages,
    )
    BV = min(64, V)
    _carry_kernel[(B * H, V // BV)](
        Wt, Ut, QG, KGL, AQK, DEC, S0.float(), O, ST,
        T, H=H, K=K, V=V, C=chunk, BV=BV, num_warps=8,
        num_stages=3 if chunk <= 16 else 1,
    )
    return O, ST
