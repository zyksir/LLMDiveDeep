"""REGISTER / inventory of every KDA speculative-decode (MTP verify) kernel.

This file is not a kernel — it holds the INVENTORY of closures (one per
framework kernel, on identical inputs) that
``benchmarks/bench_kda_spec_verify.py`` checks and times. Spec decode makes
the target model process T = 1+gamma tokens per request (bonus token + gamma
drafts); sampling then accepts only a prefix, known AFTER the forward, so
every scheme must leave the recurrent state recoverable to any accepted
prefix without re-forwarding accepted tokens. ../KDA.md section 2 develops
the taxonomy.

THE THREE SCHEMES (KDA.md section 2):

  save_ssm           schema 1 — cache full states, commit the accepted one
                     (SGLang main, TRT-LLM default)
                     -> :func:`sglang_verify_closure` (with snapshots)
                        + :func:`commit_gather_closure`
  replay_ssm         schema 2 — cache per-token records, replay accepted prefix
    replay_ssm_split   2.1 — cache raw inputs, fold after sampling
                       (SGLang PR #32541 "ReplaySSM")
                       -> :func:`sglang_verify_closure` (no snapshots)
                          + :func:`ring_store_closure`
                          + :func:`fold_replay_closure`
    replay_ssm_fused   2.2 — cache solved updates, replay inside verify
                       (TRT-LLM ``use_replay_state_update``)
                       -> :func:`trt_closure`

Given the same history + draft tokens and full acceptance, every safe-gate
scheme must leave the same committed SSM. SGLang Triton's verify is
softplus-only (timed for latency; not used in correctness).

b10's generated CuTeDSL counterparts live in ``kda/b10/``
(``b10_kda_save_ssm_*`` / ``b10_kda_replay_ssm_*``) and are adapted
through :func:`replay_ssm_closure` / :func:`save_ssm_closure`.

Head count is a module global (``H``, set via :func:`set_heads`) because every
tensor builder and closure reads it; K = V = 128 throughout (Kimi K3).
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from frameworks import _import_trtllm_fla
from kda.kda_attention import kda_recurrent_reference
from kda.kda_replayssm_fold import commit_kda_replayssm_spec

H, K, V = 12, 128, 128

VERIFY_HIST = 16
VERIFY_PNAT = 8  # steady-state ring fill used for perf shapes
VERIFY_LOWER_BOUND = -2.0

# scheme names used in every table (KDA.md section 2)
SAVE_SSM = "save_ssm"                  # schema 1 — cache full states
REPLAY_SSM_SPLIT = "replay_ssm_split"  # schema 2.1 — raw inputs, fold after sampling
REPLAY_SSM = "replay_ssm"              # schema 2.2 — solved updates, replay in verify

# fp32-equivalent bytes persisted per draft token per head, the memory axis of
# the tradeoff (see the scheme inventory above)
STORED_BYTES_PER_TOKEN_HEAD = {
    SAVE_SSM: K * V * 4,                          # full state snapshot
    REPLAY_SSM_SPLIT: 2 * V + 2 * K + 4 * K + 4,  # raw v,k bf16 + g_k fp32 + beta fp32
    REPLAY_SSM: 2 * V + 2 * K + 4 * K,      # u,k_norm bf16 + G fp32
}


def set_heads(heads: int) -> None:
    """Set the module-global head count every builder/closure reads."""
    global H
    H = heads


def _safe_gate(g_raw, A_log, dt_bias):
    """FlashKDA/TRT bounded gate: lb * sigmoid(exp(A_log) * (g_raw + dt_bias))."""
    x = g_raw.float() + dt_bias.view(H, K)
    return VERIFY_LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log.float()).view(H, 1) * x
    )


def verify_tensors(batch: int, draft: int, pnat: int, seed: int):
    """Contract tensors incl. rings with *consistent* content: ``pnat`` history
    tokens are actually replayed from S0 to cook the ring records, exactly as
    previous verify launches would have left them."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    bf = lambda *sh: (
        0.1 * torch.randn(*sh, device=dev, dtype=torch.bfloat16, generator=gen)
    ).contiguous()
    fp = lambda *sh: torch.randn(*sh, device=dev, generator=gen)
    q, k, v = bf(batch, draft, H, K), bf(batch, draft, H, K), bf(batch, draft, H, V)
    g_raw = (0.5 * fp(batch, draft, H, K) - 1).contiguous()
    beta_raw = fp(batch, draft, H).contiguous()
    A_log = torch.zeros(H, device=dev)
    dt_bias = torch.zeros(H * K, device=dev)

    s0 = 0.1 * fp(batch, H, K, V)  # internal math layout [B,H,K,V] fp32
    old_u = torch.zeros(batch, 2, VERIFY_HIST, H, V, device=dev, dtype=torch.bfloat16)
    old_k = torch.zeros(batch, 2, VERIFY_HIST, H, K, device=dev, dtype=torch.bfloat16)
    old_G = torch.zeros(batch, 2, H, K, VERIFY_HIST, device=dev)

    # cook the active ring half by replaying pnat fresh history tokens from s0
    s_run = s0.clone()
    G = torch.zeros(batch, H, K, device=dev)
    if pnat:
        hk = bf(batch, pnat, H, K)
        hv = bf(batch, pnat, H, V)
        hg = _safe_gate((0.5 * fp(batch, pnat, H, K) - 1), A_log, dt_bias)
        hbeta = torch.sigmoid(fp(batch, pnat, H))
        for t in range(pnat):
            kn = torch.nn.functional.normalize(hk[:, t].float(), dim=-1)
            G = G + hg[:, t]
            s_run = s_run * torch.exp(hg[:, t])[..., None]
            u = hbeta[:, t][..., None] * (
                hv[:, t].float() - torch.einsum("bhk,bhkv->bhv", kn, s_run)
            )
            old_u[:, 0, t] = u.to(torch.bfloat16)
            old_k[:, 0, t] = kn.to(torch.bfloat16)
            old_G[:, 0, :, :, t] = G
            s_run = s_run + kn[..., None] * u[:, :, None, :]

    # logical state as the kernels see it: fold of the *quantized* records
    s_logical = s0 * torch.exp(G)[..., None]
    for t in range(pnat):
        dec = torch.exp(G - old_G[:, 0, :, :, t])  # [B,H,K]
        s_logical = s_logical + (old_k[:, 0, t].float() * dec)[..., None] * old_u[
            :, 0, t
        ].float()[:, :, None, :]

    return {
        "q": q, "k": k, "v": v, "g_raw": g_raw, "beta_raw": beta_raw,
        "A_log": A_log, "dt_bias": dt_bias,
        # contract checkpoint layout is [B,H,V,K] (K contiguous), as in TRT
        "S": s0.transpose(-1, -2).contiguous(),
        "old_u": old_u, "old_k": old_k, "old_G": old_G,
        "pnat": torch.full((batch,), pnat, dtype=torch.int32, device=dev),
        "buf_idx": torch.zeros(batch, dtype=torch.int32, device=dev),
        "s_logical": s_logical,  # oracle-only, [B,H,K,V]
    }


def verify_reference(t):
    """Oracle: plain recurrence over the T new tokens from the logical state."""
    g_new = _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"])
    return kda_recurrent_reference(
        t["q"], t["k"], t["v"], g_new, torch.sigmoid(t["beta_raw"].float()),
        initial_state=t["s_logical"],
    )


def snapshot_oracle(t):
    """Per-token state oracle for save_ssm: state AFTER each token (safe gate)."""
    batch, draft = t["q"].shape[:2]
    g = _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"])
    beta = torch.sigmoid(t["beta_raw"].float())
    s = t["s_logical"].clone()  # [B,H,K,V]
    o = torch.empty(batch, draft, H, V, device="cuda")
    snap = torch.empty(batch, draft, H, K, V, device="cuda")
    for tt in range(draft):
        kn = torch.nn.functional.normalize(t["k"][:, tt].float(), dim=-1)
        qn = torch.nn.functional.normalize(t["q"][:, tt].float(), dim=-1)
        s = s * torch.exp(g[:, tt])[..., None]
        u = beta[:, tt][..., None] * (
            t["v"][:, tt].float() - torch.einsum("bhk,bhkv->bhv", kn, s))
        s = s + kn[..., None] * u[:, :, None, :]
        o[:, tt] = torch.einsum("bhk,bhkv->bhv", qn, s) * (K ** -0.5)
        snap[:, tt] = s
    return o, snap


def _load_trt_cached_replay():
    _import_trtllm_fla("op")  # installs the tensorrt_llm shims
    utils = sys.modules["tensorrt_llm._utils"]
    if not hasattr(utils, "get_sm_version"):
        utils.get_sm_version = lambda: 100
        utils.is_sm_100f = lambda: True
    return _import_trtllm_fla("cached_replay")


# ---------------------------------------------------------------------------
# save_ssm / replay_ssm_split shared: the SGLang Triton verify kernel
# ---------------------------------------------------------------------------


def sglang_verify_closure(t, *, snapshots: bool, qkv=None):
    """SGLang's recurrent T-loop verify kernel; with per-step full-state
    snapshots (save_ssm) or as the pure output kernel (replay_ssm_split's
    verify half). Softplus (canonical) gate. ``qkv`` overrides the window's
    q/k/v (each [B,T,H,*] bf16), e.g. with a real conv kernel's output."""
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    batch, draft = t["q"].shape[:2]
    total = batch * draft
    q, k, v = qkv if qkv is not None else (t["q"], t["k"], t["v"])
    sq = q.reshape(1, total, H, K)
    sk = k.reshape(1, total, H, K)
    sv = v.reshape(1, total, H, V)
    # the wrapper reads the per-token gate stride as a.stride()[-2]: KDA wants
    # the flattened [1, T, H*K] layout, not [1, T, H, K]
    sa = t["g_raw"].view(1, total, H * K)
    sb = t["beta_raw"].view(1, total, H)
    cu = torch.arange(0, total + 1, draft, dtype=torch.int32, device="cuda")
    # SGLang's state/snapshot layout is [slots,H,V,K] fp32 (offset v*K + k)
    state = t["s_logical"].transpose(-1, -2).contiguous()
    idx = torch.arange(batch, dtype=torch.int32, device="cuda")
    snap = torch.empty(batch, draft, H, V, K, device="cuda") if snapshots else None

    def run():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=t["A_log"], a=sa, dt_bias=t["dt_bias"],
            softplus_beta=1.0, softplus_threshold=20.0,
            q=sq, k=sk, v=sv, b=sb,
            initial_state_source=state, initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=snap,
            intermediate_state_indices=idx if snapshots else None,
            cache_steps=draft if snapshots else None,
        )

    return run, snap, state


def commit_gather_closure(snap, state, accept_len: int):
    """save_ssm commit: scatter the accepted snapshot into the canonical slot.
    An elementwise copy — shown for completeness of the scheme's launch list."""
    batch = snap.shape[0]
    rows = torch.arange(batch, device="cuda")
    step = torch.full((batch,), accept_len - 1, device="cuda")

    def run():
        # snapshots and canonical states share the [H,V,K] layout: pure gather
        state.copy_(snap[rows, step])

    return run


# ---------------------------------------------------------------------------
# replay_ssm_split: raw-input ring store + exact fold
# ---------------------------------------------------------------------------


def ring_buffers(batch: int, cache_len: int):
    return {
        "rawv": torch.zeros(batch, H, cache_len, V, device="cuda", dtype=torch.bfloat16),
        "rawk": torch.zeros(batch, H, cache_len, K, device="cuda", dtype=torch.bfloat16),
        "gk": torch.zeros(batch, H, cache_len, K, device="cuda"),
        "beta": torch.zeros(batch, H, cache_len, device="cuda"),
    }


def ring_store_closure(t, rings):
    """Write the verify window's raw inputs into the per-slot rings. In the PR
    these stores are fused into the verify kernel; standalone they are four
    strided copies — the natural fusion target for a generated kernel."""
    draft = t["q"].shape[1]
    g_act = _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"])  # [B,T,H,K] fp32
    beta_act = torch.sigmoid(t["beta_raw"].float())

    def run():
        rings["rawv"][:, :, :draft] = t["v"].transpose(1, 2)
        rings["rawk"][:, :, :draft] = t["k"].transpose(1, 2)
        rings["gk"][:, :, :draft] = g_act.transpose(1, 2)
        rings["beta"][:, :, :draft] = beta_act.transpose(1, 2)

    return run


def fold_replay_closure(t, rings, accept_len: int):
    """replay_ssm_split's second kernel: replay the accepted prefix into the
    fp32 checkpoint in place (vendored SGLang PR kernel, bit-exact with decode)."""
    batch = t["q"].shape[0]
    cache_len = rings["rawv"].shape[2]
    # checkpoint in the fold kernel's [slots,H,V,K] layout
    ckpt = t["S"].clone()
    idx = torch.arange(batch, dtype=torch.int32, device="cuda")
    accepts = torch.full((batch,), accept_len, dtype=torch.int32, device="cuda")

    def run():
        commit_kda_replayssm_spec(
            checkpoint_state=ckpt,
            rawv_cache=rings["rawv"], rawk_cache=rings["rawk"],
            gk_cache=rings["gk"], beta_cache=rings["beta"],
            ssm_state_indices=idx, accept_lens=accepts,
            max_cache_len=cache_len, num_k_heads=H,
            null_block_id=-1,
        )

    return run, ckpt


# ---------------------------------------------------------------------------
# replay_ssm_fused: the TRT-LLM cached-replay verify kernel
# ---------------------------------------------------------------------------


def trt_closure(t, qkv=None):
    mod = _load_trt_cached_replay()
    batch = t["S"].shape[0]
    q, k, v = qkv if qkv is not None else (t["q"], t["k"], t["v"])
    bufs = {n: t[n].clone() for n in ("S", "old_u", "old_k", "old_G")}
    old_beta = torch.zeros(batch, 2, H, VERIFY_HIST, device="cuda")
    idx = torch.arange(batch, dtype=torch.int32, device="cuda")

    def run():
        return mod.fused_recurrent_gated_delta_rule_cached_replay_update(
            q=q, k=k, v=v, g=t["g_raw"], beta=t["beta_raw"],
            A_log=t["A_log"], dt_bias=t["dt_bias"],
            lower_bound=VERIFY_LOWER_BOUND,
            ssm_states=bufs["S"], state_indices=idx,
            old_u=bufs["old_u"], old_k=bufs["old_k"], old_G=bufs["old_G"],
            old_beta=old_beta,
            cache_buf_idx=t["buf_idx"], prev_num_accepted_tokens=t["pnat"],
            history_size=VERIFY_HIST, use_qk_l2norm_in_kernel=True,
        )

    return run, bufs


# ---------------------------------------------------------------------------
# b10 generated kernels: adapters onto the same tensors
# ---------------------------------------------------------------------------


def replay_ssm_closure(fn, t):
    """A b10 kda_replay_ssm_fused entry on cloned persistent buffers."""
    bufs = {n: t[n].clone() for n in
            ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")}

    def run():
        return fn(t["q"], t["k"], t["v"], t["g_raw"], t["beta_raw"],
                  t["A_log"], t["dt_bias"], VERIFY_LOWER_BOUND,
                  bufs["S"], bufs["old_u"], bufs["old_k"], bufs["old_G"],
                  bufs["pnat"], bufs["buf_idx"])

    return run, bufs


def save_ssm_closure(fn, t):
    """A b10 kda_save_ssm (save_ssm) entry. Builds the varlen contract from
    the verify window: pre-activated gate/beta, [TT,H,*] packed tokens, read-only
    S0 = the logical state, and a [TT,H,K,V] fp32 snapshot the kernel fills
    (state after every token). Contract from kda_specdec_snapshot_spec.md:
        kda_save_ssm(q, k, v, g, beta, S0, cu_seqlens, snapshot) -> o
    """
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    q = t["q"].reshape(total, H, K).contiguous()
    k = t["k"].reshape(total, H, K).contiguous()
    v = t["v"].reshape(total, H, V).contiguous()
    g = _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"]).reshape(total, H, K).contiguous()
    beta = torch.sigmoid(t["beta_raw"].float()).reshape(total, H).contiguous()
    s0 = t["s_logical"].contiguous()  # [B,H,K,V] fp32, read-only
    cu = torch.arange(0, total + 1, draft, dtype=torch.int32, device="cuda")
    snapshot = torch.empty(total, H, K, V, device="cuda", dtype=torch.float32)

    def run():
        return fn(q, k, v, g, beta, s0, cu, snapshot)

    return run, snapshot


def _conv_tensors(t, seed: int = 0):
    """Raw packed mixed_qkv + conv contract tensors for the conv-fused verify
    entries. The window's q/k/v are treated as the RAW pre-conv stream (the
    conv OUTPUTS therefore differ from t's q/k/v — the composed-reference
    correctness of the fused conv is proven by each module's __main__
    self-test; these closures exist to time the fused kernels on contract
    shapes with realistic memory traffic)."""
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    cdim = 3 * H * K
    x = torch.cat(
        [t["q"].reshape(total, H * K), t["k"].reshape(total, H * K),
         t["v"].reshape(total, H * V)], dim=1).contiguous()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    conv_state = (0.1 * torch.randn(batch, cdim, 3, device="cuda",
                                    generator=gen)).to(torch.bfloat16)
    conv_weight = (0.5 * torch.randn(cdim, 4, device="cuda",
                                     generator=gen)).to(torch.bfloat16)
    conv_win = torch.empty(total, cdim, 3, device="cuda",
                           dtype=torch.bfloat16)
    return x, conv_state, conv_weight, conv_win


def save_ssm_conv_closure(fn, t):
    """A b10 kda_save_ssm_conv entry: save_ssm with the width-4 causal
    conv + SiLU input stage fused in (replaces the separate SGLang
    causal_conv1d_update launch). conv_state is READ-ONLY (rollback
    semantics); per-token raw window snapshots land in conv_win.
        kda_save_ssm_conv(x, g, beta, S0, cu_seqlens, snapshot,
                          conv_state, conv_weight, conv_win) -> o
    """
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    x, conv_state, conv_weight, conv_win = _conv_tensors(t)
    g = _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"]).reshape(total, H, K).contiguous()
    beta = torch.sigmoid(t["beta_raw"].float()).reshape(total, H).contiguous()
    s0 = t["s_logical"].contiguous()
    cu = torch.arange(0, total + 1, draft, dtype=torch.int32, device="cuda")
    snapshot = torch.empty(total, H, K, V, device="cuda", dtype=torch.float32)

    def run():
        return fn(x, g, beta, s0, cu, snapshot, conv_state, conv_weight,
                  conv_win)

    return run, snapshot, conv_win


def replay_ssm_conv_closure(fn, t):
    """A b10 kda_replay_ssm_fused_conv entry: replay_ssm_fused with the conv
    input stage fused in, on cloned persistent buffers (same rollback
    semantics as :func:`replay_ssm_closure`; conv_state READ-ONLY).
    """
    bufs = {n: t[n].clone() for n in
            ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")}
    x, conv_state, conv_weight, conv_win = _conv_tensors(t)

    def run():
        return fn(x, t["g_raw"], t["beta_raw"], t["A_log"], t["dt_bias"],
                  VERIFY_LOWER_BOUND, bufs["S"], bufs["old_u"], bufs["old_k"],
                  bufs["old_G"], bufs["pnat"], bufs["buf_idx"],
                  conv_state, conv_weight, conv_win)

    return run, bufs, conv_win


# ===========================================================================
# The e2e verify PIPELINE (conv4+SiLU -> verify recurrence -> gated RMSNorm):
# shared tensors, torch oracle, and one closure per implementation. Used by
# ``benchmarks/bench_kda_spec_verify_e2e.py``. All implementations consume
# the SAME raw pre-conv inputs and must produce the same post-norm ``o``, the
# same committed SSM, and the same per-scheme side buffers (full-state
# snapshots for save_ssm; solved-update ring records for replay_ssm_fused).
# ===========================================================================

CONV_WIDTH = 4
NORM_EPS = 1e-5


def conv_silu_reference(raw, conv_weight, conv_state):
    """fp32 width-4 depthwise causal conv + SiLU over the draft window.

    raw [B,T,D] bf16, conv_weight [D,4] bf16, conv_state [B,D,3] bf16 (the
    last 3 RAW committed tokens, slot 2 most recent — sglang's
    ``causal_conv1d_update`` roll convention). conv_state is NOT modified.
    Returns [B,T,D] fp32.
    """
    B, T, D = raw.shape
    x = torch.cat([conv_state.float().transpose(1, 2), raw.float()], dim=1)
    acc = torch.zeros((B, T, D), dtype=torch.float32, device=raw.device)
    for j in range(CONV_WIDTH):
        acc = acc + conv_weight[:, j].float() * x[:, j:j + T]
    return acc * torch.sigmoid(acc)


def gated_norm_reference(r, z, w, eps=NORM_EPS):
    """FusedRMSNormGated(activation="sigmoid") in fp32: per (token, head)
    o = r * rsqrt(mean(r^2) + eps) * w * sigmoid(z)."""
    rf = r.float()
    rstd = torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps)
    return rf * rstd * w.float() * torch.sigmoid(z.float())


def _canonical_gate(g_raw, A_log, dt_bias):
    """Canonical KDA gate: -exp(A_log) * softplus(g_raw + dt_bias) — what
    SGLang's verify kernel activates in-kernel."""
    x = g_raw.float() + dt_bias.view(H, K)
    return -torch.exp(A_log.float()).view(H, 1) * F.softplus(x)


def activate_gate(t, lower_bound):
    """[B,T,H,K] fp32 activated gate; ``lower_bound=None`` -> canonical."""
    if lower_bound is None:
        return _canonical_gate(t["g_raw"], t["A_log"], t["dt_bias"])
    return _safe_gate(t["g_raw"], t["A_log"], t["dt_bias"])


def verify_conv_tensors(batch: int, draft: int, pnat: int, seed: int):
    """:func:`verify_tensors` plus the conv input stage and the gated-norm
    epilogue. The draft window's q/k/v are REPLACED by the conv+SiLU of a
    raw packed stream (kept in both fp32 and bf16), so conv-fused one-kernel
    rows (raw in) and conv-composed chains (bf16 post-conv in) run the same
    pipeline."""
    t = verify_tensors(batch, draft, pnat, seed)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed + 7777)
    D = 3 * H * K
    raw = (0.4 * torch.randn(batch, draft, D, device=dev, generator=gen)
           ).to(torch.bfloat16).contiguous()
    conv_weight = (0.5 * torch.randn(D, CONV_WIDTH, device=dev, generator=gen)
                   ).to(torch.bfloat16).contiguous()
    conv_state = (0.5 * torch.randn(batch, D, CONV_WIDTH - 1, device=dev,
                                    generator=gen)
                  ).to(torch.bfloat16).contiguous()
    x = conv_silu_reference(raw, conv_weight, conv_state)      # [B,T,D] fp32
    q32 = x[:, :, :H * K].reshape(batch, draft, H, K)
    k32 = x[:, :, H * K:2 * H * K].reshape(batch, draft, H, K)
    v32 = x[:, :, 2 * H * K:].reshape(batch, draft, H, V)
    t["q"] = q32.to(torch.bfloat16).contiguous()
    t["k"] = k32.to(torch.bfloat16).contiguous()
    t["v"] = v32.to(torch.bfloat16).contiguous()
    t["qkv_f32"] = (q32, k32, v32)
    t["mixed_qkv"] = raw
    t["conv_weight"] = conv_weight
    t["conv_state"] = conv_state
    t["z"] = (0.5 * torch.randn(batch, draft, H, V, device=dev, generator=gen)
              ).to(torch.bfloat16).contiguous()
    t["w"] = (torch.rand(V, device=dev, generator=gen) * 1.5 + 0.25
              ).float().contiguous()
    return t


def e2e_oracle(t, *, lower_bound, bf16_conv: bool, bf16_r: bool):
    """Full-pipeline torch oracle on the shared tensors: conv -> gate
    activation -> per-token recurrence from the logical state -> gated
    RMSNorm. The two flags round to bf16 where a pipeline materializes
    tensors between kernel launches — ``bf16_conv`` at the conv->recurrence
    boundary, ``bf16_r`` at the recurrence->norm boundary. Fully fused
    kernels keep both fp32 on-chip (False, False); a kernel taking post-conv
    bf16 q/k/v with the norm fused is (True, False); a 3-launch chain is
    (True, True). Match each row to its pipeline's rounding.

    Returns (o_post [B,T,H,V] fp32, snap [B,T,H,K,V] fp32, S_final
    [B,H,K,V] fp32).
    """
    batch, draft = t["q"].shape[:2]
    q, k, v = t["qkv_f32"]
    if bf16_conv:
        q, k, v = (x.to(torch.bfloat16).float() for x in (q, k, v))
    g = activate_gate(t, lower_bound)
    beta = torch.sigmoid(t["beta_raw"].float())
    s = t["s_logical"].clone()                                 # [B,H,K,V]
    r = torch.empty(batch, draft, H, V, device="cuda")
    snap = torch.empty(batch, draft, H, K, V, device="cuda")
    for tt in range(draft):
        kn = torch.nn.functional.normalize(k[:, tt], dim=-1)
        qn = torch.nn.functional.normalize(q[:, tt], dim=-1)
        s = s * torch.exp(g[:, tt])[..., None]
        u = beta[:, tt][..., None] * (
            v[:, tt] - torch.einsum("bhk,bhkv->bhv", kn, s))
        s = s + kn[..., None] * u[:, :, None, :]
        r[:, tt] = torch.einsum("bhk,bhkv->bhv", qn, s) * (K ** -0.5)
        snap[:, tt] = s
    if bf16_r:
        r = r.to(torch.bfloat16).float()
    return gated_norm_reference(r, t["z"], t["w"]), snap, s


def logical_s_from_ring(bufs, n: int):
    """Logical SSM after ``n`` new ring records (replay kernels do not write
    S in the steady regime): fold the active ring half into the checkpoint,
    same unrolled form as the history cook in :func:`verify_tensors`."""
    bi = int(bufs["buf_idx"][0].item()) if "buf_idx" in bufs else 0
    s = bufs["S"].transpose(-1, -2).float().clone()            # [B,H,K,V]
    G_last = bufs["old_G"][:, bi, :, :, n - 1]
    s = s * torch.exp(G_last)[..., None]
    for step in range(n):
        dec = torch.exp(G_last - bufs["old_G"][:, bi, :, :, step])
        s = s + (bufs["old_k"][:, bi, step].float() * dec)[..., None] * bufs[
            "old_u"
        ][:, bi, step].float()[:, :, None, :]
    return s


# --- b10 one-kernel rows ---------------------------------------------------


def b10_save_ssm_conv_gated_closure(t, lower_bound):
    """The b10 save_ssm e2e row: ONE kernel does conv + recurrence with
    per-token full-state snapshots + gated RMSNorm. Takes pre-activated
    gate/beta, so it runs under either gate flavor (``lower_bound=None`` ->
    canonical, matching SGLang; a float -> safe gate, matching TRT).
    Returns (run, snapshot [TT,H,K,V] fp32)."""
    from kda.b10.b10_kda_save_ssm_conv_gated_cutedsl import (
        kda_save_ssm_conv_gated,
    )

    batch, draft = t["q"].shape[:2]
    total = batch * draft
    mixed = t["mixed_qkv"].reshape(total, 3 * H * K).contiguous()
    g = activate_gate(t, lower_bound).reshape(total, H, K).contiguous()
    beta = torch.sigmoid(t["beta_raw"].float()).reshape(total, H).contiguous()
    s0 = t["s_logical"].contiguous()
    cu = torch.arange(0, total + 1, draft, dtype=torch.int32, device="cuda")
    snapshot = torch.empty(total, H, K, V, device="cuda")
    z = t["z"].reshape(total, H, V).contiguous()

    def run():
        return kda_save_ssm_conv_gated(
            mixed, t["conv_weight"], t["conv_state"], g, beta, s0, cu,
            snapshot, z, t["w"])

    return run, snapshot


def b10_save_ssm_gated_closure(t, lower_bound):
    """The conv-less build of the same kernel (POST-conv bf16 q/k/v in):
    the ladder row that pays a separate conv launch."""
    from kda.b10.b10_kda_save_ssm_gated_cutedsl import (
        kda_save_ssm_gated,
    )

    batch, draft = t["q"].shape[:2]
    total = batch * draft
    q = t["q"].reshape(total, H, K).contiguous()
    k = t["k"].reshape(total, H, K).contiguous()
    v = t["v"].reshape(total, H, V).contiguous()
    g = activate_gate(t, lower_bound).reshape(total, H, K).contiguous()
    beta = torch.sigmoid(t["beta_raw"].float()).reshape(total, H).contiguous()
    s0 = t["s_logical"].contiguous()
    cu = torch.arange(0, total + 1, draft, dtype=torch.int32, device="cuda")
    snapshot = torch.empty(total, H, K, V, device="cuda")
    z = t["z"].reshape(total, H, V).contiguous()

    def run():
        return kda_save_ssm_gated(
            q, k, v, g, beta, s0, cu, snapshot, z, t["w"])

    return run, snapshot


def b10_replay_ssm_conv_gated_closure(t, qkv=None):
    """The b10 replay-SSM e2e row: ONE kernel does conv + verify
    recurrence with ring append + gated RMSNorm, on cloned persistent
    buffers (rollback semantics: S written only on overflow, pnat/buf_idx
    untouched). Safe gate (in-kernel activation, like TRT). ``qkv`` switches
    to the conv-less build on POST-conv q/k/v. Returns (run, bufs)."""
    from kda.b10.b10_kda_replay_ssm_conv_gated_cutedsl import (
        kda_replay_ssm_conv_gated,
    )
    from kda.b10.b10_kda_replay_ssm_gated_cutedsl import kda_replay_ssm_gated

    bufs = {n: t[n].clone() for n in
            ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")}
    common = (t["g_raw"], t["beta_raw"], t["A_log"], t["dt_bias"],
              VERIFY_LOWER_BOUND, bufs["S"], bufs["old_u"], bufs["old_k"],
              bufs["old_G"], bufs["pnat"], bufs["buf_idx"], t["z"], t["w"])

    if qkv is None:
        def run():
            return kda_replay_ssm_conv_gated(
                t["mixed_qkv"], t["conv_weight"], t["conv_state"], *common)
    else:
        q, k, v = (x.contiguous() for x in qkv)

        def run():
            return kda_replay_ssm_gated(q, k, v, *common)

    return run, bufs


def b10_replay_ssm_conv_gated_wychunk_closure(t):
    """Accepted b10 CuTeDSL WY kernel: conv + replay-SSM + gated norm."""
    from kda.b10.b10_kda_replay_ssm_conv_gated_wychunk_cutedsl import (
        kda_replay_ssm_conv_gated_wychunk,
    )

    batch, draft = t["q"].shape[:2]
    raw_q, raw_k, raw_v = (
        x.view(batch, draft, H, K)
        for x in t["mixed_qkv"].split(H * K, dim=-1)
    )
    assert not raw_q.is_contiguous()
    assert not raw_k.is_contiguous()
    assert not raw_v.is_contiguous()
    bufs = {
        name: t[name].clone()
        for name in ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")
    }
    out = torch.empty(
        batch, draft, H, V, device="cuda", dtype=torch.bfloat16
    )

    def run():
        return kda_replay_ssm_conv_gated_wychunk(
            raw_q,
            raw_k,
            raw_v,
            t["conv_weight"],
            t["conv_state"],
            t["g_raw"],
            t["beta_raw"],
            t["A_log"],
            t["dt_bias"],
            VERIFY_LOWER_BOUND,
            bufs["S"],
            bufs["old_u"],
            bufs["old_k"],
            bufs["old_G"],
            bufs["pnat"],
            bufs["buf_idx"],
            t["z"],
            t["w"],
            out=out,
        )

    return run, bufs


# --- framework chains (the production pipelines, N kernel launches) --------


def sglang_conv_closure(t):
    """The separate sglang ``causal_conv1d_update`` launch (Triton, from the
    installed wheel) on the raw window: the conv leg every non-conv-fused
    chain pays. conv_state is cloned because the kernel rolls it in place.
    Returns (run -> [B,D,T] bf16 conv+SiLU output, fresh_input_maker)."""
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )

    def make():
        return (t["mixed_qkv"].transpose(1, 2).contiguous(),
                t["conv_state"].clone())

    x_bdt, cs = make()

    def run():
        return causal_conv1d_update(x_bdt, cs, t["conv_weight"], None,
                                    activation="silu")

    return run, make


def sglang_norm_closure(z_like, w):
    """The separate sglang ``FusedRMSNormGated`` launch. Returns
    run(r_2d, z_2d) -> post-norm bf16 (NB: overwrites r_2d in place)."""
    from sglang.srt.layers.attention.fla.fused_norm_gate import (
        FusedRMSNormGated,
    )

    mod = FusedRMSNormGated(V, eps=NORM_EPS, activation="sigmoid",
                            device="cuda", dtype=torch.float32)
    with torch.no_grad():
        mod.weight.copy_(w.float())

    def run(r_2d, z_2d):
        return mod(r_2d, z_2d)

    return run


def sglang_save_ssm_chain_closure(t):
    """SGLang's production save_ssm e2e pipeline: THREE kernel launches
    (causal_conv1d_update -> verify T-loop with full-state snapshots ->
    FusedRMSNormGated) timed back-to-back. The launches run on shape-correct
    staged buffers with no host glue between them — production connects them
    with strided views, so the kernel sum is the honest e2e latency; the
    data-connected output for correctness comes from
    :func:`sglang_save_ssm_stitched`. Returns (run, snap)."""
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    conv_run, _ = sglang_conv_closure(t)
    verify_run, snap, _ = sglang_verify_closure(t, snapshots=True)
    norm_run = sglang_norm_closure(t["z"], t["w"])
    r_2d = torch.empty(total * H, V, device="cuda", dtype=torch.bfloat16)
    z_2d = t["z"].reshape(total * H, V)

    def run():
        conv_run()
        verify_run()
        return norm_run(r_2d, z_2d)

    return run, snap


def sglang_save_ssm_stitched(t):
    """One-shot data-connected sglang chain (host glue copies allowed, not
    timed): real conv kernel -> verify with snapshots on the conv output ->
    real gated-norm kernel. Returns (o [B,T,H,V] fp32 post-norm,
    snap [B,draft,H,V,K] fp32)."""
    batch, draft = t["q"].shape[:2]
    conv_run, make = sglang_conv_closure(t)
    x_bdt, cs = make()
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )
    out = causal_conv1d_update(x_bdt, cs, t["conv_weight"], None,
                               activation="silu")
    xt = out.transpose(1, 2).contiguous()                   # [B,T,D] bf16
    q = xt[:, :, :H * K].reshape(batch, draft, H, K).contiguous()
    k = xt[:, :, H * K:2 * H * K].reshape(batch, draft, H, K).contiguous()
    v = xt[:, :, 2 * H * K:].reshape(batch, draft, H, V).contiguous()
    verify_run, snap, _ = sglang_verify_closure(t, snapshots=True,
                                                qkv=(q, k, v))
    o = verify_run()
    o = (o[0] if isinstance(o, tuple) else o).reshape(batch, draft, H, V)
    norm_run = sglang_norm_closure(t["z"], t["w"])
    o_post = norm_run(o.reshape(-1, V).float().clone(),
                      t["z"].reshape(-1, V))
    return o_post.float().reshape(batch, draft, H, V), snap


def trt_replay_chain_closure(t):
    """TRT-LLM's production replay_ssm_fused e2e pipeline: THREE launches
    (conv -> cached-replay verify -> gated norm), timed back-to-back on
    staged buffers (same convention as the sglang chain; TRT also runs a
    separate conv update and norm around its verify kernel).
    Returns (run, bufs)."""
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    conv_run, _ = sglang_conv_closure(t)
    verify_run, bufs = trt_closure(t)
    norm_run = sglang_norm_closure(t["z"], t["w"])
    r_2d = torch.empty(total * H, V, device="cuda", dtype=torch.bfloat16)
    z_2d = t["z"].reshape(total * H, V)

    def run():
        conv_run()
        verify_run()
        return norm_run(r_2d, z_2d)

    return run, bufs


def trt_replay_stitched(t):
    """One-shot data-connected TRT chain: real conv kernel -> cached-replay
    verify on the conv output -> real gated-norm kernel. Returns
    (o [B,T,H,V] fp32 post-norm, bufs)."""
    batch, draft = t["q"].shape[:2]
    _, make = sglang_conv_closure(t)
    x_bdt, cs = make()
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )
    out = causal_conv1d_update(x_bdt, cs, t["conv_weight"], None,
                               activation="silu")
    xt = out.transpose(1, 2).contiguous()
    q = xt[:, :, :H * K].reshape(batch, draft, H, K).contiguous()
    k = xt[:, :, H * K:2 * H * K].reshape(batch, draft, H, K).contiguous()
    v = xt[:, :, 2 * H * K:].reshape(batch, draft, H, V).contiguous()
    verify_run, bufs = trt_closure(t, qkv=(q, k, v))
    o = verify_run()
    o = (o[0] if isinstance(o, tuple) else o).reshape(batch, draft, H, V)
    norm_run = sglang_norm_closure(t["z"], t["w"])
    o_post = norm_run(o.reshape(-1, V).float().clone(),
                      t["z"].reshape(-1, V))
    return o_post.float().reshape(batch, draft, H, V), bufs


# --- chunk-form replay verify (way 2 of KDA.md section 2.1, Triton) ---------


def triton_chunk_closure(t, qkv=None):
    """The chunk-form (WY) replay verify kernel: same buffers and rollback
    semantics as :func:`trt_closure`, but the T-loop is replaced by
    tensor-core GEMMs + an exactly-terminating Neumann solve
    (``kda/kda_chunk_verify_triton.py``, KDA.md Appendix A)."""
    from kda.kda_chunk_verify_triton import kda_chunk_verify

    batch, draft = t["q"].shape[:2]
    q, k, v = qkv if qkv is not None else (t["q"], t["k"], t["v"])
    bufs = {n: t[n].clone() for n in
            ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")}
    out = torch.empty(batch, draft, H, V, device="cuda")

    def run():
        return kda_chunk_verify(
            q, k, v, t["g_raw"], t["beta_raw"], t["A_log"], t["dt_bias"],
            VERIFY_LOWER_BOUND, bufs["S"], bufs["old_u"], bufs["old_k"],
            bufs["old_G"], bufs["pnat"], bufs["buf_idx"], out=out)

    return run, bufs


def triton_chunk_chain_closure(t):
    """The chunk kernel in the SAME three-launch pipeline as TRT's
    (conv -> verify -> gated norm), so the two chains differ only in the
    verify kernel. Returns (run, bufs)."""
    batch, draft = t["q"].shape[:2]
    total = batch * draft
    conv_run, _ = sglang_conv_closure(t)
    verify_run, bufs = triton_chunk_closure(t)
    norm_run = sglang_norm_closure(t["z"], t["w"])
    r_2d = torch.empty(total * H, V, device="cuda", dtype=torch.bfloat16)
    z_2d = t["z"].reshape(total * H, V)

    def run():
        conv_run()
        verify_run()
        return norm_run(r_2d, z_2d)

    return run, bufs


def triton_chunk_stitched(t):
    """One-shot data-connected chunk chain: real conv kernel -> chunk verify
    on the conv output -> real gated-norm kernel. Returns
    (o [B,T,H,V] fp32 post-norm, bufs)."""
    batch, draft = t["q"].shape[:2]
    _, make = sglang_conv_closure(t)
    x_bdt, cs = make()
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )
    out = causal_conv1d_update(x_bdt, cs, t["conv_weight"], None,
                               activation="silu")
    xt = out.transpose(1, 2).contiguous()
    q = xt[:, :, :H * K].reshape(batch, draft, H, K).contiguous()
    k = xt[:, :, H * K:2 * H * K].reshape(batch, draft, H, K).contiguous()
    v = xt[:, :, 2 * H * K:].reshape(batch, draft, H, V).contiguous()
    verify_run, bufs = triton_chunk_closure(t, qkv=(q, k, v))
    o = verify_run()
    norm_run = sglang_norm_closure(t["z"], t["w"])
    o_post = norm_run(o.reshape(-1, V).to(torch.bfloat16).float(),
                      t["z"].reshape(-1, V))
    return o_post.float().reshape(batch, draft, H, V), bufs
