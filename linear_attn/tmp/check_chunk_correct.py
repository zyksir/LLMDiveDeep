"""Thorough correctness for kda_chunk_verify vs fp32 oracle + b10 for-loop.

Checks o, ring appends (old_u / old_k / old_G), and S-on-overflow across
pnat ∈ {0, 8, overflow}, T ∈ {2,4,8}, H ∈ {12,96}, several seeds.

TRT is skipped here (current Triton rejects its `cache_results=` autotune
kwarg); b10's for-loop kernel is the shipping same-contract peer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from kda import kda_verify_register as R
from kda.kda_chunk_verify_triton import kda_chunk_verify
from kda.b10.b10_kda_replay_ssm_cutedsl import kda_replay_ssm

ATOL_O, RTOL_O = 2e-2, 1.5e-2
ATOL_RING, RTOL_RING = 4e-3, 1.5e-2
ATOL_S, RTOL_S = 4e-3, 1.5e-2


def _max_err(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), (d / (b.float().abs() + 1e-6)).max().item()


def _run_chunk(t):
    bufs = {n: t[n].clone() for n in
            ("S", "old_u", "old_k", "old_G", "pnat", "buf_idx")}
    o = kda_chunk_verify(
        t["q"], t["k"], t["v"], t["g_raw"], t["beta_raw"],
        t["A_log"], t["dt_bias"], R.VERIFY_LOWER_BOUND,
        bufs["S"], bufs["old_u"], bufs["old_k"], bufs["old_G"],
        bufs["pnat"], bufs["buf_idx"],
    )
    return o, bufs


def _run_b10(t):
    run, bufs = R.replay_ssm_closure(kda_replay_ssm, t)
    o = run()
    return o, bufs


def _ring_write_half(pnat, T, hist=R.VERIFY_HIST):
    overflow = pnat + T > hist
    return (1 if overflow else 0), (0 if overflow else pnat), overflow


def _seq_oracle_window(t):
    """Sequential recurrence from s_logical; returns o, u, kn, G_store, S_final.
    G_store is checkpoint-relative (append: +g_start; overflow/fold: cumsum only).
    """
    B, T = t["q"].shape[:2]
    H, K, V = R.H, R.K, R.V
    pnat = int(t["pnat"][0].item())
    overflow = pnat + T > R.VERIFY_HIST
    g = R._safe_gate(t["g_raw"], t["A_log"], t["dt_bias"])
    beta = torch.sigmoid(t["beta_raw"].float())
    s = t["s_logical"].clone()
    o = torch.empty(B, T, H, V, device="cuda")
    u_out = torch.empty(B, T, H, V, device="cuda")
    k_out = torch.empty(B, T, H, K, device="cuda")
    G_out = torch.empty(B, T, H, K, device="cuda")
    g_start = (t["old_G"][:, 0, :, :, pnat - 1] if pnat > 0
               else torch.zeros(B, H, K, device="cuda"))
    Gcum = torch.zeros_like(g_start) if overflow else g_start.clone()
    for tt in range(T):
        kn = torch.nn.functional.normalize(t["k"][:, tt].float(), dim=-1)
        qn = torch.nn.functional.normalize(t["q"][:, tt].float(), dim=-1)
        Gcum = Gcum + g[:, tt]
        s = s * torch.exp(g[:, tt])[..., None]
        u = beta[:, tt][..., None] * (
            t["v"][:, tt].float() - torch.einsum("bhk,bhkv->bhv", kn, s))
        s = s + kn[..., None] * u[:, :, None, :]
        o[:, tt] = torch.einsum("bhk,bhkv->bhv", qn, s) * (K ** -0.5)
        u_out[:, tt] = u
        k_out[:, tt] = kn
        G_out[:, tt] = Gcum
    return o, u_out, k_out, G_out, s


def check_case(H, B, T, pnat, seed):
    R.set_heads(H)
    t = R.verify_tensors(B, T, pnat, seed)
    o_ref, u_ref, k_ref, G_ref, _ = _seq_oracle_window(t)
    # also the register's packaged oracle (same math)
    o_pkg, _ = R.verify_reference(t)

    o_c, buf_c = _run_chunk(t)
    o_b, buf_b = _run_b10(t)

    half, woff, overflow = _ring_write_half(pnat, T)
    rows = []

    def add(name, a, b, atol, rtol):
        mae, mre = _max_err(a, b)
        good = mae <= atol or mre <= rtol
        rows.append((name, good, mae, mre))
        return good

    ok = True
    ok &= add("o vs seq-oracle", o_c, o_ref, ATOL_O, RTOL_O)
    ok &= add("o vs pkg-oracle", o_c, o_pkg, ATOL_O, RTOL_O)
    ok &= add("o vs b10", o_c, o_b, ATOL_O, RTOL_O)

    u_c = buf_c["old_u"][:, half, woff:woff + T]          # [B,T,H,V]
    k_c = buf_c["old_k"][:, half, woff:woff + T]
    G_c = buf_c["old_G"][:, half, :, :, woff:woff + T]    # [B,H,K,T]
    u_b = buf_b["old_u"][:, half, woff:woff + T]
    k_b = buf_b["old_k"][:, half, woff:woff + T]
    G_b = buf_b["old_G"][:, half, :, :, woff:woff + T]

    ok &= add("old_u vs oracle", u_c, u_ref, ATOL_RING, RTOL_RING)
    ok &= add("old_k vs oracle", k_c, k_ref, ATOL_RING, RTOL_RING)
    ok &= add("old_G vs oracle", G_c.permute(0, 3, 1, 2), G_ref,
              ATOL_RING, RTOL_RING)
    ok &= add("old_u vs b10", u_c, u_b, ATOL_RING, RTOL_RING)
    ok &= add("old_k vs b10", k_c, k_b, ATOL_RING, RTOL_RING)
    ok &= add("old_G vs b10", G_c, G_b, ATOL_RING, RTOL_RING)
    ok &= add("S vs b10", buf_c["S"], buf_b["S"], ATOL_S, RTOL_S)

    if overflow:
        # S must equal the fold of the ACTIVE half's pnat history records
        # (S_logical of the history, in [B,H,V,K] layout)
        s_log = t["s_logical"].transpose(-1, -2).contiguous()
        ok &= add("S vs history-fold", buf_c["S"], s_log, ATOL_S, RTOL_S)
        s_changed = (buf_c["S"] - t["S"]).abs().max().item()
        rows.append(("S wrote on overflow", s_changed > 1e-6, s_changed, 0.0))
        ok &= s_changed > 1e-6
    else:
        s_same = (buf_c["S"] - t["S"]).abs().max().item()
        rows.append(("S untouched (no overflow)", s_same < 1e-6, s_same, 0.0))
        ok &= s_same < 1e-6

    tag = f"H={H} B={B} T={T} pnat={pnat} seed={seed} overflow={overflow}"
    print(f"\n=== {tag}  {'PASS' if ok else 'FAIL'} ===")
    for name, good, mae, mre in rows:
        mark = "OK" if good else "XX"
        print(f"  [{mark}] {name:28s}  max_abs={mae:.3e}  max_rel={mre:.3e}")
    return ok


def main():
    cases = []
    for H in (12, 96):
        for T in (2, 4, 8):
            for pnat in (0, 8):
                cases.append((H, 4, T, pnat, 0))
            # force overflow: pnat + T > 16
            ov = max(0, 17 - T)
            cases.append((H, 4, T, ov, 0))
        cases.append((H, 16, 4, 8, 1))
        cases.append((H, 16, 4, 13, 2))

    n_pass = n_fail = 0
    fails = []
    for c in cases:
        try:
            ok = check_case(*c)
        except Exception as e:
            print(f"\n=== H={c[0]} B={c[1]} T={c[2]} pnat={c[3]} EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            ok = False
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            fails.append(c)
        torch.cuda.empty_cache()

    print(f"\n======== SUMMARY: {n_pass} PASS / {n_fail} FAIL / {len(cases)} total ========")
    if fails:
        print("failures:", fails)
        sys.exit(1)
    print("ALL CORRECT")


if __name__ == "__main__":
    main()
