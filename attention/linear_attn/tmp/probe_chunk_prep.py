"""Stage-by-stage cost of the chunk-verify PREP kernel (see
kda_chunk_verify_triton.py). Cumulative variants:

  0  q/k/g loads + gate/cumsum/norm + stack store
  1  + ring loads (rk, rG, gs) + RC dots + rc store
  2  + Amat/Bmat + Neumann operators + mc store
  3  + ring k append + G append            (= the full prep kernel)
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.append('.')
sys.path.append('..')

TP = tl.constexpr(16)


@triton.jit
def _prep_probe(
    q_ptr, k_ptr, graw_ptr, braw_ptr, alog_ptr, dtb_ptr,
    ok_ptr, oG_ptr, pnat_ptr, bidx_ptr,
    stack_ptr, rc_ptr, mc_ptr,
    lower_bound,
    VARIANT: tl.constexpr,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    offs_k = tl.arange(0, K)
    offs_r = tl.arange(0, HIST)
    offs_t = tl.arange(0, TP)
    tmask = offs_t < T
    pnat = tl.load(pnat_ptr + b)
    bi = tl.load(bidx_ptr + b)

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
    Gc = tl.cumsum(g, axis=0)
    el = tl.exp(Gc)
    kn = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-12)[:, None]
    qn = q * tl.rsqrt(tl.sum(q * q, axis=1) + 1e-12)[:, None] * (K ** -0.5)
    kn = tl.where(tmask[:, None], kn, 0.0)
    qn = tl.where(tmask[:, None], qn, 0.0)
    kl = kn * el
    ql = qn * el

    st_base = stack_ptr + (b * Hn + h) * (2 * TP * K) + offs_t[:, None] * K
    if VARIANT == 0:
        tl.store(st_base + offs_k[None, :], ql.to(stack_ptr.dtype.element_ty))
        tl.store(st_base + TP * K + offs_k[None, :],
                 kl.to(stack_ptr.dtype.element_ty))
        return

    rmask = offs_r < pnat
    ok_base = ok_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * K) + h * K
    rk = tl.load(ok_base + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    oG_base = oG_ptr + ((b * 2 + bi) * Hn + h) * (K * HIST)
    rG = tl.load(oG_base + offs_k[:, None] * HIST + offs_r[None, :],
                 mask=rmask[None, :], other=0.0)
    gs = tl.load(oG_base + offs_k * HIST + tl.maximum(pnat - 1, 0))
    gs = tl.where(pnat > 0, gs, 0.0)
    rk_dec = tl.trans(rk) * tl.exp(gs[:, None] - rG)
    rk_dec = tl.where(rmask[None, :], rk_dec, 0.0)
    egs = tl.exp(gs)
    tl.store(st_base + offs_k[None, :],
             (ql * egs[None, :]).to(stack_ptr.dtype.element_ty))
    tl.store(st_base + TP * K + offs_k[None, :],
             (kl * egs[None, :]).to(stack_ptr.dtype.element_ty))
    rc_base = rc_ptr + (b * Hn + h) * (2 * TP * HIST) + offs_t[:, None] * HIST
    tl.store(rc_base + offs_r[None, :], tl.dot(ql, rk_dec))
    tl.store(rc_base + TP * HIST + offs_r[None, :], tl.dot(kl, rk_dec))
    if VARIANT == 1:
        return

    ki = kn / el
    Amat = tl.dot(kl, tl.trans(ki))
    N = tl.where(offs_t[:, None] > offs_t[None, :], Amat * beta[:, None], 0.0)
    eye = tl.where(offs_t[:, None] == offs_t[None, :], 1.0, 0.0)
    Minv = eye
    p = eye
    for _ in tl.static_range(T - 1):
        p = -tl.dot(N, p)
        Minv = Minv + p
    Mt = Minv * beta[None, :]
    Bmat = tl.dot(ql, tl.trans(ki))
    Bmat = tl.where(offs_t[:, None] >= offs_t[None, :], Bmat, 0.0)
    Cmat = tl.dot(Bmat, Mt)
    mc_base = mc_ptr + (b * Hn + h) * (2 * TP * TP) + offs_t[:, None] * TP
    tl.store(mc_base + offs_t[None, :], Mt)
    tl.store(mc_base + TP * TP + offs_t[None, :], Cmat)
    if VARIANT == 2:
        return

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


def main():
    from kda import kda_verify_register as R
    R.set_heads(96)
    B, T, Hn, K, HIST = 64, 4, 96, 128, 16
    t = R.verify_tensors(B, T, 8, seed=1)
    stack = torch.empty(B, Hn, 32, K, device="cuda", dtype=torch.bfloat16)
    rc = torch.empty(B, Hn, 32, 16, device="cuda")
    mc = torch.empty(B, Hn, 2, 16, 16, device="cuda")
    for variant in (0, 1, 2, 3):
        for nw in (1, 2, 4):
            def launch():
                return _prep_probe[(B * Hn,)](
                    t["q"], t["k"], t["g_raw"], t["beta_raw"], t["A_log"],
                    t["dt_bias"], t["old_k"], t["old_G"], t["pnat"],
                    t["buf_idx"], stack, rc, mc, R.VERIFY_LOWER_BOUND,
                    VARIANT=variant, B=B, Hn=Hn, T=T, HIST=HIST, K=K,
                    num_warps=nw)
            h = launch()
            ms = triton.testing.do_bench(launch, warmup=20, rep=100)
            print(f"variant={variant} warps={nw}: {ms*1e3:7.1f} us"
                  f"  regs={h.n_regs} spills={h.n_spills}")


if __name__ == "__main__":
    main()
