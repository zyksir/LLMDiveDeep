"""Floor probe for the chunk-verify tiling: how fast can a Triton program
per (b,h,V-tile) stream the fp32 state through one stacked [32,K]x[K,BV]
dot? Variants add back the prep stages to find what kills the full kernel.

  A  s0 load + stacked dot + store          (streaming floor)
  B  A + q/k/g loads, gate+cumsum+norm prep (serial ALU/SFU chain)
  C  B + ring load/correction + Neumann     (the full chain, ~ the kernel)
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.append('.')
sys.path.append('..')


@triton.jit
def _floor_kernel(
    q_ptr, k_ptr, v_ptr, graw_ptr, braw_ptr, alog_ptr, dtb_ptr,
    S_ptr, ou_ptr, ok_ptr, oG_ptr, o_ptr, lower_bound,
    VARIANT: tl.constexpr, Hn: tl.constexpr, T: tl.constexpr,
    HIST: tl.constexpr, K: tl.constexpr, Vd: tl.constexpr, BV: tl.constexpr,
):
    TP: tl.constexpr = 16
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    vt = tl.program_id(1)
    offs_k = tl.arange(0, K)
    offs_v = vt * BV + tl.arange(0, BV)
    offs_t = tl.arange(0, TP)
    offs_r = tl.arange(0, HIST)
    tmask = offs_t < T

    S_base = S_ptr + (b * Hn + h) * (Vd * K)
    s0 = tl.load(S_base + offs_v[:, None] * K + offs_k[None, :])  # [BV,K]

    if VARIANT == 0:
        stack = tl.full((2 * TP, K), 0.01, tl.float32)
    else:
        tk_base = (b * T + offs_t[:, None]) * (Hn * K) + h * K + offs_k[None, :]
        q = tl.load(q_ptr + tk_base, mask=tmask[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_ptr + tk_base, mask=tmask[:, None], other=0.0).to(tl.float32)
        graw = tl.load(graw_ptr + tk_base, mask=tmask[:, None], other=0.0)
        alog = tl.load(alog_ptr + h)
        dtb = tl.load(dtb_ptr + h * K + offs_k)
        g = lower_bound * tl.sigmoid(tl.exp(alog) * (graw + dtb[None, :]))
        g = tl.where(tmask[:, None], g, 0.0)
        Gc = tl.cumsum(g, axis=0)
        kn = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-12)[:, None]
        qn = q * tl.rsqrt(tl.sum(q * q, axis=1) + 1e-12)[:, None] * (K ** -0.5)
        el = tl.exp(Gc)
        stack = tl.join(qn * el, kn * el).reshape(2 * TP, K)

    if VARIANT == 2:
        ok_base = ok_ptr + ((b * 2) * HIST + offs_r[:, None]) * (Hn * K) + h * K
        rk = tl.load(ok_base + offs_k[None, :]).to(tl.float32)
        ou_base = ou_ptr + ((b * 2) * HIST + offs_r[:, None]) * (Hn * Vd) + h * Vd
        ru = tl.load(ou_base + offs_v[None, :]).to(tl.float32)
        oG_base = oG_ptr + ((b * 2) * Hn + h) * (K * HIST)
        rG = tl.load(oG_base + offs_k[:, None] * HIST + offs_r[None, :])
        rk_dec = tl.trans(rk) * tl.exp(-rG)
        rc = tl.dot(stack, rk_dec)          # [2TP, HIST]
        big = tl.dot(stack, tl.trans(s0)) + tl.dot(rc, ru)
    else:
        big = tl.dot(stack, tl.trans(s0))   # [2TP, BV]

    oS, w = tl.split(big.reshape(TP, 2, BV).trans(0, 2, 1))
    if VARIANT == 2:
        beta = tl.sigmoid(tl.load(braw_ptr + (b * T + offs_t) * Hn + h,
                                  mask=tmask, other=0.0))
        tv = (b * T + offs_t[:, None]) * (Hn * Vd) + h * Vd + offs_v[None, :]
        v = tl.load(v_ptr + tv, mask=tmask[:, None], other=0.0).to(tl.float32)
        rhs = beta[:, None] * (v - w)
        N = tl.full((TP, TP), 0.001, tl.float32)
        N = tl.where(offs_t[:, None] > offs_t[None, :], N, 0.0)
        u = rhs
        p = rhs
        for _ in tl.static_range(T - 1):
            p = -tl.dot(N, p)
            u = u + p
        oS += tl.dot(N, u)
    tv_base = (b * T + offs_t[:, None]) * (Hn * Vd) + h * Vd + offs_v[None, :]
    tl.store(o_ptr + tv_base, oS + 0.0 * w, mask=tmask[:, None])


def main():
    from kda import kda_verify_register as R
    R.set_heads(96)
    B, T, Hn, K, Vd, HIST = 64, 4, 96, 128, 128, 16
    t = R.verify_tensors(B, T, 8, seed=1)
    out = torch.empty(B, T, Hn, Vd, device="cuda")
    for variant in (0, 1, 2):
        for BV, nw in [(32, 4), (64, 4), (64, 8), (128, 8)]:
            grid = (B * Hn, Vd // BV)
            def launch():
                return _floor_kernel[grid](
                    t["q"], t["k"], t["v"], t["g_raw"], t["beta_raw"],
                    t["A_log"], t["dt_bias"], t["S"], t["old_u"], t["old_k"],
                    t["old_G"], out, R.VERIFY_LOWER_BOUND,
                    VARIANT=variant, Hn=Hn, T=T, HIST=HIST, K=K, Vd=Vd, BV=BV,
                    num_warps=nw)
            h = launch()
            ms = triton.testing.do_bench(launch, warmup=20, rep=100)
            print(f"variant={variant} BV={BV:3d} warps={nw}: {ms*1e3:7.1f} us"
                  f"   regs={h.n_regs} spills={h.n_spills}")


if __name__ == "__main__":
    main()
