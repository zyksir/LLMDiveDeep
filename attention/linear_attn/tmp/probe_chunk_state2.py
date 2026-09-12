"""Slim state-kernel candidates for the two-kernel chunk verify: separate
q/k dots (no [2TP,BV] reshape/trans/split), deferred tiny loads. Perf-only
(reads real buffers, writes real-shaped outputs; math matches the module)."""
import sys

import torch
import triton
import triton.language as tl

sys.path.append('.')
sys.path.append('..')

TP = tl.constexpr(16)


@triton.jit
def _state2(
    v_ptr, S_ptr, ou_ptr, pnat_ptr, bidx_ptr,
    stack_ptr, rc_ptr, mc_ptr, o_ptr,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr, Vd: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    vt = tl.program_id(1)
    offs_k = tl.arange(0, K)
    offs_v = vt * BV + tl.arange(0, BV)
    offs_r = tl.arange(0, HIST)
    offs_t = tl.arange(0, TP)
    tmask = offs_t < T

    pnat = tl.load(pnat_ptr + b)
    bi = tl.load(bidx_ptr + b)

    st_base = stack_ptr + (b * Hn + h) * (2 * TP * K)
    stq = tl.load(st_base + offs_t[:, None] * K + offs_k[None, :]).to(tl.float32)
    stk = tl.load(st_base + (TP + offs_t[:, None]) * K + offs_k[None, :]).to(tl.float32)
    S_base = S_ptr + (b * Hn + h) * (Vd * K)
    s0 = tl.load(S_base + offs_v[:, None] * K + offs_k[None, :])
    s0t = tl.trans(s0)
    oS = tl.dot(stq, s0t)
    w = tl.dot(stk, s0t)

    rc_base = rc_ptr + (b * Hn + h) * (2 * TP * HIST)
    rcq = tl.load(rc_base + offs_t[:, None] * HIST + offs_r[None, :])
    rck = tl.load(rc_base + (TP + offs_t[:, None]) * HIST + offs_r[None, :])
    ou_base = ou_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * Vd) + h * Vd
    ru = tl.load(ou_base + offs_v[None, :],
                 mask=(offs_r < pnat)[:, None], other=0.0).to(tl.float32)
    oS += tl.dot(rcq, ru)
    w += tl.dot(rck, ru)

    tv_base = (b * T + offs_t[:, None]) * (Hn * Vd) + h * Vd + offs_v[None, :]
    v = tl.load(v_ptr + tv_base, mask=tmask[:, None], other=0.0).to(tl.float32)
    d = v - w
    mc_base = mc_ptr + (b * Hn + h) * (2 * TP * TP)
    Mt = tl.load(mc_base + offs_t[:, None] * TP + offs_t[None, :])
    Cmat = tl.load(mc_base + TP * TP + offs_t[:, None] * TP + offs_t[None, :])
    u = tl.dot(Mt, d)
    o = oS + tl.dot(Cmat, d)
    tl.store(o_ptr + tv_base, o, mask=tmask[:, None])

    overflow = pnat + T > HIST
    half = tl.where(overflow, 1 - bi, bi)
    woff = tl.where(overflow, 0, pnat)
    ou_w = ou_ptr + ((b * 2 + half) * HIST + woff + offs_t[:, None]) * (Hn * Vd) + h * Vd
    tl.store(ou_w + offs_v[None, :], u.to(ou_ptr.dtype.element_ty),
             mask=tmask[:, None])


def main():
    from kda import kda_verify_register as R
    R.set_heads(96)
    B, T, Hn, K, Vd, HIST = 64, 4, 96, 128, 128, 16
    t = R.verify_tensors(B, T, 8, seed=1)
    out = torch.empty(B, T, Hn, Vd, device="cuda")
    stack = torch.randn(B, Hn, 32, K, device="cuda", dtype=torch.bfloat16)
    rc = torch.randn(B, Hn, 32, 16, device="cuda")
    mc = torch.randn(B, Hn, 2, 16, 16, device="cuda")
    for BV, nw in [(32, 4), (64, 4), (64, 8), (128, 4), (128, 8)]:
        def launch():
            return _state2[(B * Hn, Vd // BV)](
                t["v"], t["S"], t["old_u"], t["pnat"], t["buf_idx"],
                stack, rc, mc, out,
                B, Hn=Hn, T=T, HIST=HIST, K=K, Vd=Vd, BV=BV, num_warps=nw)
        h = launch()
        ms = triton.testing.do_bench(launch, warmup=20, rep=100)
        print(f"state2 BV={BV:3d} warps={nw}: {ms*1e3:7.1f} us"
              f"  regs={h.n_regs} spills={h.n_spills}")


if __name__ == "__main__":
    main()
