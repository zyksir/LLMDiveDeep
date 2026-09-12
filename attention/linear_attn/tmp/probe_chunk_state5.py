"""State kernel with a pipelined K-loop for the single big dot: per
iteration load stack_kb [32,BK] bf16 + s0_kb [BV,BK] fp32->bf16, one dot.
Everything else identical to probe_chunk_state4."""
import sys

import torch
import triton
import triton.language as tl

sys.path.append('.')
sys.path.append('..')

TP = tl.constexpr(16)


@triton.jit
def _state5(
    v_ptr, S_ptr, ou_ptr, pnat_ptr, bidx_ptr,
    stack_ptr, rc_ptr, mc_ptr, o_ptr,
    B, Hn: tl.constexpr, T: tl.constexpr, HIST: tl.constexpr,
    K: tl.constexpr, Vd: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // Hn, pid % Hn
    vt = tl.program_id(1)
    offs_c = tl.arange(0, BK)
    offs_v = vt * BV + tl.arange(0, BV)
    offs_r = tl.arange(0, HIST)
    offs_t = tl.arange(0, TP)
    offs_2t = tl.arange(0, 2 * TP)
    tmask = offs_t < T

    pnat = tl.load(pnat_ptr + b)
    bi = tl.load(bidx_ptr + b)

    st_base = stack_ptr + (b * Hn + h) * (2 * TP * K)
    S_base = S_ptr + (b * Hn + h) * (Vd * K)
    big = tl.zeros((2 * TP, BV), tl.float32)
    for kb in tl.range(0, K, BK):
        cols = kb + offs_c
        stack = tl.load(st_base + offs_2t[:, None] * K + cols[None, :])
        s0 = tl.load(S_base + offs_v[:, None] * K + cols[None, :])
        big += tl.dot(stack, tl.trans(s0.to(tl.bfloat16)))

    rc_base = rc_ptr + (b * Hn + h) * (2 * TP * HIST)
    rc = tl.load(rc_base + offs_2t[:, None] * HIST + offs_r[None, :])
    ou_base = ou_ptr + ((b * 2 + bi) * HIST + offs_r[:, None]) * (Hn * Vd) + h * Vd
    ru = tl.load(ou_base + offs_v[None, :],
                 mask=(offs_r < pnat)[:, None], other=0.0)
    big += tl.dot(rc, ru)
    oS, w = tl.split(big.reshape(2, TP, BV).trans(1, 2, 0))

    tv_base = (b * T + offs_t[:, None]) * (Hn * Vd) + h * Vd + offs_v[None, :]
    v = tl.load(v_ptr + tv_base, mask=tmask[:, None], other=0.0).to(tl.float32)
    d = (v - w).to(tl.bfloat16)
    mc_base = mc_ptr + (b * Hn + h) * (2 * TP * TP)
    mc = tl.load(mc_base + offs_2t[:, None] * TP + offs_t[None, :])
    ud = tl.dot(mc, d)
    u, cd = tl.split(ud.reshape(2, TP, BV).trans(1, 2, 0))
    o = oS + cd
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
    rc = torch.randn(B, Hn, 32, 16, device="cuda", dtype=torch.bfloat16)
    mc = torch.randn(B, Hn, 32, 16, device="cuda", dtype=torch.bfloat16)
    for BV, BK, nw, ns in [(128, 32, 4, 2), (128, 32, 4, 3), (128, 32, 4, 4),
                           (128, 64, 4, 2), (128, 64, 4, 3),
                           (128, 32, 8, 3), (64, 32, 4, 3), (64, 64, 4, 2)]:
        def launch():
            return _state5[(B * Hn, Vd // BV)](
                t["v"], t["S"], t["old_u"], t["pnat"], t["buf_idx"],
                stack, rc, mc, out,
                B, Hn=Hn, T=T, HIST=HIST, K=K, Vd=Vd, BV=BV, BK=BK,
                num_warps=nw, num_stages=ns)
        h = launch()
        ms = triton.testing.do_bench(launch, warmup=20, rep=100)
        print(f"state5 BV={BV:3d} BK={BK} warps={nw} stages={ns}: "
              f"{ms*1e3:7.1f} us  regs={h.n_regs} spills={h.n_spills}")


if __name__ == "__main__":
    main()
