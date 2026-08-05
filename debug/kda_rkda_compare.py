#!/usr/bin/env python3
"""flashinfer recurrent_kda (PR #4262, CAKE SM100a) vs the b10 chunk
prefill champion + layer glue, at the K3 TP8 shard (H=12).

The flashinfer kernel fuses EVERYTHING the layer needs in-kernel
(q/k l2norm, safe-gate activation via use_gate_in_kernel+lower_bound,
beta sigmoid) and claims 2.05x geomean over MoonshotAI FlashKDA.
Contract quirks found so far: prefill requires cu_seqlens (T>1
without it is rejected), and the state pool is BF16 (production runs
fp16+stochastic-rounding / fp8 snapshots, so bf16 is at-or-above
production precision - acceptable per 2026-08-05 discussion).

If this wins at 4k/8k/16k it replaces kda_chunk_prefill + ALL the
glue in bench_kda_prefill_layer.py's opt path.

  CUDA_VISIBLE_DEVICES=<free gpu> python3 debug/kda_rkda_compare.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_LLMDIR), str(_LLMDIR / "linear_attn")]

from kda.b10.b10_kda_chunk_prefill_cutedsl import kda_chunk_prefill  # noqa: E402
from flashinfer.kda_decode import recurrent_kda  # noqa: E402

H, D, LB = 12, 128, -5.0


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda").manual_seed(3)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=gen, device="cuda",
                            dtype=torch.float32) * sc).to(torch.bfloat16)

    print(f"{'S':>7} {'b10+glue':>10} {'fi_rkda':>10} {'gain':>7} "
          f"{'cosine':>8}")
    for S in (4096, 8192, 16384):
        q, k = rn(1, S, H, D), rn(1, S, H, D)
        v = rn(1, S, H, D, sc=0.5)
        raw_g = rn(1, S, H, D, sc=0.5)
        beta_l = rn(1, S, H, sc=0.5)
        A_log = torch.zeros(H, device="cuda") - 0.5
        dt_bias = torch.zeros(H * D, device="cuda") + 0.1
        s0 = torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32)
        s0b = s0.to(torch.bfloat16)
        cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)

        def b10():
            qn = F.normalize(q.float(), p=2, dim=-1).to(
                torch.bfloat16) * (D ** -0.5)
            kn = F.normalize(k.float(), p=2, dim=-1).to(torch.bfloat16)
            g = LB * torch.sigmoid(
                torch.exp(A_log)[None, None, :, None]
                * (raw_g.float() + dt_bias.view(1, 1, H, D)))
            beta = torch.sigmoid(beta_l.float()).contiguous()
            return kda_chunk_prefill(qn, kn, v, g.contiguous(),
                                     beta, s0)[0]

        beta_sig = torch.sigmoid(beta_l.float()).to(torch.bfloat16)

        def fik():
            # contract: beta PRE-SIGMOIDED bf16; state bf16 [N,HV,V,K]
            return recurrent_kda(
                q, k, v, raw_g, beta_sig, A_log=A_log, dt_bias=dt_bias,
                scale=D ** -0.5, initial_state=s0b.clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                lower_bound=LB, cu_seqlens=cu)[0]

        o1, o2 = b10(), fik()
        torch.cuda.synchronize()
        # GUARD: on a contended GPU the fi kernel was observed to
        # return in ~10 us with NaN output (silent dispatch failure).
        # Refuse to print timings for garbage.
        for name, o in (("b10", o1), ("fi_rkda", o2)):
            if not torch.isfinite(o.float()).all():
                raise RuntimeError(
                    f"{name} produced non-finite output at S={S} - "
                    "kernel dispatch failed or GPU is contended; "
                    "rerun on a clean GPU")
        cos = F.cosine_similarity(o1.float().flatten(),
                                  o2.float().flatten(), dim=0).item()
        if cos < 0.98:
            print(f"  WARNING S={S}: cosine {cos:.4f} - outputs "
                  "disagree, timings below are not comparable")
        t1, t2 = bench(b10), bench(fik)
        print(f"{S:>7} {t1:>10.1f} {t2:>10.1f} "
              f"{(t1 - t2) / t1 * 100:>+6.1f}% {cos:>8.5f}", flush=True)


if __name__ == "__main__":
    main()
