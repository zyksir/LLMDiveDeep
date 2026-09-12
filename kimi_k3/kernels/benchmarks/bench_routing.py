#!/usr/bin/env python3
"""Kimi-K3 routing: correctness + perf, ours vs the TRT-LLM baseline.

Single GPU (routing is replicated on every TP rank). Run in trt-dev:

  docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep \
      && python3 kimi_k3/kernels/bench_routing.py"

Compared end to end from the hidden state (GEMM + routing) and as the
routing kernel alone:

  trtllm     dsv3_router_gemm_op (fp32 logits) + noaux_tc_op
  ours       bf16 logits read in place + Triton pack kernel (also emits
             the trtllm-gen packed ids, so the fused MoE skips its own
             routing stage)

Correctness: both must select the same expert SET as the fp32 torch
oracle and reproduce its weights (bf16 tolerance for ours).
Prints one markdown table; saves nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from common.kernel_bench import bench_cuda  # noqa: E402
from kimi_k3.config import HIDDEN, NUM_EXPERTS  # noqa: E402
from kimi_k3.kernels.routing import (  # noqa: E402
    gate_gemm_trtllm,
    route_for_fused_moe,
    route_pack,
    routing_ref,
    routing_trtllm,
    radix_available,
    route_radix_for_trtllm_gen,
    unpack_ids,
)


def check(batch: int, gate_w, gate_bias) -> None:
    g = torch.Generator(device="cuda").manual_seed(100 + batch)
    h = torch.randn(batch, HIDDEN, generator=g, device="cuda",
                    dtype=torch.float32).to(torch.bfloat16)
    logits_bf16 = h @ gate_w.T
    logits_f32 = logits_bf16.float()  # sigmoid input bit-identical
    ref_ids, ref_w = routing_ref(logits_f32, gate_bias.float())

    ids_t, w_t = routing_trtllm(logits_f32, gate_bias.float())
    assert (torch.sort(ids_t)[0] == torch.sort(ref_ids)[0]).all(), \
        f"trtllm expert set mismatch B={batch}"
    order = torch.argsort(ids_t, dim=-1)
    ref_order = torch.argsort(ref_ids, dim=-1)
    err_t = (torch.gather(w_t, 1, order)
             - torch.gather(ref_w, 1, ref_order)).abs().max().item()
    assert err_t < 1e-5, f"trtllm weights err={err_t} B={batch}"

    packed, w_o = route_pack(logits_bf16, gate_bias.float())
    ids_o = unpack_ids(packed)
    assert (torch.sort(ids_o)[0] == torch.sort(ref_ids)[0]).all(), \
        f"ours expert set mismatch B={batch}"
    order_o = torch.argsort(ids_o, dim=-1)
    err_o = (torch.gather(w_o.float(), 1, order_o)
             - torch.gather(ref_w, 1, ref_order)).abs().max().item()
    assert err_o < 4e-3, f"ours weights err={err_o} B={batch}"  # bf16 out

    # the SHIPPED router (Routing.RADIX): expert sets must match the oracle
    if radix_available():
        ids_r, w_r = route_radix_for_trtllm_gen(logits_bf16, gate_bias.float())
        assert (torch.sort(ids_r)[0] == torch.sort(ref_ids)[0]).all(), \
            f"radix expert set mismatch B={batch}"
        err_r = (torch.gather(w_r.float(), 1, torch.argsort(ids_r, dim=-1))
                 - torch.gather(ref_w, 1, ref_order)).abs().max().item()
        assert err_r < 4e-3, f"radix weights err={err_r} B={batch}"  # bf16
    else:
        err_r = float("nan")

    ids_f, scales_f = route_for_fused_moe(logits_bf16, gate_bias.float())
    assert (torch.sort(ids_f)[0] == torch.sort(ref_ids)[0]).all()
    err_f = (torch.gather(scales_f, 1, torch.argsort(ids_f, dim=-1))
             - torch.gather(ref_w, 1, ref_order)).abs().max().item()
    assert err_f < 1e-5, f"ours fp32 scales err={err_f} B={batch}"
    print(f"  B={batch:<5} expert sets match; weight err "
          f"trtllm {err_t:.2e}, ours(bf16) {err_o:.2e}, "
          f"ours(fp32) {err_f:.2e}, radix(bf16) {err_r:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 8192])
    args = parser.parse_args()

    import tensorrt_llm  # noqa: F401 - registers torch.ops.trtllm

    torch.cuda.set_device(0)
    g = torch.Generator(device="cuda").manual_seed(7)
    gate_w = (torch.randn(NUM_EXPERTS, HIDDEN, generator=g, device="cuda",
                          dtype=torch.float32) * 0.02).to(torch.bfloat16)
    gate_bias = torch.randn(NUM_EXPERTS, generator=g, device="cuda",
                            dtype=torch.float32) * 0.5

    print("== correctness (vs fp32 torch oracle) ==")
    for batch in (1, 8, 64, 1024):
        check(batch, gate_w, gate_bias)

    print("\n== latency (us; kernel = routing only, e2e = GEMM+routing) ==")
    rows = []
    for batch in args.tokens:
        gb = torch.Generator(device="cuda").manual_seed(batch)
        h = torch.randn(batch, HIDDEN, generator=gb, device="cuda",
                        dtype=torch.float32).to(torch.bfloat16)
        logits_bf16 = h @ gate_w.T
        logits_f32 = logits_bf16.float()
        bias_f = gate_bias.float()

        t_kernel_trt = bench_cuda(
            lambda: routing_trtllm(logits_f32, bias_f))
        t_kernel_ours = bench_cuda(
            lambda: route_pack(logits_bf16, bias_f))
        t_kernel_radix = (bench_cuda(
            lambda: route_radix_for_trtllm_gen(logits_bf16, bias_f))
            if radix_available() else float("nan"))
        t_e2e_trt = bench_cuda(
            lambda: routing_trtllm(gate_gemm_trtllm(h, gate_w), bias_f))
        t_e2e_ours = bench_cuda(
            lambda: route_pack(h @ gate_w.T, bias_f))
        rows.append((batch, t_kernel_trt, t_kernel_ours, t_kernel_radix,
                     t_e2e_trt, t_e2e_ours))

    print("| tokens | trtllm kernel | triton kernel | radix (shipped) | "
          "trtllm e2e | ours e2e | e2e speedup |")
    print("|---|---|---|---|---|---|---|")
    for batch, kt, ko, kr, et, eo in rows:
        print(f"| {batch} | {kt:.2f} | {ko:.2f} | {kr:.2f} | {et:.2f} "
              f"| {eo:.2f} | {et / eo:.2f}x |")


if __name__ == "__main__":
    main()
