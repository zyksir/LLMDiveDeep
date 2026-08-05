#!/usr/bin/env python3
"""Which side of the KDA prefill layer bench deviates: compare baseline
(TRT chunk pipeline) and opt (b10 kernel + safe-gate glue) against the
fp32 recurrence oracle at S=512, plus glue-variant probes."""
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]
import tensorrt_llm  # noqa: F401,E402

pkg = types.ModuleType("kimi_k3_layer")
pkg.__path__ = [str(_LLMDIR / "kimi_k3_layer")]
sys.modules.setdefault("kimi_k3_layer", pkg)
for p in (str(_LLMDIR), str(_LLMDIR / "linear_attn")):
    sys.path.insert(0, p)

from kimi_k3_layer.kda_trtllm_kimi_k3 import _graft  # noqa: E402

_graft("tensorrt_llm._torch.modules.stochastic_rounding",
       "_torch/modules/stochastic_rounding.py")
for leaf in ("utils", "op", "index", "l2norm", "cumsum", "solve_tril",
             "chunk_delta_h", "chunk_o", "chunk_scaled_dot_kkt",
             "wy_fast", "chunk_kda"):
    mod = _graft(f"tensorrt_llm._torch.modules.fla.{leaf}",
                 f"_torch/modules/fla/{leaf}.py")
chunk_kda_with_fused_gate = mod.chunk_kda_with_fused_gate

from kda.b10.b10_kda_chunk_prefill_cutedsl import kda_chunk_prefill  # noqa: E402
from kda.kda_attention import activate_kda_gate, kda_recurrent_reference  # noqa: E402

torch.cuda.set_device(0)
S, H, D = 512, 12, 128
LB = -5.0
g0 = torch.Generator(device="cuda").manual_seed(3)


def rn(*s, sc=1.0):
    return (torch.randn(*s, generator=g0, device="cuda",
                        dtype=torch.float32) * sc).to(torch.bfloat16)


q = rn(1, S, H, D)
k = rn(1, S, H, D)
v = rn(1, S, H, D, sc=0.5)
raw_g = rn(1, S, H, D, sc=0.5)
beta_l = rn(1, S, H, sc=0.5)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32) - 0.5
dt_bias = torch.zeros(H * D, device="cuda", dtype=torch.float32) + 0.1
state = torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32)
idx = torch.zeros(1, device="cuda", dtype=torch.int32)

# oracle: safe gate, l2norm, scale
g_safe = activate_kda_gate(raw_g.view(1, S, H, D).float(),
                           A_log, dt_bias,
                           lower_bound=LB).view(1, S, H, D)
qn = F.normalize(q.float(), p=2, dim=-1)
kn = F.normalize(k.float(), p=2, dim=-1)
beta = torch.sigmoid(beta_l.float())
o_ref, _ = kda_recurrent_reference(
    q.float(), k.float(), v.float(), g_safe, beta)
o_ref = o_ref * D ** -0.5  # reference normalizes qk itself; scale is linear

# baseline: TRT chunk pipeline (fused safe gate)
state.zero_()
o_trt, _ = chunk_kda_with_fused_gate(
    q=q, k=k, v=v,
    raw_g=raw_g.view(1, S, H, D),
    raw_beta=beta_l.view(1, S, H).float(),
    A_log=A_log, g_bias=dt_bias,
    scale=D ** -0.5,
    initial_state=state,
    initial_state_indices=idx,
    inplace_indexed_state_update=True,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=torch.tensor([0, S], device="cuda", dtype=torch.int32),
)

# opt: b10 kernel + glue
s0 = torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32)
o_b10, _ = kda_chunk_prefill(
    qn.to(torch.bfloat16), kn.to(torch.bfloat16), v,
    g_safe.contiguous(), beta.contiguous(), s0)
o_b10 = o_b10.float() * D ** -0.5


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(),
                               dim=0).item()


# register-identical b10 call: canonical gate, RAW q/k (no l2norm)
g_canon = activate_kda_gate(raw_g.view(1, S, H, D).float(), A_log, dt_bias)
o_b10c, _ = kda_chunk_prefill(
    q, k, v, g_canon.contiguous(),
    beta.contiguous(),
    torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32))
o_refc, _ = kda_recurrent_reference(
    q.float(), k.float(), v.float(), g_canon, beta)
print(f"cos(b10-canon, canon-oracle) = {cos(o_b10c, o_refc):.6f}")
o_refs_nonorm, _ = kda_recurrent_reference(
    qn, kn, v.float(), g_safe, beta, normalize_qk=False)
print(f"cos(b10-safe, safe-oracle-prenorm) = {cos(o_b10, o_refs_nonorm):.6f}")
print(f"cos(trt, oracle) = {cos(o_trt, o_ref):.6f}")
print(f"cos(b10, oracle) = {cos(o_b10, o_ref):.6f}")
print(f"cos(b10, trt)    = {cos(o_b10, o_trt):.6f}")
