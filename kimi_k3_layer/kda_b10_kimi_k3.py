"""Kimi-K3 KDA decode, b10-optimized: CuTeDSL kernels on the baseline.

``KimiK3KDAB10(KimiK3KDA)`` keeps the projections, AttnRes and state
layout of the TRT-LLM baseline (kda_trtllm_kimi_k3.py) and swaps the
conv + KDA recurrence + gated RMSNorm for the b10 CuTeDSL kernels
(linear_attn/kda/b10, sm_100a). Two backends = the conv-fusion
ablation:

b10_fused    ONE CuTeDSL kernel: causal conv1d + SiLU + KDA delta-rule
             recurrence + gated RMSNorm (kda_decode_conv_gated_raw).
             Beats trt_fused because the Triton kernel is launch- and
             occupancy-limited at decode: one CTA per (batch, head)
             with the state resident in registers, one pass over HBM.
b10_overlap  the pre-fusion pipeline with the TRT conv kernel issued
             on an aux stream so it overlaps the f_b_proj GEMM, then
             the gated CuTeDSL decode (kda_decode_gated_raw_strided).
             Kept to show WHY the conv fusion matters: overlap hides
             the conv behind the GEMM but still pays a second QKV
             round-trip through HBM.

State layout: the b10 kernels interpret the [B, H, 128, 128] state as
[K, V]; TRT interprets it as [V, K] - ``comparable_ssm_state``
transposes so the bench compares states in one layout.

Single GPU by design, like the baseline (K3Shard sets local shapes;
the TP collective story lives with the MoE layer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .config import K3Shard
from .kda_trtllm_kimi_k3 import KimiK3KDA

# the b10 CuTeDSL kernels live in the sibling linear_attn package; its
# kda/__init__ is import-clean (NO tensorrt_llm stubbing - that lives
# in frameworks.py, which must never be imported next to the real
# runtime)
_LINEAR = Path(__file__).resolve().parents[1] / "linear_attn"
if str(_LINEAR) not in sys.path:
    sys.path.insert(0, str(_LINEAR))

from kda.b10.b10_kda_decode_conv_gated_cutedsl import (  # noqa: E402
    kda_decode_conv_gated_raw,
    kda_decode_gated_raw_strided,
)

B10_BACKENDS = ("b10_fused", "b10_overlap")


class KimiK3KDAB10(KimiK3KDA):
    """Baseline layer with the kernel core swapped for CuTeDSL."""

    BACKENDS = B10_BACKENDS

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        backend: str = "b10_fused",
        *,
        layer_idx: int = 1,
        device: str = "cuda",
    ) -> None:
        super().__init__(shard, batch, backend,
                         layer_idx=layer_idx, device=device)
        self.register_buffer(
            "b10_norm_weight",
            torch.ones(shard.head_dim, device=device, dtype=torch.float32))
        self.conv_stream = (
            torch.cuda.Stream(device=torch.device(device))
            if backend == "b10_overlap" else None
        )

    def comparable_ssm_state(self) -> torch.Tensor:
        return self.ssm_state.transpose(-1, -2)

    def _kernel_forward(
        self,
        qkv: torch.Tensor,
        output_gate: torch.Tensor,
        f_a: torch.Tensor,
        raw_beta: torch.Tensor,
    ) -> torch.Tensor:
        heads, dim = self.shard.heads_local, self.shard.head_dim
        if self.backend == "b10_fused":
            with torch.profiler.record_function("kda.f_b_proj"):
                raw_gate = self.f_b_proj(f_a)
            raw_q, raw_k, raw_v = (
                item.view(self.batch, heads, dim)
                for item in qkv.split(self.shard.proj_dim, dim=-1)
            )
            with torch.profiler.record_function("kda.b10_conv_gated"):
                output = kda_decode_conv_gated_raw(
                    raw_q, raw_k, raw_v,
                    self.conv_weight, self.conv_state,
                    raw_gate.view(self.batch, heads, dim),
                    raw_beta.view(self.batch, heads),
                    self.A_log, self.dt_bias,
                    self.ssm_state,
                    output_gate.view(self.batch, heads, dim),
                    self.b10_norm_weight,
                )
            return output.unsqueeze(0)

        # b10_overlap: TRT conv on an aux stream behind the f_b GEMM
        current_stream = torch.cuda.current_stream(qkv.device)
        self.conv_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.conv_stream):
            with torch.profiler.record_function("kda.conv_aux_stream"):
                convolved = self._conv(qkv)
        with torch.profiler.record_function("kda.f_b_proj"):
            raw_gate = self.f_b_proj(f_a)
        current_stream.wait_stream(self.conv_stream)

        q, k, v = (
            item.view(self.batch, heads, dim)
            for item in convolved.split(self.shard.proj_dim, dim=-1)
        )
        with torch.profiler.record_function("kda.b10_gated"):
            output = kda_decode_gated_raw_strided(
                q, k, v,
                raw_gate.view(self.batch, heads, dim),
                raw_beta.view(self.batch, heads),
                self.A_log, self.dt_bias,
                self.ssm_state,
                output_gate.view(self.batch, heads, dim),
                self.b10_norm_weight,
            )
        return output.unsqueeze(0)
