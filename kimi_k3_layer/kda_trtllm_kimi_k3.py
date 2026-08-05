"""Kimi-K3 KDA decode baseline on the REAL TensorRT-LLM kernels.

``KimiK3KDA`` is the pure-decode path of TRT-LLM ``KimiDeltaAttention``
(kda_mixer.py) plus the AttnRes block residual, on a single GPU: the
in_proj GEMM, causal conv1d update, fused sigmoid-gating recurrence,
gated RMSNorm and o_proj, with real TRT kernels for every step. Tensor
parallelism changes local weight/head shapes only (``K3Shard``);
collectives are intentionally omitted - the KDA layer's TP comm story
is the MoE layer's (attn and MoE share the sequence AR).

Backends:

trt_fused    production: ONE Triton kernel for conv + KDA recurrence +
             gated RMSNorm (fused_kda_decode).
trt_unfused  the pre-fusion pipeline (conv kernel, recurrence kernel,
             gated-norm kernel) - what SGLang still runs; kept as the
             fusion ablation.

Runs inside the trt-dev container (installed tensorrt_llm 1.3.0rc23).
The three modules the public wheel does not ship yet (attn_res,
fused_kda_decode, its stochastic_rounding dependency) are grafted from
the local trt-llm checkout INTO the installed package namespace, so
their relative imports resolve against the container's runtime - no
sys.modules stubbing (which is what used to shadow the real
``tensorrt_llm`` and break every later import; see bench_comm.py
history).

The b10-optimized subclass lives in ``kda_b10_kimi_k3.py``; benchmark
both via ``bench_kda_kimi_k3.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

from .config import (
    ATTN_RES_BLOCK_SIZE,
    GATE_LOWER_BOUND,
    NUM_ATTN_RES_BLOCKS,
    RMS_EPS,
    K3Shard,
    is_kda_layer,
)

_CHECKOUT = Path(__file__).resolve().parents[2] / "trt-llm"

TRT_BACKENDS = ("trt_fused", "trt_unfused")


def _graft(name: str, relpath: str):
    """Load a module from the local trt-llm checkout under its
    installed-package name (relative imports resolve against the
    container's tensorrt_llm). REPLACES a wheel-shipped copy that
    ``import tensorrt_llm`` may already have loaded - the wheel's KDA
    modules are stale (pre-gate_activation) - and patches the parent
    package attribute so ``from pkg import mod`` also sees the graft."""
    path = _CHECKOUT / "tensorrt_llm" / relpath
    existing = sys.modules.get(name)
    if existing is not None and getattr(
            existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, mod)
    return mod


def trt_kda_modules():
    """(attn_res, conv, fused_decode, recurrent, gated_norm), ALL from
    the checkout: the wheel's copies predate the KDA work (its
    layernorm_gated has no sigmoid gate_activation, no fused decode).
    Their non-KDA dependencies (_utils, fla.utils, torch.utils) resolve
    against the installed runtime - verified present in 1.3.0rc23."""
    import tensorrt_llm  # noqa: F401  (installed runtime, first)

    _graft("tensorrt_llm._torch.modules.stochastic_rounding",
           "_torch/modules/stochastic_rounding.py")
    conv = _graft("tensorrt_llm._torch.modules.mamba.causal_conv1d_triton",
                  "_torch/modules/mamba/causal_conv1d_triton.py")
    recurrent = _graft(
        "tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent",
        "_torch/modules/fla/fused_sigmoid_gating_recurrent.py")
    norm = _graft("tensorrt_llm._torch.modules.mamba.layernorm_gated",
                  "_torch/modules/mamba/layernorm_gated.py")
    fused = _graft("tensorrt_llm._torch.modules.mamba.fused_kda_decode",
                   "_torch/modules/mamba/fused_kda_decode.py")
    attn_res = _graft("tensorrt_llm._torch.modules.attn_res",
                      "_torch/modules/attn_res.py")
    return attn_res, conv, fused, recurrent, norm


class KimiK3KDA(nn.Module):
    """Decode path of TRT-LLM ``KimiDeltaAttention`` + AttnRes."""

    BACKENDS = TRT_BACKENDS

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        backend: str = "trt_fused",
        *,
        layer_idx: int = 1,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if backend not in self.BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected {self.BACKENDS}")
        if not is_kda_layer(layer_idx):
            raise ValueError(
                f"layer {layer_idx} is not a KDA layer in Kimi-K3")
        (self._attn_res_mod, self._conv_mod, self._fused_mod,
         self._recurrent_mod, self._norm_mod) = trt_kda_modules()

        self.shard = shard
        self.batch = batch
        self.backend = backend
        self.layer_idx = layer_idx
        self.is_block_write_layer = layer_idx % ATTN_RES_BLOCK_SIZE == 0
        self.block_write_idx = layer_idx // ATTN_RES_BLOCK_SIZE
        self.prev_valid_blocks = (
            layer_idx + ATTN_RES_BLOCK_SIZE - 1
        ) // ATTN_RES_BLOCK_SIZE

        heads, dim = shard.heads_local, shard.head_dim
        projection = shard.proj_dim
        self.in_proj_split_sizes = [3 * projection, projection, dim, heads]
        local_output = sum(self.in_proj_split_sizes)
        self.in_proj_padding = -local_output % 16

        self.in_proj_qkvgfab = nn.Linear(
            shard.hidden, local_output + self.in_proj_padding,
            bias=False, device=device, dtype=torch.bfloat16,
        )
        self.f_b_proj = nn.Linear(
            dim, projection, bias=False, device=device,
            dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(
            projection, shard.hidden, bias=False, device=device,
            dtype=torch.bfloat16,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(shard.qkv_dim, shard.conv_size,
                        device=device, dtype=torch.bfloat16)
        )
        self.A_log = nn.Parameter(
            torch.zeros(heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.zeros(projection, device=device, dtype=torch.float32))
        self.o_norm = self._norm_mod.RMSNorm(
            dim, eps=RMS_EPS, norm_before_gate=True,
            device=device, dtype=torch.bfloat16,
            gate_activation="sigmoid",
        )
        self.attn_res = self._attn_res_mod.AttnRes(
            shard.hidden, RMS_EPS, dtype=torch.bfloat16,
            device=torch.device(device),
        )
        self.input_norm_weight = nn.Parameter(
            torch.ones(shard.hidden, device=device, dtype=torch.bfloat16))

        self.register_buffer(
            "conv_state",
            torch.zeros(batch, shard.qkv_dim, shard.conv_size - 1,
                        device=device, dtype=torch.bfloat16))
        self.register_buffer(
            "ssm_state",
            torch.zeros(batch, heads, dim, dim,
                        device=device, dtype=torch.float32))
        self.register_buffer(
            "state_indices",
            torch.arange(batch, device=device, dtype=torch.int32))
        self.register_buffer(
            "prefix_sum",
            torch.randn(batch, shard.hidden,
                        device=device, dtype=torch.bfloat16))
        self.register_buffer(
            "delta",
            torch.randn(batch, shard.hidden,
                        device=device, dtype=torch.bfloat16))
        self.register_buffer(
            "block_residual",
            torch.randn(batch, NUM_ATTN_RES_BLOCKS, shard.hidden,
                        device=device, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.in_proj_qkvgfab, self.f_b_proj, self.o_proj):
            nn.init.normal_(linear.weight, std=0.02)
        nn.init.normal_(self.conv_weight, std=0.1)
        nn.init.ones_(self.o_norm.weight)
        nn.init.ones_(self.attn_res.norm_weight)
        nn.init.normal_(self.attn_res.proj_weight, std=0.02)

    def comparable_ssm_state(self) -> torch.Tensor:
        """TRT stores its state as [V, K]; subclasses that store [K, V]
        override this so states compare in one layout."""
        return self.ssm_state

    # -- shared pieces -------------------------------------------------

    def _project(self, hidden: torch.Tensor):
        with torch.profiler.record_function("kda.in_proj_qkvgfab"):
            projected = self.in_proj_qkvgfab(hidden)
        if self.in_proj_padding:
            projected = projected[..., : -self.in_proj_padding]
        return projected.split(self.in_proj_split_sizes, dim=-1)

    def _conv(self, qkv: torch.Tensor) -> torch.Tensor:
        return self._conv_mod.causal_conv1d_update(
            qkv, self.conv_state, self.conv_weight,
            activation="silu", conv_state_indices=self.state_indices,
        )

    # -- backend kernels ------------------------------------------------

    def _kernel_forward(
        self,
        qkv: torch.Tensor,
        output_gate: torch.Tensor,
        f_a: torch.Tensor,
        raw_beta: torch.Tensor,
    ) -> torch.Tensor:
        heads, dim = self.shard.heads_local, self.shard.head_dim
        if self.backend == "trt_fused":
            with torch.profiler.record_function("kda.f_b_proj"):
                raw_gate = self.f_b_proj(f_a).view(
                    1, self.batch, heads, dim)
            with torch.profiler.record_function("kda.trt_fused_decode"):
                return self._fused_mod.fused_kda_decode(
                    projected_qkv=qkv,
                    conv_weight=self.conv_weight,
                    conv_state=self.conv_state,
                    raw_gate=raw_gate,
                    raw_beta=raw_beta.view(1, self.batch, heads),
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    state_indices=self.state_indices,
                    state=self.ssm_state,
                    output_gate=output_gate,
                    norm_weight=self.o_norm.weight,
                    lower_bound=GATE_LOWER_BOUND,
                    norm_eps=RMS_EPS,
                )
        # trt_unfused: conv kernel + recurrence kernel + gated-norm kernel
        with torch.profiler.record_function("kda.conv"):
            qkv = self._conv(qkv)
        with torch.profiler.record_function("kda.f_b_proj"):
            raw_gate = self.f_b_proj(f_a)
        with torch.profiler.record_function("kda.trt_recurrent"):
            output = self._recurrent_mod.fused_kda_packed_decode(
                qkv, raw_gate, raw_beta,
                self.A_log, self.dt_bias,
                self.ssm_state, self.state_indices,
                lower_bound=GATE_LOWER_BOUND,
            )
        with torch.profiler.record_function("kda.gated_rmsnorm"):
            return self.o_norm(
                output, output_gate.view(1, self.batch, heads, dim))

    def forward(self) -> torch.Tensor:
        with torch.profiler.record_function("kda.attn_res"):
            hidden = self.attn_res(
                self.prefix_sum,
                self.block_residual,
                delta=self.delta,
                num_blocks=self.prev_valid_blocks,
                block_write_idx=(
                    self.block_write_idx
                    if self.is_block_write_layer else -1
                ),
                output_norm_weight=self.input_norm_weight,
                output_norm_eps=RMS_EPS,
            )
        qkv, output_gate, f_a, raw_beta = self._project(hidden)
        output = self._kernel_forward(qkv, output_gate, f_a, raw_beta)
        with torch.profiler.record_function("kda.o_proj"):
            return self.o_proj(
                output.reshape(self.batch, self.shard.proj_dim))
