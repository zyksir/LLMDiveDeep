"""Unified Kimi-K3 KDA reference and B10 layer implementations.

The stateful layer owns projections, recurrent/conv state, AttnRes state, and
the decode-versus-prefill policy. Stateless CUDA/Triton/CuTeDSL entry points
remain in their kernel packages.

The default strategy is selected from the requested token count:

* ``tokens <= DECODE_MAX_TOKENS``: decode
* ``tokens > DECODE_MAX_TOKENS``: prefill

Explicit strategies and backends are retained for correctness checks and
ablation benchmarks. The prefill implementations are the algorithms formerly
isolated in the standalone prefill benchmark; they have not been revalidated
on a GPU as part of the module consolidation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Literal

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
DECODE_MAX_TOKENS = 128
TRT_BACKENDS = ("trt_fused", "trt_unfused", "trt_prefill")
B10_BACKENDS = ("b10_fused", "b10_overlap", "b10_prefill")
ALL_BACKENDS = (*TRT_BACKENDS, *B10_BACKENDS)
Strategy = Literal["auto", "decode", "prefill"]


def _b10_decode_kernels():
    from .kernels.kda_decode import decode_kernels

    return decode_kernels()


def _b10_prefill_kernel():
    from .kernels.kda_prefill import prefill_kernel

    return prefill_kernel()


def _graft(name: str, relpath: str):
    """Load a checkout module under its installed TRT-LLM package name."""
    path = _CHECKOUT / "tensorrt_llm" / relpath
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, module)
    return module


def trt_kda_modules():
    """Return checkout KDA modules against the installed TRT-LLM runtime."""
    import tensorrt_llm  # noqa: F401

    # Only newer fork checkouts ship this helper; none of the grafted KDA
    # modules below import it in older checkouts, so it is optional.
    if (_CHECKOUT / "tensorrt_llm/_torch/modules/stochastic_rounding.py").is_file():
        _graft(
            "tensorrt_llm._torch.modules.stochastic_rounding",
            "_torch/modules/stochastic_rounding.py",
        )
    conv_decode = _graft(
        "tensorrt_llm._torch.modules.mamba.causal_conv1d_triton",
        "_torch/modules/mamba/causal_conv1d_triton.py",
    )
    recurrent = _graft(
        "tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent",
        "_torch/modules/fla/fused_sigmoid_gating_recurrent.py",
    )
    norm = _graft(
        "tensorrt_llm._torch.modules.mamba.layernorm_gated",
        "_torch/modules/mamba/layernorm_gated.py",
    )
    fused = _graft(
        "tensorrt_llm._torch.modules.mamba.fused_kda_decode",
        "_torch/modules/mamba/fused_kda_decode.py",
    )
    attn_res = _graft(
        "tensorrt_llm._torch.modules.attn_res",
        "_torch/modules/attn_res.py",
    )
    return attn_res, conv_decode, fused, recurrent, norm


def trt_prefill_modules():
    """Return the production prefill conv/chunk pipeline and l2norm helper."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

    # optional in older checkouts (see trt_kda_modules)
    if (_CHECKOUT / "tensorrt_llm/_torch/modules/stochastic_rounding.py").is_file():
        _graft(
            "tensorrt_llm._torch.modules.stochastic_rounding",
            "_torch/modules/stochastic_rounding.py",
        )
    module = None
    for leaf in (
        "utils",
        "op",
        "index",
        "l2norm",
        "cumsum",
        "solve_tril",
        "chunk_delta_h",
        "chunk_o",
        "chunk_scaled_dot_kkt",
        "wy_fast",
        "chunk_kda",
    ):
        module = _graft(
            f"tensorrt_llm._torch.modules.fla.{leaf}",
            f"_torch/modules/fla/{leaf}.py",
        )
    assert module is not None
    l2norm = sys.modules["tensorrt_llm._torch.modules.fla.l2norm"]
    return causal_conv1d_fn, module.chunk_kda_with_fused_gate, l2norm.l2norm_fwd


def _backend_strategy(backend: str) -> Literal["decode", "prefill"]:
    return "prefill" if backend.endswith("_prefill") else "decode"


class KimiK3KDA(nn.Module):
    """TRT/reference KDA layer with automatic decode/prefill dispatch."""

    BACKENDS = TRT_BACKENDS
    DEFAULT_DECODE_BACKEND = "trt_fused"
    DEFAULT_PREFILL_BACKEND = "trt_prefill"

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        backend: str | None = None,
        *,
        layer_idx: int = 1,
        device: str = "cuda",
        strategy: Strategy = "auto",
    ) -> None:
        super().__init__()
        if batch < 1:
            raise ValueError("requested token/batch size must be positive")
        if backend is not None and backend not in self.BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {self.BACKENDS}"
            )
        if strategy not in ("auto", "decode", "prefill"):
            raise ValueError("strategy must be 'auto', 'decode', or 'prefill'")
        if backend is not None:
            backend_strategy = _backend_strategy(backend)
            if strategy != "auto" and strategy != backend_strategy:
                raise ValueError(
                    f"backend {backend!r} is incompatible with strategy {strategy!r}"
                )
            strategy = backend_strategy
        if not is_kda_layer(layer_idx):
            raise ValueError(f"layer {layer_idx} is not a KDA layer in Kimi-K3")

        self.requested_tokens = batch
        self.strategy = (
            strategy
            if strategy != "auto"
            else ("decode" if batch <= DECODE_MAX_TOKENS else "prefill")
        )
        self.backend = backend or (
            self.DEFAULT_DECODE_BACKEND
            if self.strategy == "decode"
            else self.DEFAULT_PREFILL_BACKEND
        )
        # Decode treats requested tokens as independent batch elements. The
        # preserved prefill implementation is one packed B=1 sequence.
        self.batch = batch if self.strategy == "decode" else 1

        (
            self._attn_res_mod,
            self._conv_mod,
            self._fused_mod,
            self._recurrent_mod,
            self._norm_mod,
        ) = trt_kda_modules()
        self._prefill_modules = None

        self.shard = shard
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
            shard.hidden,
            local_output + self.in_proj_padding,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.f_b_proj = nn.Linear(
            dim,
            projection,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(
            projection,
            shard.hidden,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(
                shard.qkv_dim,
                shard.conv_size,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.A_log = nn.Parameter(
            torch.zeros(heads, device=device, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(projection, device=device, dtype=torch.float32)
        )
        self.o_norm = self._norm_mod.RMSNorm(
            dim,
            eps=RMS_EPS,
            norm_before_gate=True,
            device=device,
            dtype=torch.bfloat16,
            gate_activation="sigmoid",
        )
        self.attn_res = self._attn_res_mod.AttnRes(
            shard.hidden,
            RMS_EPS,
            dtype=torch.bfloat16,
            device=torch.device(device),
        )
        self.input_norm_weight = nn.Parameter(
            torch.ones(shard.hidden, device=device, dtype=torch.bfloat16)
        )

        self.register_buffer(
            "conv_state",
            torch.zeros(
                self.batch,
                shard.qkv_dim,
                shard.conv_size - 1,
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "ssm_state",
            torch.zeros(
                self.batch,
                heads,
                dim,
                dim,
                device=device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "state_indices",
            torch.arange(self.batch, device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "prefix_sum",
            torch.randn(
                self.batch,
                shard.hidden,
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "delta",
            torch.randn(
                self.batch,
                shard.hidden,
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "block_residual",
            torch.randn(
                self.batch,
                NUM_ATTN_RES_BLOCKS,
                shard.hidden,
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "prefill_has_initial_state",
            torch.zeros(1, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "prefill_reset_indices",
            torch.zeros(1, device=device, dtype=torch.long),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.in_proj_qkvgfab, self.f_b_proj, self.o_proj):
            nn.init.normal_(linear.weight, std=0.02)
        nn.init.normal_(self.conv_weight, std=0.1)
        nn.init.ones_(self.o_norm.weight)
        nn.init.ones_(self.attn_res.norm_weight)
        nn.init.normal_(self.attn_res.proj_weight, std=0.02)

    def comparable_ssm_state(self) -> torch.Tensor:
        """Return recurrence state in the reference [V, K] layout."""
        return self.ssm_state

    def _project(self, hidden: torch.Tensor):
        with torch.profiler.record_function("kda.in_proj_qkvgfab"):
            projected = self.in_proj_qkvgfab(hidden)
        if self.in_proj_padding:
            projected = projected[..., : -self.in_proj_padding]
        return projected.split(self.in_proj_split_sizes, dim=-1)

    def _conv(self, qkv: torch.Tensor) -> torch.Tensor:
        return self._conv_mod.causal_conv1d_update(
            qkv,
            self.conv_state,
            self.conv_weight,
            activation="silu",
            conv_state_indices=self.state_indices,
        )

    def _attn_res_hidden(self) -> torch.Tensor:
        with torch.profiler.record_function("kda.attn_res"):
            return self.attn_res(
                self.prefix_sum,
                self.block_residual,
                delta=self.delta,
                num_blocks=self.prev_valid_blocks,
                block_write_idx=(
                    self.block_write_idx if self.is_block_write_layer else -1
                ),
                output_norm_weight=self.input_norm_weight,
                output_norm_eps=RMS_EPS,
            )

    def _decode_kernel(
        self,
        qkv: torch.Tensor,
        output_gate: torch.Tensor,
        f_a: torch.Tensor,
        raw_beta: torch.Tensor,
    ) -> torch.Tensor:
        heads, dim = self.shard.heads_local, self.shard.head_dim
        if self.backend == "trt_fused":
            with torch.profiler.record_function("kda.f_b_proj"):
                raw_gate = self.f_b_proj(f_a).view(1, self.batch, heads, dim)
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
        with torch.profiler.record_function("kda.conv"):
            qkv = self._conv(qkv)
        with torch.profiler.record_function("kda.f_b_proj"):
            raw_gate = self.f_b_proj(f_a)
        with torch.profiler.record_function("kda.trt_recurrent"):
            output = self._recurrent_mod.fused_kda_packed_decode(
                qkv,
                raw_gate,
                raw_beta,
                self.A_log,
                self.dt_bias,
                self.ssm_state,
                self.state_indices,
                lower_bound=GATE_LOWER_BOUND,
            )
        with torch.profiler.record_function("kda.gated_rmsnorm"):
            return self.o_norm(
                output, output_gate.view(1, self.batch, heads, dim)
            )

    def decode(self) -> torch.Tensor:
        hidden = self._attn_res_hidden()
        qkv, output_gate, f_a, raw_beta = self._project(hidden)
        output = self._decode_kernel(qkv, output_gate, f_a, raw_beta)
        with torch.profiler.record_function("kda.o_proj"):
            return self.o_proj(output.reshape(self.batch, self.shard.proj_dim))

    def _prefill_front(
        self, hidden: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._prefill_modules is None:
            self._prefill_modules = trt_prefill_modules()
        causal_conv1d_fn, _, _ = self._prefill_modules
        qkv, output_gate, f_a, raw_beta = self._project(hidden)
        convolved = causal_conv1d_fn(
            qkv.transpose(0, 1).contiguous(),
            self.conv_weight,
            activation="silu",
            conv_states=self.conv_state,
            has_initial_state=self.prefill_has_initial_state,
            cache_indices=self.state_indices,
            query_start_loc=cu_seqlens,
        ).transpose(0, 1)
        return convolved, output_gate, self.f_b_proj(f_a), raw_beta

    def _prefill_tail(
        self, output: torch.Tensor, output_gate: torch.Tensor
    ) -> torch.Tensor:
        tokens = output.shape[0]
        normed = self.o_norm(
            output.view(
                1, tokens, self.shard.heads_local, self.shard.head_dim
            ),
            output_gate.view(
                1, tokens, self.shard.heads_local, self.shard.head_dim
            ),
        )
        return self.o_proj(normed.reshape(tokens, self.shard.proj_dim))

    def _validate_prefill(
        self, hidden: torch.Tensor | None, cu_seqlens: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            raise ValueError("prefill dispatch requires hidden [S, hidden]")
        if hidden.shape != (self.requested_tokens, self.shard.hidden):
            raise ValueError(
                "hidden must have shape "
                f"[{self.requested_tokens}, {self.shard.hidden}]"
            )
        if hidden.dtype != torch.bfloat16 or not hidden.is_cuda:
            raise ValueError("hidden must be a CUDA bfloat16 tensor")
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(
                [0, self.requested_tokens],
                device=hidden.device,
                dtype=torch.int32,
            )
        if (
            cu_seqlens.shape != (2,)
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != hidden.device
        ):
            raise ValueError("cu_seqlens must be CUDA int32/int64 [0, S]")
        return hidden, cu_seqlens

    def prefill(
        self, hidden: torch.Tensor | None, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        hidden, cu_seqlens = self._validate_prefill(hidden, cu_seqlens)
        if self._prefill_modules is None:
            self._prefill_modules = trt_prefill_modules()
        _, chunk_kda_with_fused_gate, _ = self._prefill_modules

        # Preserve the production reset and fresh-cu-seqlens host-sync behavior
        # that the old prefill benchmark measured.
        reset_indices = self.state_indices[~self.prefill_has_initial_state]
        self.ssm_state[reset_indices] = 0.0
        self.conv_state[reset_indices] = 0.0
        cu_seqlens = cu_seqlens.clone()
        convolved, output_gate, raw_gate, raw_beta = self._prefill_front(
            hidden, cu_seqlens
        )
        tokens = hidden.shape[0]
        heads, dim = self.shard.heads_local, self.shard.head_dim
        q, k, v = (
            item.view(1, tokens, heads, dim)
            for item in convolved.split(self.shard.proj_dim, dim=-1)
        )
        output, _ = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, tokens, heads, dim),
            raw_beta=raw_beta.view(1, tokens, heads).float(),
            A_log=self.A_log,
            g_bias=self.dt_bias,
            scale=dim**-0.5,
            initial_state=self.ssm_state,
            initial_state_indices=self.state_indices,
            inplace_indexed_state_update=True,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        return self._prefill_tail(output.view(tokens, heads, dim), output_gate)

    def forward(
        self,
        hidden: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.strategy == "decode":
            if hidden is not None or cu_seqlens is not None:
                raise ValueError("decode dispatch does not accept prefill inputs")
            return self.decode()
        return self.prefill(hidden, cu_seqlens)


class KimiK3KDAB10(KimiK3KDA):
    """B10 KDA layer sharing the reference layer's state and dispatch API."""

    BACKENDS = B10_BACKENDS
    DEFAULT_DECODE_BACKEND = "b10_fused"
    DEFAULT_PREFILL_BACKEND = "b10_prefill"

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        backend: str | None = None,
        *,
        layer_idx: int = 1,
        device: str = "cuda",
        strategy: Strategy = "auto",
    ) -> None:
        super().__init__(
            shard,
            batch,
            backend,
            layer_idx=layer_idx,
            device=device,
            strategy=strategy,
        )
        self.register_buffer(
            "b10_norm_weight",
            torch.ones(shard.head_dim, device=device, dtype=torch.float32),
        )
        self.conv_stream = (
            torch.cuda.Stream(device=torch.device(device))
            if self.backend == "b10_overlap"
            else None
        )

    def comparable_ssm_state(self) -> torch.Tensor:
        return self.ssm_state.transpose(-1, -2)

    def _attn_res_hidden(self) -> torch.Tensor:
        if os.environ.get("B10_ATTNRES_KERNEL", "0") == "0":
            return super()._attn_res_hidden()
        from .kernels.attn_res_cutedsl import attn_res as attn_res_cute

        with torch.profiler.record_function("kda.attn_res_cute"):
            return attn_res_cute(
                self.prefix_sum,
                self.delta,
                self.block_residual,
                self.attn_res.norm_weight,
                self.attn_res.proj_weight,
                self.input_norm_weight,
                self.prev_valid_blocks,
                self.block_write_idx if self.is_block_write_layer else -1,
                eps=self.attn_res.variance_epsilon,
                out_eps=RMS_EPS,
            )

    def _decode_kernel(
        self,
        qkv: torch.Tensor,
        output_gate: torch.Tensor,
        f_a: torch.Tensor,
        raw_beta: torch.Tensor,
    ) -> torch.Tensor:
        kda_decode_conv_gated_raw, kda_decode_gated_raw_strided = (
            _b10_decode_kernels()
        )
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
                    raw_q,
                    raw_k,
                    raw_v,
                    self.conv_weight,
                    self.conv_state,
                    raw_gate.view(self.batch, heads, dim),
                    raw_beta.view(self.batch, heads),
                    self.A_log,
                    self.dt_bias,
                    self.ssm_state,
                    output_gate.view(self.batch, heads, dim),
                    self.b10_norm_weight,
                )
            return output.unsqueeze(0)

        if self.conv_stream is None:
            raise RuntimeError("b10_overlap requires its auxiliary CUDA stream")
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
                q,
                k,
                v,
                raw_gate.view(self.batch, heads, dim),
                raw_beta.view(self.batch, heads),
                self.A_log,
                self.dt_bias,
                self.ssm_state,
                output_gate.view(self.batch, heads, dim),
                self.b10_norm_weight,
            )
        return output.unsqueeze(0)

    def _safe_gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        dim = self.shard.head_dim
        gate = (raw_gate.float() + self.dt_bias) * torch.exp(self.A_log.float())[
            None, :
        ].repeat_interleave(dim, 1)
        return GATE_LOWER_BOUND * torch.sigmoid(gate)

    def prefill(
        self, hidden: torch.Tensor | None, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        hidden, cu_seqlens = self._validate_prefill(hidden, cu_seqlens)
        if self._prefill_modules is None:
            self._prefill_modules = trt_prefill_modules()
        _, _, l2norm_fwd = self._prefill_modules

        # Sync-free checkout-HEAD reset retained from the old optimized bench.
        self.ssm_state.index_fill_(0, self.prefill_reset_indices, 0.0)
        self.conv_state.index_fill_(0, self.prefill_reset_indices, 0.0)
        convolved, output_gate, raw_gate, raw_beta = self._prefill_front(
            hidden, cu_seqlens
        )
        tokens = hidden.shape[0]
        heads, dim = self.shard.heads_local, self.shard.head_dim
        q, k, v = (
            item.view(1, tokens, heads, dim)
            for item in convolved.split(self.shard.proj_dim, dim=-1)
        )
        q = l2norm_fwd(q.contiguous()).mul_(dim**-0.5)
        k = l2norm_fwd(k.contiguous())
        gate = self._safe_gate(raw_gate).view(1, tokens, heads, dim)
        beta = torch.sigmoid(raw_beta.float()).view(1, tokens, heads).contiguous()
        output, _ = _b10_prefill_kernel()(
            q, k, v, gate, beta, self.ssm_state.zero_()
        )
        return self._prefill_tail(output.view(tokens, heads, dim), output_gate)


__all__ = [
    "ALL_BACKENDS",
    "B10_BACKENDS",
    "DECODE_MAX_TOKENS",
    "KimiK3KDA",
    "KimiK3KDAB10",
    "TRT_BACKENDS",
    "_graft",
]
