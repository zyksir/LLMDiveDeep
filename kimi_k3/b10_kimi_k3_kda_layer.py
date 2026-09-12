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


def native_replay_verify_op():
    """Resolve the rc19-native fused MTP verify kernel (wychunk replay).

    Prefers the installed ``tensorrt_llm`` runtime (the rc19 deployment
    image, ``44d66979``, ships the op). If the installed wheel predates it,
    grafts the module from the checkout — ``12e51f84`` (b10-1.3.0rc19), the
    exact commit of the native TRT baseline trace — so the executed source
    matches the baseline either way.
    """
    import tensorrt_llm  # noqa: F401

    try:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.kda_replay_ssm_conv_gated_wychunk import (  # noqa: E501
            kda_replay_ssm_conv_gated_wychunk,
        )

        return kda_replay_ssm_conv_gated_wychunk
    except ImportError:
        # The module's relative import needs this helper resolvable first.
        _graft(
            "tensorrt_llm._torch.modules.stochastic_rounding",
            "_torch/modules/stochastic_rounding.py",
        )
        module = _graft(
            "tensorrt_llm._torch.cute_dsl_kernels.blackwell."
            "kda_replay_ssm_conv_gated_wychunk",
            "_torch/cute_dsl_kernels/blackwell/"
            "kda_replay_ssm_conv_gated_wychunk.py",
        )
        return module.kda_replay_ssm_conv_gated_wychunk


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


MTP_WIDTH = 2  # MTP1: every request contributes 1 draft + 1 bonus token
_MTP_RING_LEN = 4  # ReplaySSM ring records per slot (>= MTP_WIDTH, x4 rounded)
_MTP_MAX_TOKENS = 8  # TGV small-M window and kda_decode_mtp NUM_SPEC cover


class _KimiK3KdaMtpBlock(KimiK3KDA):
    """Shared MTP1 (target-verify, M = 2 tokens/request) KDA block machinery.

    Trace-aligned per ``kimi_k3/traces/mtp1_decode_layer_kernels.md``. This
    base implements the sglang-style middle section — the fused ``[q|k|v|g]``
    in-proj on the SGLang TGV CuTe DSL GEMM (main stream), the ``[f_a|beta]``
    and ``f_b`` projections on SGLang's tiny GEMV kernels (aux stream), and the
    fused KDA MTP verify CuTe DSL kernel (conv1d update + lower-bound gate +
    delta rule + ReplaySSM ring writes + gated RMSNorm) in CACHE_RING mode —
    used by ``SglKimiK3KdaBlock`` and by ``TrtKimiK3KdaBlock`` in its
    fork-provenance ``sgl_replay`` mode; the TRT block's DEFAULT decode
    overrides the middle with the native-rc19 sequence (see its docstring).
    Subclasses supply the stack-specific AttnRes kernel, o_proj GEMM, and
    all-reduce/residual tail.

    ``batch`` keeps the file's token-count convention: it is the TOTAL token
    count M (must be a multiple of ``MTP_WIDTH``); requests = batch // 2.
    Primary alignment case is batch=2 (1 request). Kernel-name alignment holds
    for the tp8 shard (the traces' TP8 shapes: 144-row ``[f_a|beta]`` and
    1536x128 ``f_b`` tiny GEMMs, 6144x7168 TGV in-proj).

    Shared deviations from production (documented per the bench charter):

    * State pools (conv/recurrent/ring/intermediate-window) are sized to the
      bench batch instead of the serving cache manager's pool, so the CuTe
      kernel-name suffix that embeds the pool shape differs (see the trace
      note); slot indices and cu_seqlens are static ``arange`` buffers.
    * The once-PER-STEP ``kda_replayssm_exact_fold_kernel`` (ring -> live
      state commit, launched by the cache manager across all layers, 12x in
      each trace) is outside the single-layer boundary and is not reproduced;
      the recurrent state is read-only across forwards exactly as in the
      production verify kernel.
    * ``conv_weight``/gated-norm weight are fp32 copies of this file's bench
      bf16 parameters (production loads fp32 straight from the checkpoint).
    * The tiny-GEMV chain always runs on the block's aux stream; sglang gates
      that fork on CUDA-graph capture mode and the TRT fork on token count.
      The kernels and their order are identical either way, and the bench
      times decode under graphs, matching the gated baselines.
    """

    BACKENDS = ("mtp1",)
    DEFAULT_DECODE_BACKEND = "mtp1"
    DEFAULT_PREFILL_BACKEND = "mtp1"
    RECORD_SPAN = "KimiK3KdaMtp"

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        backend: str | None = None,
        *,
        layer_idx: int = 1,
        device: str = "cuda",
        strategy: Strategy = "auto",
        collectives=None,
    ) -> None:
        if batch % MTP_WIDTH:
            raise ValueError(
                f"MTP1 blocks need batch % {MTP_WIDTH} == 0 total tokens"
            )
        if not MTP_WIDTH <= batch <= _MTP_MAX_TOKENS:
            raise ValueError(
                f"MTP1 blocks cover {MTP_WIDTH}..{_MTP_MAX_TOKENS} tokens "
                "(the traces' TGV/tiny-GEMM small-M window)"
            )
        if strategy == "prefill":
            raise ValueError("MTP1 blocks are decode(verify)-only")
        super().__init__(
            shard,
            batch,
            backend,
            layer_idx=layer_idx,
            device=device,
            strategy="decode",
        )
        if self.prev_valid_blocks < 1:
            raise ValueError(
                "MTP1 blocks need layer_idx >= 1 (a non-empty AttnRes bank)"
            )
        self._collectives = collectives
        self.num_requests = batch // MTP_WIDTH
        heads, dim = shard.heads_local, shard.head_dim
        self._n_qkvg = 4 * shard.proj_dim
        requests = self.num_requests
        dev = torch.device(device)

        # Request-granular pools replace the parent's token-granular ones.
        self.ssm_state = torch.zeros(
            requests, heads, dim, dim, device=dev, dtype=torch.float32
        )
        self.state_indices = torch.arange(
            requests, device=dev, dtype=torch.int32
        )
        self.register_buffer(
            "inter_state_indices",
            torch.arange(requests, device=dev, dtype=torch.int32),
        )
        self.register_buffer(
            "cu_seqlens",
            torch.arange(requests + 1, device=dev, dtype=torch.int32)
            * MTP_WIDTH,
        )
        # The verify kernel consumes fp32 conv weights and fp32 norm weight
        # (checkpoint dtypes in production; copies of the bench params here).
        self.register_buffer("conv_weight_f32", self.conv_weight.float())
        self.register_buffer("o_norm_weight_f32", self.o_norm.weight.float())
        # ReplaySSM rings (CACHE_RING mode, both baselines): raw v/k bf16,
        # log-decay gate and beta fp32; ring length rounded to 4 records.
        self.register_buffer(
            "ring_rawv",
            torch.zeros(
                requests, heads, _MTP_RING_LEN, dim,
                device=dev, dtype=torch.bfloat16,
            ),
        )
        self.register_buffer("ring_rawk", torch.zeros_like(self.ring_rawv))
        self.register_buffer(
            "ring_g",
            torch.zeros(
                requests, heads, _MTP_RING_LEN, dim,
                device=dev, dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "ring_beta",
            torch.zeros(
                requests, heads, _MTP_RING_LEN,
                device=dev, dtype=torch.float32,
            ),
        )
        self.aux_stream = torch.cuda.Stream(device=dev)

    def attach_collectives(self, collectives) -> None:
        self._collectives = collectives

    #: Collectives.all_reduce impl for the TP tail; subclasses pin the
    #: backend whose kernel matches their baseline trace.
    AR_IMPL = "auto"

    def _reduce(self, partial: torch.Tensor) -> torch.Tensor:
        """TP tail; see subclass docstrings for the per-stack wiring.

        The bench's single-GPU convention runs tp-shard math on one rank, so
        with no (or world==1) Collectives no AR kernel is launched at all;
        at world > 1 the AR routes through ``AR_IMPL``."""
        if self._collectives is not None and self._collectives.world > 1:
            return self._collectives.all_reduce(partial, impl=self.AR_IMPL)
        return partial

    def _project_mtp(self, hidden: torch.Tensor):
        """Trace positions 2-4: TGV [q|k|v|g] on the main stream, the
        [f_a|beta] + f_b tiny GEMV chain on the aux stream."""
        from .kernels.sgl_adapters.gemm import (
            cutedsl_bf16_gemm,
            kimi_k3_tiny_gemm,
        )

        # Detached: the vendored kernels export via dlpack, which rejects
        # requires_grad tensors (production weights are plain tensors).
        weight = self.in_proj_qkvgfab.weight.detach()
        f_b_weight = self.f_b_proj.weight.detach()
        dim, heads = self.shard.head_dim, self.shard.heads_local
        current = torch.cuda.current_stream(hidden.device)
        self.aux_stream.wait_stream(current)
        with torch.cuda.stream(self.aux_stream):
            with torch.profiler.record_function("kda.bfa_tiny_gemm"):
                # Pad rows stay in the slice: the (144, 7168) shape selects
                # the compiled tiny-N kernel (the trace's kernel #3).
                fab = kimi_k3_tiny_gemm(hidden, weight[self._n_qkvg :])
            with torch.profiler.record_function("kda.f_b_tiny_gemm"):
                raw_gate = kimi_k3_tiny_gemm(fab[:, :dim], f_b_weight)
        with torch.profiler.record_function("kda.in_proj_qkvg_tgv"):
            qkvg = cutedsl_bf16_gemm(hidden, weight[: self._n_qkvg])
        current.wait_stream(self.aux_stream)
        if not torch.cuda.is_current_stream_capturing():
            # Under graph capture the pool owns cross-stream lifetimes and
            # record_stream is disallowed; eager runs still need it.
            fab.record_stream(current)
            raw_gate.record_stream(current)
        beta = fab[:, dim : dim + heads]
        return qkvg, raw_gate, beta

    def _conv_views(self):
        """(cs_q, cs_k, cs_v, ic_q, ic_k, ic_v) in the kernel's [slot(, t),
        channel, window] layout; storage layout is per-stack."""
        raise NotImplementedError

    def _kda_verify(
        self,
        qkvg: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Trace position 5: one fused CuTe DSL KDA MTP verify launch."""
        from .kernels.sgl_adapters.kda import fused_kda_decode_mtp_dspark

        tokens = self.batch
        heads, dim, proj = (
            self.shard.heads_local,
            self.shard.head_dim,
            self.shard.proj_dim,
        )
        x_q, x_k, x_v, output_gate = (
            t.view(1, tokens, heads, dim)
            for t in qkvg.split([proj] * 4, dim=-1)
        )
        w_q, w_k, w_v = self.conv_weight_f32.split([proj] * 3, dim=0)
        cs_q, cs_k, cs_v, ic_q, ic_k, ic_v = self._conv_views()
        with torch.profiler.record_function("kda.mtp_verify_cute"):
            output = fused_kda_decode_mtp_dspark(
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                cs_q=cs_q,
                cs_k=cs_k,
                cs_v=cs_v,
                g=raw_gate.view(1, tokens, heads, dim),
                beta=beta.view(1, tokens, heads),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                recurrent_state=self.ssm_state,
                intermediate_ssm=None,
                intermediate_state_indices=self.inter_state_indices,
                intermediate_conv_q=ic_q,
                intermediate_conv_k=ic_k,
                intermediate_conv_v=ic_v,
                ssm_state_indices=self.state_indices,
                cu_seqlens=self.cu_seqlens,
                lower_bound=GATE_LOWER_BOUND,
                scale=dim**-0.5,
                replayssm_rawv=self.ring_rawv,
                replayssm_rawk=self.ring_rawk,
                replayssm_g=self.ring_g,
                replayssm_beta=self.ring_beta,
                onorm_gate=output_gate,
                onorm_weight=self.o_norm_weight_f32,
                onorm_eps=RMS_EPS,
            )
        return output.view(tokens, proj)

    def prefill(self, hidden=None, cu_seqlens=None):
        raise RuntimeError("MTP1 KDA blocks are decode(verify)-only")

    def forward(
        self,
        hidden: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is not None or cu_seqlens is not None:
            raise ValueError("MTP1 dispatch does not accept prefill inputs")
        with torch.profiler.record_function(self.RECORD_SPAN):
            return self.decode()


class TrtKimiK3KdaBlock(_KimiK3KdaMtpBlock):
    """Kimi-K3 KDA MTP1-verify block following the TRT-LLM stack.

    Two documented provenances, selected by ``sgl_replay``:

    **Default (``sgl_replay=False``) — NATIVE b10-1.3.0rc19.** Aligns to
    ``traces/trt_rc19_decode_mtp1_bs1_rank0.trace.json.gz`` (native rc19
    server, checkout ``12e51f84e1``, image ``44d66979``; per-layer sequence in
    ``traces/mtp1_decode_layer_kernels.md``). Source: ``kda_mixer.py::
    KimiDeltaAttention.forward``'s ``use_cute_mtp_replay`` branch — the
    default target-verify path with ``TLLM_K3_KDA_SGL_REPLAY`` unset.
    Per-layer kernels: (1) TRT ``attn_res_fwd_online_v2`` (bank aggregation +
    pending o_proj ``delta`` add + input RMSNorm), (2-3) ONE cuBLASLt fused
    in-proj GEMM over the full ``in_proj_qkvgfab`` weight incl. the
    ``[f_a|beta]`` and pad rows (``nvjet_..._splitK_TNT`` +
    ``splitKreduce``), (4) plain ``f_b_proj`` GEMM (``nvjet_..._TNN``),
    (5) the graph-stable Philox seed advance (``seed.add_(1)``, an int64
    elementwise add — production runs an FP16 recurrent-state cache with
    stochastic-rounding checkpoint stores), (6) the rc19-native fused verify
    ``kda_replay_ssm_conv_gated_wychunk`` (conv4 + SiLU + WY chunk replay-SSM
    + gated RMSNorm, imported via :func:`native_replay_verify_op`),
    (7) o_proj ``nn.Linear`` (nvjet TNT), (8) TP8 all-reduce.

    **``sgl_replay=True`` — OLD fork baseline.** Reproduces
    ``traces/baseline_trt_mtp1_decode.trace.json.gz`` (fork branch
    ``yikai/k3-decode-opt`` @ ``f44b703966`` with ``TLLM_K3_KDA_SGL_REPLAY=1``,
    side-chain form ``sgl_form`` — ``kda_mixer.py::KimiDeltaAttention.
    _sgl_replay_verify``): the base class's sglang-vendored middle — TGV
    ``[q|k|v|g]`` in-proj, tiny ``[f_a|beta]``/``f_b`` GEMVs (aux stream),
    ``kda_decode_mtp`` dspark verify with ReplaySSM rings — between the same
    AttnRes/o_proj/AR ends. The constructor flag mirrors the fork env switch:
    when ``sgl_replay=None``, ``TLLM_K3_KDA_SGL_REPLAY=1`` selects fork mode.

    The AR kernel is identical in both baselines: TRT-LLM's
    ``allreduce_fusion_kernel_oneshot_lamport`` (pattern 0/kNone: a plain
    AR — the residual add is deferred into the NEXT layer's AttnRes delta,
    which kernel #1 reproduces). At world > 1 the block reaches that exact
    kernel through ``Collectives.all_reduce(impl="trt")``, whose trt backend
    calls ``torch.ops.trtllm.allreduce`` with the MIN_LATENCY strategy and
    fusion NONE (pattern 0) on a real TRT-LLM ``Mapping``/Lamport workspace.
    At world = 1 no AR kernel launches (shared bench convention).

    Native-mode deviations beyond the shared ones in the base class: the
    recurrent-state pool is FP16 (the production ``mamba_ssm_cache_dtype``
    that makes the seed advance appear; fork mode keeps the base's FP32
    pool), and the replay ring (``old_u``/``old_k``/``old_G``/``pnat``/
    ``buf_idx``) is bench-batch sized with zero accepted history, so every
    forward verifies from the checkpoint exactly like the production kernel
    on a fresh request.
    """

    BACKENDS = ("trt_mtp1",)
    DEFAULT_DECODE_BACKEND = "trt_mtp1"
    DEFAULT_PREFILL_BACKEND = "trt_mtp1"
    RECORD_SPAN = "TrtKimiK3Kda"
    AR_IMPL = "trt"

    #: rc19 replay-ring geometry (kernel constants HIST and the two ring
    #: halves; see ``kda_replay_ssm_conv_gated_wychunk``).
    REPLAY_HIST = 16
    REPLAY_HALVES = 2

    def __init__(self, *args, sgl_replay: bool | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sgl_replay is None:
            sgl_replay = os.environ.get("TLLM_K3_KDA_SGL_REPLAY") == "1"
        self.sgl_replay = bool(sgl_replay)
        requests = self.num_requests
        heads, dim = self.shard.heads_local, self.shard.head_dim
        qkv_dim = self.shard.qkv_dim
        device = self.conv_weight.device
        # TRT hybrid-cache layouts: conv state [slot, channel, window],
        # intermediate window [slot, token, channel, window].
        self.conv_state = torch.zeros(
            requests, qkv_dim, self.shard.conv_size - 1,
            device=device, dtype=torch.bfloat16,
        )
        self.register_buffer(
            "inter_conv_window",
            torch.zeros(
                requests, MTP_WIDTH, qkv_dim, self.shard.conv_size - 1,
                device=device, dtype=torch.bfloat16,
            ),
        )
        if self.sgl_replay:
            self._native_op = None
            return
        self._native_op = native_replay_verify_op()
        # FP16 checkpoint pool (production cache dtype in the rc19 baseline;
        # replaces the base's FP32 pool used by the dspark path).
        self.ssm_state = torch.zeros(
            requests, heads, dim, dim, device=device, dtype=torch.float16
        )
        # rc19 replay ring, zero accepted history: old_u/old_k BF16
        # [slot, half, hist, head, dim], old_G FP32 [slot, half, head, hist,
        # dim], pnat/buf_idx int32 per slot.
        self.register_buffer(
            "replay_old_u",
            torch.zeros(
                requests, self.REPLAY_HALVES, self.REPLAY_HIST, heads, dim,
                device=device, dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "replay_old_k", torch.zeros_like(self.replay_old_u)
        )
        self.register_buffer(
            "replay_old_g",
            torch.zeros(
                requests, self.REPLAY_HALVES, heads, self.REPLAY_HIST, dim,
                device=device, dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "replay_pnat",
            torch.zeros(requests, device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "replay_buf_idx",
            torch.zeros(requests, device=device, dtype=torch.int32),
        )
        # Graph-stable Philox seed; production advances it once per launch.
        self.register_buffer(
            "rand_seed",
            torch.zeros(1, device=device, dtype=torch.int64),
        )

    def _conv_views(self):
        proj = self.shard.proj_dim
        cs_q, cs_k, cs_v = self.conv_state.split([proj] * 3, dim=1)
        ic_q, ic_k, ic_v = self.inter_conv_window.split([proj] * 3, dim=2)
        return cs_q, cs_k, cs_v, ic_q, ic_k, ic_v

    def _native_verify(self, hidden: torch.Tensor) -> torch.Tensor:
        """Contract positions 2-6 of the native-rc19 sequence.

        Mirrors ``kda_mixer.py``'s ``use_cute_mtp_replay`` branch: one fused
        in-proj GEMM over the full packed weight, the plain ``f_b``
        projection, the Philox seed advance, then the wychunk kernel
        consuming strided views of the packed projection directly.
        """
        requests, width = self.num_requests, MTP_WIDTH
        heads, dim, proj = (
            self.shard.heads_local,
            self.shard.head_dim,
            self.shard.proj_dim,
        )
        with torch.profiler.record_function("kda.in_proj_qkvgfab"):
            # nvjet splitK TNT + splitKreduce (positions 2-3).
            projected = self.in_proj_qkvgfab(hidden)
        if self.in_proj_padding:
            projected = projected[..., : -self.in_proj_padding]
        # Detached: the CuTe wrapper exports via dlpack, which rejects
        # requires_grad outputs (production weights are plain tensors).
        projected = projected.detach()
        qkv_raw, output_gate, f_a, beta = projected.split(
            self.in_proj_split_sizes, dim=-1
        )
        with torch.profiler.record_function("kda.f_b_proj"):
            raw_gate = self.f_b_proj(f_a).detach()  # nvjet TNN (position 4).
        replay_shape = (requests, width, heads, dim)
        # Row strides of the packed projection are preserved; the kernel
        # consumes these views without a QKV packing launch.
        raw_q, raw_k, raw_v = (
            t.view(replay_shape)
            for t in qkv_raw.view(requests, width, -1).split([proj] * 3, dim=-1)
        )
        # Position 5: the int64 in-place seed add (FP16 checkpoint pool with
        # graph-stable Philox stochastic rounding, as in the baseline server).
        self.rand_seed.add_(1)
        with torch.profiler.record_function("kda.mtp_replay_wychunk"):
            output = self._native_op(
                q=raw_q,
                k=raw_k,
                v=raw_v,
                conv_weight=self.conv_weight.detach(),
                conv_state=self.conv_state,
                conv_state_out=self.inter_conv_window,
                g_raw=raw_gate.view(replay_shape),
                beta_raw=beta.view(requests, width, heads),
                A_log=self.A_log.detach(),
                dt_bias=self.dt_bias.detach(),
                lower_bound=GATE_LOWER_BOUND,
                S=self.ssm_state,
                state_indices=self.state_indices,
                old_u=self.replay_old_u,
                old_k=self.replay_old_k,
                old_G=self.replay_old_g,
                replay_indices=self.inter_state_indices,
                pnat=self.replay_pnat,
                buf_idx=self.replay_buf_idx,
                z=output_gate.view(replay_shape),
                w=self.o_norm.weight.detach(),
                norm_eps=RMS_EPS,
                enable_state_commit=True,
                rand_seed=self.rand_seed,
            )
        return output.view(self.batch, proj)

    def decode(self) -> torch.Tensor:
        hidden = self._attn_res_hidden()  # kernel 1 (attn_res_fwd_v2)
        if self.sgl_replay:
            # Fork provenance: TGV + tiny GEMVs + dspark verify (kernels 2-5
            # of the OLD fork contract).
            qkvg, raw_gate, beta = self._project_mtp(hidden)
            output = self._kda_verify(qkvg, raw_gate, beta)
        else:
            # Native rc19: fused in-proj + f_b + seed add + wychunk verify
            # (kernels 2-6 of the native contract).
            output = self._native_verify(hidden)
        with torch.profiler.record_function("kda.o_proj"):
            partial = self.o_proj(output)  # nvjet TNT
        return self._reduce(partial)  # AR (see docstring)


class SglKimiK3KdaBlock(_KimiK3KdaMtpBlock):
    """Kimi-K3 KDA MTP1-verify block following the sglang implementation.

    Aligns to ``traces/baseline_sglang_mtp1_decode.trace.json.gz`` (per-layer
    sequence in ``traces/mtp1_decode_layer_kernels.md``). Source: sglang @
    ``f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`` — ``srt/models/kimi_k3.py::
    KimiK3DeltaAttention.forward_qkvbfg_fused`` + ``srt/layers/attention/
    linear/kda_backend.py::_forward_target_verify -> _run_dspark_cutedsl_mtp``
    (ReplaySSM on), ``srt/layers/attn_residual.py`` (fast TMA path) and
    ``srt/layers/k3_ar_fusion.py``. All device code is imported from
    ``kernels/sgl_copied_kernels`` (vendored at that same commit).

    Per-layer kernels: (1) ``attn_res_fused_tma`` on the pre-added prefix,
    (2) TGV ``[q|k|v|g]`` in-proj, (3-4) tiny ``[f_a|beta]``/``f_b`` GEMVs,
    (5) fused CuTe KDA verify with ReplaySSM rings, (6) o_proj on the TGV
    GEMM (``cutedsl_bf16_gemm``, the trace's second TGV launch), (7) fused
    push all-reduce + residual fold.

    Deviations beyond the shared ones in ``_KimiK3KdaMtpBlock``:

    * Trace kernel #7 (``all_reduce_push_res_kernel``, CustomAllReduceV2 1shot
      push with the residual/prefix add folded in) runs GENUINELY when the
      lazily resolved fused-AR state is live (``kernels.sgl_adapters.comm.
      get_sgl_ar_state`` on first decode — sglang's ``k3_ar_fusion._get_state``
      pattern; or injected via :meth:`attach_sgl_ar`). When it resolves None
      the harness falls back to ``Collectives.all_reduce`` at world > 1 (none
      at world = 1) plus ONE explicit elementwise add for the residual fold,
      writing the next prefix into a persistent buffer.
    * The conv-state pool is stored in sglang's [slot, window, channel] order
      and viewed transposed for the kernel, exactly like
      ``_run_dspark_cutedsl_mtp`` does.
    """

    BACKENDS = ("sgl_mtp1",)
    DEFAULT_DECODE_BACKEND = "sgl_mtp1"
    DEFAULT_PREFILL_BACKEND = "sgl_mtp1"
    RECORD_SPAN = "SglKimiK3Kda"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sgl_world = 0  # 0 = fallback tail; a live AR state enables push
        # Fused-AR state (kernels.sgl_adapters.comm.SglArState): resolved
        # lazily on first decode, or injected via attach_sgl_ar.
        self._sgl_ar_state = None
        self._sgl_ar_resolved = False
        requests = self.num_requests
        qkv_dim = self.shard.qkv_dim
        device = self.conv_weight.device
        # sglang mamba-pool layouts: conv state [slot, window, channel],
        # intermediate window [slot, token, window, channel].
        self.conv_state = torch.zeros(
            requests, self.shard.conv_size - 1, qkv_dim,
            device=device, dtype=torch.bfloat16,
        )
        self.register_buffer(
            "inter_conv_window",
            torch.zeros(
                requests, MTP_WIDTH, self.shard.conv_size - 1, qkv_dim,
                device=device, dtype=torch.bfloat16,
            ),
        )
        # attn_res_fused_tma consumes the precomputed bf16 product
        # score_norm_weight * score_proj_weight (attn_residual.get_cw).
        self.register_buffer(
            "attn_res_cw",
            (
                self.attn_res.norm_weight.float()
                * self.attn_res.proj_weight.reshape(-1).float()
            ).to(torch.bfloat16),
        )
        self.register_buffer(
            "next_prefix", torch.zeros_like(self.prefix_sum)
        )

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "attn_res_cw"):
            self.attn_res_cw.copy_(
                (
                    self.attn_res.norm_weight.float()
                    * self.attn_res.proj_weight.reshape(-1).float()
                ).to(torch.bfloat16)
            )

    def attach_sgl_ar(self, state) -> None:
        """Inject a prebuilt fused-AR state (overrides the lazy self-init).

        ``state`` is the ``SglArState`` from ``kernels.sgl_adapters.comm``
        (``build_sgl_ar_state`` / ``get_sgl_ar_state``); it owns the live
        CustomAllReduceV2 push planes. Normally the block resolves this
        itself on first decode. The o_proj partial ([T, 7168] bf16, 28 KB
        at M=2) fits the tuned 768 KB push slot."""
        from .kernels.sgl_adapters import all_reduce as sgl_ar

        # Idempotent for the same communicator; the state constructors
        # already registered it.
        sgl_ar.register_comm(state.comm)
        self._sgl_ar_state = state
        self._sgl_ar_resolved = True
        self._sgl_world = int(state.world_size)

    def _ensure_sgl_ar(self) -> None:
        """Resolve the fused-AR state once, lazily, on first decode —
        sglang's ``k3_ar_fusion._get_state`` pattern. All ranks reach decode
        in lockstep, so the collective build inside ``get_sgl_ar_state`` is
        safe; an "unavailable" verdict is cached and the block keeps its
        documented fallback tail."""
        if self._sgl_ar_resolved:
            return
        self._sgl_ar_resolved = True
        from .kernels.sgl_adapters import comm as sgl_comm

        state = sgl_comm.get_sgl_ar_state()
        if state is not None:
            self.attach_sgl_ar(state)

    def _conv_views(self):
        proj = self.shard.proj_dim
        cs_q, cs_k, cs_v = (
            t.transpose(-1, -2)
            for t in self.conv_state.split([proj] * 3, dim=-1)
        )
        ic_q, ic_k, ic_v = self.inter_conv_window.transpose(-1, -2).split(
            [proj] * 3, dim=-2
        )
        return cs_q, cs_k, cs_v, ic_q, ic_k, ic_v

    def _attn_res_hidden(self) -> torch.Tensor:
        from .kernels.sgl_adapters.attn_res import attn_res_fused_tma

        out = torch.empty_like(self.prefix_sum)
        with torch.profiler.record_function("kda.attn_res_tma"):
            attn_res_fused_tma(
                self.prefix_sum,
                self.block_residual,
                self.attn_res_cw,
                # Detached for dlpack export (see _project_mtp).
                self.input_norm_weight.detach(),
                out,
                self.prev_valid_blocks,
                RMS_EPS,
                write_prefix=self.is_block_write_layer,
            )
        return out

    def decode(self) -> torch.Tensor:
        self._ensure_sgl_ar()
        hidden = self._attn_res_hidden()  # trace kernel 1 (fused TMA)
        qkvg, raw_gate, beta = self._project_mtp(hidden)  # kernels 2-4
        output = self._kda_verify(qkvg, raw_gate, beta)  # kernel 5
        with torch.profiler.record_function("kda.o_proj_tgv"):
            from .kernels.sgl_adapters.gemm import cutedsl_bf16_gemm

            # Detached for dlpack export (see _project_mtp).
            partial = cutedsl_bf16_gemm(
                output, self.o_proj.weight.detach()
            )  # kernel 6
        if self._sgl_world > 1:
            # Kernel 7, genuine: 1shot push AR with the prefix/residual
            # add folded in, in place on the o_proj partial.
            from .kernels.sgl_adapters.all_reduce import all_reduce_push_res

            with torch.profiler.record_function("kda.ar_push_res"):
                return all_reduce_push_res(
                    self._sgl_world, partial, self.prefix_sum
                )
        reduced = self._reduce(partial)
        # Kernel 7 stand-in (no communicator attached): production folds
        # this add into all_reduce_push_res_kernel (see class docstring).
        with torch.profiler.record_function("kda.ar_residual_fold"):
            torch.add(self.prefix_sum, reduced, out=self.next_prefix)
        return self.next_prefix


__all__ = [
    "ALL_BACKENDS",
    "B10_BACKENDS",
    "DECODE_MAX_TOKENS",
    "KimiK3KDA",
    "KimiK3KDAB10",
    "MTP_WIDTH",
    "SglKimiK3KdaBlock",
    "TRT_BACKENDS",
    "TrtKimiK3KdaBlock",
    "_graft",
]
