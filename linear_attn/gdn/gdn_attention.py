"""GDN attention: input contract, exact reference, and registered backends.

GDN (scalar-gate) backends — one registered builder per framework kernel
(FLA, SGLang recurrent/packed, vLLM, TRT-LLM, FlashQLA). The vLLM/TRT-LLM
leaf-module import shims live in ``frameworks.py`` (shared with ``kda/``);
the registry mechanism lives in ``linear_attention.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frameworks import _import_trtllm_fla, _import_vllm_fla  # noqa: E402
from linear_attention import BackendRegistry, Shape  # noqa: E402


@dataclass
class GDNInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    beta_logit: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def log_decay(self) -> torch.Tensor:
        return -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.raw_gate.float() + self.dt_bias.view(1, 1, -1)
        )

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_logit.sigmoid()

    @property
    def mixed_qkv(self) -> torch.Tensor:
        return torch.cat(
            [self.q[:, 0].flatten(1), self.k[:, 0].flatten(1), self.v[:, 0].flatten(1)],
            dim=-1,
        ).contiguous()


def make_gdn_inputs(
    batch: int,
    seq_len: int,
    shape: Shape,
    *,
    seed: int = 0,
    device: str = "cuda",
) -> GDNInputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *s: torch.randn(
        *s, device=device, dtype=torch.bfloat16, generator=generator
    )
    return GDNInputs(
        q=(0.1 * randn(batch, seq_len, shape.qk_heads, shape.key_dim)).contiguous(),
        k=(0.1 * randn(batch, seq_len, shape.qk_heads, shape.key_dim)).contiguous(),
        v=(0.1 * randn(batch, seq_len, shape.value_heads, shape.value_dim)).contiguous(),
        raw_gate=(0.5 * randn(batch, seq_len, shape.value_heads) - 1).contiguous(),
        beta_logit=(0.5 * randn(batch, seq_len, shape.value_heads)).contiguous(),
        A_log=(0.2 * torch.randn(shape.value_heads, device=device, generator=generator)).float(),
        dt_bias=(0.1 * torch.randn(shape.value_heads, device=device, generator=generator)).float(),
        state=torch.zeros(
            batch,
            shape.value_heads,
            shape.value_dim,
            shape.key_dim,
            device=device,
            dtype=shape.torch_state_dtype,
        ),
        state_indices=torch.arange(batch, device=device, dtype=torch.int32),
        cu_seqlens=torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=device,
            dtype=torch.int32,
        ),
    )


@torch.no_grad()
def gdn_recurrent_reference(
    inputs: GDNInputs,
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(inputs.q.float(), dim=-1) * shape.key_dim**-0.5
    k = F.normalize(inputs.k.float(), dim=-1)
    groups = shape.value_heads // shape.qk_heads
    q, k = q.repeat_interleave(groups, 2), k.repeat_interleave(groups, 2)
    state = inputs.state.float().transpose(-1, -2).contiguous()
    out = torch.empty_like(inputs.v)
    for t in range(inputs.q.shape[1]):
        state.mul_(inputs.log_decay[:, t].exp()[..., None, None])
        residual = inputs.v[:, t].float() - torch.einsum(
            "bhk,bhkv->bhv", k[:, t], state
        )
        state.add_(
            torch.einsum(
                "bhk,bhv->bhkv",
                k[:, t],
                inputs.beta[:, t].float()[..., None] * residual,
            )
        )
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], state).to(out.dtype)
    return out, state.transpose(-1, -2).contiguous()


# ---------------------------------------------------------------------------
# Decode backends — one registered builder per framework kernel
# ---------------------------------------------------------------------------

GDN_DECODE = BackendRegistry("one-token GDN recurrence, matched inputs")


def _gdn_recurrent_kwargs(inputs: GDNInputs, shape: Shape) -> dict:
    return dict(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        g=inputs.log_decay,
        beta=inputs.beta,
        scale=shape.key_dim**-0.5,
        initial_state=inputs.state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


@GDN_DECODE.register("fla_gdn_recurrent")
def _fla_gdn_recurrent(inputs: GDNInputs, shape: Shape) -> Callable:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    kwargs = _gdn_recurrent_kwargs(inputs, shape)

    def run():
        return fused_recurrent_gated_delta_rule(**kwargs, state_v_first=True)

    return run


@GDN_DECODE.register("sglang_gdn_recurrent")
def _sglang_gdn_recurrent(inputs: GDNInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    kwargs = _gdn_recurrent_kwargs(inputs, shape)

    def run():
        return fused_recurrent_gated_delta_rule(**kwargs)

    return run


@GDN_DECODE.register("sglang_gdn_packed")
def _sglang_gdn_packed(inputs: GDNInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    packed_mixed_qkv = inputs.mixed_qkv
    packed_out = inputs.v.new_empty(
        inputs.q.shape[0], 1, shape.value_heads, shape.value_dim
    )

    def run():
        return fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=packed_mixed_qkv,
            a=inputs.raw_gate[:, 0],
            b=inputs.beta_logit[:, 0],
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            scale=shape.key_dim**-0.5,
            initial_state=inputs.state,
            out=packed_out,
            ssm_state_indices=inputs.state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    return run


@GDN_DECODE.register("vllm_gdn_recurrent")
def _vllm_gdn_recurrent(inputs: GDNInputs, shape: Shape) -> Callable:
    if inputs.q.shape[1] != 1:
        raise RuntimeError("vLLM decode kernel expects one token per request")
    vllm = _import_vllm_fla("fused_recurrent")
    kwargs = {
        key: value
        for key, value in _gdn_recurrent_kwargs(inputs, shape).items()
        if key not in ("initial_state", "output_final_state")
    }
    # vLLM reserves state slot 0; shift indices by one.
    state = torch.cat(
        [inputs.state.new_zeros(1, *inputs.state.shape[1:]), inputs.state]
    )
    state_indices = inputs.state_indices + 1

    def run():
        out, final_state = vllm.fused_recurrent_gated_delta_rule(
            **kwargs,
            initial_state=state,
            inplace_final_state=True,
            ssm_state_indices=state_indices,
        )
        return out, final_state[1:]

    return run


@GDN_DECODE.register("trtllm_gdn_recurrent")
def _trtllm_gdn_recurrent(inputs: GDNInputs, shape: Shape) -> Callable:
    trt = _import_trtllm_fla("fused_recurrent")
    kwargs = _gdn_recurrent_kwargs(inputs, shape)

    def run():
        return trt.fused_recurrent_gated_delta_rule(**kwargs)

    return run


def get_gdn_decode_backends(
    inputs: GDNInputs,
    shape: Shape,
) -> tuple[dict[str, Callable], dict[str, str]]:
    return GDN_DECODE.build(inputs, shape)


# ---------------------------------------------------------------------------
# Chunk-prefill backends — one registered builder per framework kernel
# ---------------------------------------------------------------------------

GDN_PREFILL = BackendRegistry("GDN chunk prefill, matched packed-varlen inputs")


def _gdn_chunk_kwargs(inputs: GDNInputs, shape: Shape) -> dict:
    pack = lambda tensor: tensor.reshape(1, -1, *tensor.shape[2:]).contiguous()
    return dict(
        q=pack(inputs.q),
        k=pack(inputs.k),
        v=pack(inputs.v),
        g=pack(inputs.log_decay),
        beta=pack(inputs.beta),
        scale=shape.key_dim**-0.5,
        initial_state=inputs.state,
        cu_seqlens=inputs.cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )


@GDN_PREFILL.register("fla_gdn_chunk")
def _fla_gdn_chunk(inputs: GDNInputs, shape: Shape) -> Callable:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    kwargs = _gdn_chunk_kwargs(inputs, shape)

    def run():
        # Force the Triton path: FLA auto-dispatches to FlashQLA when
        # installed, which is measured separately as flash_qla_gdn_chunk.
        import os

        prev = os.environ.get("FLA_FLASH_QLA")
        os.environ["FLA_FLASH_QLA"] = "0"
        try:
            return chunk_gated_delta_rule(
                **kwargs, output_final_state=True, state_v_first=True
            )
        finally:
            if prev is None:
                os.environ.pop("FLA_FLASH_QLA", None)
            else:
                os.environ["FLA_FLASH_QLA"] = prev

    return run


@GDN_PREFILL.register("sglang_gdn_chunk")
def _sglang_gdn_chunk(inputs: GDNInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    kwargs = _gdn_chunk_kwargs(inputs, shape)

    def run():
        return chunk_gated_delta_rule(
            **kwargs, initial_state_indices=inputs.state_indices
        )

    return run


@GDN_PREFILL.register("vllm_gdn_chunk")
def _vllm_gdn_chunk(inputs: GDNInputs, shape: Shape) -> Callable:
    vllm = _import_vllm_fla("chunk")
    kwargs = _gdn_chunk_kwargs(inputs, shape)

    def run():
        return vllm.chunk_gated_delta_rule(**kwargs, output_final_state=True)

    return run


@GDN_PREFILL.register("trtllm_gdn_chunk")
def _trtllm_gdn_chunk(inputs: GDNInputs, shape: Shape) -> Callable:
    trt = _import_trtllm_fla("chunk")
    kwargs = _gdn_chunk_kwargs(inputs, shape)

    def run():
        return trt.chunk_gated_delta_rule(**kwargs, output_final_state=True)

    return run


@GDN_PREFILL.register("flash_qla_gdn_chunk")
def _flash_qla_gdn_chunk(inputs: GDNInputs, shape: Shape) -> Callable:
    # QwenLM/FlashQLA: TileLang GDN chunk kernels, FLA-compatible API.
    from flash_qla import chunk_gated_delta_rule

    kwargs = dict(
        _gdn_chunk_kwargs(inputs, shape),
        cu_seqlens=inputs.cu_seqlens.to(torch.int64),
    )

    def run():
        with torch.inference_mode():
            return chunk_gated_delta_rule(
                **kwargs, output_final_state=True, state_v_first=True
            )

    return run


def get_gdn_prefill_backends(
    inputs: GDNInputs,
    shape: Shape,
) -> tuple[dict[str, Callable], dict[str, str]]:
    return GDN_PREFILL.build(inputs, shape)
