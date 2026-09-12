"""Isolated Kimi-style attention modules for implementation comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from linear_attention import Shape


@dataclass(frozen=True)
class AttentionShape:
    hidden_size: int = 2048
    heads: int = 16
    head_dim: int = 128
    conv_size: int = 4

    @property
    def core(self) -> Shape:
        return Shape(self.heads, self.heads, self.head_dim, self.head_dim)


class SGLangStyleKDAAttention(nn.Module):
    """Self-contained TP=1 equivalent of SGLang's fused KDA attention path."""

    def __init__(
        self,
        shape: AttentionShape,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from sglang.srt.layers.attention.fla.fused_norm_gate import (
            FusedRMSNormGated,
        )
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        self.shape = shape
        qkv_dim = 3 * shape.heads * shape.head_dim
        # q/k/v/beta plus the two low-rank gate inputs in one GEMM.
        self.in_proj = nn.Linear(
            shape.hidden_size,
            qkv_dim + shape.heads + 2 * shape.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.fg_weight = nn.Parameter(
            torch.empty(
                2,
                shape.heads * shape.head_dim,
                shape.head_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.conv_weight = nn.Parameter(
            torch.empty(
                qkv_dim,
                shape.conv_size,
                device=device,
                dtype=torch.float32,
            )
        )
        self.A_log = nn.Parameter(
            torch.zeros(shape.heads, device=device, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(
                shape.heads * shape.head_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        self.o_norm = FusedRMSNormGated(
            shape.head_dim,
            eps=1e-5,
            activation="sigmoid",
            device=torch.device(device),
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            shape.heads * shape.head_dim,
            shape.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.kda = TritonKDAKernel()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.fg_weight, std=0.02)
        nn.init.normal_(self.conv_weight, std=0.02)
        nn.init.ones_(self.o_norm.weight)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )

        shape = self.shape
        fused = self.in_proj(hidden_states)
        mixed_qkv, beta, fg = torch.split(
            fused,
            [
                3 * shape.heads * shape.head_dim,
                shape.heads,
                2 * shape.head_dim,
            ],
            dim=-1,
        )
        fg = fg.view(-1, 2, shape.head_dim).transpose(0, 1)
        forget_gate, norm_gate = torch.bmm(
            fg, self.fg_weight.transpose(1, 2)
        ).unbind(0)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv_weight,
            activation="silu",
            conv_state_indices=state_indices,
        )
        out = self.kda.packed_decode(
            mixed_qkv,
            forget_gate,
            beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=shape.head_dim**-0.5,
            ssm_states=recurrent_state,
            cache_indices=state_indices,
            num_v_heads=shape.heads,
            head_v_dim=shape.head_dim,
        )
        out = self.o_norm(
            out,
            norm_gate.view(1, -1, shape.heads, shape.head_dim),
        )
        return self.o_proj(out.squeeze(0).flatten(-2))


@torch.no_grad()
def get_attention_runners(
    batch: int,
    shape: AttentionShape,
    *,
    seed: int = 0,
) -> tuple[dict[str, Callable[[], torch.Tensor]], dict[str, str]]:
    """Build as-shipped FLA and serving-style SGLang attention-layer runners."""

    torch.manual_seed(seed)
    hidden = torch.randn(
        batch,
        1,
        shape.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    runners: dict[str, Callable[[], torch.Tensor]] = {}
    unavailable: dict[str, str] = {}

    try:
        from fla.layers.kda import KimiDeltaAttention
        from fla.models.utils import Cache

        fla_layer = KimiDeltaAttention(
            hidden_size=shape.hidden_size,
            head_dim=shape.head_dim,
            num_heads=shape.heads,
            num_v_heads=shape.heads,
            mode="fused_recurrent",
            use_short_conv=True,
            conv_size=shape.conv_size,
            layer_idx=0,
        ).to(device="cuda", dtype=torch.bfloat16).eval()
        fla_cache = Cache()
        fla_layer(hidden, past_key_values=fla_cache, use_cache=True)

        @torch.no_grad()
        def upstream_fla_layer():
            return fla_layer(
                hidden,
                past_key_values=fla_cache,
                use_cache=True,
            )[0]

        runners["upstream_fla_attention"] = upstream_fla_layer
    except Exception as exc:
        unavailable["upstream_fla_attention"] = f"{type(exc).__name__}: {exc}"

    try:
        sglang_layer = SGLangStyleKDAAttention(shape).eval()
        qkv_dim = 3 * shape.heads * shape.head_dim
        conv_state = torch.zeros(
            batch,
            qkv_dim,
            shape.conv_size - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        recurrent_state = torch.zeros(
            batch,
            shape.heads,
            shape.head_dim,
            shape.head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        state_indices = torch.arange(batch, device="cuda", dtype=torch.int32)

        @torch.no_grad()
        def sglang_attention():
            return sglang_layer(
                hidden.squeeze(1),
                recurrent_state,
                conv_state,
                state_indices,
            )

        sglang_attention()
        runners["sglang_fused_attention"] = sglang_attention
    except Exception as exc:
        unavailable["sglang_fused_attention"] = f"{type(exc).__name__}: {exc}"

    return runners, unavailable
