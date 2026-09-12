"""High-level contracts shared by the linear-attention benchmarks.

This module intentionally stays small. It holds only:

* ``Shape`` — the per-rank attention shape every bench sweeps over;
* ``BackendRegistry`` — re-exported from ``common/kernel_bench.py``, the
  mechanism through which each framework kernel (FLA, SGLang, vLLM,
  TRT-LLM, FlashKDA, FlashQLA, ...) is registered.

The actual backend registrations live next to their input contracts:

* ``gdn/gdn_attention.py`` — GDN (scalar-gate) decode/prefill backends;
* ``kda/`` — KDA (per-channel-gate): naive math in ``kda_attention.py``,
  registers in ``kda_*_register.py``, b10 kernels under ``kda/b10/``;
* ``frameworks.py`` — shared vLLM/TRT-LLM leaf-module import shims.

Tensor layouts follow SGLang:

* packed decode input: ``mixed_qkv[B, 2*H*K + HV*V]``
* recurrent state pool: ``state[B, HV, V, K]`` (V-first)
* prefill input: ``q/k/v/g[1, total_tokens, heads, dim]`` plus ``cu_seqlens``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.kernel_bench import BackendRegistry  # noqa: E402,F401  (re-export)


@dataclass(frozen=True)
class Shape:
    """Per-rank KDA/GDN shape.

    The KDA benches sweep Kimi K3's shard shapes by default: H=12 (one TP8
    rank of 96 global heads) and H=96 (a TP1/pipeline-parallel rank). The
    dataclass default of 16 (one TP=2 shard of Kimi-Linear's 32 heads) is
    kept for the GDN/module benches. The default fp32 state follows SGLang's
    state-pool policy; use ``state_dtype="bfloat16"`` to study the
    lower-memory override.
    """

    qk_heads: int = 16
    value_heads: int = 16
    key_dim: int = 128
    value_dim: int = 128
    state_dtype: str = "float32"

    @property
    def qkv_dim(self) -> int:
        return (
            2 * self.qk_heads * self.key_dim
            + self.value_heads * self.value_dim
        )

    @property
    def state_bytes_per_request(self) -> int:
        element_size = 4 if self.state_dtype == "float32" else 2
        return element_size * self.value_heads * self.value_dim * self.key_dim

    @property
    def torch_state_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.state_dtype]


