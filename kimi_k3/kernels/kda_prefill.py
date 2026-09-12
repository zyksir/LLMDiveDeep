"""Kimi-K3-owned access to the shared B10 prefill kernel.

The concrete CuTeDSL implementation stays in ``linear_attn`` because it is
shared by that package's benchmark registry. Kimi-K3 imports it only here.
"""

from __future__ import annotations

import sys
from pathlib import Path


def prefill_kernel():
    """Load the B10 chunk-prefill entry point lazily."""
    linear_attn = Path(__file__).resolve().parents[2] / "linear_attn"
    if str(linear_attn) not in sys.path:
        sys.path.insert(0, str(linear_attn))
    from kda.b10.b10_kda_chunk_prefill_cutedsl import kda_chunk_prefill

    return kda_chunk_prefill
