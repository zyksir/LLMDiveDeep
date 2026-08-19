"""Kimi-K3-owned access to the shared B10 decode kernels.

The concrete CuTeDSL implementation remains in ``linear_attn`` because its
benchmark registry and non-Kimi callers share it. This module is the narrow
ownership boundary used by the Kimi-K3 layer.
"""

from __future__ import annotations

import sys
from pathlib import Path


def decode_kernels():
    """Load the fused and overlap-path B10 decode entry points lazily."""
    linear_attn = Path(__file__).resolve().parents[2] / "linear_attn"
    if str(linear_attn) not in sys.path:
        sys.path.insert(0, str(linear_attn))
    from kda.b10.b10_kda_decode_conv_gated_cutedsl import (
        kda_decode_conv_gated_raw,
        kda_decode_gated_raw_strided,
    )

    return kda_decode_conv_gated_raw, kda_decode_gated_raw_strided
