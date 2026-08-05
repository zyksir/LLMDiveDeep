"""CuTeDSL KDA decode recurrence with fused causal conv1d."""

from ._b10_kda_decode_impl_cutedsl import kda_decode_conv_step

__all__ = ["kda_decode_conv_step"]
