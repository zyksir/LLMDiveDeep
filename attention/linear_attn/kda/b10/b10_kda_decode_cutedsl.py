"""CuTeDSL KDA decode recurrence without conv or gated RMSNorm."""

from ._b10_kda_decode_impl_cutedsl import kda_decode_step

__all__ = ["kda_decode_step"]
