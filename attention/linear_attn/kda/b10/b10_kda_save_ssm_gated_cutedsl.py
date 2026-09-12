"""CuTeDSL save-SSM recurrence with fused gated RMSNorm."""

from ._b10_kda_save_ssm_conv_gated_impl_cutedsl import (
    kda_save_ssm_gated_noconv as kda_save_ssm_gated,
)

__all__ = ["kda_save_ssm_gated"]
