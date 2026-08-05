"""CuTeDSL replay-SSM recurrence with fused gated RMSNorm."""

from ._b10_kda_replay_ssm_conv_gated_impl_cutedsl import (
    kda_replay_ssm_gated_noconv as kda_replay_ssm_gated,
)

__all__ = ["kda_replay_ssm_gated"]
