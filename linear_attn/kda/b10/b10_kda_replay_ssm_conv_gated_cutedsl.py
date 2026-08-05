"""CuTeDSL replay-SSM with fused causal conv1d and gated RMSNorm."""

from ._b10_kda_replay_ssm_conv_gated_impl_cutedsl import (
    kda_replay_ssm_conv_gated,
)

__all__ = ["kda_replay_ssm_conv_gated"]
