"""CuTeDSL replay-SSM recurrence with fused causal conv1d."""

from ._b10_kda_replay_ssm_impl_cutedsl import (
    kda_replay_ssm_fused_conv as kda_replay_ssm_conv,
)

__all__ = ["kda_replay_ssm_conv"]
