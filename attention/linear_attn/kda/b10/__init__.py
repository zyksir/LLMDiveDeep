"""b10-authored KDA kernels (CuTeDSLGen champions + the Triton FlashKDA port).

Every module here is ``b10_``-prefixed so provenance is obvious in imports,
tables, and profiles. Each CuTeDSL module's docstring opens with:

- ``IDENTICAL TO`` — the exact SGLang / TRT-LLM / FLA Triton (or CUTLASS)
  kernel(s) it replaces: same end-to-end *function*, just faster.
- ``DROP-IN API?`` — whether the Python signature matches that Triton entry
  so a user can one-line swap. Today: none are true YES; two are ALMOST
  (gated RMSNorm, TRT ``replay_ssm_fused`` verify). Everything else needs the
  thin adapter in ``kda_*_register.py``.

All CuTeDSL kernels are compiled for head_dim K = V = 128 (Kimi K3) and
assert that at their entry points.

Public module names follow:
``b10_kda_<operation>[_conv][_gated][_wychunk]_<backend>``.
``conv`` means causal conv1d is fused, ``gated`` means gated RMSNorm is fused,
and the final suffix is always the implementation backend.

| module | entry |
|---|---|
| b10_kda_decode_cutedsl | kda_decode_step |
| b10_kda_decode_gated_cutedsl | kda_decode_gated |
| b10_kda_decode_conv_cutedsl | kda_decode_conv_step |
| b10_kda_decode_conv_gated_cutedsl | kda_decode_conv_gated |
| b10_kda_gated_rmsnorm_cutedsl | gated_rmsnorm (standalone, not decode) |
| b10_kda_prefill_cutedsl | kda_spec_decode |
| b10_kda_chunk_prefill_cutedsl | kda_chunk_prefill |
| b10_kda_prefill_triton | kda_chunk_prefill |
| b10_kda_save_ssm_cutedsl | kda_save_ssm |
| b10_kda_save_ssm_gated_cutedsl | kda_save_ssm_gated |
| b10_kda_save_ssm_conv_cutedsl | kda_save_ssm_conv |
| b10_kda_save_ssm_conv_gated_cutedsl | kda_save_ssm_conv_gated |
| b10_kda_replay_ssm_cutedsl | kda_replay_ssm |
| b10_kda_replay_ssm_gated_cutedsl | kda_replay_ssm_gated |
| b10_kda_replay_ssm_conv_cutedsl | kda_replay_ssm_conv |
| b10_kda_replay_ssm_conv_gated_cutedsl | kda_replay_ssm_conv_gated |
| b10_kda_replay_ssm_conv_gated_wychunk_cutedsl | kda_replay_ssm_conv_gated_wychunk |
"""
