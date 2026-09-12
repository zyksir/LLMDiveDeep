"""4-transformer-block Kimi-K3 CP-vs-TP demo (3 KDA + 1 MLA blocks + MoE).

Reuses the ``tp_baseline`` MoE layer unchanged; attention modules are imported
directly from the installed TRT-LLM build (KimiDeltaAttention, KimiMLAAttention
via ``_kimi_attention_factory``). One code path; TP vs CP is config-only.
"""
