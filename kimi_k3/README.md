# kimi_k3 — Kimi-K3 decode-layer optimization workspace (MoE / KDA)

Self-contained production blocks (`b10_kimi_k3_moe_layer.py`,
`b10_kimi_k3_kda_layer.py`) and their trace-alignment benches. Kernel code
lives under `kernels/`; sglang functionality is reached exclusively through
`kernels/sgl_adapters/` (single sanctioned surface — the vendored tree at
`kernels/sgl_copied_kernels/` stays pristine and swappable).

## Arch gaps (retained from the removed capabilities.py)

* trtllmGen's `MxE4m3MxE2m1BlockScaleMoERunner` ships **SwiGlu-only** MXFP4
  MoE GEMM kernels on **sm_103** (B300/GB300): constructing it with
  `act=SiTU` raises "No kernel found" (measured on GB300 with a direct
  runner unit test). sm_100 (B200/GB200) ships both. Fallback on sm_103 with
  SiTU experts: the FlashInfer op backend.
* **SiTU is not signalled by `activation_type`.** TRT-LLM computes
  `_is_situ_activation = (activation_type == Swiglu) and
  pretrained_config.hidden_act == "situ"` (`fused_moe_trtllm_gen.py`), so a
  caller passing `ActivationType.Swiglu` still runs SiTU on a real K3
  checkpoint (`hidden_act` comes from the checkpoint's config.json). This
  repo's `k3_pretrained_config()` sets `hidden_act="situ"` deliberately so
  the bench measures production's SiTU expert path; the sm_103 gap therefore
  APPLIES to any serving path that pins the trtllmGen runner with SiTU.
* SiTU exists **only** in the fork: stock rc19/rc23 have no
  `_torch/modules/situ.py` and no SiTU reference in `moe_op_backend.py`.
