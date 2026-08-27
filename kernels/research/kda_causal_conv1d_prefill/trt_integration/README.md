# TRT-LLM integration surface

Snapshot of the TensorRT-LLM side of the KDA prefill causal-conv path, kept
here so the LLMDiveDeep commit contains the complete kernel work. The live
copies are in the `/node-storage/trt-llm` working tree (uncommitted there).

| File | Role |
| --- | --- |
| `causal_conv1d_prefill.py` | New stable integration surface: row-major one-launch `causal_conv1d_prefill(...)` plus the `b10_kda_conv1d_prefill_kernel` named Triton wrapper with grouped q/k/v output (`qkv_group_size` / `qkv_group_tokens`). Destination in TRT-LLM: `tensorrt_llm/_torch/modules/mamba/causal_conv1d_prefill.py`. |
| `causal_conv1d_triton.patch` | Diff against TRT-LLM git HEAD for `tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py`: adds `CONV_FWD_BLOCK_N`, `out=`/`kernel=`/`kernel_meta=` plumbing, and keeps FP32 operand promotion in the fwd accumulate (precision requirement). |
| `test_causal_conv1d_prefill.py` | Unit tests (21 cases): pipeline equivalence vs the unfused native path, padded-slot skip semantics, and bit-exact grouped-output checks. Destination: `tests/unittest/_torch/modules/mamba/test_causal_conv1d_prefill.py`. |

The CuTe DSL kernel in the parent package implements the same call signature
(verified in `local_results/r3_trt_surface_alignment.json`), so it can replace
the Triton kernel inside `causal_conv1d_prefill` without caller changes.
