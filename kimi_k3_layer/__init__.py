"""Kimi-K3 decode-layer optimization workspace (comm / MoE / KDA).

Intentionally no package-level re-exports: files here must be
importable next to the INSTALLED tensorrt_llm wheel inside the trt-dev
container, so nothing at import time may touch sys.path/sys.modules
(the old linear_attn.frameworks stubs poisoned `import tensorrt_llm`).
Import submodules explicitly (kimi_k3_layer.comm,
kimi_k3_layer.moe_b10_kimi_k3, ...). See README.md.
"""
