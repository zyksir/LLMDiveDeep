"""Vendored sglang Kimi-K3 MLA decode ops (see package docstring).

The four kernels sglang's K3 MLA decode path launches between the shared
TGV GEMMs / flashinfer norms and the trtllm-gen fmha:

* ``dsv3_fused_a_gemm`` — the min-latency fused [q_a | kv_a | k_rope]
  down-projection (``sglang::fused_a_gemm_kernel``), M in [1, 16],
  ``mat_b`` is ``weight.T`` (column-major view of the row-major weight).
* ``concat_mla_absorb_q`` — [q_nope_absorbed 512 | q_pe 64] -> 576 concat.
* ``set_mla_kv_buffer`` — TMA bulk-store scatter of packed
  [k_nope | k_rope] rows into the paged latent KV pool.
* ``kimi_k3_mla_output_gate`` — ``out = x * sigmoid(gate)`` in one launch.

On the fp8-KV decode path sglang fuses the quantize + KV scatter + q concat
into ONE launch (``set_mla_kv_concat_q_fp8``); the ops above are its
documented fallback chain (the unfused path also spends three aten
``.to(fp8)`` casts — ``mla_quantize_without_rope_for_fp8`` in sglang's
``kernels/ops/attention/utils.py`` is exactly ``concat_mla_absorb_q`` plus
those three casts, replicated at call sites instead of vendoring the
triton-heavy utils module).

Baseline-name note: the production v0.5.18 trace shows this fused step as
``set_mla_kv_concat_q_fp8_triton_kernel``; at the vendored commit
(f8cbf000f4) upstream had replaced that Triton kernel with the CUDA JIT
``sglang::set_mla_kv_concat_q_fp8`` — same single launch, same fusion
boundary, different kernel name.
"""

from ..sgl_copied_kernels.ops.attention.concat_mla import concat_mla_absorb_q
from ..sgl_copied_kernels.ops.attention.set_mla_kv_concat_q import (
    covered_fp8 as set_mla_kv_concat_q_fp8_covered,
)
from ..sgl_copied_kernels.ops.attention.set_mla_kv_concat_q import (
    set_mla_kv_concat_q_fp8,
)
from ..sgl_copied_kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm
from ..sgl_copied_kernels.ops.kimi_k3.mla_output_gate import (
    covered as mla_output_gate_covered,
)
from ..sgl_copied_kernels.ops.kimi_k3.mla_output_gate import (
    kimi_k3_mla_output_gate,
)
from ..sgl_copied_kernels.ops.kvcache.set_mla_kv_buffer import (
    can_use_set_mla_kv_buffer,
    set_mla_kv_buffer,
)

__all__ = [
    "can_use_set_mla_kv_buffer",
    "concat_mla_absorb_q",
    "dsv3_fused_a_gemm",
    "kimi_k3_mla_output_gate",
    "mla_output_gate_covered",
    "set_mla_kv_buffer",
    "set_mla_kv_concat_q_fp8",
    "set_mla_kv_concat_q_fp8_covered",
]
