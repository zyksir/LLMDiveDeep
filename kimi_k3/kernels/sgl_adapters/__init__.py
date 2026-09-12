"""Harness-facing adapters over the vendored sglang stack.

This package is the SINGLE sanctioned surface through which harness code
(blocks, benches, tools) reaches sglang functionality. The vendored tree at
``kernels/sgl_copied_kernels/`` stays a pristine copy of the upstream sglang
sources (kernels + JIT loaders + shims; see its README for the pinned commit
and per-file provenance) so it can be swapped wholesale when re-vendoring to
a newer sglang commit. Nothing outside this package may import
``sgl_copied_kernels`` internals directly.

Modules (each wraps exactly the surface the harness uses — no speculative
API):

* ``comm`` — CustomAllReduceV2 workspace setup: :class:`~.comm.SglArState`,
  the lazy per-process :func:`~.comm.get_sgl_ar_state` (sglang's
  ``k3_ar_fusion._get_state`` pattern) and the collective
  :func:`~.comm.build_sgl_ar_state`.
* ``all_reduce`` — the fused AR ops (``register_comm``, push family
  ``all_reduce_push_res`` / ``all_reduce_push_norm`` /
  ``finalize_all_reduce_push_norm``, NVLS pull family
  ``all_reduce_pull_res`` / ``all_reduce_pull_norm``, ``NORM_DIM``).
* ``comm_gemm`` — the fused GEMM+comm ops (``o_proj_gemm_ar`` +
  ``ensure_gemm_ar``/``gemm_ar_fits``, ``gemm_ag_up_proj``).
* ``moe`` — the fused route+quant front (``route_quant_fused`` /
  ``route_quant_fused_covered``) and the radix router fallback
  (``route_radix`` / ``route_radix_covered``).
* ``activation`` — ``situ_and_mul`` (SiTU shared-experts activation).
* ``gemm`` — ``cutedsl_bf16_gemm`` (TGV GEMM) and ``kimi_k3_tiny_gemm``.
* ``kda`` — ``fused_kda_decode_mtp_dspark`` (fused KDA MTP verify).
* ``attn_res`` — ``attn_res_fused_tma`` (attention-residual TMA kernel).
* ``mla`` — the K3 MLA decode ops (``dsv3_fused_a_gemm``,
  ``concat_mla_absorb_q``, ``set_mla_kv_buffer``,
  ``kimi_k3_mla_output_gate``).

Import adapter modules lazily (inside methods), like the vendored modules
they wrap: importing them pulls in the sglang JIT stack (tvm-ffi, CuTe DSL)
and may trigger kernel builds on first use.
"""
