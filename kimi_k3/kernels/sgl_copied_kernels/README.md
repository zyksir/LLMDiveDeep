# sgl_copied_kernels — vendored sglang Kimi-K3 MoE kernels

Faithful copies of the sglang runtime-compiled kernels the Kimi-K3 MoE path
uses, so they build and run from this bench without a sglang install.

- **Source checkout:** `/node-storage/sglang`
- **Source commit:** `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`
- **Copy date:** 2026-09-10

## What was changed (and nothing else)

Copied `.py` files got a mechanical import-root rewrite only:

- `sglang.kernels.jit` / `sglang.kernels.ops` / `sglang.kernels.kernel_api_logging`
  → `kimi_k3.kernels.sgl_copied_kernels.{jit,ops,kernel_api_logging}`
- the handful of `sglang.srt.*` / `sglang.utils` imports (`envs`, `is_in_ci`,
  `is_npu`, `is_sm100_supported`, `get_cuda_version`,
  `direct_register_custom_op`, `register_custom_op`) → local stubs in
  `_sgl_shims/` (see its docstring).

CUDA sources (`.cu`/`.cuh`/`.h`) and the `jit/include/sgl_kernel/` header tree
are byte-identical copies. `jit/utils/compile/paths.py` resolves `csrc/` and
`include/` relative to the (renamed) `...sgl_copied_kernels.jit` package, so
the preserved directory layout keeps `load_jit` working unchanged.

## Provenance (upstream path → vendored path)

All upstream paths are relative to `python/sglang/` at the commit above.

| Vendored | Upstream | Notes |
|---|---|---|
| `jit/__init__.py`, `jit/utils/**/*.py` (13 files) | `kernels/jit/` same relative paths | tvm-ffi JIT build stack (arch/deps/cache/ninja/loader/spec/toolchain/cpp_args) |
| `jit/include/sgl_kernel/**` (entire tree, 27 headers) | `kernels/jit/include/sgl_kernel/**` | shared header closure (utils/math/vec/warp/runtime/tensor/type/tile/mbarrier/ffi, `distributed/{communicator,ptx}.cuh`, `deepseek_v4/fp8_utils.cuh`, ...) |
| `jit/csrc/kimi_k3/comm/{ar_fusion,gemm_ar,gemm_ag}.cuh` | `kernels/jit/csrc/kimi_k3/comm/` | fused push/pull AR (+RMSNorm, +deferred MoE finalize), GEMM+AR, GEMM+AG |
| `jit/csrc/kimi_k3/situ_and_mul.cuh` | `kernels/jit/csrc/kimi_k3/` | SiTU activation (+masked post-quant variant) |
| `jit/csrc/kimi_k3/attn_res/fused_tma.cuh` | `kernels/jit/csrc/kimi_k3/attn_res/` | attn-residual fused TMA kernel |
| `jit/csrc/moe/{route_quant_fused,route_radix}.cuh` | `kernels/jit/csrc/moe/` | K3 routing (radix top-16; fused route+pack+group-quant) |
| `jit/csrc/moe/moe_finalize_fuse_shared.cu` + `tvm_ffi_utils.h` | `kernels/jit/csrc/moe/` | finalize fused with shared-expert add |
| `jit/csrc/gemm/per_token_group_quant.cuh` | `kernels/jit/csrc/gemm/` | include dependency of `route_quant_fused.cuh` |
| `jit/csrc/elementwise/add3.cuh` | `kernels/jit/csrc/elementwise/` | 3-way add tail kernel |
| `jit/csrc/distributed/{custom_all_reduce,ipc,registry}.cuh` | `kernels/jit/csrc/distributed/` | CustomAllReduceV2 device side + tvm-ffi `Communicator` object |
| `ops/kimi_k3/{__init__,all_reduce,moe,activation,gemm_ag,gemm_ar}.py` | `kernels/ops/kimi_k3/` | python loaders/wrappers for the K3 kernels |
| `ops/moe/{moe_route_radix,moe_route_quant_fused,moe_finalize_fuse_shared}.py` | `kernels/ops/moe/` | routing / finalize loaders |
| `ops/gemm/cutedsl_bf16_gemm.py` | `kernels/ops/gemm/` | CuTe DSL TGV bf16 GEMM (runtime-compiled via `nvidia-cutlass-dsl`) |
| `ops/elementwise/add3.py` | `kernels/ops/elementwise/` | add3 loader |
| `ops/communication/all_reduce.py` | `kernels/ops/communication/` | CustomAllReduceV2 push/pull/barrier planes host objects |
| `kernel_api_logging.py` | `kernels/kernel_api_logging.py` | self-contained; `debug_kernel_api` / `debug_torch_op` |
| `jit/csrc/gemm/tiny_gemm.cuh` | `kernels/jit/csrc/gemm/` | tiny bf16 GEMM (N- and K-variant), byte-identical (KDA bench addition, 2026-09-10) |
| `ops/gemm/tiny_gemm.py` | `kernels/ops/gemm/` | tiny GEMM loader/wrapper (KDA `[f_a\|beta]` + `f_b` GEMVs; import-root rewrite only) |
| `ops/kimi_k3/attn_res.py` | `kernels/ops/kimi_k3/` | attn-residual fused TMA python wrapper (`attn_res_fused_tma`; csrc was already vendored; import-root rewrite only) |
| `ops/kimi_k3/kda_decode_mtp.py` | `kernels/ops/kimi_k3/` | CuTe DSL KDA MTP verify kernel + `fused_kda_decode_mtp_dspark`, byte-identical (no sglang imports) |
| `_sgl_shims/custom_op.py` | `srt/utils/custom_op.py` | faithful copy (import roots only) |
| `_sgl_shims/__init__.py` | — | NEW: minimal stubs for `sglang.srt`/`sglang.utils` symbols |
| `__init__.py`, `ops/__init__.py`, `ops/{moe,gemm,elementwise,communication}/__init__.py` | — | NEW: vendoring glue; upstream `ops/__init__` wires a kernel registry this subset does not carry |
| `_sgl_shims/custom_all_reduce_v2.py` | `srt/distributed/device_communicators/custom_all_reduce_v2.py` | CustomAllReduceV2 workspace construction + symm-mem rendezvous + multicast binding (import roots only; the relative `.configs.custom_all_reduce_v2` / `.custom_all_reduce_utils` imports are unchanged — the shims dir mirrors that layout). Supersedes the "workspace plumbing ... NOT vendored" note under Runtime requirements below; the standalone entry point is `kimi_k3/kernels/sgl_adapters/comm.py` (`get_sgl_ar_state`, lazy — the former `kimi_k3/sgl_ar_setup.py` was folded into it) |
| `_sgl_shims/configs/custom_all_reduce_v2.py` | `srt/distributed/device_communicators/configs/custom_all_reduce_v2.py` | byte-identical: tuned per-arch/world-size dispatch tables (SM90/SM100/SM107) |
| `_sgl_shims/custom_all_reduce_utils.py` | `srt/distributed/device_communicators/custom_all_reduce_utils.py` | capability gate (full-NVLink pynvml check, batched IPC P2P self-test, NVLink-clique probe); import roots only. Caveat inherited from upstream: `gpu_p2p_access_check` re-runs this file via `python <file>` in a subprocess, so the repo root must be importable there (PYTHONPATH), exactly as upstream needs `sglang` importable; `SGLANG_SKIP_P2P_CHECK=1` bypasses it |
| `_sgl_shims/cuda_wrapper.py` | `srt/distributed/device_communicators/cuda_wrapper.py` | ctypes cudart wrapper for the P2P self-test (import roots only) |
| `_sgl_shims/cuda_vmm_utils.py` | `srt/utils/cuda_vmm_utils.py` | VMM/fabric helpers: `is_vmm_pointer`, `VmmGraphInputManager`, `_gpu_fabric_clique` (import roots only; `cuda.bindings` + pynvml imports are try-guarded upstream and stay that way) |
| `_sgl_shims/parallel_state.py` | — | NEW shim: faithful copies of `in_the_same_node_as` (`srt/distributed/parallel_state.py`) and `make_shm_name` (`srt/utils/stale_shm_cleanup.py`), plus a minimal `get_world_group` stub (only `.local_rank`/`.barrier()`, consumed by `gpu_p2p_access_check`) |
| `_sgl_shims/environ.py` | — | NEW shim: the five `envs` fields the vendored CustomAllReduceV2 closure reads (`SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB`, the two `SGLANG_FORCE_..._SIZE_KB`, `SGLANG_MEMORY_SAVER_CUDA_GRAPH`, `SGLANG_CACHE_DIR`), same names/defaults as `srt/environ.py` |
| `_sgl_shims/srt_utils.py` | — | NEW shim: `is_cuda`/`is_hip`/`is_musa` (faithful bodies from `srt/utils/common.py`) and `log_info_on_rank0` (upstream's non-runtime-context fallback branch) |
| `_sgl_shims/tc_piecewise_cuda_graph.py` | — | NEW stub: `is_in_tc_piecewise_cuda_graph` always False (the tc_piecewise runner that sets the upstream flag never runs in this bench) |
| `_sgl_shims/configs/__init__.py` | — | NEW: vendoring glue for the mirrored `configs/` subpackage |
| `jit/csrc/gemm/dsv3_fused_a_gemm.cuh` | `kernels/jit/csrc/gemm/` | min-latency fused QKV-A GEMM, byte-identical (MLA bench addition, 2026-09-10) |
| `jit/csrc/elementwise/concat_mla.cuh` | `kernels/jit/csrc/elementwise/` | MLA absorbed-q / k concat kernels, byte-identical |
| `jit/csrc/elementwise/set_mla_kv_buffer.cuh` | `kernels/jit/csrc/elementwise/` | TMA bulk-store MLA latent-KV scatter, byte-identical |
| `jit/csrc/kimi_k3/mla_output_gate.cuh` | `kernels/jit/csrc/kimi_k3/` | K3 MLA output gate `x * sigmoid(gate)`, byte-identical |
| `ops/gemm/dsv3_fused_a_gemm.py` | `kernels/ops/gemm/` | JIT loader/wrapper for the fused-A GEMM (import-root rewrite only; `direct_register_custom_op` -> `_sgl_shims`) |
| `ops/attention/concat_mla.py` | `kernels/ops/attention/` | `concat_mla_absorb_q` / `concat_mla_k` loaders (import-root rewrite only) |
| `ops/kvcache/set_mla_kv_buffer.py` | `kernels/ops/kvcache/` | TMA `set_mla_kv_buffer` loader (import-root rewrite only) |
| `ops/kimi_k3/mla_output_gate.py` | `kernels/ops/kimi_k3/` | MLA output-gate loader (import-root rewrite only; `is_npu` -> `_sgl_shims`) |
| `jit/csrc/elementwise/set_mla_kv_concat_q.cuh` | `kernels/jit/csrc/elementwise/` | fused fp8 quantize + KV scatter + q concat (and bf16 variant), byte-identical |
| `ops/attention/set_mla_kv_concat_q.py` | `kernels/ops/attention/` | fused set-KV+concat-q loaders (`set_mla_kv_concat_q[_fp8]`, `covered[_fp8]`; import-root rewrite only) |
| `ops/attention/__init__.py`, `ops/kvcache/__init__.py` | — | NEW: vendoring glue |

2026-09-10 KDA-bench addition: `tiny_gemm`, `attn_res.py` and
`kda_decode_mtp.py` (rows marked above) were added for the KDA MTP1 decode
blocks in `kimi_k3/b10_kimi_k3_kda_layer.py`, closing the previous
`kimi_k3_tiny_gemm` lazy-import gap.

2026-09-10 MLA-bench addition: `dsv3_fused_a_gemm`, `concat_mla`,
`set_mla_kv_buffer` and `mla_output_gate` (rows marked above) were added for
the MLA MTP1 decode block in `kimi_k3/b10_kimi_k3_mla_layer.py`. The fp8
quantize helper around them (`mla_quantize_without_rope_for_fp8`,
`kernels/ops/attention/utils.py`) is NOT vendored — it is three aten
`.to(fp8)` casts plus `concat_mla_absorb_q`, replicated at the call site.
Still NOT vendored (outside the MoE/KDA/MLA scopes):
`ops/kimi_k3/attn_res_hip.py`, `sp_collective.py`.

## Runtime requirements

- `torch`, and for the JIT kernels: `apache-tvm-ffi`, `msgspec`, `ninja`, a
  CUDA >= 12.8 `nvcc` (SM100/SM103 targets).
- CUTLASS/CuTe headers are found through an installed `flashinfer` or
  `deep_gemm` package (`jit/utils/deps.py`); the deployment image
  `baseten/dynamo-cache-aware-routing:trtllm-...` ships flashinfer 0.6.18rc1,
  which satisfies this. The CuTe DSL GEMM additionally needs
  `nvidia-cutlass-dsl` (`import cutlass`, 4.x; also present in that image).
- The fused-AR kernels (`ops/kimi_k3/all_reduce.py`) require a
  CustomAllReduceV2-style communicator registered via `register_comm(...)`;
  the host object is `ops/communication/all_reduce.py`, but the workspace
  plumbing (IPC exchange, multicast binding, VMM) lives in sglang's
  `srt/distributed/device_communicators/custom_all_reduce_v2.py` and is NOT
  vendored — see `SglKimiK3MoEBlock` in
  `kimi_k3/b10_kimi_k3_moe_layer.py` for the graceful fallback.

None of these kernels ship in the `sgl-kernel` pip wheel: they are all
runtime-JIT-compiled from the sglang source tree (the AOT `sgl-kernel`
`kernels/aot/csrc` tree at this commit contains no kimi_k3/route_radix/
route_quant/ar_fusion sources, and its in-tree version 0.4.6.post1 is newer
than the newest published wheel 0.3.21 anyway).
