# Kimi-K3 MTP1 decode — per-layer KDA kernel contract (2026-09-10 baselines)

Successor of `kimi_k3_layer/traces/mtp1_decode_layer_kernels.md` (lost with the
untracked `kimi_k3_layer/` tree in the 2026-09-10 reorganization). Extracted
with `extract_layer_kernels.py` from the updated baselines in this directory
(see `decode_mtp1_bs1_rank0.provenance.md`): middle profiled step, per-layer
slices rebased to the attention-side AttnRes, modal sequence over the step's
68 KDA layers. Durations are per-position medians (µs), 1 request, M=2
tokens, TP8 rank 0.

**Baseline change vs the previous contract:** the TRT baseline is now the
NATIVE b10-1.3.0rc19 server (`12e51f84`, image `44d66979`, run `trt_k3_02`) —
NOT the fork branch `yikai/k3-decode-opt` @ `f44b703966` with
`TLLM_K3_KDA_SGL_REPLAY=1` that `baseline_trt_mtp1_decode.trace.json.gz`
captured. The sglang baseline moved from the earlier capture to production
image `a66596c1` (v0.5.18 base, source `6728f068`) — first as
`…_baseline_…` (run `sgl_k3_02`, no overlay), then replaced 2026-09-10T23:52Z
by the MLA-overlay capture (run `sgl_mla_overlay_01`,
`mla-target-verify.patch`, KDA/MoE kernels identical, see below).

## SGLang KDA layer — 19 kernels (modal 66/68 layers in the overlay trace)

Positions 1–7 are the attention section (the mini blocks' 7-position
contract); 8–19 are the MoE section (see
`mtp1_decode_moe_layer_kernels.md`).

| # | median µs | kernel |
|---|---|---|
| 1 | 5.7 | `sglang::attn_res_fused_tma_kernel<KimiK3AttnResTrait<7168,B,S,200>,1>` (bank aggregation + prefix + input norm; B,S vary per layer) |
| 2 | 17.1 | `TgvGemmCuteExtKernel` (fused `[q\|k\|v\|g]` in-proj, main stream) |
| 3 | 7.4 | `sglang::tiny_n_gemm_kernel<2,144,7168,1,bf16,true>` (`[f_a\|beta]`, aux stream) |
| 4 | 4.7 | `sglang::tiny_k_gemm_kernel<2,1536,128,12,bf16,true>` (`f_b`, aux stream) |
| 5 | 11.3 | `kernel_cutlass_kda_decode_mtp_kernel…` (fused conv update + gate + delta rule + ReplaySSM ring writes + gated RMSNorm) |
| 6 | 7.7 | `TgvGemmCuteExtKernel` (o_proj) |
| 7 | 7.6 | `sglang::all_reduce_push_res_kernel<8,true,true>` (CustomAllReduceV2 1shot push AR + residual fold) |

Unchanged in structure from the previous sglang contract (same 7 positions,
same kernels). Notes:

* **Kernel-name drift `6728f068` → vendored `f8cbf000`:** the tiny-GEMM JIT
  kernels were refactored from explicit template params
  (`tiny_n_gemm_kernel<2u,144u,7168u,1u,bf16,true>`, this baseline) to
  GEMMTrait structs (`tiny_n_gemm_kernel<GEMMTraitN<144u,7168u,1u,32u>,2u,…>`,
  what `kernels/sgl_copied_kernels` compiles). Same kernels/launch shapes,
  different mangled names — treat as equal at these positions.
* The remaining `6728f068`→`f8cbf000` diffs in the K3 decode path
  (`kimi_k3.py`, `kda_backend.py`, `k3_ar_fusion.py`, `attn_residual.py`)
  are HIP/aiter fused decode, chain-verify/accept-state fusions (opt-in,
  post-6728f068), EP-a2a backends, comm-registration API and SM12x gating —
  none change the NVIDIA TP8 decode kernel sequence above.
* The MLA overlay (`mla-target-verify.patch`,
  `SGLANG_TRTLLM_MLA_FUSED_VERIFY_KV_CONCAT_Q=1`) does not touch KDA layers:
  KDA modal sequences of `…_baseline_…` and the overlay trace are identical.

## TRT (native rc19) KDA layer — 8 kernels + MoE section

The native rc19 KDA decode path is NOT the fork's `_sgl_replay_verify`
(sglang-vendored kernels) the previous TRT contract documented. New attention
section:

| # | median µs | kernel |
|---|---|---|
| 1 | 6.8 | `…kimi_k3::sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168,4,false,true,true,true,true,false>` |
| 2 | 19.9 | `nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT` (fused in-proj incl. `[f_a\|beta]`, cuBLASLt) |
| 3 | 3.4 | `cublasLt::splitKreduce_kernel` (bf16, pairs with #2) |
| 4 | 3.3 | `nvjet_sm103_tst_16x64_64x16_4x1_v_bz_TNN` (`f_b` projection) |
| 5 | 1.6 | `at::native::vectorized_elementwise_kernel<…CUDAFunctorOnSelf_add<long>…>` (slot/position bookkeeping) |
| 6 | 9.9 | `kernel_cutlass__kda_replay_ssm_conv_gated_wychunk_kernel…` (fused KDA verify, rc19-native CuTe kernel) |
| 7 | 5.8 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` (o_proj) |
| 8 | 7.5 | `ar_fusion::allreduce_fusion_kernel_oneshot_lamport<Pattern 0,…>` (plain AR; residual deferred into the next AttnRes delta) |

Changes vs the old fork baseline (the sequence `TrtKimiK3KdaBlock`
reproduces): in-proj moved from the SGLang TGV CuTe GEMM to a cuBLASLt nvjet
splitK pair; the `[f_a|beta]`/`f_b` tiny GEMVs are replaced by the fused
in-proj plus one nvjet TNN; the fused verify kernel is
`kda_replay_ssm_conv_gated_wychunk` (rc19-native), not the sglang-replay
`kda_decode_mtp` dspark kernel. Positions 1 (AttnRes), 7 (o_proj nvjet) and
8 (oneshot-lamport pattern-0 AR) are unchanged.

## MLA layers (for completeness; 24–25 per step)

sglang overlay: 25 kernels — `attn_res_fused_tma`, `fused_a_gemm` (q_a|kv_a),
TGV, 2× flashinfer RMSNorm, TGV, nvjet TNT,
`set_mla_kv_concat_q_fp8_triton_kernel` (the overlay's single fused
KV-quant + KV-scatter + Q-concat launch; the `…_baseline_…` trace instead has
`concat_mla_absorb_q` + 3× fp8-convert elementwise + `set_mla_kv_buffer`, 29
kernels total), `fmhaSm100fKernel…`, nvjet TNN, `mla_output_gate`, TGV
(o_proj), `all_reduce_push_res`, then the MoE section.
TRT rc19: 34 kernels — `attn_res_fwd_online_v2`, `dsv3MinLatencyKernels::
fused_a_gemm`, nvjet splitK + 2× flashinfer RMSNorm, nvjet TNN, CatArray,
splitKreduce, nvjet TNT, `applyMLARopeAndAssignQKVKernelGeneration`, fill +
`memcpy32_post`, flashinfer CuTe MLA decode split_kv + reduction, nvjet TNN,
sigmoid + mul elementwise (output gate), nvjet TNT (o_proj), pattern-0 AR,
then the MoE section.

## Mini-trace verification (mini_kimi_k3_kda.trace.json.gz, 2026-09-10 12:49)

24 reps per span, steady-state pattern read from the GPU kernel stream
(CUDA-graph replays carry no per-kernel launch correlation):

* `SglKimiK3Kda` — ALL 7 positions match the updated sglang baseline
  (`attn_res_fused_tma` → TGV in-proj → tiny_n → tiny_k → `kda_decode_mtp` →
  TGV o_proj → `all_reduce_push_res<8,true,true>`), modulo the documented
  pool-shape suffix on the CuTe kernels and the tiny-GEMM template-name
  drift above. ALIGNED.
* `TrtKimiK3Kda` — positions 1, 6, 7 match (`attn_res_fwd_online_v2`,
  o_proj `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT`, pattern-0
  oneshot-lamport AR). Positions 2–5 MISMATCH the native-rc19 baseline: the
  mini runs the fork's sglang-replay middle (TGV in-proj + tiny GEMVs +
  `kda_decode_mtp`), the new baseline runs nvjet splitK in-proj + nvjet TNN
  `f_b` + `kda_replay_ssm_conv_gated_wychunk`.
* **Realignment (2026-09-11, code-only):** `TrtKimiK3KdaBlock.decode()` now
  DEFAULTS to the native-rc19 sequence — one fused in-proj GEMM over the full
  `in_proj_qkvgfab` weight incl. the `[f_a|beta]`/pad rows (positions 2–3),
  plain `f_b_proj` GEMM (4), the graph-stable Philox `seed.add_(1)` int64
  elementwise add (5, FP16 checkpoint pool with stochastic rounding as in
  the baseline server), and the rc19-native
  `kda_replay_ssm_conv_gated_wychunk` fused verify (6), mirroring
  `kda_mixer.py`'s `use_cute_mtp_replay` branch at `12e51f84`. The fork
  middle is preserved behind `TrtKimiK3KdaBlock(sgl_replay=True)` (or env
  `TLLM_K3_KDA_SGL_REPLAY=1`, mirroring the fork switch); default is native.
  The mini trace above PREDATES this realignment; re-emit (deferred until
  GPUs are free) with the standard KDA bench command:
  `mpirun --allow-run-as-root -np 8 python3
  kimi_k3/bench_b10_kimi_k3_kda_layer.py --backends trt_block sgl_block
  --trace-out kimi_k3/traces/mini_kimi_k3_kda.trace.json.gz`.
