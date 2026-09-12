# Kimi-K3 MTP1 decode — per-layer MoE kernel contract (2026-09-10 baselines)

Successor of `kimi_k3_layer/traces/mtp1_decode_moe_layer_kernels.md` (lost
with the untracked `kimi_k3_layer/` tree in the 2026-09-10 reorganization).
Extraction identical to `mtp1_decode_layer_kernels.md`; the MoE section is
byte-identical between KDA and MLA layers of each baseline. 1 request, M=2
tokens, TP8 rank 0.

**Baseline change:** the TRT baseline is now the NATIVE b10-1.3.0rc19 server
(`12e51f84`, run `trt_k3_02`) instead of the fork branch — this is exactly
the revision `TrtKimiK3MoEBlock` pins, so the block's previous 4 documented
mismatches against the fork baseline are expected to close (see
verification below). The sglang baseline is production image `a66596c1`
(source `6728f068`); its MoE section is identical between the `…_baseline_…`
and MLA-overlay captures.

## TRT native rc19 — MoE section, 15 kernels (modal 61/68 KDA layers)

| # | median µs | kernel | role |
|---|---|---|---|
| 1 | 6.2 | `attn_res_fwd_online_v2_kernel<7168,4,false,true,false,true,true,false>` | pre-MoE AttnRes (deferred-residual delta + input norm) |
| 2 | 10.2 | `nvjet_sm103_tss_32x64_64x16_4x1_v_bz_splitK_TNN` | router gate GEMM, fp32 out |
| 3 | 10.1 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | fc1_latent_proj 7168→3584 (interleaves with shared gate_up on the aux stream) |
| 4 | 3.0 | `cublasLt::splitKreduce_kernel` (fp32) | pairs with the router GEMM |
| 5 | 11.1 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | shared gate_up |
| 6 | 3.8 | `cublasLt::splitKreduce_kernel` (bf16) | |
| 7 | 2.1 | `_situ_and_mul_kernel` | shared SiTU activation |
| 8 | 5.2 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` | shared down proj |
| 9 | 3.7 | `cublasLt::splitKreduce_kernel` (bf16) | (positions 3–9 jitter across streams; variant ×6 swaps one splitKreduce) |
| 10 | 2.7 | `tensorrt_llm::kernels::quantize_with_block_size<Type 2, bf16, 32,…>` | MXFP8 quant of the routed latent |
| 11 | 7.2 | `moe::dev::routing::routingCustom::routingIndicesSmallBsKernel<KernelParams<float, bf16, 1024, 16,…>>` | trtllm-gen routing from RAW fp32 logits (in-kernel top-16) |
| 12 | 12.7 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32…_t128x8x512_s3_et128x8_m128x8x32_c1x1x1…siTuGlu_dynB_sm100f` | experts fc1 + SiTU |
| 13 | 9.3 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32…_t128x8x512_s3_et128x8_m128x8x32_c1x1x1…dynB_sm100f` | experts fc2 (do_finalize=False) |
| 14 | 17.0 | `ar_fusion::moe::moefinalize_allreduce_fusion_kernel_oneshot_lamport<bf16,8,false,true,false,bf16,true>` | rc19 fused finalize + AR + RMSNorm + concat |
| 15 | 24.7 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` | full fc2_latent_proj 3584→7168; residual add deferred into the NEXT layer's AttnRes |

Changes vs the old fork baseline (the 4 mismatches previously documented for
`TrtKimiK3MoE`):

1. **Fused route_quant front — GONE.** The native server runs the unfused
   rc19 front: separate router GEMM, fc1 GEMM, `quantize_with_block_size`,
   `routingIndicesSmallBsKernel` from raw logits. The fork's sglang-style
   `route_quant_fused` does not appear.
2. **Finalize kernel variant — resolved to rc19.** The decode tail is the
   rc19 `moefinalize_allreduce_fusion_kernel_oneshot_lamport` epilogue,
   exactly `TrtKimiK3MoEBlock`'s `experts(do_finalize=False)` → fused
   finalize+AR+rmsnorm+concat → full fc2 path.
3. **Deferred residual add — matches rc19 semantics.** Plain pattern-0 AR on
   the attention side and residual deferral into the next layer's AttnRes
   delta; no sglang push-AR residual fold anywhere in the TRT trace.
4. **Expert GEMM tactic — now the rc19 pick**:
   `t128x8x512_s3_et128x8_m128x8x32_c1x1x1` for both bmm's (the fork/sglang
   baseline ran `t128x16x256_{s6,s5}_et128x16_m256x16x32_c2x1x1`). Whether
   the mini reproduces it depends on the runtime autotuner: confirm on
   regeneration.

## SGLang (6728f068) — MoE section, 12 kernels (modal, both captures)

| # | median µs | kernel | role |
|---|---|---|---|
| 1 | 6.5 | `sglang::attn_res_fused_tma_kernel` | pre-MoE AttnRes |
| 2 | 15.9 | `TgvGemmCuteExtKernel` | FUSED front: one merged `[H, shared gate_up \| router E \| fc1 latent]` GEMM (`_forward_fused`) |
| 3 | 3.6 | `sglang::route_quant_fused_kernel<true,float,float>` | noaux_tc top-16 + trtllm id pack + MXFP8 latent quant, ONE launch |
| 4 | 2.3 | `sglang::situ_and_mul_kernel<float,bf16,true,true>` | shared SiTU (aux stream) |
| 5 | 4.7 | `TgvGemmCuteExtKernel` | shared down proj |
| 6 | 4.4 | `routingCustom::routingIndicesBlockKernel<KernelParams<bf16, bf16, 896, 16,…>>` | trtllm-gen routing expansion of the PRE-PACKED ids (fast path; contrast the TRT float-logits variant) |
| 7 | 11.0 | `sglang::all_reduce_pull_res_kernel<8,false,true>` | low-SM NVLS pull AR of the shared partial |
| 8 | 13.1 | `bmm_MxE4m3…_t128x16x256_s6_et128x16_m256x16x32_c2x1x1…siTuGlu…` | experts fc1 |
| 9 | 9.4 | `bmm_Bfloat16…_t128x16x256_{s5,s6}_et128x16_m256x16x32_c2x1x1…` | experts fc2, do_finalize=False (stage pick s5 vs s6 varies BETWEEN server sessions — autotune, not code) |
| 10 | 8.1 | `sglang::all_reduce_push_norm_cluster_kernel<8,7,true,true>` | deferred finalize folded into 1shot push AR + RMSNorm |
| 11 | 6.9 | `sglang::gemm_ag::gemm_ag_gemv_kernel<3584,7168,2,2,true>` | fc2_latent up-proj as the gemm_ag producer |
| 12 | 2.3 | `sglang::gemm_ag::spin_add3_kernel<7168,true,true>` | consumer: shared + routed + residual `_add3` |

Both user key checks hold in the updated baseline: `route_quant_fused` is
present (position 3) and the fused AR tails are present
(`all_reduce_push_norm_cluster` position 10 plus the KDA/MLA attention
`all_reduce_push_res`). The `6728f068`→`f8cbf000` source diff contains no
kernel change for this section.

## Mini-trace verification

No MoE mini trace exists on disk (only `mini_kimi_k3_kda.trace.json.gz`
survived the reorganization; no `mini_kimi_k3_moe.trace.json.gz` anywhere).
Span-level verification therefore awaits regeneration. Structural
(code-path) expectations against the new contracts:

* `TrtKimiK3MoE` — `TrtKimiK3MoEBlock` pins exactly this native-rc19 revision
  and forward path; mismatches 1–3 should disappear structurally, and
  mismatch 4 (tactic) should match if the mini's autotuner picks the same
  `t128x8x512_s3` cubins — verify on regeneration.
* `SglKimiK3MoE` — expected REMAINING deviations (all pre-documented in
  `b10_kimi_k3_moe_layer.py`): unfused front (separate gate GEMM + fp32 cast
  + fc1 GEMM instead of the merged TGV front, positions 2), shared-expert AR
  via push/Collectives instead of `all_reduce_pull_res` (position 7), and
  the `torch.matmul + add` tail instead of `gemm_ag_gemv + spin_add3`
  (positions 11–12) unless the vendored gemm_ag path is wired up. Positions
  3 (route_quant_fused), 6 (routing expansion), 8–9 (bmm's) and 10
  (push_norm; the vendored `ar_fusion.cuh` compiles the same
  `all_reduce_push_norm_cluster_kernel`) should align when
  `attach_sgl_ar` is active.
* `TrtRc25KimiK3MoE` — no serving baseline exists for rc25; contract remains
  the code-replication documented in its class docstring.
