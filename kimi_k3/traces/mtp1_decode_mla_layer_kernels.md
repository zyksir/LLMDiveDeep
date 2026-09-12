# Kimi-K3 MLA layer — MTP1 decode kernel-sequence contract

Per-layer GPU kernel sequences for the **MLA (full-attention) layers** of
Kimi-K3 at the baseline operating point, extracted from the same two rank-0
decode traces as the KDA contract:

* **TRT-LLM**: `trt_rc19_decode_mtp1_bs1_rank0.trace.json.gz` — b10 fork
  `d9fa74fd94` (b10-1.3.0rc19), image `44d66979`, job
  `j20260910T202001_7ffe99`.
* **sglang**: `sglang_v0.5.18_decode_mtp1_bs1_rank0.trace.json.gz` —
  production image `a66596c1` (v0.5.18 base, source `6728f068`), job
  `j20260910T194750_6db34f`.

Workload: 35 input tokens, MTP1 (M = 2 tokens/step: 1 target + 1 draft
verify), concurrency 1, TP8, B200/GB300-class SM100. Sequences below are the
**modal** per-layer sequence from `extract_layer_kernels.py`
(`extracted_trt_rc19.txt`: 25 MLA slices, modal ×14;
`extracted_sglang.txt`: 24 MLA slices, modal ×23); raw outputs sit next to
this file. Layer boundaries are rebased at the attention-side AttnRes kernel,
so positions 1..N cover attention + the following MoE block.

MLA layer placement (checkpoint `config.json`, `/node-storage/var/
kimi-k3-config`): `linear_attn_config.full_attn_layers =
[4, 8, ..., 88, 92, 93]` (1-based) — every 4th layer plus the last two;
windows of four are `[KDA, KDA, KDA, MLA]` with MLA fourth.

Shard math at TP8 (config.json): 96 q-heads global → 12 local;
`q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`,
`qk_rope_head_dim=64`, `v_head_dim=128`; fused-A width = 1536+512+64 = 2112;
absorbed q head dim = 512+64 = 576. K3 MLA is **NoPE**
(`mla_use_nope=true`): sglang passes `skip_rope=True`, TRT builds an
identity cos/sin table (`use_rope=False`) — the TRT rope kernel still runs
(cache assign), sglang has no rope kernel at all. `mla_use_output_gate=true`
adds the g_proj gate GEMM and the `x * sigmoid(gate)` epilogue.

## TRT-LLM rc19 — modal MLA layer (34 kernels; attention = 1–19)

| # | kernel | med | attribution |
|---|--------|-----|-------------|
| 1 | `kimi_k3::sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168,4,...,true,true,...>` | 6.77 us | attention-side AttnRes (prefix + bank aggregation + delta fold + input RMSNorm) |
| 2 | `dsv3MinLatencyKernels::fused_a_gemm_kernel<1,2112,7168,16,8,256,16>` | 10.82 us | fused [q_a 1536 \| kv_a 512 \| k_rope 64] down-proj (`torch.ops.trtllm.dsv3_fused_a_gemm_op`) |
| 3 | `nvjet_sm103_tst_64x8..._splitK_TNT` (+ #8 `cublasLt::splitKreduce`) | 12.54 us | g_proj output-gate GEMM 7168→1536 (issued on the AttentionOutputGate aux stream, hence early launch slot)* |
| 4 | `flashinfer...rmsnormRMSNormKernel` | 8.43 us | q_a RMSNorm (1536) |
| 5 | `flashinfer...rmsnormRMSNormKernel` | 2.86 us | kv_a RMSNorm (512) |
| 6 | `nvjet_sm103_tst_16x64..._TNN` | 7.47 us | q_b up-proj 1536→2304 (12×192)* |
| 7 | `at::native::CatArrayBatchedCopy<OpaqueType<2u>,...>` | 5.63 us | `latent_cache = cat([compressed_kv, k_pe])` |
| 8 | `cublasLt::splitKreduce_kernel` | 2.59 us | splitK reduce paired with #3 (launch-order jitter swaps it in variants) |
| 9 | `nvjet_sm103_tst_64x8..._TNT` | 3.65 us | q_nope absorb BMM `q_nope @ k_b_proj_trans` [12,2,128]×[12,128,512] → written into `fused_q[..., :512]`* |
| 10 | `applyMLARopeAndAssignQKVKernelGeneration<bf16,256,512,64,KVBlockArray>` | 3.46 us | `torch.ops.trtllm.mla_rope_generation`: identity rope on q_pe/k_pe, builds `fused_q[..., 512:]`, appends latent row to the paged bf16 KV pool |
| 11 | `at::native::...FillFunctor<int>` | 1.09 us | fmha bookkeeping (workspace/semaphore fill around the flashinfer plan) |
| 12 | `memcpy32_post` | 1.73 us | fmha bookkeeping (block-table/seq-len staging) |
| 13 | `kernel_cutlass_split_kv_kernel_flashinfercute_dslattentionmonolithicmla` | 6.16 us | FlashInfer CuTeDSL monolithic MLA decode, split-KV pass (`flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl", cute_dsl_impl="monolithic")`, bf16 KV, HQk 576 / HV 512) |
| 14 | `kernel_cutlass_reduction_kernel_flashinfercute_dslattentionmonolithicmla` | 2.37 us | CuTeDSL MLA reduction pass |
| 15 | `nvjet_sm103_tst_16x64..._TNN` | 3.39 us | v absorb BMM `attn_latent @ v_b_proj` [12,2,512]×[12,512,128] |
| 16 | `at::native::...sigmoid_kernel_cuda` | 2.02 us | output gate `sigmoid(gate)` (unfused aten pair — TRT has no fused gate kernel) |
| 17 | `at::native::...MulFunctor<bf16>` | 1.38 us | output gate multiply |
| 18 | `nvjet_sm103_tst_64x8..._TNT` | 6.05 us | o_proj 1536→7168 |
| 19 | `ar_fusion::allreduce_fusion_kernel_oneshot_lamport<Pattern 0>` | 7.55 us | plain oneshot-lamport AR of the o_proj partial (residual deferred into the next AttnRes delta) |
| 20–34 | — | — | MoE block: identical to KDA-layer positions 9–23 (mlp-side AttnRes, gate+fc1 GEMMs, route, MXFP4 SiTU experts, fused finalize-AR, fc2) — see the KDA contract in `extracted_trt_rc19.txt` |

\* Positions 3/6/9 attribution among {g_proj, q_b, q-absorb-BMM} is inferred
from weight-byte/bandwidth arithmetic (22 MB / 7.1 MB / 1.5 MB) and the
multi-stream launch behavior — the three GEMMs interleave across the gate aux
stream and the ln/rope aux stream, and the observed variants swap #4/#5 and
the splitK-reduce slot. Kernel identity is unambiguous; the slot-to-op map
for these three carries that caveat.

Observed variants (6): flashinfer RMSNorm and splitKreduce launch-slot swaps
only (multi-stream jitter); one 174-kernel step-boundary slice carries
sampler/MTP-prep kernels between layers.

## sglang v0.5.18 — modal MLA layer (25 kernels; attention = 1–13)

| # | kernel | med | attribution |
|---|--------|-----|-------------|
| 1 | `sglang::attn_res_fused_tma_kernel<KimiK3AttnResTrait<7168,1,4,200>>` | 5.60 us | attention-side AttnRes (TMA fused aggregation + input RMSNorm) |
| 2 | `sglang::fused_a_gemm_kernel<1,2112,7168,16,8,256,16>` | 8.35 us | fused [q_a \| kv_a \| k_rope] down-proj (JIT `dsv3_fused_a_gemm`) |
| 3 | `kernel_cutlass_kernel_TgvGemmCuteExtKernel` | 11.68 us | g_proj output-gate GEMM 7168→1536, precomputed on the gate alt stream (`KimiK3MLAAttention._precompute_output_gate`) |
| 4 | `flashinfer...rmsnormRMSNormKernel` | 2.69 us | q_a RMSNorm (1536, main stream) |
| 5 | `flashinfer...rmsnormRMSNormKernel` | 1.60 us | kv_a RMSNorm (512, alt stream under capture) |
| 6 | `kernel_cutlass_kernel_TgvGemmCuteExtKernel` | 4.38 us | q_b up-proj 1536→2304 (TGV CuTe DSL GEMM) |
| 7 | `nvjet_sm103_tst_64x8..._TNT` | 3.74 us | q_nope absorb BMM `torch.bmm(q_nope.T, w_kc)` [12,2,128]×[12,128,512] |
| 8 | `set_mla_kv_concat_q_fp8_triton_kernel` | 2.27 us | ONE fused launch: bf16→fp8 quantize of k_nope/k_rope + scatter into the paged fp8 KV pool at the step's cache locs + fp8 [q_nope\|q_pe] 576-dim concat (`_set_kv_and_concat_q_fp8_fused`) |
| 9 | `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen` | 10.14 us | trtllm-gen fp8 MLA decode (`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, page 64, bmm1_scale = 1/√192) |
| 10 | `nvjet_sm103_tst_16x64..._TNN` | 3.55 us | v absorb BMM [12,2,512]×[12,512,128] |
| 11 | `sglang::mla_output_gate_kernel<256,true>` | 2.08 us | fused `x * sigmoid(gate)` |
| 12 | `kernel_cutlass_kernel_TgvGemmCuteExtKernel` | 6.46 us | o_proj 1536→7168 (TGV) |
| 13 | `sglang::all_reduce_push_res_kernel<8u,true,true>` | 6.88 us | fused 1shot push AR + residual (prefix) fold |
| 14 | `sglang::attn_res_fused_tma_kernel` | 6.75 us | mlp-side AttnRes (post-attention aggregation + norm) |
| 15–25 | — | — | MoE block: identical to KDA-layer positions 9–19 (front TGV, fused route+quant, SiTU shared experts, NVLS pull AR, MXFP4 experts, push-norm AR, gemm_ag + spin_add3) — see `extracted_sglang.txt` |

Observed variants (2): a routing-kernel launch-slot swap; one 153-kernel
step-boundary slice (sampler/draft prep) whose embedded MLA layer runs the
**unfused fallback** chain (`concat_mla_absorb_q` + three aten fp8 casts +
`set_mla_kv_buffer_kernel`) and the CuTeDSL MLA backend — that is the MTP
draft-model layer, not a target-model layer.

## Cross-stack differences worth knowing

* **fmha engines are swapped vs. what the names suggest**: TRT rc19 runs
  FlashInfer's **CuTeDSL monolithic MLA** on a **bf16** KV cache (2 kernels,
  8.5 us); sglang runs FlashInfer's **trtllm-gen** fp8 MLA (1 kernel,
  10.1 us) on an **fp8** KV cache.
* **KV write**: TRT folds the cache append into its rope/assign kernel
  (#10); sglang folds it into the fused quantize+concat kernel (#8).
* **Output gate**: sglang has a fused gate kernel (#11); TRT spends two aten
  launches (#16–17).
* **AR tail**: sglang's push AR folds the residual add (#13); TRT's Pattern-0
  lamport AR leaves the residual to the next layer's AttnRes delta.

## Bench-block deviations (named)

The blocks in `kimi_k3/b10_kimi_k3_mla_layer.py` reproduce the sequences
above with these documented exceptions:

1. **world == 1**: no AR kernel exists — TRT #19 and sglang #13 disappear;
   the sglang block replaces the fused AR+fold with one explicit aten add
   into the persistent `next_prefix` buffer (same convention as the KDA
   blocks).
2. **sglang #8 kernel name**: the vendored sglang commit (`f8cbf000f4`)
   replaced the baseline's Triton kernel with the CUDA JIT
   `sglang::set_mla_kv_concat_q_fp8` — one launch either way, identical
   fusion boundary.
3. **TRT #11–12** (fill + memcpy32) are runtime bookkeeping owned by the
   flashinfer wrapper/plan around the fmha; the bench issues the same
   flashinfer call and inherits whatever bookkeeping the installed wheel
   emits rather than pinning those two launches.
4. Both blocks replay a **fixed decode step** (35 cached tokens + 2 verify
   tokens at static cache slots), so KV writes land on the same rows every
   iteration — kernel identity and traffic match the baseline step;
   the cache **contents** are synthetic.
5. **mlp-side AttnRes** (TRT #20, sglang #14) belongs to the layer pair
   wiring, not the attention block — it is issued by the 4-layer mini
   (`b10_kimi_k3_mini.py`), matching production decoder-layer wiring.
