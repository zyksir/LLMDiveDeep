# MegaMoE destination-rank deduplication report

Date: 2026-08-26

Qualification: **REJECTED — correct, but no material large-batch speedup**

## Operation contract

- Meaning: EP8 fused expert dispatch, FC1, SwiGLU, FC2, weighted combine, and
  return to each token's source rank.
- Inputs: per-rank FP8 E4M3 activations with packed UE8M0 scales, global
  top-16 expert IDs, FP32 routing weights, and local MXFP4 expert weights.
- Shape: 896 experts, top-k 16, hidden 7168, intermediate 3072, EP8.
- Output: one BF16 `[local_tokens, 7168]` tensor per rank.
- Timed boundary: unchanged `deep_gemm.fp8_fp4_mega_moe`; input staging,
  routing construction, allocation, and weight transformation are excluded.
- Proposed change: transfer each activation once per `(source token,
  destination rank)`, then fan it out locally to all selected experts on that
  rank. Expert math, routing weights, and output writes must remain unchanged.

## Ecosystem implementation survey

- DeepGEMM 2.6.1, commit `559d79f`: fused MegaMoE dispatch indexes and pulls
  activations per `(token, top-k slot)`, not per destination rank.
- TensorRT-LLM commit `d9fa74f`: `MegaMoEDeepGemm` calls the unchanged
  DeepGEMM kernel. Its external `moeA2ADispatchKernel` is the surveyed
  implementation that does destination-rank deduplication using an
  `already_copied` rank bitmask.
- SGLang and vLLM MegaMoE paths call DeepGEMM and inherit its per-slot pull.
- FlashInfer's fused CuteDSL MegaMoE port also retains per-slot dispatch. Its
  available public fused kernels do not provide the target FP8-activation,
  MXFP4-weight contract as an independent performance baseline.

The unchanged TRT-LLM external A2A pipeline is useful migration data, but it is
not a same-boundary kernel baseline because dispatch, expert execution, and
combine are separate launches with a different intermediate ABI.

## Pre-candidate benchmark and speed-of-light check

Command:

```bash
mpirun --allow-run-as-root -n 8 python3 \
  kernels/research/megamoe/bench_rank_dedup.py \
  --global-batches 8,256,2048,16384 \
  --warmup 10 --iters 50 \
  --output out/megamoe_rank_locality_baseline_full.json
```

`spread` sends every token to all eight ranks, with two experts per rank.
`single_rank` sends all 16 expert assignments for a token to one rank. Tokens
rotate destination ranks so aggregate expert work remains balanced.

| Global batch | Spread, 8 ranks (us) | Single rank (us) | Single-rank change |
|---:|---:|---:|---:|
| 8 | 141.6 | 134.1 | -5.3% |
| 256 | 630.1 | 629.6 | -0.1% |
| 2,048 | 708.4 | 713.7 | +0.7% |
| 16,384 | 1,983.2 | 2,141.3 | +8.0% |

All outputs were finite. Cold setup was 5.19 seconds; the complete cached
benchmark process took 20.1 seconds.

For the proxy width, dispatch payload is 7,392 bytes per expert assignment:
7,168 FP8 bytes plus 224 scale bytes. At global batch 16,384, spread routing
nominally pulls about 212 MB of remote dispatch payload per rank. Rank dedup
would reduce that to about 106 MB. At an assumed 900 GB/s one-direction
NVLink bandwidth, the maximum non-overlapped saving is approximately 118 us,
only 6% of the measured 1,983 us, before charging the required local fanout.

At global batch 256, the corresponding maximum is under 2 us versus a 630 us
kernel. The approximately 3.9 GB local expert-weight footprint has a roughly
490 us lower bound at 8 TB/s HBM bandwidth when all local experts participate,
which better explains the intermediate-batch regime.

Decision: **PROCEED with a bounded high-batch ablation.** This locality test
does not reduce the unchanged kernel's explicit per-expert pulls; it only tests
whether peer/L2 caching already reuses repeated addresses. Its flat or negative
result rules out assuming automatic hardware deduplication, but does not measure
the proposed explicit cache. The 118 us upper bound at global batch 16,384 is
large enough to test, while the under-2-us bound at batch 256 predicts no win.

## Idea and candidate

The candidate would add a rank-local activation cache indexed by source rank
and source token. Dispatch warps would use the same destination-rank bitmask
idea as TRT-LLM's one-sided A2A kernel, push one FP8 row and scale row to each
destination cache, synchronize, and make every expert occurrence copy from
that local cache into the existing expert-major ring.

The smallest candidate deduplicates only repeated remote-rank routes. Local and
singleton routes retain the unchanged pull path. An owner occurrence fills the
cache and publishes a release flag; follower occurrences acquire that flag and
fan out locally. Routing weights, per-expert metadata, GEMMs, and combine remain
per top-k slot.

This preserves math but adds:

- a second local read/write path before FC1;
- a global synchronization dependency before local fanout;
- approximately 106 MiB of FP8/SF cache per rank at 2,048 maximum local tokens,
  or about 218 MiB at DeepGEMM's 4,224-token aligned capacity;
- extra metadata or fixed sparse cache slots;
- no reduction to per-expert ring materialization or combine traffic.

The candidate is an on/off JIT tactic controlled by
`DG_MEGA_MOE_RANK_DEDUP`. A runtime threshold
`DG_MEGA_MOE_RANK_DEDUP_MIN_TOKENS` keeps the original compiled pull path for
smaller workloads.

## Candidate results

The first exploratory sweep at hidden 3584 showed a non-monotonic 4.6% spread
win at 4,096 tokens/rank followed by a 4.3% regression at 8,192. Because the
confirmed proxy contract uses hidden 7168, the decision uses a new EP8 H7168
sweep:

| Global batch | Tokens/rank | Baseline spread (us) | Dedup spread (us) | Change |
|---:|---:|---:|---:|---:|
| 32,768 | 4,096 | 4,014.4 | 4,017.2 | +0.07% |
| 65,536 | 8,192 | 7,723.2 | 7,697.6 | -0.33% |

The stress routing that sends all 16 experts to one destination rank was also
within noise: -0.30% at 4,096 tokens/rank and +0.07% at 8,192. These confirmed
numbers use 10 warmups and 100 timed iterations with maximum-rank CUDA-event
latency.

H7168 correctness passed bit-exact comparison against the dedup-disabled
kernel for both spread and single-rank routing on every EP8 rank. The dedicated
cache is projected to add about 477 MiB/rank at 8,192 tokens/rank. The rejected
candidate is preserved as
`kernels/research/megamoe/deepgemm_rank_dedup.patch`; the pinned DeepGEMM
checkout and TensorRT-LLM remain unmodified.

Decision: **REJECT the candidate.** At the user-prioritized large batches, the
confirmed spread result ranges from a 0.07% regression to a 0.33% improvement,
which is not material and does not justify the cache memory, owner/follower
synchronization, or added implementation risk.

## Baselines and provenance

- Fused kernel: DeepSeek DeepGEMM 2.6.1, commit `559d79f`, MIT license,
  `sm100_fp8_fp4_mega_moe_impl` in
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`.
- Dedup idea source: TensorRT-LLM commit `d9fa74f`, Apache-2.0,
  `moeA2ADispatchKernel` in
  `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`.
- Hardware: eight SM100-family GPUs in the existing `trt-mega` benchmark
  environment.
- Launch mode: eager, synchronized CUDA-event timing, maximum rank latency.

## Results and limitations

- Result status: **REJECTED** after direct candidate comparison.
- NVLink traffic is algorithmically counted, not measured with fabric
  counters. Destination-rank locality is a hardware-reuse probe, not the final
  candidate ablation.
- The public DeepGEMM build uses SwiGLU. This report covers the agreed proxy
  operation, not Kimi-K3 SiTU integration.
- The full ecosystem does not provide another independently implemented,
  same-quantization fused kernel that can serve as a same-contract performance
  baseline. This would prevent a `QUALIFIED` positive speedup claim.

## Alternatives and next optimization target

- Per-rank cache: rejected; it is correct but neutral at H7168 large batches.
- Opportunistic duplicate pull reuse in L2: no code change; the flat locality
  result is consistent with communication being hidden or cached, but no
  fabric-counter claim is made.
- More promising target: the intermediate-batch expert scheduler and
  expert-weight movement. Global batches 256 through 2,048 are close to a
  weight-bandwidth/fragmentation regime, while the external TRTLLMGen pipeline
  previously matched or beat MegaMoE there.

## Integration

No TensorRT-LLM, layer, model, or deployment integration was performed because
the standalone candidate did not pass the latency gate.
