# Kimi-K3 MegaMoE practice report (fake expert weights)

Date: 2026-08-19
Qualification: applicability findings QUALIFIED (source-traced); latency
numbers UNQUALIFIED as absolutes (autotuner not warmed, practice shape) —
valid only as a same-harness comparison.
Decision (updated with the EP=8 run below): MegaMoE is an EP kernel and
must be judged as one. At EP=1 it loses 1.3-2.2x (its fused
dispatch/combine has nothing to fuse); at EP=8 attention-DP — its design
point — it WINS at every per-rank batch through 4096 (1.7x at B=1, 3.4x
at B=128) and trails ~5-8% only at 8192-16384. PURSUE for the EP serving
topology; the remaining blockers are K3's real intermediate 384 (needs
the newer upstream DeepGEMM; TRT-bundled requires dims % 512 == 0) and an
equalized-weight correctness gate.

## Operation contract

Expert region only: bf16 latent `[B, hidden]` + precomputed top-16
(IDs int32, scales) -> routed bf16 partial sums. One GPU, world-1 process
group, 112 local experts, hidden 3584, top-16, W4A8 MXFP4/MXFP8, expert
weights randomly filled (practice mode — no checkpoint). Practice
intermediate = 512; K3's real 384 is rejected by the TRT-bundled DeepGEMM
(packed scale rows must be TMA 16B-aligned => dims % 512 == 0).

## Survey — what each implementation actually does

- **TRTLLMGen `run_moe`** (production): a multi-kernel pipeline — activation
  MXFP8 quantize, post-top-k permutation, grouped GEMM1, SwiGLU, grouped
  GEMM2, weighted finalize. EP communication happens OUTSIDE, as separate
  collectives.
- **DeepGEMM `fp8_fp4_mega_moe`** (via TRT `MegaMoEDeepGemm`): ONE fused
  kernel that consumes top-k IDs directly and routes tokens through a
  symmetric-memory buffer — i.e. the EP dispatch/combine all-to-all is
  fused INTO the kernel along with both GEMMs and the activation. That is
  its entire value proposition.
- **vLLM main** ships `KimiK3MegaMoEExperts` calling the same
  `fp8_fp4_mega_moe` with K3's native intermediate 384 and SiTU activation
  — proof that upstream DeepGEMM has outgrown the TRT wheel's limits.
- **FlashInfer/TRT cute-DSL MegaMoE**: NVFP4 only — wrong dtype for K3,
  not applicable.
- DeepGEMM JIT-compiles per configuration: any deployment must warm its
  cache (same rule as the FlashInfer prebuild in kimi_k3_layer/README).

## SOL (measured probe kernels; every cell is a number)

SOL(B) = max of two real measured kernels (`sol_probe.py`):
- **bytes probe**: one bf16 GEMV whose weight matrix holds exactly the
  activated experts' bytes (2.84 MB/expert; with uniform top-16 IDs, 15
  distinct experts are already hot at B=1, all 112 by B=64 -> 43..318 MB)
  — the "load every active expert once" cost;
- **flops probe**: one FP8 dense GEMM with the region's exact FLOPs
  (M=B, K=3584, N=topk*3*inter), divided by 2 for MXFP4's 2x rate.
Bytes probe rows jump with cuBLAS tactic changes (e.g. B=4/16/32 hit ~3.5
TB/s vs ~6.5 TB/s at B>=64), so some rows are conservative — a ratio below
1.0x means the probe row, not the kernel, is slow.

## Results (us, CUDA-graph medians, identical harness, B = 1..16384)

| B | trt_run_moe | dg_mega_moe | mega vs run_moe | SOL (measured) | run_moe x SOL |
|---:|---:|---:|---:|---:|---:|
| 1 | **21.6** | 46.8 | 2.17x slower | 13.8 | 1.6x |
| 2 | **27.3** | 53.4 | 1.96x | 12.0 | 2.3x |
| 4 | **37.8** | 63.2 | 1.67x | 44.4 | 0.9x (probe conservative) |
| 8 | **51.7** | 78.0 | 1.51x | 37.2 | 1.4x |
| 16 | **65.1** | 89.0 | 1.37x | 77.0 | 0.8x (probe conservative) |
| 32 | **72.6** | 93.5 | 1.29x | 81.0 | 0.9x (probe conservative) |
| 64 | **75.6** | 104.8 | 1.39x | 48.8 | 1.5x |
| 128 | **85.5** | 110.4 | 1.29x | 47.9 | 1.8x |
| 256 | **92.0** | 136.3 | 1.48x | 48.8 | 1.9x |
| 512 | **126.5** | 167.6 | 1.32x | 48.1 | 2.6x |
| 1024 | 234.4 | **230.7** | 0.98x (tie) | 48.8 | 4.8x |
| 2048 | **400.6** | 404.5 | 1.01x (tie) | 59.0 | 6.8x |
| 4096 | **601.5** | 688.2 | 1.14x | 113.0 | 5.3x |
| 8192 | **997.9** | 1310.7 | 1.31x | 223.1 | 4.5x |
| 16384 | **1928.0** | 2589.1 | 1.34x | 537.2 | 3.6x |

Why the winner wins per regime:
- **B<=512: trt_run_moe wins** — dg_mega_moe pays its symmetric-buffer
  prepare and monolithic scheduling with no communication to fuse at EP=1,
  a fixed ~20-40 us tax that dominates while the region is small.
- **B=1024-2048: tie** — every expert is hot, both are streaming the same
  318 MB of weights; mega's schedule finally amortizes its prepare.
- **B>=4096: trt_run_moe wins again** — grouped GEMMs with tuned tiles beat
  mega's monolithic schedule once per-expert M is large enough for
  tensor-core-efficient tiles.

The large-B ratios (3.6-6.8x) overstate real headroom: the flops probe is a
perfectly dense FP8 GEMM and excludes activation quantize, the gather of
scattered tokens, and the weighted finalize that any MoE must do. The
mid-range ratios (1.5-2.6x at B=64-512) are against the plain weights-once
cost and are the credible target. The B=1024-2048 mega tie is its one
competitive window at EP=1.

## Idea check: fuse GEMM1 + activation + GEMM2 into one expert kernel

The proposal (no communication in scope): one kernel where each active
expert's CTA streams w13 and w2 exactly once, computes GEMV-like rows for
its few tokens, keeps the intermediate (2 x 512 bf16 = 2 KB/token) in
shared memory between the two layers, and never touches inactive experts.
Ideal cost = the bytes probe = one GEMV over the activated weights.

- **Survey**: DeepGEMM's mega kernel already fuses both GEMMs+activation in
  one kernel but drags the dispatch machinery (measured above: slower at
  B<=512). FlashInfer's cute-DSL path fuses act into GEMM1 and finalize
  into GEMM2 — two kernels, NVFP4-only. No shipped kernel implements the
  decode-specialized weights-once expert-GEMV form for W4A8 MXFP4.
- **Measured headroom for the idea**: run_moe is 1.5-2.6x the weights-once
  cost at B=64-512 (its gap = separate quantize kernel + two GEMM launches
  + permute/finalize traffic + grouped-GEMM tile inefficiency at tiny
  per-expert M). At B<=32 the probe's own tactic jitter hides the margin;
  B=1 shows 1.6x.
- **Risks**: in-kernel MXFP4 dequant throughput on CUDA cores at M~1-2 per
  expert (no tensor-core benefit at GEMV shapes — but the region is
  bandwidth-bound there, so dequant only needs to keep up with ~6 TB/s);
  scattered per-expert token gather; register pressure from double-buffered
  weight streaming.
- **Verdict: PROCEED-worthy** for decode/small-prefill (B<=512), target
  ~50 us at B=64-512 vs run_moe's 76-127 us. This is a real candidate
  kernel project (larger than the permutation candidate); it should start,
  per process, with an Nsight pass on run_moe's B=64-512 kernel chain to
  attribute the 1.5-2.6x before writing code.

Correctness: not yet gated — identical random bytes in differently-laid-out
raw parameters are different logical experts, so cross-backend outputs are
not comparable. A real gate needs an fp32 master weight quantized into each
backend's layout.

## Why mega loses here, and when it could win

At world-1 there is no dispatch/combine communication, so the mega kernel
pays its symmetric-buffer prepare and monolithic scheduling against a mature
multi-kernel pipeline and saves nothing — the 1.3-2.2x is that overhead.
Its honest test is EP>=2, where the production path pays separate NCCL
collectives that mega folds into the kernel. run_moe's own 1.8-3.1x over
the weight bound at B>=32 also says the *production* expert region has
headroom independent of mega (tactic warming first, then profiling).

## EP=8 results (attention-DP, 8x B200 — mega's design point)

Topology: `Mapping(tp=8, moe_ep_size=8, moe_tp_size=1,
enable_attention_dp=True)` — 112 experts/rank, each rank owns B_local
tokens, GLOBAL top-16 ids. Region per rank: bf16 [B,3584] + ids/scales
-> COMBINED routed output [B,3584]. Arms (both unchanged TRT modules):
  - **dg_mega_moe**: ONE `run_moe` call — token dispatch + both GEMMs +
    activation + combine fused in-kernel over NVLink symm memory.
  - **trt_run_moe + NCCL**: the standard DEP data path — all_gather
    (tokens+ids+scales) -> local-expert run_moe over world*B tokens ->
    reduce_scatter(bf16). "comm probe" = its AG+RS wire cost alone
    (measured, same graph timing).
Every cell CUDA-graph timed (median of replays, MAX over ranks); zero
capture failures, mega's symm-buffer path included. B = tokens PER RANK.

| B/rank | trt_run_moe + NCCL | dg_mega_moe | mega speedup | comm probe |
|---:|---:|---:|---:|---:|
| 1 | 99.6 | **58.6** | 1.70x | 43.4 |
| 2 | 108.8 | **63.2** | 1.72x | 43.7 |
| 4 | 123.6 | **72.6** | 1.70x | 43.0 |
| 8 | 145.5 | **89.8** | 1.62x | 45.0 |
| 16 | 174.6 | **99.7** | 1.75x | 45.5 |
| 32 | 180.8 | **103.5** | 1.75x | 50.0 |
| 64 | 250.9 | **114.2** | 2.20x | 63.9 |
| 128 | 418.0 | **122.7** | 3.41x | 89.6 |
| 256 | 440.5 | **156.1** | 2.82x | 104.0 |
| 512 | 487.3 | **238.3** | 2.04x | 141.3 |
| 1024 | 608.6 | **416.2** | 1.46x | 269.4 |
| 2048 | 918.2 | **762.1** | 1.20x | 387.8 |
| 4096 | **1479.6** | 1453.9 | 1.02x (tie) | 720.3 |
| 8192 | **2704.0** | 2830.5 | 0.96x | 1394.6 |
| 16384 | **5209.0** | 5651.0 | 0.92x | 2672.1 |

Why the winner wins per regime:
- **B<=2048: dg_mega_moe, 1.2-3.4x** — the production arm's separate
  collectives put the whole AG on the critical path BEFORE any compute
  and the whole RS AFTER it (43-388 us of exposed wire, the comm-probe
  column), while mega overlaps dispatch/compute/combine inside one
  kernel; the production arm's permute/finalize metadata also scales
  with the GATHERED world*B tokens (the B=128 jump to 418 us is its
  1024-token routing pipeline), where mega's per-rank buffer stays B.
- **B>=4096: tie then trt_run_moe by 4-8%** — wire time becomes
  bandwidth-bound and unhidable either way, and the grouped GEMMs'
  tuned tiles at large per-expert M beat mega's monolithic schedule
  (the same large-B mechanism as the EP=1 table).
- Fairness note: the NCCL AG/RS in the production arm is its stock DEP
  path but not the best measured collective at small B
  (communication/RESULTS.md: symm one-shot/multimem AG+AR run
  ~20-25 us at B<=8 vs this 43-45 us probe). Even crediting the
  production arm that full delta, mega still wins every size <=2048.

Qualification: same-harness comparison on FAKE weights (identical random
bytes, per-backend layouts — outputs not cross-comparable), practice
inter=512. UNQUALIFIED as absolute latencies; the regime structure and
the fused-vs-separate-comm mechanism are the finding.

## Alternatives and decision

Chosen: keep TRTLLMGen run_moe for the CURRENT TP8 layer (no attention-DP
dispatch/combine in that topology); ADOPT-track MegaMoE for EP serving —
the EP=8 table above is a 1.2-3.4x win through B/rank=2048. Rejected:
cute-DSL MegaMoE (wrong dtype). Follow-ups in value order: (1) newer
upstream DeepGEMM for K3's exact shape (inter 384, SiTU) — the one real
blocker; (2) equalized-weight correctness gate; (3) production-collective
baseline for the EP arm (symm AG / multimem) to firm the small-B margin;
(4) warm the autotuner and re-measure run_moe against the weight bound.

Harness: `kimi_k3_layer/kernel_research/kimi_k3_megamoe_practice/bench_expert_region.py` (both backends built by TRT's own
`create_moe`, kernel bodies unchanged). Raw numbers above from the
2026-08-19 run in the llmdd-route-bench container.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python3 \
    kimi_k3_layer/kernel_research/kimi_k3_megamoe_practice/bench_expert_region.py  # ~28 s
CUDA_VISIBLE_DEVICES=0 python3 \
    kimi_k3_layer/kernel_research/kimi_k3_megamoe_practice/sol_probe.py
# EP=8 attention-DP table (trt-dev container, ~3 min):
mpirun --allow-run-as-root -np 8 python3 \
    kimi_k3_layer/kernel_research/kimi_k3_megamoe_practice/bench_expert_region_ep.py
```
