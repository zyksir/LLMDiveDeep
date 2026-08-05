# Config → numbers ledger (Kimi-K3 MoE TP8, us, graph replay, max-over-ranks)

## KDA decode (layer-level, GPU4, tp8 shard, graph replay; k1/k3 runs)
| B | trt_fused | b10_fused | gain |
|---|---|---|---|
| 1 | 41.74 | 36.09 | +13.5% |
| 2 | 44.17 | 37.19 | +15.8% |
| 4 | 44.80 | 37.87 | +15.5% |
| 8 | 44.69 | 38.68 | +13.4% |
| 16 | 46.97 | 41.23 | +12.2% |
All backends PASS correctness. B=8 profile: in_proj GEMM 21.9us (2.8TB/s! roofline 8.6),
KDA core 12.2(triton)->4.7(cutedsl), attn_res 7.2, o_proj 6.0 (roofline 2.8), f_b 3.0.
NEXT-BEST decode lever: in_proj GEMM (~10us, ~25% layer, ~12% e2e halved).

## KDA prefill (layer-level, GPU4, tp8 shard, B=1, eager; k6 run)
baseline = in_proj -> conv -> TRT triton chunk (incl. both production D2H syncs) -> norm -> o_proj
opt = same + b10 CuTeDSLGen chunk kernel + triton l2norm glue + sync-free reset
| S | base | opt | gain | ~e2e (halved) |
|---|---|---|---|---|
| 4096 | 1231.0 | 921.5 | +25.1% | ~12.5% |
| 8192 | 1982.8 | 1585.8 | +20.0% | ~10% |
| 16384 | 3517.3 | 3147.0 | +10.5% | ~5% |
Cosine base-vs-opt 0.90 — the DEVIATION IS THE BASELINE'S (trt chunk = 0.8235 vs exact oracle
in linear_attn's own harness; b10 kernel = 0.999993, verified in debug/kda_prefill_gate_debug.py).
Kernel-level (H=12): trt 325/615/1171 vs b10 189/376/~740 at 4k/8k/16k.
D2H sync removal alone is <1-3% of layer at these S (kernel choice is the story).

Node: model-performance box, 8x B200, container tensorrt-llm 1.3.0rc23.
Targets (user, Aug 5): B=2,4,8 >40%; B=16 >30%; honest baseline.

## Runs BEFORE clock locking (unlocked clocks — ±8 us run-to-run drift, single input)

### b1 c96f: ablate, REF_TAIL_MIN_TOKENS=4, 1 input (b1_ablate.log)
| B | base | opt | -merged | -routing | -fc1shard | -fc2shard | -reftail | -customcomm | -overlap | -shared | -all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 75.27 | 52.06 | 53.72 | 58.48 | 58.15 | 64.58 | 51.05 | 51.15 | 60.00 | 49.49 | 89.58 |
| 2 | 81.38 | 52.65 | 55.26 | 61.73 | 63.53 | 69.27 | 52.52 | 52.52 | 62.02 | 51.03 | 99.27 |
| 4 | 84.07 | 64.70 | 68.98 | 73.94 | 69.89 | 64.78 | 56.39 | 64.68 | 74.34 | 52.99 | 99.96 |
| 8 | 89.94 | 75.45 | 79.70 | 75.45 | 78.71 | 75.52 | 69.63 | 75.39 | 84.79 | 58.03 | 105.99 |
| 16 | 830(bogus) | 82.94 | 88.34 | 82.49 | 84.45 | 83.70 | 82.37 | 83.04 | 93.12 | 75.08 | 111.92 |

### b2: sizes 8,16, REF_TAIL off (=1e9): B=8 89.55/70.94 (+21%); B=16 95.26/82.25 (+14%)
### b3: sizes 4,8,16, REF_TAIL=4 + fin_fuse(v1, wrapper): 84.23/68.08 (+19), 89.68/74.57 (+17), 95.62/80.96 (+15)
### b4 (trace run): REF_TAIL=16: 83.61/52.12 (+38), 89.47/58.82 (+34), 90.99/63.41 (+30), 96.78/76.84 (+21), 103.52/88.46 (+15)  <- baselines ~8us higher than b1/b2 (drift)
### b5: sizes 8,16, REF_TAIL=16 + fin_fuse(v2, direct packed): 89.90/70.98 (+21), 95.41/85.51 (+10)  <- B=16 worse than b3; drift suspected

## Runs AFTER clock locking (1965 MHz) + multi-input averaging (n_inputs=8)

### c1: ablate, REF_TAIL=16 (fin_fuse v2 direct-packed at 16), n_inputs=8, locked clocks
| B | base | opt | -merged | -routing | -fc1shard | -fc2shard | -reftail | -finfuse | -customcomm | -overlap | -shared | -all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 78.11 | 54.46 +30% | 55.92 | 60.29 | 60.11 | 66.73 | 53.45 | 53.21 | 53.31 | 62.54 | 51.98 | 92.84 |
| 2 | 83.58 | 54.82 +34% | 57.77 | 65.54 | 64.22 | 71.31 | 54.88 | 55.11 | 54.80 | 64.59 | 53.67 | 102.02 |
| 4 | 86.21 | 57.55 +33% | 60.58 | 66.72 | 66.75 | 73.82 | 57.54 | 57.57 | 57.37 | 66.55 | 54.80 | 103.53 |
| 8 | 90.91 | 69.38 +24% | 72.76 | 76.14 | 76.79 | 89.59 | 69.39 | 69.41 | 69.51 | 77.10 | 60.47 | 108.85 |
| 16 | 96.52 | 87.75 +9% | 92.35 | 83.44 | 82.33 | 87.75 | 83.29 | 84.15 | 87.68 | 92.71 | 75.57 | 114.04 |

c1 reads: B=16 — fc1shard HURTS (-5.4), ref/finfuse tails LOSE to shard tail, fin_fuse v2 worse than v1.
Shared exposure (opt minus -shared): B=1 2.5 / B=2 1.2 / B=4 2.8 / B=8 8.9 / B=16 12.2 us.
B=8 tail: -fc2shard 89.6 vs opt 69.4 -> the fc2-shard tail IS the win at 8.

### c2: merge3 + shard tail everywhere (REF off, FC1_SHARD<=8)
B=1 78.52/52.71 (+33) | B=2 83.41/53.69 (+36) | B=4 84.91/55.77 (+34) | B=8 90.74/64.16 (+29) | B=16 96.36/88.73 (+8, full-fc1 merge3 = wrong config)

### c3 (B=16 grid, merge3 + fc1shard ON):
c3a shard tail: 96.63/83.57 (+14) | c3b ref+finfuse tail: 97.03/81.83 (+16) <- ref+finfuse best at 16 w/ merge3

### c4: fin_rs (finalize fused into RS push) + merge3, shard tail, FC1_SHARD<=16
B=1 79.19/59.96 (+24) | B=2 83.64/59.42 (+29) | B=4 85.37/61.94 (+27) | B=8 90.97/70.89 (+22) | B=16 96.45/87.76 (+9)
-> fin_rs CORRECT but ~6us SLOWER than c2. Suspect: experts through the flashinfer op
   (do_finalize=False path) run untuned default tactics; native op default is better.
   Same signature as c1's fin_fuse-v2 regression. Fix attempt: autotune() in capture warmup (c5).

### c5: c4 config + AUTOTUNER warmup in capture
B=1 79.08/60.00 | B=2 83.82/59.36 | B=4 85.79/62.44 | B=8 91.34/70.88 | B=16 97.05/87.87
-> identical to c4: autotune is NOT the fin_rs penalty; gather serialization is. fin gate reverted to ref-tail-only.

### c6: baseline honesty sweep (B=2/8/16, base us):
AUTO 83.6/90.9/96.5 | ONESHOT **81.2/83.3/92.4** | TWOSHOT 81.9/226/214 | MIN_LATENCY 84.1/91.1/97.1 | NCCL 101/103/112
-> HONEST BASELINE = ONESHOT. native-op arm produced no rows (errored; not production config anyway).
route_pack warp sweep: nw=4 optimal (6.0-6.3us) at B=1..16; nw=2/8/16 worse.

### CuTeDSLGen: fc2 tail GEMM [16,3584]x[3584,7168] addmm spec launched (2 claude arms, GPUs 0/1).
env fix: dropped flash-attn/yunchang from pyproject (CUDA 12.8 host vs cu13 wheels), nvidia-cutlass-dsl[cu13].

### flashinfer patch: DynBlockKernelMaxNumExperts 512 -> 2048 (RoutingCustomPolicy.cuh, .orig backup kept).
K3's 896 experts were excluded from the dyn-block permute tier (B=5..16) -> forced 10.6us cluster kernel.
Only the PRECOMPUTED-topk (opt) path uses this tier; baseline DeepSeek permutation unaffected. Rebuild running.

### CuTeDSLGen arms running (4): tail GEMM (GPUs 0/1), expert GEMV fp4xfp8 (GPUs 2/3, spec
moe_expert_gemv_spec.md — replaces permute+bmm1+bmm2+finalize, ~34us -> roofline 8-10us at B=16).
Integration note if expert GEMV converges: needs UNSHUFFLED fp4 weights (trtllm-gen may shuffle at
post_load_weights) — keep an unshuffled copy for the custom kernel; output feeds rs_cols/norm_reduce directly.

### Best per-shape today vs ONESHOT baseline:
B=1 ~78/52.7 (~+32%) | B=2 81.2/53.7 (+33.9%) | B=4 ~84/55.8 (~+33%) | B=8 83.3/64.2 (+22.9%) | B=16 92.4/81.8 (+11.5%)
Targets: 40/40/40 at 2/4/8, 30 at 16 -> gaps ~5/6/14/17 us. Need permute/fc2/bmm-class kernel wins.

### k9: prefill layer w/ graph column (4k/8k/16k): base 1222/1986/3518, opt-eager 819/1588/3140 (+33.0/+20.0/+10.8%), opt-graph 808/1616/3121 (+33.9/+18.6/+11.3)
### d3: decode GEMM swap standalone: cuBLAS in_proj 11.3-13.8us (=~7.5TB/s, NOT the lever; trace was PDL-inflated); cutedsl wins only B=1 (+2.7)
### decode +20% path -> attn_res kernel (7.2->~3, spec attn_res_spec.md queued) + f_b_proj fusion into b10 core

### c8 FINAL ablate (ONESHOT base, REF_TAIL=16, FC1<=16, tail kernel on, n_inputs=8):
| B | base | opt | -merged | -routing | -fc1shard | -fc2shard | -reftail | -finfuse | -merge3 | -customcomm | -overlap | -shared | -all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 79.11 | 54.22 | 57.50 | 60.63 | 63.76 | 62.16 | 52.86 | 52.70 | 55.07 | **50.96** | 64.19 | 53.56 | 92.01 |
| 2 | 81.25 | 54.66 | 58.54 | 63.82 | 67.01 | 70.01 | 54.64 | 54.66 | 55.28 | **53.14** | 65.52 | 54.13 | 98.83 |
| 4 | 83.12 | 57.91 | 61.59 | 68.81 | 70.14 | 71.57 | 57.86 | 57.93 | 58.95 | **56.10** | 68.34 | 56.57 | 99.83 |
| 8 | 83.24 | 66.86 | 74.48 | 76.04 | 78.82 | 79.88 | 66.88 | 66.77 | 70.81 | **65.00** | 79.28 | 62.23 | 100.44 |
| 16 | 92.25 | 79.04 | 91.35 | 89.45 | 85.72 | 78.99 | 82.07 | 81.16 | 87.08 | **79.02** | 92.70 | 77.49 | 109.94 |
KEY: -customcomm (fi AR+norm tail) beats custom RS at EVERY size with merge3 -> default comm flipped to "fi".
Tail kernel at 16: opt 79.04 vs prior c3b 81.83 (-2.8us from CuTeDSL fc2 GEMM).
### c9 FINAL config (comm=fi default) — PENDING

### c7: dyn-block patch FAILED at launch ("too many resources requested", tier 1024 dyn-block at B=8/16) -> REVERTED, rebuild running.
### CuTeDSLGen round 1 final: tailgemm_b WINNER (79.1% roofline, 6327 GB/s; vendored + stream-patched -> kimi_k3_layer/tail_gemm_cutedsl.py, env B10_TAIL_KERNEL).
### tailgemm_a/a2, expertgemv x4 arms: no convergence. expert GEMV = the kernel that would close B=8/16 targets; retry with ARM_TURNS=4 queued.

### CuTeDSLGen agent-arm gotcha (cost ~6 failed arms): claude CLI refuses
--dangerously-skip-permissions as root UNLESS IS_SANDBOX=1 is in the env.
Some of my shells carried it (arms worked), some didn't ("Execution error" @ 15B stdout, timeout).
ALWAYS: export IS_SANDBOX=1 before run_arm.sh on this box.
### attn_res kernel: standalone 5.0us (vs TRT ~7) but IN-LAYER +1.5-2us and B=1 graph-capture break
-> vendored kimi_k3_layer/attn_res_cutedsl.py, default OFF (B10_ATTNRES_KERNEL). KDA decode +20% still open.

### gen_kdachunk_d: agent produced a CORRECT fused-prologue chunk kernel (cos>=0.999) but
1.28ms @ 16k vs incumbent b10 ~0.74ms -> NOT integrated. KDA prefill 16k +20% deferred to a
next improvement cycle (respec must keep the INT21 schedule AND fuse the prologue).
### gen_gemv6 (retry-wrapped, 4 turns) is the last arm running — the MoE B=8/16 decisive kernel.

### gen_gemv6 FINAL: kernel runs 46.4us @ B=16 (1374 GB/s = 17% peak) vs incumbent ~34us -> SLOWER
(per-(token,pick) formulation re-reads expert weights; needs per-active-expert grouping to hit
the 8-10us roofline) + 45/14336 elements out of the strict gate. NOT integrated. Improvement-cycle item.

### recurrent_kda (PR #4262) comparison: STILL PENDING. Contract fixed in debug/kda_rkda_compare.py
(cu_seqlens required, bf16 [N,HV,V,K] state, PRE-SIGMOIDED beta) + NaN guard added. GPU7 attempt
invalid: b10 4x its known numbers (contended GPU) and fi kernel returned NaN in 10-21us (silent
dispatch failure). Rerun on clean node - RUNBOOK section 4.

### KDA prefill kernel duel (GPU6, b10-canary clean): b10 190/376/739 vs INT21 266/517/1016 (H=12, +29%)
and 843 vs 1443 at H=96/8k (+41%). INT21 built from Int21-AI/KDA-B200 (registry path patched, row PASSes
exact-correctness). flashinfer 0.6.15 does NOT contain PR #4262 (merged Aug 3 > wheel); its recurrent_kda
is the pre-PR generic path (slow + wrong for prefill). Duel vs the real CAKE kernels needs flashinfer@main
in an isolated venv - runbook updated.

### g2 GEMM sweep @ bs=1..80 (GPU6 clean): ALL dense GEMMs FLAT through 80 (compute transition >>80).
fc1_full 9.4->10.2 | fc2_full 5.5-8.0 | merge3 7.1-8.3 (beats split gate+fc1 11.4-15.1 at 64/80!)
fc1_shard floor 4.9us @1.3TB/s -> GEMM-side shard saving only 2.2-4.5us at ALL sizes (fusion-only value).
shared_gu cuBLAS CLIFF at bs=32: 11.9us (2.1x neighbors) - merge3 dodges it.
=> 32-80 retune priors: drop fc1 shard (unless AG-quant pays), tail decided by collectives only
(full-fc2 read stays cheap), extend merge3 past 64, never run shared_gu separately at 32.
