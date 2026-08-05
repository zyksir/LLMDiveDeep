# Kimi-K3 MoE decode optimization — final config, per-BS strategy, and evidence

TP8 on 8x B200 (`model-performance` box), official
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` container, decode
batches **B = 1, 2, 4, 8, 16**, vs the production-aligned baseline of
`README.md`. The shipped configuration is the **code defaults** of
`KimiK3MoEB10` — no env vars. Raw sweep logs:
`results/experiments/` (incl. `overnight_ledger_aug5.md`, every
config → number).

## 1. Final results (validated; `results/bench_moe_kimi_k3_tp8.{csv,md,png}`)

| B | baseline (honest, ONESHOT AR) | opt | improvement | user target |
|---|---|---|---|---|
| 1 | 79.5 us | 52.6 | **+33.8%** | ≥20% ✅ |
| 2 | 81.6 | 54.0 | **+33.8%** | >40% ❌ (~5 us short) |
| 4 | 83.1 | 55.9 | **+32.7%** | >40% ❌ (~6 us short) |
| 8 | 83.7 | 65.0 | **+22.4%** | >40% ❌ (~15 us short) |
| 16 | 92.4 | 79.0 | **+14.5%** | >30% ❌ (~14 us short) |

Reproduce (defaults = shipped config; the env var is the
honest-baseline knob of §6):

```bash
docker exec trt-dev bash -c "cd /workspace/.../LLMDiveDeep && \
  BENCH_ALLREDUCE_STRATEGY=ONESHOT mpirun -x BENCH_ALLREDUCE_STRATEGY \
    -n 8 --allow-run-as-root \
    python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16 --ablate"
```

## 2. The final strategy, per batch size

| stage | B = 1 / 2 / 4 / 8 | B = 16 |
|---|---|---|
| input GEMM | merge3: ONE wide `[gate \| fc1-shard \| shared g/u]` GEMM | same |
| routing | Triton "ours" top-16, `run_moe` format | same |
| fc1 latent | column shard + one-shot AG with **fused MXFP8 quantize** | same |
| experts | production trtllm-gen kernels, finalized | same kernels, **unfinalized** (`do_finalize=False`) |
| shared expert | SiTU + down GEMM on the aux stream | same, **plus its AR(7168) on the aux stream** |
| latent reduction | fi fused AR + RMSNorm (one kernel) | **fused finalize + AR + RMSNorm** (one flashinfer kernel) |
| fc2 | per-rank **column slice**, `addmm_` into the shared partial | **FULL fc2** via the CuTeDSLGen tail GEMM (`tail_gemm_cutedsl`) |
| output collective | fi one-shot AR(7168) | **none** — every rank already holds the identical output |

There is exactly **one structural fork: the tail, at B=16**
(`REF_TAIL_MIN_TOKENS = 16`). Everything upstream of the tail is
identical at all five sizes.

## 3. Strategies that are per-batch-size (on for some B, off for others)

| strategy | ON | OFF | why the crossover sits there (c8 ablation) |
|---|---|---|---|
| `ref_tail` + `fin_fuse` + CuTeDSL tail GEMM | B ≥ 16 | B < 16 | at 16 the fused finalize+AR+norm → full-fc2 tail (no output collective, shared AR hidden) beats the shard tail by ~2 us (79.0 vs 82.1); below 16 the shard tail wins by 2–6 us — the unfinalized flashinfer expert path costs more than the finalize kernel it saves |
| `fc2_shard` tail (slice + `addmm_` + AR) | B < 16 | B ≥ 16 | superseded by the ref tail above the crossover |
| `fc1_shard` + fused AG-quant | B ≤ 16 | B > 16 | the AG's wire grows ~7 KB/token; wash-to-loss past 16 |
| *(outside the target range)* `overlap_shared` | B ≤ 64 | B > 64 | at 128 overlapping two weight-read-bound stages just serializes in DRAM |
| *(outside)* merged input GEMM → split gate/routing/fc1 | B < 64 | B ≥ 64 | split gives routing a clean window once the expert stage is long |
| *(outside)* whole opt path | B ≤ 128 | fallback to baseline | Lamport/one-shot costs grow with tokens while the savings stay flat |
| *(outside)* `prefill_opt` | tokens ≥ 192 | below | the CE all-gather's latency floor can't hide under the gate/shared compute |

## 4. Strategies that are ALWAYS on (B = 1…16)

| strategy | mechanism | evidence (c8: switch-off cost, us) |
|---|---|---|
| **merge3** | gate `[896]` + fc1-shard `[448]` + shared g/u `[1536]` in ONE `[B,7168]x[7168,2880]` GEMM — cuBLAS runs narrow-N at ~2 TB/s but wide-N at ~8, and the aux stream stops competing for DRAM | +0.9 … +3.9 |
| **routing "ours"** | one Triton CTA/token: sigmoid + bias + top-16 on the bf16 logits in place, emitting `run_moe`'s exact (ids, scales) — the runner skips its scores stage; also the pre-packed handoff `fin_fuse` needs | +6.4 … +10.4 |
| **fc1 shard + fused AG-quant** | rank computes `[B,448]` of the latent; a single Lamport kernel gathers AND quantizes to MXFP8 bit-exactly — the separate quantize kernel and the contiguous copy disappear | +6.7 … +12.4 |
| **shared-expert overlap** | SiTU chain forked onto the aux stream after routing, hidden under the expert window | +10.0 … +13.7 |
| **`comm="fi"`** | flashinfer fused AR+RMSNorm / one-shot AR for every reduction (implementation choice, applies at all sizes) | +1.5 … +3.5 vs the custom RS |
| **merged front + prefill path** | see §3 crossovers; both on throughout B=1..16 | — |

## 5. Strategies that FAILED (never on; kept as documented negatives)

| strategy | verdict |
|---|---|
| custom Lamport **RS+norm** tail (`comm="custom"` — the former default) | loses 1.5–3.5 us at every size once merge3 freed the aux stream; kept in-tree for re-tuning on other nodes. NOTE: the custom **AG** did *not* retire — it is the fc1-shard workhorse and is not gated by `comm` |
| finalize fused into the RS push (`reduce_scatter_cols_finalize`, comm_cuda.py) | correct but ~6 us SLOWER at B≤8 — the 16-way gather serializes the one-CTA-per-token push loop |
| `fin_fuse` below B=16 | the unfinalized flashinfer expert path never beat native-op + separate finalize when paired with the shard tail |
| `rs_defer` (deferred RMSNorm scale after the RS) | loses at decode sizes; default off |
| flashinfer dyn-block permute patch (`DynBlockKernelMaxNumExperts` 512→2048) | the 1024-expert tier fails to launch ("too many resources requested") — the gate is structural; reverted |
| expert fp4×fp8 GEMV kernel (would replace permute + bmm1 + bmm2 + finalize, ~34 us → 8–10 roofline) | two CuTeDSLGen near-misses: correct-but-46 us at B=16 (per-(token,pick) formulation re-reads expert weights at ~2 tokens/expert) and a strict-gate fail. **This is the queued kernel that closes the B=8 40% / B=16 30% gaps** — the winning shape is per-ACTIVE-EXPERT grouping, streaming each expert's 2.06 MB once |
| `skip_shared` | not an optimization — a timing probe (output intentionally wrong); measures shared-chain exposure |

## 6. Deployment classes (`moe_deploy_kimi_k3.py`) — weight layout fixed at init

A real TRT-LLM integration cannot flip weight layouts at runtime. Two
classes freeze the measured-best configuration per deployment shape
(`set_opt_flags` locked; `--moe-class agg|disagg` in the bench):

|  | **MoEForAgg** (prefill+decode engine) | **MoEForDisAggDecode** (decode-only) |
|---|---|---|
| fc1 | column-sharded (prefill FLOP cut + decode fused AG-quant) | column-sharded (inside merge3) |
| input weight | ONE merge3 tensor `[gate | fc1-shard | shared g/u]`; prefill GEMMs its contiguous row-slice `[:1344]`, shared g/u = the `[1344:]` view - **shared-fusion is NOT a conflict** | same |
| fc2 | **SHARDED only** (-45 MB/rank/layer); decode uses the shard tail at every size (B=16: ~+11% instead of +14.5%). Optional `full_fc2_allgather` (`B10_AGG_FULL_FC2=1`): ONE-TIME weight all-gather at init - weights are static, so it overlaps engine warmup, never a per-step cost - restores the B>=16 ref tail for +51 MB/rank/layer | **FULL + 6.4 MB slice copy**: ref tail + fused finalize + CuTeDSL tail GEMM at B>=16, shard tail below |
| prefill path | on; collective/compute pairing: CE all-gather <-> gate+routing(+shared, small S), latent allreduce <-> shared chain on the aux stream (early fork lets it slide into whichever stall window exists) | disabled (prefill never reaches this engine) |
| decode expectation | = finals at B<=8; B=16 +11% (or +14.5% with the weight AG) | = the §1 finals |

Functional verification (TP4, GPUs 4-7; TP8 pending - see below):
DisAgg decode 1/8/16 correct, +18/+11/+9%; Agg decode identical to
DisAgg (same front), Agg prefill correct, +5% at 512 but -2/-6% at
1k/4k **at TP4 on this node** - the prefill crossovers
(`PREFILL_MIN_TOKENS`, and whether the sharded path pays at all) are
TP-size and node dependent and must be re-validated at TP8 (the
old-node TP8 numbers were +12/+18/+22% at 512/1k/2k+). GPUs 0-3 are
currently wedged by orphaned CUDA contexts (host reboot required), so
the TP8 deployment-class validation is queued.

## 7. Measurement protocol (all numbers above)

* **Idle GPUs verified** (`nvidia-smi`) before every run.
* **Locked clocks** (`nvidia-smi -lgc 1965,1965`) — unlocked runs
  drifted ±8 us, larger than several optimizations.
* **CUDA-graph replay**, 100 iterations/graph, **max over ranks**.
* **Mean over 8 random inputs** (`--n-inputs 8`) — routing/EP-skew is
  data-dependent; single-input runs flipped two crossovers.
* **Autotuner warmup** before capture (production parity).
* **Honest baseline**: swept the baseline's own knobs; `ONESHOT`
  AllReduce beats production's AUTO by 2–8 us here (TWOSHOT is
  pathological: 214–226 us) — all margins quoted against ONESHOT.
* **Never take headline numbers from a `--profile` run** — CUPTI
  stays attached and taxes every later kernel (~+7 us).

## 8. Baseline anatomy (where the 80–92 us goes; B=16 trace)

gate GEMM 11.5+4.2 | fc1 (full 51 MB) 10.8 | quantize 4.3 | routing
scores 7.8 + permute 9.2 | bmm1 13.0 + bmm2 10.2 | finalize 3.8 |
cat 3.6 | fused AR 8–16 | RMSNorm 2.5 | fc2 (full 51 MB) ~11 |
add 3.6 | (aux, overlapped: shared chain ~26). Everything is
latency/weight-read bound: two replicated 51 MB latent projections,
~17 us of routing, and a serialized cat→AR→norm→GEMM→add tail — which
is exactly the list §§3–4 attack.

## 9. Full ablation (c8; every switch off one at a time, us)

| B | base | opt | -merged | -routing | -fc1shard | -fc2shard | -reftail | -finfuse | -merge3 | -customcomm | -overlap | -shared | -all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 79.1 | 54.2 | 57.5 | 60.6 | 63.8 | 62.2 | 52.9 | 52.7 | 55.1 | 51.0 | 64.2 | 53.6 | 92.0 |
| 2 | 81.3 | 54.7 | 58.5 | 63.8 | 67.0 | 70.0 | 54.6 | 54.7 | 55.3 | 53.1 | 65.5 | 54.1 | 98.8 |
| 4 | 83.1 | 57.9 | 61.6 | 68.8 | 70.1 | 71.6 | 57.9 | 57.9 | 59.0 | 56.1 | 68.3 | 56.6 | 99.8 |
| 8 | 83.2 | 66.9 | 74.5 | 76.0 | 78.8 | 79.9 | 66.9 | 66.8 | 70.8 | 65.0 | 79.3 | 62.2 | 100.4 |
| 16 | 92.3 | 79.0 | 91.4 | 89.5 | 85.7 | 79.0 | 82.1 | 81.2 | 87.1 | 79.0 | 92.7 | 77.5 | 109.9 |

(This run had `comm="custom"` as opt; its `-customcomm` column — the
fi tail — beat it everywhere and became the shipped default,
confirmed by the clean final run of §1. `-shared` is the timing
probe: opt minus `-shared` = shared-chain exposure, 1–5 us.)

## 10. What remains (roofline; the path to 40/30)

| kernel (opt path, B=16) | measured | roofline-ish | attack |
|---|---|---|---|
| routing permute (cluster kernel) | 10.6 us | ~2–3 | dies entirely inside the expert-GEMV formulation |
| expert bmms (fp4×fp8) | 23.5 | ~8–10 (active-weight read) | expert GEMV, per-active-expert grouping |
| finalize+AR (fused) | 13.1 | ~8 + EP skew | skew is data-dependent; both paths pay it |
| fc2 tail GEMM (CuTeDSL) | ~8 | 6.5 | already 79% of roofline |
| route_pack | 6.5 | ~3–4 | radix top-16 (serial 16-max chain today) |
| AG+quant | 7.3 | ~4–5 | sender-side fp8-wire push (`ag_qpush`, exists, untuned) |

B=8 needs ~15 us and B=16 ~14 us below the current opt: only the
expert-GEMV class win covers that; everything cheaper is already in.

## 11. Target-size update (Aug 5, late): B = tokens/step, range 1..80, 32 primary

The user's real target set is **B = 1,2,4,8,16,32,64,80 with 32 the
most important** — the SS1-3 tuning covered 1-16 only, and every
crossover pivots inside 16-80. A TP4 probe was started but VOIDED
(GPUs taken mid-run). The re-tune procedure, decision table, and the
communication-cost reasoning for the 32-80 tails live in
`RUNBOOK.md` SS1; nothing in SS2-4 should be treated as final for
B>=32 until that run lands on a clean TP8 node.

## 12. Node-dependence warning

Every crossover here was RE-measured on this box and several flipped
vs the box `README.md` was written on (collectives are much cheaper
here): ref-tail moved B≥4→B≥16, the custom RS went from tie to loss,
ONESHOT beats AUTO for the baseline. All thresholds are env-exposed
(`B10_REF_TAIL_MIN_TOKENS`, `B10_FC1_SHARD_MAX_TOKENS`,
`B10_TAIL_KERNEL`, …) — re-run `--ablate` after moving hardware.
