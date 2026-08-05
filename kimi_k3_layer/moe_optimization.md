# Kimi-K3 MoE decode optimization — final config, per-BS strategy, and evidence

TP8 on 8x B200 (`model-performance` box), official
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` container, decode
batches **B = 1…80 (MTP tokens/step; 32 the user's primary)**, vs the
production-aligned baseline of `README.md`. The shipped configuration
is the **code defaults** of `KimiK3MoEB10` — no env vars. Raw sweep
logs: `results/experiments/` (incl. `overnight_ledger_aug5.md`);
the B=1..80 retune logs: `/tmp/retune_b80.log`, `/tmp/combo_b80.log`,
`/tmp/strategy_hunt.log` on the node (runner scripts:
`run_retune_b80.sh`, `run_combo_b80.sh`).

## 1. Final results (validated; `results/bench_moe_kimi_k3_tp8.{csv,md,png}`)

Aug-5 late retune at the full target range, GPU-clean, locked clocks,
8-input mean. **Positive at every size** (the user's floor):

| B | baseline (honest, ONESHOT AR) | opt | improvement |
|---|---|---|---|
| 1 | 77.3 us | 53.1 | **+31%** |
| 2 | 80.1 | 54.0 | **+33%** |
| 4 | 82.0 | 56.3 | **+31%** |
| 8 | 83.4 | 65.7 | **+21%** |
| 16 | 92.3 | 78.4 | **+15%** |
| 32 | 106.9 | 90.5 | **+15%** |
| 64 | 126.8 | 115.4 | **+9%** |
| 80 | 143.2 | 119.7 | **+16%** |

(B=16 flaps 78.4–84.6 across otherwise-identical runs — the AG
autotune pick and EP skew move it ~6 us; every other size repeats
within ~1 us. B=64 is the weakest point: it pays the noaux routing
tax AND sits at the ref tail's worst spot — the expert-GEMV kernel of
§5/§10 is the fix.)

Reproduce (defaults = shipped config; the env var is the
honest-baseline knob of §6):

```bash
docker exec trt-dev bash -c "cd /workspace/.../LLMDiveDeep && \
  BENCH_ALLREDUCE_STRATEGY=ONESHOT mpirun -x BENCH_ALLREDUCE_STRATEGY \
    -n 8 --allow-run-as-root \
    python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16,32,64,80 --ablate"
```

## 2. The final strategy, per batch size

| stage | B ≤ 8 | B = 16 / 32 | B = 64 | B = 80 |
|---|---|---|---|---|
| input GEMM | merge3: ONE wide `[gate \| fc1-shard \| shared g/u]` GEMM | same | **split** gate → routing → fc1 (routed pipeline) | two-way merge `[gate \| fc1-shard]`, shared g/u on aux |
| routing | Triton "ours" top-16, `run_moe` format | same | same | **stock noaux** (in-kernel) — ours loses ~8 us here |
| fc1 latent | column shard + one-shot AG with **fused MXFP8 quantize** | same | same | column shard + AG (bf16) + quantize |
| experts | production trtllm-gen kernels, finalized | 16+: **unfinalized** (`do_finalize=False`) | unfinalized | finalized (fin_fuse needs "ours" routing) |
| shared expert | SiTU + down GEMM on the aux stream | same, plus its AR(7168) on the aux stream | aux stream | aux stream (overlap gate now 96) |
| latent reduction | fi fused AR + RMSNorm | **fused finalize + AR + RMSNorm** | fi fused AR + RMSNorm | fi fused AR + RMSNorm |
| fc2 | per-rank **column slice**, `addmm_` | **FULL fc2** via the CuTeDSL tail GEMM | **column slice** (shard tail — ref tail skipped at 64) | **FULL fc2** via the CuTeDSL tail GEMM |
| output collective | fi one-shot AR(7168) | none | fi one-shot AR(7168) | none |

## 3. Strategies that are per-batch-size (on for some B, off for others)

All crossovers re-measured Aug-5 at B=1..80 with fc1-shard always-on
(forced-extremes + combos, `run_combo_b80.sh` E1–E5 + strategy-hunt):

| strategy | ON | OFF | why the crossover sits there |
|---|---|---|---|
| `ref_tail` (+`fin_fuse` where routing="ours") + CuTeDSL tail GEMM | B ≥ 16 **except 64** (`REF_TAIL_SKIP`) | B < 16, B = 64 | at 16–32 it beats the shard tail by 2–6 us; at exactly 64 the shard tail wins by ~2.7 (115.3/115.5 vs 118.0, twice-confirmed); at 80 ref wins by ~7 vs shard (121.8 vs 129.0) and ~21 vs fat-AR |
| `fc2_shard` tail (slice + `addmm_` + AR) | B < 16, B = 64 | elsewhere | the complement of the row above |
| routing "ours" (`ROUTING_OURS_MAX_TOKENS=64`) | B ≤ 64 | B > 64 | one CTA/token with a serial 16-max chain: +16 us at 32, +3 at 64, **−8 at 80** vs stock noaux |
| `overlap_shared` (`B10_OVERLAP_MAX_TOKENS=96`) | B ≤ 96 | above | worth **19 us at B=80** (140.9→121.6); the old ≤64 gate was measured under hot routing that no longer exists |
| `merge3` (`MERGE3_MAX_TOKENS=64`) | B ≤ 64 (auto-off at 64 via routed pipeline) | B > 64 | at 80 the wide-N GEMM has started scaling with B; two-way merge + aux shared GEMM is ~2.7 us faster |
| merged input GEMM → split gate/routing/fc1 | B < 64 | B ≥ 64 (needs routing "ours", so 64 only) | split gives routing a clean window once the expert stage is long |
| *(outside the target range)* whole opt path | B ≤ 128 | fallback to baseline | Lamport/one-shot costs grow with tokens while the savings stay flat |
| *(outside)* `prefill_opt` | tokens ≥ 192 | below | the CE all-gather's latency floor can't hide under the gate/shared compute. **WARNING: agg-prefill measured −11/−62/−4% at 1k/4k/8k on this node (step 7b) — needs a re-tune or a higher floor before any agg deployment** |

## 4. Strategies that are ALWAYS on (B = 1…80)

| strategy | mechanism | evidence (switch-off cost, us) |
|---|---|---|
| **fc1 shard (+ fused AG-quant where routing="ours")** | rank computes `[B,448]` of the latent; a single Lamport kernel gathers AND quantizes to MXFP8 bit-exactly. **Aug-5 forced-extremes verdict: ALWAYS on** — shard-on beats shard-off at every size (89.4→83.3 @16, 99.0→90.6 @32, 132→118 @64, 154→148 @80) with zero small-B regression, the old ≤16 gate was wrong on this node. Also what the deploy classes need (no full-fc1 fallback exists) | +5 … +14 |
| **merge3** (≤64; see §3) | gate `[896]` + fc1-shard `[448]` + shared g/u `[1536]` in ONE `[B,7168]x[7168,2880]` GEMM — cuBLAS runs narrow-N at ~2 TB/s but wide-N at ~8, and the aux stream stops competing for DRAM | +0.9 … +3.9 |
| **routing "ours"** (≤64; see §3) | one Triton CTA/token: sigmoid + bias + top-16 on the bf16 logits in place, emitting `run_moe`'s exact (ids, scales) — the runner skips its scores stage; also the pre-packed handoff `fin_fuse` needs | +6.4 … +10.4 |
| **shared-expert overlap** (≤96) | SiTU chain forked onto the aux stream after routing, hidden under the expert window | +10 … +21 |
| **`comm="fi"`** | flashinfer fused AR+RMSNorm / one-shot AR for every reduction (implementation choice, applies at all sizes) | +1.5 … +3.5 vs the custom RS |
| **merged front + prefill path** | see §3 crossovers | — |

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
| **quantize-then-allgather** (`ag_qpush`: sender-side MXFP8, fp8 wire = half the AG payload) | bit-exact vs the receiver-side fused AG+quant, but **loses at every size** — 4.9→5.5 us @8 up to 6.0→8.1 @80 (tmp_fc1_ag_duel, grids 8/32/64 tuned). The [B,3584] gather is latency-bound through B=80; halving payload bytes buys nothing while the double completion (payload+scales sentinels) adds fixed cost. Kept as an autotune candidate (never picked); receiver-side fused AG+quant stays the fc1-shard workhorse |

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

TP8 validation (Aug-5 retune, steps 7a-c — note these ran BEFORE the
routing/overlap/tail gates landed, so their 64/80 numbers are stale):
DisAgg decode = the b10 opt numbers. Agg decode +6% @64 / +1% @80
(shard tail everywhere dodged the then-broken ref-tail-at-64). Agg +
one-time fc2 weight AG: matched b10 at ≤32 but −5/−8% at 64/80 (ref
tail at 64 — now skipped by default). **Agg prefill at TP8 is BROKEN
on this node: −11% @1k, −62% @4k, −4% @8k** — do not deploy the
sharded prefill path here before re-tuning `PREFILL_MIN_TOKENS` /
investigating the 4k cliff (the old-node TP8 numbers were
+12/+18/+22% at 512/1k/2k+; strongly node-dependent).

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

## 9. Full ablation (c8, B=1..16, PRE-retune — kept for the small-B
switch-off costs; the B=1..80 post-retune ablation lives in
`results/experiments/retune_aug5/` and `results/bench_moe_kimi_k3_tp8.md`)

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
| route_pack | 6.5 | ~3–4 | radix top-16 (serial 16-max chain today); above 64 tokens we now fall back to noaux entirely |
| AG+quant | 7.3 | ~4–5 | sender-side fp8-wire push tried (`ag_qpush`) — measured SLOWER at all sizes (§5); the remaining gap is the Lamport rendezvous, not payload |

B=8 needs ~15 us and B=16 ~14 us below the current opt: only the
expert-GEMV class win covers that; everything cheaper is already in.

## 11. B=1..80 retune (Aug 5, late) — DONE; decisions shipped as code defaults

The full-range retune ran GPU-clean with locked clocks
(`run_retune_b80.sh`: AG duel, forced extremes, deploy classes,
traces; then `run_combo_b80.sh` E1–E5 and the strategy-hunt combos).
Decisions, all now code defaults:

1. **fc1 shard: always on** (`FC1_SHARD_MAX_TOKENS` default 1e9).
   Forced-extremes beat the old ≤16 gate at every size; no small-B
   regression (§4).
2. **routing "ours" gated at 64** (`ROUTING_OURS_MAX_TOKENS=64`,
   batch-gated in `forward`, graph-capture-safe): +16 us at 32 but
   −8 at 80 vs stock noaux.
3. **overlap gate 64 → 96** (`B10_OVERLAP_MAX_TOKENS`): worth 19 us
   at B=80; the old gate came from a hot-routing config.
4. **ref tail skipped at exactly B=64** (`REF_TAIL_SKIP=64`): the
   shard tail wins 115.3 vs 118.0 there (twice-confirmed) while the
   ref tail wins at 80 by ~7 us vs shard, ~21 vs fat-AR.
5. **merge3 gated at 64** (`MERGE3_MAX_TOKENS`): −2.7 us at 80.
6. **quantize-then-AG rejected** (§5 `ag_qpush` row): the fc1 AG is
   latency-bound through B=80; receiver-side fused AG+quant stays.
7. **`quant_slice_mxfp8`** covers the quantize fusion on non-shard
   paths (now only the routing="noaux" B=80 path and ablations).

Result: §1's table — positive at all 8 sizes, B=32 (primary) +15%,
weakest point B=64 +9%. Open items: B=64 (needs the expert-GEMV
kernel), the agg-prefill regression (§6 warning), and the B=16
autotune flap.

## 12. Node-dependence warning

Every crossover here was RE-measured on this box and several flipped
vs the box `README.md` was written on (collectives are much cheaper
here): ref-tail moved B≥4→B≥16, the custom RS went from tie to loss,
ONESHOT beats AUTO for the baseline. All thresholds are env-exposed
(`B10_REF_TAIL_MIN_TOKENS`, `B10_FC1_SHARD_MAX_TOKENS`,
`B10_TAIL_KERNEL`, …) — re-run `--ablate` after moving hardware.
