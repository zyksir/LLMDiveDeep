# RUNBOOK — re-validating the Kimi-K3 layer work on a fresh node

> **B300/GB300:** see [B300_RETUNE.md](B300_RETUNE.md) — retuned
> thresholds, per-kernel B300 reports, and the serving-composition fixes
> (2026-08-21). The sections below predate it and are B200-specific.

Everything below is ready to run. The Aug-5 B=1..80 retune, the
regime simplification, AND the evening validation pass
(`moe_optimization.md` Appendix C.1–C.4) completed GPU-clean on the
`model-performance` box; the shipped code defaults embody those
decisions. Resolved Aug 5 evening: the tail-fork merge validated
(§1 artifacts regenerated), deploy classes re-checked (agg needs
`B10_AGG_FULL_FC2=1` for B>32), and the prefill regression
root-caused and FIXED (`_ar_prefill` AUTO; `moe_optimization.md`
§3 — +5…+33% at 512–8k on the latest idle-node re-run; earlier
+30…+56% readings had a degraded baseline column). Remaining for
a next session: §5
kernel cycles (expert GEMV) and the TP4 `PREFILL_MIN_TOKENS`
decision (§3 below).

## 0. Node setup (once)

```bash
# container (caches JIT builds across restarts via /root/.cache)
docker run -d --name trt-dev --gpus all --ipc=host --network host \
  --shm-size=32g -v /workspace:/workspace -v /root/.cache:/root/.cache \
  -w /workspace/<path>/LLMDiveDeep \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23 sleep infinity
# measurement hygiene - both are LOAD-BEARING (±8 us drift unlocked)
nvidia-smi -pm 1 && nvidia-smi -lgc <max_sm_clock>,<max_sm_clock>
nvidia-smi   # verify 0 MiB / 0 % on ALL GPUs before every run
```

First run JIT-builds `k3_comm_cuda` (~5 min) and flashinfer
`fused_moe_trtllm_sm100` (~25 min) — one-time, cached.

## 1. MoE: full re-tune at the REAL target sizes (B = tokens/step)

> An optional local campaign driver may live at
> `local_debug/kimi_k3_run_retune_b80.sh`; it runs §§1–3 plus an
> optional historical AG duel (qpush-vs-receiver-side per B, with
> bit-exactness asserted), fc1-shard ALWAYS/NEVER, ref-tail
> extremes, traces, and the final figure regen.
>
> **RESOLVED on the model-performance box (Aug 5)** — decisions are
> code defaults now; see `moe_optimization.md` §2 and Appendix C: fc1 shard / overlap / merged GEMM ALWAYS on, one
> regime boundary `SMALL_BATCH_MAX_TOKENS=32` (routing ours + merge3
> + AG-fused quant below, stock above), tail fork at
> `REF_TAIL_MIN_TOKENS=16`, qpush rejected. Logs:
> `local_result/experiments/retune_aug5/`. Re-run this section only after
> hardware or kernel changes.

Targets: **1, 2, 4, 8, 16, 32, 64, 80 — 32 is the most important.**
The shipped defaults were tuned across the full 1–80 range on the
old box; on a new node re-decide the two thresholds from this run:

```bash
docker exec trt-dev bash -c "cd .../LLMDiveDeep && \
  BENCH_ALLREDUCE_STRATEGY=ONESHOT mpirun -x BENCH_ALLREDUCE_STRATEGY \
   -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
   --sizes 1,2,4,8,16,32,64,80 --ablate"
```

Decision guide (read the ablation columns at 32/64/80):

ONE threshold exists (post-simplification, moe_optimization.md
Appendix C); everything else is unconditional:

| threshold (env) | decided by | prior knowledge |
|---|---|---|
| `B10_SMALL_BATCH_MAX_TOKENS` (regime: ours routing + merge3 + AG-fused quant + shard tail vs stock + ref tail) | `-routing`/`-merge3`/`-reftail` columns at 16/32/64/80 | this node: 32 (ours −8 us at 80; large regime beat every small-regime variant at 64; tail fork within ~2 us of the boundary on both sides) |

(`B10_REF_TAIL_MIN_TOKENS` decouples the tail fork from the boundary
if a node's crossover genuinely detaches — the old node measured 4;
`B10_FC1_SHARD_MAX_TOKENS` / `B10_FC2_SHARD_MAX_TOKENS` still exist
for ablation forcing.)

If retained locally, run the optional honest-baseline sweep once
(`local_debug/base_sweep.sh`) — ONESHOT beat AUTO by 2–8 us on the
last box; confirm per node.
NEVER take headline numbers from a `--profile` run (CUPTI taxes every
later kernel ~+7 us); capture traces in a separate final run.
One-off absurd rows (an 830 us baseline once) = re-run that size.

## 1a. RESOLVED-REJECTED: prefill shared-down reorder (Aug 5)

Hypothesis (from historical optional local wire-bandwidth evidence):
delay the shared DOWN GEMM into the latent-AR window, ~+30/60 us at
4k/8k. Clean-node A/B REFUTED it: −77/−141/−262/−403 us at
S=1024/2048/4096/8192. Root cause in the 4k traces: the NCCL LDMC
AR kernel is SM-resident; the delayed GEMM held the SMs at its
launch point and the AR went 88 → 163 us. Reverted — the shipped
schedule keeps the whole shared chain EARLY (over routing topk +
expert metadata, which are latency-bound). Rule: never schedule
compute into an AR window; SM-free AR (CE/multimem) is the only way
to recover that time. Details: moe_optimization.md §3.

## 1b. RESOLVED: always-ref rejected; fc2 TRANSPOSED; tail fork
## merged into the boundary (moe_optimization.md Appendix C.3)

The three-arm run (clean GPUs) killed always-ref: 5.5–7.8 us
structural loss at B ≤ 8 (the full 51 MB fc2 read vs the 6.4 MB
shard, unhideable at tiny B; `fin_fuse` innocent). The memory saving
shipped anyway via the fc2 TRANSPOSE: one `[3584, 7168]` tensor, the
per-rank shard is a contiguous row view, the `_fc2_local` clone is
deleted, and cuBLAS NN == TN at every size
(optional `local_debug/kimi_k3_fc2_transpose_probe.py` evidence). The
CuTeDSL tail GEMM is unwired
(needs K-contiguous weights; ≤0.8 us edge, negative at 64 — re-wire
after a K-major respec, §5 queue). `REF_TAIL_MIN_TOKENS` now
defaults to boundary+1 → ONE batch threshold total.

Still to run on a clean node (~1 min): the post-merge validation
sweep — expect ~+2 us at B=16 and ~+1 at B=32 vs the §1 table,
identical elsewhere:

```bash
docker exec trt-dev bash -c "cd .../LLMDiveDeep && \
  BENCH_ALLREDUCE_STRATEGY=ONESHOT mpirun -x BENCH_ALLREDUCE_STRATEGY \
   -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
   --sizes 1,2,4,8,16,32,64,80"
# and the tail-fork sanity arm (decoupled fork at the old 16):
... B10_REF_TAIL_MIN_TOKENS=16 mpirun -x B10_REF_TAIL_MIN_TOKENS ... \
    --sizes 16,32
```

## 2. GEMM-vs-batch sweep (informs §1's crossovers)

```bash
CUDA_VISIBLE_DEVICES=0 python3 local_debug/gemm_bs_sweep.py
```

Measures where each layer GEMM ([bs,7168]x[7168,3584] etc.) stops
being weight-read-flat and starts scaling with bs.

**HISTORICAL: `quant_slice_mxfp8`** (removed Aug-19 together with the
old `moe_kimi_k3.py` layer): ONE Triton pass turned the old merged
GEMM's strided latent slice into (e4m3, ue8m0) bit-exactly vs
`mxfp8_quantize` — 1.1-1.4 us vs 1.9-4.6 for copy+quantize at
bs=1..80, no collective. The current layer
(`b10_kimi_k3_moe_layer.py`) has no strided-slice quantize site: its
fronts either go through `all_gather_col_quant` (the same MXFP8
recipe fused into the AG epilogue, `communication/kernels/
comm_cuda.py`) or quantize a contiguous full latent with the stock
trtllm op, so the kernel was deleted as redundant.

**The decision rule all shard crossovers follow** (sharding = trade
communication for computation): shard wins iff
`saved_compute(bs) > comm_cost(bs)`.
- weight-bound regime (small bs): saved compute is FLAT
  (weight_bytes/world/HBM_bw, ~5-6 us for fc1) while comm grows
  ~linearly in bs -> shard pays only below a crossover (~16 here).
- middle bs (~32-128 decode): DEAD ZONE - compute saving still flat,
  comm has outgrown it, GEMM not yet compute-bound -> sharding is a
  pure loss; drop fc1/fc2 shards and keep only flat-cost
  optimizations (routing, merge3, overlap, tail choice).
- compute-bound regime (prefill): saved compute grows with tokens
  (real 1/world FLOP cut) and comm is hideable -> shard wins again.
The bs where this sweep's times stop being flat marks where the
third regime begins for each GEMM.

## 3. Unified MoE strategy (TP8)

```bash
# One B10 layer automatically selects decode, baseline, or SP prefill.
... bench_moe_kimi_k3.py --sizes 1,8,16,32,64,80,512,1024,4096,8192

# Report-only one-axis search; does not update defaults or configuration.
python3 kimi_k3_layer/search_best_strategy.py --world 8 \
  --sizes 1,8,16,32,64,80,512,1024,4096,8192
```

Validated Aug 5 (TP8, clean node): the full-fc2/ref-tail decode
strategy is +30/+14/+12% at B=4/16/64. Prefill is +5…+33% after
the `_ar_prefill` fix
(`moe_optimization.md` §3; measure both columns together — the
baseline AR strategy is node/state-dependent). Remaining open item: the TP4 smoke
tests predate that fix (they read −2/−6% at 1k/4k) — re-run at TP4
before deciding `PREFILL_MIN_TOKENS` there.

## 4. KDA

```bash
# unified decode/prefill layer (single GPU): automatic boundary at 128 tokens
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/bench_b10_kimi_k3_kda_layer.py \
  --shards tp8 --token-sizes 1 2 4 8 16 32 64 128 4096 8192 16384
# 3-way kernel duel (registry rows now exist: fi_recurrent_kda,
# flashkda_ptx_int21 - INT21 builds from github.com/Int21-AI/KDA-B200,
# clone to ../kda_b200_install + pip install -e . --no-build-isolation)
cd linear_attn/benchmarks && CUDA_VISIBLE_DEVICES=0 python3 \
  bench_kda_prefill.py --heads 12 --batch-sizes 1 \
  --seq-lens 4096 8192 16384 \
  --backends fi_recurrent_kda flashkda_ptx_int21 b10_kda_chunk_prefill
# MEASURED Aug 5 (GPU-clean, b10-canary verified): b10 190/376/739 beats
# INT21 266/517/1016 by ~29% at H=12 (and ~41% at H=96: 843 vs 1443).
# flashinfer PR #4262 CANNOT be tested on 0.6.15 - the wheel predates
# the merge (its recurrent_kda is the old generic path: slow AND wrong
# for prefill). To duel the real CAKE kernels: install flashinfer from
# main in an ISOLATED venv (do NOT upgrade the container's pinned
# 0.6.15 - it underpins the whole MoE stack), then rerun the command
# above; the fi_recurrent_kda registry row is contract-correct
# (cu_seqlens, pre-sigmoided beta, bf16 [N,HV,V,K] state).
```

## 5. CuTeDSLGen improvement cycles (the kernels that move the big targets)

Repo: `../CuTeDSLGen` (symlink `/workspace/CuTeDSLGen` may need
recreating). **Gotchas that burned a night**: `export IS_SANDBOX=1`
(claude CLI refuses `--dangerously-skip-permissions` as root without
it), trust the workspace in `/root/.claude.json`, use the
`claude_429_retry.sh` wrapper via `generation.run` for rate-limit
resilience, and don't run agent arms concurrently with TP8 benches
(they grab GPUs; killing them can wedge GPUs — see top).

Queue, by expected value:
1. **moe_expert_gemv** (`evaluation/decomposition_ab/moe_expert_gemv_spec.md`)
   — per-ACTIVE-EXPERT grouping this time (near-misses documented in
   moe_optimization.md Appendix A). This is the kernel for the B=8–80 targets.
2. **kda_chunk_fused** (`kda_chunk_fused_spec.md`) — must keep the
   INT21 schedule AND fuse the prologue (first attempt: correct but
   1.28 ms vs incumbent 0.74 at 16k). May be obsoleted by §4's
   flashinfer kernel — measure that first.
3. attn_res respec with PDL chaining (in-layer regression documented
   in kda_optimization.md).

## 6. Wedged-GPU recovery (what happened on the old node)

Kill-9'ing processes mid-CUDA-kernel (agent arms, mpirun ranks) can
leave orphaned contexts with a spinning Lamport kernel: 100% util,
full memory, invisible to `nvidia-smi --query-compute-apps`, survives
container restarts, and per-GPU reset is "Not Supported" on NVSwitch
boxes. Only fix: host reboot. Prevention: stop benches with the
harness (let ranks exit), never `pkill -9` a rank mid-graph-replay.
