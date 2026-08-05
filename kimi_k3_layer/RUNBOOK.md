# RUNBOOK — re-validating the Kimi-K3 layer work on a fresh node

Everything below is ready to run; the Aug-5 numbers in
`moe_optimization.md` / `kda_optimization.md` came from the
`model-performance` box, which ended the session with GPUs 0–3 wedged
(orphaned CUDA contexts, `nvidia-smi -r` unsupported on NVSwitch —
host reboot required) and GPUs 4–6 taken by other jobs, so the
B=32/64/80 and TP4 deployment numbers are INCOMPLETE/VOID. Run this
list top-to-bottom on the new node.

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

> **One-command version**: `kimi_k3_layer/run_retune_b80.sh` (host
> side) waits for idle GPUs, locks clocks, then runs §§1–3 plus the
> AG duel (`tmp_fc1_ag_duel.py`: qpush-vs-receiver-side per B, with
> bit-exactness asserted), fc1-shard ALWAYS/NEVER, ref-tail extremes,
> traces, and the final figure regen. Log: `/tmp/retune_b80.log`.
>
> **RESOLVED on the model-performance box (Aug 5)** — decisions are
> code defaults now; see `moe_optimization.md` §11: fc1 shard ALWAYS,
> routing "ours" ≤64, overlap ≤96, ref tail 16+ except 64, merge3
> ≤64, qpush rejected. Logs: `results/experiments/retune_aug5/`.
> Re-run this section only after hardware or kernel changes.

Targets: **1, 2, 4, 8, 16, 32, 64, 80 — 32 is the most important.**
The shipped defaults were tuned for 1–16; every threshold pivots
inside 16–80 and must be re-decided from this run:

```bash
docker exec trt-dev bash -c "cd .../LLMDiveDeep && \
  BENCH_ALLREDUCE_STRATEGY=ONESHOT mpirun -x BENCH_ALLREDUCE_STRATEGY \
   -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
   --sizes 1,2,4,8,16,32,64,80 --ablate"
```

Decision guide (read the ablation columns at 32/64/80):

| threshold (env) | decided by | prior knowledge |
|---|---|---|
| `B10_REF_TAIL_MIN_TOKENS` (tail: shard vs ref+finfuse) | `opt` vs `-reftail` vs `-finfuse` | old node: ref wins 32–64; this node at ≤16: shard wins <16 |
| `B10_FC2_SHARD_MAX_TOKENS` (shard tail vs fat AR) | `-fc2shard` | fat AR won at 128 (old); one-shot AR wire = (w−1)·N — grows fast |
| `B10_FC1_SHARD_MAX_TOKENS` | `-fc1shard` | wash at 32 on the old node; AG wire ~7 KB/token |
| `B10_OVERLAP_MAX_TOKENS` | `-overlap` | helped through 64 (old); check 80 |
| `B10_ROUTED_SPLIT_MIN_TOKENS` | `-merged` at 64/80 | split ≥64 (old) |

Also run the honest-baseline sweep once (`debug/base_sweep.sh`) —
ONESHOT beat AUTO by 2–8 us on the last box; confirm per node.
NEVER take headline numbers from a `--profile` run (CUPTI taxes every
later kernel ~+7 us); capture traces in a separate final run.
One-off absurd rows (an 830 us baseline once) = re-run that size.

## 2. GEMM-vs-batch sweep (informs §1's crossovers)

```bash
CUDA_VISIBLE_DEVICES=0 python3 debug/gemm_bs_sweep.py
```

Measures where each layer GEMM ([bs,7168]x[7168,3584] etc.) stops
being weight-read-flat and starts scaling with bs.

**NEW since the sweep: `quant_slice_mxfp8`** (kimi_k3_layer/
quant_slice.py, wired behind `B10_QUANT_SLICE`, default on): ONE
Triton pass turns the merged GEMM's strided latent slice into
(e4m3, ue8m0) BIT-EXACTLY vs `mxfp8_quantize` - 1.1-1.4 us vs
1.9-4.6 for copy+quantize at bs=1..80, NO collective. This decouples
the quantize fusion from the fc1 shard: in the SS1 ablation,
`-fc1shard` now keeps the fusion, so read that column as the pure
comm-vs-compute shard verdict. Expect fc1 shard OFF at 32-80 (and
possibly everywhere) with no fusion loss.

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

## 3. Deployment classes (TP8)

```bash
# decode-only engine (expect = the §1 opt numbers)
... bench_moe_kimi_k3.py --moe-class disagg --sizes 1,2,4,8,16,32,64,80
# aggregated engine: decode (shard tail everywhere) + prefill
... bench_moe_kimi_k3.py --moe-class agg --sizes 1,8,16,32,64,80,512,1024,4096,8192
# agg + one-time fc2 weight all-gather (unlocks the ref tail)
B10_AGG_FULL_FC2=1 ... --moe-class agg --sizes 16,32,64,80
```

Open item from TP4 smoke tests: agg-prefill read −2/−6% at 1k/4k at
TP4 — decide `PREFILL_MIN_TOKENS` (or whether the sharded prefill
path pays at all) from the TP8 numbers, per node.

## 4. KDA

```bash
# decode layer (single GPU): baseline vs b10, B=1..80 now
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/bench_kda_kimi_k3.py \
  --shards tp8 --batch-sizes 1 2 4 8 16 32 64 80
# prefill layer: baseline (with production D2H syncs) vs b10+glue
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/bench_kda_prefill_layer.py
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
   moe_optimization.md §5). This is the kernel for the B=8–80 targets.
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
