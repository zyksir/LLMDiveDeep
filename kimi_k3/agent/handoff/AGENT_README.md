# Kimi-K3 B300 optimization — agent handoff

You are continuing GB300/TP8 optimization of the Kimi-K3 MoE layer. Read this
top to bottom before running anything. Everything here was established on the
original node (8×GB300); the numbers need re-measuring on yours, but the code,
the bugs, and the method transfer directly.

Two repos:
- **LLMDiveDeep** (`/node-storage/LLMDiveDeep`, this repo) — the light
  experiment harness. Per-kernel benches, the MoE layer, the ablation, the
  strategy search. Develop and measure here first.
- **trt-llm** (`/node-storage/trt-llm`, branch `optimized/k3` @ `165cdc3dcf`) —
  production. The fork edits are delivered as a patch (see §3); apply it there.

---

## 1. The node is a GB300, not what nvidia-smi says

`nvidia-smi` reports **"NVIDIA L20D / Ada / cc 8.9"** on this hardware. It lies.
The CUDA runtime reports **cc (10,3), 148 SMs, 268 GB, NVLink-5** — it is a
**B300/GB300 (`sm_103a`)**. ALWAYS derive arch from
`torch.cuda.get_device_capability()`, never from the device name.
`common/arch.py` does this correctly. If your node is a real B200 (cc 10,0),
everything still works — the code is arch-generic — but retune (§6).

## 2. Containers

The layer benches (LLMDiveDeep) run in the stock NGC image:
```
docker run -d --name trt-dev --gpus all --ipc=host --network host --shm-size=32g \
  -v /node-storage:/node-storage -v /node-storage/var/cache:/root/.cache \
  -w /node-storage/LLMDiveDeep nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23 sleep infinity
```
**Serving (trt-llm `optimized/k3`) needs the fork's C++ built** — it adds ops
the stock wheel lacks (`turn_start_pair`, `moe_finalize_allreduce_rmsnorm_concat`,
a wider `KVCacheManager` ctor). Either use the fork-built prod image
`baseten/dynamo-cache-aware-routing:trtllm-165cdc3dcf-...` (needs docker login),
or build from source in the rc19 container:
```
docker run -d --name trt-k3 --gpus all --ipc=host --network host --shm-size=32g \
  -v /node-storage:/node-storage -v /node-storage/var/cache19:/root/.cache \
  -w /node-storage/trt-llm nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19 sleep infinity
docker exec -w /node-storage/trt-llm trt-k3 bash -c \
  "TORCH_CUDA_ARCH_LIST=10.3a python3 scripts/build_wheel.py -a 103-real -j 200 --skip-stubs --fast_build --trt_root /usr/local/tensorrt"
# the wheel-packaging step may fail; the .so's under cpp/build ARE built — stage them
# into dist-packages/tensorrt_llm/{libs,nanobind,...} (see B300_RETUNE.md §container).
pip install orjson                          # fork adds this dep
pip install --no-deps "flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer.git@v0.6.18rc1"
```
**flashinfer must be 0.6.18rc1** (the `requirements.txt` pin) — it has
`ActivationType.Situ`; the images ship 0.6.12/0.6.15 which do NOT, and SiTU
experts then fail at construction. Assert `flashinfer.__version__` in CI.

Overlay the fork's Python onto the wheel (the ConfigMap-wrap pattern): copy
`tensorrt_llm/**/*.py` + the `kimi_k3_optim` `.cu/.cuh/.h` into
`dist-packages/tensorrt_llm/`, leaving the compiled `.so`s in place.

## 3. Apply the fork patch

`optimized_k3_b300.patch` (beside this file) applies cleanly on
`optimized/k3` @ `165cdc3dcf`:
```
cd /node-storage/trt-llm && git checkout optimized/k3 && git apply --check \
  /node-storage/LLMDiveDeep/kimi_k3_layer/agent/handoff/optimized_k3_b300.patch
git apply /node-storage/LLMDiveDeep/kimi_k3_layer/agent/handoff/optimized_k3_b300.patch
```
It carries 14 files (9 edited, 5 new). What it does, by category — every item
is a real bug fixed on GB300:

**Arch portability (the dominant bug class — B200-baked flags fatal on B300):**
- `b10_kernels/_arch.py` (NEW): single arch authority for the fork, mirrors
  `common/arch.py`. Honours `B10_FORCE_SM`.
- `routing_radix.py`: derived `SGL_CUDA_ARCH`/`TVM_FFI_CUDA_ARCH_LIST`/name
  (was hardcoded 1000/10.0a/`_sm100`); arch added to the cache key.
- `dual_out_gemm_cutedsl.py`: `--gpu-arch` from `_arch`; compile cache key
  carries arch; `_TUNED` arch-keyed; `_bucket` cap 128→256.
- `b10_multimem.py`: gencode + module name from `_arch` (was `sm_100a`).
- `config.py`: `MOE_INTER 384→3072`, `NUM_SHARED_EXPERTS 16→2` (HF config).

**Serving integration (why layer wins didn't compose):**
- `serving.py`: guard generalized to TP {2,4,8}, names the failing predicate,
  and rejects head-tiling-incompatible degrees (TP4: 24 heads/rank); decode
  path ENABLED (`KIMI_B10_DECODE_MAX_BATCH`, default on); decode-dim AR-map
  seeds + allreduce_norm seeds; col-AG engine sized (`col_ag_max_output_columns`).
- `moe_layer.py`: expert backend arch+SiTU gated (native only on sm_100 or
  non-SiTU) at BOTH prefill and **decode** call sites (the decode one was a
  hardcoded NATIVE default → `No kernel found ... mEltwiseActType:2` on
  sm_103+SiTU); capabilities probe + `filter`; col-AG pre-tune per bucket.
- `collectives.py`: `KIMI_B10_COMM_FALLBACK={trtllm,nccl}` kill switch; fused
  `allreduce_norm` via trt; fused col-AG (`col_quant*` ported from LLMDiveDeep).
- `capabilities.py` (NEW): per-stage kill switches (`KIMI_B10_DISABLE_STAGES`),
  rank-agreed.
- `col_quant.py`, `col_quant_cuda.py`, `col_quant_backend.py` (NEW): the fused
  column-AG(+MXFP8) engine, ported from LLMDiveDeep.
- `setup.py`: package the b10 JIT sources (else wheel warmup can't find them).
- `cpp_custom_ops.py`: register the fork-only fake op conditionally (so
  `import tensorrt_llm` works on any wheel).

## 4. THE TASK — run this

Everything is staged in `/node-storage/var/run_b300_tp8_program.sh` (copied into
this handoff dir too). It **gates on a genuinely free node** (all GPUs <10 GiB —
DO NOT run against a shared/occupied node, the numbers are invalid and it OOMs),
locks clocks, then runs in order:
1. **TP8 contribution table, bs 1..64** — the headline. One row per idea
   (fc1 shard / fc2 shard / merged front / comm / routing), `step_saved_us` each.
2. TP8 full layer table (`--sizes all`).
3. Crossover sweeps: `decode_tail` @16..128, `decode_front` @32,64 — to EXPLAIN
   why a strategy wins at some sizes and not others.
4. Profiled pairs at B=8,B=64 (baseline+opt chrome traces) — review each strategy
   for unnecessary copies / kernels off their SOL.
5. TP4 ablation bs 1..64 — kernels MUST stay TP4-capable (not everyone has 8 GPUs).

```
nohup /node-storage/var/run_b300_tp8_program.sh > /node-storage/var/program.log 2>&1 &
```
Watch `program.log` for `NODE FREE`, per-stage `EXIT=`, `Wrote`, `PROGRAM DONE`.
Outputs: CSVs in `kimi_k3_layer/local_results/`, traces alongside.

## 5. The dev loop (follow it; it is in the skill)

1. bench each kernel (`kernels/benchmarks/bench_*.py`, `communication/bench_comm_graph.py`,
   `communication/kernel_benchmarks/bench_*.py`) — reproduce the report number.
2. no report on this node/arch/width? produce one first.
3. most kernels must BEAT their open-source/native baseline; if not, record why.
4. `bench_b10_kimi_k3_moe_layer.py` — whole layer vs the report table.
5. migrate to trt-llm, get the E2E number. Serving must re-establish every
   invariant the bench provides (autotuned comm map, `tune_col_ag`, fused
   dispatch) — a layer win does NOT compose for free.
Develop small (4-layer, TP2, dummy weights) then validate full (TP8, situ).

## 6. Open questions / where to push next

- **The headline gains at real width are ~+11–38% (decode compresses to ~+11%
  at bs 32–128).** Reason: the optimizations save a near-constant ~30 µs/layer,
  but the checkpoint's experts are 8× wider than the old 384 tables assumed, so
  the baseline denominator grew — the % shrank, the µs held. To beat ~15% at
  mid-batch you must attack the EXPERT GEMM itself: `kernels/specs/
  moe_expert_gemv_spec.md` is the unimplemented spec. This is the highest-value
  open kernel.
- **fc1-shard goes negative at TP4/bs=64** (−12.6 µs) but the very next rung
  (fused front) is +12.4 — a ladder-order artifact (sum ≈ 0), not a regression.
  Confirm at TP8 (was +35 µs in every TP8 table). The gather kernel itself is
  fine (16.3 vs 73.7 µs standalone).
- **comm skeleton is negative at TP4** (collectives cheap at low degree),
  positive at TP8. Degree economics — document, don't "fix".
- **Strategy selection must be reusable across nodes.** The thresholds
  (`DECODE_MAX_TOKENS`, `SHARDED_FC2_MAX_TOKENS`, `PREFILL_*`) are measured, not
  derived. Re-measure via the sweeps in §4.3 and update `measured_config`.
  `DECODE_MAX_TOKENS` was moved 256→128 at real width (decode loses at 256,
  prefill wins) — verify on your node.
- **b10_permute_v1_pdl** cuts the routing region 25–47% at B≤256 (loses ≥1024).
  Not yet promoted into the layer — needs bypassing run_moe's internal permute.

## 7. Gotchas that cost hours

- **Warm-up**: situ fused-moe JIT ≈ **913 s**, FMHA ≈ 61 s/process. Prebuild via
  `serving_startup/prebuild_jit.py` (arch-selected, covers the radix + comm
  targets now) and persist `~/.cache/flashinfer`. Put caches on LOCAL disk —
  `/root` here is JuiceFS @ 0.26 GB/s and builds stall in D-state; use
  `/node-storage/var/cache`.
- **Memory at real width**: the ablation's `init_weights` stages expert tensors
  on HOST (fixed) — do not revert it, the GPU-side int64 randint is 73.5 GiB.
- **Shared node = invalid numbers.** If any GPU shows held memory you don't own,
  the run is contaminated; the program gates against it. Do not squat/OOM other
  jobs to clear it — reset via host (`systemctl restart nvidia-fabricmanager;
  nvidia-smi --gpu-reset`) or a scheduler.
- **Harness honesty**: health-poll loops must FAIL on timeout (not fall through
  to "UP"); stream events that are errors are NOT tokens; persist per-wave
  results. All fixed in `/node-storage/var/*.sh` — reuse them.

## 8. Env-flag reference (all default to the optimized path)

- `TRTLLM_KIMI_B10_MOE=1` master on; `=0` stock layer.
- `KIMI_B10_DECODE_MAX_BATCH` decode cap (default `DECODE_MAX_TOKENS`); `=0`
  prefill-only.
- `KIMI_B10_DISABLE_STAGES=fused_front_cute,radix_routing,multimem_tail,...`
  per-stage rollback.
- `KIMI_B10_COMM_FALLBACK=trtllm` (stock AllReduce AUTO) or `=nccl` (all NCCL).
- `B10_FORCE_SM=10.3` force arch for GPU-less builds.

Full evidence and the reproduction commands: `../B300_RETUNE.md`.
