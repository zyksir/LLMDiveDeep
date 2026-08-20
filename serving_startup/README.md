# Serving bring-up time: where KimiK3's ~30 minutes goes

Why a KimiK3 replica takes ~30 minutes to serve its first token, which parts of
that are parallelizable, and how much each fix is worth.

Code read: `/node-storage/trt-llm` @ `b7e8269` (Baseten fork, 1.3.0rc23-class).
Measurements: 8×B200 node, 252 cores, 2.8 TB RAM, container
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23`.

**Status of the evidence.** Everything marked *(measured)* was measured on this
node and is host- or compile-side. The GPU-resident phases (H2D copy rate,
warm-up, graph capture) are **not yet measured** — this node's NVLink fabric is
stuck in `In Progress` and `cuInit` returns 802, so no CUDA context can be
created. The harnesses for those phases are in this directory and ready to run.

## 1. Summary

Bring-up is two serial halves — **get the weights onto the GPUs**, then **warm
up** — and both are slower than they need to be for different reasons.

| # | Finding | Evidence | Worth |
|---|---|---|---|
| 1 | ~44 min of kernel JIT compile is paid at first use, in the serving process | *(measured)* cold builds below | up to ~44 min on a cold node, → 0 |
| 2 | `FLASHINFER_NVCC_THREADS` defaults to 1 and nothing passes `--split-compile`, so one `ptxas` runs ~10 min while 251 cores idle | *(measured)* 586 s → 129 s | 4.5× on the compile that remains |
| 3 | Checkpoint read from the network FS is the floor: ~0.26 GB/s single-stream, ~0.9 GB/s saturated | *(measured)* | K3 is 7–26 min of pure I/O |
| 4 | The parallel read pass silently disables itself when the checkpoint approaches host RAM, dropping to single-stream faulting | `hf/weight_loader.py:86` | 3.5× on the dominant phase |
| 5 | KimiK3/DSv3 copy weights **single-threaded** while the generic path uses a thread pool | `modeling_deepseekv3.py:486` | unquantified, ≥1.3× host-side |
| 6 | Weight H2D copies are `non_blocking=True` from **pageable** mmap memory, i.e. effectively synchronous | `fused_moe/quantization.py:625` | 2–4× on the H2D leg |
| 7 | KV-cache estimation builds a full executor, warms it up, captures graphs, then throws it away | `_util.py:770` | one whole warm-up |
| 8 | Every parameter has its final CUDA address **before** the first checkpoint byte is read | `model_loader.py:430` | makes load ∥ warm-up possible at all |

Ordered by (value ÷ risk), the work is:

1. **Bake the JIT into the image** (finding 1+2). No correctness risk, no GPU
   needed to build. Biggest single win.
2. **Skip the KV estimation pass** (7) by setting `kv_cache_config.max_tokens`.
   Config-only.
3. **Always stream in parallel, into pinned staging** (4+6), and thread the
   K3/DSv3 module walk (5).
4. **Read once per cluster, fan out over the fabric** (3). Architectural, and the
   only thing that beats the per-node storage ceiling.
5. **Overlap warm-up compile with weight streaming** (8). Highest complexity;
   do it last, and keep the autotuner out of the overlap.

## 2. What bring-up actually does, in order

`LLM(...)` → `create_py_executor` (`pyexecutor/py_executor_creator.py:527`).
The phases are a genuine sequence, so they are numbered:

| # | Phase | Code | Needs weight *values*? |
|---|---|---|---|
| 1 | `import tensorrt_llm` | — | no |
| 2 | config load + validation | `_load_and_validate_config` | no |
| 3 | construct model on `meta` device | `model_loader.py:384` `MetaInitMode()` | no |
| 4 | **allocate every parameter as an empty CUDA tensor** | `model_loader.py:430-470` | no |
| 5 | checkpoint read → host page cache | `hf/weight_loader.py:162` `prefetch_files` | — |
| 6 | per-module walk: slice + H2D copy | `modeling_deepseekv3.py:486` (K3/DSv3), `modeling_utils.py:1104` (generic) | yes |
| 7 | `post_load_weights()` layout transforms | `fused_moe/*`, quant methods | yes |
| 8 | KV-cache capacity estimation: build a throwaway `PyExecutor` (which warms up and captures graphs), run dummy inference, tear it down | `_util.py:770-835` | yes |
| 9 | real `PyExecutor` → `model_engine.warmup()` | `py_executor.py:787` | yes |
| 9a | attention warm-up | `model_engine.py:1313` | yes |
| 9b | general warm-up (torch.compile specialization) | `model_engine.py:1319` | no¹ |
| 9c | autotuner warm-up | `model_engine.py:1332` | no¹ |
| 9d | CUDA-graph capture | `model_engine.py:1339` | no¹ |
| 9e | max-shape memory-pool pre-population | `model_engine.py:1343` | no |

¹ needs allocated buffers at final addresses, not correct values — see §7.

Two structural facts fall out of this table:

* **Phase 4 precedes phase 5.** Parameters have their final CUDA addresses
  before any checkpoint byte is read. Phases 9b/9d depend on shapes, addresses
  and kernel selection, not on values. That is the opening for overlap.
* **Phase 8 duplicates phase 9.** The estimation pass constructs an executor,
  warms it up including graph capture, and discards it. Setting
  `kv_cache_config.max_tokens` explicitly skips the whole pass.

## 3. Measured: ~44 minutes of JIT compile — over half of it a download nobody ran

In a fresh container the MoE kernels this stack calls are JIT-compiled on first
use. **The most useful fix is not compiling faster but not compiling at all:**
FlashInfer publishes its entire JIT cache as a wheel, and the NGC images ship
`flashinfer-python` with an *empty* AOT directory, so every replica rebuilds
kernels upstream already built.

`flashinfer-jit-cache` (2.0 GB, 854 prebuilt modules) covers every FlashInfer
target this stack uses — including `fused_moe_103`, so B300 needs no local nvcc
run, and the index carries aarch64 wheels for the Grace boards. Measured on the
rc19 image after `pip install --no-deps "flashinfer-jit-cache==0.6.12+cu130"
--extra-index-url https://flashinfer.ai/whl/cu130/`:

| Module | compiled locally | loaded from wheel |
|---|---|---|
| `moe_utils` | 638 s | **3.6 s** |
| `fused_moe_trtllm_sm100` | 559 s | **0.7 s** |
| `cutlass_fused_moe` (`fused_moe_100/103/90/120`) | 302 s | **0.0 s** |
| **subtotal** | **1499 s ≈ 25 min** | **4.3 s** |

The wheel version must match `flashinfer-python` exactly (`0.6.12` for rc19,
`0.6.15` for rc23) — worth a CI assertion, so an image bump fails the build
instead of silently reverting to a 25-minute cold start.

What remains is genuinely ours to compile, being `.cu` source in this repo that
no upstream artifact can cover:

| Module | Cold build |
|---|---|
| `kimi routing/permutation impls` (2 llmdd modules) | 1153 s |
| `k3_comm_cuda`, `b10_multimem_ar`, `low_contention_fused_copy` | 232 s (concurrent) |
| **subtotal, serialized** | **~23 min** (~7 min via `prebuild_jit.py`) |

So the 44 min splits into ~25 min that should be a `pip install` and ~23 min that
is ours. **Everything below about `--split-compile` and build scheduling applies
to those four modules only** — it was originally measured against the FlashInfer
modules, which no longer need building.

Open question, not yet measured: the routing/permutation build compiles
FlashInfer's own `trtllm_fused_moe_routing_*.cu` alongside our binding, so part
of its 1153 s may also disappear once the wheel supplies the surrounding
artifacts. Worth one A/B before optimizing that build further.

Two things make this avoidable rather than inherent:

**It parallelizes.** `FLASHINFER_NVCC_THREADS` defaults to **1**
(`jit/cpp_ext.py:94`) and nothing passes `--split-compile`, so a single `ptxas`
grinds for ~10 minutes on one core. `FLASHINFER_EXTRA_CUDAFLAGS` is the
supported injection point. A/B on `moe_utils`, isolated
`FLASHINFER_WORKSPACE_BASE`, shared cubin dir so only compile time is measured:

| Flags | Build |
|---|---|
| default | **586.1 s** |
| `--split-compile=0`, `FLASHINFER_NVCC_THREADS=8`, `MAX_JOBS=128` | **129.1 s** (4.5×) |

**It needs no GPU.** With `FLASHINFER_CUDA_ARCH_LIST=10.0` and
`TORCH_CUDA_ARCH_LIST=10.0a` the entire set builds on a node whose `cuInit`
fails — that is literally how the table above was produced. The artifacts are
shape-generic, so this belongs in the container image, not in a serving process.

One caveat for the torch extensions: `col_quant_cuda.get_module()` calls
`torch.cuda.get_device_capability()` eagerly to populate
`TORCH_CUDA_ARCH_LIST`, which fails without a device. Prebuilding it in a
GPU-less builder needs that value to come from the environment instead. A
two-line fix upstream would be to honour an already-set `TORCH_CUDA_ARCH_LIST`;
`prebuild_jit.py` stubs the capability for the build process instead, so no
shipped code changes for a build-time concern.

`prebuild_jit.py` in this directory is the tool: it builds all seven modules
arch-pinned, in parallel processes, with split-compile on, and reports per-module
and wall time. Run it in an image build stage and keep the cache tree
(`$BASE/.cache/flashinfer`, `$TORCH_EXTENSIONS_DIR`) in the layer. A second run
is a no-op, which is the check that the cache is actually being hit.

## 4. Measured: the storage path is the floor for weight load

Cold reads, incompressible data, one fresh never-read file set per concurrency
level (32 × 256 MiB). This methodology matters: JuiceFS caches on read into a
local NVMe cache, so re-reading the same files reports ~4.9 GB/s and hides the
real cost.

| Path | 1 stream | 8 streams | 32 streams |
|---|---|---|---|
| network FS (JuiceFS, `/root`) — cold | 260 MB/s | 774 MB/s | **901 MB/s** |
| network FS — warm (served from its local NVMe cache) | — | 4931 MB/s | 4188 MB/s |
| local NVMe (`/node-storage`) | 2013 MB/s | 4659 MB/s | ~4561 MB/s |

More readers do not help past ~32: the network FS plateaus at ~0.9 GB/s.

*(Correction to numbers quoted in chat earlier: an intermediate run computed
throughput against a 16 GiB dataset that was actually 8 GiB, which doubled the
network-FS ceiling and the NVMe figures. The table above is the corrected set —
FS ceiling ~0.9 GB/s, not ~1.4; NVMe ~4.8 GB/s, not ~9.5.)*

### What this means for KimiK3

Sizing from `kimi_k3_layer/config.py`: 93 layers, 896 experts, expert
intermediate 384, latent 3584, hidden 7168. Routed experts alone are
896·3·384·3584·93 ≈ **344 B params**; ~385 B total → **≈400 GB fp8 /
≈800 GB bf16**.

| Scenario | 400 GB (fp8) | 800 GB (bf16) |
|---|---|---|
| network FS, single stream (0.26 GB/s) | 26 min | 51 min |
| network FS, saturated (0.9 GB/s) | 7.4 min | 15 min |
| network FS warm cache / local NVMe (4.8 GB/s) | 1.4 min | 2.8 min |

The reported ~30 min is exactly the "single-stream from the network FS" row, plus
the compile tax from §3. Note also the last row: because the FS caches to local
NVMe on read, the *second* start on a node is already ~5.5× faster on this phase.
That is worth confirming in production — if restarts are much faster than cold
starts, the storage path is confirmed as the dominant term.

## 5. What the loader does well, and the cliff it falls off

`prefetch_files` (`hf/weight_loader.py:174`) is well designed: local ranks take
**disjoint** file subsets (`file_names[local_mpi_rank()::local_mpi_size()]`),
each with up to 16 threads — up to 128 concurrent streams, which *(measured)* is
more than enough to saturate this FS.

But it is gated:

```python
enable_prefetch = (prefetch_size < self._get_local_available_host_memory() * 0.9
                   and num_layers == 0)
```

For a 400–800 GB checkpoint on a 2–2.8 TB node this can flip either way, and
**when it flips off the parallel read pass disappears entirely**. The mmap pages
are then faulted in one at a time inside the per-module walk — a single stream at
~0.26 GB/s instead of ~0.9 GB/s. A 3.5× swing on the dominant phase, decided by a
volatile `psutil.virtual_memory().available` reading. For KimiK3 this is likely
the highest-value single fix.

The fix is not to raise the threshold — it is to stop conditioning a *streaming*
decision on *total* memory. Stream always, bounded by a fixed staging budget.

Related, and the opposite of what the code reads like:
`safetensors.torch.load_file` returns **mmap-backed** tensors, not copies
*(measured: RSS grew 0.01 GiB after loading a 1.00 GiB file)*. So the 8 ranks on
a node share page cache rather than each materializing the checkpoint — there is
no 8× host-RAM blow-up. The flip side is that the sharing only holds while the
checkpoint fits in RAM and the ranks stay in lockstep; past that, ranks re-read
from storage.

## 6. Two more load-path findings

**K3/DSv3 copy weights single-threaded.** `DeepseekV3WeightLoader.load_weights`
walks modules in a plain loop:

```python
for name, module in tqdm(all_named_modules.items(), desc="Loading weights"):
```

(`modeling_deepseekv3.py:486`), while the generic path
(`modeling_utils.py:1104`) uses `run_concurrently` with a thread pool. The
largest models in the fleet take the serial path. *(measured, host-side only:
threading that inner loop — mmap fault + TP slice + contiguous + copy over
8 GiB — moved 3.4 s → 2.6 s on local NVMe with 8 shards. That 1.3× is a floor,
not the expected win: with 8 files there is little parallelism to find, and on a
0.26 GB/s-per-stream network FS the gap should be much larger.)*

**H2D copies come from pageable memory.** Expert copies pass
`non_blocking=True` (`fused_moe/quantization.py:625`) but the source is a
pageable mmap tensor, so the copy is effectively synchronous through a staging
buffer. `maybe_pin_memory` exists in `_utils.py:1379` but is used only by runtime
metadata paths — never by weight loading.

Both are fixed by the same change: a shard-level pipeline that reads shard *i+1*
while copying shard *i* through double-buffered pinned staging.

## 7. Can weight load and warm-up run in parallel?

Yes — in tiers, ordered by risk.

**Tier 0 — stop doing the work twice.** Set `kv_cache_config.max_tokens`
explicitly (or reuse the estimation executor instead of destroying it) to skip
phase 8, which builds, warms up, graph-captures and discards a whole executor.
Config-only change; costs one warm-up.

**Tier 1 — take compile out of startup.** Prebuild every JIT artifact into the
image, arch-pinned, with `--split-compile=0` + `FLASHINFER_NVCC_THREADS=8` +
`MAX_JOBS=128`, and build the modules in **parallel processes** — they are
independent ninja invocations, and today they run one after another. Worth up to
~44 min on a cold node at zero correctness risk.

**Tier 2 — overlap I/O with everything that isn't I/O.**
* Kick off `prefetch_files` in a background pool at process start, before
  `import tensorrt_llm` finishes and before the model is constructed. Phases 1–4
  are pure host and compile work that today runs with the NIC idle.
* Replace "prefetch everything, then walk modules" with a shard-level pipeline
  over pinned staging buffers (fixes §6 at the same time).
* Delete the `enable_prefetch` cliff (§5).
* Thread the K3/DSv3 module walk, as the generic path already does.

**Tier 3 — overlap warm-up with weight streaming.** Because phase 4 allocates
parameters at their final addresses, 9b (torch.compile specialization, mostly CPU
work) and 9d (graph capture, which *records* kernels rather than depending on
values) can in principle run while the DMA fills those buffers. Three
constraints, all real:

1. `post_load_weights()` transforms must not reallocate parameters after
   capture, or captured graphs hold stale pointers. Make layout
   value-independent and run it on the empty buffers first.
2. **Keep the autotuner (9c) out of the overlap.** It selects kernels by measured
   time; a PCIe/HBM link saturated by weight DMA would bias every pick. It
   belongs after the load barrier.
3. Zero-init rather than leaving garbage — denormals and NaNs in bf16/fp8 change
   kernel timing and can trip nan-checks.

Expected shape: `[read + H2D + post_load] ∥ [JIT + torch.compile + capture]`,
barrier, then autotuner, then serve. Upper bound on the win is
`min(load_time, warmup_compile_time)`, so it is only worth building **after**
Tiers 0–2 have shrunk the load side — otherwise it hides behind I/O anyway.

### The warm-up itself: what is already cacheable

Two of the three expensive warm-up phases can be made to survive a process
restart today, with no code change:

**Autotuner profiling (phase 9c) has a persistent cache that is off by
default.** `autotune(cache_path=...)` loads previously-saved tactics and skips
profiling on every cache hit (`autotuner.py:255-305`), and the serving path
already wires it to an env var:

```python
cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)   # model_engine.py:1500
with self.no_cuda_graph(), autotune(cache_path=cache_path):
```

Unset means every replica re-profiles every op × shape bucket from scratch. Set
it to a warm file and phase 9c becomes a JSON load. The cache is rank-aware
(shared entries plus per-rank entries for INDEPENDENT ops), so one file serves a
whole TP group.

> **Trap.** The file records `lib_version`, `device_name` and
> `device_capability`, but `_deserialize_metadata` (`autotuner.py:626`)
> *overwrites* the live values with the file's rather than validating them. A
> cache captured on different hardware or a different TRT-LLM build loads
> silently and applies stale tactics — slow kernels, not wrong results, so it
> surfaces as an unexplained perf regression rather than an error. Key the
> filename by `(device_name, capability, lib_version, tp_size)` until that check
> exists upstream; the hardening patch is small and worth sending.

**CUDA-graph capture (phase 9d) is a cross product**, not a fixed cost:
`_capture_generation_cuda_graphs` (`model_engine.py:1623`) iterates batch sizes ×
draft lengths × sparse-attention seq-len variants × a greedy/non-greedy pass.
With MTP draft lengths and a wide `cuda_graph_config.batch_sizes` that is dozens
to hundreds of captures, each a forward plus an instantiate. Capture cannot be
parallelized (single context, driver-serialized), so the only levers are
capturing fewer graphs — pruning `batch_sizes` to what the scheduler actually
schedules, paid for with more runtime padding — and not doing it twice (Tier 0).
Needs the fabric back before any of it can be quantified.

**torch.compile (phase 9b)** has no persistence to exploit here:
`enable_inductor` defaults to false, so the cost is TRT-LLM's own backend
tracing and pattern matching (`_torch/compilation/backend.py`), not inductor
codegen, and there is no on-disk cache to warm.

## 8. Multi-node: using each node's bandwidth

Per-node ingest from the network FS is capped at ~0.9 GB/s *(measured)*, and
more readers do not help. Every node pulling the full checkpoint is therefore
the thing to eliminate, in this order:

1. **Read once per cluster, fan out over the fabric.** N nodes each read a
   disjoint 1/N (aggregate N × 0.9 GB/s), then all-gather over IB
   (8 × 400 Gb/s = 400 GB/s/node). 400 GB across 8 nodes = 50 GB/node ≈ 56 s of
   storage read plus a fabric all-gather, versus 7.4 min saturated / 26 min
   single-stream today. In-node, the same trick over NVLink (900 GB/s) instead of
   relying on page-cache sharing — which also removes the "checkpoint must fit
   in RAM" condition from §5.
2. **Local NVMe as a weight cache**, keyed by checkpoint hash *(measured
   4.8 GB/s → 400 GB in ~1.4 min)*. Every start after the first on that node is
   NVMe-speed. Highest value per unit of work for a fleet that restarts often —
   and the FS is already doing a weaker version of this by accident (§4).
3. **Reuse what is already in-tree.** `LoadFormat.GMS` (`model_loader.py:536`)
   keeps weights in a node-shared **GPU** pool for zero-copy sharing between
   instances, so a restart against a warm pool skips disk entirely. The
   MX/ModelExpress loader (`checkpoints/mx/checkpoint_loader.py`) already does
   RDMA/NIXL replica→replica transfer. What is missing is the cold-start case
   (storage→cluster), which is the same shard-and-gather pattern as (1).

## 9. Harnesses in this directory

`bench_startup.py` wraps the real bring-up functions (it does not
re-implement them) and prints a per-phase table:

```bash
python3 serving_startup/bench_startup.py --model /node-storage/models/tinyllama \
    --label baseline
```

Variants that isolate one phase each: `--dummy-weights` (warm-up only, no weight
I/O), `--kv-tokens N` (skips phase 8), `--no-cuda-graph`, `--no-torch-compile`,
`--tp 8`. Results append to `local_debug/startup/startup_phases.jsonl`.

`bench_weight_io.py` covers the host side, no GPU required:

```bash
# what the loader does today: full prefetch pass, then load_file per shard
python3 serving_startup/bench_weight_io.py --dir <ckpt> --mode prefetch_load
# no prefetch pass / only this rank's TP shard / the serial module walk
python3 serving_startup/bench_weight_io.py --dir <ckpt> --mode load_only
python3 serving_startup/bench_weight_io.py --dir <ckpt> --mode mmap_slice --world 8
python3 serving_startup/bench_weight_io.py --dir <ckpt> --mode module_loop --workers 1
# per-node duplication: N ranks each loading the whole checkpoint
python3 serving_startup/bench_weight_io.py --dir <ckpt> --mode ranks --nranks 8
```

`make_shards.py` builds a synthetic multi-shard checkpoint. It writes **random**
content deliberately: zero-filled files are served from zero-block dedup by
thin-provisioned storage and inflate read throughput by 2× or more.

Two methodology notes worth keeping, both learned the hard way here:
drop the page cache **and** use a fresh never-read file set per data point (the
FS caches on read), and check dataset size against the byte counter before
believing any MB/s number.

## 10. Open items

* **Blocked on the node fabric**: every GPU-resident phase. H2D copy rate from
  pageable vs pinned, the phase-9 warm-up breakdown, whether phase 8 really costs
  a full second warm-up, and the Tier-3 overlap prototype.
* Not yet A/B'd: `--split-compile` on `trtllm_gen_fused_moe_sm100` and
  `cutlass_fused_moe_sm100` (only `moe_utils` was measured).
* `mmap_slice` mode did not reduce bytes read in the synthetic test because the
  generated tensor rows were not divisible by the TP world size, so it fell back
  to whole-tensor reads. Fix the generator before trusting that mode.
* Unmeasured on real hardware: the claim that the IB all-gather in §8.1 is cheap
  relative to storage. Worth a 2-node measurement before building anything on it.

## 11. Baseline containers (updated)

Two containers, deliberately:

| container | image | role |
|---|---|---|
| `trt-dev19` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19` | **aligned baseline** — matches `feat/k3`'s own `__version__` (1.3.0rc19) |
| `trt-dev` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` | the version `agent/moe_optimization.md` measured the shipped table on |

Both mount the `feat/k3` checkout as a sibling of `LLMDiveDeep`, because the K3
layers do not use the installed model code — `_graft`
(`b10_kimi_k3_kda_layer.py:58`) loads fork modules from
`<parent>/trt-llm/tensorrt_llm/...` under their installed package names. That
indirection is load-bearing: **stock TRT-LLM has no SiTU at all** (verified on
the rc23 image: no `_torch/modules/situ.py`, zero SiTU references in
`moe_op_backend.py`), so the K3 activation only exists on the fork side.

Keeping both is not indecision. The shipped Aug-19 table was measured on
rc23-class containers, so reproducing it on rc19 introduces a runtime-version
variable on top of the node change; running both attributes any delta instead of
guessing. JIT caches do not collide — the FlashInfer cache path is
`$BASE/.cache/flashinfer/<flashinfer-version>/<arch>/` — but each image needs
its own `prebuild_jit.py` pass.

### rc19 needs the arch pin to import at all

On the rc19 image, `import tensorrt_llm` **raises** when the GPU cannot be
queried:

```
flashinfer/jit/core.py:108 in check_cuda_arch
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

`check_cuda_arch` iterates `current_compilation_context.TARGET_CUDA_ARCHS`,
which is empty when the device query fails, so "no arch detected" is reported as
"arch too old". Setting `FLASHINFER_CUDA_ARCH_LIST=10.0` (or `10.3`) populates
that set from the environment and the import succeeds. rc23 tolerated the same
condition, so this is an rc19-vs-rc23 behavioural difference, not a node
problem.

Worth pinning in production regardless of this node's broken fabric: it makes
bring-up independent of a transient device-query failure, and it is the same
variable the prebuild uses, so one setting covers both.
