# Kimi-K3 layers

This package contains two stateful Kimi-K3 layer implementations:

- `b10_kimi_k3_kda_layer.py`: KDA reference/TRT-LLM (`KimiK3KDA`) and optimized B10
  (`KimiK3KDAB10`) layers.
- `b10_kimi_k3_moe_layer.py`: MoE reference (`KimiK3MoEReference`) and
  optimized B10 (`B10KimiK3MoELayer`) layers. Every collective goes
  through one `communication.collective.Collectives` instance; the
  strategy is the frozen `measured_config(tokens)` plan
  (see `agent/moe_optimization.md`).

Package `__init__.py` files intentionally perform no heavy imports. Import the
layer or kernel module you need directly.

## Supported parts, and what degrades where

One build serves every Blackwell datacenter part we deploy on. `common/arch.py`
is the single place that decides compile targets — `sm_100a` for B200/GB200,
`sm_103a` for B300/GB300 — and every JIT/AOT cache key carries the arch, so a
B200 cubin can never be replayed on a B300 (torch's `cpp_extension` keys its
build dir on name + source hash only, hence `arch.ext_name()`).

`capabilities.py` gates each stage independently: a part missing one kernel
loses **that stage only** and keeps every other optimization, instead of falling
back to the reference forward. Ladders, each ending somewhere every part runs:

| stage | fast | then | then |
|---|---|---|---|
| `fused_front_cute` | fused CuTeDSL dual-out | separate sharded fc1 | separate full fc1 |
| `radix_routing` | radix kernel | reference gate + top-k | |
| `multimem_tail` | multimem full-fc2 | sharded fc2 | packed latent |
| `sharded_prefill_fc1` | sharded + DMA gather | full fc1 | |
| `fc2_shard_prefill_tail` | fc2-shard tail | packed tail | |
| `native_experts` | native trtllm-gen runner | FlashInfer op backend | |

Known gap today, and it is **activation-conditional**: on **sm_103** the
trtllm-gen MXFP4 MoE GEMM ships SwiGlu-only kernels, so `ExpertBackend.NATIVE`
is unusable there *when the runner is invoked with SiTU*.

This layer does not do that. It hands the runner `ActivationType.Swiglu`
(`b10_kimi_k3_moe_layer.py:360`) and applies SiTU itself — grafted `SiTUAndMul`
for the shared branch, `situ_and_mul` for the inline path — and SwiGlu is
exactly the variant sm_103 has. So **on B300 this layer degrades nothing**; the
gap fires only for a caller that pins the trtllm-gen runner *and* passes SiTU,
which is the fork-integrated serving path. `probe(situ_experts=...)` is
therefore read from what the layer actually configures, so if that ever changes
the gate starts applying by itself.

Worth knowing for context: SiTU exists **only** in the fork. Stock rc19 and
rc23 both ship no `_torch/modules/situ.py` and contain no SiTU reference in
`moe_op_backend.py` — which is why `_graft` is load-bearing rather than a
convenience.

Decisions are made once in `init_optimized`, outside any timed or captured
region, and **all-reduced MIN across ranks** — a collective-bearing stage must
be on everywhere or nowhere, or ranks deadlock instead of failing. Every
downgrade logs as `[k3-caps]`, and `B10_DISABLE_STAGES=stage1,stage2` forces a
stage off in production without a code change.

Two caveats:

- **Availability is not tuning.** `measured_config` thresholds were measured on
  B200 (`TUNED_ARCHES`); on B300 they are correct but not optimal, and the layer
  says so at init. Re-run `agent/RUNBOOK.md` §1 there to retune.
- The gap table is static plus operator override. Catching a kernel failure at
  first use and downgrading live is **not** implemented — that needs hardware to
  build safely.

Self-checks, no GPU required (`common.arch` needs only stdlib; the
capabilities check needs the TRT-LLM python environment for the layer enums,
but never touches a device):

```bash
python3 -m common.arch                    # flags and cache names for both targets
python3 -m kimi_k3_layer.capabilities     # every ladder rung (TRT-LLM env)
```

## Runtime dispatch

KDA dispatch is automatic from the requested token count:

- 1–128 tokens: decode.
- More than 128 tokens: prefill.

The optimized MoE layer dispatches on `measured_config(tokens)`: the
optimized decode path through 256 tokens (+17.6/+21.0% at 192/256),
the full-fc1 + fc2-shard-tail prefill for 257–2047 tokens
(+16.6/+22.3% at 512/1024 — there is no baseline window any more),
the sharded-fc1 prefill with the zero-SM DMA-mover gather from 2048
(+21.1/+31.8/+35.1/+41.3% at 2048/4096/8192/16384), and the native
expert backend from 8192. Unsupported ranges fall back to the reference forward
without changing the layer contract.

## Dependencies

Run from the repository root in the TRT-LLM environment. The layers require
PyTorch, the installed `tensorrt_llm` runtime, and their existing CUDA/Triton/
CuTeDSL dependencies. Multi-GPU MoE paths also use the primitives in
[`../communication/`](../communication/).

## FlashInfer prebuild (do this first, always)

FlashInfer is a core dependency: it packages the production TensorRT-LLM Gen
MoE routing/permutation kernels that our benchmarks and layers call, and its
modules are JIT-compiled on first use (minutes of nvcc). At the **start of any
task**, kick off the FlashInfer module build in the background so it is cached
before any subtask needs it; the artifacts are shape-generic (batch size is
runtime dispatch), so this is a one-time per-environment cost, never a
per-shape cost:

```bash
# warm the FlashInfer JIT cache in the background (no-op once cached)
python3 -c "
from kimi_k3_layer.kernels.routing_permutation_impls import load_impl
for name in ('trt_moe_sort', 'trt_routing_custom'):
    m = load_impl(name); m.load(); print(name, 'ready')
" &
```

Never let a benchmark or layer path trigger this compile inside a timed or
interactive loop, and never introduce a kernel that recompiles per shape.

## Layer benchmarks

One benchmark covers each complete layer:

```bash
# KDA decode and prefill; writes under kimi_k3_layer/local_result/
python3 kimi_k3_layer/bench_b10_kimi_k3_kda_layer.py

# MoE, TP=8; correctness-gated CUDA-graph benchmark
mpirun -n 8 --allow-run-as-root \
  python3 kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --sizes all
```

`search_moe_strategies.py` is report-only: EXP-mode coordinate-descent
over the `ExperimentConfig` axes; it writes a comparison report but never
changes source defaults.

```bash
mpirun -n 8 --allow-run-as-root \
  python3 kimi_k3_layer/search_moe_strategies.py --sizes 16
```

## Kernel verification

Each focused kernel has a standalone correctness-and-performance script:

```bash
python3 kimi_k3_layer/kernels/bench_routing.py
python3 kimi_k3_layer/kernels/bench_split_projection.py
python3 kimi_k3_layer/kernels/bench_rmsnorm.py
python3 kimi_k3_layer/kernels/bench_attn_res.py
python3 kimi_k3_layer/kernels/bench_kda_decode.py
python3 kimi_k3_layer/kernels/bench_kda_prefill.py
```

The KDA scripts compare B10 against trusted TRT-LLM paths. The other scripts
compare against TRT-LLM or direct PyTorch references as appropriate.

## Outputs and maintainer notes

Generated CSVs, figures, traces, and experiments belong in ignored
`local_result/`; they are local evidence, not package API.

Historical optimization records, runbooks, design specifications, and figures
live in [`agent/`](agent/). They provide maintainer context and are not user
entry points.
