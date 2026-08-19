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
