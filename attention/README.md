# Attention, from equations to kernels

This is a **kernel tutorial**, not a Transformer-layer benchmark. Start with
already projected tensors. No QKV/output linear layers, MoE, residual blocks,
model loading, or distributed collectives are part of the new benchmark rows.

## Reading order

For external papers and author-written blogs, follow the annotated
[reading list](READING_LIST.md). It includes a suggested order, local code
companions, and questions to bring to a kernel walkthrough.

| Step | Read | Follow in code |
|---|---|---|
| 1. Dense softmax and online softmax | [attention.md](attention.md) | [dense_attention.py](dense_attention.py) |
| 2. FA2 → FA3 → FA4; actual backend calls | [attention.md](attention.md#4-what-changes-between-fa2-fa3-and-fa4) | [backends.py](backends.py) |
| 3. Local attention and tile skipping | [sliding_window_attention.md](sliding_window_attention.md) | [sliding_window_attention.py](sliding_window_attention.py) |
| 4. Compression, selection, sparse softmax | [compressed sparse attention](sparse_attn/compressed_sparse_attention.md) | [reference operators](sparse_attn/compressed_sparse_attention.py), [adapters](sparse_attn/backends.py) |
| 5. DeepSeek V4.1 / CSA2 specifically | [DeepSeek research](../deepseek/research.md), [SGLang alignment](../deepseek/sglang_alignment.md) | [DeepSeek references](../deepseek/README.md) |
| 6. KDA recurrence, chunking, decode, verification | [KDA kernel guide](linear_attn/kda/KERNELS.md) | [source inventory](linear_attn/kda/SOURCE_INVENTORY.md) |

## Benchmark entry points — prepared, not executed

Use module invocation **from the repository root**. The new drivers require
`--run`; without it they only print help. `--list` only lists adapter names.
Importing these modules does not launch kernels. PyTorch is still required to
import them. No benchmarks, numerical checks, or GPU profiling were run while
creating this tutorial; source/static checks are not runtime validation.

```bash
python -m attention.bench_dense_attention --list
python -m attention.bench_sliding_window_attention --list
python -m attention.bench_compressed_sparse_attention --list
```

The following are **future commands**, not results:

```bash
python -m attention.bench_dense_attention --run --backends torch_eager torch_sdpa_math torch_sdpa_flash fa4
python -m attention.bench_sliding_window_attention --run --window 128 --backends torch_eager torch_online torch_flex fa4 sglang_fa4
python -m attention.bench_compressed_sparse_attention --run --backends torch_gather sglang_flashmla flashmla flashinfer_dsv4
```

For a future small CPU reference check, choose `--device cpu --dtype float32`
and only PyTorch reference backends. Optional GPU backends require compatible
hardware, package versions, compiler/toolkit, shapes, and dtypes. See the adapter
tables before installing anything; no optional packages were installed here.

Every new driver compares an output suffix with a bounded FP32 math oracle
before timing, checks the entire output for finite values, excludes first-call
compilation/warmup, and records median forward-call time, input shape, package
versions, and the measured boundary. This sampled check is not an exhaustive
correctness suite. Missing dependencies/restricted adapters report `SKIP`;
incorrect outputs and unexpected backend failures report `ERROR` and a nonzero
exit status. There is no silent replacement of FA4 with another implementation.

These are **forward kernel API microbenchmarks**: wrapper overhead and any
allocations inside the call remain included. They are not CUDA-graph kernel
traces, backward benchmarks, model throughput tests, or a universal speed ranking.
The PyTorch references intentionally pay for intermediate tensors.

## Results and retained legacy work

Future generated files belong under [results/](results/README.md), divided by
`dense/`, `sliding_window/`, `compressed_sparse/`, `kda/`, and `legacy/`. New
drivers reject CSV paths outside this directory. Old default output paths are
also redirected here. Shell redirection is your responsibility: put logs here too.

Existing CSV/log/PNG artifacts and `local_results` have been cleared. Older
Markdown measurements are historical notes, not regenerated evidence or current
rankings. Source code is retained to avoid destroying previous experiments:

- `linear_attn/benchmarks/`: KDA/GDN/linear-kernel harnesses, plus an explicitly
  legacy layer harness. Consult the KDA guide for which operations each row fuses.
- `sparse_attn/DSA.py`, `dsv4_layer.py`, `bench_*modules.py`, `bench_dsv4_layer.py`:
  older mixed kernel/layer studies; not the entry points for this tutorial.
- `mla/bench_kda_layer.py`, `linear_attn/kda/attention_modules.py`, and the
  `fusefb` experiment: retained layer/projection-fusion studies, outside matched
  attention-core comparisons.
- `linear_attn/tmp/`: historical probes, not a benchmark suite or recommended
  implementation. They may reference old external environments.

Do not treat an old benchmark name, a copied kernel, and a serving framework as
three independent algorithms. Follow the wrapper to the actual kernel first.

