# communication — autotuned collectives for single-node TP (B200)

One dispatching class, `Collectives` (`collective.py`), over every
collective implementation we have; all implementation code lives in
`backends/`, one module per backend. `impl="auto"` dispatches through
a profiled `(op, dim, token_bucket) -> impl` map that fills lazily on
first use (or upfront via `autotune()` / `import_auto_map()`).

## Ops

    all_gather           [B, H] shard -> [world*B, H]
    reduce_scatter       [world*B, H] partials -> [B, H]
    all_reduce           [B, H] partials -> [B, H]
    quantized_all_reduce lossy INT8/FP8-wire sum -> BF16 [B, H]
    all_to_all           [world*B, H] -> [world*B, H]
    all_gather_col_quant column AG + fused MXFP8 write-out
    allreduce_norm       rmsnorm(allreduce(x) [+ residual], gamma, eps)
    gemm_allreduce       allreduce(x @ w.T)
    allreduce_norm_gemm  rmsnorm(allreduce(x) + residual, gamma, eps) @ w.T

No output buffers anywhere: ops return the tensors their kernels
produced (backend allocations, CUDA-graph-pool tensors, or views of
internal staging — valid until the next call; clone to persist).
For the fused GEMM ops, ``seq`` is the multi-kernel native baseline
(each stage dispatched through this class, so it automatically rides
the autotuned winner of the inner op); every other impl name is a
SINGLE kernel.

## Backends (`impl=` names)

| impl | what | where |
|---|---|---|
| `torch_symm:multimem` / `:1shot` / `:2shot` | torch symmetric-memory kernels (all available iff `torch.ops.symm_mem` loads) | `torch_symm_backend.py` |
| `torch_low_contention` | torch's low-contention AG / RS | `torch_lc_backend.py` |
| `flashinfer:1shot` / `:2shot` | FlashInfer Lamport AR; fused AR+residual+RMSNorm for the norm ops | `flashinfer_backend.py` |
| `trt` | TRT-LLM custom AR, MIN_LATENCY (+ fused RESIDUAL_RMS_NORM) | `trt_backend.py` |
| `sgl:push_res` / `sgl:pull_res` / `sgl:push_norm` / `sgl:pull_norm` / `sgl:gemm_ar` | sglang's vendored Kimi-K3 fused MNNVL kernels (1shot multicast-push AR, low-SM NVLS 2shot pull AR on the torch_symm staging, fused-RMSNorm variants, single-kernel GEMM+AR); SM100/103 + multicast + bf16 + whole-world group only — probes and registers nothing elsewhere. MoE-finalize AR and up_proj GEMM+AG are separate entry points on the backend (no op grammar fits them) | `sglang_backend.py` (via `kimi_k3/kernels/sgl_adapters/`) |
| `vllm_int8` / `vllm_fp8` | vLLM/Kraken two-shot per-group quantized AR (explicit lossy API only) | `quantized_backend.py` |
| `nccl_symm` | NCCL over symmetric-memory staging (NVLS when IMEX is provisioned) | `nccl_symm_backend.py` |
| `nccl` | plain torch.distributed (unbounded fallback) | `nccl_backend.py` |
| `seq` | multi-kernel composition of the fused ops | dispatcher |
| **ours:** `b10_multimem` (+`:lamport`, non-default) | NVLS multimem AR | `kernels/b10_multimem.py` |
| **ours:** `b10_copy_engine` (+`:sm`) | rotated-schedule DMA / SM-copy movers (off by default, `enable_b10`) | `kernels/b10_copy_engine.py` |
| **ours:** `cutedsl` | experimental fused AR+norm and GEMM+AR (off by default, `enable_cutedsl`) | `kernels/cutedsl_*.py` |

Backend adapters live in `backends/`; custom implementation code lives
in `kernels/`. `kernels/col_quant.py` provides column all-gather with
optional fused MXFP8 output.

**Own-kernel margin gate**: our impls (`b10_*`, `col_*`, `cutedsl`) must beat
the best external candidate (torch / flashinfer / trt / nccl / seq) by
≥5% to win a bucket (`_OWN_WIN_MARGIN`) — sub-5% gains don't justify
custom kernel surface. `b10_multimem:lamport` is kept but not a
default candidate (its only win was ~2.5% at bs=64).

## Quick start

```python
from communication.collective import Collectives

comm = Collectives(group, max_numel=8192 * 7168, max_hidden=7168)
y = comm.all_reduce(x)                       # lazy-autotuned dispatch
q = comm.quantized_all_reduce(x, impl="vllm_int8")  # explicit lossy API
y = comm.gemm_allreduce(x, w)                # allreduce(x @ w.T)
norm, res = comm.allreduce_norm(x, gamma, 1e-5, residual=res_in)
y, res = comm.allreduce_norm_gemm(x, res_in, gamma, w)

comm.import_auto_map(
    json.load(open("communication/local_result/lookup_map.json"))
)
```

Constructor knobs: `enable_flashinfer` / `enable_trt` / `enable_b10` /
`enable_cutedsl`
(movers off by default; explicit `impl=` builds them on demand),
`lazy_autotune`, `skip_impls`. The NVLink-domain check is opt-in
(`assert_single_node()`; GB200 NVL72 domains legitimately span hosts).

## Benchmark

`bench_comm.py` checks every available backend against NCCL, benchmarks
all requested operations, and writes Markdown, CSV, and the preloadable
dispatch map under `communication/local_result/`.

```bash
timeout 5m mpirun -n 8 python3 communication/bench_comm.py --report
# Include cached experimental fused kernels:
timeout 5m mpirun -n 8 python3 communication/bench_comm.py --report --enable-cutedsl
```

The report preset uses one process-group schedule for AllReduce,
AllGather, ReduceScatter, and AllToAll at
`1,2,4,8,16,32,64,128,256,512,1K,2K,4K,8K,16K`, including NCCL,
NCCLSym, Torch
symmetric memory, FlashInfer/TRT where applicable, and local kernels.
Quantized AllReduce is reported separately with INT8/FP8 error metrics
and speedup against every exact backend; it never participates in exact
`all_reduce(auto)` dispatch.
The validated cold TP8 `--report` run completes in about 51 seconds.
CuTeDSL stays opt-in so the default report has no compiler dependency.
Its launcher exports rank-specific AOT objects and loads them in later
processes without JIT compilation; changing token count reuses each
dynamic-row kernel rather than compiling a new shape. Run
`kernel_benchmarks/prebuild_cutedsl.py` once (or in the Docker build) to
create the Kimi-K3 objects. The AR+norm+GEMM CuTeDSL kernel is retained
for investigation but excluded from report/autotune candidates.

The report uses a stable long-table schema: shape, backend, latency,
payload, algorithm bandwidth, bus bandwidth, theoretical speed of
light, efficiency, correctness, winner, hardware, and notes. Use
`--check-only` to skip timing.

Backend-specific correctness and replay benchmarks live beside their
kernels in `kernel_benchmarks/`.
