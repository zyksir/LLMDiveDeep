# Linear Attention / KDA — reproducible kernels

This directory isolates the linear-attention work from `sparse_attn/`. It has
one exact PyTorch KDA recurrence, thin wrappers around available SGLang kernels,
and one benchmark driver for:

1. **decode** latency over batch size, with context length as a control axis;
2. **prefill** throughput over batch size and sequence length;
3. a Chrome-compatible **decode profile**.

No model weights or server are required. Read `SURVEY.md` for the math, model
status, implementation comparison, and measured B200 results.

## Environment

The shared environment is `LLMDiveDeep/.venv`:

```bash
cd /path/to/LLMDiveDeep
uv venv .venv
source .venv/bin/activate
uv pip install -r linear_attn/requirements.txt
# Direct upstream-FLA comparison (local reference checkout).
uv pip install -e ../references/flash-linear-attention --no-deps
cd linear_attn
```

Validated environment: NVIDIA B200 (SM100), CUDA 13, Python 3.12,
PyTorch 2.11.0+cu130, Triton 3.6.0, SGLang 0.5.15.post1.

### CuTeDSL and CUDA requirements

**CUDA.** Running the benches needs no system CUDA toolkit: the `+cu130`
PyTorch wheels bundle the CUDA 13.0 runtime, and Triton / CuTeDSL JIT with
their own bundled `ptxas`. What is required:

- a driver new enough for CUDA 13 (r580+; this machine: 590.48.01, which
  `nvidia-smi` reports as "CUDA Version 13.1" — that is driver capability,
  not an installed toolkit);
- an SM100 GPU for the full matrix. Per-row floors: SGLang CuTeDSL decode
  SM90+; FlashKDA CUTLASS SM90+; TRT-LLM `cute` prefill SM100 + head_dim 128
  + bf16 only; INT21 KDA-B200 SM100/SM103 only.

A system `nvcc` is only needed to **build** the optional native packages
(FlashKDA, INT21), and it must match `torch.version.cuda` (13.0). The
system's nvcc 12.8 fails twice there: torch's extension build rejects the
version mismatch, and ptxas 12.8 rejects the >48 KB static shared memory the
INT21 sm_100a kernels use. Workaround (see SURVEY §INT21): install the
CUDA 13.0 toolkit from pip wheels (`nvidia-cuda-nvcc==13.0.*` etc.), point
`CUDA_HOME` at it, and keep the wheel pins at `13.0.*` — 13.3 nvcc breaks on
torch 2.x headers.

**CuTeDSL.** `nvidia-cutlass-dsl[cu13]==4.5.2` (the `cu13` extra must match
the torch CUDA stack). Two hard-won constraints:

- The install must be *self-consistent*: the package's pure-Python wrappers
  and its compiled MLIR bindings ship in three dists
  (`nvidia-cutlass-dsl`, `-libs-base`, `-libs-cu13`) that must all be the
  same version. A half-upgraded mix fails during JIT with misleading errors
  ("constructor mismatch", `unexpected keyword argument 'target_tensors'`).
  When in doubt: `uv pip install --force-reinstall --no-deps` all three.
- Version pins differ across frameworks: the TRT-LLM branch pins 4.5.0, but
  4.5.2 compiles and validates all its KDA kernels here (4.5.0 aborted in
  this venv). CuTeDSL kernels also bake *dtypes* at compile time — TRT's
  `cute` prefill needs a bf16 beta and SGLang's decode a bf16 `dt_bias`, or
  they silently NaN — the registrations in `kda/kda_*_register.py` handle this.

Optional FlashKDA prefill kernel ([MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)):

```bash
# SM90+ CUTLASS build; needs CUDA toolkit near the PyTorch CUDA version.
# If host nvcc is older than torch.version.cuda, patch the version check
# (see SURVEY) or point CUDA_HOME at a matching toolkit.
cd ../references && git clone --recursive https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
FLASH_KDA_CUDA_ARCHS=100a uv pip install -v --no-build-isolation \
  --python ../../LLMDiveDeep/.venv/bin/python .
```

Once installed, `bench_kda_prefill.py` includes `flash_kda` plus a matched
`fla_kda_safe_triton` baseline (safe gate, `FLA_FLASH_KDA=0`). FlashKDA is
not used by the canonical-gate Triton rows.

Optional INT21 KDA-B200 PTX kernel ([Int21-AI/KDA-B200](https://github.com/Int21-AI/KDA-B200),
`flashkda_ptx_int21` row): interface-identical plain CUDA/PTX rewrite of
FlashKDA, ~1.5x faster on B200 but limited to <= 262144 total tokens (int32
offsets). Its package is *also* named `flash_kda`, so it is built into an
isolated directory (`../kda_b200_install`, see SURVEY for the CUDA 13 build
recipe) and loaded under a `flash_kda_ptx` alias by `kda/kda_prefill_register.py`.

Optional FlashQLA GDN kernel ([QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA),
TileLang, SM90/SM100; pins `tilelang==0.1.9`):

```bash
cd ../references && git clone --recursive https://github.com/QwenLM/FlashQLA.git flash-qla
uv pip install -v --python ../LLMDiveDeep/.venv/bin/python ./flash-qla
```

Once installed, `bench_gdn_attention.py --mode prefill` includes a
`flash_qla_gdn_chunk` row (kernels JIT-compile on first use). FLA
auto-dispatches its GDN `chunk_gated_delta_rule` to FlashQLA when installed,
so the bench pins the `fla_gdn_chunk` row to Triton via `FLA_FLASH_QLA=0`.

The inspected SGLang checkout has no FlashInfer KDA decode adapter. CuTeDSL
rows (SGLang decode, TRT-LLM `cute` prefill) have their own install
constraints — see "CuTeDSL and CUDA requirements" above.

## Run

Quick decode:

```bash
python benchmarks/bench_linear_attention.py --mode decode \
  --batch-sizes 1 8 32 128 \
  --context-lens 4096 32768 131072 1048576
```

Matched decode implementations:

```bash
python benchmarks/bench_linear_attention.py --mode decode \
  --backends fla_kda_recurrent \
             sglang_kda_split \
             sglang_kda_packed \
  --batch-sizes 1 8 32 128 --context-lens 4096 \
  --csv results/bench_linear_attention_impls.csv
```

This compares the same projected Q/K/V/gate inputs, V-first fp32 recurrent
state, in-place state update, and one-token-per-request decode contract.
`fla_kda_recurrent` imports the current FLA checkout directly;
`sglang_kda_split` uses SGLang's vendored FLA-derived path; and
`sglang_kda_packed` is SGLang's serving-specific projection-output-to-state
fusion. (Older result CSVs use the pre-rename names `upstream_fla_recurrent`
/ `sglang_triton_split` / `sglang_triton_packed`.)

Cross-framework GDN (decode plus chunk prefill, including FlashQLA):

```bash
python benchmarks/bench_gdn_attention.py --mode both \
  --batch-sizes 1 8 32 128 \
  --prefill-batch-sizes 1 4 \
  --prefill-seq-lens 512 2048 8192 \
  --csv results/bench_gdn_attention.csv
```

The recurrent GDN rows pass identical preactivated scalar log-decays and
post-sigmoid beta to upstream FLA, SGLang, vLLM, and TensorRT-LLM kernels.
The SGLang packed row additionally measures its serving fusion from packed
Q/K/V and raw gate/beta logits. Prefill uses the same packed-varlen contract
and sequence boundaries across all GDN sources. The TensorRT-LLM rows are its
vendored Triton kernels with fp32 state; its production FlashInfer bf16-state
paths are a separate comparison.

Cross-framework KDA — exactly three bench files, one per phase:

```bash
# 1. decode: the matched one-token recurrence
python benchmarks/bench_kda_decode.py --batch-sizes 1 8 32 128 \
  --csv results/bench_kda_decode.csv

# 2. prefill: fused raw-gate chunk kernels (incl. FlashKDA and the
#    FlashKDA-style Triton port); --mode sweep runs the full B x S grid
#    with per-shape subprocess isolation, --mode stages splits SGLang's
#    pipeline into gate / WY / state / output leaf timings
python benchmarks/bench_kda_prefill.py --batch-sizes 1 4 --seq-lens 512 2048 8192 \
  --csv results/bench_kda_prefill_quick.csv
python benchmarks/bench_kda_prefill.py --mode sweep    # full grid -> results/bench_kda_prefill.csv
python benchmarks/bench_kda_prefill.py --mode stages --csv results/bench_kda_stages.csv

# 3. spec verify: per-kernel inventory of the MTP-verify design space
#    (save_ssm / replay_ssm_split / replay_ssm_fused; see KDA.md §2)
python benchmarks/bench_kda_spec_verify.py --check
```

KDA decode is compared across FLA, SGLang (split and packed), vLLM, and
TensorRT-LLM (`feat/kda` branch) with the same Q/K/V, beta, shape, fp32
state, and timing loop. b10 CuTeDSLGen kernels live under `kda/b10/`
(`b10_kda_*_cutedsl.py`) and are registered as ordinary backend rows in
`kda/kda_*_register.py`; every module entry point asserts the baked
head_dim=128 contract, and each is validated against the exact recurrence
oracle alongside the framework kernels.

Isolated attention layer:

```bash
python benchmarks/bench_attention_layer.py \
  --batch-sizes 1 8 32 128 \
  --csv results/bench_attention_layer.csv
```

This includes input/gate projections, short convolution, KDA, sigmoid-gated
RMSNorm, and output projection. It excludes the decoder pre-norm/residual,
MoE/MLP, collectives, and scheduler. `upstream_fla_attention` is FLA's actual
module; `sglang_fused_attention` is a small TP=1 reproduction of SGLang's
attention-only dataflow using SGLang kernels, without model-runner plumbing.

For a **full K3 transformer layer** (attn + LatentMoE, tp8/pp8), see
`../kimi_k3_layer/` — not this package.

The context lengths are intentional labels, not allocations. KDA decode reads
one fixed recurrent state per request, so latency should remain flat as context
grows. Any slope indicates hidden replay, state reconstruction, or another
non-constant operation around the kernel.

Prefill:

```bash
python benchmarks/bench_linear_attention.py --mode prefill \
  --prefill-batch-sizes 1 4 \
  --prefill-seq-lens 512 2048 8192
```

Prefill defaults to safe-gate KDA (`--lower-bound -5`). Use
`--lower-bound nan` for the original unbounded
`-exp(A_log) * softplus(raw_gate + dt_bias)` gate. With the `flash_kda`
package installed, the `sglang_flashkda` row runs SGLang's FlashKDA kernel
object for safe-gate prefill with 64 <= sequence length <= 2048; outside
that window (or with the canonical gate) it is reported unavailable because
SGLang intentionally falls back to Triton there.

Decode trace:

```bash
python benchmarks/bench_linear_attention.py --mode decode \
  --batch-sizes 32 --context-lens 1048576 \
  --profile --profile-backend sglang_kda_packed \
  --trace results/kda_decode_trace.json
```

Open the trace in Perfetto or `chrome://tracing`.

Useful flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--qk-heads / --value-heads` | `16 / 16` | QK and value heads (`bench_linear_attention.py` / GDN benches) |
| `--heads` | `12 96` | KDA benches sweep per-rank head counts: Kimi K3's TP8 shard (12) and TP1/PP shard (96) |
| `--key-dim / --value-dim` | `128 / 128` | per-head state dimensions |
| `--state-dtype` | `float32` | recurrent-state dtype; SGLang default |
| `--warmup / --iters / --repeats` | `10 / 50 / 5` | CUDA-event timing budget |
| `--backends ...` | all discovered | restrict backend names |
| `--csv PATH` | `results/bench_linear_attention.csv` | raw result output |
| `--figure PATH` | CSV path with `.png` suffix | result figure output |

Every benchmark run also renders a PNG figure next to the CSV (decode panels
plot latency versus batch size; prefill panels plot throughput versus total
tokens, one line per backend and batch). Read the figure; the CSV is just the
raw data behind it.

## Files

| File | Purpose |
|---|---|
| `linear_attention.py` | high-level contracts only: `Shape` and the `BackendRegistry` mechanism |
| `frameworks.py` | shared vLLM / TRT-LLM leaf-module import shims |
| `kda/kda_attention.py` | naive PyTorch KDA math (gate + exact recurrence) — start here |
| `kda/inputs.py` | synthetic decode/prefill input contracts shared by all backends |
| `kda/kda_decode_register.py` | **register** of every one-token decode backend |
| `kda/kda_prefill_register.py` | **register** of every chunk-prefill backend |
| `kda/kda_verify_register.py` | **register** / inventory of MTP-verify kernels (`save_ssm` / `replay_ssm_split` / `replay_ssm_fused`) |
| `kda/kda_replayssm_fold.py` | vendored SGLang PR #32541 exact-fold Triton kernel (`replay_ssm_split` `fold_replay`) |
| `kda/b10/` | b10-authored kernels (CuTeDSLGen champions + Triton FlashKDA port); each docstring names the original kernel it aligns to |
| `gdn/gdn_attention.py` | GDN input contract, exact recurrence, and registered backends |
| `benchmarks/bench_kda_decode.py` | cross-repository KDA one-token decode sweep |
| `benchmarks/bench_kda_prefill.py` | cross-repository KDA chunk-prefill sweep (`--mode prefill/stages/sweep`) |
| `benchmarks/bench_kda_spec_verify.py` | per-kernel MTP-verify (spec decode) benchmark |
| `benchmarks/bench_gdn_attention.py` | cross-repository GDN sweeps |
| `benchmarks/bench_linear_attention.py` | SGLang-centric decode/prefill sweeps + profiler trace |
| `benchmarks/bench_attention_layer.py` | attention-module decode sweep and profiler trace |
| `../common/kernel_bench.py` | the kernel-bench framework: timing, correctness, registry, table/CSV/figure output |
| `SURVEY.md` / `KDA.md` | survey + KDA kernel deep dive |
| `results/` | measurement artifacts |
| `tmp/` | one-off analysis scripts; safe to delete |

## Scope and caveats

- `bench_linear_attention.py` is kernel-level.
  `bench_gdn_attention.py`, `bench_kda_decode.py`, and `bench_kda_prefill.py`
  also stay at the kernel boundary: their vLLM and TensorRT-LLM rows load
  vendored FLA leaf modules without model/runtime initialization.
  `bench_attention_layer.py` includes the complete KDA attention module but
  excludes MoE, collectives, scheduling, CUDA graphs, and periodic
  full-attention layers.
- The default 16-head shape is one TP=2 local shard of Kimi-Linear's 32 global
  KDA heads. Kimi K3's exact heads, dimensions, layer map, and gate
  parameterization were not public on 2026-07-21.
- The state is mutated in place during timing, matching serving. Inputs use a
  stable negative gate distribution so repeated updates remain finite.
