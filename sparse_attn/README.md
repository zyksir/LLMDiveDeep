# Sparse & Linear Attention Benchmarks — how to reproduce

Reproducible microbenchmarks for the non-dense attention paths that show up
in modern open-source LLMs — DeepSeek-V3.2 / V4 / GLM-5 / GLM-5.2 sparse
attention, plus Qwen3-Next / Kimi-Linear linear attention. All numbers come
from **synthetic tensors** through the same Triton / CuTe / DeepGEMM kernels
the production models run; no model weights are loaded, no server setup is
required.

**Results are in `SURVEY.md`.** This file only covers *how* to run.

Three questions the benches are designed to answer, each covered by a
different driver:

1. **Where does DSA time go?** (indexer vs sparse-FA vs dense-FA at long
   context) → `bench_indexer_cost.py`, results in `SURVEY.md §3`.
2. **How do sparse and linear attention compare with dense at matched shape?**
   (dense FA baseline, block-sparse FA at fixed sparsity, indexer alone,
   combined DSA, linear-attention GDN) → `bench_sparse_attention_kernels.py`
   (six parts A–F).
3. **What does the *whole* nn.Module cost?** — the module-level bench
   (DSv3.2 / GLM-5 / DSv4-C4 with projections, LayerNorm, RoPE, FP8 quant
   included) → `bench_sparse_attention_modules.py`.

All numbers on a single **NVIDIA B200**, `lmsysorg/sglang:dev-cu13` /
`torch 2.11.0+cu130` / `triton 3.6.0`.

---

## 1. The wider landscape (what else sglang ships)

The scripts here focus on DSA-style sparse attention because that is the
family with the most active development in 2025–2026. But sglang actually
ships several other non-dense attention paths. If you're wondering "does the
picture change with a different attention family?", the answer is on this
list.

### Linear / sub-quadratic (recurrent state, O(N) cost)

| Family | Models that use it | Kernel |
|---|---|---|
| **Gated Delta Net (GDN)** | Qwen3-Next, Qwen3.5-VL, Jet-Nemotron, InternS2 | `fla/chunk.py::chunk_gated_delta_rule` |
| **Kimi Delta Attention (KDA)** | Kimi-Linear | `fla/kda.py::chunk_kda` |
| **Lightning / SegLa** | Bailing-MoE-Linear, Bailing-MoE-v2.5 | `linear/seg_la.py::seg_la_fwd` |
| **Mamba-2 SSM** | Falcon-H1, Nemotron-H, Granite-MoE-Hybrid, LFM2, Zaya | `mamba/ops/ssd_combined.py` |

**GDN is benchmarked in PART F** of `bench_sparse_attention_kernels.py` as the
canonical linear-attention example; **KDA (Kimi-Linear)** is covered alongside
it: `LinearAttention.py` ships a portable pure-torch gated-delta-rule
reference for *both* GDN and KDA (runs with no sglang), plus thin wrappers
around the FLA `chunk_gated_delta_rule` / `chunk_kda` kernels for the fast
path. The remaining two (SegLa, Mamba-2) share the same $O(S)$ scaling and
the "linear state instead of softmax score matrix" shape; expect similar
curves.

### Sparse (subset of KV, still softmax)

| Family | Models | Kernel |
|---|---|---|
| **DSA / lightning indexer** (V3.2, GLM-5, GLM-5.2 IndexShare) | DeepSeek-V3.2, GLM-5 / 5.2 | `dsa/dsa_indexer.py`, `deep_gemm.fp8_mqa_logits` |
| **DSA-C4** (DSA + compressed K stream) | DeepSeek-V4 | `dsv4/indexer.py`, `dsv4/compressor.py` |
| **Quest** (page-wise KV subset selection at runtime) | Model-agnostic (`--hisparse-config`) | `mem_cache/sparsity/algorithms/quest_algorithm.py` |
| **Dual-chunk vertical / slash** (MInference-style pattern) | Qwen2, Qwen3-MoE, GLM-4 (with `dual_chunk_attention_config`) | `sgl_kernel.sparse_flash_attn.sparse_attn_func` |
| **Sliding window + attention sinks** | gpt-oss, Mistral, Ministral, Gemma-2/3, Olmo-2, Phi-MoE, Cohere-2 | Window enforced inside FA / Triton metadata |
| **Video block-sparse (VSA)** | WanVideo DiT, Causal-WanVideo, Lingbot-World | `multimodal_gen/.../sparse_attn/video_sparse_kernel.py` |

DSA / DSA-C4 are benchmarked in **PARTs C, D, E** of the kernel bench and in
the new headline bench `bench_indexer_cost.py`. PARTs B, D, and E use a
generic block-sparse Triton kernel for their "sparse-FA" column — the same
kernel WanVideo uses for video block-sparse attention.

### Eviction / cache tiering

**HiSparse** — host↔device KV tiering that coordinates with DSA / DSv4 / MLA
backends. Not a stand-alone attention kernel; not benchmarked here.

### What's *not* in sglang

Notably absent (as of the version in this container): SnapKV, H2O, RocketKV,
MoBA, InfLLM, MInference (as a named integration), RWKV, RetNet, MiniMax-M3.

---

## 2. Environment

You need a Hopper (SM_90) or Blackwell (SM_100) GPU and CUDA 12.6+/13. All
numbers in `SURVEY.md` were validated on a single **NVIDIA B200**, CUDA 13,
Python 3.12, `torch` 2.11.0+cu130, `triton` 3.6.0.

### Installing the dependencies

**sglang is a hard dependency, not optional.** The DSA indexer kernel
(`deep_gemm.fp8_mqa_logits`), the FA4 dense baseline, and the linear-attention
kernels (`fla.chunk` / `fla.kda`) all ship as part of sglang and its
dependency set — installing sglang pulls them in, no separate build step.
The recommended flow is a dedicated virtualenv at the repo root
(`LLMDiveDeep/.venv`), shared by the `sparse_attn` and `quantization` suites.
We use [`uv`](https://docs.astral.sh/uv/) — a faster drop-in replacement for
`venv` + `pip` (`pip install uv` if you don't have it):

```bash
cd /path/to/LLMDiveDeep
uv venv .venv
source .venv/bin/activate

# Everything in one shot: torch/triton/numpy + sglang[diffusion]==0.5.15.post1
# (which brings the DSA indexer kernel, the FA4 dense baseline, and the
# GDN/KDA linear-attention kernels). PyPI ships a prebuilt cp310-cp313
# manylinux wheel for that pin, so this needs NO source build and NO Rust
# toolchain.
uv pip install -r sparse_attn/requirements.txt

# One extra for the new bench_indexer_cost.py plot output.
uv pip install matplotlib
```

> **Building sglang from source instead (only if you want to modify it).**
> The prebuilt wheel above is enough to run every bench. If you need an
> editable checkout, the source build uses `setuptools-rust`, so install a
> Rust toolchain first (`cargo` must be on PATH), then `-e` install:
>
> ```bash
> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
>     | sh -s -- -y --default-toolchain stable --profile minimal
> source "$HOME/.cargo/env"
> git clone https://github.com/sgl-project/sglang.git
> git -C sglang checkout v0.5.15.post1
> uv pip install -e "sglang/python[diffusion]"
> ```

> **FA4-cute JIT caveat.** The FA4 CuTe-DSL kernel JIT-compiles through
> `nvidia-cutlass-dsl`. **Outside** the `lmsysorg/sglang:dev-cu13` container
> that compile can fail with `GPUModuleOp.__init__(): incompatible function
> arguments` — a cutlass-dsl / MLIR-binding version skew. When it does,
> `bench_indexer_cost.py` (which uses `torch.nn.functional.
> scaled_dot_product_attention` and hits the same FA family) still runs;
> `bench_sparse_attention_kernels.py`'s PARTs C and F are unaffected, but
> the dense `speedup` columns in PARTs A/D show `nan`. Run inside the dev
> container to get the dense baseline for the kernel bench.

> **Block-sparse kernel path.** The Triton block-sparse FA used as the
> LM sparse-attention step (PARTs B/D/E of the kernel bench) is **not part
> of upstream sglang** — it lives in an external package. Point the
> benchmarks at whatever directory contains that `sparse_attn/` package via:
>
> ```bash
> export SPARSE_ATTN_KERNELS_DIR=/path/to/kernels
> ```
>
> Without it, PARTs C and F still run (and PART A when the FA4 baseline
> compiles); the block-sparse parts skip. `bench_indexer_cost.py` does *not*
> need this — its sparse-FA measurement uses standard SDPA over the top-K
> selected KV slice.

### Turnkey image (fastest path)

If you have access to the public sglang dev image, every dependency above is
pre-installed:

```bash
docker pull lmsysorg/sglang:dev-cu13

docker run -dit --name sgl_diff \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/LLMDiveDeep:/workspace/LLMDiveDeep \
  -v $HOME/.cache:/root/.cache \
  lmsysorg/sglang:dev-cu13 bash

docker exec -it sgl_diff bash
cd /workspace/LLMDiveDeep/sparse_attn
```

---

## 3. Running the benchmarks

All commands run from `LLMDiveDeep/sparse_attn` (with the `.venv` active, or
inside the container).

### 3.1 `bench_indexer_cost.py` — the two-question headline bench (recommended first)

Self-contained. Requires `torch`, `deep_gemm`, `matplotlib`, and `sglang`
(for the fused Hadamard / `act_quant` kernels). Produces one table + one
two-panel PNG plot answering the two questions in `SURVEY.md §3`: (Q1) full
indexer vs sparse-FA cost, (Q2) sparse-FA vs dense-FA cost. The indexer is
measured **at two levels**:

- `mqa_logits` — the `deep_gemm.fp8_mqa_logits` score kernel alone (step 7 of
  the indexer pipeline).
- `full_indexer` — the whole per-decode-step pipeline (steps 1–8): `wq_b` +
  `wk` + `k_norm` + `weights_proj` + RoPE + Hadamard + FP8-quant +
  `fp8_mqa_logits` + `topk`.

`setup_overhead = full_indexer − mqa_logits − topk` isolates the "everything
except the score kernel" plumbing, which turns out to dominate below ~64 K
context.

```bash
# ~1 min: sweeps B ∈ {1, 16}, S ∈ {32K, 128K, 1M} with the DSv3.2 shape.
python bench_indexer_cost.py

# Custom sweep — measure only decode with a single batch:
python bench_indexer_cost.py --B 1 --S 8192 32768 131072 --warmup 5 --iters 25

# The full sweep with high iters for tight variance:
python bench_indexer_cost.py --B 1 16 --warmup 10 --iters 50
```

Outputs written to the current directory:

| File | What |
|---|---|
| `bench_indexer_cost.csv` | Raw timings (all five callables per `(B, S)` row). |
| `bench_indexer_cost.png` | Two log-log panels: indexer vs sparse; dense vs sparse. |

Available flags:

| Flag | Meaning | Default |
|---|---|---|
| `--B 1 16` | batch sizes to sweep (query count = B; each has its own S KV history) | `1 16` |
| `--S 32768 131072 1048576` | KV sequence lengths to sweep | `32K 128K 1M` |
| `--warmup / --iters` | GPU-event timing budget | `5 / 25` |
| `--csv / --plot` | output paths | `bench_indexer_cost.{csv,png}` |

To retarget a different model, edit the `Shape` dataclass at the top of the
script (`H_I`, `D_I`, `TOPK`, `H_A`, `D_A`, `H_KV`, `BLOCK_SIZE`).

### 3.2 `bench_sparse_attention_kernels.py` — the six-part kernel bench

Isolates the three primitives that make DSA / IndexShare run in isolation:
dense FA, block-sparse FA, and the lightning indexer. Also produces a
combined "indexer + sparse-FA vs dense-FA" table (PART D) and a **quality
table** measuring cos-sim of sparse-attention output vs dense at matched
sparsity (PART E). PART F adds a linear-attention (GDN) column.

Quick smoke test (~20 s):

```bash
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention_kernels.py --all \
    --seq_lens 4096 8192 16384 --warmup 2 --iters 5 --index_topk_freq 4
```

Full sweep (~2 min):

```bash
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention_kernels.py --all \
    --warmup 10 --iters 50 --index_topk_freq 4 2>&1 | tee kernel_bench.log
```

Uses the default `--seq_lens 512 1024 … 65536`.

Individual parts (each independently runnable):

```bash
python bench_sparse_attention_kernels.py --dense       # PART A: dense FA baseline (FA4)
python bench_sparse_attention_kernels.py --sparse_gqa  # PART B: block-sparse FA at fixed sparsity
python bench_sparse_attention_kernels.py --indexer     # PART C: lightning indexer alone
python bench_sparse_attention_kernels.py --combined    # PART D: headline (indexer + sparse) vs dense
python bench_sparse_attention_kernels.py --quality     # PART E: cos-sim sparse vs dense
python bench_sparse_attention_kernels.py --linear      # PART F: linear attention (GDN)
```

Useful flags:

| Flag | Meaning | Default |
|---|---|---|
| `--seq_lens 4096 8192 16384` | which sequence lengths to sweep | `512 … 65536` |
| `--sparsity 0.05` | fraction of KV blocks kept in PART B/D | `0.05` |
| `--sparsities 0.02 0.05 0.10 0.25` | sweep multiple sparsities in PART B | none |
| `--index_topk_freq 4` | IndexShare period (1 = DSv3.2/GLM-5, 4 = GLM-5.2) | `1` |
| `--warmup / --iters` | GPU-event timing budget | `10 / 50` |
| `--causal` | causal masking in PART A | non-causal |

### 3.3 `bench_sparse_attention_modules.py` — module-level bench

Wraps the same hot path in a real `nn.Module` mirroring the production
DSv3.2, GLM-5, and DSv4 indexers, so the timing includes projections,
LayerNorm, RoPE, and FP8 quant — everything a deployed model actually pays
for.

```bash
python bench_sparse_attention_modules.py --all \
    --seq_lens 4096 8192 16384 32768 --index_topk_freq 4
```

Individual modules:

```bash
python bench_sparse_attention_modules.py --dsv32 --index_topk_freq 1
python bench_sparse_attention_modules.py --glm   --index_topk_freq 4  # GLM-5.2 IndexShare
python bench_sparse_attention_modules.py --dsv4                       # DSv4 compressor path
```

---

## 4. Files in this directory

| File | What it is |
|---|---|
| `README.md` | This file — how to run every bench in this directory. |
| `SURVEY.md` | The narrative + benchmark results. Read this first for the "why". |
| `ATTENTION_MATH.md` | Math-forward deep-dive: sparse (DSA) and linear (GDN) formulas, cost breakdowns. |
| `requirements.txt` | Explicit dependency list. |
| `DSA.py` | Shared MLA + DSA implementation: paper-form and weight-absorbed MLA (`MLAReference`), lightning indexers (`DsaIndexer` / `Dsv4Indexer` / `C4Compressor`), FP8 quant, optional kernel backends. |
| `LinearAttention.py` | Companion to `DSA.py`: pure-torch gated-delta-rule reference for **GDN** (Qwen3-Next) and **KDA** (Kimi-Linear), plus the FLA-kernel wrappers. |
| **`bench_indexer_cost.py`** | **New — the two-question headline bench.** Self-contained; only depends on torch + deep_gemm + matplotlib. Produces the plot in `SURVEY.md §3.4`. |
| `bench_sparse_attention_kernels.py` | Six-part kernel-level bench driver. Imports impl from `DSA.py`. |
| `bench_sparse_attention_modules.py` | Module-level bench driver. Times the real `nn.Module` indexers from `DSA.py` at production shapes. |
| `bench_indexer_cost.csv` | Raw CSV data output by `bench_indexer_cost.py`. Regenerate by running the script. |
| `bench_indexer_cost.png` | Two-panel log-log plot output by `bench_indexer_cost.py`. |
| `../common/kernel_bench.py` | Shared kernel-bench framework (`bench_cuda` / `print_section` and more), reused by the `quantization` suite too. |

The model logic is intentionally visible in `DSA.py`: `MLAReference.project`
builds the compressed cache, `attend_absorbed` shows K/V weight absorption,
`DsaIndexer.forward` computes the lightning-indexer top-K, and
`DeepSeekSparseAttentionReference.forward` connects those pieces. Optimized
SGLang / DeepGEMM kernels are used only as interchangeable low-level backends.

---

## 5. Troubleshooting

- **"deep_gemm MISSING" in the env report**: PART C and the module bench will
  use a torch FP32 reference (~50× slower). Only expected on non-Blackwell /
  non-Hopper GPUs.
- **`bench_indexer_cost.py`: `SDPA … kernel not used` warnings.** SDPA
  chooses a backend based on shape and dtype; the warnings say which backends
  it *couldn't* pick — that's fine so long as at least one of
  FLASH_ATTENTION / CUDNN_ATTENTION did work. The script forces
  FLASH_ATTENTION where possible.
- **`bench_sparse_attention_kernels.py`: "flash_attn.cute MISSING"**:
  PART A / PART D / PART F dense baseline falls back to
  `sglang.jit_kernel.flash_attention_v4` (`sgl_fa4`, same FA4 kernel family).
  If that is also missing the dense column is skipped.
- **"sparse FA backend unavailable"**: the block-sparse Triton kernel isn't
  on the Python path. Set `SPARSE_ATTN_KERNELS_DIR` to the directory that
  contains the `sparse_attn/video_sparse_kernel.py` package. Without it,
  PARTs B/D/E of the kernel bench skip and the rest of the suite still runs.
- **"chunk_gated_delta_rule not importable"**: PART F is skipped. This is
  expected outside sglang containers; there's no FLA equivalent shipped
  independently.
- **OOM at 65 K+**: PART E and PART C's largest sequence lengths allocate
  large scratch tensors even after streaming. Reduce `--seq_lens` or drop
  PART E from the sweep. For `bench_indexer_cost.py` at $S = 1\text{M}$,
  $B = 16$: the dense-FA tensors are ~8 GB; if you're tight on VRAM run
  `--B 1` only.

---

## 6. Further reading

### In this directory

- `SURVEY.md` — model-by-model comparison of GQA / MLA / DSA (DSv3.2 / GLM-5
  / GLM-5.2 / DSv4 CSA + HCA), with the measured B200 numbers.
- `ATTENTION_MATH.md` — the sparse + linear attention formulas behind those
  numbers, with a to-read list.
- `DSA.py` — sparse (DSA) implementation; `LinearAttention.py` — linear
  (GDN + KDA) implementation.

### Papers & model cards

- DeepSeek V3 paper (MLA): https://arxiv.org/abs/2412.19437
- DeepSeek V3.2 report (DSA / lightning indexer): https://arxiv.org/abs/2509.19000
- DeepSeek V4 paper (CSA + HCA): https://arxiv.org/abs/2606.19348
- GLM-5 paper ("from Vibe Coding to Agentic Engineering"): https://arxiv.org/abs/2602.15763
- GLM-5.2 blog (IndexShare, ~2.9× at 1 M): https://z.ai/blog/glm-5.2
- IndexCache (empirical basis for IndexShare): https://github.com/THUDM/IndexCache
- Qwen3-Next blog (Gated Delta Net): https://qwenlm.github.io/blog/qwen3-next/
- Gated DeltaNet paper: https://arxiv.org/abs/2412.06464
- Kimi Linear technical report (KDA): https://arxiv.org/abs/2510.26692
- Kimi-Linear repo (open KDA kernel + vLLM): https://github.com/MoonshotAI/Kimi-Linear
- Kimi-Linear-48B-A3B model card: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- Kimi K2 technical report (MLA flagship): https://arxiv.org/abs/2507.20534
- Su Jianlin, *"From MHA, MQA, and GQA to MLA"* (MLA derivation): https://spaces.ac.cn/archives/10091

### Blogs & explainers

- DeepSeek Sparse Attention — Sebastian Raschka: https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/
- A visual guide to attention variants — Sebastian Raschka: https://magazine.sebastianraschka.com/p/visual-attention-variants
- DeepSeek Sparse Attention on GPU cloud (2026 guide) — Spheron: https://www.spheron.network/blog/deepseek-sparse-attention-long-context-llm-gpu-cloud/
- Kimi Linear / KDA hardware-aware algorithms — DigitalOcean: https://www.digitalocean.com/community/tutorials/kimi-linear-moonshot-ai
- Kimi K2.6 complete guide (2026) — Codersera: https://codersera.com/blog/kimi-k2-6-complete-guide-2026/
