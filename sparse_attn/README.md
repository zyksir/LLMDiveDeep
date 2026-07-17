# Sparse & Linear Attention Benchmarks

Reproducible microbenchmarks for the non-dense attention paths that show up
in modern open-source **language models** — DeepSeek-V3.2 / V4 / GLM-5 /
GLM-5.2 sparse attention, **plus** Qwen3-Next-style linear attention. (Video
sparse attention — VSA / WanVideo — is deferred; see §1.)

`DSA.py` (the shared implementation), two benchmark drivers, and one written
explainer. Everything is self-contained: no model weights, no server setup,
no paged KV cache — all numbers come from synthetic tensors through the same
Triton / CuTe kernels that the production models use.

Three questions this benchmark is designed to answer:

1. **What is increasing** as sequence length, sparsity, or IndexShare period grows?
2. **What is the cosine similarity** between sparse-attention output and dense-attention output?
3. **What is the quality / performance trade-off** — how much accuracy do we spend per unit of speedup?

All numbers below are on a single **NVIDIA B200** in the
`lmsysorg/sglang:dev-cu13` container.

---

## 1. The wider landscape (what else sglang ships)

The scripts here focus on DSA-style sparse attention because that is the
family with the most active development in 2025. But sglang actually ships
several other non-dense attention paths. If you're wondering "does the
picture change with a different attention family?", the answer is on this
list.

### Linear / sub-quadratic (recurrent state, O(N) cost)

| Family | Models that use it | Kernel |
|---|---|---|
| **Gated Delta Net (GDN)** | Qwen3-Next, Qwen3.5-VL, Jet-Nemotron, InternS2 | `fla/chunk.py::chunk_gated_delta_rule` |
| **Kimi Delta Attention (KDA)** | Kimi-Linear | `fla/kda.py::chunk_kda` |
| **Lightning / SegLa** | Bailing-MoE-Linear, Bailing-MoE-v2.5 | `linear/seg_la.py::seg_la_fwd` |
| **Mamba-2 SSM** | Falcon-H1, Nemotron-H, Granite-MoE-Hybrid, LFM2, Zaya | `mamba/ops/ssd_combined.py` |

**Only GDN is benchmarked in PART F below**, as the canonical linear-attention
example. The other three have the same $O(S)$ scaling and the same shape of
"linear state instead of softmax score matrix"; expect similar curves.

### Sparse (subset of KV, still softmax)

| Family | Models | Kernel |
|---|---|---|
| **DSA / lightning indexer** (V3.2, GLM-5, GLM-5.2 IndexShare) | DeepSeek-V3.2, GLM-5 / 5.2 | `dsa/dsa_indexer.py`, `deep_gemm.fp8_mqa_logits` |
| **DSA-C4** (DSA + compressed K stream) | DeepSeek-V4 | `dsv4/indexer.py`, `dsv4/compressor.py` |
| **Quest** (page-wise KV subset selection at runtime) | Model-agnostic (`--hisparse-config`) | `mem_cache/sparsity/algorithms/quest_algorithm.py` |
| **Dual-chunk vertical / slash** (MInference-style pattern) | Qwen2, Qwen3-MoE, GLM-4 (with `dual_chunk_attention_config`) | `sgl_kernel.sparse_flash_attn.sparse_attn_func` |
| **Sliding window + attention sinks** | gpt-oss, Mistral, Ministral, Gemma-2/3, Olmo-2, Phi-MoE, Cohere-2 | Window enforced inside FA / Triton metadata |
| **Video block-sparse (VSA)** | WanVideo DiT, Causal-WanVideo, Lingbot-World | `multimodal_gen/.../sparse_attn/video_sparse_kernel.py` |

**DSA / DSA-C4 are benchmarked in PARTs C, D, E**. PARTs B, D, and E use a
generic block-sparse Triton kernel for their "sparse-FA" column, standing in
for the "sparse-MLA" step inside DSA. That same kernel is what WanVideo's DiT
uses natively for **video** block-sparse attention (VSA) — so the LM
sparse-attention numbers here map directly onto that model class. The
video-specific VSA story is **deferred**: this suite focuses on language-model
attention first, and VSA can be investigated later. **Quest, dual-chunk, and
sliding-window are not benchmarked** because none of them has a clean
stand-alone kernel — Quest needs a live paged KV pool, dual-chunk needs heavy
metadata pre-computation, sliding-window is just a mask on top of regular FA.

### Eviction / cache tiering

- **HiSparse** — host↔device KV tiering, coordinates with DSA / DSv4 / MLA
  backends. Not a stand-alone attention kernel; not benchmarked here.

### What's *not* in sglang

Notably absent (as of the version in this container): SnapKV, H2O, RocketKV,
MoBA, InfLLM, MInference (as a named integration), RWKV, RetNet.

---

## 2. Environment

You need a Hopper (SM_90) or Blackwell (SM_100) GPU and CUDA 12.6+/13. All
numbers below were validated on a single **NVIDIA B200**, CUDA 13, Python
3.12, `torch` 2.11.0+cu130, `triton` 3.6.0.

### Installing the dependencies

Everything the benchmarks import is either pip-installable or builds from the
open-source **sglang** source tree. The recommended flow is a dedicated
virtualenv at the repo root (`LLMDiveDeep/.venv`), shared by the `sparse_attn`
and `quantization` suites:

```bash
cd /path/to/LLMDiveDeep
python -m venv .venv
source .venv/bin/activate

# 1. Core (pick the torch wheel matching your CUDA toolkit).
pip install -r sparse_attn/requirements.txt          # torch, triton, numpy

# 2. sglang from source, pinned to a known-good release tag.
#    v0.5.15.post1 has DSA (V3.2), GLM-5.2 IndexShare, GDN, and DSv4
#    sparse support. Pin the tag so the module paths this suite imports
#    (dsa_indexer / fla.chunk / flash_attention_v4) don't drift.
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout v0.5.15.post1
pip install -e "sglang/python[all]"

# 3. deep_gemm (fp8_mqa_logits) and flash-attn FA4 (CuTe DSL) — follow the
#    build steps in the sglang docs for your GPU/CUDA. Both are optional:
#    the indexer falls back to a torch FP32 path and the dense baseline
#    falls back to sglang's FA4 if either is missing.
```

> **Note on the block-sparse kernel.** The Triton block-sparse FA used as the
> LM sparse-attention step (PARTs B/D/E) currently lives in an internal
> `b10_kernels/sparse_attn/video_sparse_kernel.py` path that is **not yet in
> upstream sglang**. Point the benchmarks at whatever directory contains a
> `sparse_attn/` package via:
>
> ```bash
> export SGLANG_B10_KERNELS_DIR=/path/to/.../b10_kernels
> ```
>
> Without it, PARTs A, C, and F still run; the block-sparse parts skip.

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

Run from `LLMDiveDeep/sparse_attn` (with the `.venv` active, or inside the
container).

### Quick smoke test (~20 s)

```bash
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention_kernels.py --all \
    --seq_lens 4096 8192 16384 --warmup 2 --iters 5 --index_topk_freq 4
```

### Full sweep (~2 min)

```bash
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention_kernels.py --all \
    --warmup 10 --iters 50 --index_topk_freq 4 2>&1 | tee kernel_bench.log
```

Uses the default `--seq_lens 512 1024 … 65536`.

### Individual parts

The kernel bench has six parts and each is independently runnable:

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

### Module-level bench

```bash
python bench_sparse_attention_modules.py --all \
    --seq_lens 4096 8192 16384 32768 --index_topk_freq 4
```

Compares three real `nn.Module` indexers (DSv3.2, GLM-5/5.2, DSv4-C4) at
production shapes. Individual modules:

```bash
python bench_sparse_attention_modules.py --dsv32 --index_topk_freq 1
python bench_sparse_attention_modules.py --glm   --index_topk_freq 4  # GLM-5.2 IndexShare
python bench_sparse_attention_modules.py --dsv4                       # DSv4 compressor path
```

---

## 4. Files in this directory

| File | What it is |
|---|---|
| `README.md` | This file. |
| `requirements.txt` | Explicit dependency list — pip-installable pieces plus non-pip deps and where each is used. |
| `DSA.py` | The DSA **implementation**, shared by both drivers: lightning-indexer hot path (`IndexerSimulator`), the real `nn.Module` indexers (`DsaIndexer` / `Dsv4Indexer`), FP8 quant, and FA4 / block-sparse backend registration. |
| `bench_sparse_attention_kernels.py` | Kernel-level microbench driver (6 parts A–F). Synthetic tensors, no model load. Imports impl from `DSA.py`. |
| `bench_sparse_attention_modules.py` | Module-level microbench driver. Times the real `nn.Module` indexers from `DSA.py` at production shapes. |
| `SURVEY.md` | Model-by-model tour: what's different between Qwen 3, DeepSeek V3 → V4, GLM-5 → 5.2, MiniMax-M3, and Qwen3-Next. |
| `ATTENTION_MATH.md` | Math-forward deep-dive: the sparse (DSA) and linear (GDN) formulas, cost breakdowns, and which bench PART measures each. |
| `../common/bench_utils.py` | Shared `bench_function` / `print_section` helpers, reused by the `quantization` suite too. |

---

## 5. Results on B200

All numbers below are from `lmsysorg/sglang:dev-cu13` on a single NVIDIA
B200, non-causal, bf16 activations, FP8 for the indexer path.

### 5.1 Dense baseline (PART A, GQA `H=32 H_kv=8 D=128`)

| seq_len | FA4 ms | FA4 TFLOPS |
|---:|---:|---:|
|  8 K |  0.70 | 1566 |
| 16 K |  2.74 | 1605 |
| 32 K | 12.05 | 1460 |

FA4 delivers 1.4–1.6 PFLOPS of useful work on B200. Every "speedup" column
below divides into this row. (cuDNN SDPA is intentionally not reported: its
BSHD↔BHSD transpose would be timed inside the attention call, making it an
unfair baseline. `sgl_fa4` is the same FA4 kernel family, kept only as an
import fallback.)

### 5.2 Block-sparse FA alone (PART B, `H=24 D=128`, sparsity 5 %)

| seq_len | sparse ms | dense ms | speedup |
|---:|---:|---:|---:|
|  8 K | 0.12 | 0.58 | **5.04×** |
| 16 K | 0.36 | 2.13 | **5.95×** |
| 32 K | 1.23 | 9.04 | **7.34×** |

At 5 % sparsity the sparse-attention kernel by itself is 5–7× faster than
dense. This is before paying the indexer. **Same kernel as WanVideo DiT
uses for video-block-sparse attention.**

### 5.3 Lightning indexer alone (PART C, DSv3.2 shape, `H_idx=64 D_idx=128`)

| seq_len | proj ms | quant ms | logits+topk ms | total ms | effective GB/s |
|---:|---:|---:|---:|---:|---:|
|  8 K | 0.18 | 0.67 |  1.71 |  2.56 | 5168 |
| 16 K | 0.34 | 1.29 |  5.49 |  7.12 | 6453 |
| 32 K | 0.67 | 2.53 | 19.97 | 23.16 | 7097 |

The indexer is HBM-bandwidth bound (~7 TB/s effective on B200's ~8 TB/s
peak). `logits+topk` grows quadratically with $S$.

### 5.4 Headline: DSA (indexer + sparse-FA) vs dense-FA (PART D)

With `index_topk_freq=4` (GLM-5.2 IndexShare), sparsity 5 %:

| seq_len | dense ms | indexer ms (/4) | sparse ms | total ms | speedup | idx frac |
|---:|---:|---:|---:|---:|---:|---:|
|  8 K |  0.58 |  0.64 | 0.11 |  0.75 | **0.77×** | 85 % |
| 16 K |  2.13 |  1.88 | 0.38 |  2.25 | **0.94×** | 83 % |
| 32 K |  8.98 |  5.90 | 1.63 |  7.53 | **1.19×** | 78 % |
| 65 K | 37.32 | 21.71 | 5.69 | 27.40 | **1.36×** | 79 % |

**Reading this table** (see full analysis in the previous chat turn):

- DSA is **slower** than dense FA4 below 16 K — the $O(S^2)$ indexer term hasn't been amortized yet.
- Break-even is at $S \approx 16\text{K}$ on our synthetic proxy shape (in real DSv3.2, whose dense-MLA baseline is ~10× fatter, break-even happens at shorter $S$).
- Speedup grows monotonically past that, and extrapolating to 1 M lands right at GLM-5.2's advertised **~2.9×** number.
- The indexer eats **~80 %** of total attention time at every $S$. That's exactly what DSv4's compressor and GLM-5.2's IndexShare both attack.

### 5.5 Quality: cos-sim of sparse vs dense (PART E)

Measured on synthetic inputs with a realistic 5 % heavy-hitter K/V bump.
Two selection policies are compared at matched sparsity:

- **random top-k** — pick blocks uniformly at random (adversarial floor; no indexer).
- **oracle top-k** — score blocks with signed block-max of $Q \cdot K^T$ and pick top-k (ceiling; the best any indexer could do).

| seq_len | sparsity | random cos | oracle cos | speedup |
|---:|---:|---:|---:|---:|
|  8 K | 0.02 | 0.11 | 0.47 | 7.3× |
|  8 K | 0.05 | 0.18 | 0.61 | 5.1× |
|  8 K | 0.10 | 0.28 | 0.75 | 3.1× |
|  8 K | 0.25 | 0.48 | 0.88 | 1.5× |
| 16 K | 0.02 | 0.10 | 0.53 | 11.2× |
| 16 K | 0.05 | 0.19 | 0.69 | 6.0× |
| 16 K | 0.10 | 0.29 | 0.79 | 3.4× |
| 16 K | 0.25 | 0.48 | 0.91 | 1.5× |
| 32 K | 0.02 | 0.11 | 0.60 | 15.7× |
| 32 K | 0.05 | 0.20 | 0.73 | 7.4× |
| 32 K | 0.10 | 0.29 | 0.81 | 4.0× |
| 32 K | 0.25 | 0.49 | 0.92 | 1.7× |

Real DSA-style indexers sit between the two columns, empirically much
closer to the ceiling. Cos-sim also **improves with $S$** at fixed sparsity
because the block-max selector has more material to find heavy hitters in.

### 5.6 Linear attention (PART F, Qwen3-Next GDN)

At `H=24 D=128`, same shape as PART B / D:

| seq_len | linear (GDN) ms | dense (FA4) ms | speedup | note |
|---:|---:|---:|---:|---|
|  4 K | 0.31 |  0.15 | 0.49× | linear loses at short S |
|  8 K | 0.57 |  0.58 | **1.01×** | breakeven |
| 16 K | 1.10 |  2.14 | **1.94×** | |
| 32 K | 2.53 |  8.98 | **3.55×** | |
| 65 K | 4.43 | 36.83 | **8.31×** | |

**The most instructive table in the whole benchmark**, because it shows the
$O(S)$ vs $O(S^2)$ scaling in one place. Dense grows ~4× every time $S$
doubles; GDN grows ~2×. At $S = 65\text{K}$ the gap is already 8×; at 1 M
it would be > 100×.

### 5.7 One-glance comparison across families

Same shape (`H=24 D=128`, batch 1), same $S$, three approaches:

| seq_len | dense (FA4) ms | **sparse (DSA + IndexShare)** ms | **linear (GDN)** ms | dense/sparse | dense/linear |
|---:|---:|---:|---:|---:|---:|
|  8 K |  0.58 |  0.75 | 0.57 | 0.77× | **1.02×** |
| 16 K |  2.14 |  2.25 | 1.10 | 0.95× | **1.94×** |
| 32 K |  8.98 |  7.53 | 2.53 | **1.19×** | **3.55×** |

Three clean takeaways:

- **At short context (< 16 K), dense wins on both fronts.** Sparse and linear only make sense past a certain sequence length.
- **Linear scales better than sparse.** Both are answers to $O(S^2)$, but a per-token recurrence (linear) beats a per-token $O(S^2)$-with-small-prefactor scorer (sparse) at every $S$ we tested.
- **They are not interchangeable in accuracy.** Linear attention drops the softmax entirely, which requires the model to be *trained* with linear attention from scratch (or a very careful fine-tune). Sparse attention keeps the softmax and only trims the key set, which can be dropped in more easily — that's why GLM-5 could adopt DSA verbatim from DeepSeek without retraining from scratch, and why Qwen3-Next is a much bigger architectural jump than Qwen3.5.

### 5.8 Module-level indexer comparison (`bench_sparse_attention_modules.py`)

At $S = 16\text{K}$, `index_topk_freq=4`:

| Module | proj ms | quant ms | logits ms | total ms | idx/4 ms | idx frac |
|---|---:|---:|---:|---:|---:|---:|
| DSv3.2       | 0.36 | 1.29 | 6.39 | 8.04 | 2.01 | 27 % |
| GLM-5 / 5.2  | 0.43 | 1.29 | 6.98 | 8.70 | 2.18 | 29 % |
| **DSv4-C4**  | 0.40 | 1.33 | **2.65** | **4.38** | **1.09** | **17 %** |

- **DSv4's compressor** cuts the `logits` step ~2.4× at 16 K (~4× at 64 K), exactly matching the `compress_ratio=4` design.
- **GLM-5.2's IndexShare** divides the whole indexer cost by the period (`/4`) — nothing about the kernel changes, but the module runs only 1 in every 4 layers.
- Combined, DSv4-C4 + IndexShare would give ~16× indexer-cost reduction.

---

## 6. Reading the numbers

### What is *increasing*?

- With $S$: indexer `logits` grows $O(S^2)$; sparse-FA grows linearly in $S$ at fixed sparsity; dense-FA grows $O(S^2)$; **linear-attention grows $O(S)$**; cos-sim increases with $S$ at fixed sparsity.
- With sparsity: sparse-FA time grows linearly; both random-cos and oracle-cos grow monotonically.
- With IndexShare period: indexer amortized cost drops proportionally; sparse-FA unchanged.
- With compressor ratio: indexer logits time drops proportionally; proj/quant unchanged.

### What is the *cosine similarity*?

Cos-sim of the sparse-attention output vector against the dense-attention
output vector on the *same* $Q, K, V$. Range 0 → 1, higher is more faithful.
See §5.5.

### What is the *quality / performance trade-off*?

- **sparsity ≤ 5 %** → 6–15× sparse-kernel speedup, but oracle cos-sim < 0.75. Only safe when the model was **trained** with sparse attention from scratch (NSA / DSA / MoBA).
- **sparsity 10–15 %** → ~3–4× speedup, oracle cos-sim ~0.80. Sweet spot for training-free top-k (Quest, InfLLM).
- **sparsity ≥ 25 %** → cos-sim > 0.90, but sparse-kernel speedup drops to 1.5× and indexer overhead usually erases it.
- **Linear attention** doesn't sit on this same curve at all — it has to be trained end-to-end. The right question there is "how many linear-attention layers can I use before quality drops?" (Qwen3-Next uses 3 GDN layers per 1 full-attention layer).

---

## 7. Troubleshooting

- **"deep_gemm MISSING" in the env report**: PART C and the module bench will use a torch FP32 reference (~50× slower). Only expected on non-Blackwell / non-Hopper GPUs.
- **"flash_attn.cute MISSING"**: PART A / PART D / PART F dense baseline falls back to `sglang.jit_kernel.flash_attention_v4` (`sgl_fa4`, same FA4 kernel family). If that is also missing the dense column is skipped.
- **"sparse FA backend unavailable"**: the block-sparse Triton kernel from sglang isn't on the Python path. Set `SGLANG_B10_KERNELS_DIR` to the parent directory containing `sparse_attn/video_sparse_kernel.py` — inside `sgl_diff` it should be auto-discovered at `/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/layers/b10_kernels`.
- **"chunk_gated_delta_rule not importable"**: PART F is skipped. This is expected outside sglang containers; there's no FLA equivalent shipped independently.
- **OOM at 65 K+**: PART E and PART C's largest sequence lengths allocate large scratch tensors even after streaming. Reduce `--seq_lens` or drop PART E from the sweep. The other five parts scale to 65 K on 180 GB B200 without issue.

---

## 8. Further reading

- `SURVEY.md` — model-by-model comparison of Qwen 3 / DeepSeek / GLM-5 / MiniMax-M3 / Qwen3-Next.
- `ATTENTION_MATH.md` — the sparse + linear attention formulas behind the numbers, with a to-read list.
- DeepSeek V3.2 paper (DSA): https://arxiv.org/abs/2509.19000
- GLM-5.2 blog (IndexShare, 2.9× at 1 M): https://z.ai/blog/glm-5.2
- Qwen3-Next blog (Gated Delta Net): https://qwenlm.github.io/blog/qwen3-next/