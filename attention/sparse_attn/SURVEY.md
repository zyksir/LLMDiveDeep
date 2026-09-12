# Sparse Attention with MLA — DeepSeek and GLM

> Historical study / retained source context. Start with the [kernel tutorial](../README.md)
> and [current reading guide](compressed_sparse_attention.md). Old CSV/log/PNG and local_results artifacts
> were removed for regeneration; measurements below were not rerun or revalidated.
> Future outputs belong under attention/results/, not the paths in old examples.
> Layer/projection benchmarks below are outside the new attention-core comparisons.


*A human-readable tour of the DSA family: what changes from plain **GQA**, to
**MLA**, to **MLA + top-K sparse attention** (DSv3.2 / GLM-5 / DSv4). Focuses
on the one thing that is actually new in the sparse-attention wave — the
**lightning indexer** — first at the math level, then in real SGLang code, then
with measured cost curves on a **B200**.*

The three orthogonal branches (linear attention, sliding-window, block-sparse
on plain GQA) are summarised in the appendix; they are *not* variants of the
same idea and the reader interested in "why is DSA fast" can skip them.

---

## 0. Introduction

Every LLM you can download today has to answer one hard question: how do we
run attention at $S \gtrsim 128\text{K}$ context without paying the naïve
$O(S^2)$ compute and per-token $O(H_{kv} \cdot D)$ KV cache? Two families of
answers have converged in 2025–2026:

1. **Reduce the KV cache** (per-token state), keep the score matrix dense —
   DeepSeek-V2/V3 **MLA (Multi-head Latent Attention)**.
2. **Only compute part of the score matrix** — Deep­Seek-V3.2 **DSA (DeepSeek
   Sparse Attention)**, adopted verbatim by GLM-5 and extended in DSv4.

These stack: DSA is *always run on top of MLA*. This document reads them in
that order.

---

## 1. Baseline: from GQA to MLA

### 1.1 GQA — the pre-2024 standard

Plain grouped-query attention. Each head has its own Q, and $H_{kv}$ KV heads
are shared by groups of Q heads (Llama-3.1 uses 64 Q / 8 KV; Qwen-3 8 Q / 2
KV). Nothing sparse, nothing latent. Per token per layer the cache holds:

$$
\underbrace{H_{kv}\cdot D_h}_{K}\;+\;\underbrace{H_{kv}\cdot D_h}_{V}
\quad\text{bf16 elements}.
$$

At $H_{kv}=8$, $D_h=128$ that's ≈ 4 KB / token / layer, or **≈ 480 MB** for a
64-layer model at $S=$ 128 K. The score matrix and the FA kernel both scale
$O(S^2)$. This is the "before" point for every subsequent trick.

### 1.2 MLA — cache the latent, not per-head K/V

MLA (introduced in DeepSeek-V2, kept unchanged in V3) does not change the
attention score matrix. It changes what you *cache* per token, cutting cache
size by ~7× at the same head count. Su Jianlin's blog [*"From MHA, MQA, and
GQA to MLA"*](https://spaces.ac.cn/archives/10091) is the reference derivation;
the rest of this section restates it in three ideas.

**Idea 1 — one shared latent for K and V.** Introduce a low-rank projection
$W^{DKV}\in\mathbb{R}^{d_c\times d}$ with $d_c \ll H\cdot D_h$ (DSv3 uses
$d_c=512$; Qwen3-like GQA would use $H_{kv}\cdot D_h = 1024$). Cache only

$$
c_i^{KV}=W^{DKV}x_i \in \mathbb{R}^{d_c}
$$

instead of the $2H_{kv}D_h$ K/V vectors. When you need K or V for head $h$,
reconstruct on the fly:

$$
k_{i,h}^{C}=W_h^{UK}c_i^{KV},
\qquad
v_{i,h}=W_h^{UV}c_i^{KV}.
$$

You have traded a small on-the-fly matmul (per-token, per-head) for a
$2H_{kv}D_h/d_c \approx 2\times$ smaller cache.

**Idea 2 — weight absorption removes the reconstruction at inference.** The
attention score is

$$
(q_{t,h}^{C})^\top k_{i,h}^{C}
=(q_{t,h}^{C})^\top W_h^{UK} c_i^{KV}
=\Bigl((W_h^{UK})^\top q_{t,h}^{C}\Bigr)^\top c_i^{KV}.
$$

So instead of materialising $k_{i,h}^{C}$ once per token, you can bake the
K up-projection into the Q side once, and dot the *transformed Q* directly
against the cached latent. In closed-book decoding the cache is literally just
$c_i^{KV}$; there is no per-head K to store. The V branch admits the same
trick — the O-projection can absorb $W^{UV}$ so no per-head V ever materialises
either. Cache size stops depending on $H$.

**Idea 3 — RoPE side branch.** Absorption breaks the moment you try to rotate
K. RoPE inserts a position-dependent rotation $R_{i-t}$ between Q and K:

$$
(q_{t,h}R_t)^\top(k_{i,h}R_i)
=q_{t,h}^\top R_{i-t}k_{i,h},
$$

and $R_{i-t}$ does not commute with $W_h^{UK}$, so we can't precompute a single
"absorbed Q'". DeepSeek's solution: **leave the big content branch unrotated
and give RoPE its own small K branch shared across heads**. The final
per-head Q and K are concatenations:

$$
\begin{aligned}
q_{i,h}&=\bigl[\,W_h^{UQ}c_i^{Q}\,;\ R_i W_h^{QR}c_i^{Q}\,\bigr],\\
k_{i,h}&=\bigl[\,W_h^{UK}c_i^{KV}\,;\ R_i W^{KR}x_i\,\bigr],\\
v_{i,h}&=W_h^{UV}c_i^{KV}.
\end{aligned}
$$

The first slice ($d_c$ wide) is absorbed and doesn't need to be cached
per-head; the second slice (small, DSv3 uses $d_r=64$) *is* rotated and cached
once per token, shared across all heads. Effective cache per token:

$$
\underbrace{d_c}_{c^{KV}}\;+\;\underbrace{d_r}_{\text{RoPE K}}
= 512+64 = 576\ \text{bf16 elements}\;\approx\;\textbf{1.15 KB / token / layer}.
$$

That is roughly **7× smaller** than an equivalent 64-Q / 8-KV GQA layout, and
crucially **independent of the number of Q heads**. MLA is what makes running
DeepSeek-V3 (128 attention heads, 61 layers) at 128 K context viable at all.

For the rest of this document, when we say "the main attention" it means MLA
in the absorbed form — MQA-shaped at the kernel level: many Q heads, one KV
head of $d_c+d_r=576$ dims.

---

## 2. Sparse Attention: top-K KV selection

### 2.1 The core idea

MLA fixed the cache. It did *not* fix compute: every layer still evaluates
attention against all $S$ past tokens. At $S=1\text{M}$, even MLA's
memory-efficient MQA form spends most of a decode-time layer reading the
1 M-entry KV latent.

DSA's observation is unremarkable in retrospect: **most of those $S$ scores
are near-zero and can be dropped without changing the softmax output**.
Formalising:

```
scores       = lightning_indexer(q_index, k_index_cache)  # [T, S]  cheap
top_k_idx    = scores.topk(K=2048, dim=-1).indices        # [T, K]
attn_output  = MLA(q, kv_latent[top_k_idx])               # only K keys read
```

Every model in the DSA family (DSv3.2, GLM-5, GLM-5.2, DSv4 CSA) is a variation
of this three-liner. Both **the sparse-attention step** and **MLA itself** are
unchanged from V3 — only the top-K keys change. What is *new* is the indexer.
So the rest of §2 zooms into just that.

### 2.2 The indexer, mathematically

The **lightning indexer** is a tiny attention-shaped scorer that runs in
parallel with MLA every layer. It has three learned projections and one
non-standard aggregation:

$$
\begin{aligned}
q_{t,r}^{I}&=\operatorname{RoPE}_t(W_{Q,r}^{I}\,c_t^Q)
&&\text{(indexer Q, one per head }r=1..h_I\text{)},\\
k_i^{I}&=\operatorname{RoPE}_i(W_K^{I}x_i)
&&\text{(indexer K, single MQA head, shared across }r\text{)},\\
g_t&=W_g\,x_t/\sqrt{h_I}\in\mathbb{R}^{h_I}
&&\text{(per-head gate).}
\end{aligned}
$$

The score for query $t$ against past key $i$ is:

$$
\boxed{\;
I_{t,s}\;=\;\sum_{r=1}^{h_I}g_{t,r}\cdot\operatorname{ReLU}\!\bigl(\langle q_{t,r}^{I},\,k_s^{I}\rangle\bigr)
\;}
\qquad
\mathcal{T}_t=\operatorname{TopK}_{s\le t}(I_{t,s},\,K).
$$

Three things to notice, because they are why the indexer is cheap:

1. **Single K head (MQA).** One $k_s^{I}$ is shared across all $h_I$ query
   heads. Only one 128-dim K per token needs to be cached alongside MLA's
   latent.
2. **Small dimensions.** $h_I=64$ (DSv3.2) or 32 (GLM-5); $D_I=128$. So the
   indexer's Q is roughly $64\times128=8{,}192$ dims — a *fifth* of the main
   attention's Q ($128\times 576 \approx 74\text{K}$).
3. **No softmax.** ReLU-weighted head-sum, not softmax-weighted. Since top-K is
   invariant to monotonic post-processing, softmax over $S$ would be wasted
   work. This one design decision removes an $O(S)$ reduction from the hot
   path.

Cost model (per layer, decode with $T$ queries against $S$ cached KV):

$$
\underbrace{O(T\cdot h_I\cdot D_I\cdot S)}_{\text{indexer FLOPs, memory-bound in FP8}}
\;+\;
\underbrace{O(T\cdot H_{MLA}\cdot D_{MLA}\cdot K)}_{\text{main MLA, }K\ll S}.
$$

The indexer is HBM-bandwidth-bound because it must read all $S$ cached
indexer-Ks; that's why the DeepGEMM `fp8_mqa_logits` kernel does everything —
the FP8 MMA, the ReLU, the per-head gate, and the head-sum — in one epilogue.

### 2.3 The indexer, in real SGLang code

The reference implementation lives in
`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`. The single
`Indexer.forward_cuda` runs the pipeline below every layer that uses DSA:

| # | Step | Kernel(s) | Notes |
|---|---|---|---|
| 1 | `wq_b(q_lora) → query`  | GEMM | Q up-projection from MLA's compressed Q latent $c^Q$. |
| 2 | `wk(x) → key`; `k_norm` | GEMM + LayerNorm | Indexer K down-projection. Single K head shared across all $h_I$ Q heads. |
| 3 | RoPE on Q's & K's leading `rope_head_dim=64` slice | `rotary_emb` + write-back | In DSA, three separate small kernels (rotate Q, rotate K, in-place write-back). |
| 4 | Hadamard on Q, K | `rotate_activation` (128-pt Hadamard) | Separate kernel; enhances FP8 numerics. |
| 5 | FP8-quantise Q; store FP8-quantised K into paged Index-K cache | `act_quant` + `fused_store_index_k_cache` | The latter fuses per-block absmax → scale → e4m3 quant → strided page write. |
| 6 | `weights_proj(x) → head-gate` | small GEMM + `softplus` | Produces the $g_{t,r}$ tensor and pre-multiplies $q_\text{scale}\cdot d_I^{-1/2}$ into it. |
| 7 | **`deep_gemm.fp8_mqa_logits`** | **the one kernel that matters** | FP8 WGMMA `Q · Kᵀ` + fused epilogue: ReLU, per-head gate multiply, head-sum, KV-scale rescale. Emits `logits[T, S]` in a single kernel. |
| 8 | `torch.topk(logits, K=2048, dim=-1)` | radix sort | Returns the token indices consumed by the main MLA attention. |

The Q-side is unfortunately **not fused** in DSA today: RoPE, Hadamard, and
FP8-quant are three separate kernels (`rotary_emb`, `rotate_activation`,
`act_quant`), plus a write-back. DSv4 shipped the fully-fused variant
`fused_q_indexer_rope_hadamard_quant` on both the Q and K sides
(`python/sglang/srt/layers/attention/dsv4/indexer.py:697-711`), which the DSA
path has *not* been back-ported to. Rough impact: about **five separate
kernel launches** on the DSA Q path where DSv4 has one, saving a few
launch-overhead-plus-HBM-round-trips at short T. See the "what is fused"
walk-through in an earlier chat turn for the full breakdown.

Two things worth stressing that fall out of the cache design:

- **Only K, not V, is cached for the indexer.** `fp8_mqa_logits` needs only Q
  and K; the scores are aggregated across the $h_I$ heads without ever mixing
  values. So the Index-K cache stores 128 fp8 + 4-byte scale = **132 B / token
  / layer** — ~11 % of MLA's own 1152 B / token / layer.
- **Softmax is intentionally absent.** The DeepGEMM kernel does WGMMA →
  `fmaxf(x, 0)` → gate multiply → head-sum → write. No exponentiation, no
  cross-lane $S$-dim reduction. See `sm90_fp8_paged_mqa_logits.cuh:301-304`
  (or the SM100 variant at `sm100_fp8_paged_mqa_logits.cuh:392-395`).

A ~200-line stand-alone version of the same math lives in `DSA.py::DsaIndexer`
in this directory — same shapes, same DeepGEMM call, no paged KV cache. Useful
as a mental model when reading the production code.

### 2.4 What varies across models

Everything below is a bolt-on to the §2.2 skeleton; the score formula, the
top-K, and the sparse MLA step are unchanged.

| Model | What changes vs §2.2 | Cost impact |
|---|---|---|
| **DSv3.2** | canonical (all §2.2 defaults) | baseline |
| **GLM-5 / 5.1** | halves the indexer head count (`index_n_heads=32`) alongside main-attention head count; RoPE is GPT-J interleaved instead of NeoX | ~2× cheaper indexer per layer |
| **GLM-5.2** | **IndexShare**: run the indexer only on 1 in every 4 layers; the sparse-attention step still runs on every layer, reusing the previous top-K | indexer amortised /4 → ~2.9× per-token FLOPs @ 1 M in GLM's own numbers |
| **DSv4 CSA** ($m=4$) | compresses K by 4× *before* the indexer via `Compress_m` (softmax-gated pool with per-slot APE, not a mean pool); indexer scores compressed entries; sparse core reads compressed entries directly. A 128-token uncompressed sliding-window attention (SWA) branch runs in parallel and handles the tail ($S \bmod 4$) | ~4× indexer scan; ~4× main-attn KV memory |
| **DSv4 HCA** ($m'=128$) | compresses K by 128× and **drops the indexer entirely** — every query attends *densely* to all $\lceil S/128\rceil$ compressed entries. Same SWA branch as CSA | at $S = 1\text{M}$, 7.8 K compressed entries → dense global memory becomes cheap |
| **DSv4 (Pro, 61 L)** | alternates CSA / HCA (layers 0–1 HCA bootstrap; 2–60 CSA/HCA; final MTP block SWA-only). Every CSA/HCA layer's output is `compressed_branch + swa_branch` | ~4× total FLOPs reduction vs DSA at 128 K |

References: DSv3.2 report (arxiv `2509.19000`), DSv4 paper (arxiv `2606.19348`),
GLM-5.2 blog ([z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)), IndexCache
empirical basis ([THUDM/IndexCache](https://github.com/THUDM/IndexCache)).

---

## 3. Benchmarks on B200

All numbers below were captured by `bench_indexer_cost.py` on a single
**NVIDIA B200**, using synthetic tensors — no model weights. The script is
self-contained; only dependencies are `torch`, `deep_gemm`, and `matplotlib`.

### 3.1 Setup

**Shape.** MLA-absorbed **MQA**: 32 query heads, 1 KV head, head-dim 128
(simplified — real DSv3.2 has 128 Q heads and $d_c+d_r=576$; the *ratios*
generalise, the absolute numbers do not). Indexer follows DSv3.2 defaults:
$h_I=64$, $D_I=128$, top-K = 2048.

**Scenario.** Decode-time attention: $Q_{\text{len}} = 1$ per request, batch
$B \in \{1, 16\}$, KV history length $S \in \{32\text{K}, 128\text{K},
1\text{M}\}$. We *deliberately skip* $S \le 8\text{K}$; below that, the
indexer isn't amortised and the story is misleading. Prefill (Q_len = S) is
not measured here — the $O(S^2)$ savings picture there is qualitatively
similar but the absolute numbers differ by a factor of $S$.

**Six hot-path callables timed per $(B, S)$.** All timings via CUDA events,
trimmed mean of 25 iterations after 5 warm-up.  We measure the indexer at
**two levels** so the reader can see what "indexer cost" really includes:

| Callable | What it measures |
|---|---|
| `mqa_logits`     | `deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w, ks, ke)` — the fp8 WGMMA + ReLU+gate+head-sum epilogue only.  **Step 7 of §2.3** in isolation. |
| `topk`           | `logits.topk(K=2048, dim=-1)` — radix sort on the score matrix.  Step 8. |
| **`full_indexer`** | **Steps 1–8 all together**: `wq_b` + `wk` + `k_norm` + `weights_proj` + RoPE + Hadamard (sglang's `rotate_activation`) + FP8 `act_quant` + `fp8_mqa_logits` + `topk`.  This is what a deployed DSA layer actually spends per decode step. |
| `gather`         | `k[topk_idx]`, `v[topk_idx]` — building the sparse KV slice at decode. |
| `sparse_fa`      | PyTorch SDPA (`FLASH_ATTENTION`, MQA via `enable_gqa=True`) over the K=2048 KV slice. |
| `dense_fa`       | The same PyTorch SDPA over all $S$ KV tokens (the "no DSA" baseline). |

We derive **`setup_overhead = full_indexer − mqa_logits − topk`** as an
implicit column — the cost of everything *except* the score kernel and top-K,
i.e. all the projections / RoPE / Hadamard / FP8-quant plumbing.

We use PyTorch SDPA rather than FA4-cute because the cutlass-DSL JIT fails
outside the `lmsysorg/sglang:dev-cu13` container (documented in
`README.md §2`, `flash_attn.cute` `GPUModuleOp` MLIR skew). SDPA hits the
same FlashAttention family on Blackwell and gives correct scaling.

### 3.2 Q1 — how much of DSA time is the indexer?

At every point we measured, **the (full) indexer dominates DSA end-to-end time
by ≥ 15× over sparse-FA**, and the ratio grows linearly with $S$:

| B | S | mqa_logits ms | topk ms | setup ms | **full indexer ms** | sparse_fa ms | **idx / sparse** | indexer share of DSA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  |  32 K | 0.090 | 0.096 | 0.287 |  0.473 | 0.031 |   **15.4×** | 90 % |
| 1  | 128 K | 0.339 | 0.106 | 0.314 |  0.760 | 0.027 |   **28.2×** | 94 % |
| 1  |   1 M | 2.671 | 0.197 | 0.301 |  3.169 | 0.029 |  **109.7×** | 98 % |
| 16 |  32 K | 0.091 | 0.106 | 0.284 |  0.481 | 0.030 |   **16.1×** | 87 % |
| 16 | 128 K | 0.344 | 0.129 | 0.292 |  0.765 | 0.027 |   **27.9×** | 92 % |
| 16 |   1 M | 2.708 | 0.302 | 0.302 |  3.312 | 0.028 |  **118.3×** | 98 % |

Three reasons the ratio blows out with $S$:

- **`fp8_mqa_logits` scales linearly with $S$.** It is fundamentally HBM-bound:
  reads all $S$ FP8 K entries (128 B/token + 4 B scale ≈ 132 B/token) plus
  writes the `[T, S]` logits. At $S = 1\text{M}$ that's ≈ 130 MB of K per
  query and 4 MB of logits per query. The measured 2.7 ms at $S = 1\text{M}$
  corresponds to ~50 GB/s effective bandwidth — well below B200's ~8 TB/s
  peak, suggesting the ragged `fp8_mqa_logits` path (T ∈ {1, 16}) is *not*
  the fully-optimised decode path (that would be `fp8_paged_mqa_logits` with
  proper block metadata). Realistic decode-mode DSA in SGLang lands closer
  to 1–1.5 ms at $S = 1\text{M}$ for the score kernel.
- **The setup overhead is $\approx$ 0.3 ms and essentially constant in both
  $B$ and $S$.** Setup only touches the $T = B$ new tokens (projections,
  norm, RoPE, Hadamard, quant), so it scales with $T$ — but even at $B = 16$
  the compute is trivial. What dominates instead is the *number of kernel
  launches*: seven separate ops on the DSA Q path plus a few on K (see the
  "what is fused" table in §2.3 — DSv4 fuses these into one, DSA has not).
  At 5–10 µs per launch that's already 100–200 µs of pure overhead, matching
  the measured ~0.29 ms floor. This is exactly the hot spot
  `fused_q_indexer_rope_hadamard_quant` was written to close (DSv4 only).
- **Sparse-FA is constant in $S$.** It always attends to exactly K = 2048
  tokens, no matter how large the underlying KV history is. The measured
  27–31 µs is essentially SDPA launch overhead — the K = 2048 workload is
  too small to saturate the kernel.

**Setup vs score, at a glance.** For the record, this is how much of the
full indexer time each piece consumes:

| B | S | setup % of indexer | mqa_logits % of indexer | topk % of indexer |
|---:|---:|---:|---:|---:|
| 1  |  32 K | **61 %** | 19 % | 20 % |
| 1  | 128 K |   41 %   | **45 %** | 14 % |
| 1  |   1 M |   10 %   | **84 %** |  6 % |
| 16 |   1 M |    9 %   | **82 %** |  9 % |

At **short $S$** (32 K), the projections + norm + RoPE + Hadamard + FP8-quant
*plumbing* is the majority of indexer time; the score kernel is only a fifth.
At **long $S$** (1 M), the score kernel dominates and the plumbing fades to
<10 %. This is the empirical case for a fused Q-side kernel: below ~64 K
context, most of DSA's indexer time is *not* the score kernel and *is*
recoverable by fusion.

### 3.3 Q2 — how much does sparse-FA save vs dense-FA?

Sparse-FA is *constant* in $S$; dense-FA grows with $S$. The savings therefore
grow monotonically with context length:

| B | S | sparse_fa ms | dense_fa ms | **dense / sparse** |
|---:|---:|---:|---:|---:|
| 1  |  32 K | 0.031 | 0.044 |  **1.4×** |
| 1  | 128 K | 0.027 | 0.056 |  **2.1×** |
| 1  |   1 M | 0.029 | 0.193 |  **6.7×** |
| 16 |  32 K | 0.030 | 0.074 |  **2.5×** |
| 16 | 128 K | 0.027 | 0.233 |  **8.6×** |
| 16 |   1 M | 0.028 | 1.699 | **60.1×** |

Two effects to notice:

- **Dense-FA is memory-bandwidth-bound in MQA form.** At $B=16$, $S=1\text{M}$
  the kernel reads $16 \times 1\text{M} \times 128 \times 2 = 4$ GB of K plus
  4 GB of V; the measured 1.696 ms corresponds to ≈ 4.7 TB/s — within a factor
  of two of B200's 8 TB/s peak. At $B=1$ the launch overhead is not amortised
  and the numbers are much smaller.
- **Sparse-FA sits at kernel-launch-overhead-land.** For every $B$ and $S$ we
  measured, the K = 2048 attention finishes in about 27 µs (SDPA's minimum
  overhead). In practice the FA kernel could probably do it in < 10 µs if
  batched with something else, but as a standalone kernel call it is bounded
  from below by launch overhead, not by the actual K = 2048 work.

### 3.4 The measured plot

![indexer cost](bench_indexer_cost.png)

Two panels, each log–log in $S$:

- **Left (Q1)**: Full-indexer time (solid) grows *sub-linearly* at small $S$
  (where the ~0.3 ms setup floor dominates) and *linearly* at large $S$
  (where `fp8_mqa_logits` dominates); sparse-FA (dashed) is flat throughout.
  The gap is ≈ 15× at $S = 32\text{K}$ and ≈ 100× at $S = 1\text{M}$. Batch
  size barely moves either curve — indexer setup is launch-overhead-bound,
  the score kernel is K-read-bound, and sparse-FA is stuck at SDPA launch
  overhead.
- **Right (Q2)**: Dense-FA (solid) also grows with $S$ (nearly linearly at
  $B=16$; sub-linearly at $B=1$ where the kernel isn't amortised); sparse-FA
  (dashed) is again flat. The dense/sparse gap widens from ~1.5× at
  $S = 32\text{K}$ to ~60× at $S = 1\text{M}$ (for $B=16$).

### 3.5 What this means for real DSA deployments

Pulling the two questions together, the DSA end-to-end picture at decode is:

$$
\underbrace{T_{\text{DSA}}}_{\text{indexer + gather + sparse-FA}} \;\approx\; T_{\text{full indexer}}
\quad\text{(dominant)}\qquad\text{vs.}\qquad
T_{\text{dense-MLA}}.
$$

Four concrete takeaways:

1. **DSA is a large win for prefill and a modest win for decode — but only at
   long context.** Our decode-time numbers show `dense_fa / dsa_total`
   ratios of 0.06 to 0.50 (i.e. DSA is *slower* than dense MLA at all
   measured points), because MQA-shaped dense MLA is already remarkably
   cheap on B200 and the indexer eats most of the DSA budget. In prefill,
   dense scales $O(S^2)$ and the win is unambiguous.
2. **The indexer is the single thing to optimise.** ~80–98 % of DSA time is
   the indexer. Every architectural knob that helps — **IndexShare** (run the
   indexer 1/4 as often, GLM-5.2), **CSA compression** ($4\times$ fewer K
   tokens to score, DSv4), **HCA** (drop the indexer entirely, DSv4), and
   **fused Q kernel** (`fused_q_indexer_rope_hadamard_quant`, DSv4) — is aimed
   at exactly this hot spot.
3. **Sparse-FA is effectively free.** If you ever consider skipping the
   top-K step and doing something adaptive at inference, remember: the
   sparse-attention kernel isn't where you'll find speedups. The whole DSA
   design is a bet that a small, memory-friendly scorer + a tiny sparse
   attention over the top 2048 KV tokens is *cheaper* than dense attention
   over all $S$ — a bet that pays off cleanly for prefill and, at long enough
   $S$, for decode with a fatter main-attention shape than the MQA-simplified
   one we measured here.
4. **Below ≈ 64 K context, the "indexer" is really the *setup* of the
   indexer, not the score kernel.**  At $S = 32\text{K}$, ≈ 61 % of the full
   indexer time is projections + norm + RoPE + Hadamard + FP8-quant *plumbing*
   (see the "setup vs score" table above); `fp8_mqa_logits` itself is only
   19 %.  This is the empirical case for porting DSv4's fused
   `fused_q_indexer_rope_hadamard_quant` back to the DSA path — at short
   context it collapses 7 small kernels into one and could roughly halve
   full-indexer time.  At long context the win narrows sharply because the
   score kernel dominates absolutely.

Raw data is in `bench_indexer_cost.csv`; run instructions live in `README.md
§3`.

---

## 4. Compact difference matrix

|  | GQA (Qwen-3.5) | MLA (DSv3) | **DSv3.2 (DSA)** | **GLM-5** | **GLM-5.2** | **DSv4 CSA** | **DSv4 HCA** |
|---|---|---|---|---|---|---|---|
| KV per token/layer | $H_{kv}\cdot D$ | $d_c + d_r$ (~576) | latent + tiny Index-K (~132 B) | same as DSv3.2 | same, indexer runs every 4 layers | latent + Index-K (both compressed 4×) | latent (compressed 128×) |
| Score matrix | full, $O(S^2)$ | full, $O(S^2)$ on latent | top-K = 2048 | top-K = 2048 | top-K = 2048 (reused across 4 layers) | top-K = $K$ compressed entries | dense over $\lceil S/128\rceil$ entries |
| Indexer? | — | — | FP8 MQA, $h_I=64$, $D_I=128$ | FP8 MQA, $h_I=32$, $D_I=128$ | same, called 1/4 layers | FP4 MQA on compressed K | none |
| Softmax? | yes (over $S$) | yes (over $S$) | yes (over top-K) | yes (over top-K) | yes (over top-K) | yes (over top-K) | yes (over $\lceil S/128\rceil$) |
| Layer-level cost | $O(S^2)$ | $O(S^2)$ on latent | $O(S)\cdot h_I D_I + O(K)\cdot\text{MLA}$ | same | indexer /4 amortised | $O(S/m)\cdot h_I D_I + O(K)\cdot\text{MLA}$ | $O(S/m')\cdot\text{MLA}$ (dense) |
| Trained sparse? | no | no | yes | yes | yes (inherited) | yes | yes |

Note on GLM-5 indexer heads: the released `config.json` uses `index_n_heads =
32`, half of DSv3.2's 64. GLM-5 also halves the *main* attention head count
(128 → 64). The `glm` spec in `DSA.py` reflects this.

---

## References

### Primary sources

- DeepSeek V3 report (MLA): https://arxiv.org/abs/2412.19437
- DeepSeek V3.2-Exp report (DSA / lightning indexer): https://arxiv.org/abs/2509.19000
- DeepSeek V4 paper (CSA + HCA): https://arxiv.org/abs/2606.19348
- GLM-5 paper ("from Vibe Coding to Agentic Engineering"): https://arxiv.org/abs/2602.15763
- GLM-5.2 blog (IndexShare, ~2.9× at 1 M): https://z.ai/blog/glm-5.2
- IndexCache (empirical basis for IndexShare): https://github.com/THUDM/IndexCache
- Su Jianlin, *"From MHA, MQA, and GQA to MLA"* (the reference MLA derivation): https://spaces.ac.cn/archives/10091

### Blogs & explainers

- DeepSeek Sparse Attention — Sebastian Raschka: https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/
- A visual guide to attention variants — Sebastian Raschka: https://magazine.sebastianraschka.com/p/visual-attention-variants
- DeepSeek Sparse Attention on GPU cloud — Spheron: https://www.spheron.network/blog/deepseek-sparse-attention-long-context-llm-gpu-cloud/

### In this directory

- `DSA.py` — readable MLA + DSA reference (paper-form and weight-absorbed MLA, `DsaIndexer`, `Dsv4Indexer`, `C4Compressor`, end-to-end sparse MLA).
- `bench_indexer_cost.py` — the self-contained bench used in §3.
- `bench_sparse_attention_kernels.py`, `bench_sparse_attention_modules.py` — older, larger benches (dense FA baseline, block-sparse FA, module-level DSv3.2 / GLM-5 / DSv4 comparisons).
- `ATTENTION_MATH.md` — the sparse + linear-attention formulas behind the numbers.
- `README.md` — how to reproduce every number in this file.

---

## Appendix — orthogonal branches (not sparse-with-MLA)

Three other 2025–2026 designs answer the "less than $O(S^2)$" question via
paths that are *not* variations of the §2 skeleton. They are included here for
completeness so the reader knows where they sit.

- **Linear attention (Qwen3-Next/Qwen3.5, Kimi-Linear/Kimi K3).** Replace most
  attention layers with an $O(S)$ recurrence — Gated Delta Net (Qwen) or Kimi
  Delta Attention (Kimi). The softmax is gone, replaced by a finite-state RNN
  memory. Requires end-to-end training with the new layer; not a drop-in
  optimisation. The full math, implementation comparison, decode/prefill B200
  sweeps, and Kimi K3 support gaps now live in `../linear_attn/SURVEY.md`.
- **Block-sparse on plain GQA (MiniMax-M3).** Same top-K skeleton as DSA but
  (a) no MLA underneath, (b) block-granularity from the start (max score
  per block, not per token), (c) forced attention-sink and sliding-window
  blocks. Not currently in SGLang (`Glob "**/minimax*"` returns only
  `minimax_m2.py`, which is dense GQA + MoE).
- **Sliding-window + attention sinks (gpt-oss, Mistral, Gemma).** A fixed
  static pattern — no indexer, no learned top-K. Cheap and easy but loses the
  ability to attend to arbitrary far-away tokens.

The DSA family in the main body is what most of the compute-heavy MoE
generation converged on because it is the only one of the three that (a)
keeps the softmax, (b) works with MLA, and (c) can be adopted by *fine-tuning*
an already-trained dense model.

