# Sparse Attention Across Modern LLMs

*A short human-readable tour: what is actually different about attention in Qwen3 / Qwen3-Next, DeepSeek V3 / V3.2 / V4, GLM-5 / 5.2, and MiniMax-M3, and why any of it matters.*

---

## Introduction

Every LLM you can download today uses one of a small number of attention families. From the outside the model card just says "transformer, 671 B, 128 K context" — but on the inside, the choice of how attention is done is now the single largest architectural difference between recent open-weight models. This document is a per-model walk-through.

The core problem is well known. Self-attention costs $O(N^2)$ in the sequence length. At $N = 1{,}000{,}000$ that number is unpayable in both compute and KV-cache memory. Every model below is a different answer to the question **"how do we spend less than $O(N^2)$?"**. The answers roughly cluster into three approaches:

1. **Reduce the KV cache** (per-token state) without changing the score matrix — DeepSeek V3's *Multi-head Latent Attention* (MLA).
2. **Only compute part of the score matrix** — everything the industry now calls "sparse attention". Sub-variants: static patterns (sliding window + sinks), content-based top-k (DSA, MoBA, MiniMax-M3), and eviction (H2O, SnapKV, RocketKV).
3. **Replace attention with a recurrence** — Qwen3-Next's Gated Delta Net, Mamba-style linear attention. This is orthogonal to the sparse-attention story.

This file zooms into six specific models and shows what changes between them, in order. For the formulas behind each family, see `ATTENTION_MATH.md`; for the benchmarks that quantify the trade-offs, see `README.md`.

---

## The main chain, at a glance

The recent open-source attention story reads as one chain plus a couple of independent branches:

```
Qwen 3.5   ─▶   DeepSeek V3   ─▶   DeepSeek V3.2   ─▶   DeepSeek V4
 dense GQA      MLA                MLA + DSA            MLA + DSA-C4
                (small KV cache)   (top-k per query)    (compressed K stream)
                                        │
                                        ▼
                                      GLM-5   ─▶   GLM-5.2
                                      = DSv3.2    = DSv3.2 + IndexShare
                                                    (1 indexer / 4 layers)
```

Independent branches that ship in the same generation:

| Model | Idea | Where it sits in the taxonomy |
|---|---|---|
| **Qwen3-Next** | replace most attention layers with a linear-attention recurrence (Gated Delta Net) | approach #3 |
| **MiniMax-M3** | block top-k sparse attention on plain GQA — no MLA underneath | approach #2, content-based top-k |
| **gpt-oss** | sliding window + attention sinks (fixed pattern, no indexer) | approach #2, static |
| **RocketKV** | evict most of the KV cache after prefill; small dense KV at decode | approach #2, eviction |

The single most important sentence to remember: **GLM-5 did not invent sparse attention. It adopted DeepSeek's DSA verbatim, and GLM-5.2 added one engineering optimisation on top — IndexShare — which alone gives roughly a 2.9× per-token FLOPs reduction at 1 M context.**

---

## What "different" actually means, model by model

### Qwen 3.5 — dense GQA baseline

This is the "before" point. Every attention layer is plain multi-head attention with grouped queries (16 Q heads sharing 2 KV heads is a common shape). No sparsity, no top-k, no compression. Its "sparse" refers to Mixture-of-Experts in the MLP, not attention.

Cost per attention layer at sequence length $S$: $O(H \cdot S^2 \cdot D)$. At $S = 128\text{K}$ this term dominates wall-clock and KV memory. Every model that follows is attacking exactly this term.

### DeepSeek V3 / V3.1 — MLA: small KV cache, dense score matrix

MLA (Multi-head Latent Attention) does **not** sparsify the QK matrix. It sparsifies the **KV cache** itself: instead of caching `num_kv_heads × head_dim` per token, it caches one **latent KV vector** of size ~512 per token. Each head's K and V are recovered on the fly by a per-head up-projection.

Q is also split into two parts: a **no-RoPE** part (typically 192 dims) and a **RoPE** part (typically 64 dims). Only the RoPE half carries positional information into the dot product. This split is what lets DeepSeek quantize the no-RoPE part very aggressively without touching the positional signal.

Result: a much smaller per-token KV cache, at essentially the same accuracy. The score matrix is still $O(S^2)$; MLA does not fix that. But the smaller cache is a **prerequisite** for the next step — you cannot afford to run a per-query indexer against a full-sized KV cache at 1 M tokens.

### DeepSeek V3.2 — MLA + DSA: the first content-based sparse attention that "just works"

V3.2 adds **DSA (Deep Sparse Attention)** on top of MLA. This is the design that started the whole "lightning indexer + top-k" line.

The new component is a tiny MQA scorer — DeepSeek calls it the **lightning indexer**. It takes the same Q activation the MLA path already computed, projects it to 64 small heads of dimension 128, quantizes both Q and K to FP8, and runs an MQA `fp8_paged_mqa_logits` kernel to score *every K token against every Q token*. Then a `top-k = 2048` picks the survivors.

The main attention (still MLA) then runs only against those 2048 KV tokens per query. Score shape drops from $[S, S]$ to $[S, 2048]$ and, crucially, becomes **constant in $S$**.

Why it works at all:

- The indexer is ~25–50× cheaper than the main attention it replaces, because it is MQA (one K head, not many), FP8 not bf16, and outputs only scores (no softmax·V matmul).
- The indexer reuses MLA's Q activation, so no additional projection cost from the residual stream.
- The training is joint — the model *learns* to make the top-k selection informative. This is why DSA works out-of-the-box, whereas earlier training-free top-k schemes (Quest, InfLLM) always trailed dense accuracy by a small but persistent gap.

Cost picture:

| Component | Grows with $S$? |
|---|---|
| Indexer projections | No (linear in $S$, constant per token) |
| Indexer MQA logits | Yes, $O(S^2)$ but with a tiny prefactor |
| Sparse MLA attention | **No** (constant in $S$: $O(K \cdot H_{mla} \cdot D)$) |

So the dominant $O(S^2)$ term is the indexer, and its prefactor is small enough that even at 128 K it's cheaper than dense MLA.

### DeepSeek V4 — DSA-C4: score against a compressed K stream

V4 keeps the DSA algorithm and adds one component: a **compressor**. Every 4 consecutive tokens are pooled through a learned projection into a single "compressor token", and the indexer scores against the compressed sequence instead of the raw one. Same top-k, but on a 4× shorter sequence.

Result: the indexer's dominant $O(S^2)$ term shrinks by 4×. Everything else — the top-k, the sparse MLA, the KV cache — is unchanged. At $S = 64\text{K}$ we measured the indexer's logits step drop from ~85 ms (V3.2) to ~22 ms (V4), a clean ~4× that exactly matches the compression ratio.

The compressor is trained jointly with the rest of the model, so it does not cost accuracy — its whole design is to preserve the information the top-k step cares about.

### GLM-5 — DSA verbatim

GLM-5 is the DeepSeek V3.2 architecture with different head counts and MoE hyperparameters. Same MLA, same DSA indexer, same top-k = 2048, same FP8 MQA scorer. The only meaningful attention-level difference from V3.2 is:

- 64 attention heads (vs V3.2's 128).
- RoPE is applied in interleaved (GPT-J) style, not NeoX.

Nothing else about attention has moved. This is worth stating clearly because papers and blog posts often present GLM-5 as though it introduces its own sparse attention — it does not. It re-uses the DeepSeek design and validates that DSA transfers cleanly to a different labs' full-scale training run.

### GLM-5.2 — IndexShare on top of DSA

GLM-5.2 is where a genuinely new engineering idea shows up. Empirically the top-k that the indexer picks for a token is **very stable across consecutive layers** — the same KV tokens are selected in layer $l$ and layer $l+1$ ~90 % of the time. So why run the indexer on every layer?

IndexShare is exactly that observation, turned into a knob. With period $N = 4$, only 1 layer in every 4 actually runs the indexer; the other 3 reuse the previous layer's top-k indices. The sparse-attention step still runs on every layer, unchanged. The indexer's amortized cost drops 4×.

That single change gives the headline **~2.9× per-token FLOPs reduction at 1 M context** in the GLM-5.2 blog, without changing the model's output distribution meaningfully. Combined with V4's compressor (which is orthogonal — compression per call vs fewer calls), the two together would give ~16× indexer-cost reduction.

### MiniMax-M3 — sparse top-k without MLA

MiniMax-M3 is the odd one out in this list, and it's the interesting one. It doesn't use MLA; the attention is plain GQA. But it does use content-based top-k sparse attention.

The differences from DSA:

- The indexer is smaller: one K head (MQA), no FP8 quantization, just bf16.
- The score is a **per-block max** rather than a per-token score. Each query scores every KV block by taking the maximum inner product across positions in that block. The top-k selects **blocks**, not tokens.
- The first `init_blocks` KV blocks (the sequence start) and the last `local_blocks` (the sliding window) are **forced** into the keep set. This is a learned analogue of StreamingLLM's attention-sink + sliding-window pattern.

The rest is familiar: dense GQA over the kept blocks only. The overall skeleton — Q projection → cheap score → top-k → sparse attention — is the same as DSA, but adapted to (a) GQA instead of MLA, and (b) block granularity from the start instead of token granularity.

Why it matters: MiniMax-M3 shows that content-based top-k works even without MLA. That's a useful data point because MLA is expensive to bring up (it changes weight layout, quantization strategy, and training) and many teams don't want to adopt it just to unlock sparse attention.

### Qwen3-Next — a completely different axis

Qwen3-Next replaces *most* attention layers with a **Gated Delta Net** recurrence — a linear-attention variant with $O(S)$ instead of $O(S^2)$ cost. A minority of layers stay full-attention to preserve long-range mixing. This is not sparse attention at all — it is approach #3 from the introduction.

The two axes (sparse attention and linear attention) are **orthogonal**: sparse attention keeps the softmax and only runs it on a subset of keys; linear attention gets rid of the softmax and rewrites attention as a recurrence. Neither subsumes the other, and it is entirely plausible that a future model combines them (linear-attention layers for cheap positional context, sparse-attention layers for retrieval).

For the "what's different?" reader: if you are switching from Qwen 3.5 to Qwen3-Next, essentially every attention layer has been replaced by something that looks more like a Mamba block than like an attention block. That is a much bigger change than any of the DeepSeek-line moves.

---

## Compact difference matrix

|  | Qwen 3.5 | DSv3 / 3.1 | DSv3.2 | DSv4 | GLM-5 | GLM-5.2 | MiniMax-M3 | Qwen3-Next |
|---|---|---|---|---|---|---|---|---|
| Attention type | dense GQA | MLA | MLA + DSA | MLA + DSA-C4 | MLA + DSA | MLA + DSA + IndexShare | GQA + block top-k | Gated Delta Net |
| Softmax? | yes | yes | yes (over top-k) | yes (over top-k) | yes | yes | yes | no (linear) |
| KV cache per token | $H_{kv} D$ | latent (~512) | latent + tiny index-K | latent + compressor-K | latent + tiny index-K | latent + index-K every 4 layers | full GQA + tiny index-K | recurrence state only |
| Indexer? | — | — | FP8 MQA, 64 heads, D=128 | FP8/FP4 MQA on 4× compressed K | FP8 MQA, 64 heads, D=128 | FP8 MQA, 1 call / 4 layers | bf16 MQA, block-max score | — |
| Top-k / kept fraction | 100 % | 100 % | 2048 tokens | 2048 tokens (of $S/4$) | 2048 tokens | 2048 tokens | block-level top-k | — |
| Cost scaling in $S$ | $O(S^2)$ | $O(S^2)$ (on latent) | $O(S)\cdot\text{indexer}+O(1)\cdot\text{MLA}$ | same but indexer / 4 | same as DSv3.2 | indexer / 4 amortized | $O(S)\cdot\text{indexer}+O(K)\cdot\text{GQA}$ | $O(S)$ |
| Trained sparse? | no | no | yes | yes | yes (inherited) | yes | yes | n/a (recurrent) |

---

## Benchmarks in this directory

Two runnable microbenchmarks accompany this document. Each is self-contained and does not require model weights.

### `bench_kernels_sparse_attention.py` — kernel level

Isolates the three primitives that make DSA / IndexShare actually run: dense FA, block-sparse FA, and the lightning indexer. Also produces a headline "(indexer + sparse-FA) vs dense-FA" table, and a **quality table** measuring the cosine similarity between sparse-attention output and dense-attention output at matched sparsity for two selection policies (random and oracle block-max).

Headline numbers on B200 (see the script's docstring for the full findings):

- Below $S \approx 8\text{K}$: DSA (with IndexShare period 4) is *slower* than dense FA4. The indexer overhead is not amortized.
- $S \approx 16\text{K}$: near breakeven.
- $S = 1\text{M}$ (extrapolated from GLM-5.2's own numbers): ~2.9× FLOPs reduction.

Quality table (16 K sequence, heavy-hitter K/V input, matched sparsity):

| sparsity | random top-k cos-sim | oracle top-k cos-sim | speedup |
|---:|---:|---:|---:|
| 0.02 | 0.10 | 0.53 | 11.0× |
| 0.05 | 0.19 | 0.69 |  6.0× |
| 0.10 | 0.29 | 0.79 |  3.4× |
| 0.25 | 0.48 | 0.91 |  1.5× |

The **oracle** column is the ceiling any indexer can reach; the **random** column is the floor with no indexer at all. Real DSA-style indexers sit between the two, empirically closer to the ceiling. The trade-off is exactly the one you expect: below 5 % sparsity you get 6–11× kernel speedup but < 0.70 cos-sim (only acceptable if the model was **trained** with sparse attention from scratch); at 10–15 % sparsity you get ~3× speedup and ~0.80 cos-sim (the training-free sweet spot); above 25 % sparsity you recover > 0.90 cos-sim but the indexer overhead almost erases the speedup.

### `bench_modules_sparse_attention.py` — module level

Wraps the same hot path in a real `nn.Module` mirroring the production DSv3.2, GLM-5, and DSv4 indexers, so the timing includes projections, LayerNorm, RoPE, and FP8 quant — everything a deployed model actually pays for.

The clean takeaways from the module table at $S = 16\text{K}$ with IndexShare period 4:

- **DSv3.2 and GLM-5 are essentially identical.** Small differences in projection dimension shift `total_ms` by ~10 %.
- **DSv4's compressor cuts total indexer time roughly in half at $S = 16\text{K}$** and closer to 4× at $S = 64\text{K}$ — matching the `compress_ratio=4` design.
- **IndexShare (period 4) drops the indexer's amortized cost 4×** in every row, without touching the sparse-attention step. Combined with the compressor it would give the full ~16× indexer-cost reduction.

---

## References

- DeepSeek V3 paper (MLA): https://arxiv.org/abs/2412.19437
- DeepSeek V3.2 (DSA / lightning indexer): https://arxiv.org/abs/2509.19000
- GLM-5.2 blog (IndexShare headline): https://z.ai/blog/glm-5.2
- IndexCache (empirical basis for IndexShare): https://github.com/THUDM/IndexCache
- Qwen3-Next (Gated Delta Net): https://qwenlm.github.io/blog/qwen3-next/
- MiniMax-M3 technical report: https://github.com/MiniMax-AI
- NSA (natively-trained sparse attention, DeepSeek/Tsinghua): "Native Sparse Attention", 2025
- MoBA (mixture of block attention, Kimi): 2025
- Benchmarks + how to run them: `README.md`