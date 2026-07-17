# Sub-quadratic Attention, in Formulas

High-level math for the two escape hatches from $O(S^2)$ attention:
**sparse** (keep softmax, trim the keys) and **linear** (drop softmax, keep a
recurrent state). Companion to `SURVEY.md` (per-model tour) and `README.md`
(benchmarks). Figures are marked `TODO` for screenshots.

## Notation

$S$ sequence length · $d$ head dim · $H$ heads · $q_t,k_s,v_s\in\mathbb{R}^{d}$.
Dense attention for query $t$:

$$
o_t=\sum_{s} a_{t,s}\,v_s,\qquad
a_{t,s}=\frac{\exp(q_t^\top k_s/\sqrt d)}{\sum_{s'}\exp(q_t^\top k_{s'}/\sqrt d)} .
$$

Cost is $O(S^2 d)$ compute and $O(S)$ KV per token. Everything below attacks the
$S^2$.

| Family | What it keeps | What it drops | Cost | Retrofit? |
|---|---|---|---|---|
| Dense | softmax, all keys | — | $O(S^2 d)$ | — |
| **Sparse** | softmax | most keys (keep top-$k$) | $O(S\,k\,d)+\text{index}$ | drop-in-ish |
| **Linear** | — | softmax entirely | $O(S\,d^2)$ | needs training |

> **TODO figure:** the "three families" taxonomy tree (dense → sparse → linear). Source: `SURVEY.md` §"main chain", or redraw.

---

# Part I — Sparse attention (keep softmax, pick top-$k$ keys)

Idea: softmax mass concentrates on a few keys, so compute it over a per-query
subset $\mathcal{S}_t\subset\{1..S\}$, $|\mathcal{S}_t|=k\ll S$:

$$
o_t=\sum_{s\in\mathcal{S}_t} a_{t,s}\,v_s .
$$

The whole game is choosing $\mathcal{S}_t$ **cheaply** yet close to the true
top-$k$. Cost splits into a small *indexer* plus a cheap *sparse attention*:

$$
\underbrace{O\!\big(S^2 H_I d_I\big)}_{\text{indexer (small }H_I,\ \text{FP8})}
\;+\;
\underbrace{O\!\big(S\,k\,d\big)}_{\text{sparse attn}} .
$$

> **TODO figure:** dense score matrix vs. block-sparse selected blocks. Source: NSA paper Fig. 2 (Native Sparse Attention), or MoBA Fig. 1.

## DSA — DeepSeek Sparse Attention (V3.2)

A **lightning indexer** scores every (query $t$, past key $s$) pair with a
tiny, low-precision head, then takes top-$k$:

$$
I_{t,s}=\sum_{j=1}^{H_I} w_{t,j}\,\operatorname{ReLU}\!\big(q^{I}_{t,j}\!\cdot k^{I}_{s}\big),
\qquad
\mathcal{S}_t=\operatorname{top\text-}k_s\, I_{t,s}.
$$

- Indexer heads $q^I\in\mathbb{R}^{H_I\times d_I}$, **one** key head $k^I\in\mathbb{R}^{d_I}$ (MQA-style), per-head gate $w_t\in\mathbb{R}^{H_I}$.
- FP8 with per-block scales makes the $O(S^2)$ term affordable; in code the $q$-scale is folded into $w$ and the dot-products run through `deep_gemm.fp8_mqa_logits` (→ `DSA.py::IndexerSimulator.logits_step`).
- Main attention (MLA) then runs only over $\mathcal{S}_t$ — the block-sparse FA kernel (PART B/D).

> **TODO figure:** DSA lightning-indexer → top-$k$ → sparse MLA pipeline. Source: DeepSeek-V3.2-Exp report, Fig. 1/2.

**Quality vs. cost.** At matched sparsity $k/S$, cos-sim(sparse, dense) rises
with $k$; a real indexer sits between a random floor and the oracle
$I_{t,s}=\max\text{-block}(q_t^\top k_s)$ (PART E). The indexer's own
$O(S^2)$ term is what the next two tricks amortize.

## IndexShare (GLM-5.2)

Reuse one index set $\mathcal{S}_t$ across a period of $p$ layers instead of
recomputing per layer. Amortized indexer cost drops $\times\tfrac1p$:

$$
\text{cost}_{\text{idx}}^{\text{amort}}=\frac1p\,\text{cost}_{\text{idx}} .
$$

Sparse attention is unchanged; only the scorer is shared (`--index_topk_freq p`).

## DSA-C4 (DeepSeek V4)

Compress the key stream by ratio $c$ before scoring: pool every $c$ tokens
into one "compressor token", so the indexer scores against $S/c$ keys:

$$
O(S^2 H_I d_I)\ \longrightarrow\ O\!\big(\tfrac{S^2}{c} H_I d_I\big).
$$

Top-$k$ is taken in compressor space, then expanded back to token positions
(`DSA.py::Dsv4Indexer`, `C4Compressor`). Composes with IndexShare
(≈ $c\cdot p$ indexer-cost reduction).

---

# Part II — Linear attention (drop softmax, keep a state)

Replace the softmax kernel with a feature map $\phi(\cdot)$ so the sum
factorizes and becomes a **recurrence** over a fixed-size state
$H_t\in\mathbb{R}^{d\times d}$:

$$
o_t=\frac{\phi(q_t)^\top\!\sum_{s\le t}\phi(k_s)v_s^\top}
        {\phi(q_t)^\top\!\sum_{s\le t}\phi(k_s)},
\qquad
H_t=H_{t-1}+\phi(k_t)v_t^\top .
$$

State size $d\times d$ is independent of $S$ ⇒ **$O(S\,d^2)$**, $O(1)$ KV per
token. The price: no softmax, so the model must be **trained** this way.

> **TODO figure:** softmax attention (grows with $S$) vs. linear recurrent state (fixed $d\times d$). Source: Linear Transformers / RWKV / Mamba-2 explainer.

## Delta rule (DeltaNet)

Plain additive state never forgets. The delta rule does an online
least-squares correction — write $v_t$ but subtract what the state already
predicts for $k_t$:

$$
H_t=H_{t-1}\big(I-\beta_t k_t k_t^\top\big)+\beta_t\,v_t k_t^\top ,
\qquad \beta_t\in[0,1].
$$

## Gated Delta Net (Qwen3-Next)

Add a data-dependent **decay** gate $\alpha_t\in(0,1)$ so old memory fades:

$$
\boxed{\,H_t=\alpha_t\,H_{t-1}\big(I-\beta_t k_t k_t^\top\big)+\beta_t\,v_t k_t^\top\,},
\qquad
o_t=H_t^\top q_t .
$$

$\alpha_t$ (forget) and $\beta_t$ (write strength) come from learned per-token
projections; $q,k$ are L2-normalized. This is the recurrence sglang's
`chunk_gated_delta_rule` implements (PART F).

> **TODO figure:** Gated DeltaNet update (decay × delta-correction). Source: "Gated Delta Networks" (Yang et al.), Fig. 1; or Qwen3-Next blog.

## Why "chunked": making the recurrence GPU-friendly

The naive recurrence is sequential. Split into chunks of size $C$: compute
**intra-chunk** contributions as dense matmuls (parallel) and carry only the
**inter-chunk** state $H$ across chunk boundaries:

$$
\text{cost}=\underbrace{O(S\,C\,d)}_{\text{intra, parallel}}+\underbrace{O\!\big(\tfrac{S}{C} d^2\big)}_{\text{inter, sequential}} .
$$

Still linear in $S$, but now bandwidth-bound matmuls instead of a token loop.

> **TODO figure:** chunk-parallel form (intra-chunk parallel + inter-chunk state pass). Source: GLA / DeltaNet-chunk paper, or FlashLinearAttention README.

---

# Part III — Sparse vs. linear, in one look

| | Sparse (DSA) | Linear (GDN) |
|---|---|---|
| Softmax | kept | dropped |
| Cost | $O(Sk d)+O(\tfrac{S^2}{cp}H_Id_I)$ | $O(S d^2)$ |
| KV / token | $O(S)$ (full cache) | $O(1)$ (state) |
| Scaling | sub-quadratic | **linear** |
| Adoption | near drop-in (GLM took DSA from DeepSeek) | needs from-scratch training |

- **Short $S$:** dense wins both — the indexer / state overhead isn't amortized (PART D, PART F break even near $S\!\approx\!8\text{–}16\text{K}$).
- **Long $S$:** linear scales best ($O(S)$); sparse keeps softmax fidelity but pays an $O(S^2)$ indexer that IndexShare + compressor beat down.
- **Mix in practice:** Qwen3-Next interleaves ~3 linear layers per 1 full-attention layer — linear for cheap context, full attention to preserve exact recall.

> **TODO figure:** measured $O(S)$ vs $O(S^2)$ curves (linear/sparse/dense ms vs seq_len). Source: `README.md` §5.7, or plot the bench output.

## Where each formula is measured

| Formula | Code | README |
|---|---|---|
| Dense $o_t$ | `bench_..._kernels.py` PART A | §5.1 |
| Sparse $o_t$ over $\mathcal{S}_t$ | PART B | §5.2 |
| Indexer $I_{t,s}$ + top-$k$ | PART C, `IndexerSimulator` | §5.3 |
| Indexer + sparse vs dense | PART D | §5.4 |
| cos-sim(sparse, dense) | PART E | §5.5 |
| Gated Delta Net $H_t$ | PART F | §5.6 |
| IndexShare $/p$, DSA-C4 $/c$ | module bench | §5.8 |

---

# Further reading (to-read)

Curated; details to be folded into Part II later.

**Linear attention**
- 杨松琳 (Songlin Yang, MIT) — talk *可扩展线性RNN的进展：DeltaNet及其变体* (progress in scalable linear RNNs: DeltaNet and variants). Bilibili [FAI]. Slides + video: <https://drive.google.com/file/d/1jF5BYeZuOM_Q2b04Hl1GP6MjK1Hgwa0k/view>
- *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang et al. (the chunk-parallel DeltaNet form in Part II). arXiv:2406.06484.
- *Gated Delta Networks: Improving Mamba2 with the Delta Rule* — Yang et al. (the GDN recurrence Qwen3-Next uses; PART F). arXiv:2412.06464.
- 苏剑林 (Su Jianlin), Jun 20 2025 — *线性注意力简史：从模仿、创新到反哺* (a short history of linear attention). 科学空间 / Scientific Spaces, kexue.fm.

**Sparse attention**
- DeepSeek-V3.2-Exp report (DSA lightning indexer): <https://arxiv.org/abs/2509.19000>
- GLM-5.2 blog (IndexShare, ~2.9× at 1 M): <https://z.ai/blog/glm-5.2>
- NSA / MoBA (natively-trained sparse attention) — see `SURVEY.md` §"Further reading".
