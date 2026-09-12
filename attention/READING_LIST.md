# Attention reading list: papers → ideas → local kernels

Selected on 2026-09-11 for this repository's **kernel-only** tutorial. These
are primarily papers and posts from the authors or implementing projects.
The order below is a learning recommendation, not a performance ranking.
Read the algorithm and implementation sections first; model-training and
evaluation sections can wait. No experiments are needed for this reading pass.

## 1. Dense attention and FlashAttention

Start with these in order. Read the blog first where one is provided, then
use the paper to work through the equations and pseudocode.

1. **FlashAttention-2 introduction — the accessible starting point.**
   [Author's Stanford CRFM blog](https://crfm.stanford.edu/2023/07/17/flash2.html),
   followed later by the [FA2 paper](https://arxiv.org/abs/2307.08691).
   Focus on memory traffic, query-tile parallelism, and work partitioning among
   warps. Question: why can fewer shared-memory exchanges matter even when the
   matrix-multiplication FLOP count barely changes?

2. **Online normalizer calculation for softmax — derive the core recurrence.**
   [Paper](https://arxiv.org/abs/1805.02867).
   Track the running maximum and rescaled denominator. Then extend the same
   reasoning to attention's weighted-value numerator in
   [dense_attention.py](dense_attention.py).
   Question: when a later tile raises the maximum, why must both old partial
   sums be rescaled?

3. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**
   [Original paper](https://arxiv.org/abs/2205.14135).
   Focus on tiling, the memory hierarchy, and avoiding materialized attention
   matrices. Question: what becomes linear in memory, and what remains quadratic
   in arithmetic? Pair with [attention.md](attention.md).

4. **FlashAttention-3 — learn Hopper pipelining.**
   [Author's blog](https://tridao.me/blog/2024/flash3/),
   [paper](https://arxiv.org/abs/2407.08608).
   Focus first on TMA, WGMMA, producer/consumer warps, and overlapping softmax
   with matrix operations. Leave the FP8 discussion for a second pass.
   Question: which operations can overlap, and which dependency requires a wait?

5. **FlashAttention-4 — learn why the bottleneck moves on Blackwell.**
   [Author's blog](https://tridao.me/blog/2026/flash4/),
   [paper](https://arxiv.org/abs/2603.05451),
   [official implementation](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
   Focus on asymmetric hardware scaling, softmax/exponential work, asynchronous
   pipelines, and tensor memory. Question: why does faster matrix multiplication
   increase the importance of optimizing non-matmul work?

The FA papers describe different hardware targets and experimental settings.
Do not transplant a reported speedup into our benchmark expectations without
matching shapes, precision, hardware, and the measured boundary.

## 2. Sliding windows and programmable sparsity

6. **FlexAttention — the most directly useful implementation blog here.**
   [PyTorch team's post](https://pytorch.org/blog/flexattention/).
   Focus on `score_mod`, `mask_mod`, and `BlockMask`. Question: why does masking
   a score not automatically mean its matrix-multiplication work was skipped?
   Pair with [sliding_window_attention.md](sliding_window_attention.md) and
   the `torch_flex` adapter in [backends.py](backends.py).

7. **Mistral 7B — optional sliding-window motivation.**
   [Paper](https://arxiv.org/abs/2310.06825).
   Read the sliding-window and rolling-cache discussion; skip the model quality
   tables initially. Question: how do logical token position and physical cache
   slot differ? This is architectural motivation, not a detailed FA kernel manual.

## 3. Sparse attention and DeepSeek compression

8. **Native Sparse Attention — useful background, not a substitute for CSA2.**
   [Paper](https://arxiv.org/abs/2502.11089).
   Focus on the distinction between compression, selected blocks, and a local
   window. Question: does selecting a block mean retaining its individual KV
   vectors or replacing them with one pooled vector?
   Pair with [our sparse tutorial](sparse_attn/compressed_sparse_attention.md).

9. **DeepSeek V4 → V4.1 / CSA2 — the target-specific reading.**
   For V4, read the official
   [`Compressor` and `Indexer` source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)
   alongside our sparse tutorial. Then read the pinned
   [V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)
   and [V4.1 inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py),
   focusing on CSA2, cache ownership, Full/Reindex/Reuse, and hierarchical
   indexing. Question: which costs are reduced by compression, by top-k, and
   by sharing across layers? Use [deepseek/research.md](../deepseek/research.md)
   as the step-by-step companion.

Keep model architecture and installed kernel support separate. Our
[SGLang alignment audit](../deepseek/sglang_alignment.md) is tied to its inspected
revision; a newer report does not establish support in an older serving checkout.

## 4. KDA: understand the recurrence before the optimized implementation

10. **Kimi Linear: An Expressive, Efficient Attention Architecture.**
    [Paper](https://arxiv.org/abs/2510.26692).
    Focus on KDA's channel-wise gating, delta-rule update, and chunkwise
    formulation. Question: why must the delta correction predict from the
    decayed state? Pair with [kda_attention.py](linear_attn/kda/kda_attention.py)
    and the first sections of [KERNELS.md](linear_attn/kda/KERNELS.md).

11. **FlashKDA v1: A Deep Dive — my first choice for the KDA kernel walkthrough.**
    [Moonshot's implementation post](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md).
    Focus on chunk size 16, the triangular inverse, fusion boundaries, and
    precision tradeoffs. Question: how does chunk size simultaneously affect
    matrix work, numerical range, and temporary storage? Then trace the local
    Triton and CuTe prep/carry kernels using [KERNELS.md](linear_attn/kda/KERNELS.md).
    Treat this as the design of that FlashKDA version, not every local variant.

If the KDA chunk algebra feels abrupt, insert these two prerequisite papers:

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484):
  focus on expressing delta-rule dependencies through a chunkwise formulation.
  Question: which work becomes parallel within a chunk, and what state must
  still pass between chunks?
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464):
  focus on combining forgetting with the delta update. Question: what changes
  when a head-level decay is replaced by KDA's channel-wise decay?

## A manageable first pass

Do not try to finish the entire list before asking questions. My suggested
starting set is **FA2 blog → online softmax → FlexAttention → Kimi Linear's KDA
section → FlashKDA deep dive**. Then branch into FA3/FA4 or DeepSeek CSA2 based
on the kernel you want to inspect next.

Bring a paper equation, paragraph, or code symbol. Useful questions include:

- “Can we derive this equation with a four-token example?”
- “Which variables in our PyTorch reference correspond to these symbols?”
- “Which CTA/warp owns this tensor, and where does it live?”
- “Is this changing the attention rule, the numerical approximation, or only
  the execution schedule?”

Those questions map naturally to this repository; no layer benchmark is needed.
