# Compressed sparse attention, step by step

Snapshot: 2026-09-10. Start here for kernel math. For the newly released
**DeepSeek V4.1-Flash / CSA2**, continue to the existing
[DeepSeek research](../../deepseek/research.md) and
[SGLang compatibility audit](../../deepseek/sglang_alignment.md).
V4 CSA, V4 HCA, V4.1 CSA2, and arbitrary block sparsity are not interchangeable.

## 1. Separate three problems

```text
projected values + pooling logits ── compression ──► compressed cache
index queries + index keys ── scoring / top-k ──────► selected IDs
main queries + caches + selected IDs ── softmax ────► attention output
```

These are different kernel boundaries. Compression changes **what is stored**;
selection changes **which entries are visited**; fused attention changes **how
the chosen softmax is evaluated**. Quantization and physical paging add storage
contracts, not new definitions of top-k. The new benchmark times only the
last arrow, on identical caches and IDs for every backend.

The PyTorch operators in [compressed_sparse_attention.py](compressed_sparse_attention.py)
cover all three arrows for study. Projection GEMMs, normalization/RoPE, cache
insertion, trained weights, and scheduling are intentionally outside these
functions. They are not a complete DeepSeek layer or a quantized deployment.

## 2. Why compress before selecting?

Selecting $k$ historical tokens reduces expensive main attention work but does
not automatically shrink the stored history or the cost of finding those
tokens. Pooling groups of $r$ tokens creates roughly $T/r$ global entries.
The selector can search this smaller cache, while a local raw-token window
preserves recent detail. This trades representational resolution for storage
and retrieval cost; it is not generally identical to dense token attention.

A useful common vocabulary is **representation + selection + aggregation**:

| Method | Representation | Selection | Is it learned compressed sparse attention? |
|---|---|---|---|
| Dense attention | Raw K/V | All causal tokens | No |
| Sliding window | Raw K/V | Recent W tokens | No; a local-only limiting case |
| DSA-style token selection | Raw/indexed token entries | Learned top-k | No pooling; identity-compression analogy only |
| Block sparse / NSA selected branch | Blocks of token K/V | Selected blocks | Blocks do not automatically become pooled vectors |
| V4 CSA | Local raw + r=4 pooled global entries | Local window + top-k compressed entries | Yes |
| V4 HCA | Local raw + r=128 pooled global entries | Local window + all completed global entries | Compression without global top-k |
| V4.1 CSA2 | Shared/quantized caches; r=2 or identity paths | Full/Reindex/Reuse, hierarchical candidate reuse | A revised architecture, not V4 with a renamed kernel |

This framework organizes the directory without making the false claim that
every sparse method implements DeepSeek's compressor. NSA itself has distinct
compression, selected-block, and window branches. [NSA paper][nsa]

## 3. Derive channel-wise gated pooling

Let projected vectors $x_u\in\mathbb R^d$ and projected pooling logits
$g_u\in\mathbb R^d$ already be supplied. In a nonoverlapping complete group
$B_j=\{jr,\ldots,(j+1)r-1\}$, with optional learned within-group bias $a$:

$$
\pi_{u,c}=\frac{\exp(g_{u,c}+a_{u-jr,c})}
{\sum_{v\in B_j}\exp(g_{v,c}+a_{v-jr,c})},\qquad
\bar c_{j,c}=\sum_{u\in B_j}\pi_{u,c}x_{u,c}.
$$

The softmax axis is **positions within the group, separately for every
channel**. This is not a single scalar weight per token, not mean pooling,
and not attention from a main query. Once created, an entry can serve many
queries. With r=1 it reduces to the supplied projected vector.

`gated_compress` computes this pooling in FP32 and drops incomplete groups.
In a real incremental implementation, those incomplete values/logits must stay
in a small accumulator buffer until the group completes; they must not vanish.
Normalization, positional encoding, and quantization follow pooling in the
model-specific path, and are not silently folded into this reference.

### V4 CSA has an important overlap rule

For r=4 the official V4 compressor projects **2d** channels for values and
logits. Split them into A/B halves. Entry j pools the previous group's A half
and the current group's B half: eight positions, producing one d-vector.
At j=0 the absent previous half has zero values and $-\infty$ logits, so it
does not dilute the first entry. The code then applies RMSNorm and partial
RoPE using the current group's first position jr. HCA's r=128 path does not
use this overlap. [Official V4 `Compressor`][v4model]

That source also has separate main/index compression paths and quantization
choices. The new pooling function reproduces the projected-tensor pooling
rule, not those entire paths. In V4.1, the r=2 compressor is nonoverlapping
and removes V4's learned within-group bias; the index key comes from a
projection of the main compressed representation before its RoPE. Follow the
pinned [CSA2 reference walkthrough](../../deepseek/research.md), not an r=4
kernel with its ratio changed by hand. [Official V4.1 code][v41model]

## 4. Visibility is determined by completion time

An entry whose current group is j becomes visible no earlier than

$$
e_j=(j+1)r-1,\qquad e_j\le t.
$$

The number of completed entries at zero-based position t is
$\lfloor(t+1)/r\rfloor$. Overlap does not make the next group available early.
For r=4, position 2 sees no compressed entries; position 3 sees entry 0;
position 6 still sees only entry 0; position 7 sees entries 0 and 1.
This must remain true even if prefill has physically computed a full cache
containing entries from later positions.

`indexer_topk` masks incomplete entries **before top-k**.
`combined_selection` applies the completion rule again when building main
attention indices. An early query can have fewer than k legal entries; padded
slots carry -1, not a duplicated valid ID.

## 5. The indexer is not the main attention score

An unquantized indexer reference uses small index heads:

$$
I_{t,j}=\sum_{h=1}^{H_I}w_{t,h}\,
\operatorname{ReLU}\!\left((q^I_{t,h})^T k^I_j\right),\qquad
J_t=\operatorname{TopK}_{j:e_j\le t}(I_{t,j}).
$$

Weights include the model's scaling and can be signed. Do not add an extra
softmax to this selector or assume ReLU makes the weighted total nonnegative.
Main attention uses different queries/scores. The reference function takes
the small projected tensors directly. Quantized score kernels and different
top-k tie breaking may change selected IDs even if the final attention kernel
is numerically correct. [V4 indexer implementation][v4model]

The main attention selection is shared across its heads. Each head still
computes its own weights over those selected entries. HCA replaces this top-k
step with the list of all completed heavy-compression entries.

## 6. One softmax over local and compressed entries, plus a sink

Use a shared latent cache with K=V, as in the studied DeepSeek sparse core.
For each query let $A_t$ contain recent raw entries and selected compressed
entries. These are distinct representations even if their underlying raw
token coverage overlaps. For a finite learned sink logit $s_h$:

$$
z_{t,h,j}=q_{t,h}^T c_j/\sqrt d,\qquad
o_{t,h}=\frac{\sum_{j\in A_t}e^{z_{t,h,j}}c_j}
{e^{s_h}+\sum_{j\in A_t}e^{z_{t,h,j}}}.
$$

The sink has **zero value**. It reduces output magnitude by absorbing some
probability mass. It is not a historical KV token and not an extra residual
vector. Local and global outputs cannot be separately normalized and added:
they must share a denominator, or be merged with correct log-sum-exp weights.
Inverse RoPE and any following projection are outside this core boundary.
[FlashMLA sparse interface and sink semantics][flashmla]

[`gather_attention`](compressed_sparse_attention.py) gathers selected entries
then uses einsum/softmax. The independent dense oracle computes scores over
the full conceptual cache and masks down to the same support. Duplicate active
IDs are rejected by the adapter because duplication changes normalization;
invalid slots are not replaced by repeated token zero. Both references define
an entirely empty support without a sink to output zero.

## 7. Follow logical IDs into actual kernels

Our conceptual per-request cache is `[raw[0:T] | compressed[0:N]]`.
This is convenient for learning, not a claim about production storage.

| Backend row | Actual callable | Prepared layout / restriction |
|---|---|---|
| `torch_gather` | PyTorch gather + einsum + softmax | Gather materialization included; shared cache concatenation excluded |
| `torch_dense_mask` | PyTorch full scores + support mask | Independent oracle; intentionally computes unselected scores |
| `sglang_flashmla` | `sgl_kernel.flash_mla.flash_mla_sparse_fwd` | BF16 D=512; flattened batches, `[Q,1,Kpad]` IDs |
| `flashmla` | `flash_mla.flash_mla_sparse_fwd` | Upstream version of the same kernel family, not an independent algorithm |
| `flashinfer_dsv4` | `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` | Explicit TRTLLM-GEN BF16 adapter, SM100/103, Q=1, W=128, H=64/128 |

FlashMLA's prefill API accepts selected physical rows; it does not infer
causality from them. We compact valid IDs to a prefix, pad the selection to
64-slot alignment, pass active lengths, and offset each batch's rows into a
flattened cache. Calling this API with Q=1 still uses that API, not a claim
that it is the library's optimal specialized decode path. [Upstream API][flashmla]

For FlashInfer's V4 TRTLLM-GEN path, the first fixed 128 selection slots address
the SWA pool, the remaining slots address the separate compressed pool, and
indices are flattened to `[B*Q,L]`. Its active length includes the fixed SWA
capacity. The adapter builds contiguous 64-token pages, physical offsets,
original sequence lengths, and reusable workspace outside timing. It
deliberately requires fully valid selections and capacity divisible by four;
earlier-prefix/padded cases report `SKIP`. This is a source-reviewed adapter,
not a GPU-validated integration. [FlashInfer V4 API][fi]

## 8. What is currently implemented, and what is only surveyed?

The inspected SGLang snapshot is
`52c191da52390fa5508de98eddd1e3eca2dbcfb2`. Its V4 model/indexer/backend paths
are traced in [sglang_alignment.md](../../deepseek/sglang_alignment.md).
At that revision the V4 compression-ratio contract is 0/4/128. It does **not**
establish a complete CSA2 path for V4.1's cache owners, ratios, and selection
reuse. A compatible final sparse softmax kernel is only one component of model
support. Do not advertise full V4.1 serving compatibility from this benchmark.

Other efficient paths worth studying, but not dishonestly aliased to the same
BF16 benchmark row:

- FlashInfer's **SM120/121 sparse path** uses packed FP8 or NVFP4 records and
  separate segment metadata. A plain BF16 `[pages,1,64,512]` tensor is not that
  storage ABI. Its **CuTe DSL HCA** path consumes compressed page IDs rather
  than arbitrary compressed token selections. [Current FlashInfer API][fi]
- vLLM has its own [V4 FlashInfer integration][vllm]. This is a serving adapter
  over kernel implementations, not an independent mathematical attention rule.
- FlexAttention's block-mask path and FlashInfer's [block-sparse wrapper][fiblock]
  are useful for structured sparsity. A scattered top-k token list can fill
  many blocks, losing most block-skipping benefit; neither supplies DeepSeek's
  learned compression automatically.
- DeepSeek's [DeepGEMM][deepgemm] provides optimized math used in indexing and
  related sparse workloads. Read its operation signature before treating it
  as a replacement for the whole compression→selection→attention pipeline.

Upstream `main` APIs and documentation are moving references inspected on the
date above. Installed wheels can lag them. Optional import failures and ABI
errors must be visible, not handled by falling back to a reference and recording
that time under the optimized backend name.

## 9. Derive costs before comparing implementations

Ignoring projection/quantization costs, a causal decode step has approximately:

$$
\begin{aligned}
\text{stored global vectors}&\sim T/r,\\
\text{indexer scoring}&\sim H_I d_I\,T/r\quad\text{(plus top-k selection)},\\
\text{CSA main core}&\sim H d\,[W+\min(k,T/r)],\\
\text{HCA main core}&\sim H d\,[W+T/128].
\end{aligned}
$$

Thus fixed top-k can bound main-core work while the indexer still grows with
context. Over a full prefill, exhaustive index scoring can remain quadratic
in sequence length divided by r. Quantized entries reduce bytes, but introduce
scale loads/conversions and numerical changes. Irregular gathers reduce usable
bandwidth; grouping/reordering loads can help only if IDs and normalization
semantics are preserved. Padding to a kernel tile may do extra work beyond k.

CSA2 additionally shares caches and selections across layers and restricts
later decoder indexers to a candidate pool. These optimizations target costs
the final softmax microbenchmark cannot measure. The [V4.1 guide](../../deepseek/research.md)
derives the Full/Reindex/Reuse rules and their ownership implications.

## 10. What to benchmark later

The new [driver](../bench_compressed_sparse_attention.py) creates synthetic
projected data, gated-pooled entries, and causal IDs **before** timing. It uses
an independent dense-selected-support oracle for an output suffix, includes
sink behavior, and records the precise core API boundary. The fixtures are
not calibrated checkpoint activations and do not reproduce FP4 accuracy.

Later, test r-boundaries, missing first overlap groups, fewer-than-k entries,
batch offsets, different windows, repeated-ID rejection, and sink extremes.
For pipeline analysis, separately time compression, score generation, top-k,
index conversion, sparse attention, and any quantization. Do not add a full
layer row to the same attention-core ranking. No results were generated here.

[nsa]: https://arxiv.org/abs/2502.11089
[v4model]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
[v41model]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
[flashmla]: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py
[fi]: https://docs.flashinfer.ai/generated/flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4.html
[fiblock]: https://docs.flashinfer.ai/generated/flashinfer.sparse.BlockSparseAttentionWrapper.html
[vllm]: https://docs.vllm.ai/en/stable/api/vllm/models/deepseek_v4/nvidia/flashinfer_sparse/
[deepgemm]: https://github.com/deepseek-ai/DeepGEMM
