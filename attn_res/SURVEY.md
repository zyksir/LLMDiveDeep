# Kimi-K3 Attention Residuals: From a Snapshot Bank to One Blackwell Kernel

Attention Residual is easy to misread as ordinary token attention. It is not.
There is no query token scanning a KV cache, and no interaction between tokens.
Each token independently chooses how much information to take from a small bank
of residual-stream snapshots collected across model depth.

This article follows the implementation that currently serves Kimi-K3 in
SGLang. It answers six practical questions:

1. Where do the “keys” come from?
2. Given those keys, what is the exact attention-residual computation?
3. Where does it run in a decoder layer?
4. How do SGLang's PyTorch, Triton, and SM100 paths implement it?
5. Why is the kernel difficult despite the simple math?
6. What does one decoder layer cost, and what remains worth fusing?

The high-level history of the idea is not repeated here. The interesting part
for inference is the concrete dataflow and the code that makes it cheap.

## 1. Start with the actual state

For a forward pass with \(T\) tokens and hidden size \(H\), SGLang allocates an
attention-residual bank

\[
B \in \mathbb{R}^{T \times N_B \times H}.
\]

Kimi-K3 uses \(H=7168\) and at most eight valid bank rows. The bank is created
once per model forward in `AttnResidual`:

```python
self.block_residual = hidden_states.new_empty(
    (num_tokens, block_num, hidden_size)
)
self.num_valid_blocks = 0
```

Source:
`sglang-opensource/python/sglang/srt/layers/attn_residual.py`.

The important layout choice is token-major:

```text
bank[token, snapshot, hidden]
```

Attention Residual never mixes different tokens. Token \(t\) only reads
`bank[t, :, :]`.

### 1.1 How a bank row is produced

Every `attn_res_block_size` decoder layers, the attention-side input prefix is
copied into the next bank row:

```python
self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
```

and:

```python
bank[:, self.num_valid_blocks, :].copy_(prefix_sum)
self.num_valid_blocks += 1
```

The stored feature is therefore a frozen snapshot of the running residual
prefix at a depth boundary. It is not an attention K tensor produced by a
learned \(H\times H\) projection.

On the SM100 fast path, even this copy disappears as a separate launch. The
prefix row is already in registers while being scored, so the kernel stores it
directly into `bank[:, nvb, :]`.

### 1.2 “Key” and “value” are two views of the same snapshot

For one token, let the valid frozen snapshots be

\[
b_1,b_2,\ldots,b_n \in \mathbb{R}^{H},
\]

and let \(c\) be the current, not-yet-frozen prefix. The candidate rows are

\[
x_1=b_1,\ldots,x_n=b_n,\quad x_{n+1}=c.
\]

Each raw row \(x_i\) is the value. Its RMS-normalized form is used to compute a
scalar key score:

\[
k_i = \operatorname{RMSNorm}_{\gamma_s}(x_i), \qquad
s_i = w^\top k_i.
\]

Here \(w\in\mathbb{R}^{H}\) is a learned, static projection vector belonging to
the consumer boundary. It is not computed from the current token.

SGLang precomputes

\[
c_w = \gamma_s \odot w,
\]

so the runtime score is directly

\[
s_i =
\frac{x_i^\top c_w}
{\sqrt{\frac{1}{H}\sum_{h=1}^{H}x_{i,h}^2+\epsilon}}.
\]

This removes a materialized normalized-key tensor and folds the score RMSNorm
weight into the projection weight.

## 2. From keys to the attention residual

The scores compete through a softmax across depth sources:

\[
\alpha_i =
\frac{\exp(s_i)}
{\sum_{j=1}^{n+1}\exp(s_j)}.
\]

The raw, unnormalized rows are then mixed:

\[
m = \sum_{i=1}^{n+1}\alpha_i x_i.
\]

Finally, the consumer applies its own RMSNorm:

\[
y = \operatorname{RMSNorm}_{\gamma_o}(m).
\]

This distinction matters:

\[
y \ne \operatorname{RMSNorm}
\left(c+\sum_{i=1}^{n}\alpha_i b_i\right).
\]

The current prefix is itself a softmax candidate:

\[
y =
\operatorname{RMSNorm}
\left(
\sum_{i=1}^{n}\alpha_i b_i + \alpha_{n+1}c
\right).
\]

Its coefficient is learned and normalized together with the block features; it
is not fixed to one.

### 2.1 A minimal Python implementation

The complete operation is small enough to write directly:

```python
def attention_residual_python(prefix, bank, cw, ow, nvb, eps=1e-6):
    rows = torch.cat((bank[:, :nvb], prefix[:, None]), dim=1)
    rows_f = rows.float()

    rrms = torch.rsqrt(rows_f.square().mean(dim=-1) + eps)
    scores = torch.einsum("trh,h->tr", rows_f, cw.float()) * rrms
    probabilities = torch.softmax(scores, dim=-1)

    mixed = torch.einsum("tr,trh->th", probabilities, rows_f)
    return F.rms_norm(
        mixed.to(prefix.dtype),
        (prefix.shape[-1],),
        weight=ow,
        eps=eps,
    )
```

This implementation now lives in `attn_res/bench_attn_res.py` and is the
correctness reference for every optimized backend.

## 3. Where it runs in one decoder layer

Kimi-K3 invokes the primitive twice per decoder layer with different learned
score parameters.

```text
previous decoder output
        │
        ▼
attention-side aggregation
  score_proj = self_attention_res_proj
  score_norm = self_attention_res_norm
  out_norm   = input_layernorm
  optionally snapshot the current prefix
        │
        ▼
KDA or MLA attention
        │
        ▼
o_proj and TP/SP communication
        │
        ▼
MLP-side aggregation
  score_proj = mlp_res_proj
  score_norm = mlp_res_norm
  out_norm   = post_attention_layernorm
        │
        ▼
dense or MoE MLP
  folds the raw prefix into its output
```

There is one additional attention-residual aggregation after the last decoder
layer, using `output_attn_res_proj`, before the model output norm.

### 3.1 Attention-side prefix

At layer entry, SGLang may carry a delayed residual prefix \(p\) and a new
delta \(h\). It first forms

\[
c_{\text{attn}} =
\begin{cases}
h, & p=\varnothing,\\
p+h, & \text{otherwise}.
\end{cases}
\]

The bank plus \(c_{\text{attn}}\) is aggregated and normalized for attention.
At a block-write layer, \(c_{\text{attn}}\) is also frozen into the bank, then
the delayed prefix is reset because the bank now owns that checkpoint.

### 3.2 MLP-side prefix

Let \(a\) be the reduced attention output. The MLP-side current candidate is

\[
c_{\text{mlp}} =
\begin{cases}
a, & p=\varnothing,\\
p+a, & \text{otherwise}.
\end{cases}
\]

It is mixed with the same frozen bank but scored by a separate MLP-side vector.
The normalized mixture is the MLP input. The raw \(c_{\text{mlp}}\) is also
passed to the MLP tail, where it becomes the ordinary residual contribution.

The implementation is in
`sglang-opensource/python/sglang/srt/models/kimi_k3.py`,
`KimiK3DecoderLayer._forward_attn_residual`.

## 4. How SGLang implements the primitive

SGLang has three levels of implementation. They share the same mathematical
contract.

### 4.1 PyTorch reference

`aggregate_stream_torch` materializes

```text
rows: [T, nvb + 1, H]
```

then executes RMSNorm, scalar projection, softmax, weighted sum, and output
conversion through regular PyTorch operators.

This is useful for understanding and testing. It is not a serving kernel:
every expression can become an allocation or launch, and the source rows may
be reread several times.

### 4.2 Triton fallback

The portable SGLang fallback uses three launches:

1. `_score_kernel`: one CTA per `(token, source row)`, reducing over \(H\) to
   produce one FP32 score.
2. `_combine_kernel`: one CTA per `(token, 1024-hidden chunk)`, recomputing the
   tiny softmax locally and combining raw source rows.
3. the standard optimized RMSNorm kernel.

For \(H=7168\), the combine stage launches seven hidden-dimension CTAs per
token. Recomputing a softmax over at most nine rows is almost free and avoids
communicating probabilities between those CTAs.

The drawback is HBM traffic. Every source row is read once for scoring and
again for combining. The intermediate score tensor is small, but the
\([T,R,H]\) row traffic is not.

### 4.3 SM100 TMA fast path

For BF16, \(H=7168\), one to eight snapshots, and SM100+, SGLang dispatches
`attn_res_fused_tma`.

The kernel uses:

* one persistent CTA per active token slot;
* a producer warp group issuing bulk asynchronous row copies;
* two shared-memory chunk slots;
* eight consumer warps;
* FP32 row-square and row-dot reductions;
* online softmax across row chunks;
* an FP32 weighted-value accumulator;
* fused output RMSNorm;
* optional fused snapshot write.

The learned vectors `cw` and `ow` are staged in TMEM. Source rows are loaded
once from HBM, scored, and immediately folded into the online-softmax
accumulator while their fragments are still available.

The tuning table specializes `(chunk_rows, occupancy, consumer_regs)` for each
valid-bank count. For example, four snapshots plus the current prefix fit a
five-row chunk exactly.

The Python wrapper is:

`sglang-opensource/python/sglang/kernels/ops/kimi_k3/attn_res.py`.

The CUDA implementation is:

`sglang-opensource/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`.

### 4.4 Distributed fusion is part of the implementation

Attention Residual sits directly on communication boundaries, so optimizing it
in isolation is insufficient.

SGLang currently supports:

* local aggregation followed by direct multicast all-gather;
* NVLS pull reduce-scatter plus local residual add plus aggregation;
* optional bank write fused into the all-gather variant;
* normal TP all-reduce with the pending prefix add folded into the collective.

The most complete path performs

```text
o_proj TP partial
    → reduce-scatter
    → add local residual
    → score bank and current prefix
    → online softmax and weighted sum
    → output RMSNorm
```

without materializing the intermediate full-token tensor.

## 5. Why this is not a FlashAttention-4 call

The operation can be forced into attention notation:

```text
batch        = T
query length = 1
key length   = nvb + 1 ≤ 9
head dim     = 7168
```

That is the opposite of the shape FA4 is designed for. FA4's Blackwell
interface supports standard head dimensions up to 128 plus a few specialized
192/256/512 cases, not 7168.

Even a generalized FA kernel would be a poor fit:

* \(M=1\) and \(N\le9\) waste normal QK/PV tensor-core tiles;
* the key requires an RMS reduction across 7168 elements;
* the value must remain the raw, unnormalized row;
* output RMSNorm should be fused;
* block snapshot writes and distributed collectives should be fused;
* materializing Q and normalized K would add more traffic than the operation
  itself.

The custom kernel is therefore not duplicated FA4 functionality. It is a
large-width vector-reduction and online-mixture kernel.

## 6. The real implementation challenges

### 6.1 Tiny source count, huge feature dimension

There are at most nine rows, but every row is 7168 BF16 values. Parallelizing
over rows gives too little work; parallelizing over \(H\) creates reductions
for every score and for the output RMSNorm.

### 6.2 One-pass reuse

Each raw row is needed twice:

1. to compute its RMS-normalized scalar score;
2. as the value multiplied by its softmax weight.

The weight is unknown until the row reduction completes. A naïve kernel
rereads the row. The SM100 kernel keeps row fragments close enough to the
online-softmax fold that HBM sees one row read.

### 6.3 Numerically stable online softmax

Rows are processed in chunks. The kernel tracks running maximum \(m\), running
normalizer \(\ell\), and running value accumulator \(o\). For a new chunk with
maximum \(m_c\),

\[
m'=\max(m,m_c),
\]

\[
\ell' = e^{m-m'}\ell+\sum_i e^{s_i-m'},
\]

\[
o' = e^{m-m'}o+\sum_i e^{s_i-m'}x_i.
\]

The final mixture is \(o'/\ell'\). This is exactly the joint softmax, without
storing all probabilities.

### 6.4 Decode and prefill want different things

At \(T=1\), the entire vector payload is only tens of kilobytes. Latency is
dominated by launch and synchronization, so eliminating launches matters more
than theoretical bandwidth.

At large \(T\), the operation is bandwidth-bound. Reading each bank row twice,
as the Triton fallback does, becomes the dominant penalty.

### 6.5 The bank is large during prefill

Eight BF16 snapshots at \(H=7168\) consume

\[
8\times7168\times2 = 114{,}688
\]

bytes per token. At 16K tokens, that is about 1.88 GB before allocator and
alignment overhead. Sequence-parallel execution therefore keeps only each
rank's token slice and fuses aggregation with the required gather/scatter.

## 7. Cost of one current decoder layer

Consider a midpoint layer with four frozen snapshots. The aggregation sees

\[
R = nvb+1 = 5
\]

rows: four bank features and one current prefix.

### 7.1 One fused aggregation

Ignoring the small learned vectors and assuming each source row is loaded once,
the ideal vector traffic is

\[
\underbrace{RTH}_{\text{source reads}}
+
\underbrace{TH}_{\text{output write}}
=(R+1)TH
\]

BF16 elements.

For \(R=5,H=7168\):

\[
6\times7168\times2 = 86{,}016
\]

bytes per token per aggregation.

A rough operation count is

\[
\left(5R+3\right)TH
\]

FLOPs: per row, square reduction, score dot product, and weighted FMA; then
output RMSNorm. For \(R=5,H=7168\), this is approximately 200,704 FLOPs per
token, or only 2.33 FLOP/byte. Large-prefill execution is necessarily
memory-bound.

### 7.2 Two aggregations in one decoder layer

A decoder layer has attention-side and MLP-side aggregations:

\[
2\times86{,}016 = 172{,}032
\]

ideal bytes per token, excluding attention, communication, and MLP.

On a layer that writes a new snapshot, add one \(TH\) BF16 store:

\[
7168\times2 = 14{,}336
\]

bytes per token. The SM100 kernel folds that write into the attention-side
aggregation, so it adds traffic but not another launch.

### 7.3 Measured B200 cost

The current benchmark was run with:

```bash
.venv/bin/python attn_res/bench_attn_res.py \
  --tokens 1 16 256 4096 16384 \
  --nvb 4 \
  --impl torch_eager sglang_triton sglang_tma \
  --warmup 10 --iters 30 --repeats 5 \
  --csv attn_res/results/bench_attn_res_current.csv
```

Environment: NVIDIA B200, BF16, \(H=7168\), four bank rows plus current prefix.
Times are CUDA-graph replay medians. “Layer” is exactly two aggregation calls;
it still excludes attention and MLP.

| tokens | Python eager, one agg | SGLang Triton, one agg | SGLang TMA, one agg | TMA, two per layer | TMA ideal bandwidth |
|---:|---:|---:|---:|---:|---:|
| 1 | 33.22 µs | 11.94 µs | **3.13 µs** | **6.27 µs** | 27 GB/s |
| 16 | 53.39 µs | 12.94 µs | **3.15 µs** | **6.30 µs** | 437 GB/s |
| 256 | 149.32 µs | 22.92 µs | **5.88 µs** | **11.75 µs** | 3.75 TB/s |
| 4,096 | 1,987.01 µs | 231.74 µs | **56.30 µs** | **112.61 µs** | 6.26 TB/s |
| 16,384 | 7,729.46 µs | 897.55 µs | **213.46 µs** | **426.92 µs** | 6.60 TB/s |

The result is clean:

* Decode is launch-latency dominated. The one-launch TMA path is about 3 µs.
* At 16K tokens, the TMA path reaches roughly 6.6 TB/s under the one-pass
  vector-traffic model.
* Triton's second source-row read makes it about 4.2× slower at 16K.
* The eager implementation is a correctness tool, not a performance baseline.

## 8. CuTeDSLGen search and the measured speed of light

The follow-up used CuTeDSLGen's spec-driven workflow rather than hand-tuning
only the two local examples. The complete campaign is in
`CuTeDSLGen/generation/workspaces/gen_attn_res_claude_0730`; its contract,
attempts, sweep harness, profiles, and negative result are reproducible.

Three generations converged on a persistent, single-load kernel that keeps
source fragments in registers and performs online softmax. The final candidate
uses 448 threads, five-row chunks, FP32-resident `cw`, paired accumulators, and
chunk prefetch. It passes all 32 tested combinations of
\(nvb\in[1,8]\) and \(T\in\{1,16,256,4096\}\), with worst relative L2 error
\(1.64\times10^{-4}\) against an independent FP32 reference.

It does not beat SGLang:

| tokens | SGLang TMA | best generated CuTe DSL | SGLang / CuTe |
|---:|---:|---:|---:|
| 1 | 3.33 µs | 6.20 µs | 0.54× |
| 16 | 3.36 µs | 6.23 µs | 0.54× |
| 256 | 6.15 µs | 9.51 µs | 0.65× |
| 4,096 | 56.8 µs | 96.8 µs | 0.59× |
| 16,384 | 213.8 µs | 361.0 µs | 0.59× |

The \(T=16{,}384\) result reproduced at 360.81 and 360.80 µs while the
in-process SGLang canary measured 213.76 and 213.85 µs. A sweep over
threads per CTA, chunk rows, and requested CTAs per SM never exceeded 0.66×.

### 8.1 A minimum-work streaming lower bound

To distinguish a weak generated kernel from an already optimal incumbent, the
campaign built a streaming probe. It reads exactly the same \(R=nvb+1\) source
rows and writes one output row, but deliberately omits scoring, reductions,
softmax, and output RMSNorm. A correct implementation cannot perform less work
over the same byte set.

| nvb | tokens | streaming probe | SGLang TMA | SGLang / probe |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 33.11 µs | 29.42 µs | 0.888 |
| 4 | 256 | 4.23 µs | 6.15 µs | 1.451 |
| 4 | 4,096 | 54.92 µs | 56.62 µs | 1.031 |
| 4 | 16,384 | 206.77 µs | 216.81 µs | 1.049 |
| 8 | 4,096 | 86.26 µs | 86.92 µs | 1.008 |

For large \(T\), SGLang is within roughly 1–5% of a kernel that performs none
of the required arithmetic. Nsight Compute independently measured about
6.53 TB/s at \(T=16{,}384\), approximately 98% of the achievable bandwidth
observed for this read-heavy workload. The production kernel is therefore at
the practical memory speed of light for large prefill.

Its low nominal occupancy is intentional rather than evidence of a bottleneck.
The TMA producer and double-buffered shared-memory ring hide memory latency
while consumer warps perform packed FP32 work. The generated kernel instead
executes about 168 million warp instructions versus roughly 46 million for
SGLang, cannot express the incumbent's packed `fma.rn.f32x2` path efficiently,
and lacks equivalent producer/consumer decoupling. At \(T\le16\), about 2.9 µs
of extra CuTeDSL/TVM-FFI launch overhead decides the result before device work
matters.

### 8.2 The one remaining kernel-level opportunity

\(T=256\) is different. SGLang launches `min(SM_count, T)` persistent CTAs, so
the grid is quantized into two waves on this B200 and lands 45% above the
streaming bound for \(nvb=4\). This is genuine headroom, not a measurement
artifact.

Exploiting it requires at least two truly resident CTAs per SM, or a two-CTA
cluster that splits \(H\) and combines reductions through distributed shared
memory. The generated design could not do that: retaining rows pushes register
use high enough that requesting more CTAs per SM does not make them resident.
A future attempt should target this narrow \(T\approx SM\_count\) regime,
budget registers for co-residency from the start, and emit packed FP32 math
through inline LLVM assembly. For large \(T\), replacing SGLang's kernel is not
a productive target; boundary fusion is the remaining meaningful direction.

## 9. Where fusion can still help

The production kernel has already fused the central chain:

```text
score RMS statistics
  + scalar projection
  + online softmax
  + weighted raw-row sum
  + output RMSNorm
  + optional bank write
```

The remaining opportunities are at its boundaries.

### 8.1 Fuse the pending prefix add on the standalone path

Outside the distributed fused path, SGLang may currently materialize

\[
c=p+h
\]

before calling the TMA kernel. A standalone kernel accepting both `p` and `h`
could add them while constructing the current source row.

This removes:

* one intermediate prefix write;
* one subsequent prefix read;
* one add-kernel launch.

The complication is that the current TMA producer bulk-copies contiguous rows.
The virtual row \(p+h\) does not exist contiguously, so the producer needs a
vector-add path into shared memory rather than a pure TMA copy.

### 8.2 Fuse normal TP all-reduce with full aggregation

The sequence-parallel path already has

```text
reduce-scatter + residual + attention-residual + RMSNorm
```

in one kernel. The regular TP path folds the residual into all-reduce but still
invokes attention-residual aggregation afterward. A monolithic

```text
all-reduce + residual + aggregation + RMSNorm
```

could remove another materialization and launch, subject to multicast buffer
and synchronization constraints.

### 8.3 Precompute frozen-bank work across a block

Within an attention-residual block, the bank is frozen and every future
consumer's score vector is static. It is therefore possible to batch the
inter-bank score and weighted-value work for several upcoming boundaries, then
merge only the evolving current prefix at each layer using online-softmax
statistics.

This trades repeated bank reads for a large precomputed-output buffer. It is
unlikely to help decode, where the current one-launch kernel is already around
3 µs. It may help long prefill, where bank traffic dominates, but it adds:

* storage proportional to future boundaries times \(T\times H\);
* scheduling and lifetime complexity;
* pipeline-parallel transfer concerns;
* another producer/consumer synchronization problem.

This should be evaluated as a prefill-specific algorithm, not assumed to be a
universal replacement.

### 8.4 Fuse into the following projection only with strong evidence

The attention-side output immediately feeds QKV projections; the MLP-side
output feeds gate/up projections. In principle, aggregation could become a
GEMM prologue.

In practice, aggregation needs full-\(H\) reductions and online softmax before
the normalized vector is available, while the GEMMs want tensor-core tiling
and substantially different register/shared-memory budgets. Combining them
risks reducing occupancy and duplicating the aggregation across GEMM tiles.

The current 6.6 TB/s prefill result leaves limited headroom inside the
aggregation itself. Boundary fusion is more promising than forcing the whole
consumer GEMM into the same CTA.

## 10. Benchmark and experimentation workflow

`attn_res/bench_attn_res.py` now benchmarks the current production contract.
It includes:

* a readable Python reference;
* optional `torch.compile`;
* SGLang's Triton fallback;
* SGLang's SM100 TMA kernel;
* an optional CuTe DSL candidate generated from the same contract.

Examples:

```bash
# Quick correctness and latency check
.venv/bin/python attn_res/bench_attn_res.py \
  --tokens 1 256 --nvb 4 \
  --impl torch_eager sglang_triton sglang_tma

# Sweep source counts and prefill sizes
.venv/bin/python attn_res/bench_attn_res.py \
  --tokens 1 16 256 4096 16384 \
  --nvb 1 4 8

# Fair eager-launch comparison with the CuTe DSL candidate
.venv/bin/python attn_res/bench_attn_res.py \
  --tokens 1 16 256 4096 16384 \
  --nvb 4 \
  --impl cute_dsl cute_dsl_v2 sglang_tma \
  --timing eager
```

Every optimized path is checked against the Python implementation before
timing. The benchmark reports both one-aggregation latency and the two-call
residual cost of a decoder layer.

## 11. Takeaway

Kimi-K3 Attention Residual is a learned softmax mixture over a tiny number of
very wide residual snapshots:

\[
\boxed{
y =
\operatorname{RMSNorm}_{\gamma_o}
\left(
\sum_i
\operatorname{softmax}_i
\left[
\frac{x_i^\top c_w}
{\sqrt{\operatorname{mean}(x_i^2)+\epsilon}}
\right]
x_i
\right)
}
\]

The keys are RMS-normalized views of frozen residual prefixes. The values are
the raw prefixes. The current stream is another softmax candidate, not an
unweighted residual added afterward.

The math is short; the engineering is about avoiding a second read of
\([T,R,H]\), eliminating decode-time launches, and composing the operation
with distributed communication. SGLang's SM100 implementation already turns
the central operation into a one-pass kernel near the B200 bandwidth ceiling.
The best remaining opportunities are therefore boundary fusions and
prefill-specific reuse of the frozen bank, not replacing it with ordinary
FlashAttention.
