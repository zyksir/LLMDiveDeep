# CSA2 implementation alignment with SGLang

This study distinguishes **shared operator conventions** from **native model
support**. The former are implemented and checked where possible. The latter
is not present in the inspected V4 model path.

## 1. Exact version boundary

| Source | Inspected revision |
|---|---|
| DeepSeek V4.1-Flash release | `dba1be0a40aa45a94ad051997016db3960a90277` |
| SGLang upstream | `52c191da52390fa5508de98eddd1e3eca2dbcfb2` |
| Local `/node-storage/sglang` | `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e` |

The public [V4 model][sgmodel] has an explicit `MqaAttentionBase` constructor
assertion permitting compression ratios `(0, 4, 128)` (lines 666–675 at this
pin). The released [V4.1 config][dsconfig] uses `0`, `2`, and `1`, declares
`DeepseekV41ForCausalLM`, and includes separate KV-source, index-source and
candidate-source settings. An inspection of the complete upstream file tree
found no separately named `deepseek_v41` model/config path at this revision.
Neither renaming the model nor deleting the ratio assertion supplies CSA2.

This is a statement about inspected commits, not a claim that no external
branch or subsequent release supports V4.1. Generic Hugging Face “use with
SGLang” snippets do not establish support for these architectural contracts.

## 2. The source map

Paths are relative to the SGLang repository at the pin above. The linked
DeepSeek names are in the official `inference/model.py` unless stated otherwise.

| Stage | DeepSeek V4.1 source | SGLang source / symbol | Local implementation |
|---|---|---|---|
| Q/KV projections, query low rank, sink, output groups | `Attention` | `python/sglang/srt/models/deepseek_v4.py`: `MqaAttentionBase`, `MQALayer` | Inputs/outputs documented; projections are outside the operator reference |
| Main token compression | `Compressor` | `python/sglang/srt/layers/attention/dsv4/compressor.py`, `compressor_v2.py` are the older V4 paths | `gated_compress`, `StreamingCompressor`: CSA2 non-overlap semantics |
| Indexer score and head reduction | `Indexer.forward` | `python/sglang/srt/layers/attention/dsv4/indexer.py`: `C4Indexer`, `C4IndexerBackendMixin`, `fp8_paged_mqa_logits_torch` | `index_scores`, `causal_index_scores` |
| Top-k and logical→physical conversion | `Indexer.forward` returns logical IDs plus a concatenation offset | Same SGLang indexer file: `_topk_transform_vectorized`, `topk_transform_paged` calls | `select_topk`, `pack_sparse_prefill` |
| Hierarchical candidate pool | `select_candidate_blocks`, `Indexer.forward` | No matching CSA2 source-owner/candidate path identified in the inspected V4 model | `select_candidates`, `reindex_candidates` |
| Cross-layer main/index cache ownership | `SharedAttentionRuntime`, `Attention._compress_kv` | Inspected V4 module cannot express the released CSA2 ownership schedule | `config.layer_spec`, `CSA2Router` |
| Joint SWA/global sparse attention | `kernel.py::sparse_attn` | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`: `forward`, `_forward_prefill_sparse` | `joint_sparse_attention`; optional `sparse_attention_sglang` |
| Packed main-FP4 cache | `fp4_act_quant(...,16,...,scale_dtype=E4M3)` | Inspected backend includes FP8 cache contracts; its FP4 indexer is not proof of CSA2 main-FP4 compatibility | No physical FP4 packing in this tutorial |
| CED optimized prefill / SWA bounded replay | Report §3.2.2; minimal `Transformer.forward` is a full layer loop | No V4.1 EPD/replay integration validated here | Explained in research.md; not implemented |

Relevant inspected source: [DeepSeek model][dsmodel], [SGLang model][sgmodel],
[SGLang indexer][sgindex], [SGLang attention backend][sgbackend].

## 3. What the executable reference aligns

The public `csa2_reference.py` boundary takes **projected floating-point
tensors**, with normalization/RoPE/dequantization already applied except at
the explicitly pre-normalization compressor boundary. It preserves:

- one main KV latent shared by main query heads;
- the indexer's dot product → ReLU → signed head weighting → head sum order;
- causal completion masks for compressed entries;
- per-query, head-shared global selections;
- a joint local/global softmax with a per-head zero-value sink;
- Full/Reindex/Reuse responsibilities, including exact cache/selection identity
  in Reuse layers;
- the candidate block maximum and newest-block pin;
- fixed candidate-domain computation for later indexers.

It intentionally makes a few execution choices differently:

| Choice | Teaching code | Production implication |
|---|---|---|
| Precision | FP32 accumulation on floating-point inputs | FP4/FP8 quantization and fused rounding require separate parity checks |
| Top-k ties | Stable, lower-position preference | CUDA top-k may return another valid tied selection |
| Short-prefix output width | Fixed `K`, padded with `-1` | Official minimal model uses `min(K, available-cache-length)` |
| Candidate representation | Position IDs with invalid slots removed/padded | Official helper returns a block-expanded boolean mask, including masked future slots |
| Deeper indexing | Gather keys before scoring | Official minimal model computes dense scores then masks; these are semantically equivalent after causal masking |
| Full prefill indexing | Materialized dense scores | Production needs tiled/quantized score computation; the reference is not a long-context benchmark |
| Shared state | New explicit router per forward query batch | Official minimal runtime uses a single process-global object |
| Local history | Logical contiguous KV | Serving uses ring/page indirection and request-owned cache pools |

## 4. Direct SGLang sparse-core adapter

[sglang_adapter.py](sglang_adapter.py) calls the same
`sgl_kernel.flash_mla.flash_mla_sparse_fwd` entry point used by the inspected
SGLang `_forward_prefill_sparse`. This bypasses SGLang's V4 model and its ratio
checks because compression and selection have already happened upstream.
It does **not** make SGLang understand a V4.1 checkpoint.

The pinned call expects:

| Argument | Adapter shape / meaning |
|---|---|
| `q` | `[B*Q,64,512]`, BF16 |
| `kv` | `[total_workspace_entries,1,512]`, BF16 dequantized workspace |
| `indices` | `[B*Q,1,padded_K]`, int32, physical workspace offsets |
| `topk_length` | `[B*Q]`, int32 length of each valid index prefix |
| `attn_sink` | `[64]`, FP32 |
| `sm_scale` | `512**-0.5` |
| `d_v` | `512` |

The adapter concatenates local/main caches per request, flattens requests,
rebases indices by the request's workspace offset, compacts valid indices to
the front, and pads the tail with `-1`. It never conflates an original token
position, a compressed-entry ID, or a physical workspace address. In particular,
counting valid entries without removing holes is insufficient for a kernel
whose `topk_length` describes a valid prefix.

For paged decode, the inspected SGLang backend instead calls
`flash_mla_with_kvcache`, supplying a SWA cache plus `extra_k_cache`, separate
index arrays and lengths, `attn_sink`, and scheduler metadata. Passing our
logical tensors directly into that API would be wrong. The included adapter
only covers the dequantized sparse-prefill core API. [Backend source][sgbackend]

In an existing CUDA SGLang environment with a compatible FlashMLA build:

```bash
python3 -m deepseek.sglang_adapter --check
```

This compares the core output on synthetic BF16 inputs against the floating-point
oracle. It fails explicitly if CUDA or the kernel is unavailable. **This GPU
command has not been validated in the current CPU test environment.** Hardware
dispatch, build availability and numerical tolerance must be checked before
using the adapter in a performance comparison. It is not a timing harness.

## 5. Reproduce the upstream CPU helper checks

The test reads pinned, hash-verified source files and extracts just the named
function AST. It does not import DeepSeek's whole model or require SGLang's
CUDA dependency graph. The exact source hashes are in `sources.json` and are
enforced by the parity script.

```bash
deepseek_sources="$(mktemp -d /tmp/deepseek-parity-XXXXXX)"
curl -fL \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py \
  -o "$deepseek_sources/model.py"
curl -fL \
  https://raw.githubusercontent.com/sgl-project/sglang/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/layers/attention/dsv4/indexer.py \
  -o "$deepseek_sources/indexer.py"
python3 -m deepseek.check_source_parity \
  --deepseek-model "$deepseek_sources/model.py" \
  --sglang-indexer "$deepseek_sources/indexer.py"
```

Observed checks on 2026-09-10, CPU PyTorch `2.14.0+cpu`:

| Check | Result | What it establishes |
|---|---|---|
| Nine CSA2 unit tests | PASS | Compression chunk consistency; causal-prefix invariance; sparse core versus independent dense mask; sink behavior; candidate/gather equivalence; released ownership and cache counts |
| One adapter packing test | PASS | Batch rebasing, valid-prefix compaction, padding, and gathered value identity |
| DeepSeek `select_candidate_blocks` | PASS | Candidate membership after intersecting with causal visibility; widths 1, 7, 19, 65; batch 2; all prefix lengths including zero |
| SGLang `fp8_paged_mqa_logits_torch` | PASS | Exact score equality on representable FP8 inputs, signed weights, permuted page tables and unequal request lengths |
| Full quantized model, FlashMLA GPU outputs, serving performance | Not run | No claim of coverage |

The SGLang helper check uses its **FP8 reference path**, not V4.1's FP4
indexer. It isolates page decoding and the shared score formula with exactly
representable inputs. The helper zero-fills invalid positions under
`clean_logits=False`; real top-k must apply visibility separately, since
invalid zero scores could outrank valid negative scores. The tutorial masks
invalid entries to `-inf` before selection.

## 6. What native V4.1 integration would require

These are concrete missing responsibilities at the inspected SGLang boundary,
not changes to the user repository's older V4 experiments:

1. Register the V4.1 configuration/model and map the released checkpoint
   projections, norms, Engram, mHC and draft modules correctly.
2. Allocate shared global pools by **KV owner**, with consumers pointing to
   them; retain separate per-layer local SWA pools. Manage source/consumer
   lifetimes across pipeline stages and prefix-cache operations.
3. Implement the non-overlap ratio-2 compressor and ratio-1 projection path,
   deriving index K from the pre-RoPE latent before main-cache quantization.
4. Add Full/Reindex/Reuse routing with batch-specific selections and candidate
   metadata; gather candidate blocks before deeper scoring.
5. Support the released main-FP4 and index-MXFP4 layouts independently, plus
   SWA FP8, in pool allocation, cache write/dequantization and kernel dispatch.
6. Introduce CED prefill/global-memory construction and bounded SWA replay,
   keeping exact-cache and approximate-replay validations separate.
7. Validate request batching, causal ragged prefill, odd compression tails,
   cache hits, eviction/retraction, speculative verification and parallelism
   before measuring end-to-end latency or claiming model accuracy parity.

For kernel study, the immediate executable boundary is much smaller: take
correct projected tensors, compression/selection results and layout metadata,
then compare a sparse core with its dense masked oracle. That is the boundary
implemented here.

[dsmodel]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
[dsconfig]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json
[sgmodel]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/models/deepseek_v4.py
[sgindex]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/layers/attention/dsv4/indexer.py
[sgbackend]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/layers/attention/deepseek_v4_backend.py
