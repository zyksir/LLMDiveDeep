# Sliding-window attention: change the support, keep the softmax

## 1. Define the window precisely

Our window $W$ **includes the current token**. For query row $i$ at absolute
position $p_i=N_k-N_q+i$, the allowed keys are

$$
\mathcal W_i=\{j:\max(0,p_i-W+1)\le j\le p_i\}.
$$

Attention is the usual normalized weighted sum over this set. It is exact for
this windowed model, but generally differs from dense causal attention. A
finite local window is not a numerical optimization of an unchanged dense
model. With `Nk=8, Nq=2, W=3`, query positions 6 and 7 see `{4,5,6}` and
`{5,6,7}` respectively. This small case catches both alignment and off-by-one errors.

## 2. Three implementations of the same equation

1. **Dense masked reference:** compute every score, mask outside the window,
   then softmax. Correct, but still quadratic prefill work/allocation.
2. **Tiled online reference:** for a query tile, visit only key tiles intersecting
   its window band, and mask partially intersecting tiles. The local support
   is smaller before matrix multiplication. Follow `lower`, `upper`, and `keep`
   in [`online_attention`](dense_attention.py).
3. **Fused implementation:** perform tile skipping and online softmax inside
   a scheduled GPU kernel. The Python tiled version illustrates this logic,
   but is not itself a fast fused implementation.

Ideal local prefill arithmetic is $O(N_qWd)$ per head when the window is full;
boundary tiles can do extra work. With query tile size $B_q$, the union of its
windows is about $W+B_q-1$ keys. A large tile can therefore waste arithmetic
on a narrow band even though it improves matrix-multiplication utilization.

## 3. Match each open-source API

| Implementation | Parameter matching this tutorial's W | What to watch |
|---|---|---|
| PyTorch matmul | explicit boolean support | `True` means keep |
| PyTorch SDPA | explicit support mask | A forced fused backend may reject it; dense mask alone does not promise sparse execution |
| PyTorch FlexAttention | `create_block_mask(mask_mod)` | BlockMask allows entire blocks to be skipped |
| Upstream FA2/FA3/FA4 | `window_size=(W-1, 0)`, causal | Right-aligned query positions for unequal lengths |
| SGLang FA3/FA4 wrapper | same tuple on prepared varlen Q/K/V | Its model-derived sliding-window field can already be `W-1` |
| FlashInfer FA2/FA3 prefill | `window_left=W-1`, causal | This tutorial's single-request adapter requires B=1 |

Upstream FlashAttention documents local-window semantics in its [interface
documentation][fa]. FlexAttention distinguishes changing individual scores from
constructing a block mask that allows computation to be omitted; our adapter
uses the latter and excludes mask construction/compilation from timing.
[PyTorch explanation][flex]

In the inspected SGLang [backend][sglang], `fa_impl_ver` chooses FA3 or FA4
wrappers. The sliding-window value passed from its model plumbing has already
been adjusted to left-distance units. Do not subtract one twice when porting
code. The new tutorial API deliberately accepts the human-readable token count
W and converts it once. [SGLang FA4 wrapper][sglang4]

FlashInfer's [prefill interface][fi] exposes a left-window distance as well.
Neither API requires us to copy a whole model implementation just to compare
the same local softmax kernel.

## 4. Logical positions are not physical cache addresses

In serving, a local cache can be a ring or paged allocation. The query still
uses absolute positions for causality/RoPE, even when the physical slot is
`position % capacity`. Cache slots cannot be interpreted as time positions
without metadata. Eviction, cache insertion, sequence scheduling, and ring
rotation are outside the new dense/window benchmark boundary.

Our fixtures retain a contiguous full KV prefix and ask the kernel to read only
the window. Thus they compare windowed **compute**, not the memory savings of a
production ring-cache manager. Decode `Nq=1` is also not proof that the adapter
uses a library's specialized paged decode kernel.

## 5. Later benchmark and correctness checklist

Use [bench_sliding_window_attention.py](bench_sliding_window_attention.py) with
`--run` only when ready. It shares the dense backend registry and correctness
policy. Compare windowed output to windowed output, never to dense attention.

Useful future cases are W=1, W larger than the sequence, uneven tile lengths,
Q=1, unequal Q/K lengths, MHA/GQA/MQA, and prefixes shorter than W. For a timing
sweep vary N at fixed W, then W at fixed N. A mask added after full QK cannot
demonstrate O(NW) scaling merely because its output is correct. No experiments
were run during this documentation pass.

[fa]: https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention
[flex]: https://pytorch.org/blog/flexattention/
[sglang]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/layers/attention/flashattention_backend.py
[sglang4]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/kernels/ops/attention/flash_attention_v4.py
[fi]: https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html
