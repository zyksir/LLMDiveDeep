# Dense attention and the FlashAttention family

Source/API inspection: 2026-09-10. The code here is forward-only; the discussion
of backward explains design choices, not something this harness measures.

## 1. Fix the operation before optimizing it

For one head, let $Q\in\mathbb R^{N_q\times d}$,
$K,V\in\mathbb R^{N_k\times d}$. With additive mask $M$:

$$
S=QK^T/\sqrt d+M,\qquad P_{ij}=\frac{e^{S_{ij}}}{\sum_u e^{S_{iu}}},
\qquad O=PV.
$$

Masked scores are $-\infty$, not zero. Softmax normalizes over **keys**, not
query positions or heads. We implement exactly these three visible operations
in [`matmul_attention`](dense_attention.py), accumulating in FP32 and returning
the input dtype. For numerical stability, softmax subtracts each row maximum.

Our public tensors are BSHD: `Q[B,Nq,Hq,D]`, `K/V[B,Nk,Hkv,D]`.
MHA has `Hq=Hkv`; GQA has several query heads per KV head; MQA has `Hkv=1`.
The reference repeats KV heads to make GQA obvious. Production kernels should
reuse KV without that physical repetition. No projection matrix is involved.

Queries are the final `Nq` positions of a sequence of length `Nk`, with
`0 < Nq <= Nk`. Query row $i$ has absolute position $p_i=N_k-N_q+i$.
Causal visibility is $j\le p_i$. Decode-shaped `Nq=1` therefore sees the entire
prefix. This bottom-right convention is important when calling different APIs.

## 2. Where the naive implementation spends memory

The two products cost approximately $4N_qN_kd$ FLOPs per head, counting a
multiply-add as two. Causality roughly halves useful square-prefill work.
Writing and rereading both $S$ and $P$ adds quadratic memory traffic. For
`B=1,H=32,N=32768`, one full FP32 score tensor alone would occupy 128 GiB.
The persistent KV cache is a different allocation; removing score tensors does
not remove the need to retain historical K/V.

For `Nq=1`, there is no large square score matrix. Decode instead tends to
emphasize KV reads, parallelism, launch overhead, and split/reduction costs.
A prefill winner cannot simply be declared the best decode kernel.

## 3. Derive online softmax, then understand fusion

Process one query row's keys in tiles. After some tiles retain only:

$$
m=\max s_j,\qquad \ell=\sum_j e^{s_j-m},\qquad
a=\sum_j e^{s_j-m}v_j.
$$

For the next tile with score vector $s$ and values $V_b$:

$$
\begin{aligned}
m'&=\max(m,\max s), & \alpha&=e^{m-m'},\\
p&=e^{s-m'}, & \ell'&=\alpha\ell+\sum p,\\
a'&=\alpha a+pV_b, & O&=a'/\ell'.
\end{aligned}
$$

The factor $\alpha$ expresses the previous partial sum in the new maximum's
coordinate system. This is why separately normalized tile outputs cannot just
be averaged. Initialization is $m=-\infty,\ell=0,a=0$; all-masked intermediate
tiles need care to avoid $-\infty-(-\infty)$. Follow these quantities in
[`online_attention`](dense_attention.py), including that edge case.

This recurrence gives the same mathematical softmax over the same keys.
Floating-point reduction order can change the answer slightly. A real fused
kernel keeps score/probability tiles and accumulators on chip and combines
loads, matrix multiplication, exponentiation, reductions, and stores in one
scheduled computation. The Python/PyTorch tiled reference illustrates the
algorithm but still launches many operators and is **not** FlashAttention.
FlashAttention avoids materializing the quadratic intermediates; it does not
turn dense attention into a linear-arithmetic algorithm. [Original paper][fa1]

There is also a useful merge rule for independently processed key partitions:
combine their $(m,\ell,a)$ with the same maximum/rescaling equations. That is
the mathematical basis for splitting a long KV sequence across workers and
reducing partial attention, at the expense of extra work and memory traffic.

## 4. What changes between FA2, FA3, and FA4?

| Family | Main question | What to look for in a kernel |
|---|---|---|
| FA2 | How can more SMs/warps do useful attention work? | Query-tile parallelism; warp ownership; fewer non-matmul operations |
| FA3 | How can Hopper's asynchronous engines stay busy together? | Producer/consumer roles, TMA loads, WGMMA, overlapping softmax |
| FA4 | What becomes limiting when Blackwell matrix math scales faster than other resources? | Asynchronous MMA/TMEM pipelines, softmax cost, shared-memory traffic |

### FA2: partition work well

Sequence/query tiles provide additional parallel work beyond batch and heads.
Inside a block, assigning query rows to warps lets them own their output
accumulators while sharing K/V, avoiding some cross-warp partial-output
communication associated with splitting K. Reducing non-matmul work also
matters because scalar operations do not run at tensor-core throughput. The
attention formula and allowed keys remain unchanged. [FA2 paper][fa2]

When reading code, ask which thread block owns `O[q_tile]`, whether there are
enough blocks for the GPU, and whether warps must exchange partial numerators.

### FA3: make hardware operations overlap

Hopper supplies TMA for asynchronous data movement and WGMMA for asynchronous
matrix operations. Producer/consumer warp specialization and carefully staged
pipelines overlap those with softmax work. Dependencies and barriers are as
important as arithmetic: a kernel can have fast instructions but leave one
engine idle waiting for another. FA3 also studies FP8 and incoherent processing
to improve low-precision accuracy; that is a separate numerical choice from
BF16 tiling. [FA3 paper][fa3]

When reading code, track buffer ownership: which stage fills a tile, which
stage consumes it, and which barrier permits the buffer to be overwritten?

### FA4: rebalance for asymmetric hardware scaling

On Blackwell, faster tensor cores can expose exponentiation and shared-memory
bandwidth as bottlenecks. FA4 co-designs larger-tile asynchronous MMA pipelines
with softmax optimizations, including software-emulated exponentials and
conditional rescaling. Its backward design uses tensor memory and 2-CTA MMA to
reduce shared-memory traffic and atomic updates. The implementation is in
CuTe DSL; “Python source” does not mean execution as Python tensor loops.
It is not merely FA3 translated into another language. [FA4 paper][fa4]

The new adapter uses `flash_attn.cute.flash_attn_func`, installed through the
separate `flash-attn-4` distribution. Consult upstream for CUDA/toolchain and
architecture requirements. The current API can return a tensor or additional
metadata depending on options; the adapter must not blindly take `output[0]`
from a tensor and lose the batch dimension. [FA4 implementation][fa4code]

Across all generations, “exact” means the same attention operation, not
bitwise identity. Dtype, approximate exponentials, reduction order, and
low-precision options must be recorded before comparing errors or speeds.

## 5. Map names to actual callable implementations

[`backends.py`](backends.py) prepares metadata once and returns a timed callable.

| Adapter | Actual entry point / selection | Important boundary |
|---|---|---|
| `torch_eager` | matmul → softmax → matmul | FP32 intermediates, GQA expansion |
| `torch_online` | tiled PyTorch recurrence | Teaching implementation, Python loop |
| `torch_sdpa_math` | SDPA, forced `MATH` | Library math path |
| `torch_sdpa_flash` | SDPA, forced `FLASH_ATTENTION` | Bundled PyTorch implementation, not a promise of external FA4 |
| `torch_sdpa_efficient` | SDPA, forced `EFFICIENT_ATTENTION` | Support varies with shape/mask/GQA |
| `torch_sdpa_cudnn` | SDPA, forced `CUDNN_ATTENTION` | Support varies with installed cuDNN |
| `torch_flex` | compiled `flex_attention` + `BlockMask` | Mask preparation and compilation excluded |
| `fa2` | `flash_attn.flash_attn_func` | External FA2 package |
| `fa3` | `flash_attn_interface.flash_attn_func` | Upstream Hopper installation/import |
| `fa4` | `flash_attn.cute.flash_attn_func` | External CuTe DSL implementation |
| `sglang_fa3` | `sgl_kernel.flash_attn.flash_attn_varlen_func` | Kernel wrapper, not a serving request |
| `sglang_fa4` | `sglang.kernels.ops.attention.flash_attention_v4` | Current SGLang wrapper; SM120 module differs |
| `flashinfer_fa2/fa3` | `single_prefill_with_kv_cache`, forced backend | This adapter only supports batch 1 |

PyTorch dispatch may otherwise select an implementation automatically. We force
one backend per row so a fallback does not masquerade as a successful test.
`is_causal=True` in ordinary non-square SDPA uses upper-left alignment; our
adapter supplies a lower-right bias where required, and no causal mask for
the single final query. Boolean mask `True` means keep. [PyTorch SDPA][sdpa]

SGLang's inspected [attention backend][sglang] dispatches to these wrappers and
also manages serving metadata and KV-cache behavior. The tutorial borrows the
kernel entry points, not its scheduler or layer. A varlen call on prepared
tensors is not a paged-cache decode benchmark. FlashInfer's [single-request
prefill API][fi] is likewise a different calling convention, not a different
definition of softmax.

## 6. How to reason about efficiency without inventing a winner

First match output semantics, shape, dtype, masks, and timed boundary. Then
measure on the actual GPU. For long prefill, examine tensor-core utilization,
on-chip resource pressure, tile size, and overlap. For short decode, examine
KV bandwidth, GQA reuse, enough independent work, and split/reduction overhead.
Larger tiles reduce some overhead but can reduce occupancy; more splitting
increases parallelism but creates reduction work. No choice wins everywhere.

The driver [bench_dense_attention.py](bench_dense_attention.py) reports latency,
not misleading dense-equivalent TFLOPs for masked/sparse problems. It bounds
the eager oracle to prevent accidental quadratic allocations at long context.
No fresh performance conclusions are supplied here because no runs were requested.

[fa1]: https://arxiv.org/abs/2205.14135
[fa2]: https://arxiv.org/abs/2307.08691
[fa3]: https://arxiv.org/abs/2407.08608
[fa4]: https://arxiv.org/abs/2603.05451
[fa4code]: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md
[sdpa]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
[sglang]: https://github.com/sgl-project/sglang/blob/52c191da52390fa5508de98eddd1e3eca2dbcfb2/python/sglang/srt/layers/attention/flashattention_backend.py
[fi]: https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html
