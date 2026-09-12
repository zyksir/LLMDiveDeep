# KDA: equations, state contracts, and a kernel-by-kernel reading map

This guide maps the local source, not a newly measured speed ranking. No KDA
kernel, autotuner, self-test, benchmark, or profiler was executed in this pass.
Historical comments such as “champion” or “wins” describe previous experiments;
they are not evidence that a kernel is currently fastest on your shapes.

Read [SOURCE_INVENTORY.md](SOURCE_INVENTORY.md) for every Python module and
top-level symbol. This document explains the families and their differences.
Existing [KDA.md](../KDA.md) and [SURVEY.md](../SURVEY.md) are supplementary
historical notes. The new tutorial compares **kernels**, not attention layers.

## 1. Start with the recurrence, not a backend name

Use one head and key-first state $S\in\mathbb R^{K\times V}$. After L2
normalizing q/k, let $g_t\in\mathbb R^K$ be log-decay and $\beta_t$ the
post-sigmoid update strength. The reference operation is

$$
\begin{aligned}
\widetilde S_t &= \operatorname{diag}(e^{g_t})S_{t-1},\\
u_t &= \beta_t\big(v_t-\widetilde S_t^T k_t\big),\\
S_t &= \widetilde S_t+k_tu_t^T,\\
o_t &= S_t^T q_t/\sqrt K.
\end{aligned}
$$

The prediction uses the **already decayed state**. Omitting that detail changes
the recurrence. The delta correction writes the residual between the desired
value and the value predicted at key k. KDA's decay is per key channel, unlike
a scalar-per-head decay. A scalar-gate GDN kernel cannot be renamed KDA.
Read [`kda_recurrent_reference`](kda_attention.py) line by line first.

Unlike softmax attention, this recurrence retains a fixed-size state rather
than a growing retrievable KV history. Its update/read work is O(KV) per head
per token and its persistent state size is O(HKV), independent of context
length. That is a different model equation, not a faster implementation of
dense softmax. Training quality equivalence is not implied by a kernel speedup.

## 2. Gate and layout contracts can invalidate a comparison

[`activate_kda_gate`](kda_attention.py) implements two parameterizations:

$$
g_{t,h,c}=-e^{A_h}\operatorname{softplus}(x_{t,h,c}+b_{h,c})
\quad\text{(canonical)},
$$

$$
g_{t,h,c}=L\,\sigma\!\left(e^{A_h}(x_{t,h,c}+b_{h,c})\right),\quad L<0
\quad\text{(bounded / safe gate)}.
$$

These are not equivalent functions of the same raw logits. Some kernels
receive raw logits and fuse activation; others receive preactivated log-decay.
The safe-gate lower bound is part of the contract, not a universal constant to
guess from the backend name. Likewise, check raw beta logits versus post-sigmoid
beta and raw output-gate z versus preactivated sigmoid(z).

| Boundary | Local contract | Reading target |
|---|---|---|
| Math oracle | q/k/v BTHD, log-decay, post-sigmoid beta, state `[B,H,K,V]` | `kda_attention.py` |
| Serving decode fixture | packed QKV, raw gate/beta, indexed FP32 state pool | `inputs.py:DecodeInputs` |
| Serving state storage | Frequently `[slot,H,V,K]`, K contiguous | Registry conversion plus concrete kernel loads |
| Prefill fixture | Packed varlen `[1,total_tokens,H,D]`, `cu_seqlens`, raw gate, post-beta | `inputs.py:PrefillInputs` |
| Local dense chunk API | `[B,T,H,128]`, preactivated gate, explicit S0 | `b10_kda_chunk_prefill_cutedsl.py` |

A transpose view and a physical transpose are different costs. An indexed pool
is not equivalent to a contiguous per-batch state without a gather/scatter
policy. Verify whether the callable mutates S0, returns a new state, or writes
a deferred checkpoint. At K=V=128, both state orientations have identical
shapes, so shape checks alone cannot catch an orientation bug.

## 3. Separate the core from fusion boundaries

The local decode studies contain these increasingly broad operations:

```text
raw packed QKV → depthwise conv4 + SiLU → q/k normalization + gate activation
              → KDA recurrence → gated RMSNorm → output
```

The optional convolution consumes the previous three committed raw values
per channel and the current value. The optional epilogue is

$$
y=r\,[\operatorname{mean}_V(r^2)+\epsilon]^{-1/2}\odot w\odot\sigma(z).
$$

These are useful **kernel fusion** studies, but a bare recurrence time cannot
be compared directly with the entire chain. A fused path can avoid writing
post-conv q/k/v and pre-norm r to HBM. It may also keep them in FP32 where a
separate BF16 chain rounds between stages, creating expected numerical
differences. Inspect `conv4_silu_reference` and `gated_rmsnorm_reference`.

Projection GEMMs, the f_b GEMV experiment, full layer wrappers, and tensor
parallel collectives remain outside this tutorial's matched attention-core
boundary. Do not erase those older experiments, but label their scope.

## 4. Decode: understand ownership of the state tile

The state read/write volume alone is about $8BHKV$ bytes for FP32 read+write
in a plain decode step, before inputs/output. With small batches it can be
hard to expose enough independent work. Splitting the V dimension among CTAs
adds parallelism because each output column has its own recurrence; splitting
K introduces reductions for the state prediction and output.

| Public source / family | Implementation to follow | What distinguishes it |
|---|---|---|
| [`b10_kda_decode_cutedsl.py`](b10/b10_kda_decode_cutedsl.py) | [`_b10_kda_decode_impl_cutedsl.py`](b10/_b10_kda_decode_impl_cutedsl.py) | Bare recurrence; launch/configuration variants, register state, vectorized loads |
| [`b10_kda_decode_gated_cutedsl.py`](b10/b10_kda_decode_gated_cutedsl.py) | Same private implementation | Adds gated-RMSNorm output; changes reduction/ownership requirements |
| [`b10_kda_decode_conv_cutedsl.py`](b10/b10_kda_decode_conv_cutedsl.py) | Same private implementation | Adds input conv4/SiLU |
| [`b10_kda_decode_conv_gated_cutedsl.py`](b10/b10_kda_decode_conv_gated_cutedsl.py) | **Substantial separate implementation**, not just a re-export | Packed/raw/strided interfaces, launch dispatch, fully fused conv/core/norm variants |
| [`b10_kda_gated_rmsnorm_cutedsl.py`](b10/b10_kda_gated_rmsnorm_cutedsl.py) | Standalone epilogue | Useful to isolate norm cost and the launch removed by fusion |
| [`b10_kda_decode_conv_gated_fusefb_cutedsl.py`](b10/b10_kda_decode_conv_gated_fusefb_cutedsl.py) | Projection-fusion variant | Includes f_b work beyond the attention-core boundary; legacy experiment |

Begin inside `make_launcher` and follow `cute.kernel` bodies, not only the
public wrapper. Track which lane owns contiguous K channels, how state is
distributed across registers, and which reductions require shuffles or shared
memory. Register pressure can erase savings from fusion. Larger vector loads
help only if addresses and layouts actually permit coalescing.

## 5. Prefill: turn a token recurrence into chunk matrix operations

A sequential token loop is readable, but long prefill benefits from doing
intra-chunk work as matrix multiplications. Here is a derivation independent
of any particular launch schedule. Within a chunk define cumulative channel
log-decay $G_t=\sum_{a=1}^{t}g_a$, and $D_t=\operatorname{diag}(e^{G_t})$.
Unrolling gives

$$
S_t=D_t S_0+\sum_{j\le t}\operatorname{diag}(e^{G_t-G_j})k_j u_j^T.
$$

Substitute this into the prediction used to compute $u_t$:

$$
u_t=\beta_t v_t-\beta_t S_0^T D_t k_t
-\sum_{j<t} L_{tj}u_j,\qquad
L_{tj}=\beta_t k_t^T\operatorname{diag}(e^{G_t-G_j})k_j.
$$

Let R have row $\beta_t k_t^T D_t$, B be diagonal beta, and U collect rows
$u_t^T$. Then

$$
(I+L)U=BV-RS_0,\qquad
U=(I+L)^{-1}BV-(I+L)^{-1}RS_0.
$$

L is strictly lower triangular. For chunk size C, $L^C=0$, so

$$
(I+L)^{-1}=I-L+L^2-\cdots+(-L)^{C-1}.
$$

The finite series is algebraically exact, not an approximate infinite series
cut off because it “seems converged.” Kernels can organize these triangular
computations through small matrix operations. Names such as W/U/WY differ
across implementations: identify which quantity already includes beta, decay,
or the initial-state correction before equating two temporary tensors.

Now compute outputs by multiplying q against the unrolled state, including
the j=t update, and carry only the chunk's final state to the next chunk.
Prep can work in parallel across chunks; the carry stage still has inter-chunk
dependencies. This explains the usual **prep + carry/output** decomposition.
Factoring exponentials naively can overflow even if the final answer is finite;
bounded gates, chunk size, accumulation dtype, and clamping are numerical
design choices to audit, not incidental implementation details.

| Local prefill source | Main kernels | Reading questions |
|---|---|---|
| [`b10_kda_prefill_triton.py`](b10/b10_kda_prefill_triton.py) | `_prep_kernel`, `_carry_kernel` | C=16/64; gate/norm/cumsum and triangular solve; materialized W/U; V-split carry |
| [`b10_kda_chunk_prefill_cutedsl.py`](b10/b10_kda_chunk_prefill_cutedsl.py) | `_kern_prep16`, `_kern_carry16` | SC=16 mathematical subchunks inside C=64 scheduling tiles; layout/swizzle; sequential carry; preactivated gate; T divisible by 16 |
| [`b10_kda_prefill_cutedsl.py`](b10/b10_kda_prefill_cutedsl.py) | `make_launcher`, `kda_spec_decode` | Register-resident sequential T loop used as a short-prefill probe, not the chunk algorithm |

The Triton port stores prepared W/U, whereas other FlashKDA-family schedules
can store an inverse and reconstruct products in carry. This is a recompute
versus bandwidth/resource tradeoff. The local Triton code also clamps decay
exponents; do not assume agreement for all extreme canonical-gate inputs.
The CuTe chunk API clones S0 and returns the final state; its signature is not
a drop-in replacement for a serving varlen/state-pool API.

## 6. Verification: acceptance is part of the state contract

Speculative verification computes T=1+gamma candidate steps before the accepted
prefix length is known. Producing candidate outputs and committing accepted
state are separate operations. Three local families implement different
storage/commit strategies:

| Family | What verification retains | How acceptance is handled |
|---|---|---|
| `save_ssm` | Full state snapshot after each candidate | Gather the accepted snapshot into committed state |
| `replay_ssm_split` | Raw per-token records | Replay/fold exactly the accepted prefix in a separate commit kernel |
| `replay_ssm` | Checkpoint plus solved update/key/cumulative-gate records | Reconstruct logical state from accepted history; fold checkpoint when needed |

Full FP32 snapshots write approximately $4TBHKV$ bytes, while each replay
record is O(K+V), plus metadata. Replay reduces storage traffic but adds
reconstruction work and a more involved ownership contract. Rejected records
must not become accepted history. The raw convolution window must follow the
same acceptance rule. “Verify did not change S” is not enough if another ring
or convolution buffer was mutated incorrectly.

| Public source(s) | Underlying implementation / role |
|---|---|
| [`b10_kda_save_ssm_cutedsl.py`](b10/b10_kda_save_ssm_cutedsl.py), [`save_ssm_conv`](b10/b10_kda_save_ssm_conv_cutedsl.py) | [`_b10_kda_save_ssm_impl_cutedsl.py`](b10/_b10_kda_save_ssm_impl_cutedsl.py): snapshot family, bare and conv wrappers |
| [`save_ssm_gated`](b10/b10_kda_save_ssm_gated_cutedsl.py), [`save_ssm_conv_gated`](b10/b10_kda_save_ssm_conv_gated_cutedsl.py) | [`_b10_kda_save_ssm_conv_gated_impl_cutedsl.py`](b10/_b10_kda_save_ssm_conv_gated_impl_cutedsl.py): separate fused implementation |
| [`b10_kda_replay_ssm_cutedsl.py`](b10/b10_kda_replay_ssm_cutedsl.py), [`replay_ssm_conv`](b10/b10_kda_replay_ssm_conv_cutedsl.py) | [`_b10_kda_replay_ssm_impl_cutedsl.py`](b10/_b10_kda_replay_ssm_impl_cutedsl.py): checkpoint/ring family, V-split bare path |
| [`replay_ssm_gated`](b10/b10_kda_replay_ssm_gated_cutedsl.py), [`replay_ssm_conv_gated`](b10/b10_kda_replay_ssm_conv_gated_cutedsl.py) | [`_b10_kda_replay_ssm_conv_gated_impl_cutedsl.py`](b10/_b10_kda_replay_ssm_conv_gated_impl_cutedsl.py): separate one-CTA norm-capable path |
| [`b10_kda_replay_ssm_conv_gated_wychunk_cutedsl.py`](b10/b10_kda_replay_ssm_conv_gated_wychunk_cutedsl.py) | One-launch conv + small-chunk WY matrix formulation + norm, preserving replay semantics |
| [`kda_chunk_verify_triton.py`](kda_chunk_verify_triton.py) | `_chunk_prep_kernel`, `_chunk_state_kernel`, `_chunk_fold_kernel`; multi-launch chunk/replay alternative |
| [`kda_replayssm_fold.py`](kda_replayssm_fold.py) | Separate accepted-prefix commit from raw records; not the same solved-U ring ABI |

The fused replay implementation stages convolution history in a layout suited
to vector loads, transforms recurrence inputs using cumulative decay, replays
accepted records, runs the new candidate recurrence, and normalizes outputs.
Fusing a norm over all V channels favors one CTA owning the whole head state:
splitting V across CTAs would require cross-CTA normalization. Its launch
configuration also varies by T, so do not assume one fixed register count or
occupancy from a historical docstring. Inspect the actual dispatcher.

For correctness, compare **logical state reconstructed from checkpoint plus
accepted records**, not raw checkpoint tensors between different strategies.
`logical_s_from_ring`, `snapshot_oracle`, `verify_reference`, and `e2e_oracle`
in [kda_verify_register.py](kda_verify_register.py) reveal those boundaries.
No speculative serving policy or full model is needed to study these kernels.

## 7. Trace external wrappers before counting implementations

The registries are the authoritative local binding map:

- [kda_decode_register.py](kda_decode_register.py): PyTorch oracle, local B10,
  FLA recurrent, SGLang packed/split/fused, vLLM recurrent, TRT-LLM
  packed/split/fused. The `sglang_kda_cutedsl` function is present but its
  registration is disabled. The FlashInfer adapter's availability note refers
  to the old environment; it is not a current package-capability guarantee.
- [kda_prefill_register.py](kda_prefill_register.py): FLA canonical and safe-gate
  baselines, Moonshot FlashKDA, INT21 PTX, vendored chunk paths, TRT-LLM CuTe,
  local Triton/CuTe, and FlashInfer CAKE/recurrent paths. Inspect state dtype:
  the local CAKE adapters use BF16 state rather than the common FP32 state.
  That is a meaningful semantic/numerical difference, not just layout.
- [kda_verify_register.py](kda_verify_register.py): closure builders for
  snapshots, replay, commit, composed conv/norm chains, and chunk verification.
  A “stitched” or “chain” row can include several launches; the name alone
  does not tell you the timed boundary.

SGLang/vLLM/TRT-LLM vendored FLA code often shares an algorithmic lineage.
The SGLang FlashKDA wrapper can call the very same `flash_kda.fwd` as the direct
row. A package-name comparison is not necessarily an algorithm comparison.
External project entry points for further reading are [FLA][fla],
[FlashKDA][flashkda], and [INT21 KDA-B200][int21]; the local registry pins the
actual imported symbols and any environment-specific loading behavior.

`attention_modules.py`, `KDA_SGLANG_EXTEND` wrappers that include broader
serving preparation, and `bench_b10_kda_fusefb.py` must be reported separately
from a prepared recurrence core. Their source is retained for future study.

## 8. One-by-one walkthrough plan for later sessions

For **each** kernel, answer these in order:

1. Write its equation, gate parameterization, normalization epsilon, and dtype.
2. List inputs/outputs, strides, aliases, in-place writes, and accepted-prefix
   ownership. Include every state buffer, not just the main matrix.
3. Draw the CTA/warp/lane ownership of K, V, and time. Count state registers and
   identify tensor-core versus scalar/reduction work.
4. Trace global → shared → registers/TMEM → output, including transpose,
   quantization, barriers, and reused buffers.
5. Mark fused work and work prepared by the wrapper outside timing.
6. Design a small nonzero-initial-state oracle check, including rejected
   speculative tokens where applicable. Do not rely only on output cosine.
7. Only then benchmark/profiling: separate compilation, warmup, kernel time,
   commit cost, and setup. Investigate a bottleneck before choosing a rewrite.

Suggested order: math oracle → bare decode → gated decode → conv-gated decode →
Triton chunk prep/carry → CuTe chunk prep/carry → snapshots → raw-record commit →
solved-record replay → chunk verification → fusion variants. Existing
`linear_attn/tmp/` probes are historical experiments, not substitutes for this
contract audit. All future artifacts belong under `attention/results/kda/`.

[fla]: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
[flashkda]: https://github.com/MoonshotAI/FlashKDA
[int21]: https://github.com/Int21-AI/KDA-B200
