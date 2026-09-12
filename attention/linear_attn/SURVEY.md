# Linear Attention for Kimi K3 — math, implementations, and experiments

> Historical study / retained source context. Start with the [kernel tutorial](../README.md)
> and [current reading guide](kda/KERNELS.md). Old CSV/log/PNG and local_results artifacts
> were removed for regeneration; measurements below were not rerun or revalidated.
> Future outputs belong under attention/results/, not the paths in old examples.
> Layer/projection benchmarks below are outside the new attention-core comparisons.


*Snapshot: 2026-07-21. The target is eventual Kimi K3 support. K3's weights and
technical report are scheduled for 2026-07-27, so all K3-specific unknowns are
marked rather than inferred from Kimi-Linear.*

## 1. High-level summary

- Causal linear attention compresses history into a fixed-size recurrent
  matrix. Prefill scales linearly with sequence length, while one-token decode
  does not scan a token-growing KV cache.
- DeltaNet treats that matrix as fast-weight memory: it predicts the new value
  from the old state and writes the prediction error. GDN adds one forget gate
  per head; KDA uses a separate forget rate for every key channel.
- Current models are usually hybrid rather than purely linear. Kimi-Linear
  interleaves three KDA layers with one MLA layer; Qwen3.6 similarly interleaves
  three GDN layers with one gated full-attention layer. The exact Kimi K3 and
  Qwen3.7/3.8 layer configurations are not public in this snapshot.

## 2. From softmax attention to a recurrent memory

For one head, causal softmax attention is

$$
o_t=
\frac{\sum_{i\le t}\exp(q_t^\top k_i/\sqrt{d_k})v_i}
{\sum_{i\le t}\exp(q_t^\top k_i/\sqrt{d_k})}.
$$

The obstacle to reassociating the matrix products is the softmax: it is applied
after forming all query-key scores. If we first drop it, associativity changes
the non-causal computation from $(QK^\top)V$ to $Q(K^\top V)$, avoiding the
token-token matrix. For a normalized, attention-like generalization, replace
the exponential similarity with a factorized kernel

$$
\operatorname{sim}(q,k)=\phi(q)^\top\varphi(k),
$$

where early formulations choose nonnegative feature maps so the normalized
weighted average remains attention-like. Substituting this kernel and
reassociating the products gives

$$
S_t=S_{t-1}+\varphi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\varphi(k_t),
$$

$$
o_t=\frac{S_t^\top\phi(q_t)}{z_t^\top\phi(q_t)}.
$$

Thus the causal token-token matrix becomes two prefix states. Modern variants
often remove the explicit denominator and normalize the output instead. With
identity feature maps, the simplest recurrence is

$$
S_t=S_{t-1}+k_tv_t^\top,\qquad o_t=S_t^\top q_t.
$$

Here $S_t\in\mathbb{R}^{d_k\times d_v}$ is an associative memory. The full
history is compressed into $d_kd_v$ numbers per head instead of retaining every
key and value. Standard attention explicitly compares each query with all prior
keys, so prefill is $O(T^2d)$ and decode reads an $O(T)$ KV cache. The recurrent
form instead gives:

- prefill: $O(Td_kd_v)$ work;
- decode: $O(d_kd_v)$ work per token, independent of $T$;
- state: $H_vd_kd_v$ elements per request and layer, independent of $T$.

The weakness is interference: additive writes never remove stale mappings.

### 2.1 DeltaNet: write the prediction error

The state's ultimate read function is $q_t\mapsto o_t$ through
$o_t=S_t^\top q_t$. DeltaNet cannot locally supervise every future query, so it
uses the observed key-value pair $k_t\mapsto v_t$ as a self-supervised write
objective. If queries and keys share a meaningful similarity space, making the
state retrieve $v_t$ from $k_t$ helps a future query similar to $k_t$ retrieve
that value as part of its output. This is a learned assumption rather than a
mathematical guarantee.

Treating $S^\top$ as this online map, DeltaNet minimizes

$$
\mathcal L_t(S)=\tfrac12\lVert S^\top k_t-v_t\rVert_2^2.
$$

One gradient step with write strength $\beta_t\in[0,1]$ gives

$$
S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
$$

The classical **delta rule**, also called the Widrow--Hoff or LMS rule, updates
a linear map by the outer product of its input and prediction error. Here the
input is $k_t$, the target is $v_t$, and the rule becomes

$$
\hat v_t=S_{t-1}^\top k_t,\quad
e_t=v_t-\hat v_t,\quad
S_t=S_{t-1}+\beta_tk_te_t^\top.
$$

If the memory already predicts $v_t$, the update is small. This is the delta
rule's advantage over an unconditional outer-product write.

### 2.2 Gated DeltaNet: one decay per head

**Mathematical update.** GDN adds a learned scalar forget gate
$\alpha_t\in(0,1)$ to the DeltaNet update:

$$
S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
    +\beta_tk_tv_t^\top.
$$

Although this form hides the original Delta Rule, it remains a Delta Rule under
reparameterization. With $\gamma_t=\alpha_t$,
$\eta_t=\alpha_t\beta_t$, and $\tilde v_t=v_t/\alpha_t$,

$$
S_t=\gamma_tS_{t-1}
    +\eta_tk_t(\tilde v_t-S_{t-1}^\top k_t)^\top.
$$

This is one gradient step of size $\eta_t$ on the regularized objective

$$
\tfrac12\lVert S^\top k_t-\tilde v_t\rVert_2^2
+\frac{1-\gamma_t}{2\eta_t}\lVert S\rVert_F^2.
$$

The additional $L_2$ term shrinks the old state, which is exactly the
forgetting effect of $\gamma_t$.

Thus GDN is a forget gate plus a prediction-error write, rather than a different
memory-learning rule.

When $\alpha_t$ is near one, the old state is retained; when it is near zero,
the old state is mostly erased before the new association is written. Because
$\alpha_t$ is one scalar per head, all key channels forget at the same rate.
This is the linear-attention core used by Qwen3-Next/Qwen3.5/Qwen3.6.

### 2.3 KDA: one decay per key channel

KDA promotes the scalar to
$\alpha_t\in(0,1)^{d_k}$. At the recurrent-model level, this is a direct
fine-grained extension of GDN; the larger technical change is making the
channel-wise recurrence efficient during parallel prefill and training:

$$
\boxed{
S_t=(I-\beta_tk_tk_t^\top)
    \operatorname{Diag}(\alpha_t)S_{t-1}
    +\beta_tk_tv_t^\top
}
,\qquad
o_t=S_t^\top q_t/\sqrt{d_k}.
$$

Equivalently:

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1},
\quad e_t=v_t-\bar S_t^\top k_t,
\quad S_t=\bar S_t+\beta_tk_te_t^\top.
$$

This ordering is part of KDA's mathematical definition, not merely a kernel
implementation detail. For a channel-wise gate,
$\operatorname{Diag}(\alpha)kk^\top S$ is not generally the same as
$kk^\top\operatorname{Diag}(\alpha)S$. KDA first applies the diagonal decay to
the old state, then calculates and writes the prediction error.

**Gate parameterization.** The paper only says the gate
$\alpha_t=f(W_\alpha^\uparrow W_\alpha^\downarrow x_t)$ uses a GDN/Mamba-style
decay $f$. The released FLA and SGLang code both concretize it via the
log-decay $g_t=\log\alpha_t$:

$$
g_t=-\exp(A_{\log})\odot
\operatorname{softplus}(a_t+\mathrm{dt\_bias}).
$$

Here $\operatorname{softplus}(x)=\log(1+e^x)$ is a smooth ReLU: always
positive, so $g_t\le 0$ and $\alpha_t=e^{g_t}\le 1$ — but unbounded above, so
a large activation can push $\alpha_t$ arbitrarily close to $0$ and wipe the
state in one step.

The newer safe-gate path swaps softplus for a sigmoid
($\sigma(x)=1/(1+e^{-x})$, output in $(0,1)$) scaled by a fixed negative
constant $\ell$ (a lowercase "ell", typically $-5$):

$$
g_t=\ell\,
\sigma\!\left(\exp(A_{\log})\odot(a_t+\mathrm{dt\_bias})\right),
\qquad \ell<0.
$$

So $g_t\in(\ell,0)$ and $\alpha_t\in(e^{\ell},1)$, versus the canonical
gate's $g_t\in(-\infty,0)$ and $\alpha_t\in(0,1)$: the decay can never
underflow, which is what enables FlashKDA's more aggressive Tensor Core
formulation.

Note the two gates are different functions, not two implementations of the
same math: the gate is fixed at training time and the served model must use
whichever one its weights were trained against. As of this snapshot the safe
gate is **not used by any released model**. SGLang's kernels accept a
`lower_bound` argument, but the backend reads it via
`getattr(layer, "lower_bound", None)` and no model file sets it, so
Kimi-Linear-48B always runs the canonical softplus gate. The safe-gate path
is forward-looking plumbing (and a FlashKDA prerequisite) for a future model
such as Kimi K3, whose actual gate must be confirmed from its released
config/weights.

### 2.4 Chunking: matrix parallelism versus token recurrence

Following
[Linear Attention Fundamentals](https://haileyschoelkopf.github.io/blog/2024/linear-attn/),
causal linear attention admits two naive extreme algorithms. All math in this
section deliberately uses the simplest linear attention — no gate, no delta
rule — to keep the algorithmic trade-off clear; section 2.5 shows how KDA
modifies this skeleton.

**Parallel form.** Stack the tokens into $Q,K,V\in\mathbb{R}^{T\times d}$ and
compute

$$
O=(QK^\top\odot M)V,
$$

where $M$ is the lower-triangular causal mask.

- Pros: pure dense matmuls — Tensor Core friendly and parallel over the whole
  sequence, which is what training and prefill want.
- Cons: $O(T^2d)$ FLOPs and a materialized $T\times T$ matrix. The mask $M$
  is the reason: without it, $Q(K^\top V)$ costs only $O(Td_kd_v)$, but
  $\odot M$ sits between the two products and blocks that reassociation.

**Recurrent form.** Process one row $(q_t,k_t,v_t)$ at a time and carry only
the fixed-size state:

$$
S_t=S_{t-1}+k_tv_t^\top,\qquad o_t=S_t^\top q_t.
$$

- Pros: complexity — $O(Td_kd_v)$ total, linear in sequence length, and each
  step is $O(d_kd_v)$ independent of context.
- Cons: not parallel — the loop is strictly sequential over tokens, works on
  vector-sized operands, and cannot use Tensor Cores; it is also bound by
  re-reading and re-writing the state every step.

So it is the natural algorithm for decode, but slow for training and prefill.

**Chunkwise form.** Chunking is the mixture that interpolates between the two
extremes: split the rows into chunks of $C$ tokens
($Q_{[c]},K_{[c]},V_{[c]}\in\mathbb{R}^{C\times d}$). The computation
decomposes into three distinct pieces — and this decomposition *is* the
kernel structure:

1. **State recurrence** (across chunks — the only sequential piece):

$$
S_{[c]}=S_{[c-1]}+K_{[c]}^\top V_{[c]}
$$

   One step per chunk instead of per token: $T/C$ sequential steps, each a
   $(d_k\times C)(C\times d_v)$ matmul costing $O(Cd_kd_v)$, total
   $O(Td_kd_v)$. Same total FLOPs as the token recurrence, but each step is
   now Tensor Core-sized work and the chain is $C\times$ shorter.

2. **Inter-chunk output** (each token reads history from the carried state):

$$
O_{[c]}^{\text{inter}}=Q_{[c]}\,S_{[c-1]}
$$

   A $(C\times d_k)(d_k\times d_v)$ matmul per chunk, $O(Cd_kd_v)$ each,
   $O(Td_kd_v)$ total. Once the $S_{[c]}$ from step 1 are materialized, all
   chunks compute this in parallel.

3. **Intra-chunk output** (tokens attend to their own chunk, parallel form):

$$
O_{[c]}^{\text{intra}}=\big(Q_{[c]}K_{[c]}^\top\odot M\big)V_{[c]}
$$

   The quadratic part, but only within a chunk: $O(C^2d)$ each, $O(TCd)$
   total — the $T\times T$ mask has shrunk to $C\times C$. Fully parallel
   across chunks, and still Tensor Core matmuls: the mask is applied
   elementwise between the two dot products and does not block MMA tiles.
   This is the piece KDA changes (section 2.5); steps 1–2 carry over with
   only decay factors added.

The final output is $O_{[c]}=O_{[c]}^{\text{inter}}+O_{[c]}^{\text{intra}}$:
$O(Td_kd_v+TCd)$ total, still linear in $T$, with only $T/C$ sequential
steps. The chunk size $C$ is the tunable knob: larger $C$ exposes more
parallel matmul work but pays more quadratic intra-chunk work; smaller $C$
means more sequential state transfers and launch/IO overhead ($C=64$ is
typical). This chunkwise algorithm is the "flash" in the Flash Linear
Attention (FLA) library's name.

### 2.5 KDA chunking: the same skeleton, a harder intra-chunk step

Section 2.4's steps 1–2 still apply; what breaks is the *additive* state
update. Vanilla linear attention writes
$S_t=S_{t-1}+k_tv_t^\top$, so a whole chunk collapses to one matmul
$S_{[c]}=S_{[c-1]}+K_{[c]}^\top V_{[c]}$. KDA instead multiplies the old
state by a per-token transition before writing:

$$
S_t=P_tS_{t-1}+\beta_tk_tv_t^\top,
\qquad
P_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t).
$$

$P_t$ is exactly the forget-then-delta-rule map from section 2.3: first
decay every key channel by $\alpha_t$, then subtract the rank-1 prediction
error along $k_t$. Across a chunk of $C$ tokens the carried state is
transformed by the *product* $P_C\cdots P_1$, not by a sum of outer
products — so step 1 of section 2.4 no longer has a closed-form matmul.

**What changes in the algorithm.** Compress that product into a
diagonal-plus-low-rank / WY form before the chunk recurrence runs. The
pipeline is three stages, each its own kernel role:

1. **WY prep** — fully parallel over chunks. Build the $C\times C$
   lower-triangular key-interaction matrix
   $A=\operatorname{tril}(\beta\odot K_gK_g^\top)$, solve
   $(I+A)^{-1}$ by forward substitution, and form the factors $W,U$.
   Cost $O(C^2d+C^3)$ per chunk. This replaces vanilla step 3's simple
   $(QK^\top\odot M)V$. In practice this stage is often several launches
   (KKT / triangular solve / $W,U$ recompute), sometimes fused.
2. **State recurrence** — sequential over chunks, one kernel. Writes
   every $S_{[c]}$:
   $S_{[c]}=\operatorname{Diag}(\alpha_{[c]})S_{[c-1]}+K_g^\top(U-WS_{[c-1]})$.
   Same role as section 2.4 step 1. The grid is only $O(NH)$ (sequences
   $\times$ heads); chunks loop inside each program, so at small batch
   many SMs sit idle. This is the main SM-waste stage.
3. **Output** — fully parallel over chunks, a separate kernel. Reads the
   materialized $S_{[c]}$ and computes the inter-chunk plus intra-chunk
   terms (section 2.4 steps 2–3). Not fused with state recurrence: the
   sequential kernel must finish and spill every $S_{[c]}$ to HBM first.

Trade-off vs token recurrence: without WY prep you stay in the
token-level loop (no Tensor Cores, $T$ sequential steps); with it you
buy back chunkwise matmuls at the cost of an $O(C^2d+C^3)$ solve per
chunk, plus the underfilled sequential state-carry kernel. Larger batch
(e.g. $B=4$ vs $B=1$ in the prefill figure) fills more of that
$O(NH)$ grid and is why throughput jumps. The paper's KDA specialization
(binding both low-rank vectors to $k$) further cuts the WY prep relative
to a general DPLR recurrence, but does not change this three-stage
shape.

**Choosing $C$.** Same knob as section 2.4, now with an extra $O(C^3)$
triangular-solve term in the WY prep:

- larger $C$ → fewer sequential steps in the state-carry kernel ($T/C$),
  bigger MMA tiles, but more expensive WY prep and higher register/SRAM
  pressure;
- smaller $C$ → cheaper prep, more sequential state transfers and launch
  overhead.

$C=64$ is the usual default in practice.

## 3. Implementation comparison

<!-- TODO(yikai): guiding questions below — delete this paragraph once verified
     that sections 3.x / 4.x answer all of them, and let readers work through
     the material directly. -->
The interesting question is not which repo owns which file, but: **what are
the kernels, what idea does each implement, is that idea shared across
frameworks, can we bench them separately, and which one is the bottleneck?**
FLA, SGLang, vLLM, and TensorRT-LLM all implement the same KDA math; they
differ in fusion boundaries and serving wrappers. The benches in this
directory load leaf kernels only (no scheduler / CUDA graph / engine).

### 3.1 Shared attention-layer flow

A KDA layer (Kimi-Linear style; GDN layers are the same shape with a scalar
gate) is:

$$
x_t
\xrightarrow{\mathrm{proj}}
(q_t,k_t,v_t,a_t,b_t,z_t)
\xrightarrow{\mathrm{short\ causal\ conv+SiLU}}
(\tilde q_t,\tilde k_t,\tilde v_t),
$$

$$
a_t
\xrightarrow{A_{\log},\,\mathrm{dt\_bias}}
g_t=\log\alpha_t,
\qquad
b_t\xrightarrow{\sigma}\beta_t,
$$

$$
(\tilde q_t,\tilde k_t,\tilde v_t,\alpha_t,\beta_t,S_{t-1})
\xrightarrow{\mathrm{KDA\ core}}
(o_t,S_t),
$$

$$
y_t=W_o\!\left(\operatorname{RMSNorm}(o_t)\odot\sigma(z_t)\right).
$$

Same idea across frameworks. Differences are packaging: whether proj/gate/β
are fused into one GEMM, whether gate activation lives inside the KDA kernel
or outside, and how request → state-slot indices are passed. Serving extras
(prefix checkpoints, speculative replay, CUDA graphs) sit *above* this flow
and are out of scope for the kernel benches.

### 3.2 Stage inventory

All stages below are shared across frameworks (same math, different fusion
boundaries):

| Stage | Idea | Bound (typical) |
|---|---|---|
| short causal conv | depthwise FIR + SiLU on $q,k,v$ | memory |
| **KDA decode** | one-token $S_t\leftarrow P_tS_{t-1}+\ldots$, $o_t=S_t^\top q_t$ | launch + state traffic |
| **KDA prefill: WY prep** | $A$, $(I+A)^{-1}$, $W,U$ per chunk | compute / Tensor Core |
| **KDA prefill: state carry** | sequential $S_{[c]}$ over chunks | **latency / SM underfill** |
| **KDA prefill: output** | inter + intra from $S_{[c]},W,U$ | compute / Tensor Core |

The bold rows are the KDA core; the surrounding projections, conv, and gated
RMSNorm + $W_o$ are plumbing shared with GDN and ordinary hybrid layers.

### 3.3 Decode: one fused recurrence

Idea — for each request, one step of section 2.3:

$$
S\leftarrow\operatorname{Diag}(\alpha)\,S,\quad
e\leftarrow v-S^\top k,\quad
S\leftarrow S+\beta\,k\,e^\top,\quad
o\leftarrow S^\top q.
$$

State is $O(H\cdot d_v\cdot d_k)$ per request and **independent of context
length** — that is the whole point versus softmax KV cache.

Impl differences (same math):

- **preactivated** path: caller passes $\log\alpha$ and $\beta$; kernel only
  updates $S$ (FLA / vLLM recurrent).
- **packed / fused-gate** path: kernel takes packed $qkv$ plus raw logits,
  does L2-norm, softplus/sigmoid gate, and the update in one launch (SGLang
  packed; TensorRT-LLM channel-wise fused sigmoid gate).

Bound:

- low batch ($B\lesssim 32$): **launch-latency bound** — profile shows one
  fused kernel (~14–20 µs) at ~19% occupancy; further fusing pointwise ops
  does not help;
- high batch: **memory bound** on the fp32 $V{\times}K$ state
  (read–modify–write every step).

Bottleneck relative to the full attention module: the KDA core is only tens
of microseconds; fused proj+conv+norm dominate when measured as a module
(~100 µs SGLang-style vs ~300 µs unfused FLA-style at the same shape). So for
decode, optimize the surrounding GEMMs/fusion first; the recurrence is already
one launch.

### 3.4 Prefill: three kernel roles

Idea — section 2.5's three stages. End-to-end:

$$
(q,k,v,a,\beta)
\xrightarrow{\mathrm{gate+cumsum}}
g
\xrightarrow{\mathrm{WY\ prep}}
(W,U,A_{qk})
\xrightarrow{\mathrm{state\ carry}}
\{S_{[c]}\}
\xrightarrow{\mathrm{output}}
o.
$$

Impl differences (same math): fusion of gate into cumsum, whether $\beta$
sigmoid is inside or outside, how many launches WY prep splits into, and
whether state updates use indexed in-place pools. Cross-framework e2e
comparisons are in §4.7; they do **not** change the three-role structure.

Bound and bottleneck (stage-split numbers in §4.8):

1. **WY prep** — fully parallel, Tensor Core matmuls + triangular solve.
   Scales nearly linearly with total tokens → compute-bound when the grid
   is full.
2. **State carry** — sequential over $T/C$ chunks, grid only $O(NH)$.
   **Latency / SM-underfill bound** at small batch: many SMs idle while each
   program walks chunks. Latency barely moves from $B=1$ to $B=4$ at fixed
   $S$, while WY grows $\sim 4\times$ — that is why $B=4$ e2e throughput
   jumps.
3. **Output** — fully parallel Tensor Core; separate launch after $S_{[c]}$
   hits HBM. Cheapest of the three at the measured shapes.

Which stage wins depends on shape: at $B=1$, $S=8192$ state carry is
largest; at $B=4$, $S=8192$ WY prep is largest. Gate cumsum is negligible.

## 4. Experiments

### 4.1 Scope and B200 setup

- GPU: NVIDIA B200 (SM100), 183,359 MiB.
- Software: PyTorch 2.11.0+cu130, Triton 3.6.0,
  `sglang[diffusion]==0.5.15.post1`.
- Per-rank shape: $H_q=H_v=16$, $d_k=d_v=128$, bf16 inputs and fp32 state. This is one
  TP=2 shard of Kimi-Linear's released 32-global-head configuration; it is not
  presented as Kimi K3's unknown shape.
- Fixed recurrent state: 1 MiB per request per KDA layer.
- Timing: CUDA events, median of 7 repeats × 100 iterations after 20 warmups
  for the decode context sweep; 5 × 100 for the direct implementation sweep;
  3 × 20 after 5 warmups for prefill.
- Scope: core KDA only; projections, short conv, output norm/projection,
  periodic MLA, MoE, collectives, scheduler, and CUDA graph are excluded.

The exact one-step PyTorch recurrence and packed Triton decode agree:

| Quantity | Result |
|---|---:|
| output cosine | 0.9999966 |
| output max absolute error | $7.63\times10^{-6}$ |
| final-state cosine | 0.9999985 |
| final-state max absolute error | $1.08\times10^{-4}$ |

### 4.2 Decode: flat with context, scales with batch

The table condenses the four repeated context labels; ranges show min–max
latency from 4 K through 1 M.

| Batch | packed Triton | split Triton | packed throughput | recurrent state |
|---:|---:|---:|---:|---:|
| 1 | 16.35–19.56 µs | 18.74–19.72 µs | 51–61 K tok/s | 1 MiB |
| 8 | 17.93–18.04 µs | 20.25–22.98 µs | 443–446 K tok/s | 8 MiB |
| 32 | 16.41–19.80 µs | 17.89–18.02 µs | 1.62–1.95 M tok/s | 32 MiB |
| 128 | 63.93–64.50 µs | 78.33–78.81 µs | 1.98–2.00 M tok/s | 128 MiB |

Key observations:

1. **Context length has no algorithmic effect.** The labels allocate no history;
   variations across repeated labels are host/GPU timing noise, not history
   reads.
2. **Packing helps most at high batch.** It removes roughly 11–14 µs at batch
   128; at batch 8 the paths are close.
3. **Decode is state-bandwidth/parallelism work, not context work.** Per-layer
   state grows with batch and heads, not sequence length. A full model must
   multiply the 1 MiB proxy by the number of KDA layers and account for TP.
4. **Microseconds do not imply a 1 M-token model decodes this quickly
   end-to-end.** Periodic full-attention layers still read their KV cache, and
   K3's 2.8 T MoE/communication cost is outside this kernel.

### 4.3 Decode profile

The batch-32 PyTorch trace contains one
`fused_recurrent_kda_packed_decode_kernel` per step. Steady-state device
durations are about 14.4 µs under profiler instrumentation (best repeated
CUDA-event latency without profiler: 16.4 µs including launch sequencing).
The trace reports:

- one fused kernel, with no visible Q/K/V split or gate kernels;
- grid `[4, 512, 1]`, block `[32, 1, 1]`;
- 168 registers/thread;
- estimated achieved occupancy around 19%.

This is the decode optimization target: at low batch, launch latency dominates;
at higher batch, the large per-request matrix state and relatively low
occupancy matter. Kernel fusion is already present, so further work should
measure state traffic, register pressure, head/value tiling, and persistent
batch scheduling rather than fuse arbitrary extra pointwise operations.

### 4.4 Prefill: linear scaling

Safe-gate Triton chunk results:

| Batch | Sequence | Latency | Throughput | peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.124 ms | 4.14 M tok/s | 288.2 MiB |
| 1 | 2,048 | 0.224 ms | 9.15 M tok/s | 363.2 MiB |
| 1 | 8,192 | 0.851 ms | 9.63 M tok/s | 489.4 MiB |
| 4 | 512 | 0.184 ms | 11.11 M tok/s | 132.2 MiB |
| 4 | 2,048 | 0.562 ms | 14.58 M tok/s | 492.4 MiB |
| 4 | 8,192 | 2.174 ms | 15.08 M tok/s | 1,933 MiB |

Latency is approximately linear in total tokens. The peak allocator number is
process-local and affected by JIT/workspace reuse, so compare it only within
the same run.

Optional backend status in this environment:

- FlashKDA ([MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA))
  is installed and measured in §4.7; FLA auto-dispatches to it under
  `safe_gate=True` + `torch.inference_mode()`. SGLang's serving chunk path
  (`fla.kda.chunk_kda`) does not call FlashKDA, but its kernel-object
  `extend` path (`linear/kernels/kda_flashkda`) does run FlashKDA for
  safe-gate prefill with 64 ≤ sequence length ≤ 2048 now that the package is
  installed (measured in `bench_linear_attention.py --mode prefill`).
- FlashQLA ([QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)) is
  installed: Qwen's TileLang GDN chunk-prefill kernels (SM90/SM100), the GDN
  counterpart of FlashKDA, with an FLA-compatible `chunk_gated_delta_rule`
  API. Three JIT kernels (gate cumsum, KKᵀ solve = WY prep, fused chunk
  forward = state carry + output). Benchmarked as `flash_qla_gdn_chunk` in
  `bench_gdn_attention.py --mode prefill`. Like FlashKDA, FLA
  **auto-dispatches** to it when installed; the bench pins the FLA row to
  Triton with `FLA_FLASH_QLA=0` so the two rows stay distinct. Fastest GDN
  chunk prefill at every measured shape: up to ~2× over FLA Triton and ~2.3×
  over SGLang at batch 4, sequence 8,192 (`results/bench_gdn_prefill.csv`).
- CuTeDSL prefill does not support safe gate in the installed build.
- SGLang's CuTeDSL decode (`sglang_kda_cutedsl`) works after a clean
  `nvidia-cutlass-dsl==4.5.2` reinstall (the venv previously carried a
  half-upgraded install whose Python wrappers and compiled MLIR bindings
  disagreed, which surfaced as the "constructor mismatch"). Its compiled
  kernel expects a **bf16** `dt_bias`; the bench casts accordingly.
- The inspected SGLang checkout/package has no FlashInfer KDA decode module.

### 4.5 Direct implementation and attention-module comparison

The original packed-versus-split numbers compare two paths inside SGLang.
An additional run imports upstream FLA 0.5.2 directly and aligns the decode
contract: identical projected tensors, one token per request, raw gate/beta
logits, Q/K normalization in-kernel, fp32 V-first state, and in-place update.
Upstream FLA has no packed projection-output-to-recurrence API, so its row and
SGLang split form the recurrence-level comparison; SGLang packed is reported as
an additional serving-only fast path, not as a like-named FLA counterpart.

At the KDA-core boundary, SGLang's packed path wins from batch 8 upward; the
split path is slightly faster at batch 1:

| Batch | upstream FLA recurrent | SGLang split | SGLang packed |
|---:|---:|---:|---:|
| 1 | 22.91 µs | 18.27 µs | 19.61 µs |
| 8 | 23.01 µs | 20.19 µs | 18.00 µs |
| 32 | 23.40 µs | 19.67 µs | 18.64 µs |
| 128 | 76.29 µs | 74.47 µs | 63.18 µs |

The low-batch ordering is close enough to be sensitive to host launch latency
and GPU clocks. This later run had materially higher host launch overhead than
the dedicated sweep in §5.2, so its absolute microseconds should not be mixed
with that table; use it for same-run implementation ratios. The robust result
is at batch 128: SGLang packed is 1.21x
faster than current upstream FLA and 1.18x faster than SGLang split. Upstream
FLA's current recurrent kernel uses a 1-D grid with `BV=32`; SGLang's packed
kernel uses different value tiling and avoids separate tensor preparation.

The attention-module benchmark includes projection, short convolution, KDA,
gated RMSNorm, and output projection, while excluding decoder pre-norm and
residual, MoE/MLP, communication, and scheduler work:

| Batch | upstream FLA module | SGLang fused attention | speedup |
|---:|---:|---:|---:|
| 1 | 326.30 µs | 103.93 µs | 3.14x |
| 8 | 328.82 µs | 106.02 µs | 3.10x |
| 32 | 326.59 µs | 109.06 µs | 2.99x |
| 128 | 327.05 µs | 106.62 µs | 3.07x |

This is not merely a kernel result. At batch 32, the FLA trace has eight matrix
multiplications and three convolution launches per step; the SGLang-style path
has two ordinary matrix multiplications, one batched gate GEMM, one packed
convolution, and one packed KDA launch. FLA is optimized as a general
training/inference layer with separate projections; SGLang reorganizes the
same attention math for low-launch-count continuous-batching decode.

`sglang_fused_attention` is an attention-only TP=1 reproduction of SGLang's
dataflow, not a full `ModelRunner`: it deliberately excludes cache allocation,
scheduler metadata, CUDA graphs, collectives, MoE, and model weights. This
keeps the comparison at the boundary requested here while retaining the
operations that distinguish the attention modules.

Raw data: `results/bench_linear_attention_decode.csv`,
`results/bench_linear_attention_prefill.csv`,
`results/bench_linear_attention_impls.csv`, and
`results/bench_attention_layer.csv`. Traces: `results/kda_decode_trace.json`,
`results/fla_kda_attention_trace.json`, and
`results/sglang_kda_attention_trace.json`.

### 4.6 FLA versus SGLang versus vLLM versus TensorRT-LLM

`bench_gdn_attention.py`, `bench_kda_decode.py`, and `bench_kda_prefill.py` add matched
comparisons (measured before the GDN/KDA file split; the kernels and input
contracts are unchanged). Decode recurrent rows
receive identical bf16 Q/K/V, preactivated scalar log-decay, post-sigmoid beta,
Q/K normalization, and fp32 V-first state. The SGLang packed row is a separate
serving path that also fuses Q/K/V extraction and raw gate/beta activation.
Prefill packs each batch into `[1, total_tokens, ...]` with identical sequence
boundaries and state slots.

All four GDN implementations match the PyTorch recurrence: output cosine is
1.0 and final-state cosine is at least 0.99999988. For a two-sequence,
128-token packed prefill, all four chunk paths have output cosine at least
0.9999903 and state cosine at least 0.9999951. The three KDA decode outputs
agree with upstream FLA at cosine 1.0 for SGLang and 0.9999969 for vLLM.

Matched decode latency on B200:

| Batch | FLA GDN | SGLang GDN | vLLM GDN | TRT-LLM GDN | SGLang packed GDN |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.31 µs | 21.59 µs | 25.76 µs | 23.13 µs | 18.20 µs |
| 8 | 26.88 µs | 21.81 µs | 25.64 µs | 23.00 µs | 18.05 µs |
| 32 | 27.23 µs | 21.96 µs | 25.57 µs | 23.26 µs | 18.14 µs |
| 128 | 48.97 µs | 56.08 µs | 58.36 µs | 48.72 µs | 51.89 µs |

SGLang packed GDN is fastest at batches 1–32. Among the unfused recurrent
interfaces, SGLang is fastest there, followed by TensorRT-LLM, vLLM, and FLA.
At batch 128, TensorRT-LLM and FLA are fastest because their smaller value
tiles expose more programs as the GPU saturates.

Matched KDA decode (TRT-LLM rows from `feat/kda` at `3897ad1cc700`, raw data
`results/bench_kda_decode.csv`):

| Batch | FLA KDA recurrent | SGLang KDA packed | vLLM KDA recurrent | TRT-LLM KDA recurrent |
|---:|---:|---:|---:|---:|
| 1 | 23.64 µs | 19.29 µs | 19.08 µs | 25.17 µs |
| 8 | 23.13 µs | 19.25 µs | 30.31 µs | 25.54 µs |
| 32 | 23.14 µs | 19.32 µs | 29.82 µs | 25.98 µs |
| 128 | 66.23 µs | 54.88 µs | 66.10 µs | 78.55 µs |

vLLM and SGLang are effectively tied at batch 1. SGLang's packed KDA path is
fastest at batches 8–128. TRT-LLM's KDA decode kernel (adapted from SGLang's
split, non-packed kernel) tracks upstream FLA at batches 1–32 and is the
slowest row at batch 128; it matches upstream FLA's output bit-exactly in the
correctness check. Note the TRT-LLM row activates gate and beta inside the
kernel like SGLang's packed row, while the FLA and vLLM rows receive
preactivated log-decays.

Matched packed-varlen GDN prefill:

| Batch | Sequence | FLA | SGLang | vLLM | TensorRT-LLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.254 ms | 0.135 ms | 0.182 ms | 0.190 ms |
| 1 | 2,048 | 0.263 ms | 0.148 ms | 0.187 ms | 0.189 ms |
| 1 | 8,192 | 0.423 ms | 0.488 ms | 0.518 ms | 0.438 ms |
| 4 | 512 | 0.254 ms | 0.138 ms | 0.187 ms | 0.185 ms |
| 4 | 2,048 | 0.272 ms | 0.302 ms | 0.358 ms | 0.282 ms |
| 4 | 8,192 | 0.910 ms | 1.054 ms | 1.293 ms | 0.981 ms |

SGLang wins short prefills because its vendored pipeline has lower fixed
launch overhead. Current upstream FLA crosses over for larger total work: its
newer fused intra-chunk path combines KKT, triangular solve, and WY
reconstruction, while the other inspected Triton paths still launch these
stages separately. TensorRT-LLM is the closest to FLA for the largest cases;
vLLM is slowest there.

Against canonical SGLang KDA prefill measured in the same session, GDN is
within 5% at batch 1, sequence 512, then 1.46–1.69x faster at sequence
2,048–8,192 (and 1.27x faster for batch 4, sequence 512). Unlike decode, KDA
prefill pays substantially for its diagonal-plus-low-rank transition and
per-channel gate throughout the chunk algorithm. Raw data:
`results/bench_gdn_attention.csv` and
`results/bench_kda_canonical_prefill.csv`.

### 4.7 KDA prefill across FLA, SGLang, vLLM, TensorRT-LLM, and FlashKDA

Re-measured on B200 with [Moonshot FlashKDA](https://github.com/MoonshotAI/FlashKDA)
installed. Figure: `results/bench_kda_prefill.png` (one panel per batch
size, throughput vs sequence length, one color per backend; the FlashKDA
and `fla_kda_safe_triton` lines are safe-gate, the rest canonical-gate).

**Who calls FlashKDA in serving code:** TRT-LLM `feat/kda` yes (`auto` /
`flashkda`); vLLM `feat/k3` yes (`vllm._flashkda_C`); SGLang and vLLM
`b10-main` no (Triton / CuTeDSL only). FlashKDA requires the **safe gate**;
SGLang's row below is still **canonical** softplus — mark that when reading
the right panel.

Canonical-gate Triton (ms):

| Batch | Sequence | FLA | SGLang | vLLM | TRT-LLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.323 | 0.126 | 0.193 | 0.251 |
| 1 | 2,048 | 0.331 | 0.212 | 0.252 | 0.259 |
| 1 | 8,192 | 0.970 | 0.734 | 0.852 | 0.739 |
| 4 | 512 | 0.334 | 0.173 | 0.213 | 0.255 |
| 4 | 2,048 | 0.737 | 0.507 | 0.677 | 0.625 |
| 4 | 8,192 | 2.717 | 1.787 | 2.471 | 2.194 |

FlashKDA vs Triton (ms); SGLang column is canonical-gate (not safe):

| Batch | Sequence | FLA safe Triton | SGLang Triton* | **FlashKDA** | vs SGLang |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.337 | 0.126 | **0.077** | 1.64× |
| 1 | 2,048 | 0.350 | **0.212** | 0.225 | 0.94× |
| 1 | 8,192 | 0.834 | **0.734** | 0.830 | 0.88× |
| 4 | 512 | 0.345 | 0.173 | **0.087** | 1.98× |
| 4 | 2,048 | 0.598 | 0.507 | **0.268** | 1.89× |
| 4 | 8,192 | 2.137 | 1.787 | **0.976** | 1.83× |

Among canonical Triton paths, SGLang is fastest at every point. FlashKDA
beats SGLang at short prefills and at every $B=4$ point (~1.8–2×); at
$B=1$, $S\ge 2048$ SGLang Triton is slightly ahead. Matched safe-gate
FlashKDA vs FLA safe Triton is 1.0–4.4× (largest on short sequences).

Correctness against the exact fp32 recurrence for a two-sequence, 128-token
packed prefill: FLA, vLLM, and TRT-LLM reach output cosine 0.99999; SGLang's
vendored pipeline reaches output cosine 0.9855 and state cosine 0.99984 — the
same mathematics with visibly larger numerical error, consistent with
lower-precision intermediates in its older vendored FLA revision. Three
measurement caveats discovered here: vLLM's chunk pipeline writes its output
into `v` in place (`o=v`), so benchmark inputs must not be shared across
backends; TRT-LLM's autotuned chunk kernels mutate `v` and the state pool in
place, so the very first invocation — during which Triton autotuning re-runs
the kernel once per candidate config — returns corrupted output (cosine
0.9948) unless the autotune cache is warmed on throwaway inputs first; and
upstream FLA's `chunk_intra_token_parallel.py` needed a three-line local
patch (hoisting `BK` to a host-passed constexpr) because Triton 3.6 rejects
`triton.next_power_of_2` referenced inside a jitted kernel. Raw data:
`results/bench_kda_prefill.csv`.

### 4.8 Prefill stage split (where the time goes)

`bench_kda_prefill.py --mode stages` times SGLang's four leaf stages on
the same packed inputs. Latency (ms):

| B | S | gate | WY prep | state carry | output |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.017 | 0.045 | 0.024 | 0.015 |
| 1 | 2,048 | 0.018 | 0.090 | 0.067 | 0.020 |
| 1 | 8,192 | 0.037 | 0.261 | **0.329** | 0.077 |
| 4 | 512 | 0.018 | 0.088 | 0.031 | 0.020 |
| 4 | 2,048 | 0.037 | 0.263 | 0.103 | 0.077 |
| 4 | 8,192 | 0.123 | **0.921** | 0.369 | 0.282 |

Read: at single-sequence long prefill the sequential state carry dominates;
once the $O(NH)$ grid is fuller ($B=4$), WY prep takes over because it does
real Tensor Core work proportional to tokens while state carry barely grows
(0.329 → 0.369 ms from $B=1$ to $B=4$ at $S=8192$). Figure:
`results/bench_kda_stages.png`.

Together with §4.2–4.3 (decode flat in context, launch-bound at low $B$) and
§4.5 (module ≫ bare KDA core):

- **Decode bottleneck** — surrounding proj/conv/norm fusion, not the
  recurrence kernel.
- **Prefill bottleneck** — state carry when underfilling SMs; WY prep when
  the batch fills the GPU.
- **Same idea across frameworks** — yes; e2e gaps in §4.6–4.7 are fusion and
  launch overhead, not different mathematics.

## 5. Concrete path to Kimi K3 support

### Work possible before the release

1. Keep the KDA math oracle and synthetic packed/split benchmarks green.
2. Validate canonical and safe-gate prefill separately; never silently compare
   a fallback under the requested backend's name.
3. Preserve V-first `[slot, H_v, d_v, d_k]` serving state across prefill/decode.
4. Prepare tests for batch sizes 1/8/32/128, ragged prefill, CUDA graph,
   state-slot remapping, prefix checkpoint restore, and speculative rollback.
5. Profile packed decode at Kimi-Linear proxy shapes and retain raw traces.

### Blocked until K3 artifacts arrive

1. Implement the K3 config/model and weight mapping: KDA layer IDs, Gated MLA,
   AttnRes, Stable LatentMoE, vision path, and MXFP4/MXFP8 quantization.
2. Replace proxy dimensions with released per-rank KDA shapes.
3. Select canonical versus safe gate from the config/weights—not the blog.
4. Implement/validate K3's announced prefill-cache policy.
5. Run end-to-end accuracy and throughput against Moonshot reference outputs.

The highest-risk integration issue is not the recurrence kernel. It is making
recurrent-state checkpoints participate correctly in prefix caching,
continuous batching, preemption, speculative rollback, and disaggregated
prefill/decode while the periodic Gated-MLA layers maintain normal paged KV.

## References

- Kimi K3 official announcement: https://www.kimi.com/blog/kimi-k3
- Kimi Linear report: https://arxiv.org/abs/2510.26692
- Kimi-Linear repository: https://github.com/MoonshotAI/Kimi-Linear
- FLA KDA implementation: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- DeltaNet Explained, Part I (model and derivation): https://sustcsonglin.github.io/blog/2024/deltanet-1/
- DeltaNet Explained, Part II (parallel algorithm): https://sustcsonglin.github.io/blog/2024/deltanet-2/
- DeltaNet Explained, Part III (modern architecture): https://sustcsonglin.github.io/blog/2024/deltanet-3/
- Linear Attention Fundamentals (parallel, recurrent, and chunkwise forms): https://haileyschoelkopf.github.io/blog/2024/linear-attn/
- 线性注意力简史：从模仿、创新到反哺: https://kexue.fm/archives/11033
- “对角+低秩”三角阵的高效求逆方法: https://kexue.fm/archives/11072
- 为什么线性注意力要加 Short Conv？: https://kexue.fm/archives/11320
- Key 归一化助力长度外推: https://kexue.fm/archives/9859
- Qwen3.5 GDN 原理与代码分析: https://zhuanlan.zhihu.com/p/2007937984738129405
- FlashQLA：面向 GDN 的融合线性注意力算子库: https://zhuanlan.zhihu.com/p/2032898276207350901
- Linear Attention and Beyond, interactive tutorial with Songlin Yang: https://www.youtube.com/watch?v=d0HJvGSWw8A
- Linear Attention: Kimi Delta Attention, kernel-oriented walkthrough: https://jianyuh.github.io/attention/2025/12/13/KDA.html
- Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels: https://arxiv.org/abs/2503.14376
- Qwen3-Next announcement: https://qwenlm.github.io/blog/qwen3-next/
- Qwen3.5 Transformers documentation: https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
- SGLang: https://github.com/sgl-project/sglang
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM

