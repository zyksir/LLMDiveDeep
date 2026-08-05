# KDA backend notes: SGLang Triton decode paths & FlashKDA prefill

Temporary working notes. Hardware: B200 (~148 SMs). Shapes throughout are the
bench defaults ($H = H_V = 16$ heads, $d_k = d_v = 128$, fp32 state), i.e. the
Kimi-Linear per-rank KDA layer. All numbers are kernel-only (no projections,
norm, scheduler).

---

## 0. Notation

KDA is a gated DeltaNet-style layer: per head it keeps a fast-weight state
$S \in \mathbb{R}^{d_k \times d_v}$, and the kernels consume the *raw*
projection outputs plus two learned per-head parameters. Per token, per head:

- $q, k \in \mathbb{R}^{d_k}$, $v \in \mathbb{R}^{d_v}$ — query / key / value
  projections; $q, k$ are L2-normalized before use ($\tilde q, \tilde k$).
- $a \in \mathbb{R}^{d_k}$ — pre-activation **gate logits**, the
  data-dependent input to the per-channel decay.
- $b \in \mathbb{R}$ — **beta logit**; $\beta = \sigma(b) \in (0,1)$ is the
  delta-rule write strength.
- $A_{\log} \in \mathbb{R}$ — learned per-head decay **rate constant**,
  stored in log space (so $e^{A_{\log}} > 0$ without constraints).
- $b_{dt} \in \mathbb{R}^{d_k}$ — learned per-head bias added to the gate
  logits before the softplus (Mamba-style "dt bias").
- Together these form the per-channel **log-decay**
  $g = -e^{A_{\log}}\,\mathrm{softplus}(a + b_{dt}) \in \mathbb{R}^{d_k}$,
  $g \le 0$, applied to the state as $\mathrm{diag}(e^{g})$ with
  $e^{g} \in (0,1]$.

As tensors, in the SGLang calling convention ($T$ = total tokens across the
batch, $N$ = number of sequences, $H$ = heads). This is the chunk-prefill
contract of §4; the decode kernels of §1 take the same quantities but with
the raw logits $a, b$ — the activations are fused into the kernel:

| symbol | tensor | shape | dtype |
|---|---|---|---|
| $q, k$ | `q`, `k` | $[1, T, H, d_k]$ | bf16 |
| $v$ | `v` | $[1, T, H, d_v]$ | bf16 |
| $a$ | `raw_gate` (pre-activation gate logits) | $[1, T, H, d_k]$ | bf16 |
| $\beta$ | `beta` (post-sigmoid write strength) | $[1, T, H]$ | bf16 |
| $A_{\log}$ | `A_log` (per-head rate constant, log-stored) | $[H]$ | fp32 |
| $b_{dt}$ | `dt_bias` | $[H \cdot d_k]$ | fp32 |
| $S_{\mathrm{prev}}$ | `initial_state` (indexed pool) | $[N, H, d_v, d_k]$ | fp32 |
| — | `cu_seqlens` (sequence boundaries) | $[N+1]$ | int32 |

---

## 1. Decode

### 1.1 The math

One decode step per head, state $S \in \mathbb{R}^{d_k \times d_v}$. The
inputs are the raw projection outputs — packed pre-conv $q_0, k_0, v_0$,
gate logits $a$, beta logit $b$. First the Mamba-style **short causal
conv**: per channel, a width-4 convolution over the last 4 tokens (the 3
previous ones live in a per-request conv state, shifted in place each
step) followed by SiLU. $g$ and $\beta$ are *not* convolved:

$$
(q, k, v) = \mathrm{SiLU}\big(\mathrm{conv4}(q_0, k_0, v_0)\big),
$$

then the recurrence step itself:

$$
\tilde q = \frac{q}{\lVert q \rVert_2}, \qquad
\tilde k = \frac{k}{\lVert k \rVert_2}, \qquad
g = -e^{A_{\log}}\,\mathrm{softplus}(a + b_{dt}) \in \mathbb{R}^{d_k}, \qquad
\beta = \sigma(b),
$$

$$
S \leftarrow \mathrm{diag}(e^{g})\,S
  + \tilde k\,\big(\beta\,(v - S^{\top}\tilde k)\big)^{\top},
\qquad
o = S^{\top} \tilde q ,
$$

and finally the layer's sigmoid-gated output RMSNorm
($o \leftarrow \mathrm{RMSNorm}(o) \odot \sigma(z)$, $z$ = the projected
output gate). The conv and the output norm are part of the decode step's
math here — not "surrounding layer ops" — because the fastest kernels
(SGLang's and TRT-LLM's fused decodes below) execute all of it in one
launch.

### 1.2 Implementation examples

Every stack implements exactly the §1.1 math; they differ in **how much of
the step is fused into one launch** — decode at these sizes is
launch/latency-bound, so the kernel count per layer per step is the story.
The backtick names are the row names used by `bench_kda_decode.py`.

**Everything in one kernel.** SGLang (kimi-k3 branch,
`sglang_kda_fused_decode`) and TRT-LLM (`trtllm_kda_fused_decode`, the
same design in Triton) do the whole of §1.1 in one launch, replacing the
chain `causal_conv1d_update → decode → rms_norm_gated`. Both exist only
for the K3 decode regime ($H = 12$, $d = 128$, $T = 1$, no spec decode —
§3) and fall back to the unfused chain otherwise.

**Everything else** fuses less. These are SGLang kernels (the TRT-LLM
rows are forks of them):

- $T = 1$ packed decode (`sglang_kda_packed`, `trtllm_kda_packed`) — the
  serving fast path, best-performing of these: gate/beta activation, L2
  norms, and the recurrence in one kernel, slicing q/k/v straight out of
  the packed projection layout. Conv and gated output norm stay separate
  (three launches per step).
- Varlen $T > 1$ split decode (`sglang_kda_split`, `trtllm_kda_split`) —
  the general fallback, incl. speculative verify, fed by pre-split q/k/v.
  Same fusion level: the gated output norm is not fused here either.
- FLA upstream / vLLM (`fla_kda_recurrent`, `vllm_kda_recurrent`; vLLM
  vendors FLA as-is) — least efficient, **four** launches per step: conv,
  gate activation, delta-rule recurrence, gated norm.

---

## 2. Chunked algorithm: the closed-form transition

Speculative verify, SGLang chunk prefill, and FlashKDA use the same chunk
transition. Take $C$ normalized queries and keys
$Q,K\in\mathbb{R}^{C\times d_k}$, values
$V\in\mathbb{R}^{C\times d_v}$, write strengths $\beta$, log-decays $g$,
and entering state $S_0\in\mathbb{R}^{d_k\times d_v}$.

First form the chunk-local cumulative decay
$G_t=\sum_{i\le t}g_i$ and four decay-adjusted views:

$$
\begin{aligned}
K_d &= K\odot e^G, &
K_i &= K\odot e^{-G},\\
K_g &= K\odot e^{G_C-G}, &
Q_d &= Q\odot e^G\cdot\mathrm{scale}.
\end{aligned}
$$

Then the complete chunk transition is

$$
\begin{aligned}
L &=
\operatorname{tril}_{-1}
\left(\operatorname{diag}(\beta)K_dK_i^\top\right),\\
M &= (I+L)^{-1},\\
V_{\mathrm{new}}
&=M\,\operatorname{diag}(\beta)(V-K_dS_0),\\
S_{\mathrm{next}}
&:=\boxed{\operatorname{diag}(e^{G_C})S_0
{}+K_g^\top V_{\mathrm{new}}
},\\
A_{qk}&=\operatorname{tril}(Q_dK_i^\top),\\
O&=Q_dS_0+A_{qk}V_{\mathrm{new}}.
\end{aligned}
$$

Below, “formula (1)” means the decay views, “(2)” the $L/M$ triangular
solve, “(3)” $V_{\mathrm{new}}$ and the state transition, and “(4)” the
output equation.

What each line means:

- $L$ records how each token overlaps earlier tokens in the same chunk.
  $\operatorname{tril}_{-1}(X)$ means `torch.tril(X, diagonal=-1)`: retain
  entries strictly below the main diagonal and set the diagonal and
  everything above it to zero. By contrast, $\operatorname{tril}(X)$ keeps
  the main diagonal as well.
- $M$ resolves those causal, within-chunk dependencies at once.
- $V-K_dS_0$ subtracts what the entering state already predicts;
  $V_{\mathrm{new}}$ is the solved update contributed by this chunk.
- The state transition first decays $S_0$ to the chunk end, then folds all
  solved updates into it with the GEMM $K_g^\top V_{\mathrm{new}}$.
- The output has two terms: $Q_dS_0$ reads history before the chunk, while
  $A_{qk}V_{\mathrm{new}}$ adds causal contributions from this chunk.

This is the reusable closed form
$S_{\mathrm{next}}=f(S_0,K_g,M,K_d,V,\beta,G_C)$.

<details>
<summary>Optional algebra: relation to WY/UT</summary>

Define
$W=M\operatorname{diag}(\beta)K_d$ and
$U=M\operatorname{diag}(\beta)V$. Then
$V_{\mathrm{new}}=U-WS_0$. Because $L$ is strictly lower triangular, it is
nilpotent, so the triangular inverse is exact after finitely many terms.
This is the compact WY/UT representation. A related derivation is given in
[DeltaNet Explained (Part II)](https://sustcsonglin.github.io/blog/2024/deltanet-2/).

</details>

---

## 3. Speculative decoding (MTP): the state-rollback design space

### 3.1 Why speculative decoding needs rollback

This section considers a **linear draft chain** (`topk = 1`) only, not a
draft tree. The target verifies
$T=1+\gamma$ tokens—the golden token and $\gamma$ drafts—in one forward,
but sampling reveals the accepted prefix length only afterwards.

That uncertainty is the problem for KDA. Attention can reject drafts by
rewinding a KV-cache pointer. KDA instead compresses all history into one
destructively updated state $S$. During verify we therefore do not yet know
whether each draft token should update the persistent SSM.

There are two rollback strategies:

1. **Save SSM:** run the recurrence over all $T$ tokens and save the state
   after every token. Once `accept_len` is known, select that snapshot and
   discard the others.
2. **Replay SSM:** keep the checkpoint $S_0$ unchanged during verify and
   save a compact recurrence record for every token. Once `accept_len` is
   known, replay only the accepted prefix to obtain the committed state.

The first spends memory to avoid recomputation; the second spends a small
amount of recomputation to avoid writing $T$ full $K\times V$ states.

### 3.2 Implementations and fast replay

The save-SSM implementation is algorithmically trivial: run the $T$-token
recurrence as a loop, keep $S$ resident, and write one snapshot after each
step. There is no recurrence-level shortcut because every candidate state is
an observable rollback point. Optimization is therefore limited to ordinary
kernel engineering—state tiling, fusion, and snapshot-store efficiency.

Replay-SSM is more interesting, and the shipping implementations place the
replay at different points on the timeline:
- **SGLang uses two kernels.** The verify kernel keeps the checkpoint
  read-only and stores compact per-token replay inputs. After sampling, a
  second kernel replays the accepted prefix and writes the new checkpoint.
  It is sometimes summarized as “save k, not S,” but the KDA fold needs more
  than the key alone: the implementation records the value, key/decay data,
  and beta needed to reproduce the recurrence.
- **TensorRT-LLM uses one verify/replay kernel.** It stores already-solved
  update records `(u, normalized k, G)` in a ring. The checkpoint may lag
  behind the accepted sequence; each verify reads the ring as logical
  history, and physically updates the checkpoint only when the ring would
  overflow. Rejected records are ignored by shortening the accepted prefix.

TensorRT-LLM's organization is algorithmically preferable: it removes the
post-sampling launch and amortizes the expensive full-state write. The
measured difference is currently modest because its kernel has additional
precision, masking, and occupancy overhead. Setting the ring capacity to
$T$ makes it fold every step, so the same implementation can also emulate
an always-current checkpoint.

**Two lengths, two outputs.** Suppose the replay ring contains $P$ accepted
prefix tokens and the current verify window contains $T$ new draft tokens.
The layer needs two different results:

1. one committed state $S_P$ after the accepted prefix; and
2. outputs $O_D$ for all $T$ draft tokens, starting from that logical state.

The two groups do **not** share one $V_{\mathrm{new}}$: accepted history and
current drafts contain different keys, values, and residuals. They reuse the
same formula from §2, but produce
$V_{\mathrm{new}}^{(P)}$ and $V_{\mathrm{new}}^{(T)}$.
The already-solved $V_{\mathrm{new}}^{(P)}$ can be reused both to form
$S_P$ and to supply the history cross terms needed by $O_D$; it cannot
replace $V_{\mathrm{new}}^{(T)}$ for the new drafts.

For TensorRT-LLM's solved-update ring,
$V_{\mathrm{new}}^{(P)}$ was already computed when those accepted tokens
were drafts. Therefore replay needs no $P\times P$ inverse:

$$
S_P=\operatorname{diag}(e^{G_P})S_0
{}+K_{g,P}^{\top}V_{\mathrm{new}}^{(P)}.
$$

The current drafts then need only their own $T\times T$ triangular matrix
$M_T$ to compute $V_{\mathrm{new}}^{(T)}$ and $O_D$. There are two equivalent
implementations:

- materialize $S_P$ with the replay GEMM, then run the $T$-token chunk from
  $S_P$; or
- keep $S_P$ implicit and use $T\times P$ cross terms from the solved history
  records when computing the draft residuals and outputs.

TensorRT-LLM uses the second form. It still solves only the current
$T\times T$ system. It does **not** need a $P\times P$ inverse or a combined
$(P+T)\times(P+T)$ inverse.

If the ring stores raw inputs instead of solved updates, the accepted prefix
must first be processed by either a $P$-step loop or a $P\times P$ chunk
solve. The drafts still use a separate $T$-step loop or $T\times T$ solve.
A combined $(P+T)\times(P+T)$ solve is mathematically valid, but it performs
unnecessary work and couples replay to draft verification.

**TODO — choose the loop/chunk dispatch.** At $T=1$, use ordinary decode.
For very small $P$ or $T$, a direct loop may beat the setup cost of a chunk
solve. Benchmark fixed $T=4$ across `pnat` and $B\in\{1,4,8\}$, then repeat
with a fully fused chunk kernel and include $T=1,2$ before fixing a runtime
threshold.

---

## 4. SGLang chunk prefill: the whole algorithm, then the kernels

### 4.1 Stage contract — input → output of the whole thing

Everything below is per head; SGLang runs $H = 16$ heads independently.
$T$ = total tokens (e.g. 32768 at $B=4$, $S=8192$), $d_k = d_v = 128$.
Entry point: `chunk_kda` (SGLang's vendored `fla/kda.py`).

**Inputs:** the tensors of the notation table (§0) — `q`/`k`/`v`,
`raw_gate`, post-sigmoid `beta`, `A_log`, `dt_bias`, the indexed
`initial_state` pool, and `cu_seqlens`.

**Outputs:** $O$ `[1, T, H, d_v]` bf16, and the final state written back
into the pool $[N, H, d_v, d_k]$ in place.

### 4.2 The kernels SGLang launches

The complete shared mathematics and notation are in §2. SGLang chooses
$C=64$ and materializes $G$, $M$, $W$, $U$, $K_g$,
$V_{\mathrm{new}}$, and the chunk-entry states between kernels. The discussion
below maps those named objects onto launches; it does not define a second
version of the math.

SGLang cuts (1)–(4) into **8 kernel launches** (7 distinct kernels; l2norm
runs twice), orchestrated in `chunk_kda_fwd`. On small
grids ($B \cdot NT \cdot H \le 256$) kernels 4–6 fuse
into one launch, giving 6. All shapes below are written in full. $NT = T/64$
= number of chunks. Times: $B=1$, $S=8192$, per iteration.

The $64\times64$ triangular system is treated as a $4\times4$ grid of
$16\times16$ blocks. `Akkd` stores the four diagonal blocks in fp32;
`Akk` stores the completed $M$ in chunk-stacked layout.

Kernel 4 computes the four causal diagonal blocks. Kernel 5 first inverts
those diagonal blocks, then constructs every off-diagonal block of the full
inverse with block back-substitution GEMMs. Thus only four small blocks
require an explicit inverse; the rest of the $64\times64$ inverse is
tensor-core matrix multiplication.

| # | kernel | formula | key implementation detail | µs |
|---|---|---|---|---:|
| 1,2 | `l2norm_fwd_kernel` ×2 | normalize $q,k$ | one program per token/head | 24.4 |
| 3 | `kda_gate_chunk_cumsum_vector_kernel` | (1): $G$ | chunk cumsum; stores $G$ in fp32 | 34.5 |
| 4 | `chunk_kda_fwd_kernel_intra_token_parallel` | (2): diagonal blocks | four causal $16\times16$ blocks in fp32 | 98.0 |
| 5 | `chunk_kda_fwd_kernel_inter_solve_fused` | (2): solve → $M$ | diagonal inverses, then off-diagonal GEMMs | 96.2 |
| 6 | `recompute_w_u_fwd_kernel` | (2): $W,U,K_g$ | two $[64,64][64,128]$ GEMMs | 57.5 |
| 7 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | (3): state | serial over chunks; writes chunk-entry states | 321.4 |
| 8 | `chunk_gla_fwd_kernel_o` | (4): output | chunk-parallel history and local GEMMs | 78.3 |

Notes an engineer should care about:

- **Every arrow between kernels is an HBM round trip.** The intermediates
  materialized are: $G$ (fp32, input-sized), `Akkd`/`Akk`/$A_{qk}$
  ($C$-wide per token), $w, u, k_g$ (3 input-sized tensors), $v_{\mathrm{new}}$
  (input-sized), and $h$ — per-chunk state snapshots,
  $NT \cdot H \cdot 128 \cdot 128$ bf16 ≈ 64 MiB at $S = 8192$. `Akk` in
  particular exists *only* to ferry $M$ from kernel 5 to kernel 6 (the fused
  small-grid path never writes it).
- **Only kernel 7 is serial** (grid $(d_v/64,\, N \cdot H)$ = 32 CTAs at
  $B=1$; walks $NT$ chunks in order). Kernels 1–6 and 8 are
  token/chunk-parallel with thousands of CTAs. That's why kernel 7 is 45% of
  the pipeline at $B=1$ and barely grows with batch.
- The stage names used elsewhere in this doc group them as: gate cumsum
  (3), WY prep (4–6), state carry (7), output (8).
  `bench_kda_prefill.py --mode stages` times these groups on matched
  inputs (`run_stages` in `bench_kda_prefill.py`).

---

## 5. FlashKDA prefill: what the two kernels do

FlashKDA uses two kernels and `CHUNK = 16`. The fastest measured variant,
`int21`, uses the same high-level decomposition with a lower-level PTX
implementation:

1. a **prepare kernel** computes normalization, decay, the small triangular
   inverse, and output coefficients for every 16-token tile in parallel;
2. a **recurrence kernel** walks those tiles, carrying the state and
   producing outputs.

Chunk size 16 is useful for two reasons: the safe-gate decay over one tile
stays representable in bf16 without rescaling, and the $16\times16$
triangular inverse is small enough for a short Neumann-series MMA sequence.

### K1 — `_flash_kda_fwd_prepare` (token-parallel)

Grid $(\mathrm{total\ 16\text{-}token\ tiles},\; H)$, 256 threads, `__launch_bounds__(256, 8)`.
One thread block fully processes one 16-token tile of
one head — all tiles independent, so it saturates the GPU at any sequence
length. Inside (`fwd_kernel1.cuh`):

Using the formulas from §2 (with $C = 16$):

1. q/k **L2 normalization** — SGLang's kernels 1–2.
2. **Fused gate activation + intra-tile cumsum** = formula (1)'s $G$
   — safe gate $g = \mathrm{lower\_bound} \cdot \sigma(x)$ with
   the sigmoid built on `tanh.approx.f32`, and the exponent rebased to 2 so
   each decay is one `ex2.approx.ftz.f32`.
3. **Decay application** = the rest of formula (1): builds
   $Q_d$, $K_d$, $K_i$, $K_g$ in shared memory (smem buffers `q_decayed`,
   `k_decayed`, `k_inv`, `k_restored`; $K_g = K \odot e^{G_C - G}$;
   `g_total` holds $G_C$). Of these, $K_i$ is chunk-local: it feeds only the
   two MMAs below and never leaves K1 (`k_inv` is an smem union slot, not a
   workspace array).
4. **Tile MMAs** = formula (2)'s $L$ and formula (4)'s $A_{qk}$:
   $L = K_d K_i^{\top}$ and $M_{qk} = Q_d K_i^{\top}$ —
   $16 \times 16$ outputs via single-warp SM80 MMA.
5. **$\beta$ scaling + $I + L$**, then formula (2)'s
   inverse $M = (I+L)^{-1}$ by **Neumann series in fp16** (fp16
   works because the entries of $M$ are bounded in $[-1, 1]$, and fp16 MMA
   skips the fp32→bf16 cast).

K1 writes six compact per-tile arrays to a **workspace** — `k_decayed`,
`q_decayed`, `k_restored` ($[16, 128]$ bf16), `g_total` ($[128]$ fp32), `INV`,
`Mqk` ($[16, 16]$ bf16) — roughly the
size of the inputs themselves, so the round-trip is cheap.

So K1 covers formulas (1) + (2) **up to the inverse $M$**, i.e. SGLang's
kernels 1–5 — but it stops before $W$/$U$: those are never materialized;
their effect is folded into K2. This is why $K_d$ is in the workspace even
though $L$ is already consumed: K2 computes
$V_{\mathrm{new}} = M\,\mathrm{diag}(\beta)(V - K_d S_0)$, which is formula
(3)'s $U - W S_0$ with the factors kept unassembled
($W = M\,\mathrm{diag}(\beta)K_d$, $U = M\,\mathrm{diag}(\beta)V$). Storing
$\{M, K_d\}$ instead of $\{W, U\}$ swaps a $[16,128]$ tile ($U$) for a
$[16,16]$ one ($M$) and spares K1 from ever reading $V$. $K_i$, by contrast,
is only needed for $L$ and $M_{qk}$, both finished inside K1 — it never
touches HBM.

### K2 — `_flash_kda_fwd_recurrence` (head-parallel, serial over tiles)

Grid $(N_{\mathrm{seqs}}, H)$, 192 threads = 2 TMA producer warps + 4 MMA consumer
warps, warp-specialized with a 3-stage `PipelineTmaAsync`.
One block owns one (sequence, head) pair, keeps the $128 \times 128$ state in
shared memory **in bf16 for the whole sequence** (fp32 FMA updates, bf16
storage — halves smem and skips casts on the GEMM critical path), and walks
the 16-token tiles serially. Per tile, six phases
(`fwd_kernel2.cuh`):

1. dual GEMM $K_d S_0$ and $Q_d S_0$ — the history halves of formulas (3)
   and (4);
2. cast/stage, load $V$, $M$ (`INV`), $\beta$;
3. $V_{\mathrm{new}} = M\,\mathrm{diag}(\beta)(V - K_d S_0)$ — formula (3)'s
   $V_{\mathrm{new}}$, computed in one fused expression. Algebraically equal
   to $U - W S_0$, but $W$/$U$ never exist: the solve happens right here,
   next to the state;
4. $O \mathrel{+}= M_{qk} V_{\mathrm{new}}$ — formula (4)'s intra-chunk
   term, with $V_{\mathrm{new}}$ transposed **inside the register file** via
   `MOVM_T`, no smem round-trip between phases;
5. TMA store of the output tile (overlapped with the next tile's loads);
6. formula (3)'s state update
   $S \leftarrow \mathrm{diag}(e^{G_C})\,S + K_g^{\top} V_{\mathrm{new}}$.

So K2 covers formulas (3) + (4) — SGLang's kernels 6–8 — fused into the
serial walk, which is exactly what lets $W$, $U$, $v_{\mathrm{new}}$, and
the $h$ snapshots skip HBM entirely.

Why two kernels and not one: an earlier fused prototype tied K1's abundant
token-parallelism to K2's scarce head-parallelism and left SMs idle; splitting
gave ≥15% end-to-end (deep-dive doc, §2).

Note the **gate difference**: FlashKDA implements the *safe* gate
($g = \mathrm{lower\_bound} \cdot \sigma(x)$), not the canonical softplus
gate ($g = -e^{A_{\log}}\,\mathrm{softplus}(x)$). The matched apples-
to-apples Triton baseline in our bench is `fla_kda_safe_triton`, and outputs
are compared against the same reference in `bench_kda_prefill.py`.

---

## Appendix A — the chunk-form verify: derivation, fp64 proof, and a Triton kernel

This appendix gives the detailed derivation and proof behind §2's chunk form.
The recurrence over a $T$-token window is *exactly* a handful of GEMMs plus
one $T \times T$ triangular solve, with no approximation anywhere. A.1–A.4
derive it, A.5 proves it in fp64 to machine epsilon, A.6 walks through the
Triton implementation, A.7 measures it against TRT-LLM / b10 / SOL, and
A.8 is the Chinese version (中文版见 A.8).

### A.1 Setup and claim

Per token, the verify window runs the §1.1 recurrence (safe gate, inputs
already conv'd and L2-normalized):

$$
S_t = \Lambda_t\, S_{t-1} + \tilde k_t\, u_t^{\top}, \qquad
u_t = \beta_t \bigl(v_t - \tilde k_t^{\top} \Lambda_t S_{t-1}\bigr), \qquad
o_t = \tilde q_t^{\top} S_t,
$$

with $\Lambda_t = \mathrm{diag}(e^{g_t})$ a per-$K$-channel decay. Way 1
executes this literally: $T$ dependent full passes over the $K \times V$
state. The claim is that with the cumulative decay
$G_t = \sum_{s \le t} g_s$, $\lambda_t = e^{G_t}$ (a $K$-vector per
token), the whole window collapses to:

1. one $[T, K] \times [K, V]$ GEMM against the checkpoint ($W$, and its
   $\tilde q$ twin for outputs),
2. one $T \times T$ unit-lower-triangular solve (the only sequential part,
   and even that becomes $T{-}1$ tiny GEMMs via a Neumann series),
3. one $[K, T] \times [T, V]$ GEMM to commit the state.

Nothing here is new mathematics — it is §2's WY/UT chunk form
specialized to chunk size $T$ — but writing it out at $T \le 8$ shows how
little of it is actually serial.

### A.2 Lemma 1 — the state unrolls; replay is one GEMM

Divide the recurrence by $\lambda_t$ (elementwise over the $K$ axis;
$\lambda$ broadcasts over $V$). With $\hat S_t = S_t \oslash \lambda_t$:

$$
\hat S_t = \hat S_{t-1} + (\tilde k_t \oslash \lambda_t)\, u_t^{\top}
\quad\Longrightarrow\quad
S_t = \lambda_t \odot \Bigl(S_0 + \sum_{s \le t} (\tilde k_s \oslash \lambda_s)\, u_s^{\top}\Bigr).
$$

The telescoping product of diagonal decays becomes a *ratio of cumulative
gates* — that is the entire trick. Two consequences:

- **The final state is one GEMM**:
  $S_T = \lambda_T \odot S_0 + \sum_s \bigl(\tilde k_s \odot (\lambda_T \oslash \lambda_s)\bigr) u_s^{\top}$,
  a $[K, T] \times [T, V]$ contraction with pre-scaled key columns.
- **Ring replay is the same GEMM.** Replay records already store *solved*
  $(u_s, \tilde k_s, G_s)$ (cumulative decays continue across launches),
  so folding pnat of them into the logical state — the solved-history case
  in §3.2 —
  is this identity applied across launches:
  $S_{\mathrm{logical}} = e^{G_p} \odot S_0 + \sum_s (e^{G_p - G_s} \odot \tilde k_s)\, u_s^{\top}$.
  No loop, no solve: the records are history, their $u_s$ are settled.
  The factored decay $e^{G_p - G_s}$ needs $|G| \lesssim 30$ to stay in
  fp32 range — true for the bounded safe gate
  ($g \ge \mathrm{lower\_bound} = -2$, $T \le 8$, $\mathrm{HIST} = 16$),
  and the reason A.6 assumes it.

### A.3 Lemma 2 — the cross-token dependency is a $T \times T$ triangular system

The only thing coupling token $t$ to earlier tokens is the $w_t = \tilde
k_t^{\top} \Lambda_t S_{t-1}$ term inside $u_t$. Substitute Lemma 1 for
$S_{t-1}$ and note $\Lambda_t \lambda_{t-1} = \lambda_t$:

$$
w_t = (\tilde k_t \odot \lambda_t)^{\top} S_0
      + \sum_{s < t} A[t, s]\, u_s, \qquad
A[t, s] = (\tilde k_t \odot \lambda_t) \cdot (\tilde k_s \oslash \lambda_s).
$$

The first term is a row of $W = (\tilde K \odot \lambda)\, S_0$ — the big
GEMM against the checkpoint. Moving the second term to the left-hand side
of $u_t = \beta_t (v_t - w_t)$ gives, over the whole window,

$$
\bigl(I + N\bigr)\, U = \mathrm{diag}(\beta)\,(V - W), \qquad
N = \mathrm{tril}_{-1}\bigl(\mathrm{diag}(\beta)\, A\bigr),
$$

a unit-lower-triangular system in $T$ unknowns of size $V$ each. *This is
the entire sequential content of verify.* Everything $K \times V$-sized —
$W$, the outputs, the state commit — is a tensor-core GEMM around it.

### A.4 Outputs, and the exact Neumann inverse

Outputs follow from Lemma 1 the same way ($o_t = \tilde q_t^{\top} S_t$,
sum now over $s \le t$):

$$
O = (\tilde Q \odot \lambda)\, S_0 + \mathrm{tril}_0(B)\, U, \qquad
B[t, s] = (\tilde q_t \odot \lambda_t) \cdot (\tilde k_s \oslash \lambda_s).
$$

And the solve needs no forward substitution: $N$ is *strictly* lower
triangular, hence nilpotent ($N^T = 0$), so the Neumann series terminates
— it is an identity, not an approximation:

$$
(I + N)^{-1} = \sum_{j=0}^{T-1} (-N)^j.
$$

$T{-}1$ multiplications of a $T \times T$ matrix, no division, no
data-dependent loop, MMA-friendly (FlashKDA's K1 uses the same trick for
its intra-tile inverse, §5). Precompute
$\tilde M = (I+N)^{-1}\mathrm{diag}(\beta)$ once per (b, h) and both
"solves" downstream are plain GEMMs: $U = \tilde M (V - W)$ and
$\mathrm{tril}_0(B)\, U = \bigl(\mathrm{tril}_0(B)\tilde M\bigr)(V - W)$.

### A.5 The fp64 proof — `tmp/proof_chunk_form.py`

The script runs the plain for-loop recurrence in fp64, then checks each
identity above against it. In fp64 the only residue left is machine
epsilon, i.e. the chunk form is a re-association of the same arithmetic,
not an approximation:

```python
# ---- sequential reference (way 1): the for loop --------------------------
S = S0.clone()
for t in range(T):
    S = S * g[..., t, :, None].exp()                          # diag decay
    w_t = torch.einsum("bhk,bhkv->bhv", k[..., t, :], S)      # k̃ᵀS
    u_t = beta[..., t, None] * (v[..., t, :] - w_t)           # solved update
    S = S + k[..., t, :, None] * u_t[..., None, :]            # rank-1
    o_seq[..., t, :] = torch.einsum("bhk,bhkv->bhv", q[..., t, :], S)
    u_seq[..., t, :] = u_t

# ---- chunk quantities ----------------------------------------------------
G = g.cumsum(dim=-2)          # cumulative gate G_t          [B,H,T,K]
lam = G.exp()                 # λ_t = e^{G_t}
kl, ki, ql = k * lam, k / lam, q * lam    # k̃⊙λ, k̃⊘λ, q̃⊙λ

# Lemma 1: S_T = λ_T ⊙ (S_0 + Σ_s (k̃_s ⊘ λ_s) u_sᵀ)
S_unroll = lam[..., -1, :, None] * (
    S0 + torch.einsum("bhsk,bhsv->bhkv", ki, u_seq))

# Lemma 2: (I + N) U = diag(β)(V − W),  N = tril₋₁(diag(β) A)
W = torch.einsum("bhtk,bhkv->bhtv", kl, S0)
A = torch.einsum("bhtk,bhsk->bhts", kl, ki)
N = torch.tril(beta[..., None] * A, diagonal=-1)
rhs = beta[..., None] * (v - W)
resid = u_seq + torch.einsum("bhts,bhsv->bhtv", N, u_seq) - rhs

# Exact Neumann inverse (N strictly lower ⇒ N^T = 0)
M, P = eye.clone(), eye.clone()
for _ in range(T - 1):
    P = -N @ P
    M = M + P

# ---- the full chunk pipeline: no loop over T anywhere --------------------
U = torch.einsum("bhts,bhsv->bhtv", M, rhs)
Bq = torch.tril(torch.einsum("bhtk,bhsk->bhts", ql, ki))
O = torch.einsum("bhtk,bhkv->bhtv", ql, S0) \
    + torch.einsum("bhts,bhsv->bhtv", Bq, U)
Sf = lam[..., -1, :, None] * S0 \
    + torch.einsum("bhsk,bhsv->bhkv", k * (lam[..., -1:, :] / lam), U)
```

Reading guide: the reference loop *saves its own $u_t$* (`u_seq`), so
Lemma 1 and Lemma 2 are each checked in isolation before the pipeline is
assembled — Lemma 1 plugs the reference updates into the unrolled state,
Lemma 2 checks the reference updates satisfy the triangular system
(residual form, no solve involved). Only the last block is the actual
chunk pipeline: solve $U$ with the Neumann $M$, then two GEMMs for $O$
and $S_T$. Output (B=2, H=3, T=6, K=V=32, fp64):

```text
Lemma 1 (unrolled final state): 2.22e-16
Lemma 2 (triangular system residual): 3.33e-16
Neumann inverse ((I+N)M - I): 1.73e-18
chunk O  vs sequential: 6.66e-16
chunk U  vs sequential: 3.33e-16
chunk S  vs sequential: 2.78e-16
```

In bf16 the two forms differ in summation order, so kernels are *not
bit-identical* to the for-loop — the same trade `replay_ssm_fused`
already makes (§3.2); the e2e bench checks both against the fp32 oracle
with the same tolerances and both pass.

### A.6 The Triton kernel — `kda/kda_chunk_verify_triton.py`

Same contract as TRT's `cached_replay` (the `trt_closure` row): checkpoint
+ solved-update ring in, outputs + ring appends out, $S$ written only on
ring overflow. The internals are A.2–A.4. Three launches, one program per
(batch, head) each:

- **prep kernel** — everything *V-independent and serial-ish*, on
  $[16, K]$ tiles: gate activation + `tl.cumsum`, L2 norms, the decayed
  ring keys $e^{G_p - G_s} \odot \tilde k_s$ (Lemma 1), the Neumann solve
  operator $\tilde M$ *stacked with its output twin*
  $C = \mathrm{tril}_0(B)\tilde M$ (A.4), the ring correction
  $RC = [\tilde q\lambda;\, \tilde k\lambda] \cdot K_{\mathrm{dec}}$, and
  the $(\tilde k, G)$ ring appends. Writes three small bf16 scratch
  tensors (`stack`, `rc`, `mc`); never touches $S$ or the $u$ ring.
- **state kernel** — the streaming pass. The mainloop is *only* the
  checkpoint GEMM: per $K$-block, one stacked tensor-core dot
  $[O_S; W] \mathrel{+}= \mathrm{stack}_{kb} \cdot S_{0,kb}^{\top}$,
  software-pipelined over blocks (`num_stages`). The epilogue folds in
  the ring replay ($[O_S; W] \mathrel{+}= RC \cdot U_{\mathrm{ring}}$ —
  Lemma 1 across launches, ONE dot) and applies both solve operators in
  ONE stacked dot: $[U;\, C(V - W)] = [\tilde M; C]\,(V - W)$. Stores $o$
  and appends $U$ to the ring.
- **fold kernel** — the rare ring-overflow commit of
  $S_{\mathrm{logical}}$ (Lemma 1 over the history rows). Kept as its own
  launch so its registers never tax the hot state kernel; no-ops in the
  steady regime.

Two representation choices carry the perf: (i) $T$ is padded to 16 rows
so every operator tile is `tl.dot`-shaped, with padded rows zeroed so
they contribute nothing (`k̃ = β = v = q̃ = 0`); (ii) the four logical
$[T, \cdot]$ operands are *stacked pairwise into $[32, \cdot]$ tiles* so
the mainloop and the solve each cost one MMA instead of two.

Why three launches and not one — the design history (probes in
`tmp/probe_chunk_*.py`, B=64 H=96, B200): a monolithic kernel measures
~530 us — every V-tile repeats the serial prologue, ~200 live registers
kill occupancy (ncu: 57% DRAM at 18% occupancy); accumulating all six
contractions per $K$-block in one loop spills catastrophically (~1.5 ms);
even just computing the ring correction inside the state kernel costs
+120 us by breaking the mainloop's pipeline. The split lands at ~95 us
prep + ~94 us state, the state pass near its ~78 us streaming floor.

### A.7 Measured: vs TRT, vs b10, vs SOL

`bench_kda_spec_verify_e2e.py`, replay-scheme figure (T=4, pnat=8,
CUDA-graph timed, B200, peak DRAM 7.67 TB/s). The chunk row is
`triton_chunk_replay_chain`: TRT's production three-launch chain with the
verify T-loop swapped for this kernel — same conv, same gated norm, so
the delta is purely way 1 → way 2.

| H | B | TRT chain (way 1) | Triton chunk (way 2) | b10 fused (way 1) | SOL |
|---|---|---|---|---|---|
| 96 | 1 | 14.2 us | 17.3 us | 7.9 us | 1.2 us |
| 96 | 4 | 30.9 us | 23.4 us | 16.0 us | 4.5 us |
| 96 | 16 | 114.6 us | 75.9 us | 56.0 us | 18.0 us |
| 96 | 64 | 419.1 us | **234.4 us** | 197.8 us | 71.8 us |
| 12 | 64 | 58.5 us | **37.5 us** | 29.5 us | 9.0 us |

Three readings:

- **Way 2 beats way 1 at equal engineering.** Against TRT — the shipping
  chain with the same launch structure — the chunk kernel wins 1.3–1.8x
  wherever there is enough batch to fill the machine (B ≥ 4 at H=96),
  and the gap *widens* with batch: exactly the chunking argument in §3.2, since
  way 1's sequential state passes ride the critical path of every CTA
  while way 2's GEMMs just add waves. At B=1 the fourth launch and the
  prep pass dominate and TRT still wins — way 2 pays a fixed overhead
  for removing a serial one.
- **Closer to SOL, not at it.** At H=96 B=64 the replay row moves from
  17% of SOL (TRT) to 31% (chunk); the remaining 3.3x is not the
  recurrence anymore: the state pass alone runs at ~76% of its own
  streaming floor, and the other ~140 us are the prep pass (~95 us of
  serial-tile work whose *bytes* are worth ~35 us) plus conv + norm +
  launch overheads.
- **The fused for-loop still wins the row (36% of SOL)** because b10
  pays ONE launch and streams the state ONCE with everything fused
  around it. The two results compose rather than compete: way-2
  internals (this appendix) inside a single fused kernel (b10's
  structure) is the obvious next kernel — the prep tile-work overlaps
  with the state stream instead of preceding it, and the ~95 us prep
  pass disappears from the critical path.

### A.8 中文版 — chunk 形式 verify 的推导与实现

**问题。** verify 窗口对每个请求要按 §1.1 的递推连续跑 $T = 1+\gamma$ 个
token（replay 方案还要先重放 pnat 条历史记录）。way 1 直接在 kernel 里
for 循环：每一步都是对 $K \times V$ 状态的一次完整依赖遍历（对角衰减、
$\tilde k^{\top} S$ 归约、rank-1 更新），全部是标量 fp32 —— 串行关键路
径，测得每步 ~5.2 us，其中只有 ~0.8 us 是字节成本（H=96, B=64），卡在
带宽 SOL 的 ~36%，且加大 batch 不会收敛（每个 CTA 都背着同样的依赖链）。

**核心恒等式（引理 1）。** 令累积门控 $G_t = \sum_{s \le t} g_s$、
$\lambda_t = e^{G_t}$（每 token 一个 $K$ 维向量）。把递推两边除以
$\lambda_t$，对角衰减的连乘就变成了累积门控的比值，状态完全展开：

$$
S_t = \lambda_t \odot \Bigl(S_0 + \sum_{s \le t} (\tilde k_s \oslash \lambda_s)\, u_s^{\top}\Bigr).
$$

由此：(a) 最终状态是一个 $[K,T] \times [T,V]$ 的 GEMM；(b) **replay 完
全不需要循环** —— ring 里存的本来就是已解出的 $(u_s, \tilde k_s, G_s)$，
把 pnat 条记录折进逻辑状态就是同一个恒等式跨 launch 使用：
$S_{\mathrm{logical}} = e^{G_p} \odot S_0 + \sum_s (e^{G_p - G_s} \odot \tilde k_s)\, u_s^{\top}$，
一个 GEMM。分解出的衰减因子 $e^{G_p - G_s}$ 要求 $|G| \lesssim 30$ 才不
会超出 fp32 范围 —— 有界的 safe gate（lower_bound = −2、$T \le 8$、
HIST = 16）满足这一点。

**跨 token 依赖只是一个 $T \times T$ 三角方程组（引理 2）。** 把引理 1
代入 $u_t$ 里的 $w_t = \tilde k_t^{\top} \Lambda_t S_{t-1}$ 项：

$$
(I + N)\, U = \mathrm{diag}(\beta)\,(V - W), \qquad
N = \mathrm{tril}_{-1}\bigl(\mathrm{diag}(\beta) A\bigr), \quad
A[t,s] = (\tilde k_t \odot \lambda_t) \cdot (\tilde k_s \oslash \lambda_s),
$$

其中 $W = (\tilde K \odot \lambda) S_0$ 是对 checkpoint 的大 GEMM。
**verify 的全部串行内容就是这个 $T \times T$ 单位下三角方程组**；所有
$K \times V$ 规模的运算（$W$、输出、状态提交）都是 tensor-core GEMM。
而且这个"解"也不用前代消元：$N$ 严格下三角故幂零（$N^T = 0$），Neumann
级数 $(I+N)^{-1} = \sum_{j<T} (-N)^j$ 在 $T{-}1$ 项后**精确**截断 ——
是恒等式而非近似，只需 $T{-}1$ 次 $T \times T$ 小矩阵乘，无除法、无数据
依赖分支。输出同理：$O = (\tilde Q \odot \lambda) S_0 + \mathrm{tril}_0(B)\, U$。

**精确性。** `tmp/proof_chunk_form.py` 在 fp64 下把上面每条恒等式与
for 循环参考逐一对拍，最大误差全部在机器精度（~1e-16）—— chunk 形式只是
同一算术的重结合，不是近似。bf16 下两种形式求和顺序不同、不逐位相同，
与 `replay_ssm_fused` 已经接受的取舍相同（§3.2），e2e 基准里两者对
fp32 oracle 用同一容差均通过。

**Triton 实现**（`kda/kda_chunk_verify_triton.py`，接口与 TRT 的
`cached_replay` 完全一致）分三个 launch：**prep kernel** 做所有与 $V$
无关的"串行小块"工作（门控激活 + cumsum、L2 归一化、衰减后的 ring key、
Neumann 算子 $\tilde M$ 及其输出孪生 $C = \mathrm{tril}_0(B)\tilde M$、
$(\tilde k, G)$ 入环），写三个小 bf16 scratch；**state kernel** 的主循
环只做 checkpoint 流式 GEMM（每个 $K$ 块一次堆叠的 tensor-core dot，
`num_stages` 软件流水），尾声用一次 dot 折入 ring replay、再用一次堆叠
dot 同时完成解方程和输出投影；**fold kernel** 只在 ring 溢出时提交
$S_{\mathrm{logical}}$，独立成 launch 以免占用热核寄存器。两个关键表示
选择：$T$ 补零到 16 行使所有算子都是 `tl.dot` 形状；四个 $[T,\cdot]$
操作数两两堆叠成 $[32,\cdot]$，主循环与解算各只花一条 MMA。（单体 kernel
反例：~530 us，寄存器 ~200 个、占用率 18%；六个收缩量全放主循环里累加
会灾难性溢出到 ~1.5 ms —— 见 A.6 的设计史。）

**结果**（A.7 表）：对 TRT 链（同 launch 结构的 way 1）在 B ≥ 4 时快
1.3–1.8 倍，H=96、B=64 从 419 us 降到 234 us，SOL 占比 17% → 31%，且
batch 越大差距越大 —— 正是 §3.2 中 chunking 分析的预测。剩余差距已不在递推本身：state
kernel 已到自身流式下限的 ~76%，其余是 prep pass（~95 us 的小块串行工
作，按字节只值 ~35 us）与 conv/norm/launch 开销。单 launch 融合的
for-loop kernel（b10，36% SOL）目前仍领先 —— 两个结论是互补的：把本附
录的 way-2 内核结构塞进 b10 那样的单 launch 融合 kernel，prep 的工作与
状态流重叠而非串联，就是下一个该写的 kernel。

---

Shape note: the measured tables in this document use H=16 heads (the
original working shape). The benches now sweep Kimi K3's shard shapes by
default — H=12 (one TP8 rank of 96 global heads) and H=96 (TP1/PP rank) —
via `--heads`; pass `--heads 16` to reproduce these tables. Relative
rankings hold at both shard shapes, and the gaps widen at H=96 (e.g.
prefill at B=2 S=2048: FlashKDA 442 us / PTX 390 us vs 1.3-1.8 ms for every
vendored FLA Triton pipeline; verify at B=64 T=4: `verify_replay_ssm_fused`
395 us vs 281 us for the verify half of `verify_store`).

Sources checked 2026-07-28: SGLang main `85618cc` and PR #32541 head
`4051b19` (fetched from GitHub; both newer than the local checkout), local
TRT-LLM and vLLM checkouts. Reproduction commands live in `README.md` and the
bench files' docstrings (`bench_kda_decode.py`, `bench_kda_prefill.py`,
`bench_kda_spec_verify.py`).
