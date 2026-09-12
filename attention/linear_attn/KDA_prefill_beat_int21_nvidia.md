# Kimi Delta Attention (KDA) prefill on B200 — a kernel-collaboration brief for the NVIDIA kernel team

> Historical study / retained source context. Start with the [kernel tutorial](../README.md)
> and [current reading guide](kda/KERNELS.md). Old CSV/log/PNG and local_results artifacts
> were removed for regeneration; measurements below were not rerun or revalidated.
> Future outputs belong under attention/results/, not the paths in old examples.
> Layer/projection benchmarks below are outside the new attention-core comparisons.


**Purpose.** We (Baseten) build inference kernels for the Kimi-K3 / Kimi-Linear
**KDA** linear-attention layer. Our CuTeDSL prefill kernel already beats the
CUTLASS reference (`flash-kda`) at every batch size, but it still trails the
fastest known implementation — **INT21's `flashkda-ptx`** (hand-written PTX,
`Int21-AI/KDA-B200`) — at batch $\ge 4$. This document (a) states the KDA math
end to end, (b) surveys the existing implementations and what makes each fast,
(c) reports exactly where our kernel loses to INT21, and (d) asks for your help
closing the last gap. It is self-contained: everything needed is below.

**Target hardware / stack.** NVIDIA **B200 (sm_100a)**, CuTeDSL 4.5.2
(CUTLASS Python DSL). Reference points are CUTLASS/CuTe C++ (`flash-kda`) and
raw PTX (INT21). Problem sizes: heads $H=16$, head dims $d_k=d_v=128$,
sequence lengths up to $T=65536$ per sequence, batch $B\in\{2,4,8,16,64\}$,
within a total-token envelope of $T_{\text{total}}\le 262144$.

**On the token bound — please treat it as a usable assumption, not a
constraint to work around.** The $T_{\text{total}}\le 262144$ envelope (and the
per-sequence cap) is a deliberate, sensible serving-side limitation: it is
exactly the operating envelope of the INT21 target we are matching, and it
reflects how the layer is actually driven (chunked/paged prefill never hands a
single kernel launch more than this). **The kernel is free to bake it in** — e.g.
32-bit (or narrower) token/offset indexing, statically-sized workspaces and
pipeline depths, no need for a 64-bit-offset or unbounded-$T$ slow path. Assume
it wherever it simplifies the schedule or saves registers/instructions.

---

## 1. What KDA is

KDA is a **gated DeltaNet-style linear-attention** layer. Per head it keeps a
fast-weight **state** $S\in\mathbb{R}^{d_k\times d_v}$ and updates it with a
delta rule that has a per-channel forget gate. Unlike softmax attention there is
no $T\times T$ score matrix over the whole sequence — history is compressed into
$S$, and a token attends *directly* only to the tokens inside its own chunk.

One KDA layer, per token, does the following (all per head unless noted):

1. **Projections** of the residual stream $x$: a merged `qkv` projection, a
   scalar **beta logit** $b$, a low-rank **gate logit** $a\in\mathbb{R}^{d_k}$
   (`f_a→f_b`), and a low-rank **output-gate** $z\in\mathbb{R}^{d_v}$
   (`g_a→g_b`).
2. **Short causal convolution** (depthwise conv1d, width $W_c$, typically 4)
   over the token axis of the merged `qkv`, followed by **SiLU**; then split
   into $q,k,v$. (§3)
3. **Per-channel gate + L2 norm**: $\tilde q,\tilde k$ are L2-normalized;
   $g=-e^{A_{\log}}\,\mathrm{softplus}(a+b_{dt})$; $\beta=\sigma(b)$. (§4)
4. **Delta-rule recurrence** — the core; decode form in §5, chunked prefill
   form in §6.
5. **Gated output RMSNorm**: $o = \mathrm{RMSNorm}(r)\odot w\odot\sigma(z)$. (§7)

The **prefill kernel** this brief is about covers steps 3–4 for a whole
sequence (the conv and the gated RMSNorm are separate launches today, but we
also fuse the RMSNorm — see §7). Steps are consumed from the *raw* projection
outputs; the activations ($g$, $\beta$, the L2 norms) are fused into the kernel.

### Notation and tensor shapes (SGLang calling convention)

$T$ = total tokens across the batch, $N$ = number of sequences, $H$ = heads,
$d_k=d_v=128$.

| symbol | tensor | shape | dtype |
|---|---|---|---|
| $q,k$ | `q`,`k` | $[1,T,H,d_k]$ | bf16 |
| $v$ | `v` | $[1,T,H,d_v]$ | bf16 |
| $a$ | `raw_gate` (pre-activation gate logits) | $[1,T,H,d_k]$ | bf16 |
| $\beta$ | `beta` (post-sigmoid write strength) | $[1,T,H]$ | bf16 |
| $A_{\log}$ | per-head decay rate constant (log-stored) | $[H]$ | fp32 |
| $b_{dt}$ | `dt_bias` | $[H\cdot d_k]$ | fp32 |
| $S_{\text{prev}}$ | `initial_state` (indexed pool) | $[N,H,d_v,d_k]$ | fp32 |
| — | `cu_seqlens` (sequence boundaries) | $[N+1]$ | int32 |

Outputs: $O\ [1,T,H,d_v]$ bf16, and the final state written back to the pool.

---

## 2. The delta-rule recurrence (decode / ground truth)

Ground truth is the per-token recurrence. With $\tilde q_t,\tilde k_t$ the
L2-normalized query/key, $g_t$ the per-channel log-decay ($g_t\le 0$), and
$\beta_t=\sigma(b_t)$:

$$
S_t = \mathrm{diag}(e^{g_t})\,S_{t-1}
      + \tilde k_t\,\big(\beta_t\,(v_t - S_{t-1}^{\top}\tilde k_t)\big)^{\top},
\qquad
o_t = S_t^{\top}\,\tilde q_t .
$$

The bracketed term is the **delta correction**: $S_{t-1}^{\top}\tilde k_t$ is
what the current state already predicts for key $\tilde k_t$; the write is
$\beta_t$ times the residual $v_t-\hat v_t$. This is what makes it a *delta*
rule rather than plain additive linear attention, and it is what couples tokens
within a chunk (below).

---

## 3. The short causal convolution

Before the delta rule, KDA applies a **depthwise causal 1-D convolution** of
width $W_c$ (Kimi-Linear config `short_conv_kernel_size`, typically 4) along the
token axis, independently per channel of the merged `qkv`, then a **SiLU**
activation (`config.hidden_act = "silu"`; only SiLU is supported). For channel
$c$ and token $t$:

$$
\hat x_{t,c} = \mathrm{SiLU}\!\Big(b_c + \sum_{j=0}^{W_c-1} w_{c,j}\,x_{t-W_c+1+j,\,c}\Big),
\qquad \mathrm{SiLU}(u)=u\,\sigma(u),
$$

with left-padding (causal: a token sees only itself and $W_c-1$ predecessors).
$\hat x$ is then split into the $q,k,v$ that feed §2/§4. In prefill this is a
short causal conv over the full sequence; in decode it is a rolling window of
the last $W_c$ tokens kept in a small state. In our stack the conv is a separate
launch (or fused into the decode kernel); the prefill kernel receives the
already-convolved $q,k,v$. We mention it here because a fully fused KDA prefill
would absorb it, and because it changes the read pattern at the sequence edges.

---

## 4. Gating and normalization (fused into the kernel)

$$
\tilde q=\frac{q}{\lVert q\rVert_2},\qquad
\tilde k=\frac{k}{\lVert k\rVert_2},\qquad
g=-e^{A_{\log}}\,\mathrm{softplus}(a+b_{dt})\in\mathbb{R}^{d_k},\qquad
\beta=\sigma(b).
$$

The decay applied to the state is $\mathrm{diag}(e^{g})$ with $e^{g}\in(0,1]$.

**Gate variants (important for numerics).** The production model uses the
**softplus gate** above. The fastest existing kernels (CUTLASS FlashKDA, INT21)
implement the **safe gate** $g=\text{lower\_bound}\cdot\sigma(x)$ with
`lower_bound = -5`; this bounds the per-token decay so that decays over a small
chunk stay inside bf16 range without any rescaling trick. Our benchmark compares
against a matched `fla_kda_safe_triton` reference so the gate is apples-to-apples.
Either gate is just an elementwise activation; the recurrence is identical.

---

## 5. Chunked prefill algorithm (the whole computation)

Running §2 token by token is $T$ serial steps. The chunked ("chunk-parallel")
algorithm cuts the sequence into chunks of size $C$ and computes, per chunk,
**exactly** the same result from the state at the chunk boundary. Take one chunk:
$Q,K\in\mathbb{R}^{C\times d_k}$, $V\in\mathbb{R}^{C\times d_v}$,
$\beta\in\mathbb{R}^{C}$, entering state $S_0$. Everything is one head.

**(1) Decay.** With chunk-local cumsum $G_t=\sum_{i\le t} g_i$:

$$
K_d = K\odot e^{G},\quad
K_i = K\odot e^{-G},\quad
Q_d = (Q\odot e^{G})\cdot\text{scale},\quad
K_g = K\odot e^{G_C-G}.
$$

All pairwise decays $e^{G_i-G_j}$ become sums of log-decays, hence the cumsum.

**(2) WY solve (the intra-chunk delta coupling).**

$$
L=\mathrm{tril}_{\text{strict}}\!\big(\mathrm{diag}(\beta)\,K_d K_i^{\top}\big),
\quad
M=(I-L)^{-1},
\quad
W=M\,\mathrm{diag}(\beta)\,K_d,
\quad
U=M\,\mathrm{diag}(\beta)\,V.
$$

$L_{ij}$ measures how much token $i$'s write overlaps token $j$'s ($i>j$).
Unrolling the within-chunk recursion is exactly a lower-triangular solve, so one
$C\times C$ inverse replaces $C$ dependent rank-1 updates. $W,U$ are the batched
"WY representation" of those updates.

**(3) State carry (the only serial part).**

$$
V_{\text{new}} = U - W S_0,
\qquad
S_{\text{next}} = \mathrm{diag}(e^{G_C})\,S_0 + K_g^{\top} V_{\text{new}}.
$$

Chunk $n{+}1$ needs $S_{\text{next}}$ of chunk $n$ — this recurrence over chunks
is the serial spine.

**(4) Output.**

$$
A_{qk}=\mathrm{tril}\!\big(Q_d K_i^{\top}\big),
\qquad
O = Q_d\,S_0 + A_{qk}\,V_{\text{new}}.
$$

$Q_d S_0$ is the history term (everything before this chunk, via $S_0$);
$A_{qk} V_{\text{new}}$ routes within-chunk contributions directly.
$A_{qk}[i,j]=\langle q_i,k_j\rangle e^{G_i-G_j}\cdot\text{scale}$ for $i\ge j$ —
a plain causal score matrix, no softmax, no inverse. $L$, $M$, $A_{qk}$ are all
**per-chunk $C\times C$** matrices, never $T\times T$.

Every framework implements formulas (1)–(4); they differ only in chunk size $C$
and in how the work is cut into kernels.

---

## 6. The gated output RMSNorm

The delta-rule output $r=o_t$ (pre-norm) is finished by a **gated RMSNorm**,
`FusedRMSNormGated(activation="sigmoid")` over the $d_v=128$ head dim, with a
learned weight $w$ and the output gate $z$ from §1:

$$
o = \Big(r\cdot\mathrm{rsqrt}\big(\tfrac{1}{d_v}\textstyle\sum_c r_c^2 + \varepsilon\big)\Big)\odot w \odot \sigma(z),
\qquad \varepsilon=10^{-5}.
$$

(RMSNorm — no mean subtraction. The `"sigmoid"` activation multiplies by
$\sigma(z)$; the `"swish"/"silu"` variant would multiply by $z\,\sigma(z)$.) In
production this is a separate launch after the core; we also ship a version that
fuses it into the epilogue of the delta-rule kernel, which is essentially free
because the kernel already holds $r$ in registers.

---

## 7. Existing implementations and their kernel strategies

All of the following compute the same formulas (1)–(4); the differences are
chunk size and kernel decomposition.

### 7.1 FLA / vLLM (reference, `chunk_kda`)
The upstream flash-linear-attention chunk kernel, chunk $C=64$, Triton. vLLM
vendors it as-is. Correct and general (any $T$, varlen) but not tuned for B200
throughput. This is the semantic reference everyone starts from.

### 7.2 SGLang Triton (`chunk_kda_fwd`, 8 launches)
Cuts (1)–(4) into **8 kernel launches** (7 distinct kernels; L2-norm runs
twice): two L2-norms, gate+cumsum, the $L$/inverse build, the $W$/$U$ solve, the
$A_{qk}$ score GEMM, and the serial state scan
(`chunk_gated_delta_rule_fwd_kernel_h_*`, the dominant one). $C=64$. Each launch
round-trips the per-chunk matrices ($L,M,A_{qk}$ stacked as $[1,T,H,64]$)
through HBM. This is the throughput baseline we must "largely beat"; we have a
2-launch CuTeDSL fusion of it that already does.

### 7.3 CUTLASS FlashKDA (MoonshotAI, `references/flash-kda/`, CuTe C++)
Restructures the math into **two kernels with $C=16$** (so $e^{G}$ over 16
tokens stays in bf16 range and $(I-L)^{-1}$ is a cheap Neumann series):

- **K1 `_flash_kda_fwd_prepare`** — token-parallel, grid (16-token-tiles, $H$),
  256 threads, `__launch_bounds__(256,8)`. Does L2-norm, fused gate+intra-tile
  cumsum, builds $Q_d,K_d,K_i,K_g$ in smem, the two $16\times16$ MMAs
  ($L=K_dK_i^\top$, $M_{qk}=Q_dK_i^\top$), and $M=(I-L)^{-1}$ by **fp16 Neumann
  series**. Writes compact per-tile arrays ($K_d,Q_d,K_g$ $[16,128]$ bf16,
  $G_C$ $[128]$ fp32, $M,M_{qk}$ $[16,16]$) to a workspace. $W,U$ are never
  materialized.
- **K2 `_flash_kda_fwd_recurrence`** — head-parallel, grid $(N,H)$, 192 threads
  (2 TMA producer + 4 MMA consumer warps), 3-stage `PipelineTmaAsync`. Keeps the
  $128\times128$ state in **smem in bf16** for the whole sequence and walks the
  16-token tiles serially: dual GEMM $K_dS_0/Q_dS_0$, then
  $V_{\text{new}}=M\,\mathrm{diag}(\beta)(V-K_dS_0)$ (the solve fused next to the
  state, $W/U$ never formed), $O\mathrel{+}=M_{qk}V_{\text{new}}$ (with
  $V_{\text{new}}$ transposed in-register via `MOVM_T`), TMA-store $O$, then the
  state update $S\leftarrow\mathrm{diag}(e^{G_C})S+K_g^\top V_{\text{new}}$.

Splitting prep (abundant token-parallelism) from the serial carry (scarce
head-parallelism) was worth ≥15% over an earlier fused prototype.

### 7.4 INT21 `flashkda-ptx` — the target to beat
Hand-written PTX ([`Int21-AI/KDA-B200`](https://github.com/Int21-AI/KDA-B200),
`csrc/flash_kda_cuda.cu`), same two-kernel $C=16$ shape as FlashKDA but ~1.5×
faster than CUTLASS. Its public entry is
`fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state,
final_state, cu_seqlens)` → two kernels, `prepare` + `recurrence_mma`. Note the
signature: it takes **no conv weight and no output-gate / norm weight**, so it
fuses the *input* activations (safe gate, β-sigmoid, q/k L2-norm) and the delta
recurrence, but **not the short conv (§3) and not the gated output RMSNorm (§6)**
— those remain separate launches (verified from the source). What makes it fast,
as far as we can tell:
- **warp `mma.sync`** (not tcgen05) with the $128\times128$ state kept
  **register-resident** for the whole serial walk;
- **1 CTA/SM + an 8-deep software pipeline** on the carry;
- **cross-stream 3-segment overlap** and a **V-split across CTAs** to feed the
  serial carry more parallelism at large batch;
- an exact **$C=16$ doubling inverse** and a gate rebased to base-2
  (`ex2.approx`) so each decay is one instruction.

INT21 is our explicit target: **beat it at every batch size within the
$\le262144$-token envelope.**

### 7.5 Our CuTeDSL kernel (what we have)
Two kernels mirroring the prep/carry split, ported to CuTeDSL 4.5.2, with a
$C=16$ "INT21-shaped" schedule (`kda16.py`) and a $C=64$ fallback. Notable
choices reached by ablation: the WY sweep folded into the coupling GEMM so $W/U$
are never formed; the carry keeps `acc_S` (the $128\times128$ state) in
registers (exactly 128 regs/thread at rows-per-warp 32); 2-warp carry CTAs to
cover all 148 SMs; prep freed of ~45 KB smem to reach `min_blocks_per_mp=3`. It
also optionally fuses the gated RMSNorm (§6) into the output epilogue.

### 7.6 The fusion landscape — nobody fuses the whole layer

Worth stating plainly, because it is an open opportunity: **no existing
implementation fuses the entire KDA layer.** Every stack treats the layer as a
pipeline of separate launches, and the two "wrapper" ops — the **short causal
conv (§3)** on the front and the **gated output RMSNorm (§6)** on the back — are
*always* separate kernels from the delta-rule core:

| impl | conv (§3) | gate+norm+delta core (§4–5) | gated RMSNorm (§6) |
|---|---|---|---|
| SGLang Triton | separate | 8 launches | separate |
| CUTLASS FlashKDA | separate | 2 kernels (K1+K2) | separate |
| **INT21 `flashkda-ptx`** | **separate** | 2 kernels (prep+recurrence) | **separate** |
| ours (CuTeDSL) | separate | 2 kernels | **fused into K2 epilogue** (optional) |

INT21 — the fastest core — fuses neither wrapper (confirmed from its `fwd`
signature: no conv weight, no `z`/`w`). We already fuse the gated RMSNorm into
our carry's epilogue at ~zero cost (it reuses the $r$ already in registers). That
leaves **the short conv as the one piece nobody has folded in**, and **a fully
fused conv → delta → gated-RMSNorm prefill as unclaimed territory** — a real
lever on the extra HBM round-trips those separate launches pay, on top of beating
INT21's core.

---

## 8. Where we stand vs INT21 (measured, B200)

End-to-end prefill, safe-gate, correctness-verified against the shared reference.
Ratios are **ours / INT21** (`< 1` = we win):

| shape | vs CUTLASS FlashKDA | vs INT21 `flashkda-ptx` |
|---|---|---|
| $B=2$ (T=4096 / 65536) | win | **0.88 / 0.87** (we win) |
| $B=4$ (T=65536) | win (~4.69 ms) | ~**1.07** (we trail ~7%) |
| $B=8$ (T=4096) | win | trails |
| $B=16$ (T=4096) | win | ~**1.8** → halved in round 2 |

We **beat CUTLASS FlashKDA at every $B>1$** and **beat INT21 at $B=2$**, but
**trail INT21 at $B\ge4$**, worst around $B=8\text{–}16$.

**Our profiling conclusions (so you don't have to rediscover them):**
- The **prep** kernel (formulas 1–2) is 58–63% of runtime and is
  **bandwidth-closed at ~58% of HBM peak** — the workspace round-trip is 105 KB
  of the 201 KB moved per 64 tokens per (b,h).
- The **carry** kernel (formulas 3–4) is **latency-bound, not
  L1TEX-bandwidth-bound** at large batch: 2 warps/scheduler, and the warps are
  **register-closed** (`acc_S` alone is 128 regs/thread at rows-per-warp 32).
- The one shape round 2 could not move is **$B=8$ ($H\cdot B=128$)**: it wants
  the L1TEX saving of rows-per-warp 32, but 128 CTAs × 4 warps leave one warp
  per scheduler, and every CTA shape that gives 4× operand replication loses
  there.
- We priced **tcgen05/TMEM-resident state** (to read the carry's replicated
  operands once per CTA instead of 4×) and **rejected** it: the prize is ~33% of
  the carry *if the operands were free*, but the fp32-accumulator→bf16-A-operand
  TMEM round-trip costs back ~2.5× what round-6 measured. `OperandSource.TMEM`
  exists in this build; the premise was sound, the economics were not.

---

## 9. The ask for NVIDIA's kernel team

We want a KDA prefill kernel that **beats INT21 `flashkda-ptx` at every batch
size** on B200, ideally in CuTeDSL/CUTLASS (so it stays maintainable) but we are
open to whatever wins. You may freely assume the $T_{\text{total}}\le 262144$
envelope above (static workspaces, narrow indexing, fixed pipeline depths — no
unbounded-$T$ path required). Concretely, we would value your guidance on:

1. **The $B=8$ occupancy cliff.** At $H\cdot B=128$ the serial carry gets one
   warp per scheduler and the register-resident $128\times128$ state can't be
   made cheaper without going to TMEM. Is there a B200 scheduling pattern
   (cluster/DSMEM state sharing, PDL, distributed shared memory for the
   cross-chunk carry, 2-CTA clusters splitting $d_v$) that gives the serial
   recurrence more independent warps **without** paying the TMEM round-trip we
   measured?
2. **tcgen05 for a serially-carried state.** Our TMEM attempt lost to the
   accumulator↔operand round-trip on a state that must be both an MMA
   accumulator (updated every chunk) and an A/B operand (read every chunk). Is
   there a tcgen05/TMEM dataflow on sm_100a where the state stays in TMEM across
   chunks and feeds the next MMA **without** a bf16 restage — i.e. an in-TMEM
   accumulate-then-consume pattern we're missing?
3. **prep↔carry fusion at $H\cdot B\ge N_{\text{SM}}$.** The largest un-priced
   idea: fusing prep into the carry deletes the 105 KB workspace round-trip
   (prep is bandwidth-closed at 58% of HBM peak). We rejected fusion earlier
   because one $H\cdot B$-CTA grid on a $T/C$-long serial chain starves the SMs —
   but that reason **stops applying once $H\cdot B\ge N_{\text{SM}}$**, which is
   exactly the $B\ge8$ regime where we trail. Is a fused single-kernel prep+carry
   the right structure there, and what's the cleanest way to keep prep's
   token-parallelism alive inside a head-serial kernel?
4. **The $(I-L)^{-1}$ solve.** At $C=16$ we do a 3-step tensor-core doubling
   (fp16). Is there a faster exact small-triangular-inverse primitive on B200
   (or a way to avoid materializing $M$ at all beyond what the WY fold already
   does)?
5. **The prep bandwidth bound.** prep is L1TEX-bound with ~82% of wavefronts in
   shared memory at $C=64$; the $C=16$ port cut that but prep is now HBM-bound at
   58%. Is there a smem/TMA layout for the six per-tile workspace arrays
   ($K_d,Q_d,K_g,G_C,M,M_{qk}$) that we're leaving on the table?

Reproduction, the CuTeDSL sources, the CUTLASS `flash-kda` and INT21
`flashkda-ptx` references, and the correctness/perf harness are all in this
repository; we can walk through any of it and profile live on B200.

Beyond matching INT21's core, the **full-layer fusion** of §7.6 (conv → delta →
gated-RMSNorm in one prefill) is on the table too — the gated RMSNorm we already
fuse; the short conv is the remaining unfused piece.

---

## References

- **INT21 `flashkda-ptx`** (the target): <https://github.com/Int21-AI/KDA-B200>
  — kernel: `csrc/flash_kda_cuda.cu` (`prepare` + `recurrence_mma`); Python entry
  `flash_kda.fwd(...)`. In-repo copy: `KDA-B200/`.
- **CUTLASS FlashKDA** (MoonshotAI): <https://github.com/MoonshotAI/FlashKDA>
  — CuTe C++ `_flash_kda_fwd_prepare` (K1) + `_flash_kda_fwd_recurrence` (K2),
  $C=16$; design deep-dive under `docs/`. In-repo copy: `references/flash-kda/`.
- **FLA (flash-linear-attention)** — reference `chunk_kda`; gated-RMSNorm
  reference `fla/modules/fused_norm_gate.py`.
- **SGLang** — `chunk_kda_fwd` (Triton, 8 launches); the KDA layer, conv, and
  `FusedRMSNormGated` wiring in `sglang/srt/models/kimi_linear.py`; SGLang's own
  CuTeDSL decode kernel `sglang/jit_kernel/cutedsl_kda.py`.
- **Our CuTeDSL kernels + harness** — this repository (`LLMDiveDeep/linear_attn/`).

