# Idea ledger

## Survey

### TensorRT-LLM native CUDA causal-conv1d

Source: `cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu`.

The unchanged native algorithm assigns a CTA to `(sequence, channel)`, streams
the temporal axis in 128-bit per-thread chunks, exchanges one neighboring chunk
through warp shuffle/shared memory, accumulates in FP32, and updates one state
row per channel. Its channel-major input makes temporal loads contiguous. The
KDA caller pays a separate token-major-to-channel-major transpose before this
kernel and later consumes a transposed view.

Useful mechanisms:

- fixed `W` specialization and fully unrolled FP32 FMA;
- 128-bit vector transfer along the physically contiguous dimension;
- exact short-sequence state update;
- direct initial-state injection into the first temporal window.

Mismatch with the requested direct path:

- its `(sequence, channel)` geometry serializes all tokens of one channel;
- token-major KDA input requires a physical layout-copy kernel first;
- native output is channel-major, not the requested contiguous token-major
  package output.

### TensorRT-LLM Triton prefill

Source lineage: git `HEAD` of
`tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py`.

The one-launch Triton kernel uses `(sequence, 8-token chunk, 256-channel tile)`.
It reads token-major physical storage through a transposed metadata view,
parallelizes channels and temporal chunks, uses FP32 accumulator preload, and
has the same rolling state logic. The working tree has two local changes:
explicit FP32 multiply conversion and optional padded-input copy. Those changes
are not treated as an unchanged baseline.

This is the strongest algorithmic comparison: direct row-major access and one
launch. It is also the nearest geometry to test rather than merely comparing
against the multi-launch native pipeline.

### CuTeDSL vectorized elementwise template

Source:
`optimizations/blackwell/dict/templates/elementwise_vectorized_fp16/run.py`.

Useful mechanisms:

- `cute.zipped_divide` over the contiguous last dimension;
- one thread owns an aligned vector and uses `cute.autovec_copy`;
- FP16 input is widened to FP32 register values;
- cached `cute.compile` callable with DLPack outside timed execution.

Mismatch:

- causal convolution gathers up to four token rows and has sequence boundaries;
- state update is an additional side effect;
- a bulk-plus-tail two-kernel template is forbidden by the one-launch contract.

### CuTeDSL depthwise-convolution template

Source:
`optimizations/blackwell/dict/templates/depthwise_conv_ref/run.py`.

Useful mechanisms:

- recognizes depthwise convolution as a low-arithmetic-intensity stencil rather
  than a tensor-core GEMM;
- vectorizes the contiguous dimension;
- compares no-SMEM register-window and SMEM-staging variants;
- preserves FP32 accumulation in accepted variants.

Mismatch:

- the template is fixed-shape NCHW 2-D spatial convolution;
- it has no packed-varlen boundaries, cached history, or in-place slot update;
- its per-plane/row geometry does not directly map to token-major channels.

### CuTeDSL cumsum scan template

Source:
`optimizations/blackwell/dict/templates/cumsum_scan_ref/run.py`.

Useful lesson: serial dependence along one axis should remain local to a thread
or warp while independent axes are parallelized. For this operation `W <= 4`,
so there is no long scan: every output token can independently gather its
bounded causal window. State update needs only the final CTA/chunk.

### Conv1d implicit-GEMM template

Source:
`optimizations/blackwell/dict/templates/conv1d_implicit_gemm_ref`.

Rejected as a starting algorithm. It is a dense cross-channel convolution with
im2col TMA and tcgen05 MMA, requires transformed channels-last storage and large
shape divisibility, and has no cached state semantics. This contract is
depthwise with only `W <= 4`; tensor-core setup would dominate and would not
remove packed-sequence control flow.

### Upstream example index

The skill's listed Ampere `elementwise_add.py` and `elementwise_apply.py` raw
URLs returned HTTP 404 on 2026-08-26. The local promoted vectorized elementwise
and depthwise templates above are newer CuTeDSL 4.5.2 references and were used
instead.

## Falsifiable ideas

| ID | Idea | Prediction | Test | Status |
|---|---|---|---|---|
| I00 | Direct token-major SIMT kernel, one CTA per `(sequence, token tile)`, vectorized over channels | Removes native layout copies and exposes channel/token parallelism; should beat the native pipeline at main BF16 W4 shapes | Compare full-pipeline native latency and one-launch CuTe latency | **confirmed:** beats native on 72/72 matrix shapes, 2.15–4.90x |
| I01 | One thread owns 8 contiguous FP16/BF16 channels, loading each causal row as a 128-bit vector | Coalesced row-major accesses and one 128-bit output store should reduce instruction count versus scalar channels | Compare against one-launch TRT Triton | **useful:** beats Triton on 52/72 shapes; scalar/vector-4 not pursued because the direct vector-8 result already won short/medium shapes and long shapes showed a token-tiling trend |
| I02 | Use 8-token CTA tiles like TRT Triton | Bounded register work with enough CTAs; should compare closely to Triton's one-launch path | Sweep token tile 4/8/16 with identical vector width | **refuted as universal:** tile 4 wins short/medium; tile 16 wins long; adaptive selection retained |
| I03 | Let only the CTA containing the sequence tail update state | Avoids atomics and duplicate state stores while preserving one launch | Correctness on short, mixed-state, permuted-slot cases | **refuted:** tail CTA races chunk-zero initial-state reads. Replaced by chunk-zero snapshot-then-update, matching native/Triton ordering |
| I04 | Preload channel weights/bias once per thread and reuse across token tile | Reduces repeated weight traffic; larger token tiles improve reuse until register pressure/serial work dominates | Tile sweep and latency by T/B | **confirmed tradeoff:** tile 4/8/16 large-shape latency 0.1581/0.1036/0.09195 ms; compile 1.46/2.60/4.98 s |
| I05 | Match native fast exponential for SiLU | Preserve accepted TRT tolerance while avoiding a slower high-precision path | Adversarial SiLU comparison | **confirmed:** `cute.math.exp(..., fastmath=True)` passes native tolerance; worst focused errors FP16 1.22e-4, BF16 1.95e-3, state bitwise |

## Round 2 independent hard acceptance gates

The selected implementation may dispatch only among CuTe DSL kernels using
input metadata. It must satisfy both independent gates below on every primary
BF16/W4 shape:

```text
D in {1536,3072}, T in {128,1024,8192}, B in {1, uneven 8}
```

1. **TRT gate:** beat unchanged exact git-`HEAD` TRT Triton with a defensible
   repeatable positive margin on all 12 shapes.
2. **SOL gate:** for long memory-bound shapes, latency must be at most `1.25x`
   the defensible same-clock measured bandwidth lower bound (at least 80%
   analytical SOL efficiency) and NCU Memory Throughput/SOL must be at least
   80%. For short/crossover launch-bound shapes, latency must be at most `1.25x`
   the same-clock measured launch/dispatch floor. High HBM utilization is not
   required for underfilled launch-bound kernels.

Analytical lower-bound efficiency (`lower_bound / benchmark_latency`) and NCU
peak-sustained throughput/SOL are distinct metrics and must both be reported.
If authoritative NCU evidence identifies another necessary resource bound, the
defensible roof must be recomputed and documented; the 80% threshold is not
relaxed. Final measurements must remeasure attainable bandwidth and launch
floor in the same fixed-clock environment.

Any comparison within 5% is inconclusive until repeated rounds establish
dispersion and confidence. If either gate fails on one primary shape after the
profile-driven campaign, the CuTe candidate is `REJECTED`; wins over the native
three-kernel pipeline do not relax either gate. Equal output/state work, FP32
accumulation, one launch, padded/state semantics, and timing boundaries are
immutable. Candidate-only untimed conversions and Triton fallback are forbidden.

## Round 2 profile-driven candidates

The long tile-16 kernel uses 84 registers/thread, is register-limited to 31.25%
theoretical occupancy, achieves 27.3–27.4%, and spends 35.2–44.3% of stall
cycles on L1TEX scoreboard dependencies. Its 852–1029 GB/s DRAM rate reaches
only 17.2–23.1% of the analytical bound. TRT Triton uses 32 registers/thread,
73–89% achieved occupancy, and remains 1.34–1.57x faster.

| ID | Controlled change | Counter prediction | Latency prediction / rejection rule | Status |
|---|---|---|---|---|
| R2-I01 | Replace W independent row loads per output with a per-thread W-entry circular token window: load W-1 history vectors once per tile, then one new vector per output | Executed load instructions and L1TEX scoreboard stalls fall materially; DRAM obligations unchanged | Long T8192 latency improves; reject if register growth offsets the reduced loads or stalls do not move | **confirmed:** first streaming V4/tile16 cut D3072/B8 from 0.09283 to 0.06605 ms |
| R2-I02 | Reduce channel vector width 8→4 while keeping FP32 weights/accumulators | Live weight, bias, accumulator, and state registers roughly halve; target <=64 regs/thread and >=50% theoretical occupancy | Long latency improves through more resident warps; reject if doubled thread/CTA work dominates | **confirmed mechanism:** early streaming profile reached 55 regs and 56.25% theoretical occupancy; final lookahead/TensorSSA path uses 68 regs |
| R2-I03 | Sweep CTA threads 64/128/256 and corresponding channel tiles under vector width 4 | 64-thread/256-channel CTAs approach TRT's channel granularity and increase independent CTAs; 128/256 threads test launch versus occupancy | Select by long B1/B8 latency and achieved eligible warps; reject topologies that only increase instruction/launch overhead | **confirmed 128:** tile32 D3072/B8 was 0.07294/0.07354/0.07561 ms for 128/64/256 threads |
| R2-I04 | Sweep streaming token tiles 8/16 after register reduction | Tile 16 amortizes history/weights; tile 8 doubles independent CTAs and may hide current-row latency | Keep per-regime winner only if crossover is repeatable; no single universal tile assumed | **extended and confirmed tile12:** tile12 beats 8/16 on the final packed grid and is selected for W4 |
| R2-I05 | Prefetch the next current-row vector before consuming the current vector, using one extra input fragment | More independent L1TEX requests and lower long-scoreboard percentage with a small register increase | Accept only if NCU shows higher eligible warps or lower stalls and latency improves over plain streaming | **confirmed:** required for D3072; removing it regressed long B1 from about 0.0451 to 0.0501 ms |
| R2-I06 | Stage token rows in shared memory only for channel tiles that can reuse them | Could reduce redundant history loads without additional registers | Defer unless circular registers fail: channels are disjoint across threads, so SMEM cannot share values across threads and likely adds barriers | deferred by data-reuse analysis |
| R2-I07 | Preserve the existing direct vector-8/tile-4 kernel for short/launch-bound shapes and dispatch long/crossover shapes to the best lower-register streaming kernel | T128 latency remains unchanged while T8192 uses the profile winner | Final dispatcher must beat Triton on all 12 shapes, including repeated <=5% margins | **refuted:** the streaming tile12 W4 kernel is also the selected short/crossover regime; direct remains only for W2/W3 |
| R2-I08 | Use persistent channel-owner CTAs that traverse multiple token chunks with a width-4 register ring | Eliminates inter-chunk W-1 reloads and weight reloads; sharply reduces grid/launch work while retaining one current-row load/output | Accept only if long analytical SOL efficiency and NCU memory SOL both move toward 80%; reject if too few CTAs underfill B1 | **rejected as pure persistence:** D1536/B1 exposes too few channel CTAs; bounded 12-token persistence retains enough grid parallelism |
| R2-I09 | Double-buffer current-row vector loads inside persistent CTAs | Increases independent memory requests per warp and lowers long-scoreboard stalls without restoring tile-wide live values | Require lower scoreboard stalls and improved latency versus plain ring at comparable registers | **confirmed as one-vector lookahead:** final B1 profile lowers scoreboard stalls to 36.1%, but remains far from SOL |
| R2-I10 | Evaluate CuTe DSL architecture-specific async/global-to-shared memory paths for cooperative token-channel staging | May overlap memory latency at CTA scope, but only if the staged layout creates cross-thread reuse and avoids barriers on every token | Use only after proving legal coalesced copies and measurable counter gains; no architecture-specific mechanism is accepted by assumption | **L2 prefetch refuted; SMEM deferred:** `prefetch.global.L2` regressed D3072/B8 0.06499→0.06683 ms; register ring already captures cross-token reuse |
| R2-I11 | Replace rectangular `(max_blocks * B)` launch space with exact packed token-block prefixes embedded from existing CPU sequence metadata | Eliminates inactive B8 CTAs without a conversion kernel or extra launch | B8 long latency approaches B1 while all state boundaries remain exact | **confirmed:** D3072/T8192/B8 fell from 0.06499 to 0.04610 ms and final B1/B8 are 0.04508/0.04616 ms |

### One-launch state-ordering invariant for streaming variants

Each `(sequence, channel tile, token chunk)` is one CTA. Only chunk zero reads
cached history. It snapshots the `W-1` initial values into its circular register
window before writing final state, then computes every output that can consume
cached history (`token_tile >= W-1`). Later chunks initialize history only from
immutable `projected` rows and never touch cached input state. Therefore the
single chunk-zero state write cannot race any state read, and padded slots skip
both state reads and writes.

## Round 3 scope addition: Kimi-K3 production strided shape

The gate matrix now also covers the production geometry: `D=4608` (packed QKV
`3x1536`, 12 heads, head dim 128), BF16, W4, SiLU, no bias, with the input as
a row-strided view `buffer[:, :4608]` of the `[T, 4752]` in_proj output
(`stride(0)=4752`). `T in {128,1024,8192}`, `B in {1, uneven 8}`. TRT Triton
and the native extract kernel accept arbitrary strides, so both baselines run
the identical strided view unchanged. A bias variant is covered in
correctness. Alignment facts are measured, not assumed: rows of the strided
view are `9504 = 32 x 297` bytes apart, so they are 16- AND 32-byte aligned
(the pre-round claim that `9504 % 32 != 0` is arithmetically wrong), but they
are NOT 128-byte aligned (`9504 % 128 = 32`), which costs partial DRAM lines
at row edges. The wrapper derives `assumed_align` and the effective vector
width from real pointers and strides per call.

## Round 3 roof calibration (results/r3_roof_calibration.json)

Hypothesis to test first: the 1 GiB-copy bandwidth roof (6335.9 GB/s) may be
loose for ~100 MB working sets. Method: minimal same-byte-volume,
same-access-pattern CuTe streaming copy kernels (one CTA per channel-block x
token-tile, one vector load + one vector store per token) plus torch flat
copies at the exact payloads, all at the 1500 MHz lock.

Findings:

- **Refuted for contiguous long shapes.** The same-pattern copy at
  T8192/D3072 attains 6957 GB/s (vec8/tile4), above both the 1 GiB copy
  (6336 GB/s) and the same-payload torch flat copy (6392 GB/s). The round-2
  roof was attainable; the round-3 defensible roof for contiguous long shapes
  tightens to 6956.6 GB/s.
- **Confirmed lower for the strided production pattern.** Best of
  vec4/8/16 x tile12/24 sweeps is 5844.8 GB/s (vec16/tile12). Rows are 32B-
  but not 128B-aligned, so every 9216-byte row read spans partial 128B DRAM
  lines at both edges. The strided long roof is re-derived to 5844.8 GB/s
  with this evidence; the 80% threshold itself is unchanged.
- **Medium/short payloads cannot reach copy-peak either.** Same-payload torch
  copies attain 2848 GB/s at T1024/D3072 (4.42 us) and the same-pattern T128
  copy takes 5.87 us against the 2.963 us launch floor. These are reported as
  supplementary calibration; the strict launch-floor bound is retained for
  short/crossover gate rows because raising a floor relaxes the gate and a
  faster small-payload streamer was not exhaustively ruled out.

## Round 3 falsifiable ideas

| ID | Controlled change | Prediction | Status |
|---|---|---|---|
| R3-I01 | Same-pattern streaming-copy calibration of the bandwidth roof | The 1 GiB roof is loose at ~100 MB | **refuted for contiguous** (6957 > 6336 GB/s); **confirmed for strided** (5845 GB/s) and short/medium payloads |
| R3-I02 | Replace SiLU `x/(1+exp(-x))` (fastmath exp + full-precision divide) with `0.5x(1+tanh(x/2))` via one fastmath MUFU.TANH | The long kernel is issue-limited (59% compute SOL, 138 warp-instructions per warp-token-vector), so removing the divide sequence cuts latency materially | **confirmed:** dense long B8 0.0457->0.0382 ms (-16%), strided long -25%; correctness preserved vs native tolerance |
| R3-I03 | 16-byte vectors (vector_width=8) after the calibration showed vec8 streams 6744 vs vec4 5375 GB/s | Halving load/store/addressing instructions outweighs register growth | **refuted in the conv kernel:** 0.0425 vs 0.0382 ms (dense long); vec16 worse still (0.0595); register/occupancy cost dominates, unlike the pure copy |
| R3-I04 | 32-byte vectors (vector_width=16, LDG.256-class) with measured 32B alignment | Quarter the memory instruction count | **refuted:** slowest of the three widths at every long shape |
| R3-I05 | Larger token tiles (20/24) to cut the `(W-1)/tile` redundant history reloads | Less read amplification approaches the one-read-per-row ideal | **split:** tile20 wins strided long (0.0495 vs 0.0501 ms) but loses short/dense; tile24 loses everywhere |
| R3-I06 | Remove the one-vector software lookahead under tanh SiLU (shorter dependency chains may not need it) | Neutral-to-positive | **confirmed for dense long:** no-prefetch tanh is the dense-long winner 0.0353 ms (B8) vs 0.0382 with lookahead; strided long slightly prefers lookahead |
| R3-I07 | CTA width 64/256 threads at tile12 | More CTAs (64) may improve tail/occupancy; 256 reduces launch count | **confirmed 64 for short/medium:** t64 wins T128/T1024 (0.00756 ms) and strided medium; t256 loses long shapes badly |
| R3-I08 | Group-batched loads: issue all `group_span` independent row loads of a token group before computing (separate history fragments instead of the aliased ring) | Round-3 NCU shows long-scoreboard stalls are 69%/58% of 14.2/13.0 cycles-per-issue on dense/strided long with only ~one 8B load in flight per thread; more bytes in flight should cut the stall share and raise memory SOL | **confirmed, largest single win of round 3:** dense long B8 0.0353->0.0286 ms (-19%), strided long 0.0495->0.0408 (-18%); span 8 + tile 24 best without F32 ring; NCU confirms long-scoreboard share fell and DRAM SOL rose to ~53-58% |
| R3-I09 | Wider group span (8, 12) and tiles (16-32) on the group-loads path | More loads in flight keeps cutting scoreboard stalls until register pressure bites | **span 8 optimal without F32 ring; span 12 regressed** (register growth); with F32 ring the optimum shifts to span 4 (see R3-I11) |
| R3-I10 | FP32 parameter fragments (`KDA_CUTE_FP32_PARAMS=1`) to remove per-tap BF16->FP32 weight converts | NCU shows ALU pipe at 64-66% (top pipe); weight converts are per-tap-per-vector | **marginal** (<2% moves, within noise on most shapes): compiler already hoists weight converts out of the token loop; outputs bitwise-identical (precision audit) |
| R3-I11 | F32 ring: store ring/current history in FP32 so each loaded value is converted BF16->FP32 exactly once instead of once per consuming tap | Remaining ALU-pipe pressure is input-value converts (W=4 taps consume each row 4x) | **confirmed at span 4:** dense long B8 0.0286->0.0268 ms, strided long 0.0408->0.0399/0.0401; at span 8 it regresses (F32 fragments double register bytes; occupancy drops). New overall best |

## Round 3 precision audit (results/r3_silu_precision_audit.json)

User directive: no silent numeric weakening. Audit on every W4 correctness
case (normal, unscaled, adversarial extremes, short, padded, strided
production, fp16) plus both long gate geometries, all versus the unchanged
TRT native pipeline:

- **FP32 accumulation is mandatory and present in every variant** (audited in
  code: accumulator fragments are F32; taps convert operands at the multiply).
- **BF16 parameter fragments are bit-identical to the FP32-parameter path**
  (outputs and states bitwise equal on all cases, both SiLU modes): source
  weights/bias are BF16, and BF16->FP32 conversion at the multiply is exact.
  This is a register-pressure choice, not a numeric downgrade.
- **Config invariance proven bitwise:** tile12/tile20, 128/64 threads, and
  prefetch on/off produce bitwise-identical outputs and states (same FP32
  accumulation order per output).
- **SiLU formulations, measured against native:** expdiv (fastmath exp +
  IEEE divide) worst max-abs 1.95e-3 / worst max-rel 7.6e-3 and bit-identical
  to native on several cases; tanh (MUFU.TANH) worst max-abs 3.13e-2
  (bf16 unscaled; atol gate 1e-1) and max-rel 1.0 only where the reference
  magnitude is below atol (abs error there <= 4.9e-4). Worst direct
  tanh-vs-expdiv output delta 3.13e-2. All cases pass the frozen tolerances
  (fp16 rtol 1e-2/atol 1e-2; bf16 rtol 1e-2/atol 1e-1) in both modes, and
  conv states are bitwise exact in both modes. The tanh form is selected for
  speed (dense long 0.0353 vs 0.0457 ms) with this delta reported, per the
  directive; expdiv stays available via `KDA_CUTE_SILU_MODE=expdiv`.

## Decision gate

### Pre-candidate evidence

Machine floors at the fixed 1500 MHz clock:

- one-launch floor: 0.002963 ms;
- 1 GiB device copy: 0.338939 ms, 6335.91 GB/s counting read plus write;
- runtime-reported SM count/capability: 148 / 10.3;
- FP32 peak used for the arithmetic lower bound: 56.832 TFLOP/s
  (`148 * 128 lanes * 2 FLOP * 1.5 GHz`).

Main BF16 W4 baseline range:

- native three-kernel pipeline: 0.02959–0.25206 ms;
- git-`HEAD` one-launch Triton: 0.02924–0.05879 ms.

The largest shape `(T=8192,D=3072)` has:

- mandatory-byte lower bound: 100.73–100.92 MB;
- bandwidth floor: 0.01590–0.01593 ms;
- arithmetic floor: 0.00354 ms;
- serial four-FMA chain floor: 0.0000107 ms;
- defensible combined floor: 0.01590–0.01593 ms;
- measured Triton: 0.05097 ms (B1), 0.05879 ms (B8), or 31.2%/27.1%
  of the lower-bound SOL respectively.

Small/medium shapes are launch dominated by the conservative model, while their
measured ~0.029 ms latency remains roughly 10x the isolated launch floor. The
native path loses up to 4.3x to the one-launch Triton path at large B8 shapes,
confirming that layout copies and channel-serial geometry are material.

### Round-2 decision (superseded): REJECTED/UNQUALIFIED

The round-2 selected dispatcher passed the TRT gate 12/12 with five-round
`1.125–3.466x` speedups, but failed the then-hard SOL gate 12/12 at only
25.6–35.5% analytical efficiency (long NCU Memory Throughput/SOL 20.3–20.5%
versus the 80% requirement).

### Round-3 selection: single configuration for all shapes

User acceptance update: one kernel configuration for every shape (no
per-regime dispatcher), must beat unchanged git-HEAD TRT Triton on all 12
primary dense shapes plus the 6 strided production shapes; the SOL gate is
softened to best-effort with plateau evidence.

Selected config (now the backend default with no env vars set):
**stream algorithm (W4), vector width 4, 128 threads/CTA, token tile 16,
group-batched loads with span 4, F32 ring, tanh SiLU, BF16 parameter
fragments (bit-identical to FP32 params), FP32 accumulation.** W2/W3
correctness shapes use the direct kernel (stream is W4-only); no primary gate
shape is W2/W3.

Quick-matrix worst-case sacrifice versus per-shape best across all round-3
sweeps: +3.7% (dense long B8: 0.02779 vs 0.02680 ms for tile24-span4, which
loses strided-medium by 15%). All other measured shapes are within ~2.3% of
their per-shape best. Final verdicts live in
`results/r3_acceptance_summary.json` and REPORT.md.
