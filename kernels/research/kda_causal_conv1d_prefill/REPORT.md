# KDA causal-conv1d prefill report

## Outcome

Round 3 delivers a **single CuTe DSL kernel configuration for every shape**
(no per-regime dispatcher) for the one-launch packed-varlen KDA prefill
depthwise causal conv1d contract, covering both the original dense gate matrix
and the Kimi-K3 production row-strided shape.

Status per the round-3 acceptance framing (Triton gate hard, SOL gate
best-effort with plateau evidence):

- **TRT Triton gate: PASS 18/18.** The selected config beats unchanged
  SHA-checked git-`HEAD` TRT Triton on all 12 primary dense BF16/W4 shapes AND
  all 6 strided production shapes, with five-round repeatable margins of
  **1.774–3.949x**. The worst margin is 77%, so no shape is inside the 5%
  confidence-rerun band; every candidate round beats every Triton round on
  every shape.
- **Correctness: PASS.** Case suite (17 semantic cases) and the full matrix
  (dense 72 + strided production variants, 108 correctness rows) pass against
  the unchanged TRT native pipeline; conv states are bitwise exact.
- **SOL, best-effort:** against the round-3 *calibrated attainable* roofs the
  selected config reaches **52.0–52.1%** on dense D3072 long, **59.7–60.7%**
  on dense D1536 long, and **64.7–64.9%** on the strided production long
  shapes. Short/medium shapes sit at 31–40% of the conservative launch-floor
  bound. The 80% target was not reached; plateau evidence below documents
  where the residual goes (DRAM latency under occupancy-limited hiding, with
  no saturated pipe).

No TensorRT-LLM file was edited, no KDA integration was added, no Triton
fallback exists, and no commit was created.

## Selected single configuration

One setting for all shapes, batch sizes, and layouts — now the backend
default when no environment variable is set:

| Knob | Value |
|---|---|
| algorithm | stream (W4 gate shapes; W2/W3 correctness shapes use the direct kernel — stream is W4-only and no gate shape is W2/W3) |
| vector width | 4 (8-byte LDG/STG; clamped by measured alignment) |
| threads per CTA | 128 |
| token tile | 16 |
| group-batched loads | on, span 4 |
| history/current fragments | FP32 ring (each loaded value converted BF16→FP32 exactly once) |
| SiLU form | tanh (`0.5x(1+tanh(x/2))`, MUFU.TANH; expdiv available via `KDA_CUTE_SILU_MODE=expdiv`) |
| parameters | BF16 weight/bias fragments — bitwise identical outputs/states vs FP32 fragments (precision audit) |
| accumulation | FP32, mandatory, in every variant |

Worst-case sacrifice versus the per-shape best variant observed in the
round-3 sweeps is **+3.7%** (dense D3072 T8192 B8: 0.02779 vs 0.02680 ms for
tile24/span4, which in turn loses a strided medium shape by 15%). All other
measured shapes are within ~2.3% of their per-shape best, so no dispatch is
needed.

## Hard gates (round-3 framing)

1. Beat SHA-checked git-`HEAD` TRT Triton with repeatable positive margin on
   every one of the 12 primary dense BF16/W4 shapes (`D={1536,3072}`,
   `T={128,1024,8192}`, `B={1,uneven 8}`) **and** the 6 production strided
   shapes (`D=4608`, row stride 4752, no bias, same T/B grid). **PASS 18/18.**
2. SOL is best-effort per the user acceptance update: report achieved
   fractions against the calibrated roofs plus plateau evidence, instead of a
   REJECTED-on-SOL verdict. Precision rules stay binding: FP32 accumulation
   everywhere, documented SiLU-form deltas, no silent downgrades.

Full FP16/BF16 W2/W3/W4 correctness (dense 72 + strided production), FP32
accumulation, equal output/state and padded-slot work, one launch, equal
timing boundaries, and no candidate-only untimed conversion remain mandatory
and all hold.

## Hardware, sources, and floors

- CUDA runtime identity: `NVIDIA L20D`, capability 10.3 (SM103), 148 SMs,
  287,428,771,840 bytes; host `nvidia-smi` reports capability 8.9. No B200
  claim is made.
- Fixed benchmark SM clock: 1500 MHz through the repository GPU lease.
- Environment caveat: per user authorization, runs used GPU 7 with 0%
  utilization but co-resident idle process memory (~18–43 GiB held by
  long-lived containers). Utilization was verified 0% at claim time for every
  timed/NCU run and the five-round dispersion is tight (see receipt).
- Launch floor (this run): 0.002369 ms. 1 GiB copy: 6319.4 GB/s.
- Modeled FP32 peak at 1500 MHz: 56.832 TFLOP/s.
- CuTe DSL 4.5.2; container TensorRT-LLM 1.3.0rc23, torch
  2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2.
- git-`HEAD` Triton SHA256:
  `68860408e57d076765d1e0be13cc87b9a3811ea9424de670d1c39821e5332432`.

### Calibrated attainable roofs (`results/r3_roof_calibration.json`)

The round-2 roof (1 GiB contiguous copy) was tested with minimal
same-byte-volume, same-access-pattern streaming copy kernels at the real
payloads and the 1500 MHz lock:

| Pattern | Calibrated roof | vs 1 GiB copy (6336 GB/s) |
|---|---:|---|
| contiguous T8192 D3072 (~100.7 MB) | 6956.6 GB/s | tighter — the roof was attainable and then some |
| contiguous T8192 D1536 (~50.4 MB) | 5813.5 GB/s | lower — smaller payload cannot sustain copy peak |
| strided T8192 D4608 rs4752 (~151 MB) | 5844.8 GB/s | lower — rows are 32B- but not 128B-aligned; every 9216 B row read spans partial 128 B DRAM lines at both edges |
| T1024 / T128 payloads | far below copy peak | launch/ramp dominated; the strict launch-floor bound is retained because raising a floor relaxes the gate |

## Round-3 variant campaign

Representative quick-matrix medians at 1500 MHz (dense D3072 T8192 B8 /
strided D4608 T8192 B1), each mechanism layered on the previous winner:

| Mechanism | dense long B8 | strided long B1 |
|---|---:|---:|
| round-2 selected (tile12, expdiv SiLU, lookahead) | 0.0455 ms | — |
| tanh SiLU (R3-I02) | 0.0382 ms | 0.0501 ms |
| best per-shape tuning of tile/prefetch/threads (R3-I05–I07) | 0.0353 ms | 0.0495 ms |
| group-batched loads, span 8, tile 24 (R3-I08/I09) | 0.0286 ms | 0.0408 ms |
| FP32 parameter fragments (R3-I10) | ±<2% (bitwise-identical outputs; compiler already hoists converts) | ±<2% |
| **FP32 ring, span 4, tile 16 (R3-I11, selected)** | **0.0279 ms** | **0.0399 ms** |

Refuted mechanisms with evidence in the ledger: 16/32-byte vectorization
(register/occupancy cost dominates, unlike the pure-copy calibration where
vec8 wins), L2 prefetch hints, span 12, FP32 ring at span 8 (register
pressure), 256-thread CTAs.

## Final gate matrix

Medians of five rounds, 50 warmups and 500 timed iterations per round,
1500 MHz. SOL fraction = calibrated defensible lower bound / measured.
Receipt: `results/r3_acceptance_summary.json`.

| D | T | B | layout | CuTe ms | Triton ms | speedup | SOL vs calibrated |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1536 | 128 | 1 | dense | 0.007480 | 0.028333 | 3.788x | 31.7% |
| 1536 | 128 | 8 | dense | 0.007515 | 0.028637 | 3.810x | 31.5% |
| 1536 | 1024 | 1 | dense | 0.007525 | 0.029423 | 3.910x | 31.5% |
| 1536 | 1024 | 8 | dense | 0.007469 | 0.029182 | 3.907x | 31.7% |
| 1536 | 8192 | 1 | dense | 0.014270 | 0.029157 | 2.043x | 60.7% |
| 1536 | 8192 | 8 | dense | 0.014532 | 0.030017 | 2.066x | 59.7% |
| 3072 | 128 | 1 | dense | 0.007353 | 0.029005 | 3.945x | 32.2% |
| 3072 | 128 | 8 | dense | 0.007434 | 0.029156 | 3.922x | 31.9% |
| 3072 | 1024 | 1 | dense | 0.007771 | 0.028989 | 3.730x | 30.5% |
| 3072 | 1024 | 8 | dense | 0.007459 | 0.029083 | 3.899x | 31.8% |
| 3072 | 8192 | 1 | dense | 0.027778 | 0.050785 | 1.828x | 52.1% |
| 3072 | 8192 | 8 | dense | 0.027900 | 0.058693 | 2.104x | 52.0% |
| 4608 | 128 | 1 | strided | 0.007324 | 0.028652 | 3.912x | 32.3% |
| 4608 | 128 | 8 | strided | 0.007395 | 0.029205 | 3.949x | 32.0% |
| 4608 | 1024 | 1 | strided | 0.007416 | 0.028922 | 3.900x | 40.5% |
| 4608 | 1024 | 8 | strided | 0.008010 | 0.028814 | 3.597x | 38.0% |
| 4608 | 8192 | 1 | strided | 0.039847 | 0.070683 | 1.774x | 64.9% |
| 4608 | 8192 | 8 | strided | 0.040031 | 0.082812 | 2.069x | 64.7% |

Short/medium SOL fractions are measured against the strict launch-floor
bound; those shapes are launch/ramp-bound at ~7.3–8.0 µs versus a 2.4–3.0 µs
isolated launch floor and beat Triton by ~3.6–3.9x.

## Correctness and precision

- Full matrix: 108/108 correctness rows pass vs unchanged native (dense 72
  incl. FP16/W2/W3 plus strided production variants); conv states bitwise
  exact. Case suite 17/17 (unscaled, adversarial extremes, short, padded,
  mixed initial state, permuted slots, no-bias/no-activation, strided).
- Precision audit (`results/r3_silu_precision_audit.json`): FP32 accumulation
  present in every variant; BF16 parameter fragments bitwise-identical to the
  FP32-parameter path; tile/threads/prefetch config choices bitwise-invariant;
  SiLU tanh form vs native worst max-abs 3.13e-2 (BF16 unscaled, atol gate
  1e-1) with abs error <= 4.9e-4 wherever the reference magnitude is below
  atol; expdiv form worst max-abs 1.95e-3. tanh was selected for speed
  (dense long 0.0353 vs 0.0457 ms at selection time) with this delta
  documented; expdiv remains selectable.

## Final NCU (selected config) and plateau evidence

NCU replay clocks ~2.0 GHz despite the benchmark lock, so NCU durations are
not substituted for 1500 MHz latencies. `results/ncu/ncu_sol_report.json` is
canonical.

| Profile | duration | compute SOL | DRAM SOL | regs | occ theory / achieved | bench vs calibrated floor |
|---|---:|---:|---:|---:|---:|---:|
| selected T8192 D3072 B1 | 24.5 µs | 47.6% | 31.8% | 72 | 43.8% / 38.4% | 1.92x |
| selected T8192 D3072 B8 | 25.2 µs | 48.1% | 31.0% | 72 | 43.8% / 37.6% | 1.92x |
| selected T8192 D4608s B1 | 33.9 µs | 53.4% | 42.6% | 64 | 50.0% / 43.7% | 1.54x |
| selected T8192 D4608s B8 | 34.4 µs | 54.4% | 41.8% | 64 | 50.0% / 44.2% | 1.55x |
| Triton T8192 D4608s B1 | 56.8 µs | 80.7% | 25.9% | 32 | 100% / 89.7% | 2.73x |
| Triton T8192 D4608s B8 | 67.5 µs | 73.9% | 22.0% | 32 | 100% / 89.7% | 3.20x |

Counter evolution across the round (dense long B1): DRAM SOL 20.5% →
22.8% (tanh/no-prefetch) → 31.4% (group loads) → 31.8% (selected); duration
38.7 → 34.7 → 24.4 → 24.5 µs.

Plateau evidence — where the residual 1.5–1.9x above the calibrated floor is
spent:

- **No unit is saturated.** Compute SOL 47.6–54.4%, DRAM SOL 31.0–42.6%; the
  busiest pipe is ALU at 37.0–38.6% (FMA 26–28%, XU/SFU 12.7–13.7%, LSU
  9.2–9.6%). The FP32 ring specifically cut ALU from 64–66% (span-8 profile)
  by removing repeated BF16→FP32 converts, and cycles-per-issue fell from
  ~14.2 to 10.3–11.5.
- **The dominant residual is DRAM latency under occupancy-limited hiding:**
  long-scoreboard stalls are 5.2–6.0 of the 10.3–11.5 cycles per issued
  instruction (~52%), even after group-batched loads quadrupled the bytes in
  flight per thread. Occupancy is capped by registers (72/64 per thread →
  43.8–50% theoretical, 37.6–44.2% achieved); every attempt to push more
  bytes in flight (span 8/12, FP32 ring at span 8, wider vectors) raised
  register pressure and regressed.
- **Mechanisms exhausted:** SiLU reformulation, vector widths 2–16, tiles
  4–32, CTA widths 64–256, L2 prefetch hints, software lookahead, group-load
  spans 4–12, FP32 parameter fragments, FP32 ring. The remaining gap would
  require latency-hiding machinery (cp.async/TMA-style staging through shared
  memory) that CuTe DSL's SIMT path would have to express without inflating
  the register budget that already binds — no untried counter-indicated
  mechanism remains within the current kernel architecture, so exploration
  stops per the acceptance update.

For reference, unchanged Triton on the strided production long shapes runs at
80.7%/73.9% compute SOL with only 25.9%/22.0% DRAM SOL — it is
instruction-bound and 1.77–2.07x slower.

## Compile, timing, and artifacts

- Full-matrix correctness wall time: 193.5 s; case suite 27.4 s; five-round
  main benchmark 38.3 s.
- Compilation, metadata preparation, and output allocation are outside
  timing; the timed body is exactly one GPU launch.

Canonical artifacts:

- `results/r3_acceptance_summary.json` — final per-shape gate verdicts/SOL
- `results/r3_selected_main_confidence.json` — five-round benchmark
- `results/r3_selected_full_correctness.json`, `results/r3_selected_case_correctness.json`
- `results/r3_silu_precision_audit.json` — precision audit
- `results/r3_roof_calibration.json` — calibrated roofs
- `results/ncu/ncu_sol_report.json`, `results/ncu/r3_selected_*.ncu-rep/.csv`,
  `results/ncu/r3_triton_T8192_D4608s_*.ncu-rep/.csv`

## Reproduction

From `/node-storage/CuTeDSLGen`, preserving the GPU lease and 1500 MHz lock
(the selected config is the default; no env vars needed):

```bash
# Correctness (case suite + full matrix) and five-round confidence benchmark
GPU_POOL=7 IDLE_GATE_MEM_MIB=60000 bash evaluation/decomposition_ab/gpu_run.sh -t 7200 -- \
  bash /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/r3_final_inner.sh

# Final NCU profiles + parse
CPID=$(docker inspect -f '{{.State.Pid}}' trt-dev)
GPU_POOL=7 IDLE_GATE_MEM_MIB=60000 bash evaluation/decomposition_ab/gpu_run.sh -t 5400 -- \
  env KDA_NCU_PROFILE_SET=round3-final \
  bash /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/ncu_inner_profiles.sh \
  "$CPID" /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill \
  /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/results/ncu
docker exec -w /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill trt-dev \
  python3 parse_ncu_results.py

# Acceptance receipt
python3 /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/build_r3_acceptance.py
```

## Limitations

- SOL fractions plateau at 52–65% of the calibrated roofs on long shapes; the
  documented blocker is DRAM latency with register-bound occupancy, and no
  counter-indicated mechanism remains inside the current SIMT architecture.
  A shared-memory staged (cp.async/TMA) redesign is the only identified path
  and is out of round-3 scope.
- Short shapes remain ~3x above the isolated launch floor (launch/ramp
  dominated) while beating Triton ~3.9x.
- Timing was collected on a shared-memory-resident device (utilization
  verified 0% before each run) per user authorization.
- The native extension binary was not rebuilt from the inspected checkout,
  although observed semantics match the source. Native remains authoritative
  for padded output because exact git-`HEAD` Triton leaves it uninitialized.

Decision: **Triton gate PASS 18/18 with the single selected configuration;
SOL reported best-effort with plateau evidence per the user acceptance
update.**
