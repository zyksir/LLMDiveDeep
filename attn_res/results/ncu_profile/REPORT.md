# SGLang Kimi-K3 `attn_res_fused_tma` — ncu Analysis Report

**Date**: 2026-07-29  
**Platform**: NVIDIA B200 × 1 (SM100a, 148 SMs, 183 GB HBM3e)  
**Driver**: 590.48 · CUDA CC: 10.0  
**Profiler**: ncu 2025.1.1 (`--set full`, 39 passes per kernel)  
**Python env**: `.venv` under `LLMDiveDeep/`  
**SGLang source**: `sglang-opensource/python/sglang/kernels/ops/kimi_k3/attn_res.py`  
**CUDA kernel**: `sglang-opensource/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`  

---

## Kernel Identity

```
void attn_res_fused_tma_kernel<KimiK3AttnResTrait<7168, 4, 5, 200>, 1>(AttnResTMAParams)
```

Template instantiation for **nvb=4**:  
`KimiK3AttnResTrait<kDim=7168, kNumBankRows=4, kChunkRows=5, kConsumerRegs=200>`  
Occupancy=1 CTA/SM (from `_TMA_BEST_CONFIG[4] = (5, 1, 200)`).

### Architecture
| Parameter | Value |
|-----------|-------|
| Warps/CTA | 12 (8 consumer + 4 producer) |
| Threads/CTA | 384 |
| Consumer warp groups | 2 × 4 warps |
| kChunkRows (nvb=4) | 5 (exactly 1 chunk = (4+1+5-1)/5 = 1) |
| kNumStages (ring slots) | 2 (double-buffered) |
| kNumChunks per token | 1 — the 5 source rows fit one chunk |
| Producer copies rows | via `cp.async.bulk` (TMA) → smem, bypassing L1/L2 |
| cw / ow | loaded once to TMEM per CTA at startup |
| PDL | enabled; `PDLTriggerSecondary` after all tokens |
| Smem per CTA | **143.87 KB** dynamic + 1.02 KB driver = 144.89 KB |
| Registers/thread | **168** |
| SM register budget | 168 × 384 = 64,512 / 65,536 max → 1 block/SM |
| Theoretical occupancy | **18.75%** (1 CTA / SM, 12 warps / 64 max) |

---

## Hardware Reference

| Spec | Value | Source |
|------|-------|--------|
| B200 HBM3e peak bandwidth | **8.0 TB/s** | NVIDIA spec |
| ncu-implied achievable BW | **~6.65 TB/s** | DRAM SOL% = measured_BW / this |
| SM count | 148 | `cuda.get_device_properties(0).multi_processor_count` |
| Max SM clock | 1.97 GHz | nvidia-smi |
| Observed SM clock (ncu) | 1.08–1.13 GHz | SM Frequency in profiles |
| DRAM clock | 3.97–4.00 GHz | ncu |

---

## Algorithmic Bytes (one-pass contract, nvb=4, H=7168, BF16)

For every token, the kernel reads (prefix + 4 bank rows) and writes 1 output row:

```
Read  = rows × T × H × 2 bytes = 5 × T × 14,336 B = T × 71,680 B
Write = 1    × T × H × 2 bytes = 1 × T × 14,336 B = T × 14,336 B
────────────────────────────────────────────────────────────────
One-pass total = 6 × T × 14,336 B = T × 85,016 B
```

Per-CTA weight overhead (cw + ow loaded once per CTA, amortized):
```
cw + ow = 2 × 7,168 × 2 = 28,672 bytes per CTA
148 CTAs × 28,672 = 4.24 MB  (negligible at large T, ~1.2% at T=16,384)
```

| T | One-pass read (MB) | Write (MB) | Total one-pass (MB) | +weights (MB) |
|---|---|---|---|---|
| 1 | 0.0716 | 0.0143 | 0.0859 | 0.115 |
| 256 | 18.35 | 3.67 | 22.02 | 26.26 |
| 4,096 | 293.6 | 58.72 | 352.3 | 356.6 |
| 16,384 | 1,174.4 | 234.9 | 1,409.3 | 1,413.5 |

---

## Benchmark Timing (steady-state, CUDA-graph + eager)

```
python attn_res/bench_attn_res.py --impl sglang_tma --nvb 4
  --tokens 1 256 4096 16384 --warmup 10 --iters 30 --repeats 5
```

| T | Aggregation (µs) | Decoder layer (µs) | Bench ideal_GBps |
|---|---|---|---|
| 1 | **3.09** | 6.19 | 27.8 GB/s |
| 256 | **5.91** | 11.81 | 3,729 GB/s |
| 4,096 | **56.28** | 112.56 | **6,260 GB/s** |
| 16,384 | **213.27** | 426.53 | **6,608 GB/s** |

> Note: ncu single-kernel duration is **~1.9–4× higher** than benchmark (9–12 µs PDL cold-start
> penalty: `PDLWaitPrimary` stalls waiting for no prior kernel signal in single-shot profiling,
> plus the absence of steady-state cache warmth from CUDA-graph replay).
> All bandwidth and SOL analysis uses **benchmark timing** as the authoritative duration.

---

## Raw ncu Metrics (per-size)

### T=1 (grid 1×384)

| Metric | Value |
|--------|-------|
| ncu Duration | 12.64 µs |
| Steady-state Duration | **3.09 µs** |
| DRAM read (ncu) | 135.17 KB |
| DRAM write (ncu) | 0 bytes (output cached in L2) |
| DRAM SOL% (ncu) | 0.16% (1-CTA, 1-SM workload) |
| L1/TEX Hit Rate | **82.35%** (cw/ow cached after warmup) |
| L2 Hit Rate | 45.91% |
| Compute SOL% | 0.13% |
| SM Active Cycles | 56.42 |
| Theoretical Occupancy | 18.75% |
| Achieved Occupancy | **13.48%** |
| Registers/thread | 168 |
| Dynamic Smem/block | 143.87 KB |
| Top stall: long_scoreboard | 1.49 ratio |
| Top stall: wait (named barrier) | 1.36 ratio |
| Top stall: no_instruction | 0.79 ratio |
| Grid | **(1, 1, 1)** × 384 — only 1 SM used |

### T=256 (grid 148×384)

| Metric | Value |
|--------|-------|
| ncu Duration | 18.43 µs |
| Steady-state Duration | **5.91 µs** |
| DRAM read (ncu) | 18.41 MB |
| DRAM write (ncu) | 0 bytes (output ≤14 KB/CTA, fully buffered in L2) |
| DRAM BW (ncu) | 999 GB/s |
| DRAM SOL% (ncu) | **15.07%** |
| L2 Hit Rate | 3.90% |
| L1/TEX Hit Rate | 78.96% |
| Compute SOL% | 18.42% |
| Theoretical Occupancy | 18.75% |
| Achieved Occupancy | **13.76%** |
| Top stall: wait (named barrier) | 1.23 ratio |
| Top stall: long_scoreboard | 1.48 ratio |
| Top stall: short_scoreboard | 0.82 ratio |

### T=4,096 (grid 148×384)

| Metric | Value |
|--------|-------|
| ncu Duration | 109.31 µs |
| Steady-state Duration | **56.28 µs** |
| DRAM read (ncu) | **293.68 MB** |
| DRAM write (ncu) | **43.59 MB** |
| DRAM total (ncu) | **337.27 MB** |
| DRAM BW (ncu) | **3.09 TB/s** |
| DRAM SOL% (ncu) | **46.41%** |
| L2 Hit Rate | **0.27%** (nearly all reads miss to HBM) |
| L1/TEX Hit Rate | 32.05% |
| L2 Cache Throughput% | 39.53% |
| Compute SOL% | 41.60% |
| FMA pipeline utilization | **36.2%** (highest pipeline, not bottleneck) |
| Theoretical Occupancy | 18.75% |
| Achieved Occupancy | **14.00%** |
| Top stall: wait (named barrier) | **0.99 ratio** |
| Top stall: short_scoreboard | 0.65 ratio |
| Top stall: long_scoreboard | 0.64 ratio |
| Top stall: not_selected | 0.40 ratio |
| Top stall: math_pipe_throttle | 0.37 ratio |
| Top stall: barrier (mbarrier) | 0.26 ratio |
| Top stall: dispatch_stall | 0.21 ratio |
| Excessive L2 sectors (uncoalesced) | 928,256 (32% of 2,895,872 total) |

### T=16,384 (grid 148×384)

| Metric | Value |
|--------|-------|
| ncu Duration | 397.02 µs |
| Steady-state Duration | **213.27 µs** |
| DRAM read (ncu) | **1.17 GB** |
| DRAM write (ncu) | **219.32 MB** |
| DRAM total | **~1.393 GB** |
| DRAM BW (ncu) | **3.51 TB/s** |
| DRAM SOL% (ncu) | **52.78%** |
| L2 Hit Rate | **0.07%** |
| L1/TEX Hit Rate | 11.05% |
| Compute SOL% | 44.58% |
| FMA pipeline utilization | **37.6%** |
| Theoretical Occupancy | 18.75% |
| Achieved Occupancy | **14.04%** |
| Top stall: wait (named barrier) | **0.97 ratio** |
| Top stall: short_scoreboard | 0.64 ratio |
| Top stall: long_scoreboard | 0.59 ratio |
| Top stall: not_selected | 0.40 ratio |
| Top stall: math_pipe_throttle | 0.37 ratio |
| Excessive L2 sectors (uncoalesced) | 928,256 (11% of 8,400,896 total) |

---

## Bandwidth & SOL Summary

> B200 peak: 8.0 TB/s. Achievable (ncu-implied): ~6.65 TB/s.  
> Benchmark BW = ncu DRAM bytes / steady-state duration.

| T | DRAM bytes (ncu) | Steady-state (µs) | Achieved BW | SOL vs 8 TB/s | SOL vs achievable |
|---|---|---|---|---|---|
| 1 | 135.17 KB | 3.09 | 43.7 GB/s (1 SM) | ~68% (1 SM) | — |
| 256 | ~18.4 MB | 5.91 | **3.11 TB/s** | **38.9%** | ~47% |
| 4,096 | 337.27 MB | 56.28 | **5.99 TB/s** | **74.9%** | **90.1%** |
| 16,384 | ~1,393 MB | 213.27 | **6.53 TB/s** | **81.6%** | **~98%** |

### Defensible Copy-Bandwidth Roofline

Minimum time for a pure `memcpy` kernel reading/writing the algorithmic bytes:

| T | Algo bytes | @8 TB/s (µs) | @6.65 TB/s (µs) | Actual (µs) | SOL (vs 8) | SOL (vs 6.65) |
|---|---|---|---|---|---|---|
| 256 | 26.26 MB | 3.28 | 3.95 | 5.91 | **55.5%** | **66.8%** |
| 4,096 | 356.6 MB | 44.6 | 53.6 | 56.28 | **79.2%** | **95.3%** |
| 16,384 | 1,413.5 MB | 176.7 | 212.6 | 213.27 | **82.8%** | **99.7%** |

---

## Findings

### 1. Warp Stall Breakdown (T=4,096 / T=16,384 — production sizes)

The dominant stall is **`wait` (named barrier)** at ratio ≈1.0 per issued instruction. Named barriers in this kernel are:

```cpp
::ptx::named_barrier_sync(kConsumerBarId, kNumConsumerThreads);  // 2× per chunk
```

The two named barriers per token (cross-warp RMS/dot reduction; cross-warp ssq reduction for output norm) require all 8 consumer warps (256 threads) to rendezvous. With only 12 warps/SM and 4 SMSPs/SM, each SMSP has 3 warps active (2 consumer + 1 producer). The consumer warps that arrive early at the barrier stall until stragglers finish — manifesting as `wait` stalls.

| Stall | T=4096 | T=16384 | Interpretation |
|-------|--------|---------|----------------|
| wait (named_bar) | **0.99** | **0.97** | Cross-warp reduction rendezvous |
| short_scoreboard | 0.65 | 0.64 | Dep stall within warp (fp32 chain latency) |
| long_scoreboard | 0.64 | 0.59 | L2/DRAM latency (cw/ow global loads) |
| not_selected | 0.40 | 0.40 | Scheduler choosing another warp |
| math_pipe_throttle | 0.37 | 0.37 | FMA pipeline backpressure |
| barrier (mbarrier) | 0.26 | — | TMA bulk-copy arrival wait |
| dispatch_stall | 0.21 | 0.21 | Instruction decode/dispatch |

### 2. Memory Access Pattern

- **Source rows (bank + prefix)**: Loaded via `cp.async.bulk` (TMA), bypassing L1/L2 directly into smem. L2 hit rate 0.07–0.27% confirms these never enter the cache hierarchy — exactly one DRAM read per row per token.
- **cw / ow weights**: Loaded to TMEM once per CTA at startup via normal global loads → L1 (82% hit at T=1, decreasing as more CTAs compete; still 32% at large T for warm cache).
- **Output stores**: Regular global stores → L1 → L2 → DRAM. Write bytes lag algorithmic expectation (e.g., 43.59 MB vs 58.72 MB at T=4096) because L2 write-buffers partially coalesce dirty lines.

### 3. Shared Memory Bank Conflicts

The kernel exhibits modest shared-memory bank conflicts (summed across all CTAs):

| T | Total conflicts | Load conflicts | Store conflicts |
|---|---|---|---|
| 256 | 3,544 | 499 | 646 |
| 4,096 | 91,850 | 11,367 | 28,985 |
| 16,384 | 369,010 | 46,017 | 107,777 |

These are the `warp_rms[8][5]` / `warp_dot[8][5]` partial-sum accumulation writes (stride-1 float → 8 warps × 5 rows = lanes 0–7 write row 0, lanes 0–7 write row 1, etc.). Bank conflicts are a minor contributor relative to the named-barrier stall.

### 4. Occupancy Limiter

Register count (168/thread × 384 threads = 64,512) and smem (143.87 KB) both independently cap occupancy at 1 CTA/SM (18.75% theoretical). The deliberate choice `consumer_regs=200` (setmaxnreg) trades peak occupancy for a larger consumer register file, enabling the packed fp32×2 accumulator loop without register spills. This is the right trade-off: the kernel is not occupancy-bound at large T, as DRAM bandwidth is the floor.

### 5. Uncoalesced Global Accesses

928,256 excessive sectors are constant across all T values — these are the **cw/ow TMEM stores** (`tcgen05_st_32x32b_x8`) which ncu classifies as non-optimal global accesses. The 4/32 bytes-per-sector utilization is an artifact of the TMEM store path; the actual data movement is correct and efficient (weights are loaded once per CTA). This is not an actionable bottleneck.

---

## Speed-of-Light Judgment

| T | Bottleneck | SOL vs algo roofline (6.65 TB/s) | At SOL? | Max plausible speedup |
|---|---|---|---|---|
| **1** | Kernel latency (named-bar + PDL cold-start) | 68% (single SM) | No | **~1.3×** from barrier reduction |
| **256** | Latency-bound (small tokens/CTA, long barriers relative to work) | ~67% (vs 6.65 TB/s roofline) | No | **~1.5×** (more tokens needed per CTA) |
| **4,096** | Named-barrier sync overhead | **~95% vs algo roofline** | **Near-SOL** | **~1.05×** (diminishing returns) |
| **16,384** | — | **~99.7% vs algo roofline** | **YES, at SOL** | **<1.02×** (noise-level headroom) |

**Summary**: At T≥4096, `attn_res_fused_tma` runs at 90–98% of achievable HBM3e bandwidth (74–82% of 8 TB/s theoretical). At T=16,384 it is effectively **at speed-of-light** for a single-pass HBM-streaming kernel on B200. Smaller T values are latency-bound, not bandwidth-bound.

---

## Recommendations

### R1 — Reduce Named Barrier Serialization (T=1..256, medium impact)
The two `named_barrier_sync(kConsumerBarId, 256)` calls per chunk (one for RMS/dot reduction, one for ssq reduction) are the primary stall source. Options:
- Use warp-level shuffle + warp-leader atomic for the reduction instead of CTA-wide named barriers (removes both barriers; replaces with 2× `__shfl_xor_sync` trees and 2× `atomicAdd` into smem).
- Reduces `wait` stall from 1.0 → ~0.3; estimated gain ~30–40% at T=1–256.

### R2 — Consider Dual-CTA Mode at T=256 (B200-specific)
At T=256, each SM processes only 1–2 tokens before exiting. If occupancy=2 were feasible (requires halving register use below 168/thread, reducing smem below 72 KB each), two CTAs could co-reside, doubling token throughput per SM. The current 143.87 KB smem precludes this. A 4-row chunk (kChunkRows=4) reduces smem to ~115 KB; still tight but combined with consumer_regs≤112 could enable occupancy=2.

### R3 — Accept Current Config for T≥4096 (no action needed)
The kernel is within 5% of the bandwidth roofline at T=4096 and at the roofline at T=16384. No code change will yield meaningful gains without changing the algorithmic contract (e.g., compute-in-register reuse).

### R4 — Monitor PDL Chain Health in Production
The 4× ncu inflation for T=1 confirms that **PDL chaining is critical** for decode-serving latency. An unintentional PDL break (e.g., kernel replaced by non-PDL variant, or inserted `torch.cuda.synchronize()`) would quadruple decode latency for T=1. Add a CI assertion that PDL is enabled for SM100+ dispatch.

---

## Artifact Paths

| File | Description |
|------|-------------|
| `attn_res/results/ncu_profile/attn_res_tma_T1.ncu-rep` | Full ncu report, T=1 |
| `attn_res/results/ncu_profile/attn_res_tma_T256.ncu-rep` | Full ncu report, T=256 |
| `attn_res/results/ncu_profile/attn_res_tma_T4096.ncu-rep` | Full ncu report, T=4096 |
| `attn_res/results/ncu_profile/attn_res_tma_T16384.ncu-rep` | Full ncu report, T=16384 |
| `attn_res/results/ncu_profile/bench_tma_nvb4.csv` | Benchmark CSV (all T, nvb=4) |
| `attn_res/run_ncu_target.py` | ncu profiling harness |
