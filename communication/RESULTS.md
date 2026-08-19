# Communication kernels — results (TP8, B200, dim 7168, bf16)

Date: 2026-08-19. **CUDA-graph timed** (median of graph replays, MAX across
ranks; host launch bubbles excluded per USER-RULES 17) via
`communication/bench_comm_graph.py`; source CSVs
`local_result/report_graph_<op>.csv`. Bold = fastest backend in the row;
`n/a (capacity)` = size-gated by the backend itself (e.g. quantized AR needs
>=8192 elements). Every backend of every op graph-captures cleanly — zero
capture failures.

**SOL** is measured-derived and reachable: max(best measured latency at the
smallest size, bus bytes / the op's own peak measured bus bandwidth).

Note on `lookup.csv`: `bench_comm.py`'s event-window numbers include host
overhead (3-8x these values) and exist only for the autotune map; they are
not report material. The CuTeDSL fused kernels (gemm_allreduce,
allreduce_norm[_gemm]) JIT-compile at first use — run
`communication/kernel_benchmarks/prebuild_cutedsl.py` once (9 s warm,
minutes cold); that was the "slow compile".

**What each backend actually does**

- `torch_symm:multimem` / `b10_multimem`: NVLS multimem — the switch does
  the reduction/broadcast; one multimem load+store per element per rank.
- `torch_symm:1shot/2shot`, `flashinfer:1shot/2shot`: peer-to-peer symmetric
  memory; 1shot = every rank reads all peers (lowest latency, (w-1)x
  traffic), 2shot = reduce-scatter + all-gather (1/4 the wire, one more
  phase).
- `trt`: TRT-LLM custom all-reduce, MIN_LATENCY mode — same symmetric-memory
  family, with an internal one-shot/two-shot switch by payload.
- `nccl` / `nccl_symm`: ring/NVLS NCCL; high latency floor, fine asymptote.
- `b10_copy_engine(:sm)`: rotated-schedule DMA (or SM-copy) movers — pure
  data movement on copy engines, zero SM occupancy.
- `vllm_int8/fp8`: lossy quantized-wire two-shot AR (separate numerics).
- `cutedsl`: this repo's fused GEMM+AR / AR+norm CuTeDSL kernels.
- `seq`: the fused op composed from the autotuned winners of its stages.


## all_reduce

Peak measured bus bandwidth 712 GB/s; latency floor 5.1 us.

| B | b10_multimem | flashinfer:1shot | torch_symm:multimem | trt | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.6 | 5.4 | 11.5 | **5.1** | 5.1 | 1.0x |
| 2 | 11.7 | 5.5 | 11.7 | **5.2** | 5.1 | 1.0x |
| 4 | 11.8 | 5.7 | 11.7 | **5.4** | 5.1 | 1.1x |
| 8 | 11.9 | 6.1 | 12.0 | **5.7** | 5.1 | 1.1x |
| 16 | 12.0 | 7.2 | 13.3 | **6.8** | 5.1 | 1.3x |
| 32 | 12.5 | 9.8 | 13.5 | **9.3** | 5.1 | 1.8x |
| 64 | **13.3** | 15.7 | 14.0 | 14.9 | 5.1 | 2.6x |
| 128 | **15.1** | 27.1 | 16.6 | 25.6 | 5.1 | 3.0x |
| 256 | **19.1** | 49.9 | 22.0 | 36.9 | 9.0 | 2.1x |
| 512 | 32.0 | 98.0 | **31.4** | 50.9 | 18.0 | 1.7x |
| 1024 | 50.2 | 190.9 | **49.7** | 80.2 | 36.1 | 1.4x |
| 2048 | **83.3** | 377.6 | 86.0 | 127.3 | 72.2 | 1.2x |
| 4096 | **160.9** | 753.5 | 176.4 | 233.7 | 144.4 | 1.1x |
| 8192 | **302.8** | 1507.2 | 351.2 | 363.6 | 288.8 | 1.0x |
| 16384 | **577.6** | 3018.6 | 689.5 | 635.2 | 577.6 | 1.0x |

Why the winner wins: bs<=64 is latency-bound — trt's one-shot has the
shortest handshake (5.1 us floor; flashinfer:1shot is 5-6% behind at the
same algorithm: its Lamport protocol must clear sentinel values after every
call and the fusion kernel carries residual/norm branches). From bs~128
NVLS multimem wins: the switch performs the reduction, so wire bytes drop
~4x vs one-shot. At the largest sizes b10_multimem edges torch's multimem
(577.6 vs 689.5 us at 16384 — its rotated store schedule keeps more links
saturated), and trt stays within reach only because MIN_LATENCY internally
switches to two-shot.

## all_gather

Peak measured bus bandwidth 720 GB/s; latency floor 9.5 us.
(Aug-19 re-measure: rows 4096-16384 previously read "n/a (capacity)"
for every non-NCCL backend — root-caused to a bench probe-shape bug:
the capacity check received the gathered OUTPUT shape and multiplied
by world again. The real operand fits the symm pool at every size;
no backend has a genuine capacity limit at this dim.)

| B | b10_copy_engine (dma) | b10_copy_engine:sm | nccl | nccl_symm | torch_low_contention | torch_symm:multimem | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.2 | 13.0 | 20.8 | 21.1 | 55.2 | **9.5** | 9.5 | 1.0x |
| 2 | 48.4 | 13.0 | 20.6 | 21.1 | 56.7 | **10.7** | 9.5 | 1.1x |
| 4 | 48.7 | 13.0 | 21.5 | 22.0 | 58.7 | **10.8** | 9.5 | 1.1x |
| 8 | 48.5 | 13.3 | 21.8 | 22.2 | 58.9 | **11.0** | 9.5 | 1.2x |
| 16 | 48.9 | 14.2 | 23.9 | 24.3 | 60.4 | **12.1** | 9.5 | 1.3x |
| 32 | 50.1 | 16.5 | 30.1 | 30.5 | 61.7 | **14.6** | 9.5 | 1.5x |
| 64 | 59.4 | 21.2 | 42.7 | 43.1 | 70.8 | **19.8** | 9.5 | 2.1x |
| 128 | 70.5 | **30.2** | 48.4 | 48.9 | 81.4 | 30.4 | 17.8 | 1.7x |
| 256 | 93.7 | **48.7** | 67.3 | 69.2 | 104.4 | 51.1 | 35.7 | 1.4x |
| 512 | 135.2 | **87.7** | 133.9 | 134.6 | 137.1 | 94.3 | 71.3 | 1.2x |
| 1024 | 218.3 | **161.7** | 193.5 | 186.2 | 219.6 | 180.2 | 142.7 | 1.1x |
| 2048 | 377.3 | **310.3** | 352.4 | 335.2 | 357.5 | 346.8 | 285.3 | 1.1x |
| 4096 | 641.3 | **618.1** | 690.1 | 640.7 | 676.8 | 687.5 | 570.6 | 1.1x |
| 8192 | **1196.9** | 1287.8 | 1329.7 | 1242.9 | 1266.1 | 1363.0 | 1141.3 | 1.0x |
| 16384 | **2282.6** | 2578.4 | 2579.3 | 2446.3 | 2464.5 | 2702.4 | 2282.6 | 1.0x |

Why the winner wins: B<=64 — torch_symm's multimem gather has the
lowest handshake (the switch broadcasts; one store per element).
B=128..4096 — the b10 SM-copy mover's fused rotated-copy kernel keeps
every link saturated with no protocol phases. B>=8192 — the pure
copy-engine (DMA) mover takes over at 1.0x SOL: cudaMemcpyAsync peer
copies stream at full link rate with ZERO SM occupancy, while the SM
kernel's issue rate becomes the limiter at the largest messages
(310 -> 618 -> 1288 us: sm scales worse than 2x per doubling from 4k).
torch_low_contention is never the best row — its schedule pays a
~46 us fixed setup that only amortizes where the b10 movers already
win. NCCL's ring pays protocol overhead per hop everywhere.

## all_gather overlapped with an independent GEMM (Aug-19)

Kimi prefill question: while the fc1 gather runs, the independent gate
+ shared-up GEMMs ([B,7168]x[7168,2432] bf16) should compute for free.
Per backend, three graph-timed measurements (MAX over ranks,
`bench_ag_gemm_overlap.py`, ~3 min): `ag` = the AG alone;
`gemm chain` = N back-to-back copies of the GEMM alone, with N chosen
as round(ag / one_gemm) so the compute leg is DELIBERATELY sized to
match the AG leg (that is why the two columns are close — by
construction); `overlapped` = the same chain concurrent with the AG
(AG on a side stream). With two equal legs, perfect overlap makes
`overlapped` ~= either leg alone; every us above that is
serialization/contention. eff = fraction of the shorter leg hidden
(1.0 = free); dilate = overlapped minus ag.

| B | backend | ag | gemm chain | overlapped | eff | dilate |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | **b10_copy_engine (dma)** | 226.6 | 170.8 | **229.7** | **0.87** | +3.1 |
| 1024 | b10_copy_engine:sm | 177.2 | 147.0 | 278.8 | -3.15 | +101.6 |
| 1024 | torch_symm:multimem | 191.5 | 148.2 | 209.4 | 0.27 | +18.0 |
| 1024 | nccl | 201.5 | 170.8 | 282.6 | -2.32 | +81.1 |
| 2048 | **b10_copy_engine (dma)** | 388.2 | 324.8 | **390.7** | **0.94** | +2.6 |
| 2048 | b10_copy_engine:sm | 323.7 | 278.4 | 520.0 | -3.23 | +196.3 |
| 2048 | torch_symm:multimem | 358.0 | 324.1 | 387.3 | 0.37 | +29.3 |
| 2048 | nccl | 364.4 | 324.1 | 479.0 | -1.47 | +114.6 |
| 4096 | **b10_copy_engine (dma)** | 661.4 | 690.2 | **785.4** | -0.26 | +124.0 |
| 4096 | b10_copy_engine:sm | 629.6 | 591.0 | 1112.3 | -3.90 | +482.7 |
| 4096 | torch_symm:multimem | 695.7 | 693.7 | 828.9 | -0.34 | +133.2 |
| 4096 | nccl | 695.2 | 697.2 | 904.2 | -1.10 | +209.0 |
| 8192 | **b10_copy_engine (dma)** | 1220.1 | 1200.6 | **1368.8** | 0.26 | +148.7 |
| 8192 | b10_copy_engine:sm | 1296.8 | 1397.9 | 2522.8 | -5.14 | +1226.0 |
| 8192 | torch_symm:multimem | 1370.7 | 1395.8 | 1610.9 | -0.20 | +240.2 |
| 8192 | nccl | 1393.9 | 1393.0 | 1858.9 | -1.34 | +465.1 |

Why: the DMA mover issues cudaMemcpyAsync peer copies on copy engines
— ZERO SM occupancy — so at 1024-2048 the GEMM chain rides under the
AG nearly free (+2.6-3.1 us). The SM-copy mover is the WORST overlap
partner everywhere (its rotated-copy kernel and the GEMM fight for
SMs and finish slower than running serially — eff -3..-5); NCCL's
SM-resident ring is similar in kind. multimem overlaps partially (the
switch reduces, but the kernel still holds SMs for loads/stores).
From B=4096 even the DMA mover shows +124-149 us dilation — the copy
engines' local HBM reads/writes start competing with the GEMM's
weight streaming for DRAM bandwidth (both legs near their bandwidth
roofs); attacking that contention (rate-limiting the engines, or
splitting the AG across the GEMM's tail) is the open item.

Consequence for the layer: any overlapped fc1-shard gather must run
on the DMA mover, never the SM variant; at 1-2k tokens the gather is
then effectively free next to gate/shared compute.

## reduce_scatter

Peak measured bus bandwidth 658 GB/s; latency floor 18.2 us.

| B | b10_copy_engine | b10_copy_engine:sm | nccl | nccl_symm | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.8 | **18.2** | 20.4 | 21.2 | 18.2 | 1.0x |
| 2 | 50.0 | **18.3** | 20.4 | 20.8 | 18.2 | 1.0x |
| 4 | 50.0 | **18.2** | 21.3 | 21.9 | 18.2 | 1.0x |
| 8 | 50.6 | **18.4** | 21.6 | 22.5 | 18.2 | 1.0x |
| 16 | 51.1 | **18.6** | 23.5 | 24.6 | 18.2 | 1.0x |
| 32 | 59.0 | **21.2** | 29.4 | 31.8 | 18.2 | 1.2x |
| 64 | 62.7 | **25.9** | 41.1 | 46.1 | 18.2 | 1.4x |
| 128 | 80.8 | **37.1** | 53.4 | 58.3 | 19.5 | 1.9x |
| 256 | 108.0 | **59.6** | 75.5 | 85.2 | 39.1 | 1.5x |
| 512 | 152.8 | **111.1** | 133.4 | 153.8 | 78.1 | 1.4x |
| 1024 | 260.9 | 206.4 | **189.4** | 225.5 | 156.2 | 1.2x |
| 2048 | 449.9 | 392.3 | **356.0** | 428.1 | 312.5 | 1.1x |
| 4096 | 796.8 | 774.3 | **694.6** | 837.1 | 625.0 | 1.1x |
| 8192 | 1493.7 | 1578.8 | **1304.2** | 1590.1 | 1250.0 | 1.0x |
| 16384 | 2863.7 | 3183.8 | **2500.0** | 3064.3 | 2500.0 | 1.0x |

Why the winner wins: like all_gather plus a reduction — the SM-copy mover
variant wins large sizes because the reduction needs arithmetic the pure DMA
engine cannot do; NCCL competes only at the smallest sizes.

## all_to_all

Peak measured bus bandwidth 717 GB/s; latency floor 14.8 us.

| B | b10_copy_engine | b10_copy_engine:sm | nccl | nccl_symm | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.5 | 14.8 | **14.8** | 15.7 | 14.8 | 1.0x |
| 2 | 48.4 | **14.3** | 18.5 | 21.8 | 14.8 | 1.0x |
| 4 | 48.8 | **14.3** | 19.9 | 22.5 | 14.8 | 1.0x |
| 8 | 48.8 | **14.4** | 21.4 | 24.1 | 14.8 | 1.0x |
| 16 | 48.7 | **14.4** | 22.7 | 25.3 | 14.8 | 1.0x |
| 32 | 49.8 | **16.5** | 27.2 | 28.8 | 14.8 | 1.1x |
| 64 | 59.8 | **21.2** | 34.5 | 34.4 | 14.8 | 1.4x |
| 128 | 70.5 | **30.3** | 49.2 | 45.3 | 17.9 | 1.7x |
| 256 | 92.2 | **49.1** | 77.9 | 68.6 | 35.8 | 1.4x |
| 512 | 140.8 | **90.9** | 123.0 | 124.8 | 71.6 | 1.3x |
| 1024 | 218.2 | **165.7** | 213.5 | 218.9 | 143.3 | 1.2x |
| 2048 | 372.6 | **314.9** | 389.0 | 406.0 | 286.5 | 1.1x |
| 4096 | 650.4 | **624.2** | 740.9 | 809.6 | 573.0 | 1.1x |
| 8192 | **1200.7** | 1297.1 | 1364.4 | 1506.3 | 1146.1 | 1.0x |
| 16384 | **2292.1** | 2604.4 | 2718.0 | 3064.6 | 2292.1 | 1.0x |

Why the winner wins: pure permutation traffic; the rotated DMA schedule
sends each hop exactly once at link rate. No arithmetic anywhere, so the SM
variant offers nothing extra here.

## quantized_all_reduce (lossy wire; separate numerics)

Peak measured bus bandwidth 317 GB/s; latency floor 17.5 us.

| B | vllm_fp8 | vllm_int8 | SOL | best x SOL |
|---:|---:|---:|---:|---:|
| 2 | **17.5** | 20.1 | 17.5 | 1.0x |
| 4 | **19.0** | 21.5 | 17.5 | 1.1x |
| 8 | **19.5** | 21.9 | 17.5 | 1.1x |
| 16 | **21.8** | 24.3 | 17.5 | 1.2x |
| 32 | **21.8** | 24.2 | 17.5 | 1.2x |
| 64 | **22.3** | 24.5 | 17.5 | 1.3x |
| 128 | **23.6** | 25.2 | 17.5 | 1.3x |
| 256 | **30.7** | 34.8 | 17.5 | 1.8x |
| 512 | **36.8** | 39.9 | 20.3 | 1.8x |
| 1024 | **48.5** | 51.9 | 40.6 | 1.2x |
| 2048 | **81.1** | 87.3 | 81.1 | 1.0x |
| 4096 | **216.3** | 222.2 | 162.3 | 1.3x |
| 8192 | **408.3** | 421.6 | 324.5 | 1.3x |
| 16384 | **784.3** | 812.3 | 649.0 | 1.2x |

Why the winner wins: int8 vs fp8 wire trade encode cost against identical
byte counts; fp8's cheaper encode wins where the kernel is quantize-bound.
Lossy — never compare into the lossless tables.

## allreduce_norm (rmsnorm(allreduce(x)+residual))

Peak measured bus bandwidth 649 GB/s; latency floor 6.5 us.

| B | b10_multimem | cutedsl | flashinfer:1shot | seq | trt | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.5 | 56.5 | 6.9 | 7.9 | **6.5** | 6.5 | 1.0x |
| 2 | 38.2 | 56.5 | 7.0 | 8.3 | **6.6** | 6.5 | 1.0x |
| 4 | 38.6 | 54.4 | 7.0 | 8.4 | **6.6** | 6.5 | 1.0x |
| 8 | 39.4 | 55.1 | 7.5 | 9.0 | **7.0** | 6.5 | 1.1x |
| 16 | 43.9 | 58.7 | 8.6 | 10.0 | **8.2** | 6.5 | 1.3x |
| 32 | 49.1 | 80.2 | 11.4 | 12.6 | **10.7** | 6.5 | 1.7x |
| 64 | 49.0 | 63.5 | 17.0 | 17.4 | **16.0** | 6.5 | 2.5x |
| 128 | 50.1 | 64.7 | 30.8 | **20.3** | 26.7 | 6.5 | 3.1x |
| 256 | 52.1 | 67.1 | 60.1 | **26.7** | 46.9 | 9.9 | 2.7x |
| 512 | 55.9 | 165.7 | 118.6 | **37.8** | 63.4 | 19.8 | 1.9x |
| 1024 | 65.7 | 75.7 | 235.2 | **59.9** | 88.5 | 39.6 | 1.5x |
| 2048 | **103.0** | 106.2 | 467.2 | 111.8 | 146.4 | 79.1 | 1.3x |
| 4096 | 198.7 | **194.2** | 934.9 | 204.7 | 266.5 | 158.2 | 1.2x |
| 8192 | 345.6 | **344.9** | 1866.0 | 388.5 | 440.1 | 316.5 | 1.1x |
| 16384 | **633.0** | 637.8 | 3740.6 | 747.2 | 787.8 | 633.0 | 1.0x |

Why the winner wins: small bs — the fused trt/flashinfer kernel saves the
separate norm launch; large bs — seq (multimem AR + separate norm) wins
because the fused one-shot kernels move (w-1)x wire while multimem moves ~1x;
the b10 cutedsl fused kernel wins the mid-range where it combines multimem
traffic with the fused epilogue.

## gemm_allreduce (allreduce(x @ w.T))

Peak measured bus bandwidth 769 GB/s; latency floor 8.0 us.

| B | cutedsl | seq | SOL | best x SOL |
|---:|---:|---:|---:|---:|
| 1 | 24.6 | **8.0** | 8.0 | 1.0x |
| 2 | 24.4 | **8.0** | 8.0 | 1.0x |
| 4 | 24.5 | **8.0** | 8.0 | 1.0x |
| 8 | 24.4 | **8.7** | 8.0 | 1.1x |
| 16 | 24.6 | **12.2** | 8.0 | 1.5x |
| 32 | 24.5 | **14.0** | 8.0 | 1.8x |
| 64 | 24.5 | **17.9** | 8.0 | 2.2x |
| 128 | 23.7 | **21.1** | 8.0 | 2.7x |
| 256 | **26.3** | 27.5 | 8.3 | 3.1x |
| 512 | 182.3 | **39.7** | 16.7 | 2.4x |
| 1024 | **56.5** | 62.7 | 33.4 | 1.7x |
| 2048 | **88.3** | 106.0 | 66.8 | 1.3x |
| 4096 | **157.5** | 195.2 | 133.6 | 1.2x |
| 8192 | **283.5** | 365.7 | 267.2 | 1.1x |
| 16384 | **534.3** | 704.3 | 534.3 | 1.0x |

Why the winner wins: the cutedsl fused GEMM+AR overlaps the epilogue's
communication with the GEMM mainloop (the AR starts while tiles finish);
seq must finish the whole GEMM before the AR starts. The fusion wins
everywhere its shape support applies; JIT-compile is cached by the prebuild
script.

## allreduce_norm_gemm (rmsnorm(allreduce(x)+res) @ w.T)

Peak measured bus bandwidth 266 GB/s; latency floor 19.8 us.

| B | norm_gemm_seq | seq | SOL | best x SOL |
|---:|---:|---:|---:|---:|
| 1 | **19.8** | 21.8 | 19.8 | 1.0x |
| 2 | **19.3** | 21.5 | 19.8 | 1.0x |
| 4 | **19.8** | 22.2 | 19.8 | 1.0x |
| 8 | **19.6** | 21.8 | 19.8 | 1.0x |
| 16 | **23.0** | 25.3 | 19.8 | 1.2x |
| 32 | **29.3** | 30.7 | 19.8 | 1.5x |
| 64 | 35.4 | **29.4** | 19.8 | 1.5x |
| 128 | 36.0 | **35.7** | 19.8 | 1.8x |
| 256 | 51.5 | **50.2** | 24.2 | 2.1x |
| 512 | **69.2** | 69.5 | 48.4 | 1.4x |
| 1024 | 116.2 | **115.7** | 96.7 | 1.2x |
| 2048 | **212.9** | 219.1 | 193.4 | 1.1x |
| 4096 | **416.2** | 428.3 | 386.9 | 1.1x |
| 8192 | **785.5** | 842.1 | 773.7 | 1.0x |
| 16384 | **1547.5** | 1673.9 | 1547.5 | 1.0x |

Why the winner wins: only sequential compositions exist (the three-stage
fusion is excluded by design); `norm_gemm_seq` (AR, then fused norm+GEMM)
beats plain seq by removing the intermediate norm round-trip. A fully fused
kernel is a candidate idea; its bound is this table's SOL column.

## Reproduce (graph-timed; ~20-45 s per op)

```bash
python3 communication/kernel_benchmarks/prebuild_cutedsl.py   # once, 9 s warm
mpirun --allow-run-as-root -np 8 python3 communication/bench_comm_graph.py \
    --ops allreduce --bs 1..16k     # repeat per op name
```
