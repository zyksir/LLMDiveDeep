# gather_quant — fused rank-block interleave + MXFP8 quantize

Date: 2026-08-19. Qualification: QUALIFIED (bit-exact gate vs the
unchanged trtllm op executed at every size; CUDA-graph timing, 1 GPU).
Prefix key: `b10_` = written in this repo, `trt_` = TensorRT-LLM op.

## Operation contract

The prefill fc1 gather returns rank-major bf16 blocks
`[world*B, 448]` (rank r's shard at rows `[r*B, (r+1)*B)`); the expert
path consumes `(e4m3 [B, 3584], ue8m0 sf u8 [B*112])` in the LINEAR
scale layout — exactly what `TRTLLMGenFusedMoE.quantize_input`
produces (it calls `mxfp8_quantize(is_sf_swizzled_layout=False)`).
448 = 14 x 32, so MXFP8 blocks never straddle rank boundaries.

## Implementations

- **trt_unfused** (previous path): interleave copy
  (`view.permute.reshape`, reads+writes bf16) then
  `torch.ops.trtllm.mxfp8_quantize` (reads bf16 again, writes fp8+sf).
- **b10_gather_quant** (`gather_quant.py`, one Triton launch): one
  read of the blocks, one write of the (half-size) fp8 payload + sf.
  Recipe = bit-exact replica of TRT's `cvt_warp_fp16_to_mxfp8`
  (per-32 block: sf = e8m0(amax/448) rounded UP saturating; value =
  e4m3(x * 2^-(sf-127)); amax==0 -> sf=0, values=0).

## Results (us, CUDA-graph medians; bold = winner; SOL measured)

SOL = the kernel's mandatory bytes (read B*3584 bf16 + write B*3584
fp8 + B*112 sf ~= 3.03 B*KB) at the 6.5 TB/s large-copy rate measured
on this GPU (torch.clone probe) — reachable, no launch floor added.

| B | trt_unfused | b10_gather_quant | saved | SOL | b10 x SOL |
|---:|---:|---:|---:|---:|---:|
| 512 | 8.3 | **3.2** | +5.1 | 0.9 | 3.6x (launch-bound) |
| 1024 | 13.1 | **5.3** | +7.7 | 1.7 | 3.1x |
| 2048 | 21.8 | **9.5** | +12.3 | 3.4 | 2.8x |
| 4096 | 41.0 | **18.0** | +23.0 | 6.8 | 2.6x |
| 8192 | 97.1 | **35.5** | +61.6 | 13.7 | 2.6x |
| 16384 | 188.5 | **69.8** | +118.7 | 27.3 | 2.6x |

Correctness: payload AND scales bit-exact vs the trtllm op at every
size (gate in the bench).

Why b10 wins: the unfused path moves the bf16 payload three times
(interleave read+write, quantize read) and the fused path once; the
2.6x residual over the bytes bound is the strided block-gather read
pattern (rank-major rows are B apart) and the small per-CTA tile —
acceptable, since the kernel sits behind the overlapped gather and
saves 12-119 us net in the layer.

Integration: `b10_kimi_k3_moe_layer._prefill_sharded` calls
`gather_quant_mxfp8(gathered, world)` and feeds the (fp8, sf) tuple
straight to the experts (skipping `quantize_input`). Layer-level
effect included in the shipping table (`agent/moe_optimization.md` §1).

Lesson recorded: the first bit-exact gate FAILED because the oracle
was called as `mxfp8_quantize(x, 32)` — the 32 landed in
`is_sf_swizzled_layout` (truthy) and produced the 128x4-swizzled,
128-row-padded sf layout. The consumer's actual call is
`(x, False, alignment=32)` = linear. Match the CALLER, not the op's
defaults.

## Reproduce (~60 s)

```bash
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/kernels/bench_gather_quant.py
```
