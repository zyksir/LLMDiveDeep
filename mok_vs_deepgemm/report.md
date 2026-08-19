# MoK versus DeepGEMM BF16 full-forward report

Date: 2026-08-18  
Qualification: **UNQUALIFIED pairwise result**

## Operation contract

- Meaning: one distributed DSV3-style MoE forward pass.
- Shape: EP4, 2,048 tokens/rank, 256 routed experts, one shared expert, H=6,144, I=2,048, top-k=8.
- Dtypes and math: BF16 activations, weights, intermediates, and output; unclamped SwiGLU; FP32 router weights.
- Timed boundary: input/routing copies, global or in-kernel scheduling, pull dispatch, shared and routed expert FC1/SwiGLU/FC2, push combine, router-weight reduction, and final BF16 output.
- Hardware: GPUs 4–7, NVIDIA B200 SM100, all-to-all NV18 connectivity.
- Launch mode: eager CUDA events; each sample is the maximum elapsed time across four ranks; 500 warmups and 100 measured samples.
- Excluded: backward, model integration, CUDA graph replay, MXFP8/MXFP4, SiTU, Kimi-K3 TP4/B8 latent semantics, and kernel modifications.

## Ecosystem survey

- Mixture-of-Kittens commit `22fc95ae6e331a738c4a58a227a8b03cac586e12`, Apache-2.0:
  `mok.functional.build_schedule` plus `mok.functional.forward`, ending in
  `dispatch_mlp_swiglu_combine_fwd_bf16` and `fwd_epilogue`.
- DeepGEMM commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`, MIT:
  `deep_gemm.bf16_mega_moe`, ending in `sm100_bf16_mega_moe_impl`.
- SGLang and TensorRT-LLM currently wrap DeepGEMM Mega MoE for quantized serving; they are not independent BF16 full-forward kernels for this contract.
- vLLM's current BF16 MoE path is grouped-GEMM based rather than the distributed BF16 Mega MoE used here.
- FlashInfer now exposes an independent BF16 CuTeDSL Mega MoE path. It was outside the user-confirmed MoK-versus-DeepGEMM pairwise scope, so this report cannot claim the ecosystem's fastest implementation.

## Correctness

All outputs were finite.

- DeepGEMM versus MoK: max abs 0.015625, relative L2 0.002105, cosine 0.9999978.
- MoK versus the unchanged MoK PyTorch reference: max abs 0.03125, relative L2 0.004104, cosine 0.9999916.
- DeepGEMM versus the same reference: max abs 0.03125, relative L2 0.004131, cosine 0.9999915.

## Result

- Tuned MoK (`fwd_num_comm_sms=16`, `minibatch_size=2048`): median 2.2469 ms, p20 2.2287 ms, p80 2.2881 ms, 619.3 TFLOP/s/rank.
- DeepGEMM BF16 Mega MoE: median 1.8464 ms, p20 1.8091 ms, p80 1.8756 ms, 753.7 TFLOP/s/rank.
- DeepGEMM is 1.217x faster by inverse latency. MoK is 21.7% slower, or DeepGEMM has 17.8% lower latency.
- Decision: **STOP for this unchanged EP4 BF16 comparison. MoK does not beat DeepGEMM.**

## Idea analysis

- Pull dispatch plus push combine is not a transferable advantage: both kernels already use it.
- Ring-buffered token staging is also shared: both recycle routed-token storage.
- MoK's distinct ideas are a deterministic reusable global schedule, dedicated communication SMs, medium-grained minibatches sized for multiple GEMM waves, CLC work stealing, and a training backward megakernel.
- The documented communication-SM/minibatch sweep selected 16 communication SMs and 2,048 routed tokens per minibatch, but the tuned result remained behind DeepGEMM.
- CLC is primarily valuable when the megakernel must yield to higher-priority inter-rack RDMA. This isolated single-node forward benchmark does not exercise that benefit.
- MoK's deterministic schedule and backward support may still win for EP64 training on GB300 NVL72, which is its intended regime. That does not imply a win for EP4 inference.

## Qualification limits

- MoK's forward writes a backward context; DeepGEMM's inference-oriented BF16 Mega MoE does not. MoK therefore performs extra writes.
- DeepGEMM applies router weights in the BF16 SwiGLU/FC2 path, while MoK performs the weighted routed reduction after FC2. The real-number math is equivalent, but BF16 rounding order differs.
- No measured launch, HBM, NVLink, or synchronization speed-of-light probes were run.
- The newly available FlashInfer BF16 Mega MoE baseline was not executed.
- These gaps make the numerical result a direct, reproducible pairwise measurement, not a qualified SOTA claim.

## Kimi-K3 relevance

MoK cannot directly test the prior Kimi-K3 TP4/B8 latent contract. Its public
workspace requires at least 512 tokens/rank and dimensions divisible by 256;
the released kernels use BF16 or MXFP8 routed weights, SwiGLU, token-owner
dispatch/combine, and a full output. The Kimi target uses B=8, H=3,584,
I=384, MXFP8 activations with MXFP4 weights, SiTU, and rank-local partial
output semantics.

## Reproduction artifacts

- Structured final result: `results/final_stable.json`
- MoK tuning sweep: `results/final.json`
- Stable runner: `run.py`
- Backend adapters: `mok_backend.py`, `deepgemm_backend.py`
- Cold AOT build: MoK 164.4 s; DeepGEMM 36.2 s.
- Cached full correctness and stable latency command: 21.1 s wall time.

No upstream kernel source was modified.
