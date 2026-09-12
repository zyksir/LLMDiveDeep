/*
 * Thin standalone launcher for the UNCHANGED TensorRT-LLM fused Kimi-K3
 * routing front (moe::dev::routing::routingCustom::run), compiled against the
 * FlashInfer-bundled sources. The Data configuration mirrors the production
 * DeepSeekV3 no-groups branch in trtllm_fused_moe_runner.cu (SigmoidBias
 * preprocess, ScaledSumNormalize postprocess, sum epsilon 1e-20f).
 *
 * This file is an adapter only; it contains no kernel code.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "tvm_ffi_utils.h"

namespace {

inline int32_t computeLog2(int32_t val) {
  int32_t n = val;
  int32_t out = 0;
  while (n >>= 1) {
    ++out;
  }
  return out;
}

void routing_custom_from_scores(
    // inputs
    int64_t scores_ptr,  // [num_tokens, num_experts] fp32 or bf16 raw logits
    int64_t bias_ptr,    // [num_experts] fp32
    // outputs
    int64_t topk_weights_ptr,  // [num_tokens, top_k] fp32 or bf16
    int64_t topk_packed_ptr,   // [num_tokens, top_k] PackedScoreIdx output (score, expert idx)
    int64_t expert_counts_ptr,  // [2 * num_experts] int32 scratch
    int64_t permuted_idx_size_ptr,
    int64_t expanded_idx_to_permuted_idx_ptr,
    int64_t permuted_idx_to_expanded_idx_ptr,
    int64_t permuted_idx_to_token_idx_ptr,
    int64_t cta_idx_to_expert_idx_ptr,
    int64_t cta_idx_to_mn_limit_ptr,
    int64_t num_non_exiting_ctas_ptr,
    // metadata
    int32_t num_tokens, int32_t num_experts, int32_t top_k, int32_t tile_tokens_dim,
    int32_t local_expert_offset, int32_t num_local_experts, double route_scale,
    bool input_is_fp32, bool output_is_fp32, bool use_pdl, int64_t cuda_stream_ptr) {
  namespace tg = batchedGemm::trtllm::gen;
  moe::dev::routing::routingCustom::Data data;

  data.mDtypeInput = input_is_fp32 ? tg::Dtype::Fp32 : tg::Dtype::Bfloat16;
  data.mDtypeOutput = output_is_fp32 ? tg::Dtype::Fp32 : tg::Dtype::Bfloat16;
  data.mUsePdl = use_pdl;
  data.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
  data.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
  data.mPtrRoutingBias = reinterpret_cast<void const*>(bias_ptr);
  data.mDtypeBias = tg::Dtype::Fp32;
  data.mRouteScale = static_cast<float>(route_scale);
  data.mSumEpsilon = 1e-20f;

  data.mPtrScores = reinterpret_cast<void const*>(scores_ptr);
  data.mPtrTopKIds = nullptr;
  data.mPtrTopKPacked = reinterpret_cast<void*>(topk_packed_ptr);
  data.mPtrExpertCounts = reinterpret_cast<int32_t*>(expert_counts_ptr);
  data.mPtrPermutedIdxSize = reinterpret_cast<int32_t*>(permuted_idx_size_ptr);
  data.mPtrExpandedIdxToPermutedIdx =
      reinterpret_cast<int32_t*>(expanded_idx_to_permuted_idx_ptr);
  data.mPtrPermutedIdxToExpandedIdx =
      reinterpret_cast<int32_t*>(permuted_idx_to_expanded_idx_ptr);
  data.mPtrPermutedIdxToTokenIdx =
      permuted_idx_to_token_idx_ptr != 0
          ? reinterpret_cast<int32_t*>(permuted_idx_to_token_idx_ptr)
          : nullptr;
  data.mPtrTopKWeights = reinterpret_cast<void*>(topk_weights_ptr);

  data.mPtrCtaIdxXyToBatchIdx = reinterpret_cast<int32_t*>(cta_idx_to_expert_idx_ptr);
  data.mPtrCtaIdxXyToMnLimit = reinterpret_cast<int32_t*>(cta_idx_to_mn_limit_ptr);
  data.mPtrNumNonExitingCtas = reinterpret_cast<int32_t*>(num_non_exiting_ctas_ptr);

  data.mNumTokens = num_tokens;
  data.mNumExperts = num_experts;
  data.mTopK = top_k;
  data.mPaddingLog2 = computeLog2(tile_tokens_dim);
  data.mTileTokensDim = tile_tokens_dim;
  data.mLocalExpertsStartIdx = local_expert_offset;
  data.mLocalExpertsStrideLog2 = 0;
  data.mNumLocalExperts = num_local_experts;

  cudaStream_t stream = cuda_stream_ptr != 0
                            ? reinterpret_cast<cudaStream_t>(cuda_stream_ptr)
                            : get_current_stream();

  moe::dev::routing::routingCustom::run(data, stream);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(llmdd_routing_custom_from_scores, routing_custom_from_scores);
