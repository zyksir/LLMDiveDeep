// LLMDiveDeep candidate: fused Kimi-K3 decode routing+permutation, B <= 128.
//
// Hypothesis: at decode the region is launch/dependency bound. The best
// unfused open-source composition pays two kernel launches, a global IDs
// round-trip, and (in TRT moe_sort) a multi-phase cluster pipeline. One CTA
// can route every token with an SGLang-style byte-radix top-16 done at warp
// scope (28 keys/lane), then build the histogram/scan/maps in shared memory,
// eliminating one launch and all intermediate global traffic.
//
// Exact semantics: FP32 sigmoid (accurate expf, no fast-math), selection by
// score+bias descending, ties by lower expert ID, weights = score/sum.
// Emits ids, weights, e2p, p2e, p2t (-1 padding), tile maps, sizes.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int kExperts = 896;
constexpr int kTopK = 16;
constexpr int kLocalStart = 224;
constexpr int kNumLocal = 112;
constexpr int kTile = 128;
// 512 threads = 128 registers/thread: the 28-key/lane selection state must
// stay in registers (1024 threads caps at 64 regs and spills to local).
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kPerLane = kExperts / 32;  // 28
constexpr int kMaxB = 128;

__device__ __forceinline__ uint32_t orderKey(float f) {
  // Monotonic float->uint transform: larger float => larger uint.
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int warpSumInt(int v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warpSumFloat(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ int warpMinInt(int v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__global__ void __launch_bounds__(kThreads) fusedRoutePermuteKernel(
    const float* __restrict__ logits, int64_t logitsStride,
    const float* __restrict__ bias, int batch,
    int* __restrict__ outIds, float* __restrict__ outWeights,
    int* __restrict__ e2p, int* __restrict__ p2e, int* __restrict__ p2t,
    int* __restrict__ tileExpert, int* __restrict__ tileMnLimit,
    int* __restrict__ paddedSize, int* __restrict__ numTiles,
    int ablate) {  // 0=full, 1=keys only, 2=+radix, 3=+ties, 4=+emit (no epilogue)
  __shared__ uint32_t sHist[kWarps][256];
  __shared__ float sBias[kExperts];
  __shared__ int sIds[kMaxB * kTopK];
  __shared__ int sTokCnt[kMaxB];
  __shared__ int sCnt[kNumLocal];
  __shared__ int sBase[kNumLocal];
  __shared__ int sOff[kNumLocal];
  __shared__ int sPadded, sTiles;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int e = tid; e < kExperts; e += kThreads) sBias[e] = bias[e];
  for (int t = tid; t < batch; t += kThreads) sTokCnt[t] = 0;
  for (int e = tid; e < kNumLocal; e += kThreads) { sCnt[e] = 0; sOff[e] = 0; }
  __syncthreads();

  // ---- route: one token per warp per round -------------------------------
  for (int token = warp; token < batch; token += kWarps) {
    const float* row = logits + token * logitsStride;
    uint32_t keys[kPerLane];
    uint32_t active = (1u << kPerLane) - 1u;
    uint32_t win = 0;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      int e = j * 32 + lane;
      float score = 1.0f / (1.0f + expf(-row[e]));
      keys[j] = orderKey(score + sBias[e]);
    }
    if (ablate == 1) {  // keep the computation observable
      if (lane == 0) sTokCnt[token] = (int)keys[0];
      continue;
    }
    int needed = kTopK;
    for (int byte = 3; byte >= 0 && needed > 0; --byte) {
#pragma unroll
      for (int k = 0; k < 8; ++k) sHist[warp][lane * 8 + k] = 0;
      __syncwarp();
#pragma unroll
      for (int j = 0; j < kPerLane; ++j)
        if (active >> j & 1u)
          atomicAdd(&sHist[warp][(keys[j] >> (byte * 8)) & 0xFF], 1u);
      __syncwarp();
      int laneSum = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) laneSum += sHist[warp][lane * 8 + k];
      int inclusive = laneSum;  // suffix over lanes >= lane
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        int v = __shfl_down_sync(0xffffffffu, inclusive, off);
        if (lane + off < 32) inclusive += v;
      }
      int suffixHi = inclusive - laneSum;  // bins owned by lanes > lane
      int running = suffixHi, tBin = -1, gtAtT = 0;
      for (int k = 7; k >= 0; --k) {
        int c = (int)sHist[warp][lane * 8 + k];
        if (running < needed && running + c >= needed) {
          tBin = lane * 8 + k;
          gtAtT = running;
        }
        running += c;
      }
      unsigned found = __ballot_sync(0xffffffffu, tBin >= 0);
      int src = __ffs(found) - 1;
      tBin = __shfl_sync(0xffffffffu, tBin, src);
      gtAtT = __shfl_sync(0xffffffffu, gtAtT, src);
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        if (!(active >> j & 1u)) continue;
        int bv = (keys[j] >> (byte * 8)) & 0xFF;
        if (bv > tBin) {
          win |= 1u << j;
          active &= ~(1u << j);
        } else if (bv < tBin) {
          active &= ~(1u << j);
        }
      }
      needed -= gtAtT;
      int totActive = warpSumInt(__popc(active));
      if (totActive == needed) {
        win |= active;
        active = 0;
        needed = 0;
      }
    }
    if (ablate == 2) {
      if (lane == 0) sTokCnt[token] = (int)win;
      continue;
    }
    while (needed > 0) {  // full-key ties: lower expert ID wins
      int myId = INT_MAX, myJ = -1;
#pragma unroll
      for (int j = kPerLane - 1; j >= 0; --j)
        if (active >> j & 1u) { myId = j * 32 + lane; myJ = j; }
      int mn = warpMinInt(myId);
      if (myId == mn && myJ >= 0) {
        win |= 1u << myJ;
        active &= ~(1u << myJ);
      }
      --needed;
    }
    if (ablate == 3) {
      if (lane == 0) sTokCnt[token] = (int)win;
      continue;
    }
#pragma unroll
    for (int j = 0; j < kPerLane; ++j)
      if (win >> j & 1u) {
        int pos = atomicAdd(&sTokCnt[token], 1);
        sIds[token * kTopK + pos] = j * 32 + lane;
      }
    __syncwarp();
    int id = -1;
    float score = 0.0f;
    if (lane < kTopK) {
      id = sIds[token * kTopK + lane];
      score = 1.0f / (1.0f + expf(-row[id]));
    }
    float sum = warpSumFloat(score);
    if (lane < kTopK) {
      outIds[token * kTopK + lane] = id;
      outWeights[token * kTopK + lane] = score / sum;
    }
  }
  __syncthreads();
  if (ablate != 0) return;

  // ---- permute: histogram -> scan/tiles -> fill -> scatter ---------------
  const int expanded = batch * kTopK;
  for (int i = tid; i < expanded; i += kThreads) {
    int e = sIds[i] - kLocalStart;
    if (0 <= e && e < kNumLocal) atomicAdd(&sCnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int base = 0, tiles = 0;
    for (int e = 0; e < kNumLocal; ++e) {
      int c = sCnt[e];
      sBase[e] = base;
      if (c > 0) {
        int nt = (c + kTile - 1) / kTile;
        for (int k = 0; k < nt; ++k) {
          tileExpert[tiles + k] = e;
          tileMnLimit[tiles + k] = min(base + (k + 1) * kTile, base + c);
        }
        tiles += nt;
        base += nt * kTile;
      }
    }
    sPadded = base;
    sTiles = tiles;
    *paddedSize = base;
    *numTiles = tiles;
  }
  __syncthreads();
  for (int p = tid; p < sPadded; p += kThreads) {
    p2e[p] = -1;
    p2t[p] = -1;
  }
  __syncthreads();
  for (int i = tid; i < expanded; i += kThreads) {
    int e = sIds[i] - kLocalStart;
    if (0 <= e && e < kNumLocal) {
      int pos = sBase[e] + atomicAdd(&sOff[e], 1);
      e2p[i] = pos;
      p2e[pos] = i;
      p2t[pos] = i / kTopK;
    } else {
      e2p[i] = -1;
    }
  }
}

// ---- candidate standalone permutation (from IDs) ---------------------------
//
// Hypothesis: TRT moe_sort pays a multi-phase cluster pipeline (3-6 us) and
// the fork kernel ~2.4 us for work whose mandatory traffic is ~KBs. One
// 512-thread CTA with a shared histogram, one warp-level two-quantity
// exclusive scan (token bases + tile bases), and a parallel scatter should
// approach the launch floor.

constexpr int kPermThreads = 512;

template <bool kPdl>
__device__ __forceinline__ void permuteBody(
    const int* __restrict__ ids, int batch,
    int* __restrict__ e2p, int* __restrict__ p2e, int* __restrict__ p2t,
    int* __restrict__ tileExpert, int* __restrict__ tileMnLimit,
    int* __restrict__ paddedSize, int* __restrict__ numTiles) {
  __shared__ int sCnt[kNumLocal];
  __shared__ int sBase[kNumLocal];
  __shared__ int sTileBase[kNumLocal];
  __shared__ int sOff[kNumLocal];
  __shared__ int sPadded, sTiles;
  const int tid = threadIdx.x;
  const int nt_ = blockDim.x;
  const int expanded = batch * kTopK;
  for (int e = tid; e < kNumLocal; e += nt_) {
    sCnt[e] = 0;
    sOff[e] = 0;
  }
#if __CUDA_ARCH__ >= 900
  if (kPdl) {  // producer results needed from here on
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
  __syncthreads();
  for (int i = tid; i < expanded; i += nt_) {
    int e = ids[i] - kLocalStart;
    if (0 <= e && e < kNumLocal) atomicAdd(&sCnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {  // exclusive scan of padded-token and tile counts (4 experts/lane)
    int tokPre[4], tilePre[4], tokSum = 0, tileSum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int e = tid * 4 + k;
      int c = e < kNumLocal ? sCnt[e] : 0;
      int nt = (c + kTile - 1) / kTile;
      tokPre[k] = tokSum;
      tilePre[k] = tileSum;
      tokSum += nt * kTile;
      tileSum += nt;
    }
    int tokExcl = tokSum, tileExcl = tileSum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int a = __shfl_up_sync(0xffffffffu, tokExcl, off);
      int b = __shfl_up_sync(0xffffffffu, tileExcl, off);
      if (tid >= off) {
        tokExcl += a;
        tileExcl += b;
      }
    }
    tokExcl -= tokSum;
    tileExcl -= tileSum;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int e = tid * 4 + k;
      if (e < kNumLocal) {
        sBase[e] = tokExcl + tokPre[k];
        sTileBase[e] = tileExcl + tilePre[k];
      }
    }
    if (tid == 31) {
      sPadded = tokExcl + tokSum;
      sTiles = tileExcl + tileSum;
      *paddedSize = sPadded;
      *numTiles = sTiles;
    }
  }
  __syncthreads();
  for (int e = tid; e < kNumLocal; e += nt_) {
    int c = sCnt[e];
    int nt = (c + kTile - 1) / kTile;
    int tb = sTileBase[e], base = sBase[e];
    for (int k = 0; k < nt; ++k) {
      tileExpert[tb + k] = e;
      tileMnLimit[tb + k] = min(base + (k + 1) * kTile, base + c);
    }
  }
  // int4 -1 fill was tried and measured ~0.2 us slower at every decode B;
  // plain int stores win here. Only permuted_to_expanded padding must be -1
  // (TRT ABI documents permuted_to_token padding as uninitialized).
  for (int p = tid; p < sPadded; p += nt_) p2e[p] = -1;
  __syncthreads();
  for (int i = tid; i < expanded; i += nt_) {
    int e = ids[i] - kLocalStart;
    if (0 <= e && e < kNumLocal) {
      int pos = sBase[e] + atomicAdd(&sOff[e], 1);
      e2p[i] = pos;
      p2e[pos] = i;
      p2t[pos] = i / kTopK;
    } else {
      e2p[i] = -1;
    }
  }
}

template <bool kPdl>
__global__ void __launch_bounds__(kPermThreads) permuteFromIdsKernel(
    const int* __restrict__ ids, int batch,
    int* __restrict__ e2p, int* __restrict__ p2e, int* __restrict__ p2t,
    int* __restrict__ tileExpert, int* __restrict__ tileMnLimit,
    int* __restrict__ paddedSize, int* __restrict__ numTiles) {
  permuteBody<kPdl>(ids, batch, e2p, p2e, p2t, tileExpert, tileMnLimit,
                    paddedSize, numTiles);
}

// ---- candidate fused v2: route CTAs + last-CTA epilogue --------------------
//
// Hypothesis: the accepted radix+permute composition still exposes the
// permutation's launch and ids round-trip. One kernel with B route CTAs
// (block-scope byte-radix top-16, 4 keys/lane over 256 threads — no per-warp
// issue bottleneck) where the last-arriving CTA runs the permutation
// epilogue removes the second launch entirely while keeping route
// parallelism.

constexpr int kV2Threads = 256;
constexpr int kV2PerLane = 4;  // ceil(896/256)

__global__ void __launch_bounds__(kV2Threads) fusedV2Kernel(
    const float* __restrict__ logits, int64_t logitsStride,
    const float* __restrict__ bias, int batch, int* __restrict__ counter,
    int* __restrict__ outIds, float* __restrict__ outWeights,
    int* __restrict__ e2p, int* __restrict__ p2e, int* __restrict__ p2t,
    int* __restrict__ tileExpert, int* __restrict__ tileMnLimit,
    int* __restrict__ paddedSize, int* __restrict__ numTiles,
    int ablate) {  // 0=full, 1=keys, 2=+passes, 3=+ties/emit, 4=no epilogue
  __shared__ uint32_t sHist2[256];
  __shared__ int sTBin, sGt, sActive, sWinCnt, sMinId, sIsLast;
  __shared__ int sWin[kTopK];
  const int tid = threadIdx.x;
  const int token = blockIdx.x;
  const float* row = logits + token * logitsStride;

  uint32_t keys[kV2PerLane];
  uint32_t active = 0, win = 0;
#pragma unroll
  for (int j = 0; j < kV2PerLane; ++j) {
    int e = tid + kV2Threads * j;
    if (e < kExperts) {
      float score = 1.0f / (1.0f + expf(-row[e]));
      keys[j] = orderKey(score + bias[e]);
      active |= 1u << j;
    }
  }
  if (ablate == 1) {
    if (tid == 0) outIds[token * kTopK] = (int)keys[0];
    return;
  }
  int needed = kTopK;
  for (int byte = 3; byte >= 0 && needed > 0; --byte) {
    sHist2[tid] = 0;
    if (tid == 0) sActive = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kV2PerLane; ++j)
      if (active >> j & 1u)
        atomicAdd(&sHist2[(keys[j] >> (byte * 8)) & 0xFF], 1u);
    __syncthreads();
    if (tid < 32) {  // suffix-scan the 256 bins, 8 per lane
      int laneSum = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) laneSum += sHist2[tid * 8 + k];
      int inclusive = laneSum;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        int v = __shfl_down_sync(0xffffffffu, inclusive, off);
        if (tid + off < 32) inclusive += v;
      }
      int running = inclusive - laneSum;
      int tBin = -1, gtAtT = 0;
      for (int k = 7; k >= 0; --k) {
        int c = (int)sHist2[tid * 8 + k];
        if (running < needed && running + c >= needed) {
          tBin = tid * 8 + k;
          gtAtT = running;
        }
        running += c;
      }
      if (tBin >= 0) {
        sTBin = tBin;
        sGt = gtAtT;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kV2PerLane; ++j) {
      if (!(active >> j & 1u)) continue;
      int bv = (keys[j] >> (byte * 8)) & 0xFF;
      if (bv > sTBin) {
        win |= 1u << j;
        active &= ~(1u << j);
      } else if (bv < sTBin) {
        active &= ~(1u << j);
      }
    }
    needed -= sGt;
    if (__popc(active)) atomicAdd(&sActive, __popc(active));
    __syncthreads();
    if (sActive == needed) {
      win |= active;
      active = 0;
      needed = 0;
    }
  }
  if (ablate == 2) {
    if (tid == 0) outIds[token * kTopK] = (int)win;
    return;
  }
  while (needed > 0) {  // full-key ties: lower expert ID wins
    if (tid == 0) sMinId = INT_MAX;
    __syncthreads();
    int myMin = INT_MAX, myJ = -1;
#pragma unroll
    for (int j = kV2PerLane - 1; j >= 0; --j)
      if (active >> j & 1u) { myMin = tid + kV2Threads * j; myJ = j; }
    if (myMin != INT_MAX) atomicMin(&sMinId, myMin);
    __syncthreads();
    if (myJ >= 0 && myMin == sMinId) {
      win |= 1u << myJ;
      active &= ~(1u << myJ);
    }
    --needed;
    __syncthreads();
  }
  if (tid == 0) sWinCnt = 0;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kV2PerLane; ++j)
    if (win >> j & 1u) sWin[atomicAdd(&sWinCnt, 1)] = tid + kV2Threads * j;
  __syncthreads();
  if (tid < 32) {
    int id = -1;
    float score = 0.0f;
    if (tid < kTopK) {
      id = sWin[tid];
      score = 1.0f / (1.0f + expf(-row[id]));
    }
    float sum = warpSumFloat(score);
    if (tid < kTopK) {
      outIds[token * kTopK + tid] = id;
      outWeights[token * kTopK + tid] = score / sum;
    }
  }

  if (ablate >= 3) return;

  // last CTA runs the permutation over all tokens' ids
  __threadfence();
  __syncthreads();
  if (tid == 0) sIsLast = (atomicAdd(counter, 1) == batch - 1);
  __syncthreads();
  if (!sIsLast) return;
  __threadfence();
  permuteBody<false>(outIds, batch, e2p, p2e, p2t, tileExpert, tileMnLimit,
                     paddedSize, numTiles);
  __syncthreads();
  if (tid == 0) *counter = 0;  // graph-replay safe
}

// ---- SOL probes -----------------------------------------------------------
//
// Practical-bound probes: the mandatory work any implementation of each stage
// must perform, with minimal coordination. Their measured times are the
// defensible per-shape lower bounds for the tables (the strict launch-floor /
// bytes bound is kept separately and is deliberately looser).

__global__ void __launch_bounds__(kV2Threads) probeRouteWorkKernel(
    const float* __restrict__ logits, int64_t logitsStride,
    const float* __restrict__ bias, float* __restrict__ out) {
  __shared__ float sRed[kV2Threads / 32];
  const int tid = threadIdx.x;
  const float* row = logits + blockIdx.x * logitsStride;
  float m = -1e30f;
#pragma unroll
  for (int j = 0; j < kV2PerLane; ++j) {
    int e = tid + kV2Threads * j;
    if (e < kExperts) m = fmaxf(m, 1.0f / (1.0f + expf(-row[e])) + bias[e]);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if ((tid & 31) == 0) sRed[tid >> 5] = m;
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int w = 1; w < kV2Threads / 32; ++w) m = fmaxf(m, sRed[w]);
    out[blockIdx.x] = m;
  }
}

__global__ void __launch_bounds__(kPermThreads) probePermuteWorkKernel(
    const int* __restrict__ ids, int expanded, int* __restrict__ out) {
  __shared__ int h[kNumLocal];
  __shared__ int sTotal;
  const int tid = threadIdx.x;
  for (int e = tid; e < kNumLocal; e += kPermThreads) h[e] = 0;
  __syncthreads();
  for (int i = tid; i < expanded; i += kPermThreads) {
    int e = ids[i] - kLocalStart;
    if (0 <= e && e < kNumLocal) atomicAdd(&h[e], 1);
  }
  __syncthreads();
  if (tid < 32) {
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int e = tid * 4 + k;
      if (e < kNumLocal) sum += ((h[e] + kTile - 1) / kTile) * kTile;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (tid == 0) sTotal = sum;
  }
  __syncthreads();  // stands in for the fill barrier
  __syncthreads();  // stands in for the scatter barrier
  if (tid == 0) *out = sTotal;
}

__global__ void __launch_bounds__(kThreads) probePhasesKernel(int phases, int* out) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  for (int i = 0; i < phases; ++i) {
    if ((threadIdx.x & 31) == 0) atomicAdd(&s, 1);
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = s;
}

__global__ void probeStreamKernel(const float4* __restrict__ in, int64_t n4,
                                  float* __restrict__ out) {
  float acc = 0.0f;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n4;
       i += (int64_t)gridDim.x * blockDim.x) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 16);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

}  // namespace

void fused_route_permute(torch::Tensor logits, torch::Tensor bias,
                         torch::Tensor ids, torch::Tensor weights,
                         torch::Tensor e2p, torch::Tensor p2e, torch::Tensor p2t,
                         torch::Tensor tileExpert, torch::Tensor tileMnLimit,
                         torch::Tensor paddedSize, torch::Tensor numTiles,
                         int64_t ablate) {
  int batch = logits.size(0);
  TORCH_CHECK(batch >= 1 && batch <= kMaxB, "B must be in [1,128]");
  TORCH_CHECK(logits.size(1) == kExperts && logits.stride(1) == 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  fusedRoutePermuteKernel<<<1, kThreads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), bias.data_ptr<float>(), batch,
      ids.data_ptr<int>(), weights.data_ptr<float>(), e2p.data_ptr<int>(),
      p2e.data_ptr<int>(), p2t.data_ptr<int>(), tileExpert.data_ptr<int>(),
      tileMnLimit.data_ptr<int>(), paddedSize.data_ptr<int>(),
      numTiles.data_ptr<int>(), (int)ablate);
}

void permute_from_ids(torch::Tensor ids, torch::Tensor e2p, torch::Tensor p2e,
                      torch::Tensor p2t, torch::Tensor tileExpert,
                      torch::Tensor tileMnLimit, torch::Tensor paddedSize,
                      torch::Tensor numTiles, bool use_pdl) {
  int batch = ids.size(0);
  TORCH_CHECK(ids.size(1) == kTopK && ids.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (use_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(kPermThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    auto err = cudaLaunchKernelEx(
        &cfg, permuteFromIdsKernel<true>, ids.data_ptr<int>(), batch,
        e2p.data_ptr<int>(), p2e.data_ptr<int>(), p2t.data_ptr<int>(),
        tileExpert.data_ptr<int>(), tileMnLimit.data_ptr<int>(),
        paddedSize.data_ptr<int>(), numTiles.data_ptr<int>());
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  } else {
    permuteFromIdsKernel<false><<<1, kPermThreads, 0, stream>>>(
        ids.data_ptr<int>(), batch, e2p.data_ptr<int>(), p2e.data_ptr<int>(),
        p2t.data_ptr<int>(), tileExpert.data_ptr<int>(),
        tileMnLimit.data_ptr<int>(), paddedSize.data_ptr<int>(),
        numTiles.data_ptr<int>());
  }
}

void fused_route_permute_v2(torch::Tensor logits, torch::Tensor bias,
                            torch::Tensor counter, torch::Tensor ids,
                            torch::Tensor weights, torch::Tensor e2p,
                            torch::Tensor p2e, torch::Tensor p2t,
                            torch::Tensor tileExpert, torch::Tensor tileMnLimit,
                            torch::Tensor paddedSize, torch::Tensor numTiles,
                            int64_t ablate) {
  int batch = logits.size(0);
  TORCH_CHECK(batch >= 1 && batch <= kMaxB, "B must be in [1,128]");
  TORCH_CHECK(logits.size(1) == kExperts && logits.stride(1) == 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  fusedV2Kernel<<<batch, kV2Threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), bias.data_ptr<float>(), batch,
      counter.data_ptr<int>(), ids.data_ptr<int>(), weights.data_ptr<float>(),
      e2p.data_ptr<int>(), p2e.data_ptr<int>(), p2t.data_ptr<int>(),
      tileExpert.data_ptr<int>(), tileMnLimit.data_ptr<int>(),
      paddedSize.data_ptr<int>(), numTiles.data_ptr<int>(), (int)ablate);
}

void probe_route_work(torch::Tensor logits, torch::Tensor bias, torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  probeRouteWorkKernel<<<(int)logits.size(0), kV2Threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), bias.data_ptr<float>(),
      out.data_ptr<float>());
}

void probe_permute_work(torch::Tensor ids, torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  probePermuteWorkKernel<<<1, kPermThreads, 0, stream>>>(
      ids.data_ptr<int>(), (int)ids.numel(), out.data_ptr<int>());
}

void probe_phases(int64_t phases, torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  probePhasesKernel<<<1, kThreads, 0, stream>>>((int)phases, out.data_ptr<int>());
}

void probe_stream(torch::Tensor in, torch::Tensor out, int64_t blocks) {
  auto stream = at::cuda::getCurrentCUDAStream();
  probeStreamKernel<<<(int)blocks, 256, 0, stream>>>(
      reinterpret_cast<const float4*>(in.data_ptr<float>()), in.numel() / 4,
      out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_route_permute", &fused_route_permute);
  m.def("fused_route_permute_v2", &fused_route_permute_v2);
  m.def("permute_from_ids", &permute_from_ids);
  m.def("probe_route_work", &probe_route_work);
  m.def("probe_permute_work", &probe_permute_work);
  m.def("probe_phases", &probe_phases);
  m.def("probe_stream", &probe_stream);
}
