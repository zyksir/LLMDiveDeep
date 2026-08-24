"""CUDA-graph surgery: swap TRT-LLM's routingIndicesClusterKernel for a
single-CTA small-batch replacement, without rebuilding TRT-LLM.

Why surgery instead of a C++ rebuild: the installed tensorrt_llm wheel
(1.3.0rc23, Baseten fork build) does not match the local checkout
(1.3.0rc19 + local commits), the container has no TensorRT build root, and
libtensorrt_llm.so is a 1.1 GB monolith - an ABI-safe library swap is not
available.  The decode path is always executed via CUDA graphs, so patching
the captured graph gives the same effect for the shipped configuration.

How it works
------------
1. Capture the decode graph with ``torch.cuda.CUDAGraph(keep_graph=True)``.
2. ``scan(graph)`` walks the graph, finds kernel nodes whose (mangled) name
   contains ``routingIndicesClusterKernel`` and returns the scalar fields of
   their single KernelParams argument.  The caller verifies these against
   known values (numExperts=896, topK=16, numLocalExperts=112, ...) - this
   is an *empirical* layout check against the live wheel, so fork drift in
   the struct layout is detected before any patching.
3. ``patch(graph, ...)`` replaces each verified node with the single-CTA
   kernel (validated bitwise in local_debug/routing_indices_smallb.py), reusing
   the exact buffer pointers from the original node's parameter struct and
   preserving dependency edges including PDL (programmatic) edge data.
4. Caller then runs ``graph.instantiate()`` and replays as usual.

Safety
------
- Nodes whose mPtrTopKIds is null (score-path routing, e.g. baseline
  strategies) are left untouched.
- Integer fingerprint mismatch raises instead of silently patching, since
  it means the wheel's struct layout drifted from the mirror below.
- The replacement kernel issues ``griddepcontrol.wait`` on entry and
  ``griddepcontrol.launch_dependents`` before its final write phase, so
  programmatic (PDL) edges in either direction stay correct.

The struct mirror below matches routingCustom::KernelParams from the local
checkout's RoutingKernel.h (KernelParamsBase + mPtrTopKPacked/mTopK); only
fields up to mTopK are read.
"""

from __future__ import annotations

import torch

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Mirror of moe::dev::routing::routingCustom::KernelParams<InputT, OutputT,
// MaxNumExperts, MaxNumTopExperts, ExpertSelectPolicy> - fields up to mTopK.
// Layout is verified empirically via scan() fingerprints before use.
// ---------------------------------------------------------------------------
// Matches flashinfer 0.6.18rc1's RoutingKernel.h KernelParamsBase + the
// routingCustom KernelParams prefix. The 2026-08-22 silent no-op was this
// mirror drifting behind (four shared-expert scalars + mPtrRoutingReplayOut
// were inserted before the derived members, and mTopK became IntFastDiv);
// every field past mNumLocalExperts read garbage and the patch skipped all
// nodes as "score path". The fingerprint TORCH_CHECKs below abort loudly on
// the next drift instead of silently no-op'ing.
struct ParamsMirror
{
    bool mUsePdl;
    bool mIsPow2;
    int32_t* mPtrExpertCounts;
    int32_t* mPtrPermutedIdxSize;
    int32_t* mPtrExpandedIdxToPermutedIdx;
    int32_t* mPtrPermutedIdxToExpandedIdx;
    int32_t* mPtrPermutedIdxToTokenIdx;
    int32_t* mPtrCtaIdxXyToBatchIdx;
    int32_t* mPtrCtaIdxXyToMnLimit;
    int32_t* mPtrNumNonExitingCtas;
    void* mPtrTopKWeights;
    int32_t* mPtrTopKIds;
    void const* mPtrScores;
    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mPaddingLog2;
    int32_t mTileTokensDim;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
    int32_t mNumFusedSharedExperts;
    int32_t mSharedExpertTokenOffset;
    int32_t mSharedExpertNumTokens;
    int32_t mTotalExpertsPerToken;
    int16_t* mPtrRoutingReplayOut;
    // ---- ExpertSelectPolicy KernelParams (derived; RoutingKernel.h:353) ----
    void* mPtrTopKPacked;  // PackedScoreIdx<bf16>: low16 = score bits, high16 = idx
    int32_t mTopK;
    // trailing ExpertSelectParams not mirrored (never read)
};

// ---------------------------------------------------------------------------
// Single-CTA replacement kernel (verbatim from local_debug/routing_indices_smallb.py,
// validated there against the reference cluster kernel), plus PDL handling.
// ---------------------------------------------------------------------------
#define NUM_THREADS 1024
// numTokens <= 128, topK = 16 -> expandedIdxSize <= 2048 -> 2 items/thread
#define MAX_ITEMS 2
// numLocalExperts <= 256 (Kimi-K3: 112 at TP8, 224 at TP4; TP2's 448
// exceeds the single-CTA histogram and is skipped python-side)
#define MAX_LOCAL_EXPERTS 256

__global__ void __launch_bounds__(NUM_THREADS) routingIndicesSmallBKernel(
    int32_t const* __restrict__ topKIds,
    int32_t idsArePacked,  // PackedScoreIdx: expert id in the high 16 bits
    uint16_t* __restrict__ topKWeightsOut,  // packed input: unpack bf16 bits
    int32_t numTokens, int32_t topK,
    int32_t localExpertsStart,
    int32_t numLocalExperts,
    int32_t tileDim,
    int32_t* __restrict__ ctaIdxXyToBatchIdx,
    int32_t* __restrict__ ctaIdxXyToMnLimit,
    int32_t* __restrict__ permutedIdxSize,
    int32_t* __restrict__ numNonExitingCtas,
    int32_t* __restrict__ expandedIdxToPermutedIdx,
    int32_t* __restrict__ permutedIdxToTokenIdx)
{
#if __CUDA_ARCH__ >= 900
    // If the incoming graph edge is programmatic (PDL), the producer of
    // topKIds may still be running when we launch - wait for it.
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    __shared__ int32_t smemHist[MAX_LOCAL_EXPERTS];
    __shared__ int32_t smemCtaOff[MAX_LOCAL_EXPERTS + 1];
    __shared__ int32_t smemWarpTotals[MAX_LOCAL_EXPERTS / 32];

    int const tid = threadIdx.x;
    int const expandedIdxSize = numTokens * topK;

    if (tid < numLocalExperts)
    {
        smemHist[tid] = 0;
    }
    __syncthreads();

    int locE[MAX_ITEMS];
    int offInE[MAX_ITEMS];
#pragma unroll
    for (int ii = 0; ii < MAX_ITEMS; ++ii)
    {
        locE[ii] = -1;
        offInE[ii] = 0;
        int const idx = tid + ii * NUM_THREADS;
        if (idx < expandedIdxSize)
        {
            int const raw = topKIds[idx];
            if (idsArePacked && topKWeightsOut != nullptr)
            {
                // The cluster kernel unpacks bf16 weight bits from the
                // packed struct for downstream finalize; replicate it.
                topKWeightsOut[idx] = static_cast<uint16_t>(raw & 0xFFFF);
            }
            int const le = (idsArePacked ? (raw >> 16) : raw)
                - localExpertsStart;
            if (le >= 0 && le < numLocalExperts)
            {
                locE[ii] = le;
                offInE[ii] = atomicAdd(&smemHist[le], 1);
            }
        }
    }
    __syncthreads();

    int myNumCta = 0;
    int scanIncl = 0;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    if (tid < MAX_LOCAL_EXPERTS)
    {
        if (tid < numLocalExperts)
        {
            myNumCta = (smemHist[tid] + tileDim - 1) / tileDim;
        }
        scanIncl = myNumCta;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
            int const n = __shfl_up_sync(0xffffffffu, scanIncl, d);
            if (lane >= d)
            {
                scanIncl += n;
            }
        }
        if (lane == 31)
        {
            smemWarpTotals[warp] = scanIncl;
        }
    }
    __syncthreads();

    if (tid < MAX_LOCAL_EXPERTS)
    {
        int prefix = 0;
        for (int w = 0; w < warp; ++w)
        {
            prefix += smemWarpTotals[w];
        }
        if (tid < numLocalExperts)
        {
            smemCtaOff[tid] = prefix + scanIncl - myNumCta;
        }
        if (tid == MAX_LOCAL_EXPERTS - 1)
        {
            int total = 0;
#pragma unroll
            for (int w = 0; w < MAX_LOCAL_EXPERTS / 32; ++w)
            {
                total += smemWarpTotals[w];
            }
            smemCtaOff[numLocalExperts] = total;
            numNonExitingCtas[0] = total;
            permutedIdxSize[0] = total * tileDim;
        }
    }
    __syncthreads();

    int const totalCtas = smemCtaOff[numLocalExperts];
    int const permSize = totalCtas * tileDim;

    for (int s = tid; s < totalCtas; s += NUM_THREADS)
    {
        int lo = 0;
        int hi = numLocalExperts - 1;
        while (lo < hi)
        {
            int const mid = (lo + hi + 1) >> 1;
            if (smemCtaOff[mid] <= s)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        ctaIdxXyToBatchIdx[s] = lo;
        int const mnLimit1 = (s + 1) * tileDim;
        int const mnLimit2 = smemCtaOff[lo] * tileDim + smemHist[lo];
        ctaIdxXyToMnLimit[s] = min(mnLimit1, mnLimit2);
    }

    for (int p = tid; p < permSize; p += NUM_THREADS)
    {
        permutedIdxToTokenIdx[p] = -1;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 900
    // Let PDL dependents launch early; they still griddepcontrol.wait for
    // our completion before consuming outputs.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

#pragma unroll
    for (int ii = 0; ii < MAX_ITEMS; ++ii)
    {
        int const idx = tid + ii * NUM_THREADS;
        if (idx >= expandedIdxSize)
        {
            break;
        }
        int p = -1;
        if (locE[ii] >= 0)
        {
            p = smemCtaOff[locE[ii]] * tileDim + offInE[ii];
            permutedIdxToTokenIdx[p] = idx / topK;
        }
        expandedIdxToPermutedIdx[idx] = p;
    }
}

// ---------------------------------------------------------------------------
// Graph surgery host code
// ---------------------------------------------------------------------------

// The probing below intentionally calls runtime APIs that can fail on
// driver-created nodes (cudaGraphKernelNodeGetParams, cudaFuncGetName).
// Those failures set the per-thread CUDA last-error, which torch's next
// kernel-launch check would then report as a bogus launch failure
// ("invalid device function"). Always swallow it before returning.
struct LastErrorClearer
{
    ~LastErrorClearer()
    {
        (void) cudaGetLastError();
    }
};

static std::string nodeKernelName(cudaGraphNode_t node)
{
    LastErrorClearer clearer;
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess
        || type != cudaGraphNodeTypeKernel)
    {
        return {};
    }
    char const* name = nullptr;
    cudaKernelNodeParams p{};
    if (cudaGraphKernelNodeGetParams(node, &p) == cudaSuccess && p.func != nullptr)
    {
        // Captured runtime launches store the host stub in p.func.
        if (cudaFuncGetName(&name, p.func) == cudaSuccess && name)
        {
            return name;
        }
        name = nullptr;
        if (cuFuncGetName(&name, reinterpret_cast<CUfunction>(p.func)) == CUDA_SUCCESS
            && name)
        {
            return name;
        }
    }
    // Driver-created nodes (e.g. CuTeDSL cuLaunchKernelEx captures): the
    // runtime GetParams refuses them, use the driver query instead.
    CUDA_KERNEL_NODE_PARAMS dp{};
    if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &dp)
        == CUDA_SUCCESS)
    {
        name = nullptr;
        if (dp.func && cuFuncGetName(&name, dp.func) == CUDA_SUCCESS && name)
        {
            return name;
        }
        name = nullptr;
        if (dp.kern && cuKernelGetName(&name, dp.kern) == CUDA_SUCCESS && name)
        {
            return name;
        }
    }
    return {};
}

static std::vector<cudaGraphNode_t> clusterNodes(cudaGraph_t graph)
{
    size_t numNodes = 0;
    TORCH_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes) == cudaSuccess,
        "cudaGraphGetNodes(count) failed");
    std::vector<cudaGraphNode_t> nodes(numNodes);
    TORCH_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes) == cudaSuccess,
        "cudaGraphGetNodes failed");
    std::vector<cudaGraphNode_t> found;
    for (auto node : nodes)
    {
        std::string const name = nodeKernelName(node);
        if (name.find("routingIndicesClusterKernel") != std::string::npos)
        {
            found.push_back(node);
        }
    }
    return found;
}

static ParamsMirror readParams(cudaGraphNode_t node)
{
    LastErrorClearer clearer;
    void** kernelParams = nullptr;
    cudaKernelNodeParams p{};
    if (cudaGraphKernelNodeGetParams(node, &p) == cudaSuccess)
    {
        kernelParams = p.kernelParams;
    }
    else
    {
        // Driver-created node: query via the driver API instead.
        CUDA_KERNEL_NODE_PARAMS dp{};
        TORCH_CHECK(cuGraphKernelNodeGetParams(
                reinterpret_cast<CUgraphNode>(node), &dp) == CUDA_SUCCESS,
            "cuGraphKernelNodeGetParams failed");
        kernelParams = dp.kernelParams;
    }
    TORCH_CHECK(kernelParams != nullptr, "node has no kernelParams");
    ParamsMirror mirror{};
    std::memcpy(&mirror, kernelParams[0], sizeof(ParamsMirror));
    return mirror;
}

// Debug helper: names of all kernel nodes (recursing into child graphs),
// prefixed with node type info for non-kernel nodes.
static void collectNames(cudaGraph_t graph, std::vector<std::string>& out, int depth)
{
    size_t numNodes = 0;
    if (cudaGraphGetNodes(graph, nullptr, &numNodes) != cudaSuccess)
    {
        out.push_back("<cudaGraphGetNodes failed>");
        return;
    }
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    for (auto node : nodes)
    {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
        {
            continue;
        }
        std::string prefix(depth * 2, ' ');
        if (type == cudaGraphNodeTypeKernel)
        {
            std::string name = nodeKernelName(node);
            out.push_back(prefix + (name.empty() ? "<kernel: name lookup failed>" : name));
        }
        else if (type == cudaGraphNodeTypeGraph)
        {
            out.push_back(prefix + "<child graph>");
            cudaGraph_t child;
            if (cudaGraphChildGraphNodeGetGraph(node, &child) == cudaSuccess)
            {
                collectNames(child, out, depth + 1);
            }
        }
        else
        {
            out.push_back(prefix + "<node type " + std::to_string((int) type) + ">");
        }
    }
}

std::vector<std::string> names(int64_t graphHandle)
{
    std::vector<std::string> out;
    collectNames(reinterpret_cast<cudaGraph_t>(graphHandle), out, 0);
    return out;
}

// Flat per-node scalar dump so python can fingerprint the layout.
std::vector<int64_t> scan(int64_t graphHandle)
{
    auto graph = reinterpret_cast<cudaGraph_t>(graphHandle);
    std::vector<int64_t> out;
    for (auto node : clusterNodes(graph))
    {
        ParamsMirror const m = readParams(node);
        out.push_back(m.mNumTokens);
        out.push_back(m.mNumExperts);
        out.push_back(m.mTopK);
        out.push_back(m.mTileTokensDim);
        out.push_back(m.mLocalExpertsStartIdx);
        out.push_back(m.mLocalExpertsStrideLog2);
        out.push_back(m.mNumLocalExperts);
        out.push_back(m.mPaddingLog2);
        out.push_back(static_cast<int64_t>(m.mIsPow2));
        out.push_back(static_cast<int64_t>(m.mUsePdl));
        out.push_back(m.mPtrTopKIds != nullptr);
        out.push_back(m.mPtrPermutedIdxToExpandedIdx != nullptr);
        out.push_back(m.mPtrExpandedIdxToPermutedIdx != nullptr);
        out.push_back(m.mPtrPermutedIdxToTokenIdx != nullptr);
        out.push_back(m.mPtrCtaIdxXyToBatchIdx != nullptr);
        out.push_back(m.mPtrNumNonExitingCtas != nullptr);
    }
    return out;
}

int64_t patch(int64_t graphHandle, int64_t expNumExperts, int64_t expTopK,
    int64_t expNumLocalExperts, int64_t maxTokens)
{
    LastErrorClearer clearer;
    auto graph = reinterpret_cast<cudaGraph_t>(graphHandle);
    int64_t patched = 0;
    for (auto node : clusterNodes(graph))
    {
        ParamsMirror const m = readParams(node);
        if (m.mPtrTopKIds == nullptr && m.mPtrTopKPacked == nullptr)
        {
            continue; // score-path routing (e.g. baseline) - leave untouched
        }
        // Integer fingerprints: mismatch means struct layout drift -> abort
        // loudly rather than corrupt the graph.
        TORCH_CHECK(m.mNumExperts == expNumExperts,
            "routing_graphpatch: mNumExperts=", m.mNumExperts, " expected ", expNumExperts);
        TORCH_CHECK(m.mTopK == expTopK,
            "routing_graphpatch: mTopK=", m.mTopK, " expected ", expTopK);
        TORCH_CHECK(m.mNumLocalExperts == expNumLocalExperts,
            "routing_graphpatch: mNumLocalExperts=", m.mNumLocalExperts);
        TORCH_CHECK(m.mLocalExpertsStrideLog2 == 0,
            "routing_graphpatch: strideLog2=", m.mLocalExpertsStrideLog2);
        TORCH_CHECK(m.mNumTokens >= 1 && m.mNumTokens <= maxTokens,
            "routing_graphpatch: mNumTokens=", m.mNumTokens, " out of range");
        TORCH_CHECK(m.mNumTokens * m.mTopK <= NUM_THREADS * MAX_ITEMS,
            "routing_graphpatch: expanded size too large for kernel");
        TORCH_CHECK(expNumLocalExperts <= MAX_LOCAL_EXPERTS,
            "routing_graphpatch: too many local experts");
        TORCH_CHECK(m.mTileTokensDim >= 1 && m.mTileTokensDim <= 256,
            "routing_graphpatch: mTileTokensDim=", m.mTileTokensDim, " implausible");
        TORCH_CHECK(!m.mIsPow2 || m.mPaddingLog2 > 0,
            "routing_graphpatch: inconsistent pow2 flags");
        TORCH_CHECK(m.mPtrPermutedIdxToExpandedIdx == nullptr,
            "routing_graphpatch: mPtrPermutedIdxToExpandedIdx is set; "
            "replacement kernel does not fill it");
        TORCH_CHECK(m.mPtrExpandedIdxToPermutedIdx && m.mPtrPermutedIdxToTokenIdx
                && m.mPtrCtaIdxXyToBatchIdx && m.mPtrCtaIdxXyToMnLimit
                && m.mPtrPermutedIdxSize && m.mPtrNumNonExitingCtas,
            "routing_graphpatch: expected output pointer is null");

        // Collect edges with edge data, so PDL ports are preserved
        // (CUDA 13: the default APIs carry cudaGraphEdgeData).
        size_t numDeps = 0;
        TORCH_CHECK(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &numDeps)
                == cudaSuccess, "GetDependencies(count) failed");
        std::vector<cudaGraphNode_t> deps(numDeps);
        std::vector<cudaGraphEdgeData> depData(numDeps);
        if (numDeps)
        {
            TORCH_CHECK(cudaGraphNodeGetDependencies(
                    node, deps.data(), depData.data(), &numDeps) == cudaSuccess,
                "GetDependencies failed");
        }
        size_t numOut = 0;
        TORCH_CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &numOut)
                == cudaSuccess, "GetDependentNodes(count) failed");
        std::vector<cudaGraphNode_t> outs(numOut);
        std::vector<cudaGraphEdgeData> outData(numOut);
        if (numOut)
        {
            TORCH_CHECK(cudaGraphNodeGetDependentNodes(
                    node, outs.data(), outData.data(), &numOut) == cudaSuccess,
                "GetDependentNodes failed");
        }

        // Build replacement node. Argument values are copied by
        // cudaGraphAddKernelNode, so stack temporaries are fine.
        int32_t const* topKIds = m.mPtrTopKIds;
        int32_t idsArePacked = 0;
        if (topKIds == nullptr)
        {
            topKIds = reinterpret_cast<int32_t const*>(m.mPtrTopKPacked);
            idsArePacked = 1;
        }
        uint16_t* weightsOut = (idsArePacked
            ? reinterpret_cast<uint16_t*>(m.mPtrTopKWeights) : nullptr);
        int32_t numTokens = m.mNumTokens;
        int32_t topK = m.mTopK;
        int32_t start = m.mLocalExpertsStartIdx;
        int32_t numLocal = m.mNumLocalExperts;
        int32_t tileDim = m.mTileTokensDim;
        int32_t* batchIdx = m.mPtrCtaIdxXyToBatchIdx;
        int32_t* mnLimit = m.mPtrCtaIdxXyToMnLimit;
        int32_t* permSize = m.mPtrPermutedIdxSize;
        int32_t* numCtas = m.mPtrNumNonExitingCtas;
        int32_t* e2p = m.mPtrExpandedIdxToPermutedIdx;
        int32_t* p2t = m.mPtrPermutedIdxToTokenIdx;
        void* args[] = {&topKIds, &idsArePacked, &weightsOut, &numTokens, &topK, &start, &numLocal, &tileDim,
            &batchIdx, &mnLimit, &permSize, &numCtas, &e2p, &p2t};

        cudaKernelNodeParams np{};
        np.func = reinterpret_cast<void*>(&routingIndicesSmallBKernel);
        np.gridDim = dim3(1, 1, 1);
        np.blockDim = dim3(NUM_THREADS, 1, 1);
        np.sharedMemBytes = 0;
        np.kernelParams = args;
        np.extra = nullptr;

        cudaGraphNode_t newNode = nullptr;
        TORCH_CHECK(cudaGraphAddKernelNode(&newNode, graph, nullptr, 0, &np)
                == cudaSuccess, "cudaGraphAddKernelNode failed");

        for (size_t i = 0; i < numDeps; ++i)
        {
            TORCH_CHECK(cudaGraphAddDependencies(
                    graph, &deps[i], &newNode, &depData[i], 1) == cudaSuccess,
                "AddDependencies(in) failed");
        }
        for (size_t i = 0; i < numOut; ++i)
        {
            TORCH_CHECK(cudaGraphAddDependencies(
                    graph, &newNode, &outs[i], &outData[i], 1) == cudaSuccess,
                "AddDependencies(out) failed");
        }
        TORCH_CHECK(cudaGraphDestroyNode(node) == cudaSuccess,
            "cudaGraphDestroyNode failed");
        ++patched;
    }
    return patched;
}
"""

_CPP_SRC = r"""
#include <cstdint>
#include <string>
#include <vector>
std::vector<std::string> names(int64_t graphHandle);
std::vector<int64_t> scan(int64_t graphHandle);
int64_t patch(int64_t graphHandle, int64_t expNumExperts, int64_t expTopK,
    int64_t expNumLocalExperts, int64_t maxTokens);
"""

_SCAN_FIELDS = (
    "num_tokens", "num_experts", "top_k", "tile_dim", "local_start",
    "stride_log2", "num_local", "padding_log2", "is_pow2", "use_pdl",
    "has_topk_ids", "has_p2e", "has_e2p", "has_p2t", "has_batch_idx",
    "has_num_ctas",
)

_ext = None

# One canonical build dir, deliberately NOT under TORCH_EXTENSIONS_DIR:
# our harnesses set TORCH_EXTENSIONS_DIR per rank (/tmp/torchext_r<rank>),
# which would give every rank its own (potentially stale) copy of this
# extension. All processes must build/load the exact same .so.
_BUILD_DIR = "/root/.cache/llmdd_graphpatch"


def _source_hash() -> str:
    import hashlib

    return hashlib.sha256((_CPP_SRC + _CUDA_SRC).encode()).hexdigest()


def _direct_import(so: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("routing_graphpatch_ext", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _module():
    """Build (rank 0) or load the extension, safely under multi-rank MPI.

    load_inline's baton/ninja path deadlocks when many ranks race it (and a
    rank killed while holding the baton orphans the lock forever). Protocol:
    rank 0 builds via load_inline and stamps source.hash; other ranks poll
    for a matching stamp and then import the .so directly - no JIT machinery
    on any rank but 0. ROUTING_GRAPHPATCH_PREBUILT=1 skips the build on
    every rank (the .so must already exist and match).
    """
    global _ext
    if _ext is not None:
        return _ext

    import os
    import time

    so = os.path.join(_BUILD_DIR, "routing_graphpatch_ext.so")
    stamp = os.path.join(_BUILD_DIR, "source.hash")
    want = _source_hash()

    def _stamp_ok() -> bool:
        try:
            with open(stamp) as f:
                return f.read().strip() == want and os.path.exists(so)
        except OSError:
            return False

    prebuilt = os.environ.get("ROUTING_GRAPHPATCH_PREBUILT") == "1"
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))

    if _stamp_ok():
        _ext = _direct_import(so)
        return _ext

    if prebuilt or rank != 0:
        deadline = time.monotonic() + (0 if prebuilt else 600)
        while not _stamp_ok():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"routing_graphpatch: prebuilt .so missing or stale at "
                    f"{so} (prebuilt={prebuilt}, rank={rank}); "
                    "build it from a single process first"
                )
            time.sleep(1.0)
        _ext = _direct_import(so)
        return _ext

    from torch.utils.cpp_extension import load_inline

    os.makedirs(_BUILD_DIR, exist_ok=True)
    _ext = load_inline(
        name="routing_graphpatch_ext",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["names", "scan", "patch"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=_BUILD_DIR,
        verbose=False,
    )
    with open(stamp, "w") as f:
        f.write(want)
    return _ext


def graph_kernel_names(graph: torch.cuda.CUDAGraph) -> list[str]:
    """Debug: names of all kernel nodes in a captured (keep_graph) graph."""
    return list(_module().names(graph.raw_cuda_graph()))


def scan_graph(graph: torch.cuda.CUDAGraph) -> list[dict]:
    """Return scalar fingerprints of every routingIndicesClusterKernel node
    in a captured (keep_graph=True, not yet instantiated) CUDA graph."""
    flat = _module().scan(graph.raw_cuda_graph())
    n = len(_SCAN_FIELDS)
    assert len(flat) % n == 0
    return [
        dict(zip(_SCAN_FIELDS, flat[i : i + n])) for i in range(0, len(flat), n)
    ]


def patch_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    num_experts: int = 896,
    top_k: int = 16,
    num_local_experts: int = 112,
    max_tokens: int = 128,
) -> int:
    """Replace every precomputed-ids routingIndicesClusterKernel node with the
    single-CTA small-batch kernel. Call between capture end and
    ``graph.instantiate()``. Returns the number of nodes patched.

    num_local_experts is TP-degree dependent (896/tp) — pass it, never rely
    on the TP8 default (the hardcoded 112 asserted at TP4 with 224). Degrees
    whose local-expert count exceeds the kernel's histogram (TP2: 448 > 256)
    are skipped gracefully: the cluster kernel stays in the graph."""
    if num_local_experts > 256:
        return 0
    return _module().patch(
        graph.raw_cuda_graph(), num_experts, top_k, num_local_experts, max_tokens
    )
