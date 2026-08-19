"""Kimi-K3 MoE routing: TRT-LLM baseline vs our Triton pack kernel.

K3 routes with DeepSeekV3-style "noaux_tc" (n_group=1): score every
expert with sigmoid(logit), pick top-16 of (sigmoid + per-expert bias),
weight each pick by its UNBIASED sigmoid normalized over the picks and
scaled by routed_scaling. Both implementations below are bit-exact
against the fp32 torch oracle (`routing_ref`).

Baseline (`routing_trtllm`)
    What the production stack runs when routing is done outside the
    fused-MoE kernel: gate GEMM (fp32 logits) then
    ``torch.ops.trtllm.noaux_tc_op``. NOTE the production K3 path
    (TRTLLMGenFusedMoE) does NOT run this op - it hands the logits to
    the fused MoE kernel, whose in-kernel routing stage costs ~29 us
    at decode batch sizes (single-CTA fallback; measured in the MoE
    bench ablation, B=8).

Ours (`route_pack`)
    One Triton CTA per token over the 896 bf16 logits, read IN PLACE
    from the merged input-GEMM output (no fp32 cast, no copy). Top-16
    selection packs each score with its reversed index into one
    order-preserving int64 key, so every pick is a single block-wide
    tl.max (value+argmax+tie-break in one reduction). Emits both the
    bf16 weights and the trtllm-gen packed ids ((expert<<16)|bf16(w)).

Honest accounting (kernels/bench_routing.py, B200) - on clean fp32
contiguous logits the stock kernel is nearly as good as ours (6.9 vs
6.0 us at B=1); the layer-level advantage comes from two things the
baseline op cannot do:

  * read the layer's ACTUAL logits - a bf16 STRIDED slice of the
    merged input-GEMM output - at full speed. The stock op accepts
    that input but slows to 8.2-10.4 us on it (B=1..256); ours runs
    6.0-7.6 us.
  * store the consumer's exact format directly (fused_moe int32 ids +
    fp32 scales via `route_for_fused_moe`, or trtllm-gen packed ids
    via `route_pack`), so no unpack/cast kernels follow. The stock
    output always needs a values.float() cast kernel (~1.2 us).

Net at decode sizes: ~6.0 us vs ~9.3 us for stock+cast, ~3.3 us per
MoE layer. Beyond ~1k tokens the cooperative stock kernel amortizes
better and layers switch back (TRITON_ROUTE_MAX_TOKENS). The MoE layer
bench (`bench_b10_kimi_k3_moe_layer.py --mode exp --sweep routing=...`)
measures routing backends in the full layer; the shipped default is
Routing.RADIX (see b10_kimi_k3_moe_layer.py).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..config import NUM_EXPERTS, ROUTED_SCALING, TOP_K

# Top-16 kernel shape (B10_ROUTE_IMPL):
#   "lane"  (default) one WARP per token, 32-bit fp32-monotone keys, no
#           smem/barrier in the 16-round chain (stock RoutingKernelTopK
#           shape, kept bit-exact via a 2nd tie-break reduction/round).
#           Ties "block" standalone; 2-3x less inflation under
#           concurrent GEMM waves (optional local evidence
#           local_debug/kimi_k3_routing_warp16.py, B200: 65-88 vs 180-258
#           us contended at B=64-80).
#   "block" the original one-CTA-per-token int64-key chain (16
#           DEPENDENT block-wide reductions crossing warps through
#           smem). Kept for A/B until a full retune retires it.
ROUTE_IMPL = os.environ.get("B10_ROUTE_IMPL", "lane")

# measured crossover vs the builtin cooperative routing (B200): near-tie
# at 4k tokens, builtin -10% at 8k. See kernels/bench_routing.py.
TRITON_ROUTE_MAX_TOKENS = 4096

# The kernel's top-16 loop is 16 DEPENDENT block-wide reductions, so
# it inflates under concurrent GEMM waves (contention probe: 10 us
# alone -> 54-60 us under big GEMMs at 4-8 warps; 2 warps is 3x more
# resilient there but ~2.5 us slower alone). In the LAYER the measured
# best is 4 warps everywhere: at small B the concurrent aux-chain
# GEMMs are small (nw=2 lost ~5 us at B=8-32), and at B>=64 the
# routed-split pipeline gives routing a clean window where the
# fastest-alone config wins.
NUM_WARPS = 4

# TODO(routing): the kernel is LATENCY-bound, not bandwidth-bound
# (~118 KB touched at B=64 = ~15 ns of DRAM time vs ~6 us measured).
# The critical path is the top-16 loop: 16 DEPENDENT block-wide max
# reductions (~3-4 us) after a ~1-2 us launch. A radix/bitonic top-k
# could break the serial chain and land near ~4 us, but the e2e ceiling
# is ~2 us out of a ~110-150 us layer (<2%) - park it until the big
# structural items (fc1 shard, comm) are settled.


@triton.jit
def _kimik3_route_pack_kernel(
    logits_ptr, bias_ptr, packed_ptr, weights_ptr,
    ids_ptr, scales_ptr,
    logits_stride,
    E: tl.constexpr, K: tl.constexpr,
    SCALE: tl.constexpr, BLOCK: tl.constexpr,
    EMIT_PACKED: tl.constexpr, EMIT_FUSED: tl.constexpr,
):
    """Per-token Kimi-K3 routing (DSv3-style noaux_tc, hard-coded to K3's
    n_group=1, 896 experts, top-16): top-K of sigmoid+bias,
    weights = selected sigmoid / sum * SCALE, packed for trtllm-gen as
    (expert_id << 16) | bf16(weight) bits. Reads bf16 logits in place
    from the merged GEMM output (sigmoid input is bit-identical to the
    fp32-cast the builtin routing uses).

    Top-16 selection: each score is packed with its reversed index into
    one order-preserving int64 key, so every pick is a single block-wide
    tl.max (value + argmax + tie-break to the smallest expert index in
    one reduction). ~13% faster than the 3-reductions-per-pick argmax
    loop; the sigmoid gathers for the picked experts are batched into
    one (K, BLOCK) masked sum after the loop."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < E
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=m, other=0.0)
    s = tl.sigmoid(x.to(tl.float32))
    b = tl.load(bias_ptr + offs, mask=m, other=0.0).to(tl.float32)
    biased = tl.where(m, s + b, -float("inf"))

    # fp32 -> order-preserving uint32, then key = value bits | reversed
    # index (low 10 bits) so equal scores pick the smaller expert first.
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    key32 = tl.where(ib >> 31 != 0, (~ib) & 0xFFFFFFFF, ib | 0x80000000)
    key = (key32 << 10) | (BLOCK - 1 - offs).to(tl.int64)

    kk = tl.arange(0, K)
    idxs = tl.zeros((K,), dtype=tl.int32)
    for k in tl.static_range(K):
        best = tl.max(key, axis=0)
        pick = (BLOCK - 1 - (best & (BLOCK - 1))).to(tl.int32)
        idxs = tl.where(kk == k, pick, idxs)
        key = tl.where(offs == pick, -1 << 62, key)

    svals = tl.sum(
        tl.where(offs[None, :] == idxs[:, None], s[None, :], 0.0), axis=1
    )
    sum_s = 0.0
    for k in tl.static_range(K):  # serial sum: keeps bit-exact pick order
        sum_s += tl.sum(tl.where(kk == k, svals, 0.0), axis=0)

    wf = svals * (SCALE / sum_s)
    if EMIT_PACKED:  # trtllm-gen routed MoE: (expert<<16) | bf16(w)
        w = wf.to(tl.bfloat16)
        wb = w.to(tl.uint16, bitcast=True).to(tl.int32)
        tl.store(packed_ptr + row * K + kk, (idxs << 16) | wb)
        tl.store(weights_ptr + row * K + kk, w)
    if EMIT_FUSED:  # CUTLASS fused_moe: int32 ids + fp32 scales,
        # stored directly so no cast kernels follow
        tl.store(ids_ptr + row * K + kk, idxs)
        tl.store(scales_ptr + row * K + kk, wf)


def route_pack(
    logits: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Our routing for the trtllm-gen ROUTED MoE binding. ``logits``:
    [B, >=896] bf16, may be a strided view of the merged input-GEMM
    output (columns beyond E are ignored). Returns (packed int32
    [B, K] as (expert<<16)|bf16(w), weights bf16 [B, K])."""
    batch = logits.shape[0]
    packed = torch.empty(
        batch, TOP_K, device=logits.device, dtype=torch.int32
    )
    weights = torch.empty(
        batch, TOP_K, device=logits.device, dtype=torch.bfloat16
    )
    _kimik3_route_pack_kernel[(batch,)](
        logits, gate_bias_f32, packed, weights, packed, weights,
        logits.stride(0),
        E=NUM_EXPERTS, K=TOP_K, SCALE=ROUTED_SCALING, BLOCK=1024,
        EMIT_PACKED=True, EMIT_FUSED=False, num_warps=NUM_WARPS,
    )
    return packed, weights


def route_for_fused_moe(
    logits: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Our routing in the CUTLASS ``torch.ops.trtllm.fused_moe`` input
    format: (ids int32 [B, K], scales fp32 [B, K]) stored DIRECTLY by
    the kernel - no unpack/cast kernels afterwards (the stock baseline
    needs a values.float() cast; ours is format-free)."""
    batch = logits.shape[0]
    ids = torch.empty(batch, TOP_K, device=logits.device,
                      dtype=torch.int32)
    scales = torch.empty(batch, TOP_K, device=logits.device,
                         dtype=torch.float32)
    _kimik3_route_pack_kernel[(batch,)](
        logits, gate_bias_f32, ids, scales, ids, scales,
        logits.stride(0),
        E=NUM_EXPERTS, K=TOP_K, SCALE=ROUTED_SCALING, BLOCK=1024,
        EMIT_PACKED=False, EMIT_FUSED=True, num_warps=NUM_WARPS,
    )
    return ids, scales


def route_for_trtllm_gen(
    logits: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Our routing in ``TRTLLMGenFusedMoE.run_moe`` input format:
    (ids int32 [B, K], scales bf16 [B, K]). Passing pre-computed top-k
    makes the runner skip its in-kernel scores+top-k stage (46.8 us
    ClusterKernel at B=64, 896 experts) and run only the cheap permute
    pipeline (RoutingFromTopKIds.cu)."""
    batch = logits.shape[0]
    ids = torch.empty(batch, TOP_K, device=logits.device,
                      dtype=torch.int32)
    scales = torch.empty(batch, TOP_K, device=logits.device,
                         dtype=torch.bfloat16)
    _kimik3_route_pack_kernel[(batch,)](
        logits, gate_bias_f32, ids, scales, ids, scales,
        logits.stride(0),
        E=NUM_EXPERTS, K=TOP_K, SCALE=ROUTED_SCALING, BLOCK=1024,
        EMIT_PACKED=False, EMIT_FUSED=True, num_warps=NUM_WARPS,
    )
    return ids, scales


# --- SGLang radix router (adopted unchanged open-source kernel) -------------
#
# Measured fastest route kernel at every shape and AT its probe-measured
# bound at decode (2.08-2.50 us vs the Triton kernel's ~6 us); see
# kernels/routing_permutation_results.md. The prebuilt artifact accepts the
# layer's actual inputs (bf16 STRIDED merged-GEMM slice or fp32 gate output)
# and applies ROUTED_SCALING in-kernel; weights come out fp32, so the
# trtllm-gen handoff pays one bf16 cast (still ~2.5 us ahead of Triton).
# With precomputed ids, run_moe executes only the post-top-k permutation
# pipeline (RoutingFromTopKIds, the moe_sort lineage) — i.e. radix+moe_sort.

_RADIX_MODULE = None


def _radix_module():
    global _RADIX_MODULE
    if _RADIX_MODULE is None:
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from kimi_k3_layer.kernel_research.kimi_k3_routing_permute_v2.sglang_radix_prebuilt \
            import load_sglang_radix

        _RADIX_MODULE = load_sglang_radix()
    return _RADIX_MODULE


def radix_available() -> bool:
    try:
        _radix_module()
        return True
    except Exception:  # noqa: BLE001 - artifact/setup missing
        return False


def _route_radix(logits: torch.Tensor, gate_bias_f32: torch.Tensor):
    """(ids int32 [B,K], weights fp32 [B,K]) with ROUTED_SCALING applied."""
    batch = logits.shape[0]
    weights = torch.empty(batch, TOP_K, device=logits.device,
                          dtype=torch.float32)
    ids = torch.empty(batch, TOP_K, device=logits.device, dtype=torch.int32)
    _radix_module().run(
        logits[:, :NUM_EXPERTS], gate_bias_f32, weights, ids,
        TOP_K, float(ROUTED_SCALING), True, True, False,
    )
    return ids, weights


def route_radix_for_fused_moe(
    logits: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radix routing in the CUTLASS fused_moe format (ids i32, scales f32)."""
    return _route_radix(logits, gate_bias_f32)


def route_radix_for_trtllm_gen(
    logits: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radix routing in the run_moe format (ids i32, scales bf16)."""
    ids, weights = _route_radix(logits, gate_bias_f32)
    return ids, weights.to(torch.bfloat16)


def unpack_ids(packed: torch.Tensor) -> torch.Tensor:
    """Expert indices from the trtllm-gen packed (id<<16)|w format."""
    return packed >> 16


def routing_trtllm(
    logits_f32: torch.Tensor, gate_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """TRT-LLM baseline: the fused noaux_tc kernel on fp32 logits
    (what DeepseekV3Gate.routing_impl.apply runs for K3's shape).
    Returns (indices int32 [B, K], weights fp32 [B, K])."""
    values, indices = torch.ops.trtllm.noaux_tc_op(
        logits_f32, gate_bias, 1, 1, TOP_K, ROUTED_SCALING
    )
    return indices.to(torch.int32), values.to(torch.float32)


_GATE_GEMM_KIND: str | None = None


def gate_gemm_trtllm(
    h: torch.Tensor, gate_w: torch.Tensor
) -> torch.Tensor:
    """Baseline gate GEMM to fp32 logits, matching DeepseekV3Gate's
    dispatch: dsv3_router_gemm_op where its shape specialization
    applies (256 experts; K3's 896 usually does not), else the CuteDSL
    bf16 GEMM on Blackwell, else a plain matmul + cast."""
    global _GATE_GEMM_KIND
    if _GATE_GEMM_KIND is None:
        try:
            torch.ops.trtllm.dsv3_router_gemm_op(
                h, gate_w.t(), bias=None, out_dtype=torch.float32)
            _GATE_GEMM_KIND = "dsv3"
        except Exception:  # noqa: BLE001 - shape not supported
            _GATE_GEMM_KIND = "matmul"
    if _GATE_GEMM_KIND == "dsv3":
        return torch.ops.trtllm.dsv3_router_gemm_op(
            h, gate_w.t(), bias=None, out_dtype=torch.float32)
    return (h @ gate_w.T).float()


def routing_ref(
    logits_f32: torch.Tensor, gate_bias_f32: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 torch oracle (same math as RefWeights.routing)."""
    scores = torch.sigmoid(logits_f32)
    sel = torch.topk(scores + gate_bias_f32, TOP_K, dim=-1).indices
    weights = torch.gather(scores, 1, sel)
    weights = weights / weights.sum(-1, keepdim=True) * ROUTED_SCALING
    return sel.to(torch.int32), weights
