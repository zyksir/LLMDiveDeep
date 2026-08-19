"""Kimi-K3 routing+permutation contract, PyTorch oracle, and checks.

Operation (whole region): FP32 contiguous logits [B, 896] and FP32 bias [896]
to top-16 expert IDs/weights plus the TP8-local permutation ABI consumed by
the grouped GEMM: expanded->permuted map, tile->local-expert map, tile M/N
limits, padded size, and non-empty tile count. Local experts are the 112
contiguous experts starting at 224; the grouped-GEMM tile is 128 tokens.

Implementations live in `routing_permutation_impls/`, one file each. The
oracle here is supplemental only; open-source kernels are the correctness and
performance baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

NUM_EXPERTS = 896
TOP_K = 16
LOCAL_EXPERT_START = 224
NUM_LOCAL_EXPERTS = 112
TILE_TOKENS = 128


def max_num_tiles(batch: int) -> int:
    """Tight worst-case tile count (TRT-LLM GroupedGemmInputsHelper formula)."""
    expanded = batch * TOP_K
    if expanded <= NUM_LOCAL_EXPERTS:
        return expanded
    return (expanded + (TILE_TOKENS - 1) * NUM_LOCAL_EXPERTS) // TILE_TOKENS


def max_permuted_tokens(batch: int) -> int:
    return max_num_tiles(batch) * TILE_TOKENS


@dataclass
class Prepared:
    """One shape-bound, CUDA-graph-capturable invocation.

    The benchmark fills `inputs` outside the timed region and calls `run()`
    on the current stream. All buffers are preallocated; `run` must not
    allocate, synchronize, or change buffer addresses.
    """

    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    run: Callable[[], None]
    charged_note: str = ""
    padding_filled: frozenset = field(default_factory=frozenset)


# Standard output names.
# route:   ids [B,16] int32, weights [B,16] fp32 (or bf16, declared per impl)
# permute: expanded_to_permuted [B*16] i32 (-1 for non-local),
#          tile_to_expert [<=max_num_tiles] i32 (local expert index),
#          tile_to_mn_limit [<=max_num_tiles] i32 (cumulative token limit),
#          padded_size [1] i32, num_tiles [1] i32,
#          optional permuted_to_expanded / permuted_to_token [max_permuted] i32
# fused:   union of both sets.


def route_reference(
    logits: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact FP32 sigmoid router with stable lower-ID tie breaking."""
    assert logits.ndim == 2 and logits.shape[1] == NUM_EXPERTS
    scores = torch.sigmoid(logits.float())
    ids = torch.argsort(
        scores + bias[None, :].float(), dim=1, descending=True, stable=True
    )[:, :TOP_K]
    picked = torch.gather(scores, 1, ids)
    weights = picked / picked.sum(dim=1, keepdim=True)
    return ids.to(torch.int32), weights


def permute_reference(ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expert-major TP8-local permutation with the TRT map/descriptor ABI."""
    assert ids.ndim == 2 and ids.shape[1] == TOP_K and ids.dtype == torch.int32
    device = ids.device
    flat = ids.reshape(-1)
    e2p = torch.full_like(flat, -1)
    p2e_chunks: list[torch.Tensor] = []
    tile_experts: list[int] = []
    tile_limits: list[int] = []
    base = 0
    for local in range(NUM_LOCAL_EXPERTS):
        sel = torch.nonzero(flat == LOCAL_EXPERT_START + local).flatten()
        count = sel.numel()
        if count == 0:
            continue
        tiles = (count + TILE_TOKENS - 1) // TILE_TOKENS
        padded = tiles * TILE_TOKENS
        e2p[sel] = torch.arange(count, dtype=torch.int32, device=device) + base
        p2e = torch.full((padded,), -1, dtype=torch.int32, device=device)
        p2e[:count] = sel.to(torch.int32)
        p2e_chunks.append(p2e)
        for t in range(tiles):
            tile_experts.append(local)
            tile_limits.append(min(base + (t + 1) * TILE_TOKENS, base + count))
        base += padded
    empty = torch.empty(0, dtype=torch.int32, device=device)
    p2e_all = torch.cat(p2e_chunks) if p2e_chunks else empty
    return {
        "expanded_to_permuted": e2p,
        "permuted_to_expanded": p2e_all,
        "permuted_to_token": torch.where(p2e_all >= 0, p2e_all // TOP_K, p2e_all),
        "tile_to_expert": torch.tensor(tile_experts, dtype=torch.int32, device=device),
        "tile_to_mn_limit": torch.tensor(tile_limits, dtype=torch.int32, device=device),
        "padded_size": torch.tensor([base], dtype=torch.int32, device=device),
        "num_tiles": torch.tensor([len(tile_experts)], dtype=torch.int32, device=device),
    }


def check_route(
    name: str,
    out_ids: torch.Tensor,
    out_weights: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_weights: torch.Tensor,
    *,
    atol: float,
) -> float:
    """IDs must match exactly (order-insensitive); weights within atol."""
    order = torch.argsort(out_ids.long(), dim=1)
    ids = torch.gather(out_ids.long(), 1, order)
    weights = torch.gather(out_weights.float(), 1, order)
    ref_order = torch.argsort(ref_ids.long(), dim=1)
    rids = torch.gather(ref_ids.long(), 1, ref_order)
    rweights = torch.gather(ref_weights.float(), 1, ref_order)
    if not torch.equal(ids, rids):
        bad = int((ids != rids).any(dim=1).sum())
        raise AssertionError(f"{name}: expert IDs mismatch on {bad} rows")
    err = float((weights - rweights).abs().max())
    if err > atol:
        raise AssertionError(f"{name}: weight max abs err {err:.3e} > {atol:.1e}")
    return err


def check_fused_region(
    name: str,
    logits: torch.Tensor,
    bias: torch.Tensor,
    out: dict[str, torch.Tensor],
    *,
    atol: float,
) -> float:
    """Order-insensitive check for a fused region without an INT32 ID output.

    The kernel's top-k slot order is internal (weights and maps stay mutually
    consistent), so validate: exact tile/padded descriptors, each local slot's
    expert identity via its permuted segment with its paired weight, and the
    per-token weight multiset for the non-local remainder.
    """
    batch = logits.shape[0]
    ref_ids, ref_weights = route_reference(logits, bias)
    ref = permute_reference(ref_ids)
    nt = int(ref["num_tiles"][0])
    ps = int(ref["padded_size"][0])
    if int(out["num_tiles"].reshape(-1)[0]) != nt or int(out["padded_size"].reshape(-1)[0]) != ps:
        raise AssertionError(f"{name}: tile count/padded size mismatch")
    for key in ("tile_to_expert", "tile_to_mn_limit"):
        if not torch.equal(out[key].reshape(-1)[:nt], ref[key]):
            raise AssertionError(f"{name}: {key} mismatch")

    device = logits.device
    dense = torch.zeros((batch, NUM_EXPERTS), device=device)
    dense.scatter_(1, ref_ids.long(), ref_weights)
    ref_local_mask = (ref_ids >= LOCAL_EXPERT_START) & (
        ref_ids < LOCAL_EXPERT_START + NUM_LOCAL_EXPERTS
    )

    counts = torch.bincount(
        ref_ids.reshape(-1)[ref_local_mask.reshape(-1)].long() - LOCAL_EXPERT_START,
        minlength=NUM_LOCAL_EXPERTS,
    )
    padded = ((counts + TILE_TOKENS - 1) // TILE_TOKENS) * TILE_TOKENS
    seg_base = torch.cumsum(padded, 0) - padded
    seg_of = torch.searchsorted(seg_base, torch.arange(ps, device=device), right=True) - 1

    e2p = out["expanded_to_permuted"].reshape(-1).long()
    local = e2p >= 0
    if int(local.sum()) != int(ref_local_mask.sum()):
        raise AssertionError(f"{name}: local slot count mismatch")
    if not torch.equal(
        local.view(batch, TOP_K).sum(1), ref_local_mask.sum(1)
    ):
        raise AssertionError(f"{name}: per-token local count mismatch")
    got = e2p[local]
    if int(torch.unique(got).numel()) != int(got.numel()):
        raise AssertionError(f"{name}: expanded_to_permuted not a bijection")
    experts = seg_of[got]  # local expert index per local slot
    if not bool((got < (seg_base + counts)[experts]).all()):
        raise AssertionError(f"{name}: slot beyond its expert's token count")
    tokens = torch.nonzero(local).flatten() // TOP_K
    ref_w_slot = dense[tokens, experts + LOCAL_EXPERT_START]
    if not bool((ref_w_slot > 0).all()):
        raise AssertionError(f"{name}: kernel selected an expert the oracle did not")
    weights = out["weights"].float().reshape(-1)
    err = float((weights[local] - ref_w_slot).abs().max()) if got.numel() else 0.0
    if err > atol:
        raise AssertionError(f"{name}: local weight max err {err:.3e} > {atol:.1e}")

    got_sorted = torch.sort(out["weights"].float(), dim=1)[0]
    ref_sorted = torch.sort(ref_weights, dim=1)[0]
    werr = float((got_sorted - ref_sorted).abs().max())
    if werr > atol:
        raise AssertionError(f"{name}: weight multiset max err {werr:.3e} > {atol:.1e}")

    if "permuted_to_expanded" in out:
        p2e = out["permuted_to_expanded"].reshape(-1).long()
        if not torch.equal(p2e[got], torch.nonzero(local).flatten()):
            raise AssertionError(f"{name}: permuted_to_expanded inconsistent")
    if "permuted_to_token" in out:
        p2t = out["permuted_to_token"].reshape(-1).long()
        if not torch.equal(p2t[got], tokens):
            raise AssertionError(f"{name}: permuted_to_token inconsistent")
    return max(err, werr)


def check_permutation(
    name: str,
    ids: torch.Tensor,
    out: dict[str, torch.Tensor],
    *,
    padding_filled: frozenset = frozenset(),
) -> None:
    """Validate the ABI invariants; within-expert order is unobservable."""
    ref = permute_reference(ids)
    nt = int(ref["num_tiles"][0])
    ps = int(ref["padded_size"][0])
    if int(out["num_tiles"].reshape(-1)[0]) != nt:
        raise AssertionError(
            f"{name}: num_tiles {int(out['num_tiles'].reshape(-1)[0])} != {nt}"
        )
    if int(out["padded_size"].reshape(-1)[0]) != ps:
        raise AssertionError(
            f"{name}: padded_size {int(out['padded_size'].reshape(-1)[0])} != {ps}"
        )
    for key in ("tile_to_expert", "tile_to_mn_limit"):
        got = out[key].reshape(-1)[:nt]
        if not torch.equal(got, ref[key]):
            raise AssertionError(f"{name}: {key} mismatch")

    flat = ids.reshape(-1).long()
    local = (flat >= LOCAL_EXPERT_START) & (flat < LOCAL_EXPERT_START + NUM_LOCAL_EXPERTS)
    e2p = out["expanded_to_permuted"].reshape(-1).long()
    if not torch.equal(e2p >= 0, local):
        raise AssertionError(f"{name}: expanded_to_permuted -1 mask mismatch")

    # Segment membership: each local assignment lands in its expert's range.
    ref_e2p = ref["expanded_to_permuted"].long()
    counts = torch.bincount(
        (flat[local] - LOCAL_EXPERT_START), minlength=NUM_LOCAL_EXPERTS
    )
    padded = ((counts + TILE_TOKENS - 1) // TILE_TOKENS) * TILE_TOKENS
    seg_base = torch.cumsum(padded, 0) - padded
    expert_of = (flat[local] - LOCAL_EXPERT_START)
    lo = seg_base[expert_of]
    hi = lo + counts[expert_of]
    got = e2p[local]
    if not bool(((got >= lo) & (got < hi)).all()):
        raise AssertionError(f"{name}: expanded_to_permuted outside expert segment")
    sorted_got, _ = torch.sort(got)
    if not torch.equal(sorted_got, torch.sort(ref_e2p[local])[0]):
        raise AssertionError(f"{name}: expanded_to_permuted is not a bijection")

    if "permuted_to_expanded" in out:
        p2e = out["permuted_to_expanded"].reshape(-1).long()
        idx = torch.nonzero(local).flatten()
        if not torch.equal(p2e[got], idx):
            raise AssertionError(f"{name}: permuted_to_expanded inconsistent")
        if "permuted_to_expanded" in padding_filled:
            mask = torch.ones(ps, dtype=torch.bool, device=ids.device)
            mask[got] = False
            if not bool((p2e[:ps][mask] == -1).all()):
                raise AssertionError(f"{name}: permuted_to_expanded padding not -1")
    if "permuted_to_token" in out:
        p2t = out["permuted_to_token"].reshape(-1).long()
        idx = torch.nonzero(local).flatten()
        if not torch.equal(p2t[got], idx // TOP_K):
            raise AssertionError(f"{name}: permuted_to_token inconsistent")
        if "permuted_to_token" in padding_filled:
            mask = torch.ones(ps, dtype=torch.bool, device=ids.device)
            mask[got] = False
            if not bool((p2t[:ps][mask] == -1).all()):
                raise AssertionError(f"{name}: permuted_to_token padding not -1")
