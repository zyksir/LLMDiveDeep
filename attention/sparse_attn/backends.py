"""Prepared sparse-core adapters. No model, projection, indexer, or cache writes.

These call real optional kernels; missing packages never fall back to PyTorch.
GPU API/layout integration is source-reviewed, not runtime-validated here.
"""

import torch

from ..backends import (
    BackendUnavailable,
    PreparedAttention,
    optional_symbol,
    result_tensor,
)
from .compressed_sparse_attention import (
    dense_selected_attention,
    gather_attention,
    selection_mask,
)


BACKENDS = (
    "torch_gather",
    "torch_dense_mask",
    "sglang_flashmla",
    "flashmla",
    "flashinfer_dsv4",
)


def prepare_sparse(name, q, raw, compressed, indices, *, window, sink=None):
    """Logical contract: q [B,Q,H,D], raw [B,T,D], compressed [B,N,D].

    IDs refer to [raw | compressed], shared by heads; -1 means padding.
    The caller constructs the causal support; kernels must not add a second
    triangular mask to these already selected physical cache rows.
    """
    if q.ndim != 4 or raw.ndim != 3 or compressed.ndim != 3:
        raise ValueError("expected q [B,Q,H,D], raw/ compressed [B,N,D]")
    b, nq, h, d = q.shape
    if min(b, nq, h, d, raw.shape[1], compressed.shape[1], window) < 1:
        raise ValueError(
            "this adapter requires positive dimensions and a nonempty compressed cache"
        )
    if any(
        x.shape[0] != b
        or x.shape[-1] != d
        or x.dtype != q.dtype
        or x.device != q.device
        for x in (raw, compressed)
    ):
        raise ValueError(
            "cache batch/channel dimensions, dtype, and device must match q"
        )
    if indices.ndim != 3 or indices.shape[:2] != (b, nq) or indices.device != q.device:
        raise ValueError("indices must be [B,Q,L] on the query device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be integers")
    total = raw.shape[1] + compressed.shape[1]
    if ((indices < -1) | (indices >= total)).any().item():
        raise ValueError("indices outside logical cache")
    sorted_ids = indices.sort(dim=-1).values
    if (
        ((sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., 1:] >= 0))
        .any()
        .item()
    ):
        raise ValueError("duplicate active IDs would change the softmax distribution")
    if sink is not None and (
        sink.shape != (h,)
        or sink.device != q.device
        or not torch.isfinite(sink).all().item()
    ):
        raise ValueError("sink must be a finite per-head tensor on the query device")
    sink = None if sink is None else sink.float().contiguous()
    cache = torch.cat((raw, compressed), dim=1).contiguous()
    if name == "torch_gather":
        return PreparedAttention(
            lambda: gather_attention(q, cache, indices, sink=sink),
            "PyTorch gather/einsum/softmax",
            "core + materialized selected-KV gather; cache concatenation excluded",
        )
    if name == "torch_dense_mask":
        mask = selection_mask(indices, total)
        return PreparedAttention(
            lambda: dense_selected_attention(q, cache, mask, sink=sink),
            "PyTorch dense selected-support oracle",
            "dense core; support-mask construction excluded",
        )
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name}")
    if q.device.type != "cuda" or q.dtype != torch.bfloat16 or d != 512:
        raise BackendUnavailable("this sparse adapter requires CUDA, BF16, and D=512")
    if name in ("sglang_flashmla", "flashmla"):
        module = "sgl_kernel.flash_mla" if name == "sglang_flashmla" else "flash_mla"
        kernel = optional_symbol(module, "flash_mla_sparse_fwd")
        # Compact valid IDs to a prefix, THEN pad to the kernel's 64-slot tile.
        # Flattened batch cache needs offsets; never let request b read b=0.
        valid = indices >= 0
        order = torch.argsort(~valid, dim=-1, stable=True)
        compact = torch.gather(indices, -1, order)
        offsets = torch.arange(b, device=q.device)[:, None, None] * total
        compact = torch.where(compact >= 0, compact + offsets, -1)
        padded = (indices.shape[-1] + 63) // 64 * 64
        packed = torch.full((b, nq, padded), -1, dtype=torch.int32, device=q.device)
        packed[..., : compact.shape[-1]] = compact.to(torch.int32)
        packed = packed.reshape(b * nq, 1, padded).contiguous()
        lengths = valid.sum(dim=-1).to(torch.int32).reshape(-1)
        flat_q = q.reshape(b * nq, h, d).contiguous()
        flat_kv = cache.reshape(b * total, 1, d)

        def run():
            out = kernel(
                flat_q,
                flat_kv,
                packed,
                sm_scale=d**-0.5,
                d_v=512,
                attn_sink=sink,
                topk_length=lengths,
            )
            return result_tensor(out).reshape(b, nq, h, d)

        return PreparedAttention(
            run,
            f"{module}.flash_mla_sparse_fwd",
            "sparse prefill-kernel API (also callable with Q=1); packing excluded",
        )

    # Deliberately narrow, source-verified BF16 TRTLLM-GEN ABI. Packed SM120
    # FP8/NVFP4 and HCA CuTe paths have different metadata and are NOT aliases.
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise BackendUnavailable(
            "flashinfer_dsv4 BF16 adapter supports SM100/SM103 only"
        )
    if nq != 1 or window != 128 or raw.shape[1] < 128 or h not in (64, 128):
        raise BackendUnavailable(
            "flashinfer_dsv4 adapter: Q=1, W=128, T>=128, H=64 or 128"
        )
    if indices.shape[-1] % 4:
        raise BackendUnavailable(
            "FlashInfer sparse capacity must be a multiple of four"
        )
    if indices.shape[-1] <= 128 or (indices < 0).any().item():
        raise BackendUnavailable(
            "flashinfer_dsv4 adapter requires a full window and nonempty, unpadded global selection"
        )
    local, global_ids = indices[..., :128], indices[..., 128:] - raw.shape[1]
    expected_local = torch.arange(raw.shape[1] - 128, raw.shape[1], device=q.device)
    if (
        not torch.equal(local, expected_local[None, None, :].expand(b, 1, -1))
        or (global_ids < 0).any().item()
    ):
        raise BackendUnavailable("expected [last 128 raw IDs | compressed IDs]")
    kernel = optional_symbol("flashinfer.mla", "trtllm_batch_decode_sparse_mla_dsv4")

    def paged(x):
        stride = (x.shape[1] + 63) // 64 * 64
        storage = x.new_zeros(b, stride, d)
        storage[:, : x.shape[1]] = x
        return storage.reshape(-1, 1, 64, d), stride

    swa, swa_stride = paged(raw)
    comp, comp_stride = paged(compressed)
    batch_offset = torch.arange(b, device=q.device)[:, None, None]
    physical = torch.cat(
        (local + batch_offset * swa_stride, global_ids + batch_offset * comp_stride),
        dim=-1,
    )
    physical = physical.to(torch.int32).reshape(b * nq, -1).contiguous()
    lengths = torch.full((b,), indices.shape[-1], dtype=torch.int32, device=q.device)
    seq_lens = torch.full((b,), raw.shape[1], dtype=torch.int32, device=q.device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    out = torch.empty_like(q)
    query = q.contiguous()

    def run():
        return result_tensor(
            kernel(
                query=query,
                swa_kv_cache=swa,
                workspace_buffer=workspace,
                sparse_indices=physical,
                compressed_kv_cache=comp,
                sparse_topk_lens=lengths,
                seq_lens=seq_lens,
                out=out,
                bmm1_scale=d**-0.5,
                bmm2_scale=1.0,
                sinks=sink,
                kv_layout="HND",
                backend="trtllm-gen",
            )
        )

    return PreparedAttention(
        run,
        "flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4 / trtllm-gen",
        "BF16 paged decode core + wrapper; physical packing/workspace setup excluded",
    )
