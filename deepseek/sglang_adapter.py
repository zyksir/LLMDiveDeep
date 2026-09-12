"""Optional SGLang FlashMLA sparse-prefill core adapter, not a V4.1 server.

Inputs are normalized, rotated, dequantized tensors. Signature follows the
SGLang pin in sources.json. GPU execution is unverified; packing is CPU-tested.
"""

import argparse

import torch

from .csa2_reference import joint_sparse_attention, window_indices


def pack_sparse_prefill(local_kv, main_kv, local_ids, main_ids):
    """Rebase [B,Q,K] logical IDs to flattened workspace IDs.

    Compact valid IDs to a prefix before padding to a multiple of 64:
    topk_length is a count, so interspersed invalid holes are unsuitable.
    """
    if local_ids.shape[:2] != main_ids.shape[:2]:
        raise ValueError("indices must share batch/query dimensions")
    b, queries = local_ids.shape[:2]
    if local_kv.shape[0] != b or main_kv.shape[0] != b:
        raise ValueError("cache and index batch sizes differ")
    for cache, ids in ((local_kv, local_ids), (main_kv, main_ids)):
        if ((ids < -1) | (ids >= cache.shape[1])).any():
            raise ValueError("cache index outside [-1, N)")
    n_local = local_kv.shape[1]
    stride = n_local + main_kv.shape[1]
    offsets = torch.arange(b, device=local_ids.device)[:, None, None] * stride
    local = torch.where(local_ids >= 0, local_ids.long() + offsets, -1)
    main = torch.where(main_ids >= 0, main_ids.long() + n_local + offsets, -1)
    ids = torch.cat((local, main), -1)
    lengths = (ids >= 0).sum(-1).reshape(-1).to(torch.int32)
    sentinel = b * stride
    ids = torch.where(ids >= 0, ids, sentinel).sort(-1).values
    ids = torch.where(ids < sentinel, ids, -1)
    width = max(64, ((ids.shape[-1] + 63) // 64) * 64)
    ids = torch.nn.functional.pad(ids, (0, width - ids.shape[-1]), value=-1)
    if b * stride >= 2**31:
        raise ValueError("physical indices exceed int32 range")
    kv = (
        torch.cat((local_kv, main_kv), 1)
        .reshape(b * stride, local_kv.shape[-1])
        .contiguous()
    )
    return (
        kv,
        ids.reshape(b * queries, 1, width).int().contiguous(),
        lengths.contiguous(),
    )


def sparse_attention_sglang(q, local_kv, main_kv, local_ids, main_ids, sink):
    if not q.is_cuda:
        raise ValueError("SGLang FlashMLA requires CUDA; use the reference for CPU")
    if q.dtype != torch.bfloat16 or q.shape[2:] != (64, 512):
        raise ValueError("adapter targets BF16 Q with 64 heads and head_dim=512")
    if local_kv.dtype != q.dtype or main_kv.dtype != q.dtype:
        raise ValueError("KV must be dequantized to BF16")
    if sink.shape != (64,):
        raise ValueError("sink must have 64 logits")
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    kv, indices, lengths = pack_sparse_prefill(local_kv, main_kv, local_ids, main_ids)
    output, _, _ = flash_mla_sparse_fwd(
        q=q.reshape(-1, 64, 512).contiguous(),
        kv=kv.unsqueeze(1),
        indices=indices,
        sm_scale=512**-0.5,
        d_v=512,
        attn_sink=sink.float().contiguous(),
        topk_length=lengths,
    )
    return output.reshape_as(q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; no SGLang GPU parity was run.")
    torch.manual_seed(31)
    b, t, h, d = 2, 16, 64, 512
    q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    local, main_kv = torch.randn_like(q[:, :, 0]), torch.randn_like(q[:, :8, 0])
    local_ids = window_indices(torch.arange(t, device="cuda"), 4, b)
    visible = (torch.arange(t, device="cuda") + 1) // 2
    ids = torch.arange(8, device="cuda")[None, :].expand(t, -1)
    main_ids = (
        torch.where(ids < visible[:, None], ids, -1).int()[None].expand(b, -1, -1)
    )
    sink = torch.randn(h, device="cuda")
    expected = joint_sparse_attention(q, local, main_kv, local_ids, main_ids, sink)
    actual = sparse_attention_sglang(q, local, main_kv, local_ids, main_ids, sink)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    error = (actual.float() - expected.float()).abs().max().item()
    print(f"PASS: SGLang sparse core; max_abs_error={error:.6g}")


if __name__ == "__main__":
    main()
