"""Module-level micro-benchmark for DeepSeek-style sparse attention indexers.

This file is the **driver** only. The real `nn.Module` indexers (DSv3.2 /
GLM-5 / DSv4-C4), the FP8 quantizer, and the FA4 / block-sparse backends all
live in `DSA.py` next to this file; here we only wrap them in a sweep and
print the per-module headline table.

Requirements
------------
See `requirements.txt` next to this file for the full list. In short:

- Python 3.10+ with a CUDA-enabled `torch>=2.6` and `triton>=3.0`.
- `deep_gemm` with `fp8_mqa_logits` (a torch FP32 fallback is used if
  missing -- correct but ~50x slower).
- The block-sparse Triton kernel from sglang's `b10_kernels/sparse_attn/`.
  Auto-discovered via `SGLANG_B10_KERNELS_DIR` or a set of well-known
  workspace paths (see `DSA.resolve_b10_sparse_path`).
- Optional: `flash_attn.cute` (FA4) for the dense baseline column.
- An NVIDIA Hopper (SM_90) or Blackwell (SM_100) GPU. README numbers are
  from a single NVIDIA B200.

Overview
--------
Counterpart to `bench_sparse_attention_kernels.py`. The kernel file isolates
the FP8 MQA logits + top-k hot path; this file wraps that hot path in a real
`nn.Module` that mirrors the production indexer's projections / norms / RoPE /
FP8 quant, so we can measure the **module forward** time -- which is what
matters at deployment.

Three modules, each in a clearly-separated section:

    DSv32 -- DeepSeek V3.2: standard DSA -- one indexer call per layer.
    GLM   -- GLM-5 / GLM-5.2 (IndexShare): same DSA indexer as DSv3.2 but the
             forward can be skipped on 3 of every 4 layers via
             `index_topk_freq=4`. Same math, quartered call frequency.
    DSv4  -- DeepSeek V4: DSA-C4 indexer with a learned compressor that
             reduces the K stream length by `compress_ratio=4` before scoring.

For each module the script reports:

    proj_ms    : Q/K/W projections + norms + RoPE         (linear in S)
    quant_ms   : FP8 per-block quantize of Q and K        (linear in S)
    logits_ms  : deep_gemm.fp8_mqa_logits + top-k         (quadratic in S)
    total_ms   : indexer module forward
    speedup    : effective speedup of (indexer + sparse-FA) vs dense-FA
    idx_frac   : indexer_ms / total_attention_ms

Common invocations (from inside the sgl container):

    python bench_sparse_attention_modules.py --all
    python bench_sparse_attention_modules.py --glm --index_topk_freq 4
    python bench_sparse_attention_modules.py --dsv4    # compressor path
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# Make `common` (one dir up) and `DSA` (same dir) importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_LLMDIVEDEEP = os.path.dirname(_HERE)
if _LLMDIVEDEEP not in sys.path:
    sys.path.insert(0, _LLMDIVEDEEP)

from common.bench_utils import bench_function, print_section  # noqa: E402
from DSA import (  # noqa: E402
    INDEXER_BLOCK_SIZE,
    SPARSE_BLOCK,
    SPECS,
    DsaIndexer,
    Dsv4Indexer,
    IndexerSpec,  # noqa: F401  (re-exported for callers importing specs from here)
    env_report,
    fp8_quant_per_block,
    register_dense_fn,
    register_sparse_fa,
    try_import_deep_gemm,
)

DEFAULT_SEQ_LENS = [4096, 8192, 16384, 32768, 65536]


def _build_sparse_run():
    """Return just the block-sparse `run(q,k,v,idx,num,sm_scale)` callable."""
    pair = register_sparse_fa()
    if pair is None:
        return None
    _, run = pair
    return run


# ===========================================================================
# Driver: per-module headline table
# ===========================================================================

def bench_indexer_module(
    spec_key: str,
    seq_lens,
    index_topk_freq: int,
    warmup: int,
    iters: int,
) -> None:
    spec = SPECS[spec_key]
    print_section(
        f"{spec.name} indexer module  "
        f"(hidden={spec.hidden_size} q_lora={spec.q_lora_rank} "
        f"H_idx={spec.index_n_heads} D_idx={spec.index_head_dim} "
        f"compress_ratio={spec.compress_ratio} index_topk_freq={index_topk_freq})"
    )
    dg = try_import_deep_gemm()
    print(f"  deep_gemm.fp8_mqa_logits: {'present' if dg else 'MISSING -- FP32 fallback'}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    if spec.compress_ratio > 1:
        mod = Dsv4Indexer(spec, dtype=dtype).to(device)
    else:
        mod = DsaIndexer(spec, dtype=dtype).to(device)

    sparse_fn = _build_sparse_run()
    dense_fn = register_dense_fn()
    if sparse_fn is None:
        print("  WARN: sparse FA backend missing; skipping speedup column")
    if dense_fn is None:
        print("  WARN: dense FA backend missing; skipping speedup column")

    # Sparse-attention shape (downstream of indexer). We pick a per-rank GQA
    # shape so the kernel runs cleanly on a single GPU.
    H_attn, D_attn = 24, 128

    header = (
        f"  {'seq_len':>10}  {'proj ms':>10}  {'quant ms':>10}"
        f"  {'logits ms':>10}  {'total ms':>10}  {'idx /4 ms':>10}"
        f"  {'sparse ms':>10}  {'dense ms':>10}  {'speedup':>9}  {'idx frac':>9}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        # All shapes must be compatible with quant block size & sparse block.
        if S % INDEXER_BLOCK_SIZE != 0 or S % SPARSE_BLOCK != 0:
            continue
        try:
            x = torch.randn(S, spec.hidden_size, dtype=dtype, device=device)
            q_lora = torch.randn(S, spec.q_lora_rank, dtype=dtype, device=device)
        except RuntimeError as e:
            print(f"  S={S}  OOM: {e}")
            continue

        # Split the module forward into the three phases manually so each phase
        # gets its own timing.
        def phase_proj():
            return mod.projections(x, q_lora) if hasattr(mod, "projections") else None

        def phase_full():
            return mod(x, q_lora)

        # Time the three internal pieces using surrogate decompositions
        # (these mirror the dataflow inside `forward`).
        try:
            proj_ms = bench_function(phase_proj, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001
            print(f"  S={S}  proj err: {e}")
            continue
        q, k, w = mod.projections(x, q_lora) if hasattr(mod, "projections") else (None, None, None)
        if isinstance(mod, Dsv4Indexer):
            # Compressor + k_norm
            def compress_step():
                _ = mod.compressor(x)
            comp_ms = bench_function(compress_step, warmup=warmup, iters=iters)
            proj_ms = proj_ms + comp_ms

        def quant_step():
            _ = fp8_quant_per_block(q)
            _ = fp8_quant_per_block(k if not isinstance(mod, Dsv4Indexer) else mod.compressor(x))
        quant_ms = bench_function(quant_step, warmup=warmup, iters=iters)

        try:
            total_ms = bench_function(phase_full, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001
            print(f"  S={S}  full err: {e}")
            continue
        logits_ms = max(total_ms - proj_ms - quant_ms, 0.0)

        idx_amortized_ms = total_ms / max(index_topk_freq, 1)

        # Sparse + dense attention reference (block-sparse FA at ~5% sparsity).
        sparse_ms = float("nan")
        dense_ms = float("nan")
        if sparse_fn is not None and dense_fn is not None:
            qkv_shape = (1, S, H_attn, D_attn)
            q_attn = torch.randn(*qkv_shape, dtype=dtype, device=device)
            k_attn = torch.randn(*qkv_shape, dtype=dtype, device=device)
            v_attn = torch.randn(*qkv_shape, dtype=dtype, device=device)
            sm_scale = D_attn ** -0.5
            sparsity = max(spec.index_topk * SPARSE_BLOCK / S, 0.01)
            sparsity = min(sparsity, 1.0)
            Nq = Nkv = S // SPARSE_BLOCK
            topk_blocks = max(1, int(round(sparsity * Nkv)))
            scores = torch.rand(1, H_attn, Nq, Nkv, device=device)
            idx = scores.topk(topk_blocks, dim=-1).indices.to(torch.int32).contiguous()
            num = torch.full((1, H_attn, Nq), topk_blocks, dtype=torch.int32, device=device)
            try:
                sparse_ms = bench_function(
                    sparse_fn, q_attn, k_attn, v_attn, idx, num, sm_scale,
                    warmup=warmup, iters=iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  S={S}  sparse_fn err: {e}")
            try:
                dense_ms = bench_function(
                    dense_fn, q_attn, k_attn, v_attn, S, False, sm_scale,
                    warmup=warmup, iters=iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  S={S}  dense_fn err: {e}")

        attention_ms = idx_amortized_ms + sparse_ms if not math.isnan(sparse_ms) else idx_amortized_ms
        if not math.isnan(dense_ms) and attention_ms > 0:
            speedup = dense_ms / attention_ms
            idx_frac = idx_amortized_ms / attention_ms
        else:
            speedup = float("nan")
            idx_frac = float("nan")

        print(
            f"  {S:>10}  {proj_ms:>10.3f}  {quant_ms:>10.3f}"
            f"  {logits_ms:>10.3f}  {total_ms:>10.3f}  {idx_amortized_ms:>10.3f}"
            f"  {sparse_ms:>10.3f}  {dense_ms:>10.3f}"
            f"  {speedup:>9.2f}  {idx_frac:>9.2%}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--glm", action="store_true", help="GLM-5 / 5.2 indexer")
    p.add_argument("--dsv32", action="store_true", help="DeepSeek V3.2 indexer")
    p.add_argument("--dsv4", action="store_true", help="DeepSeek V4 (C4) indexer")
    p.add_argument("--all", action="store_true", help="run all three")
    p.add_argument("--seq_lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    p.add_argument(
        "--index_topk_freq", type=int, default=1,
        help="IndexShare period. 1=DSv3.2/GLM-5, 4=GLM-5.2.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    print("environment:")
    for k, v in env_report().items():
        print(f"  {k:>20} : {v}")
    print("  (see requirements.txt for full dependency notes)")

    if args.all:
        args.glm = args.dsv32 = args.dsv4 = True
    if not (args.glm or args.dsv32 or args.dsv4):
        args.glm = args.dsv32 = args.dsv4 = True

    for key, flag in (("dsv32", args.dsv32), ("glm", args.glm), ("dsv4", args.dsv4)):
        if flag:
            bench_indexer_module(
                key, args.seq_lens, args.index_topk_freq,
                args.warmup, args.iters,
            )


if __name__ == "__main__":
    main()
