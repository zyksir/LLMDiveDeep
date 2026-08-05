#!/usr/bin/env python3
"""CuTe DSL candidate for the Kimi-K3 fused attention-residual operation.

This is a standalone experiment.  It intentionally has the same local
aggregation contract as SGLang's ``attn_res_fused_tma`` entry point, but does
not implement the optional prefix write or distributed fused entry points.

The kernel is specialized for B200/SM100, bf16, H=7168, and 1..8 bank rows.
It uses one 256-thread CTA per token and 128-bit vector memory operations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator

HIDDEN = 7168
THREADS = 256
WARPS = THREADS // 32
VEC = 8
FULL_ITERS = HIDDEN // (THREADS * VEC)  # 3 full 2048-element sweeps
TAIL_THREADS = (HIDDEN // VEC) - FULL_ITERS * THREADS  # 128 threads
EPS_DEFAULT = 1.0e-6


def _make_launcher(num_bank_rows: int, eps: float):
    """Build a launcher with row count and epsilon baked into generated code."""
    if not 1 <= num_bank_rows <= 8:
        raise ValueError(f"num_bank_rows must be in [1, 8], got {num_bank_rows}")
    num_rows = num_bank_rows + 1

    @cute.kernel
    def kernel(
        prefix: cute.Tensor,
        bank: cute.Tensor,
        cw: cute.Tensor,
        ow: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32

        smem = SmemAllocator()
        # Row-major partials make each row's eight warp sums contiguous.
        partial_ss = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_rows, WARPS), stride=(WARPS, 1)),
        )
        partial_dot = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_rows, WARPS), stride=(WARPS, 1)),
        )
        weights = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((num_rows,))
        )
        out_ss = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((WARPS,))
        )
        out_scale = smem.allocate_tensor(cutlass.Float32, cute.make_layout((1,)))

        xvec = cute.make_fragment(VEC, cutlass.BFloat16)
        wvec = cute.make_fragment(VEC, cutlass.BFloat16)

        # Pass 1: score every raw row in fp32.
        for row in cutlass.range_constexpr(num_rows):
            ss = cutlass.Float32(0.0)
            dot = cutlass.Float32(0.0)
            for it in cutlass.range_constexpr(FULL_ITERS):
                tile = it * THREADS + tid
                if cutlass.const_expr(row < num_bank_rows):
                    cute.autovec_copy(
                        cute.local_tile(bank[token, row, None], (VEC,), (tile,)),
                        xvec,
                    )
                else:
                    cute.autovec_copy(
                        cute.local_tile(prefix[token, None], (VEC,), (tile,)), xvec
                    )
                cute.autovec_copy(cute.local_tile(cw, (VEC,), (tile,)), wvec)
                for j in cutlass.range_constexpr(VEC):
                    x = xvec[j].to(cutlass.Float32)
                    ss = ss + x * x
                    dot = dot + x * wvec[j].to(cutlass.Float32)

            if tid < TAIL_THREADS:
                tile = FULL_ITERS * THREADS + tid
                if cutlass.const_expr(row < num_bank_rows):
                    cute.autovec_copy(
                        cute.local_tile(bank[token, row, None], (VEC,), (tile,)),
                        xvec,
                    )
                else:
                    cute.autovec_copy(
                        cute.local_tile(prefix[token, None], (VEC,), (tile,)), xvec
                    )
                cute.autovec_copy(cute.local_tile(cw, (VEC,), (tile,)), wvec)
                for j in cutlass.range_constexpr(VEC):
                    x = xvec[j].to(cutlass.Float32)
                    ss = ss + x * x
                    dot = dot + x * wvec[j].to(cutlass.Float32)

            for stage in cutlass.range_constexpr(5):
                offset = 1 << (4 - stage)
                ss = ss + cute.arch.shuffle_sync_bfly(ss, offset)
                dot = dot + cute.arch.shuffle_sync_bfly(dot, offset)
            if lane == 0:
                partial_ss[row, warp] = ss
                partial_dot[row, warp] = dot

        cute.arch.sync_threads()

        # Eight lanes cooperatively finish one row each; lane 0 then computes
        # the tiny (<=9 element) softmax and publishes weights in shared memory.
        if tid == 0:
            row_logits = cute.make_fragment(num_rows, cutlass.Float32)
            row_max = cutlass.Float32(-3.0e38)
            for row in cutlass.range_constexpr(num_rows):
                ss = cutlass.Float32(0.0)
                dot = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(WARPS):
                    ss = ss + partial_ss[row, w]
                    dot = dot + partial_dot[row, w]
                logit = dot * cute.math.rsqrt(
                    ss * cutlass.Float32(1.0 / HIDDEN)
                    + cutlass.Float32(eps),
                    fastmath=True,
                )
                row_logits[row] = logit
                row_max = cute.arch.fmax(row_max, logit)
            denom = cutlass.Float32(0.0)
            for row in cutlass.range_constexpr(num_rows):
                p = cute.math.exp(row_logits[row] - row_max, fastmath=True)
                weights[row] = p
                denom = denom + p
            inv_denom = cutlass.Float32(1.0) / denom
            for row in cutlass.range_constexpr(num_rows):
                weights[row] = weights[row] * inv_denom

        cute.arch.sync_threads()

        # Pass 2: reload raw rows and form the fp32 weighted sum. Keeping all
        # rows live would require >250 values/thread at nvb=8 and collapse
        # occupancy, so the candidate deliberately trades bandwidth for state.
        acc = [
            cute.make_fragment(VEC, cutlass.Float32)
            for _ in range(FULL_ITERS + 1)
        ]
        for it in cutlass.range_constexpr(FULL_ITERS + 1):
            for j in cutlass.range_constexpr(VEC):
                acc[it][j] = cutlass.Float32(0.0)

        for row in cutlass.range_constexpr(num_rows):
            alpha = weights[row]
            for it in cutlass.range_constexpr(FULL_ITERS):
                tile = it * THREADS + tid
                if cutlass.const_expr(row < num_bank_rows):
                    cute.autovec_copy(
                        cute.local_tile(bank[token, row, None], (VEC,), (tile,)),
                        xvec,
                    )
                else:
                    cute.autovec_copy(
                        cute.local_tile(prefix[token, None], (VEC,), (tile,)), xvec
                    )
                for j in cutlass.range_constexpr(VEC):
                    acc[it][j] = (
                        acc[it][j] + alpha * xvec[j].to(cutlass.Float32)
                    )
            if tid < TAIL_THREADS:
                tile = FULL_ITERS * THREADS + tid
                if cutlass.const_expr(row < num_bank_rows):
                    cute.autovec_copy(
                        cute.local_tile(bank[token, row, None], (VEC,), (tile,)),
                        xvec,
                    )
                else:
                    cute.autovec_copy(
                        cute.local_tile(prefix[token, None], (VEC,), (tile,)), xvec
                    )
                for j in cutlass.range_constexpr(VEC):
                    acc[FULL_ITERS][j] = (
                        acc[FULL_ITERS][j]
                        + alpha * xvec[j].to(cutlass.Float32)
                    )

        ss = cutlass.Float32(0.0)
        for it in cutlass.range_constexpr(FULL_ITERS):
            for j in cutlass.range_constexpr(VEC):
                ss = ss + acc[it][j] * acc[it][j]
        if tid < TAIL_THREADS:
            for j in cutlass.range_constexpr(VEC):
                ss = ss + acc[FULL_ITERS][j] * acc[FULL_ITERS][j]
        for stage in cutlass.range_constexpr(5):
            ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << (4 - stage))
        if lane == 0:
            out_ss[warp] = ss
        cute.arch.sync_threads()

        if tid == 0:
            total = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                total = total + out_ss[w]
            out_scale[0] = cute.math.rsqrt(
                total * cutlass.Float32(1.0 / HIDDEN)
                + cutlass.Float32(eps),
                fastmath=True,
            )
        cute.arch.sync_threads()

        scale = out_scale[0]
        ovec = cute.make_fragment(VEC, cutlass.BFloat16)
        for it in cutlass.range_constexpr(FULL_ITERS):
            tile = it * THREADS + tid
            cute.autovec_copy(cute.local_tile(ow, (VEC,), (tile,)), wvec)
            for j in cutlass.range_constexpr(VEC):
                ovec[j] = (
                    acc[it][j] * scale * wvec[j].to(cutlass.Float32)
                ).to(cutlass.BFloat16)
            cute.autovec_copy(
                ovec, cute.local_tile(out[token, None], (VEC,), (tile,))
            )
        if tid < TAIL_THREADS:
            tile = FULL_ITERS * THREADS + tid
            cute.autovec_copy(cute.local_tile(ow, (VEC,), (tile,)), wvec)
            for j in cutlass.range_constexpr(VEC):
                ovec[j] = (
                    acc[FULL_ITERS][j] * scale * wvec[j].to(cutlass.Float32)
                ).to(cutlass.BFloat16)
            cute.autovec_copy(
                ovec, cute.local_tile(out[token, None], (VEC,), (tile,))
            )

    @cute.jit
    def launch(
        prefix: cute.Tensor,
        bank: cute.Tensor,
        cw: cute.Tensor,
        ow: cute.Tensor,
        out: cute.Tensor,
    ):
        kernel(prefix, bank, cw, ow, out).launch(
            grid=(prefix.shape[0], 1, 1),
            block=(THREADS, 1, 1),
        )

    return launch


_COMPILED: dict[tuple[int, int, int, float], object] = {}


def _compiled(prefix: torch.Tensor, bank: torch.Tensor, nvb: int, eps: float):
    key = (prefix.shape[0], bank.shape[1], nvb, float(eps))
    if key not in _COMPILED:
        fake_out = torch.empty_like(prefix)
        _COMPILED[key] = cute.compile(
            _make_launcher(nvb, eps),
            from_dlpack(prefix, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(bank, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(torch.empty(HIDDEN, dtype=torch.bfloat16, device="cuda"),
                        assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(torch.empty(HIDDEN, dtype=torch.bfloat16, device="cuda"),
                        assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(fake_out, assumed_align=16, enable_tvm_ffi=True),
            options="--gpu-arch sm_100a --enable-tvm-ffi",
        )
    return _COMPILED[key]


def attn_res_cute(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float = EPS_DEFAULT,
) -> None:
    """Run the precompiled candidate, writing ``out`` in place."""
    if prefix.dtype != torch.bfloat16 or prefix.shape[1] != HIDDEN:
        raise ValueError("prefix must be [T, 7168] bf16")
    if bank.dtype != torch.bfloat16 or bank.shape[0] != prefix.shape[0]:
        raise ValueError("bank must be [T, NB, 7168] bf16")
    _compiled(prefix, bank, nvb, eps)(prefix, bank, cw, ow, out)


def reference(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    nvb: int,
    eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    rows = torch.cat((bank[:, :nvb], prefix[:, None]), dim=1).float()
    logits = (rows * cw.float()).sum(-1)
    logits *= torch.rsqrt(rows.square().mean(-1) + eps)
    mixed = (torch.softmax(logits, dim=-1).unsqueeze(-1) * rows).sum(1)
    return (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
        * ow.float()
    ).to(torch.bfloat16)


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument("--nvb", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sglang-root", type=Path, default=Path(
        "/workspace/model-performance/yikai/diffusion_inference/sglang-opensource"
    ))
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("this candidate requires SM100")

    sglang_fn = None
    sys.path.insert(0, str(args.sglang_root / "python"))
    try:
        from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma

        sglang_fn = attn_res_fused_tma
    except Exception as exc:
        print(f"SGLang import unavailable: {type(exc).__name__}: {exc}")

    for tokens in args.tokens:
        for nvb in args.nvb:
            gen = torch.Generator(device="cuda").manual_seed(tokens * 17 + nvb)
            prefix = torch.randn(
                tokens, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
            )
            bank = torch.randn(
                tokens, 8, HIDDEN, generator=gen, device="cuda",
                dtype=torch.bfloat16,
            )
            cw = (
                torch.randn(
                    HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
                )
                * 0.02
            )
            ow = torch.randn(
                HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
            )
            out = torch.empty_like(prefix)
            ref = reference(prefix, bank, cw, ow, nvb)
            attn_res_cute(prefix, bank, cw, ow, out, nvb)
            torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs()
            ok = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
            cute_us = _bench(
                lambda: attn_res_cute(prefix, bank, cw, ow, out, nvb),
                args.warmup,
                args.iters,
            )
            fields = [
                f"T={tokens:4d}",
                f"nvb={nvb}",
                f"correct={ok}",
                f"max_abs={diff.max().item():.4g}",
                f"cute={cute_us:.2f} us",
            ]
            if sglang_fn is not None:
                sgl_out = torch.empty_like(prefix)
                sglang_fn(prefix, bank, cw, ow, sgl_out, nvb, EPS_DEFAULT)
                torch.cuda.synchronize()
                sgl_diff = (sgl_out.float() - ref.float()).abs().max().item()
                sgl_us = _bench(
                    lambda: sglang_fn(
                        prefix, bank, cw, ow, sgl_out, nvb, EPS_DEFAULT
                    ),
                    args.warmup,
                    args.iters,
                )
                fields.extend(
                    (
                        f"sglang_max_abs={sgl_diff:.4g}",
                        f"sglang={sgl_us:.2f} us",
                        f"ratio={cute_us / sgl_us:.2f}x",
                    )
                )
            print("  ".join(fields))


if __name__ == "__main__":
    main()
