"""Benchmark the Kimi-K3 attention-residual aggregation used by SGLang.

This file intentionally studies one production primitive rather than the
paper's speculative two-phase schedule.  For every token, the primitive takes
``nvb`` frozen block snapshots plus the current prefix, assigns one scalar
score to each row, applies softmax across rows, forms a weighted sum of the raw
rows, and applies the consumer's RMSNorm.

Compared implementations:

* ``torch_eager``: small, readable reference implementation.
* ``torch_compile``: the same Python implementation compiled by Inductor.
* ``sglang_triton``: SGLang's portable score/combine/RMSNorm fallback.
* ``sglang_tma``: SGLang's production SM100+ one-pass persistent kernel.
* ``cute_dsl``: two-pass experimental CuTe DSL implementation.
* ``cute_dsl_v2``: single-load, register-retained CuTe DSL experiment.

The timed unit is one aggregation point.  A Kimi-K3 decoder layer executes two
such points (attention-side and MLP-side), so the report also includes
``decoder_layer_us = 2 * aggregation_us``.  Attention, o_proj communication,
and the MLP itself are deliberately excluded.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SGLANG_PYTHON = WORKSPACE / "sglang-opensource" / "python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SGLANG_PYTHON))

from common.kernel_bench import bench_cuda, check_impls, print_table, write_csv

DTYPE = torch.bfloat16
DEVICE = "cuda"
HIDDEN_SIZE = 7168
MAX_BANK_ROWS = 8
MAX_SCORE_ROWS = 16
BLOCK_H = 1024
EPS = 1.0e-6


@dataclass
class Inputs:
    prefix: torch.Tensor
    bank: torch.Tensor
    cw: torch.Tensor
    ow: torch.Tensor
    nvb: int

    @property
    def tokens(self) -> int:
        return self.prefix.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.prefix.shape[1]

    @property
    def rows(self) -> int:
        return self.nvb + 1


def make_inputs(tokens: int, hidden_size: int, nvb: int) -> Inputs:
    if not 1 <= nvb <= MAX_BANK_ROWS:
        raise ValueError(f"nvb must be in [1, {MAX_BANK_ROWS}], got {nvb}")
    generator = torch.Generator(device=DEVICE).manual_seed(20260729)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        value = torch.randn(
            *shape,
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
        return (value * scale).to(DTYPE)

    return Inputs(
        prefix=randn(tokens, hidden_size),
        bank=randn(tokens, MAX_BANK_ROWS, hidden_size),
        # cw is score_norm.weight * score_proj.weight.  The scale keeps random
        # logits in a useful range; trained checkpoints provide this vector.
        cw=randn(hidden_size, scale=hidden_size**-0.5).contiguous(),
        ow=randn(hidden_size, scale=0.1).add_(1.0).contiguous(),
        nvb=nvb,
    )


def attention_residual_python(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    nvb: int,
    eps: float = EPS,
) -> torch.Tensor:
    """Readable implementation of SGLang's exact aggregation contract.

    ``cw`` already folds the key RMSNorm weight into the scalar projection:

        score(x) = dot(x, cw) / sqrt(mean(x**2) + eps)
        output   = RMSNorm(sum(softmax(scores)[i] * x[i]))
    """
    rows = torch.cat((bank[:, :nvb], prefix.unsqueeze(1)), dim=1)
    rows_fp32 = rows.float()
    row_rrms = torch.rsqrt(rows_fp32.square().mean(dim=-1) + eps)
    scores = torch.einsum("trh,h->tr", rows_fp32, cw.float())
    scores = scores * row_rrms
    probabilities = torch.softmax(scores, dim=-1)
    mixed = torch.einsum("tr,trh->th", probabilities, rows_fp32)
    mixed = mixed.to(prefix.dtype)
    return F.rms_norm(mixed, (mixed.shape[-1],), weight=ow, eps=eps)


def build_torch_eager(inputs: Inputs) -> Callable[[], torch.Tensor]:
    return lambda: attention_residual_python(
        inputs.prefix,
        inputs.bank,
        inputs.cw,
        inputs.ow,
        inputs.nvb,
    )


def build_torch_compile(inputs: Inputs) -> Callable[[], torch.Tensor]:
    compiled = torch.compile(
        attention_residual_python,
        fullgraph=True,
        dynamic=False,
    )
    # Compile before timing. nvb is a Python integer and therefore specializes
    # the graph to the source count of this layer.
    compiled(
        inputs.prefix,
        inputs.bank,
        inputs.cw,
        inputs.ow,
        inputs.nvb,
    )
    return lambda: compiled(
        inputs.prefix,
        inputs.bank,
        inputs.cw,
        inputs.ow,
        inputs.nvb,
    )


def build_sglang_triton(inputs: Inputs) -> Callable[[], torch.Tensor]:
    """Build SGLang's current three-launch portable fallback."""
    if inputs.hidden_size % BLOCK_H:
        raise ValueError("SGLang Triton path requires H divisible by 1024")

    from sgl_kernel import rmsnorm
    from sglang.srt.layers.attn_residual import (  # type: ignore[import-not-found]
        _combine_kernel,
        _score_kernel,
    )

    scores = torch.empty(
        (inputs.tokens, MAX_SCORE_ROWS),
        dtype=torch.float32,
        device=inputs.prefix.device,
    )
    mixed = torch.empty_like(inputs.prefix)

    def run() -> torch.Tensor:
        _score_kernel[(inputs.tokens, inputs.rows)](
            inputs.prefix,
            inputs.bank,
            inputs.cw.float(),
            scores,
            inputs.nvb,
            EPS,
            inputs.prefix.stride(0),
            inputs.bank.stride(0),
            inputs.bank.stride(1),
            scores.stride(0),
            H=inputs.hidden_size,
            BLOCK_H=BLOCK_H,
            num_warps=8,
        )
        _combine_kernel[(inputs.tokens, inputs.hidden_size // BLOCK_H)](
            inputs.prefix,
            inputs.bank,
            scores,
            mixed,
            inputs.nvb,
            inputs.prefix.stride(0),
            inputs.bank.stride(0),
            inputs.bank.stride(1),
            scores.stride(0),
            mixed.stride(0),
            BLOCK_H=BLOCK_H,
            MAX_ROWS=MAX_SCORE_ROWS,
            num_warps=4,
        )
        return rmsnorm(mixed, inputs.ow, EPS)

    return run


def build_sglang_tma(inputs: Inputs) -> Callable[[], torch.Tensor]:
    """Build SGLang's production H=7168, SM100+ fused implementation."""
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("SGLang TMA kernel requires SM100+")
    if inputs.hidden_size != HIDDEN_SIZE:
        raise ValueError("SGLang TMA kernel is specialized to H=7168")

    from sglang.kernels.ops.kimi_k3.attn_res import (  # type: ignore[import-not-found]
        attn_res_fused_tma,
    )

    output = torch.empty_like(inputs.prefix)

    def run() -> torch.Tensor:
        attn_res_fused_tma(
            inputs.prefix,
            inputs.bank,
            inputs.cw,
            inputs.ow,
            output,
            inputs.nvb,
            EPS,
        )
        return output

    return run


def build_cute_dsl(inputs: Inputs) -> Callable[[], torch.Tensor]:
    """Load the optional CuTeDSLGen candidate without making it mandatory."""
    module = importlib.import_module("attn_res.cute_attn_res_candidate")
    output = torch.empty_like(inputs.prefix)

    def run() -> torch.Tensor:
        module.attn_res_cute(
            inputs.prefix,
            inputs.bank,
            inputs.cw,
            inputs.ow,
            output,
            inputs.nvb,
            EPS,
        )
        return output

    # Trigger CuTe compilation before correctness checks and timing.
    run()
    return run


def build_cute_dsl_v2(inputs: Inputs) -> Callable[[], torch.Tensor]:
    """Load the register-retained, single-source-load CuTe experiment."""
    module = importlib.import_module("attn_res.cute_attn_res_candidate_v2")
    output = torch.empty_like(inputs.prefix)

    def run() -> torch.Tensor:
        module.attn_res_cute_v2(
            inputs.prefix,
            inputs.bank,
            inputs.cw,
            inputs.ow,
            output,
            inputs.nvb,
            EPS,
        )
        return output

    run()
    return run


BUILDERS: dict[str, Callable[[Inputs], Callable[[], torch.Tensor]]] = {
    "torch_eager": build_torch_eager,
    "torch_compile": build_torch_compile,
    "sglang_triton": build_sglang_triton,
    "sglang_tma": build_sglang_tma,
    "cute_dsl": build_cute_dsl,
    "cute_dsl_v2": build_cute_dsl_v2,
}


def ideal_bytes(inputs: Inputs, implementation: str) -> int:
    """Algorithmic HBM bytes for one aggregation, excluding weight residency.

    The production TMA kernel consumes each source row once and writes one
    output row. The current CuTe baseline deliberately reloads every source row
    for the combine pass. Triton does the same and additionally materializes
    the mixed row for its separate RMSNorm kernel.
    """
    element_bytes = inputs.prefix.element_size()
    if implementation == "sglang_triton":
        vector_elements = 2 * inputs.rows * inputs.tokens * inputs.hidden_size
        vector_elements += 2 * inputs.tokens * inputs.hidden_size
    elif implementation == "cute_dsl":
        vector_elements = (2 * inputs.rows + 1)
        vector_elements *= inputs.tokens * inputs.hidden_size
    else:
        vector_elements = (inputs.rows + 1) * inputs.tokens * inputs.hidden_size
    return vector_elements * element_bytes


def approximate_flops(inputs: Inputs) -> int:
    # Per source: square reduction (~H), score dot (~2H), weighted FMA (~2H).
    # Output RMSNorm contributes square reduction + scale (~3H).
    return (5 * inputs.rows + 3) * inputs.tokens * inputs.hidden_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the production Kimi-K3 attention-residual primitive."
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--nvb", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 16, 256, 4096, 16384],
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        choices=list(BUILDERS),
        default=list(BUILDERS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--timing",
        choices=("graph", "eager"),
        default="graph",
        help=(
            "CUDA-graph replay models serving; eager compares host launch paths. "
            "The experimental CuTe TVM-FFI wrapper is eager-only."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "results" / "bench_attn_res_current.csv",
    )
    args = parser.parse_args()

    print(
        "Contract: rows=[bank[:nvb], prefix] -> score/RMS -> softmax -> "
        "weighted raw rows -> output RMSNorm"
    )
    print(
        f"Device: {torch.cuda.get_device_name()} "
        f"(SM{torch.cuda.get_device_capability()[0]}"
        f"{torch.cuda.get_device_capability()[1]})"
    )

    all_rows: list[dict] = []
    for nvb in args.nvb:
        for tokens in args.tokens:
            inputs = make_inputs(tokens, args.hidden_size, nvb)
            runners: dict[str, Callable[[], torch.Tensor]] = {}
            unavailable: dict[str, str] = {}
            for name in args.impl:
                try:
                    if name.startswith("cute_dsl") and args.timing == "graph":
                        raise RuntimeError(
                            "CuTe TVM-FFI wrapper is not CUDA-graph safe; "
                            "rerun with --timing eager"
                        )
                    runners[name] = BUILDERS[name](inputs)
                except Exception as exc:
                    unavailable[name] = f"{type(exc).__name__}: {exc}"

            reference = attention_residual_python(
                inputs.prefix,
                inputs.bank,
                inputs.cw,
                inputs.ow,
                inputs.nvb,
            )
            correctness = check_impls(
                runners,
                reference,
                normalize=lambda _name, output: output,
                atol=3e-2,
                rtol=2e-2,
            )
            print_table(
                correctness,
                columns=["impl", "pass", "max_abs", "cosine", "error"],
                title=f"Correctness: T={tokens}, H={args.hidden_size}, nvb={nvb}",
            )

            def row_extra(name: str, latency_us: float) -> dict:
                bytes_moved = ideal_bytes(inputs, name)
                return {
                    "tokens": tokens,
                    "nvb": nvb,
                    "rows": inputs.rows,
                    "aggregation_us": round(latency_us, 3),
                    "decoder_layer_us": round(2 * latency_us, 3),
                    "ideal_GB": round(bytes_moved / 1e9, 6),
                    "ideal_GBps": round(bytes_moved / (latency_us * 1e3), 1),
                    "approx_TFLOPs": round(
                        approximate_flops(inputs) / (latency_us * 1e6),
                        3,
                    ),
                }

            timing_iters = (
                args.iters if tokens <= 4096 else min(args.iters, 10)
            )
            rows: list[dict] = []
            for name, runner in runners.items():
                try:
                    latency_us = bench_cuda(
                        runner,
                        warmup=args.warmup,
                        iters=timing_iters,
                        repeats=args.repeats,
                        use_graph=args.timing == "graph",
                    )
                    row = {
                        "impl": name,
                        "latency_us": latency_us,
                        "timing": args.timing,
                    }
                    row.update(row_extra(name, latency_us))
                except Exception as exc:
                    row = {
                        "impl": name,
                        "timing": args.timing,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)

            baseline_name = (
                "sglang_tma" if "sglang_tma" in runners else "torch_eager"
            )
            baseline_us = next(
                (
                    row["latency_us"]
                    for row in rows
                    if row["impl"] == baseline_name and "latency_us" in row
                ),
                None,
            )
            if baseline_us is not None:
                for row in rows:
                    if "latency_us" in row:
                        row["speedup"] = baseline_us / row["latency_us"]
            for row in rows:
                row["device"] = torch.cuda.get_device_name()
                row["hidden_size"] = args.hidden_size
                row["available"] = True
            for name, reason in unavailable.items():
                rows.append(
                    {
                        "impl": name,
                        "tokens": tokens,
                        "nvb": nvb,
                        "rows": inputs.rows,
                        "device": torch.cuda.get_device_name(),
                        "hidden_size": args.hidden_size,
                        "available": False,
                        "error": reason,
                    }
                )
            print_table(
                rows,
                columns=[
                    "impl",
                    "aggregation_us",
                    "decoder_layer_us",
                    "speedup",
                    "ideal_GBps",
                    "approx_TFLOPs",
                    "error",
                ],
                title=f"Latency: T={tokens}, H={args.hidden_size}, nvb={nvb}",
            )
            all_rows.extend(rows)
            del inputs, runners, reference
            torch.cuda.empty_cache()

    write_csv(all_rows, args.csv)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
