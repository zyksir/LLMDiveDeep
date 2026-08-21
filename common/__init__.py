"""Shared helpers for the LLMDiveDeep micro-benchmarks.

Everything lives in ``common/kernel_bench.py`` — the single kernel-bench
framework module (timing, correctness, registry, table/CSV/figure output).

Imports are lazy (PEP 562): ``kernel_bench`` imports torch, and eagerly
importing it here broke the advertised no-GPU self-checks
(``python3 -m common.arch``, ``python3 -m kimi_k3_layer.capabilities``)
on any host without torch. ``common.arch`` stays importable bare.
"""

__all__ = [
    "BackendRegistry",
    "KernelBench",
    "bench_cuda",
    "bench_impls",
    "check_impls",
    "plot_rows",
    "print_section",
    "print_table",
    "write_csv",
]


def __getattr__(name):
    if name in __all__:
        from common import kernel_bench

        return getattr(kernel_bench, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
