"""Kernel-level bench scripts for kimi_k3/kernels/.

One standalone script per kernel entry point (correctness vs a reference
plus CUDA-event timing); run directly, e.g.::

    python3 kimi_k3/kernels/benchmarks/bench_rmsnorm.py

LAYER-level benches (``bench_b10_kimi_k3_{moe,kda}_layer.py`` and the
mini/mla ones) stay at the ``kimi_k3/`` package root — this directory is
only for single-kernel benches.
"""
