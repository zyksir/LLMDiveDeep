"""Vendored sglang Kimi-K3 MoE kernels (JIT CUDA + CuTe DSL).

Faithful copies from sglang commit f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e
(see README.md for the file-by-file provenance). Only import roots were
rewritten (``sglang.kernels`` -> this package; ``sglang.srt``/``sglang.utils``
symbols -> ``._sgl_shims``); kernel sources and loader logic are unchanged.

This top-level module is deliberately import-light: the kernel loaders pull in
``torch``, ``tvm_ffi``, ``msgspec`` and (at build time) ``ninja`` + a CUDA
toolchain, none of which are needed just to import the package. Import the
subpackages you need directly, e.g.::

    from kimi_k3.kernels.sgl_copied_kernels.ops.moe import moe_route_radix
    from kimi_k3.kernels.sgl_copied_kernels.ops.kimi_k3 import all_reduce
"""
