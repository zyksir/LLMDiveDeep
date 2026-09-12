"""Stand-in for ``sglang.srt.environ.envs`` restricted to the environment
variables the vendored CustomAllReduceV2 host plumbing reads. Names and
defaults follow sglang commit f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e
``srt/environ.py``:

* ``SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB``            EnvInt(16 * 1024)
* ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB``     EnvInt(None)
* ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB``     EnvInt(None)
* ``SGLANG_MEMORY_SAVER_CUDA_GRAPH``                     EnvBool(False)
* ``SGLANG_CACHE_DIR``                                   EnvStr(~/.cache/sglang)

The field classes are reused from ``__init__.py`` (same read semantics as
sglang's EnvBool/EnvInt/EnvStr for the get()/is_set() surface used here).
"""

from __future__ import annotations

import os

from kimi_k3.kernels.sgl_copied_kernels._sgl_shims import (
    _EnvBool,
    _EnvInt,
    _EnvStr,
)


class _CarV2Envs:
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = _EnvInt(
        "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB", 16 * 1024
    )
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = _EnvInt(
        "SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB", None
    )
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = _EnvInt(
        "SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB", None
    )
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = _EnvBool("SGLANG_MEMORY_SAVER_CUDA_GRAPH", False)
    SGLANG_CACHE_DIR = _EnvStr(
        "SGLANG_CACHE_DIR", os.path.expanduser("~/.cache/sglang")
    )


envs = _CarV2Envs()

__all__ = ["envs"]
