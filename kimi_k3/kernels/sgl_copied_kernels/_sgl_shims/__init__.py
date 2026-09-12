"""Local stand-ins for the ``sglang.srt`` / ``sglang.utils`` symbols that the
vendored kernel loaders import, so the copies run without a sglang install.

Only the exact call surface the copied files use is implemented. Semantics
follow sglang commit f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e:

* ``envs`` — the four ``SGLANG_JIT_*`` vars the JIT build stack reads
  (``sglang/srt/environ.py`` declares them as EnvBool/EnvStr/EnvInt).
* ``is_in_ci`` — ``sglang/utils.py`` (reads ``SGLANG_IS_IN_CI``).
* ``get_cuda_version`` / ``is_npu`` / ``is_sm100_supported`` /
  ``direct_register_custom_op`` — ``sglang/srt/utils/common.py``.

``custom_op.py`` next to this file is a faithful copy of
``sglang/srt/utils/custom_op.py`` (only import paths adjusted), not a shim.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, List, Optional


class _EnvField:
    def __init__(self, name: str, default):
        self._name = name
        self._default = default

    def is_set(self) -> bool:
        return self._name in os.environ

    def _raw(self) -> Optional[str]:
        return os.environ.get(self._name)


class _EnvBool(_EnvField):
    def get(self) -> bool:
        raw = self._raw()
        if raw is None:
            return self._default
        return raw.strip().lower() in ("1", "true", "yes", "on", "y")


class _EnvStr(_EnvField):
    def get(self) -> Optional[str]:
        raw = self._raw()
        return self._default if raw is None else raw


class _EnvInt(_EnvField):
    def get(self) -> Optional[int]:
        raw = self._raw()
        return self._default if raw is None else int(raw)


class _Envs:
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = _EnvBool(
        "SGLANG_JIT_KERNEL_RUN_FULL_TESTS", False
    )
    SGLANG_JIT_CACHE_DIR = _EnvStr("SGLANG_JIT_CACHE_DIR", None)
    SGLANG_JIT_CACHE_DEBUG = _EnvBool("SGLANG_JIT_CACHE_DEBUG", False)
    SGLANG_JIT_CACHE_KEEP = _EnvInt("SGLANG_JIT_CACHE_KEEP", None)


envs = _Envs()


def is_in_ci() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@functools.lru_cache(maxsize=1)
def is_npu() -> bool:
    import torch

    if not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:  # noqa: BLE001 - partial torch_npu installs
        return False


def get_cuda_version() -> tuple:
    import torch

    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


@functools.lru_cache(maxsize=1)
def is_sm100_supported(device=None) -> bool:
    """Datacenter Blackwell (SM10x) with a CUDA >= 12.8 runtime, matching
    sglang's ``_check_cuda_device_version(majors=[10], cuda_version=(12, 8))``."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major == 10 and get_cuda_version() >= (12, 8)


_sglang_lib = None


def _get_sglang_lib():
    global _sglang_lib
    if _sglang_lib is None:
        from torch.library import Library

        _sglang_lib = Library("sglang", "FRAGMENT")  # noqa: TOR901
    return _sglang_lib


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib=None,
) -> None:
    """Register ``op_func`` under ``torch.ops.sglang.<op_name>`` (CUDA dispatch
    key), skipping silently when already registered — the behaviour of
    sglang's ``direct_register_custom_op`` minus the NPU/XPU/MUSA branches."""
    import torch
    import torch.library

    my_lib = target_lib or _get_sglang_lib()
    try:
        lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "sglang"
        if hasattr(torch.ops, lib_name) and hasattr(
            getattr(torch.ops, lib_name), op_name
        ):
            return
    except (AttributeError, RuntimeError):
        pass

    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    try:
        my_lib.define(op_name + schema_str)
        my_lib.impl(op_name, op_func, "CUDA")
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as error:
        if "Tried to register an operator" in str(error) and "multiple times" in str(
            error
        ):
            pass
        else:
            raise


__all__ = [
    "envs",
    "is_in_ci",
    "is_npu",
    "get_cuda_version",
    "is_sm100_supported",
    "direct_register_custom_op",
]
