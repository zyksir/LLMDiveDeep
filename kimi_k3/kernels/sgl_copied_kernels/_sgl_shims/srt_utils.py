"""Stand-ins for the ``sglang.srt.utils`` symbols the vendored
CustomAllReduceV2 host plumbing imports (beyond what ``__init__.py`` already
provides). Semantics follow sglang commit
f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e ``srt/utils/common.py``:

* ``is_cuda`` / ``is_hip`` / ``is_musa`` — faithful copies of the platform
  probes (``is_musa`` keeps the ``torchada`` import gate).
* ``log_info_on_rank0`` — upstream resolves rank through
  ``get_parallel().tp_rank`` and falls back to ``torch.distributed.get_rank``
  when that context is unavailable; this harness never has the sglang runtime
  context, so the fallback branch IS the behaviour here.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def is_cuda() -> bool:
    return torch.cuda.is_available() and torch.version.cuda is not None


@lru_cache(maxsize=1)
def is_hip() -> bool:
    return torch.version.hip is not None


@lru_cache(maxsize=1)
def is_musa() -> bool:
    try:
        import torchada  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.version, "musa") and torch.version.musa is not None


def log_info_on_rank0(logger, msg) -> None:
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(msg)
    else:
        logger.info(msg)


__all__ = ["is_cuda", "is_hip", "is_musa", "log_info_on_rank0"]
