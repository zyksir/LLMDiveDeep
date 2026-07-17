"""Timing + printing helpers shared by every LLMDiveDeep bench script.

These two functions used to be copy-pasted into each benchmark file. They
are generic (no sparse-attention / quantization specifics) so they live here
and are imported by both the `sparse_attn` and `quantization` suites.
"""

from __future__ import annotations

from typing import Callable

import torch


def bench_function(
    func: Callable, *args, warmup: int = 10, iters: int = 50, **kwargs
) -> float:
    """Return mean GPU-event-timed ms over `iters`, after `warmup` runs."""
    for _ in range(warmup):
        func(*args, **kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def print_section(title: str) -> None:
    """Print a boxed section header."""
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
