#!/usr/bin/env python3
"""Per-kernel timing for KDA prefill backends via torch.profiler.

FlashKDA exposes a single ``fwd`` binding, so its two internal kernels
(K1 ``_flash_kda_fwd_prepare``, K2 ``_flash_kda_fwd_recurrence``) cannot be
called separately from Python. They *are* separate kernel launches, though,
so the CUDA profiler attributes time to each. The same trick gives the
SGLang chunk pipeline's leaf-kernel breakdown on identical inputs, so both
backends can be compared stage by stage.

Usage: ../.venv/bin/python profile_kda_stage_kernels.py
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from kda_attention import KDA_PREFILL, make_prefill_inputs
from linear_attention import Shape

ITERS = 20


def profile_backend(name: str, batch: int, seq_len: int, shape: Shape):
    inputs = make_prefill_inputs(batch, seq_len, shape, seed=batch + seq_len)
    runners, unavailable = KDA_PREFILL.build(inputs, shape)
    if name not in runners:
        raise RuntimeError(f"{name} unavailable: {unavailable.get(name)}")
    fn = runners[name]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    totals: dict[str, float] = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            totals[evt.key] = evt.self_device_time_total / ITERS
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def short(key: str) -> str:
    key = key.split("<")[0]
    return key.rsplit("::", 1)[-1][:60]


def main():
    shape = Shape(16, 16, 128, 128, "float32")
    for batch, seq_len in [(1, 512), (1, 2048), (1, 8192), (4, 2048), (4, 8192)]:
        for name in ("flash_kda", "sglang_kda_chunk"):
            totals = profile_backend(name, batch, seq_len, shape)
            print(f"\n== {name}  B={batch} S={seq_len}  "
                  f"(sum {sum(totals.values()):.1f} us/iter)")
            for key, us in totals.items():
                if us < 0.5:
                    continue
                print(f"  {us:10.2f} us  {short(key)}")


if __name__ == "__main__":
    main()
