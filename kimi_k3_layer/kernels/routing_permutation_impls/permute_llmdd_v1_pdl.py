"""PDL variant of llmdd_permute_v1: overlap prologue with the route kernel.

Identical kernel body; launched with programmatic stream serialization and a
griddepcontrol.wait placed after the counter-zeroing prologue, so the launch
gap and prologue hide under the producing route kernel's tail.
"""

from __future__ import annotations

from kimi_k3_layer.kernels.routing_permutation_impls import permute_llmdd_v1 as _base

NAME = "b10_permute_v1_pdl"
KIND = "permute"
PROVENANCE = {**_base.PROVENANCE, "variant": "PDL launch"}


def load() -> None:
    _base.load()


def prepare(batch, device, **kwargs):
    return _base.prepare(batch, device, use_pdl=True, **kwargs)
