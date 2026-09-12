"""CustomAllReduceV2 workspace setup for the sglang K3 fused-AR kernels.

Builds the multicast push/pull workspace planes the fused-AR tails of the
SGL blocks need, outside a sglang server, from a plain ``torch.distributed``
process group. The heavy lifting is the vendored copy of sglang's
``custom_all_reduce_v2.py`` under ``sgl_copied_kernels/_sgl_shims/`` (see
that package's README for provenance); this module wires it to the bench
environment and mirrors sglang's own lazy-state pattern
(``k3_ar_fusion._get_state``):

* :func:`sgl_ar_available` — cheap capability probe (SM arch, world size,
  NVLink multicast), no collectives, no JIT builds.
* :func:`build_sgl_ar_state` — collective constructor. Every rank must call
  it. Returns an :class:`SglArState` that OWNS the CustomAllReduceV2
  wrapper (and through it the symmetric-memory slab) — no attributes are
  monkey-patched onto the tvm-ffi ``Communicator``, which rejects instance
  attributes.
* :func:`get_sgl_ar_state` — process-lazy cached state, the entry point the
  SGL blocks self-initialize through on first forward.

Mirroring sglang (which hands its gloo ``cpu_group`` to CustomAllReduceV2),
the group must NOT be a NCCL group: the rendezvous runs CPU-side collectives
(``all_gather_object``, CPU-tensor ``all_reduce``, shared-memory node probe).
Pass ``group=None`` to build (once) a gloo group spanning WORLD.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.distributed as dist

__all__ = [
    "SglArState",
    "build_sgl_ar_state",
    "get_sgl_ar_state",
    "sgl_ar_available",
]

logger = logging.getLogger(__name__)

# Lazily created gloo group for group=None callers; one per process, like
# sglang's world cpu_group.
_GLOO_GROUP: Optional[dist.ProcessGroup] = None


class SglArState(NamedTuple):
    """Live fused-AR state, sglang ``k3_ar_fusion._State`` shaped.

    Owns the lifetime chain: ``comm`` holds raw pointers into the
    symmetric-memory slab owned by ``car_v2``, so holding the state keeps
    the workspaces alive. ``comm`` is already registered with the vendored
    AR ops (``register_comm``) by the constructors below.
    """

    comm: Any  # tvm-ffi Communicator (what register_comm / the ops consume)
    car_v2: Any  # owning CustomAllReduceV2 wrapper (slab keepalive, close())
    world_size: int
    max_push_size: int  # bytes per push slot (push-fits checks read this)
    max_pull_size: int
    has_multicast: bool


def _default_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _has_multicast_support(device: torch.device) -> bool:
    """NVLink multicast (NVLS) support via torch symmetric memory.

    This is the same capability ``_SymmetricMemory.rendezvous`` needs to
    return a nonzero ``multicast_ptr``, which the fused push kernel
    host-checks (``ar_fusion.cuh``: "the fused push needs a multicast-capable
    push plane").
    """
    from torch._C._autograd import DeviceType
    from torch._C._distributed_c10d import _SymmetricMemory

    return bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index))


def sgl_ar_available(
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> bool:
    """Cheap, non-collective probe: can ``build_sgl_ar_state`` succeed here?

    Checks the fused-AR preconditions sglang's ``k3_ar_fusion`` auto-probe
    and ``can_use_custom_all_reduce_v2`` gate on:

    * CUDA device with SM 100/103 (datacenter Blackwell — the ar_fusion
      kernels and tuned configs target B200/B300/GB200/GB300),
    * torch.distributed initialized with a supported world size (> 1, in the
      tuned config table),
    * NVLink multicast support on the device.

    The full gate (same-node probe, full-NVLink pynvml check, P2P check) runs
    collectively inside ``build_sgl_ar_state``; this probe never blocks.
    """
    if not torch.cuda.is_available():
        return False
    device = device if device is not None else _default_device()
    major, minor = torch.cuda.get_device_capability(device)
    if major * 10 + minor not in (100, 103):
        return False
    if not dist.is_initialized():
        return False
    world_size = dist.get_world_size(group=group)
    # Import here: pulls in the vendored config table only (torch + stdlib).
    from ..sgl_copied_kernels._sgl_shims.configs.custom_all_reduce_v2 import (
        get_supported_world_sizes,
    )

    if world_size <= 1 or world_size not in get_supported_world_sizes():
        return False
    return _has_multicast_support(device)


def _resolve_group(group: Optional[dist.ProcessGroup]) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError(
            "build_sgl_ar_state requires torch.distributed to be initialized"
        )
    if group is None:
        global _GLOO_GROUP
        if _GLOO_GROUP is None:
            # Collective: every rank reaches this together on first build.
            _GLOO_GROUP = dist.new_group(backend="gloo")
        return _GLOO_GROUP
    backend = dist.get_backend(group)
    if backend is not None and "nccl" in str(backend).lower():
        raise ValueError(
            "build_sgl_ar_state needs a non-NCCL (e.g. gloo) group: the "
            "CustomAllReduceV2 rendezvous runs CPU-side collectives, exactly "
            "like sglang hands its cpu_group to CustomAllReduceV2. Pass "
            "group=None to have one created over WORLD, or build one with "
            'dist.new_group(backend="gloo").'
        )
    return group


def build_sgl_ar_state(
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    *,
    max_push_bytes: Optional[int] = None,
    max_pull_bytes: Optional[int] = None,
) -> SglArState:
    """Build the CustomAllReduceV2 workspaces; return the owning state.

    Collective over ``group`` (every rank must call with the same sizes).
    JIT-compiles the vendored communicator registry on first use, allocates
    one multicast-bound symmetric-memory slab per rank
    (``[2 * world_size push slots | pull buffer | pull semaphores]``) and
    registers the communicator with the vendored AR ops.

    :param group: non-NCCL process group to reduce over (``None`` = a gloo
                  group over WORLD, created on first call). The FUSED AR
                  itself runs over NVLink on the GPUs of these ranks.
    :param device: this rank's CUDA device (default: current device).
    :param max_push_bytes: per-slot push workspace bytes. ``None`` takes the
                  tuned per-arch/world-size config (768 KB for SM100 TP8) —
                  sglang's own default. The [T, 3584] bf16 K3 latent fits
                  512 KB for T <= 73, so the default covers the MTP decode
                  batches.
    :param max_pull_bytes: pull buffer bytes; ``0`` builds a push-only
                  instance (no pull plane — enough for the
                  ``finalize_all_reduce_push_norm`` / ``all_reduce_push_res``
                  tails). ``None`` takes the tuned config clipped to 16 MB.
    :raises RuntimeError: when the capability gate rejects this topology or
                  the allocation came back without a multicast binding.
    """
    device = device if device is not None else _default_device()
    group = _resolve_group(group)

    from ..sgl_copied_kernels._sgl_shims.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    kwargs = {}
    if max_push_bytes is not None:
        kwargs["max_push_size"] = int(max_push_bytes)
    if max_pull_bytes is not None:
        kwargs["max_pull_size"] = int(max_pull_bytes)
    v2 = CustomAllReduceV2(group=group, device=device, **kwargs)
    if v2.disabled:
        raise RuntimeError(
            "CustomAllReduceV2 gate rejected this setup (see log warnings): "
            "needs a single-node full-NVLink group with a supported world "
            f"size, got world_size={dist.get_world_size(group=group)} on "
            f"{device}. sgl_ar_available() pre-checks the local conditions."
        )
    if not v2.has_multicast:
        v2.close()
        raise RuntimeError(
            "symmetric-memory allocation has no multicast binding "
            "(multicast_ptr == 0); the K3 fused push AR requires an NVLS-"
            "capable NVLink domain."
        )

    from ..sgl_copied_kernels.ops.kimi_k3 import all_reduce as ar_ops

    ar_ops.register_comm(v2.obj)
    return SglArState(
        comm=v2.obj,
        car_v2=v2,
        world_size=int(v2.obj.world_size),
        max_push_size=int(v2.max_push_size),
        max_pull_size=int(v2.max_pull_size),
        has_multicast=bool(v2.has_multicast),
    )


# Cache-once per group, mirroring sglang's ``@cache_once _get_state()``: the
# resolution (including a None "unavailable" verdict) is FROZEN at first
# call. Keyed by the group object itself (strong ref, identity hash), None
# for the default WORLD group.
_STATE_CACHE: Dict[Optional[dist.ProcessGroup], Optional[SglArState]] = {}


def get_sgl_ar_state(
    group: Optional[dist.ProcessGroup] = None,
) -> Optional[SglArState]:
    """Process-lazy fused-AR state — sglang's ``k3_ar_fusion._get_state``.

    First call probes :func:`sgl_ar_available`; when the probe passes it
    runs the COLLECTIVE :func:`build_sgl_ar_state`, so every rank must reach
    the first call together (the SGL blocks call this from forward, which
    runs in rank lockstep). The verdict — state or None — is cached for the
    process lifetime, so call it only once torch.distributed is in its final
    state (an "unavailable" probe is frozen too, exactly like sglang's
    ``cache_once``).
    """
    if group in _STATE_CACHE:
        return _STATE_CACHE[group]
    if not sgl_ar_available(group=group):
        logger.info(
            "K3 fused AR unavailable (arch/world-size/multicast probe "
            "failed); SGL blocks use their documented fallback tails."
        )
        _STATE_CACHE[group] = None
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "first get_sgl_ar_state() call landed inside CUDA graph "
            "capture; the collective workspace build cannot run there. Run "
            "one eager forward (warmup) before capturing."
        )
    state = build_sgl_ar_state(group)
    logger.info("K3 fused AR enabled (world_size=%d)", state.world_size)
    _STATE_CACHE[group] = state
    return state
