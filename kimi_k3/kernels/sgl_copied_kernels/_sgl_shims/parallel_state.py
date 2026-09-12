"""Stand-in for the two ``sglang.srt.distributed.parallel_state`` symbols the
vendored CustomAllReduceV2 host plumbing uses.

* ``in_the_same_node_as`` — faithful copy of the function (sglang commit
  f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e ``parallel_state.py``), together
  with its one helper ``make_shm_name`` (copied from
  ``srt/utils/stale_shm_cleanup.py``). It is a real collective shared-memory
  probe, not a hardcoded "same node" answer.
* ``get_world_group`` — minimal stub. Upstream returns the WORLD
  GroupCoordinator; the only consumer in the vendored closure is
  ``custom_all_reduce_utils.gpu_p2p_access_check``, which uses
  ``.local_rank`` (to elect the one process that generates the P2P cache
  file) and ``.barrier()``. This bench is single-node one-process-per-GPU,
  so local rank falls back to the global rank when ``LOCAL_RANK`` is unset.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from multiprocessing import shared_memory
from typing import List
from unittest.mock import patch

import torch
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# make_shm_name — faithful copy from sglang srt/utils/stale_shm_cleanup.py
# --------------------------------------------------------------------------

_SGL_SHM_PREFIX = "sgl_shm"


def make_shm_name(kind: str) -> str:
    """Pid-stamped name (sgl_shm_<kind>_<pid>_<rand>) the sweep can reclaim."""
    return f"{_SGL_SHM_PREFIX}_{kind}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# in_the_same_node_as — faithful copy from sglang srt/distributed/parallel_state.py
# --------------------------------------------------------------------------


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert (
        torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
    ), "in_the_same_node_as should be tested with a non-NCCL group."
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor(
        [0] * world_size, dtype=torch.int32, device="cpu"
    )

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(
                    create=True, size=128, name=make_shm_name("nodecheck")
                )
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]


# --------------------------------------------------------------------------
# get_world_group — minimal stub (see module docstring)
# --------------------------------------------------------------------------


class _WorldGroupStub:
    @property
    def local_rank(self) -> int:
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        return torch.distributed.get_rank()

    def barrier(self) -> None:
        torch.distributed.barrier()


_WORLD_GROUP_STUB = _WorldGroupStub()


def get_world_group() -> _WorldGroupStub:
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "get_world_group stub requires torch.distributed to be initialized"
        )
    return _WORLD_GROUP_STUB


__all__ = ["in_the_same_node_as", "get_world_group", "make_shm_name"]
