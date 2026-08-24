"""Peer-visible buffer allocation aligned with TRT-LLM's AllReduce.

Our custom lamport kernels (finalize_ar_norm, mega_tail, persistent_ar,
col_quant) only consume a per-rank peer-pointer table. Historically they
allocated via torch symmetric memory (CUDA IPC / NVLS), which is
intra-node only in most builds. TRT-LLM's own AllReduce survives GB300
cross-node because its workspace goes through:

  * IpcMemory  — cudaMalloc + legacy cudaIpc handles (allgathered over
    MPI). Intra-node always; cross-node wherever the IMEX daemon extends
    legacy IPC across the NVLink domain. This is the allocator behind
    the allreduce_fusion oneshot-lamport kernel that works cross-node.
  * MnnvlMemory — cuMemCreate(FABRIC) + IMEX channels for true
    multi-node NVLink, gated by MnnvlMemory.supports_mnnvl().

This module reuses those exact classes with TRT's gate order, so any
kernel built on it inherits TRT's cross-node behavior. Selection:

  K3_PEER_ALLOC=symm | ipc | auto   (default auto: ipc when available,
                                     symm as the fallback)

Returns (buffer_uint8, peer_ptrs_int64_cuda, keepalive_handle).
"""
from __future__ import annotations

import os

import torch


def _alloc_symm(group, nbytes: int):
    import torch.distributed._symmetric_memory as symm_mem
    buf = symm_mem.empty(nbytes, dtype=torch.uint8,
                         device=torch.device(
                             "cuda", torch.cuda.current_device()))
    hdl = symm_mem.rendezvous(buf, group.group_name)
    ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs],
                        dtype=torch.int64, device="cuda")
    return buf, ptrs, hdl


def _alloc_ipc(mapping, nbytes: int):
    """TRT-LLM's legacy-IPC allocator (IMEX-extended on NVL72 fabrics)."""
    from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
    if not can_access_peer(mapping):
        raise RuntimeError("peer access unsupported on this topology")
    mem = IpcMemory(mapping, nbytes, True)
    if not mem.open_ipc or mem.local_ptr == 0:
        # tp_size > gpus_per_node (set K3_GPUS_PER_NODE to the NVLink
        # DOMAIN size on NVL72 fabrics, as production launchers do)
        raise RuntimeError("IpcMemory did not open (check gpus_per_node)")
    ptrs = torch.tensor(mem.peer_ptrs, dtype=torch.int64, device="cuda")
    buf = _wrap_ptr_as_tensor(mem.local_ptr, nbytes)
    return buf, ptrs, mem


def _wrap_ptr_as_tensor(ptr: int, nbytes: int):
    """Minimal wrapper for a raw device pointer (no ownership).

    The kernels only consume data_ptr(); the sole tensor-like operation
    needed is the lamport sentinel fill at init, done via a D2D copy.
    """
    class _PtrBuf:
        def __init__(self, ptr, nbytes):
            self.ptr, self.nbytes = ptr, nbytes

        def data_ptr(self):
            return self.ptr

        def fill_int16(self, value: int) -> None:
            t = torch.full((self.nbytes // 2,), value, dtype=torch.int16,
                           device="cuda")
            import cuda.bindings.runtime as cudart
            cudart.cudaMemcpy(
                self.ptr, t.data_ptr(), self.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    return _PtrBuf(ptr, nbytes)


def alloc_peer_buffer(group, rank: int, world: int, nbytes: int,
                      sentinel_i16: int | None = None):
    """Allocate a peer-visible buffer per K3_PEER_ALLOC policy."""
    mode = os.environ.get("K3_PEER_ALLOC", "auto")
    if mode in ("auto", "ipc"):
        try:
            from tensorrt_llm.mapping import Mapping
            mapping = Mapping(world_size=world, tp_size=world, rank=rank,
                              gpus_per_node=int(os.environ.get(
                                  "K3_GPUS_PER_NODE", str(world))))
            buf, ptrs, keep = _alloc_ipc(mapping, nbytes)
            if sentinel_i16 is not None:
                buf.fill_int16(sentinel_i16)
            return buf, ptrs, keep, "ipc"
        except Exception:
            if mode == "ipc":
                raise
    buf, ptrs, keep = _alloc_symm(group, nbytes)
    if sentinel_i16 is not None:
        buf.view(torch.int16).fill_(sentinel_i16)
    return buf, ptrs, keep, "symm"
