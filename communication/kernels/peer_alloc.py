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

  K3_PEER_ALLOC=nvl | symm | ipc | auto   (default auto: nvl (cuMem
                                     FABRIC+IMEX, true multi-node) then ipc
                                     then symm)

Returns (buffer_uint8, peer_ptrs_int64_cuda, keepalive_handle).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


class _NvlAllocation:
    """Keepalive for one fabric allocation (unmap+release on GC)."""

    def __init__(self, base, aligned, handles, world):
        self.base, self.aligned = base, aligned
        self.handles, self.world = handles, world

    def __del__(self):
        try:
            try:
                from cuda.bindings import driver as cuda
            except ImportError:
                from cuda import cuda  # noqa: delayed; may be torn down
            for i in range(self.world):
                cuda.cuMemUnmap(self.base + i * self.aligned, self.aligned)
            for h in self.handles:
                cuda.cuMemRelease(h)
            cuda.cuMemAddressFree(self.base, self.aligned * self.world)
        except Exception:
            pass


def _cu(res):
    err, *vals = res
    if err != 0 and getattr(err, "value", 1) != 0:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return vals[0] if len(vals) == 1 else (vals or None)


def _alloc_nvl(group, rank: int, world: int, nbytes: int):
    """cuMemCreate(FABRIC) + IMEX, handles exchanged over torch.dist.

    Layout matches TRT-LLM MnnvlMemory: one VA reservation of
    world*aligned bytes, rank i's segment mapped at base + i*aligned,
    so the peer table is base-strided.
    """
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    from tensorrt_llm._dlpack_utils import pack_strided_memory

    dev_id = torch.cuda.current_device()
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = dev_id
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = \
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop.location = location

    gran = _cu(cuda.cuMemGetAllocationGranularity(
        prop=prop,
        option=cuda.CUmemAllocationGranularity_flags.
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
    aligned = (nbytes + gran - 1) // gran * gran

    handle = _cu(cuda.cuMemCreate(aligned, prop, flags=0))
    exported = _cu(cuda.cuMemExportToShareableHandle(
        handle, prop.requestedHandleTypes, 0))
    # fabric handle payload is bytes; exchange over the torch group
    all_data = [None] * world
    dist.all_gather_object(all_data, bytes(exported.data), group=group)

    base = int(_cu(cuda.cuMemAddressReserve(aligned * world, gran, 0, 0)))
    madesc = cuda.CUmemAccessDesc()
    madesc.location = location
    madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    handles = []
    for i, data in enumerate(all_data):
        ptr = base + i * aligned
        if i == rank:
            h = handle
        else:
            h = _cu(cuda.cuMemImportFromShareableHandle(
                data, prop.requestedHandleTypes))
        handles.append(h)
        _cu(cuda.cuMemMap(ptr, aligned, 0, h, 0))
    _cu(cuda.cuMemSetAccess(base, aligned * world, [madesc], 1))

    keep = _NvlAllocation(base, aligned, handles, world)
    strided = pack_strided_memory(base, nbytes, aligned, world,
                                  torch.uint8, dev_id)
    buf = strided[rank]
    ptrs = torch.tensor([base + i * aligned for i in range(world)],
                        dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    dist.barrier(group)
    return buf, ptrs, keep



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
    if mode in ("auto", "nvl"):
        # cuMemCreate(FABRIC)+IMEX with handles exchanged over
        # torch.distributed: works under MPI and plain torchrun, and
        # unlike torch symmetric memory it is genuinely multi-node.
        try:
            buf, ptrs, keep = _alloc_nvl(group or dist.group.WORLD, rank,
                                        world, nbytes)
            if sentinel_i16 is not None:
                buf.view(torch.int16).fill_(sentinel_i16)
            return buf, ptrs, keep, "nvl"
        except Exception:
            if mode == "nvl":
                raise
    if mode in ("auto", "ipc"):
        try:
            from tensorrt_llm.mapping import Mapping
            mapping = Mapping(world_size=world, tp_size=world, rank=rank,
                              gpus_per_node=min(
                                  world, torch.cuda.device_count()))
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
