#!/usr/bin/env python3
"""Host-side weight-load benchmark: TRT-LLM's current path vs alternatives.

No GPU needed - measures disk -> host-RAM only, which is the dominant term
for large checkpoints on a network filesystem.

Modes
  prefetch_load : what HfWeightLoader does today (full f.read() prefetch into
                  page cache, then safetensors.torch.load_file per file)
  load_only     : safetensors.torch.load_file per file (no prefetch pass)
  mmap_slice    : safe_open + get_slice, reading ONLY this rank's TP shard
  ranks         : N processes each running `prefetch_load` on the same dir,
                  i.e. the per-node duplication TRT-LLM pays today
"""
from __future__ import annotations
import argparse, glob, os, time, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


def files(d):
    return sorted(glob.glob(f"{d}/*.safetensors"))


def total_bytes(fs):
    return sum(os.path.getsize(f) for f in fs)


def prefetch(fs, workers):
    def one(f):
        with open(f, "rb") as h:
            while h.read(1 << 24):
                pass
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, fs))


def load_only(fs, workers):
    import safetensors.torch
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(safetensors.torch.load_file, fs):
            out.update(part)
    return sum(t.numel() * t.element_size() for t in out.values())


def mmap_slice(fs, workers, world, rank):
    """Read only the rows/cols this TP rank owns, via safetensors slicing."""
    from safetensors import safe_open
    got = 0

    def one(f):
        nonlocal got
        n = 0
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                shape = sl.get_shape()
                if len(shape) == 2 and shape[0] % world == 0:
                    per = shape[0] // world
                    t = sl[rank * per:(rank + 1) * per, :]
                else:
                    t = h.get_tensor(k)
                n += t.numel() * t.element_size()
        return n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(one, fs):
            got += n
    return got


def rank_worker(args):
    d, workers, idx = args
    fs = files(d)
    t = time.time()
    prefetch(fs, workers)
    load_only(fs, workers)
    return time.time() - t


def module_loop(fs, world, rank, threads):
    """Emulate the per-module inner loop DSv3/K3 runs today.

    For each checkpoint tensor: fault in the mmap pages, take this rank's
    TP row shard, make it contiguous (what load_weight_shard does), and
    copy into a destination host buffer (stand-in for the GPU parameter).
    threads=1 reproduces DeepseekV3WeightLoader's serial `for name, module`
    walk; threads>1 is the thread-pooled alternative.
    """
    import torch
    from safetensors import safe_open

    def one(f):
        n = 0
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                rows = t.shape[0]
                per = rows // world
                shard = t[rank * per:(rank + 1) * per].contiguous()
                dst = torch.empty_like(shard)
                dst.copy_(shard)
                n += shard.numel() * shard.element_size()
        return n

    with ThreadPoolExecutor(max_workers=threads) as ex:
        return sum(ex.map(one, fs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--mode", default="prefetch_load",
                   choices=("prefetch_load", "load_only", "mmap_slice", "ranks", "module_loop"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--nranks", type=int, default=8)
    a = p.parse_args()

    fs = files(a.dir)
    gb = total_bytes(fs) / 1024**3
    print(f"{len(fs)} files, {gb:.2f} GiB in {a.dir}")

    t0 = time.time()
    if a.mode == "prefetch_load":
        prefetch(fs, a.workers); tp = time.time()
        load_only(fs, a.workers); tl = time.time()
        print(f"  prefetch     {tp-t0:7.1f}s  {gb*1024/(tp-t0):7.0f} MB/s")
        print(f"  load_file    {tl-tp:7.1f}s  {gb*1024/(tl-tp):7.0f} MB/s")
        print(f"  TOTAL        {tl-t0:7.1f}s  {gb*1024/(tl-t0):7.0f} MB/s effective")
    elif a.mode == "load_only":
        load_only(fs, a.workers); d = time.time() - t0
        print(f"  load_file    {d:7.1f}s  {gb*1024/d:7.0f} MB/s")
    elif a.mode == "mmap_slice":
        n = mmap_slice(fs, a.workers, a.world, a.rank); d = time.time() - t0
        print(f"  mmap_slice   {d:7.1f}s  {n/1024**3:.2f} GiB read "
              f"({n/1024**2/d:7.0f} MB/s of shard, world={a.world})")
    elif a.mode == "module_loop":
        n = module_loop(fs, a.world, a.rank, a.workers); d = time.time() - t0
        print(f"  module_loop  {d:7.1f}s  threads={a.workers}  "
              f"{n/1024**3:.2f} GiB shard  {n/1024**2/d:7.0f} MB/s")
    else:
        with mp.Pool(a.nranks) as pool:
            ds = pool.map(rank_worker,
                          [(a.dir, a.workers, i) for i in range(a.nranks)])
        d = time.time() - t0
        print(f"  {a.nranks} ranks each loading the whole checkpoint: "
              f"wall {d:7.1f}s, per-rank {min(ds):.1f}-{max(ds):.1f}s, "
              f"aggregate {gb*a.nranks*1024/d:7.0f} MB/s, "
              f"useful {gb*1024/d:7.0f} MB/s")

if __name__ == "__main__":
    main()
