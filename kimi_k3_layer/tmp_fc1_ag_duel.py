"""Probe: fc1-shard AG kernel duel at the NEW target sizes - answers
"when does quantize-then-AG beat AG-then-quantize?" at the kernel
level, plus bit-exactness of the fp8-wire path.

Per B in 16..80, [B, 448] bf16 shard -> ([B, 3584] e4m3, scales):

  recv   receiver-side fused AG+quant (`all_gather_mxfp8`): bf16 on
         the wire (B*3584*2 bytes gathered), quantize on write-out
  qpush  sender-side quant (`all_gather_mxfp8_push`): e4m3+scales on
         the wire (~B*3584*1.03 bytes) - HALF the payload; pays once
         payload time matters (B>=~32), loses in the latency-bound
         regime if the double completion (payload+scales) costs more

Each family reports its best (grid, block) - grids 8/32/64. The two
paths are asserted BIT-IDENTICAL first (payload and scales), so a
qpush correctness bug fails fast here, not in the e2e retune.

  mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/tmp_fc1_ag_duel.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3_layer.comm import OneShotComm  # noqa: E402
from kimi_k3_layer.tmp_rs_norm_probe import bench_graph  # noqa: E402

LATENT = 3584
MAX_B = 80


def main():
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", rank))

    cols = LATENT // world  # 448
    max_bytes = MAX_B * LATENT * 2
    grids = (8, 32, 64)
    bf16 = {g: OneShotComm(rank, world, max_bytes=max_bytes, grid=g)
            for g in grids}
    fp8 = {g: OneShotComm(rank, world, max_bytes=max_bytes, grid=g,
                          wire="fp8")
           for g in grids}

    if rank == 0:
        print(f"{'B':>4} {'recv:AG+q':>10} {'(g,blk)':>9} "
              f"{'qpush:q+AG':>11} {'(g,blk)':>9} {'delta':>7}"
              f"  (us, graph replay, max over ranks; delta>0 = qpush wins)")
    for b in (8, 16, 32, 64, 80):
        torch.manual_seed(100 + rank)
        x = torch.randn(b, cols, device="cuda", dtype=torch.bfloat16)

        # bit-exactness first: qpush must equal the receiver-side path
        p_recv, s_recv = bf16[8].all_gather_mxfp8(x)
        p_push, s_push = fp8[8].all_gather_mxfp8_push(x)
        assert torch.equal(p_recv.view(torch.uint8),
                           p_push.view(torch.uint8)), f"payload B={b}"
        assert torch.equal(s_recv.view(torch.uint8),
                           s_push.view(torch.uint8)), f"scales B={b}"

        def best(comms, fn_name):
            best_t, best_cfg = float("inf"), None
            for g, comm in comms.items():
                for blk in (256, 512):
                    fn = getattr(comm, fn_name)
                    t = bench_graph(lambda: fn(x, block=blk), world)
                    if t < best_t:
                        best_t, best_cfg = t, (g, blk)
            return best_t, best_cfg

        t_r, cfg_r = best(bf16, "all_gather_mxfp8")
        t_q, cfg_q = best(fp8, "all_gather_mxfp8_push")
        if rank == 0:
            print(f"{b:>4} {t_r:10.2f} {str(cfg_r):>9} "
                  f"{t_q:11.2f} {str(cfg_q):>9} {t_r - t_q:7.2f}",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
