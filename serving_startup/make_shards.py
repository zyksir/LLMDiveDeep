"""Create a synthetic multi-shard safetensors checkpoint with RANDOM content.

Zero-filled files can be served from zero-block dedup by thin-provisioned
storage, which makes read-throughput numbers meaningless.
"""
import sys, os, torch
from safetensors.torch import save_file

out, n_shards, gb_per_shard = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
os.makedirs(out, exist_ok=True)
rows = int(gb_per_shard * 1024**3 / (7168 * 2) / 8)
for s in range(n_shards):
    tensors = {}
    for i in range(8):
        t = torch.randint(0, 255, (rows, 7168), dtype=torch.uint8)
        tensors[f"model.layers.{s}.w{i}.weight"] = t.view(torch.float8_e4m3fn)
    save_file(tensors, f"{out}/model-{s:05d}.safetensors")
    print(f"shard {s} done", flush=True)
