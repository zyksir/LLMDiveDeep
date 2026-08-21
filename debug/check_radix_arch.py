"""Does a 10.0a-pinned radix cubin load and run on this 10.3 device?"""
import torch
from common import arch
from kimi_k3_layer.kernels import routing_radix as rr
from kimi_k3_layer.config import NUM_EXPERTS, TOP_K

print("device cap      :", torch.cuda.get_device_capability())
print("build target    : sm_%s (B10_FORCE_SM)" % arch.suffix())
print("SGL_CUDA_ARCH   :", rr._cuda_arch_macro())
print("arch list       :", arch.torch_arch_list())
logits = torch.randn(8, NUM_EXPERTS, device="cuda", dtype=torch.float32)
bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)
try:
    ids, w = rr.route_radix(logits, bias, fmt="fp32")
    torch.cuda.synchronize()
    print("RESULT          : ran OK, ids", tuple(ids.shape), "dtype", ids.dtype)
except Exception as e:
    print("RESULT          : FAILED ->", type(e).__name__, str(e)[:300])
