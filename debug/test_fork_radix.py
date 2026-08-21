"""Load the FORK's routing_radix.py by path and run it on this device."""
import importlib.util, sys, torch

path = "/node-storage/trt-llm/tensorrt_llm/_torch/kimi_k3_optim/b10_kernels/routing_radix.py"
spec = importlib.util.spec_from_file_location("fork_routing_radix", path)
m = importlib.util.module_from_spec(spec)
sys.modules["fork_routing_radix"] = m
spec.loader.exec_module(m)

print("device cap    :", torch.cuda.get_device_capability())
print("target        :", m._target())
print("SGL_CUDA_ARCH :", m._cuda_arch())
print("arch list     :", m._arch_list())
print("arch tag      :", m._arch_tag())

logits = torch.randn(64, m.NUM_EXPERTS, device="cuda", dtype=torch.float32)
bias = torch.zeros(m.NUM_EXPERTS, device="cuda", dtype=torch.float32)
ids, w = m.route_radix(logits, bias, fmt="fp32")
torch.cuda.synchronize()

# correctness vs an fp32 torch oracle: expert SETS must match exactly
scores = torch.sigmoid(logits) + bias
want = torch.topk(scores, m.TOP_K, dim=1).indices
ok = torch.equal(torch.sort(ids.long(), 1).values, torch.sort(want, 1).values)
print(f"RESULT        : ran OK, ids{tuple(ids.shape)}, expert sets match={ok}")
