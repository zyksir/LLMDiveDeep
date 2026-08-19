import torch
import torch.distributed as dist

import deep_gemm

from common import Inputs, Shape


class DeepGEMMBackend:
    name = "DeepGEMM BF16 Mega MoE"

    def __init__(self, inputs: Inputs, shape: Shape) -> None:
        self.inputs = inputs
        self.shape = shape
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            shape.num_experts,
            shape.tokens_per_rank,
            shape.topk,
            shape.hidden,
            shape.intermediate,
            num_shared_experts=shape.num_shared_experts,
            mma_type="bf16xbf16",
        )

        l1_weights = torch.cat(
            (inputs.w_routed_gate, inputs.w_routed_up),
            dim=1,
        ).contiguous()
        self.l1_weights, self.l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                l1_weights,
                inputs.w_routed_down,
            )
        )
        shared_l1_weights = torch.cat(
            (inputs.w_shared_gate, inputs.w_shared_up),
            dim=0,
        ).contiguous()
        self.shared_l1_weights, self.shared_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                shared_l1_weights,
                inputs.w_shared_down,
            )
        )

    def run(self):
        tokens = self.shape.tokens_per_rank
        self.buffer.x[:tokens].copy_(self.inputs.x)
        self.buffer.topk_idx[:tokens].copy_(self.inputs.topk_experts)
        self.buffer.topk_weights[:tokens].copy_(self.inputs.router_weights)
        output = torch.empty(
            tokens,
            self.shape.hidden,
            dtype=torch.bfloat16,
            device=self.inputs.x.device,
        )
        deep_gemm.bf16_mega_moe(
            output,
            self.l1_weights,
            self.l2_weights,
            self.buffer,
            shared_l1_weights=self.shared_l1_weights,
            shared_l2_weights=self.shared_l2_weights,
            fast_math=True,
        )
        return output, None

    def close(self) -> None:
        self.buffer.destroy()
