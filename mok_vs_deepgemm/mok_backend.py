import torch.distributed as dist

from mok import functional

from common import Inputs, Shape


class MoKBackend:
    def __init__(
        self,
        inputs: Inputs,
        shape: Shape,
        *,
        fwd_num_comm_sms: int = 24,
        minibatch_size: int = 4096,
    ) -> None:
        self.name = (
            "MoK BF16 full forward "
            f"(comm_sms={fwd_num_comm_sms}, minibatch={minibatch_size})"
        )
        self.inputs = inputs
        self.num_local_experts = shape.num_experts // dist.get_world_size()
        self.config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=28,
            minibatch_size=minibatch_size,
            macrobatch_size=131072,
        )
        self.workspace = functional.get_workspace(
            self.config,
            dist.group.WORLD,
            device=inputs.x.device,
            num_local_tokens=shape.tokens_per_rank,
            hidden_size=shape.hidden,
            topk=shape.topk,
        )

    def run(self):
        schedule = functional.build_schedule(
            self.workspace,
            self.config,
            self.inputs.topk_experts,
            num_local_experts=self.num_local_experts,
        )
        return functional.forward(
            self.config,
            self.workspace,
            schedule,
            self.inputs.x,
            self.inputs.router_weights,
            self.inputs.w_shared_gate,
            self.inputs.w_shared_up,
            self.inputs.w_shared_down,
            self.inputs.w_routed_gate,
            self.inputs.w_routed_up,
            self.inputs.w_routed_down,
        )

    def close(self) -> None:
        functional.clear_workspace_cache()
