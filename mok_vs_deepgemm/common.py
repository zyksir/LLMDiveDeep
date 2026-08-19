from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Shape:
    num_experts: int = 256
    hidden: int = 6144
    intermediate: int = 2048
    topk: int = 8
    tokens_per_rank: int = 2048
    num_shared_experts: int = 1


@dataclass
class Inputs:
    x: torch.Tensor
    topk_experts: torch.Tensor
    router_weights: torch.Tensor
    w_shared_gate: torch.Tensor
    w_shared_up: torch.Tensor
    w_shared_down: torch.Tensor
    w_routed_gate: torch.Tensor
    w_routed_up: torch.Tensor
    w_routed_down: torch.Tensor


def make_inputs(rank: int, world_size: int, device: torch.device, shape: Shape) -> Inputs:
    if shape.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")

    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    num_local_experts = shape.num_experts // world_size
    logits = torch.randn(
        shape.tokens_per_rank,
        shape.num_experts,
        generator=generator,
        device=device,
    )
    topk_values, topk_experts = torch.topk(logits, shape.topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)

    hidden_scale = shape.hidden**-0.5
    intermediate_scale = shape.intermediate**-0.5
    x = torch.randn(
        shape.tokens_per_rank,
        shape.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    w_shared_gate = torch.randn(
        shape.intermediate,
        shape.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(hidden_scale)
    w_shared_up = torch.randn(
        shape.intermediate,
        shape.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(hidden_scale)
    w_shared_down = torch.randn(
        shape.hidden,
        shape.intermediate,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(intermediate_scale)
    w_routed_gate = torch.randn(
        num_local_experts,
        shape.intermediate,
        shape.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(hidden_scale)
    w_routed_up = torch.randn(
        num_local_experts,
        shape.intermediate,
        shape.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(hidden_scale)
    w_routed_down = torch.randn(
        num_local_experts,
        shape.hidden,
        shape.intermediate,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(intermediate_scale)

    return Inputs(
        x=x,
        topk_experts=topk_experts,
        router_weights=router_weights,
        w_shared_gate=w_shared_gate,
        w_shared_up=w_shared_up,
        w_shared_down=w_shared_down,
        w_routed_gate=w_routed_gate,
        w_routed_up=w_routed_up,
        w_routed_down=w_routed_down,
    )
