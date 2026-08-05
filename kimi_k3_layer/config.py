"""Official Kimi-K3 KDA decode dimensions, without collectives."""

from __future__ import annotations

from dataclasses import dataclass

HIDDEN = 7168
HEADS = 96
HEAD_DIM = 128
CONV = 4
GATE_LOWER_BOUND = -5.0
RMS_EPS = 1e-5
NUM_LAYERS = 93
ATTN_RES_BLOCK_SIZE = 12
NUM_ATTN_RES_BLOCKS = 8

# MoE (per the Kimi-K3 checkpoint config; latent MoE like TRT's NemotronHMOE).
NUM_EXPERTS = 896
TOP_K = 16
MOE_INTER = 384  # routed expert intermediate size
MOE_LATENT = 3584  # routed_expert_hidden_size (fc1/fc2 latent)
NUM_SHARED_EXPERTS = 16
SHARED_INTER = MOE_INTER * NUM_SHARED_EXPERTS  # 6144
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING = 1.0


@dataclass(frozen=True)
class K3Shard:
    name: str
    tp_size: int
    hidden: int = HIDDEN
    heads_global: int = HEADS
    heads_local: int = HEADS
    head_dim: int = HEAD_DIM
    conv_size: int = CONV
    # MoE sharding. Routed experts are EP-sharded (896/tp experts local,
    # full 384 intermediate): TRT's default moe_tp would leave intermediate
    # 384/8 = 48, which the trtllm-gen BF16 kernel cannot tile (block_k=64)
    # and which no real K3 deployment uses. The shared expert is TP-sharded
    # on its intermediate like TRT's GatedMLP; gate/fc1/fc2 are replicated.
    experts_local: int = NUM_EXPERTS
    shared_inter_local: int = SHARED_INTER

    @property
    def qkv_dim(self) -> int:
        return 3 * self.heads_local * self.head_dim

    @property
    def proj_dim(self) -> int:
        return self.heads_local * self.head_dim


def k3_shard(name: str) -> K3Shard:
    name = name.lower()
    if name == "tp8":
        return K3Shard(
            name="tp8",
            tp_size=8,
            heads_local=HEADS // 8,
            experts_local=NUM_EXPERTS // 8,
            shared_inter_local=SHARED_INTER // 8,
        )
    if name in ("tp1", "full"):
        return K3Shard(
            name="tp1",
            tp_size=1,
            heads_local=HEADS,
        )
    raise ValueError(f"unknown shard {name!r}; expected tp1 or tp8")


def is_kda_layer(layer_idx: int) -> bool:
    """The official config uses three KDA layers followed by one MLA layer."""
    one_based = layer_idx + 1
    return 1 <= one_based <= 91 and one_based % 4 != 0
