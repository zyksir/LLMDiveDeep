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

# MLA (full-attention layers; checkpoint config.json at
# /node-storage/var/kimi-k3-config). 96 q-heads shared with KDA (HEADS);
# fused [q_a | kv_a | k_rope] down-proj width = 1536 + 512 + 64 = 2112,
# absorbed q head dim = 512 + 64 = 576. K3 MLA is NoPE
# (mla_use_nope=true) with the sigmoid output gate (mla_use_output_gate).
MLA_Q_LORA_RANK = 1536
MLA_KV_LORA_RANK = 512
MLA_QK_NOPE_HEAD_DIM = 128
MLA_QK_ROPE_HEAD_DIM = 64
MLA_V_HEAD_DIM = 128
MLA_FUSED_A_DIM = MLA_Q_LORA_RANK + MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM
MLA_QK_HEAD_DIM = MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM  # 192
MLA_ABSORB_Q_DIM = MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM  # 576 (fmha HQk)

# MoE (per the Kimi-K3 checkpoint config; latent MoE like TRT's NemotronHMOE).
NUM_EXPERTS = 896
TOP_K = 16
# Aligned with the released moonshotai/Kimi-K3 config.json (2026-08-21):
# moe_intermediate_size=3072, num_shared_experts=2. The previous values
# (384 x 16) matched the shared product (6144) but ran the ROUTED experts
# 8x narrower than the checkpoint -- every routed-expert GEMM in earlier
# tables used inter=384. Shared-expert shapes were always correct.
MOE_INTER = 3072  # routed expert intermediate size (per HF config)
MOE_LATENT = 3584  # routed_expert_hidden_size (fc1/fc2 latent)
NUM_SHARED_EXPERTS = 2
SHARED_INTER = MOE_INTER * NUM_SHARED_EXPERTS  # 6144 (unchanged)
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
    # full 3072 intermediate) to match every real K3 deployment; TRT's
    # default moe_tp would shard the intermediate instead (3072/8 = 384),
    # which tiles but is not what production runs. The shared expert is TP-sharded
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
    """``'tpN'`` for any N that divides the sharded dims (1, 2, 4, 8...)."""
    name = name.lower()
    if name == "full":
        name = "tp1"
    if not name.startswith("tp") or not name[2:].isdigit():
        raise ValueError(f"unknown shard {name!r}; expected tpN (e.g. tp2)")
    tp = int(name[2:])
    if tp < 1 or HEADS % tp or NUM_EXPERTS % tp or SHARED_INTER % tp:
        raise ValueError(
            f"tp={tp} does not divide heads/experts/shared dims "
            f"({HEADS}/{NUM_EXPERTS}/{SHARED_INTER})")
    return K3Shard(
        name=f"tp{tp}",
        tp_size=tp,
        heads_local=HEADS // tp,
        experts_local=NUM_EXPERTS // tp,
        shared_inter_local=SHARED_INTER // tp,
    )


def is_kda_layer(layer_idx: int) -> bool:
    """The official config uses three KDA layers followed by one MLA layer.

    Checkpoint config.json (``linear_attn_config``): ``full_attn_layers =
    [4, 8, ..., 88, 92, 93]`` (1-based) and ``kda_layers`` everything else,
    i.e. windows of four are [KDA, KDA, KDA, MLA] with the MLA layer fourth,
    plus MLA for the final two layers (92, 93).
    """
    one_based = layer_idx + 1
    return 1 <= one_based <= 91 and one_based % 4 != 0
