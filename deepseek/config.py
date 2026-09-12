"""Released backbone schedule; IDs are zero-based (DSpark is excluded)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    stage: str
    ratio: int
    mode: str
    kv_owner: int | None
    index_owner: int | None


KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
CANDIDATE_SOURCE = 20
WINDOW = 128
TOPK = 512
CANDIDATE_BLOCKS = 2048
BLOCK_SIZE = 8
HEAD_DIM = 512
INDEX_DIM = 128
HEADS = 64
INDEX_HEADS = 32


def layer_spec(layer_id: int) -> LayerSpec:
    if not 0 <= layer_id < 40:
        raise ValueError("backbone layer_id must be in [0, 40)")
    stage = "encoder" if layer_id < 20 else "decoder"
    if layer_id < 2:
        return LayerSpec(layer_id, stage, 0, "SWA", None, None)
    mode = (
        "Full"
        if layer_id in KV_SOURCES
        else "Reindex"
        if layer_id in INDEX_SOURCES
        else "Reuse"
    )
    return LayerSpec(
        layer_id,
        stage,
        2 if layer_id < 20 else 1,
        mode,
        max(i for i in KV_SOURCES if i <= layer_id),
        max(i for i in INDEX_SOURCES if i <= layer_id),
    )


def global_cache_bytes(tokens: int) -> int:
    """Ideal payload, excluding SWA, allocator padding and runtime metadata.

    Main latent: 512/2 + 512/16 = 288 bytes.
    Index key: 128/2 + 128/32 = 68 bytes.
    Three encoder owners at ratio 2; one decoder owner at ratio 1.
    """
    if tokens < 0:
        raise ValueError("tokens must be nonnegative")
    return (3 * (tokens // 2) + tokens) * (288 + 68)


def index_positions_per_decode(tokens: int) -> int:
    """Positions scored across eight index-producing backbone layers.

    This is a work-count model, not FLOPs or measured latency.
    """
    if tokens < 0:
        raise ValueError("tokens must be nonnegative")
    return 3 * (tokens // 2) + tokens + 4 * min(tokens, CANDIDATE_BLOCKS * BLOCK_SIZE)
