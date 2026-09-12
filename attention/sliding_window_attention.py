"""Causal sliding-window attention; window W includes the current token.

Dense-mask reference shows the definition. Online reference skips outside
tiles, showing how FlashAttention can execute the same local pattern.
"""

from .dense_attention import matmul_attention, online_attention


def sliding_window_attention(q, k, v, window=128):
    return matmul_attention(q, k, v, causal=True, window=window)


def sliding_window_online(q, k, v, window=128, tile=128):
    return online_attention(q, k, v, causal=True, window=window, tile=tile)
