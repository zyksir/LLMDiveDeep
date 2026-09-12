"""4-layer mini Kimi-K3 transformer for cross-layer overlap research.

Purpose: the single-layer blocks in ``b10_kimi_k3_kda_layer.py`` /
``b10_kimi_k3_mla_layer.py`` / ``b10_kimi_k3_moe_layer.py`` align each
sublayer's kernel sequence in isolation, but the scheduling questions that
remain — side-stream reuse ACROSS layers, AttnRes prefix hand-off between
sublayers, whether the MoE shared-expert / gate-GEMM / tiny-GEMV forks of
adjacent layers serialize on each other, CUDA-graph capture of a multi-layer
span — only exist when consecutive layers run back to back. This mini stacks
four production layers with production residual wiring so those interactions
can be measured (and later A/B'd) without standing up a 93-layer server.

Layer pattern: [KDA, KDA, KDA, MLA], the real interleave — checkpoint
config.json (``/node-storage/var/kimi-k3-config``) ``linear_attn_config``
has ``full_attn_layers = [4, 8, ..., 88, 92, 93]`` (1-based), i.e. every
window of four is three KDA layers then one MLA layer
(``kimi_k3.config.is_kda_layer`` encodes the same rule). The mini
instantiates layer indices 4..7 (one-based 5..8) — a REAL window, chosen
over the first window (1..4) so no layer is an AttnRes block-write layer
(none of 5..8 hits the every-12 write) and every layer sees the same
one-valid-block AttnRes bank; the earliest window would also need the
layer-0 block-write layer the MTP blocks reject. Each layer = one attention
block + one MoE block with the production AttnRes residual-stream wiring
between them (attention delta folded into the prefix by the mlp-side
AttnRes; MoE delta folded before the next layer's attention AttnRes).

Both minis create every shared stream set ONCE and hand it to all layers:
one MoE ``aux_stream_dict`` (the ``bench_b10_kimi_k3_moe_layer.py``
session convention), one KDA tiny-GEMV aux stream rebound onto all three
KDA blocks, and the MLA aux/gate streams passed by constructor — so
cross-layer side-stream contention is real, not an artifact of per-block
stream creation.

MTP1 decode geometry throughout (M = 2 tokens/request, 35-token context;
see the block files). Per-layer ``record_function`` spans
``TrtKimiK3Mini/L0-KDA`` .. ``TrtKimiK3Mini/L3-MLA`` (and ``Sgl...``)
match ``traces/extract_layer_kernels.py``'s ``(Trt|Sgl).*Kimi`` span regex.

Named deviations (beyond each composed block's own documented ones):

* MoE blocks are constructed at ``layer_idx=0`` — the MoE math has no layer
  dependence and the bench quant_config_dict keys the MXFP4 expert config
  at layer 0 (the ``bench_b10_kimi_k3_moe_layer.py`` convention).
* TRT: production defers the MoE shared+routed add into the next AttnRes's
  ``second_delta``; the composed ``TrtKimiK3MoEBlock`` returns one summed
  tensor (``defer_shared_output_add=False``, identical math — its
  documented deviation), so the mini feeds it as a single ``delta``.
* SGL: the MoE-delta prefix fold is ONE explicit ``torch.add`` into a
  mini-owned buffer; production folds it into the MoE tail's push-AR
  (kernel-free only at world = 1, matching the attention blocks' fallback
  tail convention).
* The two stacks' return conventions differ like their blocks': the TRT
  mini returns the last layer's pending MoE delta (its blocks return
  deltas); the SGL mini returns the fully folded prefix stream.
"""

from __future__ import annotations

import torch
from torch import nn

from kimi_k3.b10_kimi_k3_kda_layer import (
    SglKimiK3KdaBlock,
    TrtKimiK3KdaBlock,
    _graft,
)
from kimi_k3.b10_kimi_k3_mla_layer import (
    DEFAULT_CONTEXT_LEN,
    SglKimiK3MlaBlock,
    TrtKimiK3MlaBlock,
)
from kimi_k3.b10_kimi_k3_moe_layer import k3_model_config
from kimi_k3.config import (
    HIDDEN,
    NUM_ATTN_RES_BLOCKS,
    RMS_EPS,
    K3Shard,
    is_kda_layer,
)

#: One-based 5..8: a real [KDA, KDA, KDA, MLA] window (see module docstring).
LAYER_IDXS = (4, 5, 6, 7)


class _KimiK3Mini(nn.Module):
    """Shared mini machinery: the mini-owned residual stream, the MoE block
    stack, collectives fan-out and the layer loop with per-layer spans.

    Residual-stream ownership follows the AttnRes design: the mini owns ONE
    ``prefix_sum`` / ``delta`` seed / ``block_residual`` bank (randn, the
    block files' convention) and rebinds every attention block's registered
    buffers onto them post-construction, so all four layers read and write
    one continuous stream exactly as consecutive production layers do —
    without editing the block classes.
    """

    SPAN_PREFIX = "KimiK3Mini"

    def __init__(
        self,
        shard: K3Shard,
        batch: int,
        *,
        context_len: int = DEFAULT_CONTEXT_LEN,
        device: str = "cuda",
        collectives=None,
    ) -> None:
        super().__init__()
        for idx in LAYER_IDXS[:3]:
            assert is_kda_layer(idx), idx
        assert not is_kda_layer(LAYER_IDXS[3]), LAYER_IDXS[3]

        self.shard = shard
        self.batch = batch
        self.context_len = context_len
        self._collectives = collectives
        dev = torch.device(device)

        self.register_buffer(
            "prefix_sum",
            torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "delta",
            torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "block_residual",
            torch.randn(
                batch, NUM_ATTN_RES_BLOCKS, HIDDEN,
                device=dev, dtype=torch.bfloat16,
            ),
        )
        # Per-layer post-attention (mlp-side) input norm weights; the
        # attention-side norm weights live inside the attention blocks.
        self.post_norm_weights = nn.ParameterList(
            nn.Parameter(torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16))
            for _ in LAYER_IDXS
        )

        # ONE shared MoE side-stream set for all four layers (the MoE bench's
        # session convention; TRT uses several AuxStreamType slots, SGL the
        # MoeShared one).
        from tensorrt_llm._torch.utils import AuxStreamType

        self.moe_aux_streams = {
            kind: torch.cuda.Stream(device=dev) for kind in AuxStreamType
        }
        rank = collectives.rank if collectives is not None else 0
        self._model_config = k3_model_config(rank, shard.tp_size)
        self.moe_blocks = nn.ModuleList(
            self._build_moe_block(dev) for _ in LAYER_IDXS
        )
        self.attn_blocks = nn.ModuleList(
            self._build_attn_blocks(dev, collectives)
        )
        self.layer_kinds = tuple(
            "KDA" if is_kda_layer(idx) else "MLA" for idx in LAYER_IDXS
        )
        # Rebind every attention block's residual-stream buffers onto the
        # mini-owned stream (registered-buffer assignment replaces the
        # entry; block code reads self.prefix_sum etc. at call time).
        for blk in self.attn_blocks:
            blk.prefix_sum = self.prefix_sum
            blk.delta = self.delta
            blk.block_residual = self.block_residual

    def _build_moe_block(self, dev: torch.device) -> nn.Module:
        raise NotImplementedError

    def _build_attn_blocks(self, dev, collectives) -> list[nn.Module]:
        raise NotImplementedError

    def attach_collectives(self, collectives) -> None:
        self._collectives = collectives
        for blk in (*self.attn_blocks, *self.moe_blocks):
            blk.attach_collectives(collectives)

    def _layer_step(self, i: int, prefix_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        hidden: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is not None or cu_seqlens is not None:
            raise ValueError("MTP1 dispatch does not accept prefill inputs")
        with torch.profiler.record_function(self.SPAN_PREFIX):
            return self.decode()


class TrtKimiK3Mini(_KimiK3Mini):
    """4-layer mini on the TRT-LLM rc19 blocks.

    Per layer: ``TrtKimiK3KdaBlock`` / ``TrtKimiK3MlaBlock`` (whose in-block
    AttnRes folds the incoming delta into the shared prefix IN PLACE —
    ``attn_res.py``'s HAS_DELTA store — and yields the attention input),
    then the mini-owned mlp-side grafted ``AttnRes`` folding the attention
    output and yielding the MoE input, then ``TrtKimiK3MoEBlock``. The MoE
    output is the next layer's ``delta`` (rebound per step, so no copy
    kernels enter the stream). All three KDA blocks share one tiny-GEMV aux
    stream (rebound post-construction; the block ctor creates a private
    one); the MLA block receives the mini's MLA aux/gate streams.
    """

    SPAN_PREFIX = "TrtKimiK3Mini"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dev = self.prefix_sum.device
        # Mlp-side AttnRes: same grafted checkout module the attention
        # blocks run for their attention-side fold (attn_res_fwd_online_v2).
        attn_res_mod = _graft(
            "tensorrt_llm._torch.modules.attn_res",
            "_torch/modules/attn_res.py",
        )
        self.mlp_attn_res = nn.ModuleList(
            attn_res_mod.AttnRes(
                HIDDEN, RMS_EPS, dtype=torch.bfloat16, device=dev
            )
            for _ in LAYER_IDXS
        )
        with torch.no_grad():
            for mod in self.mlp_attn_res:
                nn.init.ones_(mod.norm_weight)
                nn.init.normal_(mod.proj_weight, std=0.02)

    def _build_attn_blocks(self, dev, collectives):
        kda_aux_stream = torch.cuda.Stream(device=dev)
        self.mla_aux_stream = torch.cuda.Stream(device=dev)
        self.mla_gate_stream = torch.cuda.Stream(device=dev)
        blocks: list[nn.Module] = []
        for idx in LAYER_IDXS[:3]:
            blk = TrtKimiK3KdaBlock(
                self.shard,
                self.batch,
                layer_idx=idx,
                device=str(dev),
                collectives=collectives,
            )
            blk.aux_stream = kda_aux_stream  # ONE shared tiny-GEMV stream
            blocks.append(blk)
        blocks.append(
            TrtKimiK3MlaBlock(
                self.shard,
                self.batch,
                layer_idx=LAYER_IDXS[3],
                context_len=self.context_len,
                device=str(dev),
                collectives=collectives,
                aux_stream=self.mla_aux_stream,
                output_gate_aux_stream=self.mla_gate_stream,
            )
        )
        self.kda_aux_stream = kda_aux_stream
        return blocks

    def _build_moe_block(self, dev: torch.device) -> nn.Module:
        from kimi_k3.b10_kimi_k3_moe_layer import TrtKimiK3MoEBlock

        return TrtKimiK3MoEBlock(
            self._model_config,
            layer_idx=0,
            aux_stream_dict=self.moe_aux_streams,
            collectives=self._collectives,
        ).to(dev)

    def decode(self) -> torch.Tensor:
        delta = self.delta
        for i, (blk, kind) in enumerate(
            zip(self.attn_blocks, self.layer_kinds)
        ):
            with torch.profiler.record_function(
                f"{self.SPAN_PREFIX}/L{i}-{kind}"
            ):
                # The block's AttnRes reads self.delta at call time and
                # folds it into the shared prefix in place.
                blk.delta = delta
                attn_out = blk()
                with torch.profiler.record_function("mini.mlp_attn_res"):
                    moe_hidden = self.mlp_attn_res[i](
                        self.prefix_sum,
                        self.block_residual,
                        delta=attn_out,
                        num_blocks=blk.prev_valid_blocks,
                        block_write_idx=-1,
                        output_norm_weight=self.post_norm_weights[i],
                        output_norm_eps=RMS_EPS,
                    )
                delta = self.moe_blocks[i](moe_hidden)
        # The last MoE delta stays pending, as production hands it to the
        # NEXT layer's AttnRes (TRT-block return convention).
        return delta


class SglKimiK3Mini(_KimiK3Mini):
    """4-layer mini on the sglang blocks.

    Per layer: ``SglKimiK3KdaBlock`` / ``SglKimiK3MlaBlock`` (which consume
    the PRE-ADDED prefix and return the next prefix — their fold lives in
    the push-AR tail, or the explicit-add fallback at world = 1), then the
    mini-owned mlp-side ``attn_res_fused_tma`` (vendored TMA kernel with a
    per-layer precomputed ``cw`` product) yielding the MoE input, then
    ``SglKimiK3MoEBlock``. The MoE delta folds into the stream with one
    ``torch.add`` into a per-layer mini buffer (named deviation above).
    Stream sharing mirrors the TRT mini: one KDA aux stream across the
    three KDA blocks, mini-owned MLA aux/gate streams, one MoE stream set.
    """

    SPAN_PREFIX = "SglKimiK3Mini"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dev = self.prefix_sum.device
        # Per-layer mlp-side AttnRes weights + the precomputed bf16 product
        # attn_res_fused_tma consumes (the SglKimiK3KdaBlock convention).
        self.mlp_attn_res_norm = nn.ParameterList(
            nn.Parameter(torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16))
            for _ in LAYER_IDXS
        )
        self.mlp_attn_res_proj = nn.ParameterList(
            nn.Parameter(
                torch.empty(
                    NUM_ATTN_RES_BLOCKS, device=dev, dtype=torch.bfloat16
                )
            )
            for _ in LAYER_IDXS
        )
        for i in range(len(LAYER_IDXS)):
            self.register_buffer(
                f"mlp_attn_res_cw_{i}",
                torch.empty(
                    NUM_ATTN_RES_BLOCKS * HIDDEN,
                    device=dev, dtype=torch.bfloat16,
                ).view(NUM_ATTN_RES_BLOCKS, HIDDEN),
            )
            # MoE-fold landing buffer: the next layer's prefix.
            self.register_buffer(
                f"folded_prefix_{i}",
                torch.zeros(
                    self.batch, HIDDEN, device=dev, dtype=torch.bfloat16
                ),
            )
        with torch.no_grad():
            for i in range(len(LAYER_IDXS)):
                nn.init.normal_(self.mlp_attn_res_proj[i], std=0.02)
                getattr(self, f"mlp_attn_res_cw_{i}").copy_(
                    (
                        self.mlp_attn_res_norm[i].float().unsqueeze(0)
                        * self.mlp_attn_res_proj[i].float().unsqueeze(1)
                    ).to(torch.bfloat16)
                )

    def _build_attn_blocks(self, dev, collectives):
        kda_aux_stream = torch.cuda.Stream(device=dev)
        self.mla_aux_stream = torch.cuda.Stream(device=dev)
        self.mla_gate_stream = torch.cuda.Stream(device=dev)
        blocks: list[nn.Module] = []
        for idx in LAYER_IDXS[:3]:
            blk = SglKimiK3KdaBlock(
                self.shard,
                self.batch,
                layer_idx=idx,
                device=str(dev),
                collectives=collectives,
            )
            blk.aux_stream = kda_aux_stream  # ONE shared tiny-GEMV stream
            blocks.append(blk)
        blocks.append(
            SglKimiK3MlaBlock(
                self.shard,
                self.batch,
                layer_idx=LAYER_IDXS[3],
                context_len=self.context_len,
                device=str(dev),
                collectives=collectives,
                aux_stream=self.mla_aux_stream,
                gate_stream=self.mla_gate_stream,
            )
        )
        self.kda_aux_stream = kda_aux_stream
        return blocks

    def _build_moe_block(self, dev: torch.device) -> nn.Module:
        from kimi_k3.b10_kimi_k3_moe_layer import SglKimiK3MoEBlock

        return SglKimiK3MoEBlock(
            self._model_config,
            layer_idx=0,
            aux_stream_dict=self.moe_aux_streams,
            reduce_output=self.shard.tp_size > 1,
            collectives=self._collectives,
        ).to(dev)

    def attach_sgl_ar(self, state) -> None:
        """Fan a prebuilt fused-AR state out to every sgl block (each also
        resolves lazily on first decode when not injected)."""
        for blk in (*self.attn_blocks, *self.moe_blocks):
            blk.attach_sgl_ar(state)

    def decode(self) -> torch.Tensor:
        from kimi_k3.kernels.sgl_adapters.attn_res import attn_res_fused_tma

        prefix = self.prefix_sum
        for i, (blk, kind) in enumerate(
            zip(self.attn_blocks, self.layer_kinds)
        ):
            with torch.profiler.record_function(
                f"{self.SPAN_PREFIX}/L{i}-{kind}"
            ):
                # sgl blocks consume the pre-added prefix and return the
                # next one (attention delta folded by their AR tail).
                blk.prefix_sum = prefix
                attn_prefix = blk()
                moe_hidden = torch.empty_like(prefix)
                with torch.profiler.record_function("mini.mlp_attn_res_tma"):
                    attn_res_fused_tma(
                        attn_prefix,
                        self.block_residual,
                        getattr(self, f"mlp_attn_res_cw_{i}"),
                        self.post_norm_weights[i].detach(),
                        moe_hidden,
                        blk.prev_valid_blocks,
                        RMS_EPS,
                        write_prefix=False,
                    )
                moe_out = self.moe_blocks[i](moe_hidden)
                with torch.profiler.record_function("mini.moe_prefix_fold"):
                    # Named deviation: production folds this into the MoE
                    # tail's push-AR; explicit only at world = 1.
                    prefix = torch.add(
                        attn_prefix,
                        moe_out,
                        out=getattr(self, f"folded_prefix_{i}"),
                    )
        return prefix


__all__ = [
    "LAYER_IDXS",
    "SglKimiK3Mini",
    "TrtKimiK3Mini",
]
