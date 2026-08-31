#!/usr/bin/env python3
"""DP-attention variants for the Kimi-K3 MLA layer: BS-SPLIT vs Ulysses vs Ring(DCP).

Three ways to parallelize the attention sublayer over 8 GPUs for the same
global decode batch B with ISL cached history (MLA-only, per user spec):

- DP-Attention-BS-SPLIT: each rank owns B/8 whole requests, full 96-head
  module, full-length local KV. Zero communication.
- DP-Attention-Ulysses: requests token-sharded B/8 per rank; full-weight
  projections on local tokens; A2A redistributes Q by heads (96 -> 12/rank),
  allgather of the 576-dim latent; 12-head core over ALL B tokens against a
  replicated full KV cache; A2A back; local full-head gate + o_proj.
- DP-Attention-Ring: the fork's Helix DCP (decode-only): KV history sharded
  1/8 per rank block-cyclically; every rank computes all B tokens' partial
  attention (96 heads) over its shard; alltoall+helix_post_process merge to
  12 heads/rank; row-parallel o_proj allreduce restores full hidden.

Prefill chunks (B seqs x S): BS-SPLIT and Ulysses only (Helix is decode-only).

All TRT-LLM modules/kernels are IMPORTED unchanged; this file only composes
them with explicit collectives (torch.distributed NCCL) where the variant
requires them.

Launch (inside trt-k3-bench):
  mpirun --allow-run-as-root -np 8 python3 kimi_k3_layer/dp_attention/bench_dp_attention.py \
      --decode-batch-sizes 8 32 128 --isl 2048 --check
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PACKAGE_DIR))
sys.path.insert(0, str(_PACKAGE_DIR.parents[1]))  # kimi_k3_layer
sys.path.insert(0, str(_PACKAGE_DIR.parents[2]))  # repo root

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size  # noqa: E402
from tensorrt_llm.mapping import CpType, Mapping  # noqa: E402

import dp_common  # noqa: E402
from dp_common import (  # noqa: E402
    HIDDEN_SIZE,
    KV_LORA_RANK,
    NUM_HEADS,
    QK_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    TOKENS_PER_BLOCK,
    V_HEAD_DIM,
    add_requests,
    hidden_batch,
    hidden_for_reqs,
    make_kv_cache_manager,
    make_metadata,
    make_mla,
)
from bench_a2a_megamoe_pipeline import _correctness  # noqa: E402

LOCAL_RESULTS = _PACKAGE_DIR / "local_results"
HEADS_PER_RANK = NUM_HEADS // 8  # 12


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(values),
    }


# --------------------------------------------------------------------------
# Reference (single-GPU full batch; identical logical weights on every rank)
# --------------------------------------------------------------------------
class FullBatchReference:
    def __init__(self, device: torch.device):
        self.device = device
        self.mla = make_mla(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            device=device,
            q_shard=(0, 1),
            o_shard=(0, 1),
        )

    def decode(self, batch: int, isl: int, hist_fn, dec: torch.Tensor):
        cache = make_kv_cache_manager(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            batch=batch,
            local_seq_len=isl,
            device_seq_extra=4,
        )
        add_requests(cache, batch, isl, is_gen=True)
        # Real context fill so KV holds real latents (chunked; setup-only).
        from dp_common import CTX_FILL_CHUNK_REQS

        with torch.inference_mode():
            for start in range(0, 0 if dp_common.SKIP_CTX_FILL else batch, CTX_FILL_CHUNK_REQS):
                n = min(CTX_FILL_CHUNK_REQS, batch - start)
                md_ctx = make_metadata(
                    kv_cache_manager=cache, batch=n, phase="prefill",
                    seq_len=isl, cached=0,
                    request_ids=list(range(start, start + n)),
                )
                pos_ctx = torch.arange(isl, dtype=torch.int, device=self.device).repeat(n)
                self.mla.forward(pos_ctx, hist_fn(start, start + n), md_ctx)
        md_gen = make_metadata(
            kv_cache_manager=cache, batch=batch, phase="decode", seq_len=1, cached=isl
        )
        pos_gen = torch.full((batch,), isl, dtype=torch.int, device=self.device)
        with torch.inference_mode():
            out = self.mla.forward(pos_gen, dec, md_gen).clone()
        torch.cuda.synchronize()
        cache.shutdown()
        return out


# --------------------------------------------------------------------------
# Variant: BS-SPLIT
# --------------------------------------------------------------------------
class BsSplitVariant:
    name = "bs_split"
    supports_prefill = True

    def __init__(self, rank: int, world: int, device: torch.device):
        self.rank, self.world, self.device = rank, world, device
        self.mla = make_mla(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            device=device,
            q_shard=(0, 1),
            o_shard=(0, 1),
        )

    def setup_decode(self, batch: int, isl: int, hist_fn, dec: torch.Tensor):
        bl = batch // self.world
        self.cache = make_kv_cache_manager(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            batch=bl,
            local_seq_len=isl,
            device_seq_extra=4,
        )
        add_requests(self.cache, bl, isl, is_gen=True)
        from dp_common import CTX_FILL_CHUNK_REQS

        base = self.rank * bl
        with torch.inference_mode():
            for start in range(0, 0 if dp_common.SKIP_CTX_FILL else bl, CTX_FILL_CHUNK_REQS):
                n = min(CTX_FILL_CHUNK_REQS, bl - start)
                md_ctx = make_metadata(
                    kv_cache_manager=self.cache, batch=n, phase="prefill",
                    seq_len=isl, cached=0,
                    request_ids=list(range(start, start + n)),
                )
                pos_ctx = torch.arange(isl, dtype=torch.int, device=self.device).repeat(n)
                self.mla.forward(pos_ctx, hist_fn(base + start, base + start + n), md_ctx)
        torch.cuda.synchronize()
        self.md = make_metadata(
            kv_cache_manager=self.cache, batch=bl, phase="decode", seq_len=1, cached=isl
        )
        self.pos = torch.full((bl,), isl, dtype=torch.int, device=self.device)
        self.dec_local = dec[self.rank * bl : (self.rank + 1) * bl].contiguous()

    def run_decode(self) -> torch.Tensor:
        return self.mla.forward(self.pos, self.dec_local, self.md)

    def setup_prefill(self, batch: int, seq: int, hist: torch.Tensor):
        bl = batch // self.world
        self.cache = make_kv_cache_manager(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            batch=bl,
            local_seq_len=seq,
        )
        add_requests(self.cache, bl, seq, is_gen=False)
        self.md = make_metadata(
            kv_cache_manager=self.cache, batch=bl, phase="prefill", seq_len=seq, cached=0
        )
        self.pos = torch.arange(seq, dtype=torch.int, device=self.device).repeat(bl)
        self.x = (
            hist[self.rank * bl : (self.rank + 1) * bl]
            .reshape(bl * seq, HIDDEN_SIZE)
            .contiguous()
        )

    def run_prefill(self) -> torch.Tensor:
        return self.mla.forward(self.pos, self.x, self.md)

    def teardown(self):
        self.cache.shutdown()


# --------------------------------------------------------------------------
# Variant: Ulysses (A2A on Q heads, allgather latent, 12-head core, A2A back)
# --------------------------------------------------------------------------
class UlyssesVariant:
    name = "ulysses"
    supports_prefill = True

    def __init__(self, rank: int, world: int, device: torch.device):
        self.rank, self.world, self.device = rank, world, device
        # Full-weight projections / gate / o_proj (identical to bs_split module).
        self.proj = make_mla(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            device=device,
            q_shard=(0, 1),
            o_shard=(0, 1),
        )
        # 12-head core (heads-sharded module; only its core stages are used).
        self.core = make_mla(
            mapping=Mapping(world_size=world, tp_size=world, rank=rank),
            device=device,
            reduce_output=False,
            q_shard=(rank, world),
            o_shard=(rank, world),
        )

    # ---- shared projection + comm helpers ----
    def _project(self, x: torch.Tensor):
        p = self.proj
        qkv_a = p.kv_a_proj_with_mqa(x)
        q_lora, ckv, k_pe = qkv_a.split(
            [Q_LORA_RANK, KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1
        )
        q = p.q_b_proj(p.q_a_layernorm(q_lora))  # [tl, 96*192]
        ckv = p.kv_a_layernorm(ckv)
        latent = torch.cat([ckv, k_pe], dim=-1)  # [tl, 576]
        return q, ckv.contiguous(), k_pe.contiguous(), latent.contiguous()

    def _a2a_heads(self, q: torch.Tensor, tl: int) -> torch.Tensor:
        """[tl, 96*192] -> [world*tl, 12*192] (token-major rank order)."""
        w, h = self.world, HEADS_PER_RANK
        q_in = (
            q.view(tl, w, h * QK_HEAD_DIM).permute(1, 0, 2).contiguous()
        )  # [w, tl, 12*192]
        q_out = torch.empty_like(q_in)
        dist.all_to_all_single(q_out, q_in)
        return q_out.reshape(w * tl, h * QK_HEAD_DIM)

    def _a2a_heads_back(self, o: torch.Tensor, tl: int) -> torch.Tensor:
        """[world*tl, 12*128] -> [tl, 96*128]."""
        w, h = self.world, HEADS_PER_RANK
        o_in = o.view(w, tl, h * V_HEAD_DIM).contiguous()
        o_out = torch.empty_like(o_in)
        dist.all_to_all_single(o_out, o_in)
        return o_out.permute(1, 0, 2).reshape(tl, NUM_HEADS * V_HEAD_DIM)

    def _allgather(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (self.world * t.shape[0], t.shape[1]), dtype=t.dtype, device=t.device
        )
        dist.all_gather_into_tensor(out, t.contiguous())
        return out

    def _finish(self, attn_full_heads: torch.Tensor, x_local: torch.Tensor):
        p = self.proj
        gate = p.g_proj(x_local).sigmoid()
        return p.o_proj(attn_full_heads * gate)

    # ---- decode ----
    def setup_decode(self, batch: int, isl: int, hist_fn, dec: torch.Tensor):
        # Replicated full-batch KV cache for the 12-head core.
        self.cache = make_kv_cache_manager(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            batch=batch,
            local_seq_len=isl,
            device_seq_extra=4,
        )
        add_requests(self.cache, batch, isl, is_gen=True)
        # Context fill via the core's context stage (12 heads; output scratch;
        # chunked, setup-only).
        from dp_common import CTX_FILL_CHUNK_REQS

        with torch.inference_mode():
            for start in range(0, 0 if dp_common.SKIP_CTX_FILL else batch, CTX_FILL_CHUNK_REQS):
                n = min(CTX_FILL_CHUNK_REQS, batch - start)
                md_ctx = make_metadata(
                    kv_cache_manager=self.cache, batch=n, phase="prefill",
                    seq_len=isl, cached=0,
                    request_ids=list(range(start, start + n)),
                )
                pos_ctx = torch.arange(isl, dtype=torch.int, device=self.device).repeat(n)
                x = hist_fn(start, start + n)
                q, ckv, k_pe, latent = self._proj_full(x)
                scratch = x.new_empty(n * isl, HEADS_PER_RANK * V_HEAD_DIM)
                self.core.forward_context_default(
                    q, ckv, k_pe, pos_ctx, md_ctx, scratch, latent_cache=latent
                )
        torch.cuda.synchronize()

        bl = batch // self.world
        self.md = make_metadata(
            kv_cache_manager=self.cache, batch=batch, phase="decode", seq_len=1, cached=isl
        )
        self.pos_full = torch.full((batch,), isl, dtype=torch.int, device=self.device)
        self.dec_local = dec[self.rank * bl : (self.rank + 1) * bl].contiguous()
        self.bl = bl
        self.out12 = torch.empty(
            (batch, HEADS_PER_RANK * V_HEAD_DIM), dtype=torch.bfloat16, device=self.device
        )

    def _proj_full(self, x_full: torch.Tensor):
        """Projection with the CORE module's sharded q_b (context fill only)."""
        c = self.core
        qkv_a = c.kv_a_proj_with_mqa(x_full)
        q_lora, ckv, k_pe = qkv_a.split(
            [Q_LORA_RANK, KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1
        )
        q = c.q_b_proj(c.q_a_layernorm(q_lora))  # [T, 12*192]
        ckv = c.kv_a_layernorm(ckv).contiguous()
        latent = torch.cat([ckv, k_pe], dim=-1).contiguous()
        return q, ckv, k_pe.contiguous(), latent

    def run_decode(self) -> torch.Tensor:
        q, ckv, k_pe, latent = self._project(self.dec_local)
        q12 = self._a2a_heads(q, self.bl)  # [B, 12*192]
        ckv_g = self._allgather(ckv)
        kpe_g = self._allgather(k_pe)
        latent_g = self._allgather(latent)
        self.core.forward_absorption_generation(
            q12, ckv_g, kpe_g, self.md, self.out12,
            position_ids=self.pos_full, latent_cache=latent_g,
        )
        attn = self._a2a_heads_back(self.out12, self.bl)  # [bl, 96*128]
        return self._finish(attn, self.dec_local)

    # ---- prefill ----
    def setup_prefill(self, batch: int, seq: int, hist: torch.Tensor):
        w = self.world
        self.cache = make_kv_cache_manager(
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            batch=batch,
            local_seq_len=seq,
        )
        add_requests(self.cache, batch, seq, is_gen=False)
        self.md = make_metadata(
            kv_cache_manager=self.cache, batch=batch, phase="prefill", seq_len=seq, cached=0
        )
        self.pos_full = torch.arange(seq, dtype=torch.int, device=self.device).repeat(batch)
        sl = seq // w
        # Local slice: tokens [b, rank*sl:(rank+1)*sl] for every sequence b.
        self.x_local = (
            hist[:, self.rank * sl : (self.rank + 1) * sl, :]
            .reshape(batch * sl, HIDDEN_SIZE)
            .contiguous()
        )
        self.batch, self.seq, self.sl = batch, seq, sl
        self.out12 = torch.empty(
            (batch * seq, HEADS_PER_RANK * V_HEAD_DIM),
            dtype=torch.bfloat16, device=self.device,
        )

    def _gather_seq_order(self, t_a2a: torch.Tensor, width: int) -> torch.Tensor:
        """[w, B*sl, width] rank-major -> [B*S, width] sequence order."""
        w, b, sl = self.world, self.batch, self.sl
        return (
            t_a2a.view(w, b, sl, width).permute(1, 0, 2, 3).reshape(b * w * sl, width)
        ).contiguous()

    def _scatter_seq_order(self, t_full: torch.Tensor, width: int) -> torch.Tensor:
        """[B*S, width] sequence order -> local [B*sl, width] for this rank."""
        w, b, sl = self.world, self.batch, self.sl
        return (
            t_full.view(b, w, sl, width)[:, self.rank].reshape(b * sl, width).contiguous()
        )

    def run_prefill(self) -> torch.Tensor:
        tl = self.batch * self.sl
        q, ckv, k_pe, latent = self._project(self.x_local)
        w, h = self.world, HEADS_PER_RANK
        q_in = q.view(tl, w, h * QK_HEAD_DIM).permute(1, 0, 2).contiguous()
        q_out = torch.empty_like(q_in)
        dist.all_to_all_single(q_out, q_in)
        q12 = self._gather_seq_order(q_out, h * QK_HEAD_DIM)
        ckv_g = self._gather_seq_order(
            self._allgather(ckv).view(w, tl, KV_LORA_RANK), KV_LORA_RANK
        )
        kpe_g = self._gather_seq_order(
            self._allgather(k_pe).view(w, tl, QK_ROPE_HEAD_DIM), QK_ROPE_HEAD_DIM
        )
        lat_g = self._gather_seq_order(
            self._allgather(latent).view(w, tl, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        )
        self.core.forward_context_default(
            q12, ckv_g, kpe_g, self.pos_full, self.md, self.out12, latent_cache=lat_g
        )
        o_local_slices = self._scatter_seq_order(self.out12, h * V_HEAD_DIM)
        o_in = o_local_slices  # every rank now holds ITS token slice? No: out12 is full batch.
        # A2A back: send full-batch 12-head outputs, receive local-token 96-head.
        o_send = self.out12.view(w, tl, h * V_HEAD_DIM).contiguous()
        # Reorder so chunk g contains the tokens owned by rank g, in their local order.
        o_send = (
            self.out12.view(self.batch, w, self.sl, h * V_HEAD_DIM)
            .permute(1, 0, 2, 3)
            .reshape(w, tl, h * V_HEAD_DIM)
            .contiguous()
        )
        o_recv = torch.empty_like(o_send)
        dist.all_to_all_single(o_recv, o_send)
        attn = o_recv.permute(1, 0, 2).reshape(tl, NUM_HEADS * V_HEAD_DIM)
        return self._finish(attn, self.x_local)

    def teardown(self):
        self.cache.shutdown()



# --------------------------------------------------------------------------
# Variant: plain TP (production MLA baseline: 12 heads/rank, full KV per
# rank, o_proj allreduce — the thing DP-attention variants try to beat)
# --------------------------------------------------------------------------
class TpVariant:
    name = "tp"
    supports_prefill = False

    def __init__(self, rank: int, world: int, device: torch.device):
        self.rank, self.world, self.device = rank, world, device
        self.mla = make_mla(
            mapping=Mapping(world_size=world, tp_size=world, rank=rank),
            device=device,
            reduce_output=(world > 1),
            q_shard=(rank, world),
            o_shard=(rank, world),
        )

    def setup_decode(self, batch: int, isl: int, hist_fn, dec: torch.Tensor):
        # Replicated tokens: every rank runs the FULL batch over the full KV.
        self.cache = make_kv_cache_manager(
            mapping=Mapping(world_size=self.world, tp_size=self.world, rank=self.rank),
            batch=batch,
            local_seq_len=isl,
            device_seq_extra=4,
        )
        add_requests(self.cache, batch, isl, is_gen=True)
        from dp_common import CTX_FILL_CHUNK_REQS

        with torch.inference_mode():
            for start in range(0, 0 if dp_common.SKIP_CTX_FILL else batch, CTX_FILL_CHUNK_REQS):
                n = min(CTX_FILL_CHUNK_REQS, batch - start)
                md_ctx = make_metadata(
                    kv_cache_manager=self.cache, batch=n, phase="prefill",
                    seq_len=isl, cached=0,
                    request_ids=list(range(start, start + n)),
                )
                pos_ctx = torch.arange(isl, dtype=torch.int, device=self.device).repeat(n)
                self.mla.forward(pos_ctx, hist_fn(start, start + n), md_ctx)
        torch.cuda.synchronize()
        self.md = make_metadata(
            kv_cache_manager=self.cache, batch=batch, phase="decode", seq_len=1, cached=isl
        )
        self.pos = torch.full((batch,), isl, dtype=torch.int, device=self.device)
        self.dec_local = dec.contiguous()

    def run_decode(self) -> torch.Tensor:
        return self.mla.forward(self.pos, self.dec_local, self.md)

    def teardown(self):
        self.cache.shutdown()


# --------------------------------------------------------------------------
# Variant: Ring via Helix DCP (decode-only)
# --------------------------------------------------------------------------
class HelixVariant:
    name = "helix"
    supports_prefill = False

    def __init__(self, rank: int, world: int, device: torch.device):
        self.rank, self.world, self.device = rank, world, device
        self.mapping_cp = Mapping(
            world_size=world, rank=rank, tp_size=1, pp_size=1, cp_size=world,
            cp_config={
                "cp_type": CpType.HELIX,
                "tokens_per_block": TOKENS_PER_BLOCK,
                "use_nccl_for_alltoall": True,
                "fifo_version": 2,
            },
        )
        mapping_tp = Mapping(world_size=world, rank=rank, tp_size=world)
        self.mla = make_mla(
            mapping=mapping_tp,
            device=device,
            mapping_with_cp=self.mapping_cp,
            reduce_output=True,
            q_shard=(0, 1),        # q_b/kv_b unsharded under helix (tp inside MLA = 1)
            o_shard=(rank, world),  # v_b/o/g sharded by cp rank
        )
        # Context KV fill must use the BASE MLA context path: Kimi's helix
        # override only supports terminal one-token contexts (the executor
        # fills helix KV via cache transfer, never a local context pass). The
        # base path runs dense local attention (output discarded) and fills
        # this rank's KV shard correctly — the pattern of test_mla_helix.py.
        import types

        from tensorrt_llm._torch.modules.attention import MLA

        self.mla.forward_context = types.MethodType(MLA.forward_context, self.mla)

    def setup_decode(self, batch: int, isl: int, hist_fn, dec: torch.Tensor):
        w = self.world
        blocks_total = isl // TOKENS_PER_BLOCK
        assert blocks_total % w == 0, "ISL must be a multiple of tokens_per_block*world"
        local_len = isl // w
        self.cache = make_kv_cache_manager(
            mapping=self.mapping_cp, batch=batch, local_seq_len=local_len,
            device_seq_extra=4,
        )
        add_requests(self.cache, batch, local_len, is_gen=True)

        # Block-cyclic ownership: rank r owns blocks r, r+w, ... of each sequence.
        own_blocks = list(range(self.rank, blocks_total, w))
        pos_per_seq = torch.cat(
            [
                torch.arange(
                    b * TOKENS_PER_BLOCK, (b + 1) * TOKENS_PER_BLOCK,
                    dtype=torch.int, device=self.device,
                )
                for b in own_blocks
            ]
        )  # [local_len] global positions
        pos_ctx = pos_per_seq.repeat(batch)
        from _torch.modules.helix_test_utils import activate_all_ranks_for_context

        from dp_common import CTX_FILL_CHUNK_REQS

        with torch.inference_mode():
            for start in range(0, 0 if dp_common.SKIP_CTX_FILL else batch, CTX_FILL_CHUNK_REQS):
                n = min(CTX_FILL_CHUNK_REQS, batch - start)
                md_ctx = make_metadata(
                    kv_cache_manager=self.cache, batch=n, phase="prefill",
                    seq_len=local_len, cached=0, mapping=self.mapping_cp,
                    request_ids=list(range(start, start + n)),
                )
                pos_chunk = pos_per_seq.repeat(n)
                activate_all_ranks_for_context(md_ctx, pos_chunk)
                x = (
                    hist_fn(start, start + n)
                    .view(n, blocks_total, TOKENS_PER_BLOCK, HIDDEN_SIZE)[:, own_blocks]
                    .reshape(n * local_len, HIDDEN_SIZE)
                    .contiguous()
                )
                scratch = x.new_empty(n * local_len, self.mla.num_heads_tp * V_HEAD_DIM)
                self.mla.forward_impl(pos_chunk, x, md_ctx, output=scratch)
        torch.cuda.synchronize()

        self.md = self._gen_metadata(batch, local_len, isl)
        self.pos = torch.full((batch,), isl, dtype=torch.int, device=self.device)
        self.dec_full = dec.contiguous()

    def _gen_metadata(self, batch: int, local_len: int, isl: int):
        from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

        md = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([1] * batch, dtype=torch.int, device="cpu"),
            num_contexts=0,
            request_ids=list(range(batch)),
            prompt_lens=[local_len] * batch,
            max_num_requests=batch,
            max_num_tokens=max(batch, 32),
            kv_cache_manager=self.cache,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[local_len] * batch
            ),
            mapping=self.mapping_cp,
        )
        md.enable_helix = True
        inactive = [self.rank != self.world - 1] * batch
        md.helix_is_inactive_rank = torch.tensor(
            inactive, dtype=torch.bool, device="cuda"
        )
        md.helix_is_inactive_rank_cpu = (
            md.helix_is_inactive_rank.to("cpu").pin_memory()
        )
        md.helix_position_offsets = torch.full(
            (batch,), isl, dtype=torch.int, device="cuda"
        )
        md.helix_position_offsets_cpu = (
            md.helix_position_offsets.to("cpu").pin_memory()
        )
        md.helix_total_input_len = torch.full(
            (batch,), isl, dtype=torch.int, device="cuda"
        )
        md.helix_total_input_len_cpu = md.helix_total_input_len.to("cpu").pin_memory()
        causal = torch.full((batch,), 1, dtype=torch.int, device="cuda")
        md.helix_causal_seqlens_kv_global = causal
        md.helix_causal_seqlens_kv_global_cpu = causal.to("cpu").pin_memory()
        md.prepare()
        return md

    def run_decode(self) -> torch.Tensor:
        return self.mla.forward(self.pos, self.dec_full, self.md)

    def teardown(self):
        self.cache.shutdown()


VARIANTS = {"bs_split": BsSplitVariant, "ulysses": UlyssesVariant, "helix": HelixVariant, "tp": TpVariant}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _time(run: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            run()
    torch.cuda.synchronize()
    mpi_barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.inference_mode():
        for i in range(iters):
            starts[i].record()
            run()
            ends[i].record()
    torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _time_graph(run: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    with torch.inference_mode():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        mpi_barrier()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _capture_trace(run: Callable[[], torch.Tensor], out_path: Path) -> str:
    with torch.inference_mode():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            run()
            torch.cuda.synchronize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=["bs_split", "ulysses", "helix"])
    parser.add_argument("--decode-batch-sizes", type=int, nargs="*", default=[8, 32, 128])
    parser.add_argument("--isl", type=int, default=2048)
    parser.add_argument("--prefill-seqs", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--phases", nargs="+", choices=("decode", "prefill"),
                        default=["decode"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true",
                        help="cross-check each variant against the single-GPU reference")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--graph-timing", action="store_true",
                        help="time CUDA-graph replays (required for real tables)")
    parser.add_argument("--skip-ctx-fill", action="store_true",
                        help="perf-only: leave KV zero-filled (shapes/traffic "
                        "identical to a real history; disables --check validity)")
    parser.add_argument("--tag", default="dp_attention")
    args = parser.parse_args()

    rank, world = mpi_rank(), mpi_world_size()
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    # NOTE: no ambient torch.device ctx here — attention metadata internals
    # create CPU tensors implicitly (helix prepare() index_select), and this
    # bench passes explicit devices everywhere.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29911")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world)
    dp_common.SKIP_CTX_FILL = args.skip_ctx_fill

    rows = []
    checks = {}
    for phase in args.phases:
        grid = args.decode_batch_sizes if phase == "decode" else [args.prefill_seqs]
        for size in grid:
            hist_seed = args.seed + size
            if phase == "decode":
                hist_fn = lambda lo, hi: hidden_for_reqs(  # noqa: E731
                    lo, hi, args.isl, hist_seed, device
                )
                dec = hidden_batch(size, hist_seed + 1, device)
                ref_out = None
                if args.check:
                    ref = FullBatchReference(device)
                    ref_out = ref.decode(size, args.isl, hist_fn, dec)
            else:
                hist = hidden_for_reqs(0, size, args.prefill_chunk, hist_seed, device).view(
                    size, args.prefill_chunk, HIDDEN_SIZE
                )
            for name in args.variants:
                cls = VARIANTS[name]
                if phase == "prefill" and not cls.supports_prefill:
                    continue
                variant = None
                try:
                    variant = cls(rank, world, device)
                    if phase == "decode":
                        variant.setup_decode(size, args.isl, hist_fn, dec)
                        run = variant.run_decode
                    else:
                        variant.setup_prefill(size, args.prefill_chunk, hist)
                        run = variant.run_prefill
                    with torch.inference_mode():
                        run()
                    torch.cuda.synchronize()
                    ok = True
                    err = None
                except Exception as exc:  # noqa: BLE001 - recorded in receipt
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"
                if not all(mpi_allgather(ok)):
                    rows.append({
                        "phase": phase, "size": size,
                        "isl": args.isl if phase == "decode" else args.prefill_chunk,
                        "variant": name, "error": err or "failed on another rank",
                    })
                    if rank == 0:
                        print(f"[dp_attention] {phase} size={size} {name}: FAILED "
                              f"({err})", flush=True)
                    if variant is not None and hasattr(variant, "cache"):
                        variant.teardown()
                    mpi_barrier()
                    continue
                mpi_barrier()
                if args.check and phase == "decode" and ref_out is not None:
                    with torch.inference_mode():
                        out = run().clone()
                    torch.cuda.synchronize()
                    if name == "helix":
                        got, want = out, ref_out
                    else:
                        bl = size // world
                        got = out
                        want = ref_out[rank * bl : (rank + 1) * bl]
                    metrics = _correctness(got, want, atol=0.1, rtol=0.05,
                                           min_close_fraction=0.85)
                    gathered = mpi_allgather(metrics)
                    checks[f"{phase}_{size}_{name}"] = {
                        f"rank{i}": m for i, m in enumerate(gathered)
                    }
                timing = (_time_graph if args.graph_timing else _time)(
                    run, args.warmup, args.iters)
                trace_path = None
                if args.trace:
                    trace_path = _capture_trace(
                        run,
                        LOCAL_RESULTS / "traces"
                        / f"{args.tag}_{phase}_{name}_b{size}_isl{args.isl}_w{world}_rank{rank}.json",
                    )
                gathered_t = mpi_allgather(timing)
                row = {
                    "phase": phase,
                    "size": size,
                    "isl": args.isl if phase == "decode" else args.prefill_chunk,
                    "variant": name,
                    "score_median_ms": max(t["median_ms"] for t in gathered_t),
                    "per_rank": {f"rank{i}": t for i, t in enumerate(gathered_t)},
                    "traces": mpi_allgather(trace_path) if args.trace else None,
                }
                rows.append(row)
                variant.teardown()
                mpi_barrier()
                if rank == 0:
                    print(
                        f"[dp_attention] {phase} size={size} {name}: "
                        f"{row['score_median_ms']:.3f} ms median",
                        flush=True,
                    )

    receipt = {
        "kind": "dp_attention",
        "shape": {
            "hidden": HIDDEN_SIZE, "heads": NUM_HEADS, "kv_lora": KV_LORA_RANK,
            "nope": 128, "rope": QK_ROPE_HEAD_DIM, "v": V_HEAD_DIM,
            "world": world, "isl": args.isl,
        },
        "labels": {
            "bs_split": "DP-Attention-BS-SPLIT (whole requests per rank, no comm)",
            "ulysses": "DP-Attention-Ulysses (Q-head A2A + latent allgather, 12-head core)",
            "helix": "DP-Attention-Ring (fork Helix DCP: KV sharded, partial-attn merge)",
        },
        "timing": {"warmup": args.warmup, "iters": args.iters,
                   "launch": "cuda_graph" if args.graph_timing else "eager"},
        "rows": rows,
        "checks": checks,
    }
    if rank == 0:
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        out = LOCAL_RESULTS / f"{args.tag}_w{world}_isl{args.isl}.json"
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[dp_attention] receipt={out}", flush=True)


if __name__ == "__main__":
    main()
