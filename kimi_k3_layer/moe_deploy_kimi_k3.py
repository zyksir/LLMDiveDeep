"""Deployment-shaped Kimi-K3 MoE classes: weight layout fixed at init.

``KimiK3MoEB10`` proved the optimizations with every switch runtime-
flippable; a real TRT-LLM integration cannot do that - the weight
LAYOUT (sharded vs replicated, fused vs separate) is decided when the
checkpoint is loaded and never changes. These two classes freeze the
measured-best configuration per deployment shape and expose NO
runtime switches (``set_opt_flags`` is locked).

Init-time decisions and how each falls:

  fc1        COLUMN-SHARDED in both classes: prefill needs it (1/world
             FLOPs) and decode<=16 needs it (fused AG+MXFP8 quantize).
  merge3     ONE stored tensor [gate | fc1-shard | shared g/u]
             ([2880, 7168]) serves BOTH regimes with no duplication:
             decode GEMMs against all rows; prefill GEMMs against the
             contiguous ROW SLICE [:1344] and reads the shared g/u
             weight as the [1344:] view for its side-stream GEMM.
             (This is why shared-fusion is NOT actually a conflict.)
  fc2        THE one real conflict. Prefill + decode<16 want the
             column shard; the decode>=16 ref tail wants the FULL
             weight (no output collective). Resolution per class:

KimiK3MoEForAgg  (aggregated serving: prefill + decode in one engine)
    fc2 stored SHARDED ([7168, 448]/rank, saves ~45 MB/rank/layer vs
    replicating). Decode uses the shard tail at ALL sizes (measured
    cost: only B=16 regresses, 79.0 -> ~82 us, +14.5% -> ~+11%).
    Optional ``full_fc2_allgather=True`` (env B10_AGG_FULL_FC2=1):
    one-time all-gather of the fc2 shards into a persistent full
    buffer - weights are static, so this happens ONCE at init and
    overlaps engine warmup, not per step - which re-enables the
    ref tail at B>=16 for +51 MB/rank/layer.

KimiK3MoEForDisAggDecode  (disaggregated decode-only engine)
    fc2 stored FULL + one contiguous column-slice copy (6.4 MB): the
    full weight feeds the B>=16 ref tail (fused finalize+AR+norm +
    CuTeDSL tail GEMM), the slice feeds the B<16 shard tail. The
    prefill path is disabled (prefill never reaches this engine).

Measured expectations (see moe_optimization.md; c8/c10 runs):
  DisAgg decode = the validated finals: +33.8/+33.8/+32.7/+22.4/+14.5%
  Agg decode    = same at B<=8; B=16 ~+11% (shard tail) or +14.5%
                  with full_fc2_allgather.
  Agg prefill   = the b10 prefill path (+12..+22% at 512..8k tokens).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .config import NUM_EXPERTS
from .moe_b10_kimi_k3 import KimiK3MoEB10


class _DeployMoE(KimiK3MoEB10):
    """Shared plumbing: locked flags + merge3-view weight aliasing."""

    def _locked_flags(self) -> dict:
        """Frozen set_opt_flags kwargs (instance-resolved)."""
        return {}

    def init_opt(self, **kwargs) -> None:
        super().init_opt(**kwargs)
        # ONE stored input weight: the merge3 concat. The two-way
        # merged weight and the shared gate/up weight become ROW
        # VIEWS of it (contiguous slices are valid GEMM operands),
        # dropping the duplicate storage the research class kept.
        if self._world > 1:
            split = NUM_EXPERTS + self._latent_width
            self._fused_gf_w_sh = self._fused_in_w_sh[:split]
            self._shared_gate_up = self._fused_in_w_sh[split:]
        super().set_opt_flags(**self._locked_flags())
        self._flags_locked = True

    def set_opt_flags(self, **kwargs) -> None:
        if getattr(self, "_flags_locked", False):
            raise RuntimeError(
                f"{type(self).__name__} freezes its configuration at "
                "init (deployment weight layout cannot change at "
                "runtime); use KimiK3MoEB10 for ablations")
        super().set_opt_flags(**kwargs)


class KimiK3MoEForAgg(_DeployMoE):
    """Aggregated serving: prefill-first layout, decode on top.

    fc1 + fc2 SHARDED. Decode tail = shard tail at every size (the
    only stored fc2 is the column slice) unless ``full_fc2_allgather``
    materializes the full weight once at init.
    """

    def _locked_flags(self) -> dict:
        # ref tail needs the full fc2 -> only with the weight AG
        return dict(ref_tail=self._agg_full_fc2)

    def init_opt(self, *, full_fc2_allgather: bool | None = None,
                 **kwargs) -> None:
        if full_fc2_allgather is None:
            full_fc2_allgather = (
                os.environ.get("B10_AGG_FULL_FC2", "0") == "1")
        self._agg_full_fc2 = full_fc2_allgather
        super().init_opt(**kwargs)
        if self._world <= 1:
            return
        if full_fc2_allgather:
            # ONE-TIME weight all-gather (static weights: init cost,
            # overlaps engine warmup in a real integration). Rebuild
            # the full [7168, 3584] fc2 from the per-rank contiguous
            # [7168, 448] slices.
            shard = self._fc2_local.contiguous()
            parts = [torch.empty_like(shard) for _ in range(self._world)]
            dist.all_gather(parts, shard)
            self._fc2_full = torch.cat(parts, dim=1).contiguous()
        else:
            # deployment truth: the replicated weight is never
            # materialized. (The parent Linear keeps its copy only so
            # the bench's baseline/correctness reference still runs.)
            self._fc2_full = None
            # the >64-token decode fallback tail ("-fc2shard" branch)
            # would need the full weight - keep the shard tail through
            # the whole opt range instead (instance-level override)
            self._fc2_shard_max = self.OPT_MAX_TOKENS


class KimiK3MoEForDisAggDecode(_DeployMoE):
    """Disaggregated decode-only engine: decode-optimal layout.

    fc2 stored FULL (+ the 6.4 MB contiguous slice for the B<16
    tail); ref tail + fused finalize+AR+norm + CuTeDSL tail GEMM at
    B>=16. Prefill never reaches this engine: the prefill path is
    disabled and large batches fall back to the baseline forward.
    """

    def _locked_flags(self) -> dict:
        return dict(prefill_opt=False)

    def init_opt(self, **kwargs) -> None:
        kwargs["ag_ce"] = None  # no prefill copy-engine comm
        super().init_opt(**kwargs)
