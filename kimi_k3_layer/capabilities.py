"""Per-stage capability gating so one build serves B200, GB200 and B300.

The measured plan (`measured_config`) names the fastest implementation of every
stage. On a part where one of those implementations has no kernel, the layer
must degrade *that stage only* and keep every other optimization — never fall
back to the reference forward wholesale, and never crash in production.

Two mechanisms, deliberately separate:

* **Static tables** (`_ARCH_GAPS`, `_SITU_ARCH_GAPS`): kernels we know are
  absent on a target. `_ARCH_GAPS` is currently empty; the one gap we know of is
  activation-conditional. trtllmGen's MXFP4 MoE GEMM ships SwiGlu-only kernels
  on sm_103, so `ExpertBackend.NATIVE` is unusable there when the runner runs
  SiTU.

  **SiTU is not signalled by `activation_type`.** TRT-LLM computes
  `_is_situ_activation = (activation_type == Swiglu) and
  pretrained_config.hidden_act == "situ"` (`fused_moe_trtllm_gen.py:231`), so a
  caller passing `ActivationType.Swiglu` still runs SiTU on a real K3
  checkpoint -- `hidden_act` comes from the checkpoint's config.json, not from
  any code in this repo. Consequences worth keeping straight:

  * production K3 (real checkpoint, `hidden_act="situ"`) -> SiTU -> the sm_103
    gap APPLIES;
  * this repo's benchmark (`k3_pretrained_config()` sets no `hidden_act`, and a
    bare `PretrainedConfig` has none) -> SwiGlu -> the gap does not apply, and
    the bench is therefore not exercising production's expert activation.

  So `situ_experts` is read off TRT-LLM's own `_is_situ_activation` rather than
  re-derived, and never inferred from `activation_type` alone.

  Note that SiTU exists **only** in the fork: stock rc19 and rc23 have no
  `_torch/modules/situ.py` and no SiTU reference in `moe_op_backend.py` at all.
* **Operator override** (`B10_DISABLE_STAGES`): a comma-separated list of stage
  names to force off without a code change, for when production finds a gap
  this table does not know about yet.

Everything is decided once, at layer init, outside any timed or captured
region, and **agreed across ranks**: a stage that carries a collective must be
enabled on all TP ranks or none, or the ranks deadlock instead of failing. The
agreement is an all-reduce MIN over the capability bitmask.

What this module does NOT do yet: catch a kernel failure at first use and
downgrade live. That needs hardware to build safely (the failure has to be
caught outside graph capture, with the fallback path already warmed), and is
tracked as the follow-up in serving_startup/README.md.

Usage:

    caps = Capabilities.probe(group)          # once, at init
    config = caps.filter(measured_config(t))  # per dispatch, cheap
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from common import arch

if TYPE_CHECKING:  # avoid importing the layer (and TRT-LLM) at module import
    from kimi_k3_layer.b10_kimi_k3_moe_layer import ExperimentConfig

_DISABLE_ENV = "B10_DISABLE_STAGES"

# Architectures whose `measured_config` thresholds come from measurements on
# that part. Anything else runs a plan tuned elsewhere: correct, but not
# optimal, and `log_once` says so.
TUNED_ARCHES = frozenset({(10, 0)})  # B200 / GB200, Aug-19 retune

# stage name -> architectures on which it has no kernel. Entries here are
# UNCONDITIONAL; activation-dependent gaps live in _SITU_ARCH_GAPS below.
_ARCH_GAPS: dict[str, frozenset[tuple[int, int]]] = {}

# Gaps that exist only when the routed experts run the SiTU activation.
#
# trtllmGen's MxE4m3MxE2m1BlockScaleMoERunner ships SwiGlu-only MXFP4 kernels on
# sm_103; constructing it with act=SiTU raises "No kernel found" (measured on
# GB300 with a direct runner unit test). This layer hands the runner
# `ActivationType.Swiglu` and applies SiTU itself (grafted `SiTUAndMul` for the
# shared branch, `situ_and_mul` for the inline path), so the gap does NOT apply
# here -- SwiGlu is precisely the variant sm_103 has. It applies to a caller
# that pins the trtllmGen runner *and* passes SiTU, which is what the
# fork-integrated serving path does; hence the flag rather than a hard-coded
# arch entry.
_SITU_ARCH_GAPS: dict[str, frozenset[tuple[int, int]]] = {
    "native_experts": frozenset({(10, 3)}),
}

# The gated stages, in report order. Explicit rather than derived from
# `fields()` because the dataclass also carries `situ_experts`, which is probe
# *input* recorded for the report -- not a stage. Deriving from fields() would
# all-reduce it and report the normal False as a downgrade.
_STAGES: tuple[str, ...] = (
    "fused_front_cute",
    "radix_routing",
    "multimem_tail",
    "sharded_fc2_tail",
    "sharded_prefill_fc1",
    "fc2_shard_prefill_tail",
    "native_experts",
)

# Human-readable reason per gap, for the log line and the degraded report.
_GAP_REASON = {
    "native_experts": (
        "trtllmGen MXFP4 MoE GEMM ships SwiGlu-only kernels on this arch; "
        "Kimi-K3 routed experts need SiTU -> using the FlashInfer op backend"
    ),
}


@dataclass(frozen=True)
class Capabilities:
    """One flag per gated stage. True = the fast implementation may be used."""

    fused_front_cute: bool = True      # DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
    radix_routing: bool = True         # Routing.RADIX
    multimem_tail: bool = True         # DecodeTail.FULL_FC2_MULTIMEM_SHARED_REDUCE
    sharded_fc2_tail: bool = True      # DecodeTail.SHARDED_FC2_OUTPUT_REDUCE
    sharded_prefill_fc1: bool = True   # PrefillFC1.SHARDED (+ its DMA gather)
    fc2_shard_prefill_tail: bool = True  # PrefillTail.FC2_SHARD
    native_experts: bool = True        # ExpertBackend.NATIVE

    # True when the probe assumed SiTU-activated routed experts; recorded so
    # `degraded()` can explain the reason it actually used.
    situ_experts: bool = False

    # ------------------------------------------------------------------ probe
    @classmethod
    def probe(cls, group=None, *, situ_experts: bool = False) -> "Capabilities":
        """Decide every stage for this device, then agree across ranks.

        ``situ_experts`` says whether the routed-expert GEMM is invoked with the
        SiTU activation. It matters because trtllmGen's MXFP4 kernel coverage is
        activation-dependent (see ``_SITU_ARCH_GAPS``); pass what the caller
        actually configures rather than assuming the model's nominal activation.
        """
        version = arch.sm_version()
        disabled = {
            name.strip() for name in os.environ.get(_DISABLE_ENV, "").split(",")
            if name.strip()
        }
        unknown = disabled.difference(_STAGES)
        if unknown:
            raise ValueError(
                f"{_DISABLE_ENV} names unknown stage(s): {sorted(unknown)}; "
                f"valid: {sorted(_STAGES)}")

        values: dict[str, bool] = {"situ_experts": situ_experts}
        for stage in _STAGES:
            gap = set(_ARCH_GAPS.get(stage, frozenset()))
            if situ_experts:
                gap |= set(_SITU_ARCH_GAPS.get(stage, frozenset()))
            values[stage] = stage not in disabled and version not in gap
        caps = cls(**values)
        return caps._agree(group)

    def _agree(self, group) -> "Capabilities":
        """All-reduce MIN so every rank runs the same stages.

        A stage that issues a collective must be on everywhere or nowhere: if
        rank 3 downgrades the multimem tail and rank 4 does not, the job hangs
        in mismatched collectives rather than failing a check.
        """
        import torch
        import torch.distributed as dist

        if group is None or not dist.is_available() or not dist.is_initialized():
            return self
        if dist.get_world_size(group) == 1:
            return self
        # Stages only: `situ_experts` is probe input, not a capability, and
        # MIN-ing it would rewrite the reason string reported later.
        flags = torch.tensor(
            [int(getattr(self, name)) for name in _STAGES],
            dtype=torch.int32, device="cuda")
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
        return replace(self, **{
            name: bool(value)
            for name, value in zip(_STAGES, flags.tolist())})

    # ----------------------------------------------------------------- report
    def degraded(self) -> tuple[tuple[str, str], ...]:
        """((stage, reason), ...) for every stage that is switched off."""
        version = arch.sm_version()
        out = []
        for stage in _STAGES:
            if getattr(self, stage):
                continue
            gap = set(_ARCH_GAPS.get(stage, frozenset()))
            if self.situ_experts:
                gap |= set(_SITU_ARCH_GAPS.get(stage, frozenset()))
            if version in gap:
                reason = _GAP_REASON.get(stage, "no kernel on this arch")
            else:
                reason = f"disabled via {_DISABLE_ENV}"
            out.append((stage, reason))
        return tuple(out)

    def log_once(self, rank: int = 0) -> None:
        if rank != 0:
            return
        target = f"sm_{arch.suffix()}"
        if arch.sm_version() not in TUNED_ARCHES:
            # Availability and tuning are different questions: every stage
            # below may be *runnable* here while its threshold still comes
            # from another part's measurements. Say so rather than let a
            # B200-shaped plan pass for a measured one.
            print(f"[k3-caps] {target}: running the B200-tuned plan; "
                  "thresholds (DECODE_MAX_TOKENS, PREFILL_SHARDED_MIN_TOKENS, "
                  "SHARDED_FC2_MAX_TOKENS, ...) are NOT retuned for this arch "
                  "-- see agent/RUNBOOK.md §1 for the retune", flush=True)
        degraded = self.degraded()
        if not degraded:
            print(f"[k3-caps] {target}: all stages available", flush=True)
            return
        # Loud on purpose: "running degraded" must be visible in prod logs and
        # scrapeable for an alert, not inferred from a latency regression.
        print(f"[k3-caps] {target}: DEGRADED, {len(degraded)} stage(s) off",
              flush=True)
        for stage, reason in degraded:
            print(f"[k3-caps]   {stage}: {reason}", flush=True)

    # ------------------------------------------------------------------ filter
    def filter(self, config: "ExperimentConfig") -> "ExperimentConfig":
        """Rewrite a plan so it only names implementations this part has.

        Each ladder ends at something every Blackwell part can run, so the
        result is always executable; the layer contract is unchanged.
        """
        from kimi_k3_layer.b10_kimi_k3_moe_layer import (
            DecodeFront, DecodeTail, ExpertBackend, PrefillFC1, PrefillTail,
            Routing,
        )

        if not config.enabled:
            return config
        changes: dict[str, object] = {}

        # front: fused CuTeDSL dual-out -> separate sharded fc1 -> separate full
        if (not self.fused_front_cute
                and config.decode_front is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE):
            changes["decode_front"] = DecodeFront.SEPARATE_SHARDED_FC1
        # routing: radix kernel -> reference gate + topk
        if not self.radix_routing and config.routing is Routing.RADIX:
            changes["routing"] = Routing.REFERENCE
        # decode tail: multimem full-fc2 -> sharded fc2 -> packed latent
        tail = config.decode_tail
        if (not self.multimem_tail
                and tail is DecodeTail.FULL_FC2_MULTIMEM_SHARED_REDUCE):
            tail = (DecodeTail.SHARDED_FC2_OUTPUT_REDUCE if self.sharded_fc2_tail
                    else DecodeTail.PACKED_LATENT_SHARED_REDUCE)
        if (not self.sharded_fc2_tail
                and tail is DecodeTail.SHARDED_FC2_OUTPUT_REDUCE):
            tail = DecodeTail.PACKED_LATENT_SHARED_REDUCE
        if tail is not config.decode_tail:
            changes["decode_tail"] = tail
        # prefill fc1: sharded (+ DMA gather) -> full
        if (not self.sharded_prefill_fc1
                and config.prefill_fc1 is PrefillFC1.SHARDED):
            changes["prefill_fc1"] = PrefillFC1.FULL
        # prefill tail: fc2 shard -> packed
        if (not self.fc2_shard_prefill_tail
                and config.prefill_tail is PrefillTail.FC2_SHARD):
            changes["prefill_tail"] = PrefillTail.PACKED
        # experts: native trtllmGen runner -> FlashInfer op backend
        if (not self.native_experts
                and config.prefill_expert_backend is ExpertBackend.NATIVE):
            changes["prefill_expert_backend"] = ExpertBackend.FLASHINFER

        return replace(config, **changes) if changes else config


def _selftest() -> None:
    """No-GPU check of the arch table and every ladder rung.

    Run as ``python3 -m kimi_k3_layer.capabilities`` (needs TRT-LLM importable
    for the enums, but no device).
    """
    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        DecodeFront, DecodeTail, ExpertBackend, PrefillFC1, PrefillTail,
        Routing, measured_config,
    )

    original = os.environ.get(arch._FORCE_ENV)
    try:
        # B200 / GB200: everything available.
        os.environ[arch._FORCE_ENV] = "10.0"
        caps = Capabilities.probe()
        assert caps.degraded() == (), caps.degraded()
        plan = measured_config(16384)
        assert caps.filter(plan) is plan or caps.filter(plan) == plan
        assert plan.prefill_expert_backend is ExpertBackend.NATIVE
        print("sm_100a: no downgrades; 16384-token plan keeps NATIVE experts")

        # B300 / GB300 as THIS layer configures it (SwiGlu to the runner):
        # nothing degrades -- SwiGlu is the variant sm_103 ships.
        os.environ[arch._FORCE_ENV] = "10.3"
        caps = Capabilities.probe()
        assert caps.degraded() == (), caps.degraded()
        assert (caps.filter(measured_config(16384)).prefill_expert_backend
                is ExpertBackend.NATIVE)
        print("sm_103a + SwiGlu experts: no downgrades, NATIVE kept")

        # B300 / GB300 with SiTU-activated experts (fork-integrated path):
        # only the expert backend drops.
        caps = Capabilities.probe(situ_experts=True)
        assert [stage for stage, _ in caps.degraded()] == ["native_experts"]
        filtered = caps.filter(measured_config(16384))
        assert filtered.prefill_expert_backend is ExpertBackend.FLASHINFER
        assert filtered.prefill_fc1 is PrefillFC1.SHARDED
        assert filtered.prefill_tail is PrefillTail.FC2_SHARD
        assert filtered.routing is Routing.RADIX
        decode = caps.filter(measured_config(64))
        assert decode.decode_front is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
        assert decode.decode_tail is DecodeTail.FULL_FC2_MULTIMEM_SHARED_REDUCE
        caps.log_once()
        print("sm_103a + SiTU experts: only the expert backend downgraded; "
              "fused front, radix routing, multimem tail, sharded fc1 all kept")

        # Operator override exercises the remaining ladders.
        os.environ[_DISABLE_ENV] = (
            "fused_front_cute,radix_routing,multimem_tail,"
            "sharded_prefill_fc1,fc2_shard_prefill_tail")
        caps = Capabilities.probe()
        decode = caps.filter(measured_config(64))
        assert decode.decode_front is DecodeFront.SEPARATE_SHARDED_FC1
        assert decode.routing is Routing.REFERENCE
        assert decode.decode_tail is DecodeTail.SHARDED_FC2_OUTPUT_REDUCE
        prefill = caps.filter(measured_config(4096))
        assert prefill.prefill_fc1 is PrefillFC1.FULL
        assert prefill.prefill_tail is PrefillTail.PACKED
        print("overrides: every ladder rung reachable")

        os.environ[_DISABLE_ENV] = "multimem_tail,sharded_fc2_tail"
        caps = Capabilities.probe()
        assert (caps.filter(measured_config(64)).decode_tail
                is DecodeTail.PACKED_LATENT_SHARED_REDUCE)
        assert (caps.filter(measured_config(4)).decode_tail
                is DecodeTail.PACKED_LATENT_SHARED_REDUCE)
        print("both tails off: falls through to the packed latent tail")

        try:
            os.environ[_DISABLE_ENV] = "not_a_stage"
            Capabilities.probe()
        except ValueError as error:
            print(f"unknown stage rejected: {str(error)[:52]}...")
        else:
            raise AssertionError("unknown stage name must raise")
    finally:
        os.environ.pop(_DISABLE_ENV, None)
        if original is None:
            os.environ.pop(arch._FORCE_ENV, None)
        else:
            os.environ[arch._FORCE_ENV] = original
    print("kimi_k3_layer.capabilities selftest OK")


if __name__ == "__main__":
    _selftest()
