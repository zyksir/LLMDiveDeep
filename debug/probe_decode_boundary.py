#!/usr/bin/env python3
"""Retune probe: is DECODE_MAX_TOKENS=256 still the right boundary on sm_103a?

The B300 TP8 run reproduces B200 within +-1.6% everywhere except B=256, which
lands 10.1% slower (+11.9% vs +21.5%) -- and 256 is exactly the top of the
decode regime. `measured_config` sends tokens <= DECODE_MAX_TOKENS down the
decode path; this lowers that boundary so the same size is served by the
prefill plan instead, which is what RUNBOOK.md SS1 asks a new node to re-decide.

  mpirun -n 8 --allow-run-as-root python3 debug/probe_decode_boundary.py \
      --sizes 256 --iters 100 --n-inputs 8 --csv-suffix _b300_prefill256

Report-only: patches the module constants for this process, touches no source.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import kimi_k3_layer.b10_kimi_k3_moe_layer as layer_mod

_BOUNDARY = 128

# Both boundaries are bound as function-definition defaults
# (b10_kimi_k3_moe_layer.py:207 and :462), so rebinding the module constants
# is not enough -- `self.prefill_baseline_max_tokens` keeps the value captured
# at import and 256 <= 256 still returns `all_off()`, i.e. the exact reference
# path (that is the err=0.00e+00 tell). Wrap the plan function instead, which
# is the one place the layer reads both thresholds from.
layer_mod.DECODE_MAX_TOKENS = _BOUNDARY
layer_mod.B10KimiK3MoELayer.OPT_MAX_TOKENS = _BOUNDARY

_measured_config = layer_mod.measured_config


def _forced_config(tokens, **kwargs):
    kwargs["prefill_baseline_max_tokens"] = _BOUNDARY
    return _measured_config(tokens, **kwargs)


layer_mod.measured_config = _forced_config

from kimi_k3_layer.bench_b10_kimi_k3_moe_layer import main

if __name__ == "__main__":
    print(f"[probe] DECODE_MAX_TOKENS lowered to {_BOUNDARY}", flush=True)
    main()
