"""KDA (Kimi Delta Attention) — everything KDA-specific lives in this package.

Layout (start with ``kda_attention.py`` if you are new to KDA):

- ``kda_attention.py``  the NAIVE PyTorch implementation of the math — the
  gate activation and the exact token recurrence. No kernels, no frameworks;
  read this to understand what every other implementation computes.
- ``inputs.py``         synthetic input contracts/builders shared by every
  backend and bench (packed decode inputs, varlen prefill inputs).
- ``kda_decode_register.py``   registry of every one-token decode implementation
  (FLA, SGLang, vLLM, TRT-LLM, b10) and how each one is called.
- ``kda_prefill_register.py``  registry of every chunk-prefill implementation
  (FLA, FlashKDA, INT21 PTX, SGLang, vLLM, TRT-LLM, b10) + SGLang's
  module-level ``extend`` path.
- ``kda_verify_register.py``   the speculative-decode (MTP verify) kernel inventory:
  one closure per framework kernel across the ``save_ssm`` /
  ``replay_ssm_split`` / ``replay_ssm_fused`` schemes (taxonomy in
  ../KDA.md section 2).
- ``kda_replayssm_fold.py``  vendored SGLang PR #32541 fold kernel — the
  ``replay_ssm_split`` ``fold_replay`` Triton kernel timed by
  ``benchmarks/bench_kda_spec_verify.py``. Not a register; a real kernel.
- ``attention_modules.py``   full KDA attention-layer modules (projections +
  conv + recurrence + norm) for the layer-level bench.
- ``b10/``              kernels authored or generated in this repo (CuTeDSLGen
  champions + the Triton FlashKDA port), all ``b10_``-prefixed. Each module
  documents the original Triton/CUTLASS kernel it aligns to.

The registry mechanism (``BackendRegistry``) and the ``Shape`` contract live
one level up in ``linear_attention.py``; framework import shims live in
``frameworks.py``.
"""

import sys
from pathlib import Path

# Make the flat top-level modules (linear_attention, frameworks) and the
# sibling packages importable regardless of where the entry script lives.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kda.kda_attention import (  # noqa: E402, F401
    SAFE_GATE_LOWER_BOUND,
    activate_kda_gate,
    kda_recurrent_reference,
)
