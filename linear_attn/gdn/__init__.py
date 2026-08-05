"""GDN (Gated DeltaNet, scalar-gate) — input contract, exact PyTorch
reference, and the registry of framework kernels. See ``gdn_attention.py``."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
