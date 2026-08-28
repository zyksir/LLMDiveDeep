"""Import setup and provenance pinning for the unchanged TRT-LLM harness.

The routed-expert module is built through TRT-LLM's own microbenchmark
harness (``tests/microbenchmarks/bench_moe`` plus the unittest quantize
fixtures), imported UNCHANGED from the TRT-LLM checkout — the same launch
pattern as the proven ``kimi_k3_layer/bench_a2a_megamoe_pipeline.py`` /
``bench_a2a_megamoe_sweep.py`` receipts. Nothing under ``TRTLLM_REPO`` is
ever modified by this package.

To keep the baseline pinned, ``PROVENANCE.json`` records the SHA256 of every
harness file this package imports (written by ``gen_provenance.py``).
``verify_provenance()`` re-hashes them at run time; a mismatch is reported in
the receipt so a silently moved checkout can never masquerade as the
finalized baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent  # /node-storage/LLMDiveDeep
# The implementation-reference checkout (branch yikai/optimized-k3). Every
# harness file this package imports is byte-identical there to the checkout
# matching the container's installed tensorrt_llm build (verified via
# PROVENANCE.json hashes), so importing from the reference checkout is safe
# with the container's compiled bindings.
TRTLLM_REPO = Path(os.environ.get("TRTLLM_REPO", "/node-storage/trt-llm"))
PROVENANCE_JSON = PACKAGE_DIR / "PROVENANCE.json"

# Harness roots imported by this package (identical to the prior receipts).
HARNESS_SYS_PATHS = (
    TRTLLM_REPO / "tests" / "microbenchmarks",
    TRTLLM_REPO / "tests" / "unittest",
)

# Files whose bytes define the harness behavior we rely on.
HARNESS_FILE_GLOBS = (
    ("tests/microbenchmarks/bench_moe", "**/*.py"),
    ("tests/unittest/_torch/modules/moe", "quantize_utils.py"),
    ("tests/unittest/_torch/modules/moe", "moe_test_utils.py"),
    ("tests/unittest/_torch", "helpers.py"),
    ("tests/unittest/utils", "util.py"),
)


def setup_sys_path() -> None:
    for path in (*HARNESS_SYS_PATHS, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _iter_harness_files():
    for rel_dir, glob in HARNESS_FILE_GLOBS:
        base = TRTLLM_REPO / rel_dir
        for file in sorted(base.glob(glob)):
            if file.is_file() and "__pycache__" not in file.parts:
                yield file


def compute_manifest() -> dict[str, str]:
    manifest = {}
    for file in _iter_harness_files():
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest[str(file.relative_to(TRTLLM_REPO))] = digest
    return manifest


def verify_provenance() -> dict:
    """Compare current harness file hashes against the pinned manifest."""
    if not PROVENANCE_JSON.exists():
        return {"status": "missing-manifest"}
    pinned = json.loads(PROVENANCE_JSON.read_text())
    current = compute_manifest()
    mismatched = sorted(
        path
        for path in set(pinned["files"]) | set(current)
        if pinned["files"].get(path) != current.get(path)
    )
    return {
        "status": "match" if not mismatched else "MISMATCH",
        "trtllm_repo": str(TRTLLM_REPO),
        "pinned_commit": pinned.get("trtllm_commit"),
        "mismatched_files": mismatched,
    }
