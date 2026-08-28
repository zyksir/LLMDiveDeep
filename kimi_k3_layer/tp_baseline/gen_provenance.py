#!/usr/bin/env python3
"""Write PROVENANCE.json + PROVENANCE.md: import-vs-copy inventory + hashes."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date

from harness import PACKAGE_DIR, TRTLLM_REPO, compute_manifest

CONTAINER = (
    "trt-k3-prod image "
    "(baseten/dynamo-cache-aware-routing:trtllm-d9fa74fd94-d9d1a655f-c08257485a, "
    "installed tensorrt_llm 1.3.0rc19 fork with NVLinkOneSided/MegaMoE)"
)

# Everything TRT-LLM this package uses, classified per the structure directive:
# IMPORTED = imported unchanged (module path + commit); COPIED = none;
# ADAPTED = small in-package orchestration where direct import is impractical,
# with the reason recorded.
INVENTORY = [
    {
        "kind": "IMPORTED (installed package)",
        "what": "tensorrt_llm runtime: torch.ops.trtllm compiled ops, "
        "ConfigurableMoE/TRTLLMGenFusedMoE, NVLinkOneSided, GatedMLP, "
        "AllReduce, Mapping, ModelConfig, AutoTuner",
        "source": "container-installed tensorrt_llm (build of fork commit "
        "d9fa74fd94); the reference checkout's tensorrt_llm/ python diverges "
        "from the installed compiled bindings by ~129 files, so overlaying it "
        "via PYTHONPATH is not binary-safe",
    },
    {
        "kind": "IMPORTED (reference checkout)",
        "what": "bench_moe microbenchmark harness "
        "(tests/microbenchmarks/bench_moe/**): module build, mapping, quant "
        "params, routing plan + logits projection, specs, device utils",
        "source": "PYTHONPATH import from the checkout below; byte-identical "
        "to the container-build checkout (hashes pinned in this manifest)",
    },
    {
        "kind": "IMPORTED (reference checkout)",
        "what": "unittest MoE fixtures "
        "(tests/unittest/_torch/modules/moe/quantize_utils.py, "
        "moe_test_utils.py): MXFP4MXFP8QuantizeUtil weight generation "
        "(routed + shared experts), prepare_weights_from_backend, "
        "MXFP4MXFP8RefGatedMLPFusedMoE sequential reference",
        "source": "PYTHONPATH import from the checkout below; byte-identical "
        "to the container-build checkout",
    },
    {
        "kind": "IMPORTED (LLMDiveDeep)",
        "what": "bench_a2a_megamoe_pipeline._build_module/_correctness, "
        "bench_a2a_megamoe_sweep._per_rank_tokens/"
        "_prepare_local_backend_weights (cached local weights)",
        "source": "kimi_k3_layer/ in this repository (the proven MegaMoE "
        "receipt harness)",
    },
    {
        "kind": "ADAPTED (in-package)",
        "what": "bench_layer._build_routed_with_reference",
        "source": "bench_moe.build._build_moe_module — same calls, same "
        "order; direct import impractical because the stock builder discards "
        "the reference weights/module needed for the correctness gate",
    },
    {
        "kind": "ADAPTED (in-package)",
        "what": "bench_layer._run_layer_autotune",
        "source": "bench_moe.timing.autotune._run_autotune — same tuner "
        "settings; direct import impractical because the stock helper's "
        "signature is bound to a bare MoE forward, not a full-layer callable",
    },
    {
        "kind": "ADAPTED (in-package)",
        "what": "layer.KimiK3MoELayerBaseline / KimiK3SharedExperts wiring",
        "source": "tensorrt_llm/_torch/models/modeling_deepseekv3.py "
        "Deepseekv3MoE — shared GatedMLP(reduce_output=False) + AllReduce + "
        "routed MoE composition; direct import impractical because the model "
        "class pulls attention/pipeline dependencies irrelevant to a "
        "single-layer benchmark",
    },
]


def main() -> None:
    commit = subprocess.run(
        ["git", "-C", str(TRTLLM_REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(TRTLLM_REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(TRTLLM_REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    manifest = compute_manifest()
    payload = {
        "generated": date.today().isoformat(),
        "trtllm_repo": str(TRTLLM_REPO),
        "trtllm_branch": branch,
        "trtllm_commit": commit,
        "trtllm_worktree_dirty": dirty,
        "container": CONTAINER,
        "inventory": INVENTORY,
        "files": manifest,
    }
    (PACKAGE_DIR / "PROVENANCE.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Provenance: import-vs-copy inventory",
        "",
        f"Generated: {payload['generated']}",
        "",
        "All TRT-LLM kernels and Python modules are used UNCHANGED. Nothing is",
        "copied-and-edited: every piece is either imported directly (IMPORTED)",
        "or is a small, documented in-package adaptation (ADAPTED) where direct",
        "import is impractical. There are no COPIED files.",
        "`harness.verify_provenance()` re-hashes the imported harness files at",
        "every benchmark run and records the result in each JSON receipt.",
        "",
        f"- Reference checkout: `{payload['trtllm_repo']}` branch",
        f"  `{branch}` @ `{commit}`"
        + (" (worktree dirty at generation time)" if dirty else ""),
        f"- Runtime container: {CONTAINER}",
        "",
        "## Inventory",
        "",
        "| kind | what | source / reason |",
        "|---|---|---|",
    ]
    for item in INVENTORY:
        lines.append(f"| {item['kind']} | {item['what']} | {item['source']} |")
    lines += [
        "",
        "## Pinned harness file hashes",
        "",
        "| file (relative to checkout) | sha256 |",
        "|---|---|",
    ]
    for path, digest in sorted(manifest.items()):
        lines.append(f"| `{path}` | `{digest}` |")
    (PACKAGE_DIR / "PROVENANCE.md").write_text("\n".join(lines) + "\n")
    print(f"Pinned {len(manifest)} files at {branch}@{commit}")


if __name__ == "__main__":
    sys.exit(main())
