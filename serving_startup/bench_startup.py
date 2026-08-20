#!/usr/bin/env python3
"""Phase-resolved TRT-LLM startup benchmark (weight load vs warm-up).

Instruments the real serving bring-up path by wrapping the functions that
own each phase, so the numbers come from production code, not a re-implementation:

  import            `import tensorrt_llm`
  config            HF config load + validation
  model_init        ModelLoader.load: meta construct -> empty CUDA params
                    -> checkpoint read -> H2D copies -> post_load hooks
  kv_estimate       KV-cache capacity profiling pass (creates a throwaway
                    PyExecutor, which warms up + captures graphs, then tears
                    it down)
  warmup            ModelEngine.warmup, split into attention / general
                    (torch.compile) / autotuner / cuda-graph / max-shape
  ready             everything else until LLM() returns

Usage
  python3 bench_startup.py --model <dir> [--tp 1] [--dummy-weights]
                           [--no-cuda-graph] [--no-torch-compile]
                           [--kv-tokens N] [--label NAME]
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

T0 = time.time()
MARKS: list[tuple[str, float, float]] = []


def record(name, start, end):
    MARKS.append((name, start - T0, end - start))


def wrap(obj, attr, name):
    """Time every call to obj.attr, recording it as `name`."""
    original = getattr(obj, attr)

    def timed(*a, **k):
        s = time.time()
        try:
            return original(*a, **k)
        finally:
            record(name, s, time.time())

    setattr(obj, attr, timed)
    return original


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dummy-weights", action="store_true",
                   help="load_format=dummy: isolates warm-up from weight I/O")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--no-torch-compile", action="store_true")
    p.add_argument("--kv-tokens", type=int, default=0,
                   help="explicit KV token budget; skips the estimation pass")
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--label", default="baseline")
    p.add_argument("--out", default="local_debug/startup/startup_phases.jsonl")
    a = p.parse_args()

    t = time.time()
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
    record("import", t, time.time())

    # ---- instrument the phases -------------------------------------------
    from tensorrt_llm._torch.pyexecutor import model_loader, model_engine, _util
    wrap(model_loader.ModelLoader, "load", "model_init")
    ME = model_engine.PyTorchModelEngine
    wrap(ME, "warmup", "warmup_total")
    for attr, label in (("_run_attention_warmup", "warmup.attention"),
                        ("_general_warmup", "warmup.general"),
                        ("_run_autotuner_warmup", "warmup.autotuner"),
                        ("_run_cuda_graph_warmup", "warmup.cuda_graph")):
        if hasattr(ME, attr):
            wrap(ME, attr, label)
    for cls_name in ("KvCacheCreator", ):
        cls = getattr(_util, cls_name, None)
        if cls is not None:
            for attr in ("estimate_max_kv_cache_tokens", "try_prepare_estimation",
                         "configure_kv_cache_capacity"):
                if hasattr(cls, attr):
                    wrap(cls, attr, f"kv.{attr}")

    kv = KvCacheConfig(free_gpu_memory_fraction=0.6)
    if a.kv_tokens:
        kv = KvCacheConfig(max_tokens=a.kv_tokens)

    kwargs = dict(
        model=a.model,
        tensor_parallel_size=a.tp,
        max_batch_size=a.max_batch_size,
        max_seq_len=a.max_seq_len,
        kv_cache_config=kv,
    )
    if a.dummy_weights:
        kwargs["load_format"] = "dummy"
    if a.no_cuda_graph:
        kwargs["cuda_graph_config"] = None
    else:
        kwargs["cuda_graph_config"] = CudaGraphConfig()
    if a.no_torch_compile:
        kwargs["torch_compile_config"] = None

    t = time.time()
    llm = LLM(**kwargs)
    record("LLM()_total", t, time.time())

    t = time.time()
    out = llm.generate(["Hello, my name is"])
    record("first_generate", t, time.time())
    total = time.time() - T0

    print(f"\n=== startup phases [{a.label}] ===")
    print(f"{'phase':<24}{'start_s':>9}{'dur_s':>9}")
    for name, start, dur in MARKS:
        print(f"{name:<24}{start:>9.1f}{dur:>9.1f}")
    print(f"{'TOTAL to first token':<24}{'':>9}{total:>9.1f}")
    print(f"sample: {out[0].outputs[0].text[:40]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps({
            "label": a.label, "model": a.model, "tp": a.tp,
            "dummy_weights": a.dummy_weights,
            "no_cuda_graph": a.no_cuda_graph,
            "no_torch_compile": a.no_torch_compile,
            "kv_tokens": a.kv_tokens, "total_s": total,
            "phases": [{"name": n, "start_s": s, "dur_s": d} for n, s, d in MARKS],
        }) + "\n")
    llm.shutdown()


if __name__ == "__main__":
    main()
