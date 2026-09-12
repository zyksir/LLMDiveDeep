# DeepSeek V4.1-Flash: sparse attention from idea to code

Start with [research.md](research.md). It develops CSA2 step by step, with
equations, the released layer schedule, a numerical example, cache accounting,
and a prefill/decode walkthrough. This follows the math → implementation →
verification structure of the repository's linear attention and KDA studies.

The implementation is a small PyTorch reference for **projected-tensor CSA2
operators and cross-layer routing**. It runs on CPU with synthetic inputs.
It does not load the 552B backbone or implement an inference server.

**SGLang alignment has a version boundary.** The inspected SGLang commit
implements V4's `0/4/128` compression paths, not V4.1's `0/2/1` CSA2/CED
architecture. The indexer arithmetic and sparse core conventions are aligned;
native V4.1 serving and GPU-kernel parity are not claimed. Read
[sglang_alignment.md](sglang_alignment.md) for exact symbols, supported
contracts, missing integration work, and the source-parity command.

## Run from the repository root

Use an existing PyTorch environment, or create one:

```bash
python3 -m venv deepseek/.venv
deepseek/.venv/bin/python -m pip install -r deepseek/requirements.txt
deepseek/.venv/bin/python -m deepseek.demo --schedule
deepseek/.venv/bin/python -m unittest discover -s deepseek/tests -v
```

For an existing environment:

```bash
python3 -m deepseek.demo --seq-len 32 --query 17
python3 -m unittest discover -s deepseek/tests -v
```

The demo uses small dimensions and selection budgets to make every selected
position visible. It keeps the released 40-layer **ownership schedule**. Each
layer receives synthetic projected tensors; outputs are not propagated through
a trained Transformer. Printed output norms demonstrate execution, not quality.
The final memory/work table uses released dimensions and is analytical, not a
latency benchmark. Do not increase toy sequence length to one million: its Full
indexer deliberately materializes prefill scores for readability.

## Files and reading order

| File | Purpose |
|---|---|
| [research.md](research.md) | Full step-by-step explanation |
| [config.py](config.py) | Layer ownership, exact cache bytes, indexer work counts |
| [csa2_reference.py](csa2_reference.py) | Compression, selection, gather-first reindexing, sparse softmax, shared state |
| [demo.py](demo.py) | Trace all 40 layers on small projected tensors |
| [tests/test_csa2.py](tests/test_csa2.py) | Causality, streaming, numerical oracles, state ownership |
| [sglang_alignment.md](sglang_alignment.md) | Production source map and integration boundaries |
| [check_source_parity.py](check_source_parity.py) | Run two pinned upstream CPU helpers against the reference |
| [sglang_adapter.py](sglang_adapter.py) | Optional direct FlashMLA core call; CPU-tested packing, GPU execution unverified |
| [sources.json](sources.json) | Primary-source revision pins and content hashes |

Existing related material: [linear-attention survey](../attention/linear_attn/SURVEY.md),
[KDA implementation notes](../attention/linear_attn/KDA.md), and
[older V4 layer reference](../attention/sparse_attn/dsv4_layer.py).

## Verified snapshot

On 2026-09-10, using CPU PyTorch 2.14.0+cpu in an isolated temporary environment:

- Ten correctness tests passed (nine CSA2 checks and one SGLang-layout check).
- Pinned DeepSeek candidate-block helper parity and SGLang paged-FP8 indexer
  helper parity passed; see the exact coverage in `sglang_alignment.md`.
- Toy traces exercised short prefixes and the full released ownership schedule.

No GPU throughput, full-model accuracy, quantized CSA2 output parity, or native
SGLang V4.1 server run is represented by these results.
