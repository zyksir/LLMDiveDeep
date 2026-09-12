"""Compare CPU operators against two inspected upstream helper functions.

Supply the pinned files in sources.json. Hash checks precede extraction;
only the named function AST is compiled, never the model's import graph.
This is helper parity, NOT a SGLang CUDA/backend or full-model test.
"""

import argparse
import ast
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .csa2_reference import index_scores, select_candidates


def load_helper(path, name, expected_hash):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError(
            f"{path}: source hash differs from the inspected pin in sources.json"
        )
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"expected exactly one {name} function")
    namespace = {
        "torch": torch,
        "F": F,
        "Any": Any,
        "FP8_DTYPE": torch.float8_e4m3fn,
        "_arange_cache": {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def check_candidates(path):
    upstream = load_helper(
        path,
        "select_candidate_blocks",
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    )
    for n in (1, 7, 19, 65):
        lengths = torch.arange(n + 1)[:, None]
        scores = torch.randn(2, n + 1, n)
        visible = torch.arange(n)[None, :] < lengths
        scores = scores.masked_fill(~visible, -torch.inf)
        expected = upstream(scores, lengths, 2, 4) & visible
        ids = select_candidates(scores, lengths, 2, 4)
        actual = torch.zeros_like(expected)
        for b in range(2):
            for t in range(n + 1):
                valid = ids[b, t]
                actual[b, t, valid[valid >= 0].long()] = True
        torch.testing.assert_close(actual, expected)
    print(
        "PASS: DeepSeek select_candidate_blocks, clipped to causal positions (4 widths, B=2)"
    )


def check_sglang_indexer(path):
    upstream = load_helper(
        path,
        "fp8_paged_mqa_logits_torch",
        "fb7d974991001833a435df2d6f571bb749908bf801afab53d4e0913195cf54bd",
    )
    # Exact representable inputs isolate score ordering, head reduction,
    # page indirection and scale application from quantization roundoff.
    pages, page_size, d, h = 4, 64, 128, 2
    raw_keys = torch.randint(-1, 2, (pages, page_size, d)).to(torch.float8_e4m3fn)
    scales = torch.full((pages, page_size), 0.5)
    payload = torch.cat(
        (raw_keys.view(torch.uint8).flatten(1), scales.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache = payload.view(torch.float8_e4m3fn).reshape(pages, page_size, 1, d + 4)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([65, 97], dtype=torch.int32)
    q = torch.randint(-1, 2, (2, 1, h, d)).to(torch.float8_e4m3fn)
    weights = torch.tensor([[0.25, -0.5], [-0.25, 0.5]])
    actual = upstream(q, cache, weights, lengths, table, None, 128, clean_logits=False)
    keys = raw_keys.float()[table.long()].reshape(2, 128, d) * 0.5
    expected = index_scores(q, keys, weights[:, None])[:, 0]
    # This SGLang fallback zero-fills invalid slots. Serving top-k separately
    # applies causal validity; zero-filled padding must not be selected.
    expected = expected.masked_fill(torch.arange(128) >= lengths[:, None], 0.0)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    print(
        "PASS: SGLang fp8_paged_mqa_logits_torch (permuted pages, signed weights, exact inputs)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepseek-model", required=True)
    parser.add_argument("--sglang-indexer", required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.set_num_threads(1)
    check_candidates(args.deepseek_model)
    check_sglang_indexer(args.sglang_indexer)


if __name__ == "__main__":
    main()
