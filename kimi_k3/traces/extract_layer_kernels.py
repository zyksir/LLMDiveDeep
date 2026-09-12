#!/usr/bin/env python3
"""Extract ordered per-layer kernel sequences from a K3 MTP1 decode trace.

Works on both baselines in this directory (TRT-LLM rank0 kineto trace and
the sglang TP-0 trace) and on the bench mini traces:

* Baselines: kernels of ONE decode step are sliced between consecutive MoE
  routing anchors (``routingIndices*`` — one per layer, 93/step); each slice
  is one decoder layer (attention + MoE + tails). Slices are classified KDA
  (contain the fused KDA MTP verify kernel) or MLA (contain the MLA decode
  kernels). The modal sequence across all same-class layers of the middle
  step is printed, with per-position kernel medians.
* Mini traces (``--mini``): GPU kernels are attributed to the bench
  ``user_annotation`` spans (TrtKimiK3Kda / SglKimiK3Kda / TrtKimiK3MoE /
  SglKimiK3MoE / TrtRc25KimiK3MoE) via cuda_runtime launch correlation; the
  modal per-span sequence of the LAST repetitions (steady state) is printed.

Kernel names are normalized (template args stripped) for sequence
comparison; the full names are kept in the printout.
"""

import argparse
import collections
import gzip
import json
import re
import statistics


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    return data["traceEvents"] if isinstance(data, dict) else data


def normalize(name: str) -> str:
    """Collapse a kernel name to a stable identity: templates/params dropped,
    cutlass/cute suffixes with shape hashes trimmed."""
    n = name
    if n.startswith("void "):
        n = n[5:]
    n = n.split("<", 1)[0].split("(", 1)[0]
    # CuTe DSL kernels embed layout/pool shapes after the symbolic stem;
    # keep the stem up to the op name.
    for stem in (
        "kernel_cutlass__kda_replay_ssm_conv_gated_wychunk_kernel",
        "kernel_cutlass_kda_decode_mtp_kernel",
        "kernel_cutlass_kernel_TgvGemmCuteExtKernel",
        "kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel",
        "kernel_cutlass_split_kv_kernel_flashinfercute_dslattentionmonolithicmla",
        "kernel_cutlass_reduction_kernel_flashinfercute_dslattentionmonolithicmla",
    ):
        if n.startswith(stem):
            return stem
    # bmm_* trtllm-gen cubin names encode the tactic; keep the full name
    # (tactic identity is part of the contract).
    return n


def kernels_with_correlation(events):
    ker, launches = [], {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat == "kernel":
            ker.append(ev)
        elif cat in ("cuda_runtime", "cuda_driver") and (
            "LaunchKernel" in ev.get("name", "")
            or "GraphLaunch" in ev.get("name", "")
        ):
            corr = ev.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = ev
    ker.sort(key=lambda e: e["ts"])
    return ker, launches


ANCHOR = "routingIndices"  # per-layer MoE routing kernel (both stacks)
# KDA verify kernel: sglang traces name it kda_decode_mtp_kernel; the TRT
# fork's vendored copy compiles as _kda_replay_ssm_conv_gated_wychunk_kernel.
KDA_MARKS = ("kda_decode_mtp_kernel", "kda_replay_ssm")
MLA_MARKS = ("mla_decode", "MultiHeadLatentAttention", "fmhaSm100fKernel")


def slice_baseline_layers(path):
    events = load_events(path)
    ker, _ = kernels_with_correlation(events)
    anchors = [i for i, e in enumerate(ker) if ANCHOR in e["name"]]
    if not anchors:
        raise SystemExit(f"no {ANCHOR} anchors in {path}")
    n_layers = 93
    n_steps = len(anchors) // n_layers
    if len(anchors) % n_layers:
        raise SystemExit(
            f"{len(anchors)} anchors not a multiple of {n_layers} layers"
        )
    # middle step, full 93-layer window; slice k covers anchor k..k+1
    step = n_steps // 2
    step_anchor_idx = anchors[step * n_layers:(step + 1) * n_layers]
    layers = []
    for j, ai in enumerate(step_anchor_idx):
        # layer slice = kernels from just after the previous anchor's slice
        # start ... we delimit [this_anchor, next_anchor) then shift so the
        # attention front (before routing) lands with its own MoE: a layer
        # in launch order is [attn ... route ... moe tail], so the natural
        # unit is (prev_anchor, this_anchor] attention + (this..next] moe.
        # Simplest faithful unit: kernels in [anchor_j, anchor_{j+1}).
        lo = ai
        hi = step_anchor_idx[j + 1] if j + 1 < len(step_anchor_idx) else None
        if hi is None:
            # close the last layer at the next step's first anchor if any
            k = (step + 1) * n_layers
            hi = anchors[k] if k < len(anchors) else len(ker)
        layers.append(ker[lo:hi])
    return layers


def rotate_layers_to_attention_start(layers):
    """Rebase slices so each starts at the attention front instead of the
    routing anchor: concatenate and re-split before each AttnRes that is
    FOLLOWED (before the next AttnRes) by an attention kernel — both stacks
    launch a second AttnRes right before the MoE, which stays inside the
    layer."""
    flat = [e for sl in layers for e in sl]
    attn_idx = [i for i, e in enumerate(flat) if "attn_res_f" in e["name"]]
    if len(attn_idx) < 2:
        return layers
    starts = []
    for k, i in enumerate(attn_idx):
        j = attn_idx[k + 1] if k + 1 < len(attn_idx) else len(flat)
        window = " ".join(e["name"] for e in flat[i:j])
        if any(m in window for m in KDA_MARKS) or any(
            m in window for m in MLA_MARKS
        ):
            starts.append(i)
    if len(starts) < 2:
        return layers
    return [flat[a:b] for a, b in zip(starts, starts[1:])]


def classify(layer):
    names = " ".join(e["name"] for e in layer)
    if any(m in names for m in KDA_MARKS):
        return "kda"
    if any(m in names for m in MLA_MARKS):
        return "mla"
    return "other"


def modal_sequence(layer_group):
    seqs = collections.Counter(
        tuple(normalize(e["name"]) for e in layer)
        for layer in layer_group
    )
    (modal, hits), = seqs.most_common(1)
    # per-position duration medians over layers matching the modal sequence
    durs = [
        [e.get("dur", 0.0) for e in layer]
        for layer in layer_group
        if tuple(normalize(e["name"]) for e in layer) == modal
    ]
    med = [statistics.median(col) for col in zip(*durs)]
    # one full-name example
    example = next(
        layer for layer in layer_group
        if tuple(normalize(e["name"]) for e in layer) == modal
    )
    return modal, hits, len(seqs), med, example


def print_group(tag, group):
    modal, hits, variants, med, example = modal_sequence(group)
    print(f"\n### {tag}: {len(group)} layers, modal sequence x{hits} "
          f"({variants} variant(s)), {len(modal)} kernels")
    for i, (norm, dur, ev) in enumerate(zip(modal, med, example), 1):
        print(f"{i:3d}. {dur:8.2f} us  {norm}")
        full = ev["name"]
        if full != norm and not full.startswith(norm):
            print(f"          full: {full[:160]}")
    if variants > 1:
        seqs = collections.Counter(
            tuple(normalize(e["name"]) for e in layer) for layer in group
        )
        for seq, c in seqs.most_common():
            if seq == modal:
                continue
            print(f"  -- variant x{c}: {len(seq)} kernels; diff vs modal:")
            import difflib
            sm = difflib.SequenceMatcher(a=modal, b=seq)
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if op == "equal":
                    continue
                print(f"     {op}: modal[{a0}:{a1}]={list(modal[a0:a1])} "
                      f"-> variant[{b0}:{b1}]={list(seq[b0:b1])}")


def run_baseline(path):
    layers = slice_baseline_layers(path)
    layers = rotate_layers_to_attention_start(layers)
    by = collections.defaultdict(list)
    for sl in layers:
        by[classify(sl)].append(sl)
    print(f"== {path}: {len(layers)} attention-rebased layer slices "
          f"({ {k: len(v) for k, v in by.items()} })")
    for tag in ("kda", "mla", "other"):
        if by.get(tag):
            print_group(tag, by[tag])


def run_mini(path):
    events = load_events(path)
    ker, launches = kernels_with_correlation(events)
    spans = [
        e for e in events
        if e.get("cat") == "user_annotation"
        and re.match(r"(Trt|Sgl).*Kimi|TrtRc25", e.get("name", ""))
    ]
    spans.sort(key=lambda e: e["ts"])
    # map kernels to spans via launch correlation (launch inside span window
    # on the span's tid)
    # A cudaGraphLaunch replay maps ONE correlation id to MANY kernels, so
    # collect all kernels per correlation (GPU-time ordered), not just the
    # first.
    corr_to_kernels = collections.defaultdict(list)
    for e in ker:
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            corr_to_kernels[corr].append(e)
    by_span = collections.defaultdict(list)
    for corr, le in launches.items():
        kes = corr_to_kernels.get(corr)
        if not kes:
            continue
        for sp in spans:
            if (le.get("tid") == sp.get("tid")
                    and sp["ts"] <= le["ts"] < sp["ts"] + sp.get("dur", 0)):
                for ke in kes:
                    by_span[id(sp)].append((le["ts"], ke["ts"], ke))
                break
    groups = collections.defaultdict(list)
    for sp in spans:
        ks = [k for _, _, k in sorted(by_span.get(id(sp), []),
                                      key=lambda t: (t[0], t[1]))]
        if ks:
            groups[sp["name"]].append(ks)
    print(f"== {path}: spans {{name: reps}} = "
          f"{ {k: len(v) for k, v in groups.items()} }")
    for name, reps in groups.items():
        # steady state: drop the first half (JIT/tuning warmup)
        tail = reps[len(reps) // 2:]
        print_group(f"{name} (last {len(tail)} reps)", tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--mini", action="store_true",
                    help="bench mini trace (span-attributed) instead of a "
                         "server baseline")
    args = ap.parse_args()
    if args.mini:
        run_mini(args.trace)
    else:
        run_baseline(args.trace)


if __name__ == "__main__":
    main()
