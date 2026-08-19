"""REGISTER of every KDA chunk-PREFILL implementation.

This file is not a kernel — it only registers builders that map the shared
:class:`~kda.inputs.PrefillInputs` contract onto each framework's entry point.
The math they all implement is the naive recurrence in ``kda_attention.py``
(chunked derivation in ../KDA.md). Benches consume these through
:data:`KDA_PREFILL` / :func:`get_kda_prefill_backends`.

Gate parameterizations differ per kernel family: canonical-gate rows
(FLA/SGLang/vLLM Triton) receive post-sigmoid ``beta``; FlashKDA-family rows
require the safe gate and raw beta logits (``fla_kda_safe_triton`` is the
matched Triton baseline). Rows prefixed ``b10_`` are kernels authored or
generated in this repo (see ``kda/b10/``).

Also here: SGLang's module-level ``extend`` prefill registry
(:data:`KDA_SGLANG_EXTEND`), used by the layer-level bench.
"""

from __future__ import annotations

from typing import Callable

import torch

from frameworks import _import_trtllm_fla, _import_trtllm_module, _import_vllm_fla
from kda.inputs import PrefillInputs, beta_logit_of, make_prefill_inputs  # noqa: F401
from kda.kda_attention import (
    SAFE_GATE_LOWER_BOUND,
    activate_kda_gate,
    kda_recurrent_reference,  # noqa: F401  (re-export for benches)
)
from linear_attention import BackendRegistry, Shape

KDA_PREFILL = BackendRegistry("fused-raw-gate KDA chunk prefill, matched inputs")

# Back-compat alias: older notes/specs refer to _SAFE_GATE_LOWER_BOUND.
_SAFE_GATE_LOWER_BOUND = SAFE_GATE_LOWER_BOUND


@KDA_PREFILL.register(
    "fla_kda_chunk",
    note="upstream FLA Triton chunk pipeline, canonical gate (the original)",
)
def _fla_kda_chunk(inputs: PrefillInputs, shape: Shape) -> Callable:
    from fla.ops.kda import chunk_kda

    def run():
        return chunk_kda(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g=inputs.raw_gate,
            beta=inputs.beta,
            scale=shape.key_dim**-0.5,
            initial_state=inputs.state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            state_v_first=True,
            cu_seqlens=inputs.cu_seqlens,
        )

    return run


@KDA_PREFILL.register(
    "fla_kda_safe_triton",
    note="same FLA Triton pipeline, safe gate, FLA_FLASH_KDA=0 - matched baseline for flash_kda",
)
def _fla_kda_safe_triton(inputs: PrefillInputs, shape: Shape) -> Callable:
    from fla.ops.kda import chunk_kda

    beta_logit = beta_logit_of(inputs)

    def run():
        # Force the Triton path even when FlashKDA is installed.
        import os

        prev = os.environ.get("FLA_FLASH_KDA")
        os.environ["FLA_FLASH_KDA"] = "0"
        try:
            with torch.inference_mode():
                return chunk_kda(
                    q=inputs.q,
                    k=inputs.k,
                    v=inputs.v,
                    g=inputs.raw_gate,
                    beta=beta_logit,
                    scale=shape.key_dim**-0.5,
                    initial_state=inputs.state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    safe_gate=True,
                    lower_bound=SAFE_GATE_LOWER_BOUND,
                    A_log=inputs.A_log,
                    dt_bias=inputs.dt_bias,
                    state_v_first=True,
                    cu_seqlens=inputs.cu_seqlens,
                )
        finally:
            if prev is None:
                os.environ.pop("FLA_FLASH_KDA", None)
            else:
                os.environ["FLA_FLASH_KDA"] = prev

    return run


def _flashkda_fwd_builder(inputs: PrefillInputs, shape: Shape, module) -> Callable:
    """Shared adapt for the ``flash_kda.fwd`` API (the INT21 PTX package
    intentionally exposes the identical interface)."""
    beta_logit = beta_logit_of(inputs)
    dt_bias_hk = inputs.dt_bias.view(shape.value_heads, shape.key_dim)

    def run():
        out = torch.empty_like(inputs.v)
        final_state = torch.empty_like(inputs.state)
        cu = inputs.cu_seqlens.to(torch.int64)
        with torch.inference_mode():
            module.fwd(
                inputs.q.contiguous(),
                inputs.k.contiguous(),
                inputs.v.contiguous(),
                inputs.raw_gate.contiguous(),
                beta_logit.contiguous(),
                shape.key_dim**-0.5,
                out,
                A_log=inputs.A_log.contiguous(),
                dt_bias=dt_bias_hk.contiguous(),
                lower_bound=SAFE_GATE_LOWER_BOUND,
                initial_state=inputs.state.contiguous(),
                final_state=final_state,
                cu_seqlens=cu,
            )
        return out, final_state

    return run


@KDA_PREFILL.register(
    "flash_kda",
    note="MoonshotAI CUTLASS kernels, safe gate only; FLA auto-dispatches here when installed",
)
def _flash_kda(inputs: PrefillInputs, shape: Shape) -> Callable:
    import flash_kda

    return _flashkda_fwd_builder(inputs, shape, flash_kda)


import os as _os

_KDA_B200_DIR = _os.environ.get("KDA_B200_DIR") or next(
    (d for d in (
        # local clone (see RUNBOOK: github.com/Int21-AI/KDA-B200,
        # pip install -e . --no-build-isolation in trt-dev)
        str(__import__("pathlib").Path(__file__).resolve()
            .parents[3] / "kda_b200_install"),
        "/workspace/model-performance/yikai/diffusion_inference/"
        "kda_b200_install",
    ) if __import__("pathlib").Path(d).is_dir()),
    "/workspace/model-performance/yikai/diffusion_inference/"
    "kda_b200_install",
)
_FLASHKDA_PTX_MAX_TOKENS = 262144  # int32 byte offsets overflow (IMA) past this


def _import_flash_kda_ptx():
    """Load INT21's flashkda-ptx under an alias: its package is *also* named
    ``flash_kda`` (interface-compatible by design), which would collide with
    MoonshotAI's pip-installed package in the same process. Only its C
    extension (``flash_kda_cuda``) has a distinct name, so the Python package
    is loaded from file under the ``flash_kda_ptx`` module name."""
    import importlib.util
    import sys

    if "flash_kda_ptx" in sys.modules:
        return sys.modules["flash_kda_ptx"]
    if _KDA_B200_DIR not in sys.path:
        sys.path.insert(0, _KDA_B200_DIR)
    spec = importlib.util.spec_from_file_location(
        "flash_kda_ptx", f"{_KDA_B200_DIR}/flash_kda/__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["flash_kda_ptx"] = module
    spec.loader.exec_module(module)
    return module


@KDA_PREFILL.register(
    "flashkda_ptx_int21",
    note="INT21 KDA-B200: plain CUDA/PTX rewrite of FlashKDA, ~1.5x CUTLASS; "
    "int32 offsets limit it to total tokens <= 262144",
)
def _flashkda_ptx_int21(inputs: PrefillInputs, shape: Shape) -> Callable:
    total_tokens = inputs.q.shape[1]
    if total_tokens > _FLASHKDA_PTX_MAX_TOKENS:
        raise RuntimeError(
            f"flashkda-ptx int32 offsets overflow past "
            f"{_FLASHKDA_PTX_MAX_TOKENS} total tokens (got {total_tokens})"
        )
    return _flashkda_fwd_builder(inputs, shape, _import_flash_kda_ptx())


@KDA_PREFILL.register(
    "fi_recurrent_kda",
    note="flashinfer PR #4262 CAKE SM100a recurrent-KDA prefill "
    "(safe gate; l2norm+gate fused in-kernel; bf16 state; needs "
    "cu_seqlens for multi-token)",
)
def _fi_recurrent_kda(inputs: PrefillInputs, shape: Shape) -> Callable:
    import torch as _t

    from flashinfer.kda_decode import recurrent_kda

    beta = inputs.beta.to(_t.bfloat16)  # contract: PRE-sigmoided bf16

    def run():
        return recurrent_kda(
            inputs.q, inputs.k, inputs.v, inputs.raw_gate, beta,
            A_log=inputs.A_log, dt_bias=inputs.dt_bias,
            scale=shape.key_dim ** -0.5,
            initial_state=None,  # zero-init; bf16 [N,HV,V,K] pool
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=SAFE_GATE_LOWER_BOUND,
            cu_seqlens=inputs.cu_seqlens,
        )

    return run


@KDA_PREFILL.register(
    "sglang_kda_chunk",
    note="SGLang-vendored FLA chunk pipeline (adds fused-intra small-grid path)",
)
def _sglang_kda_chunk(inputs: PrefillInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.fla.kda import chunk_kda

    def run():
        return chunk_kda(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g=inputs.raw_gate,
            beta=inputs.beta,
            scale=shape.key_dim**-0.5,
            initial_state=inputs.state,
            initial_state_indices=inputs.state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs.cu_seqlens,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
        )

    return run


@KDA_PREFILL.register(
    "vllm_kda_chunk",
    note="vLLM-vendored FLA chunk pipeline + fused canonical gate",
)
def _vllm_kda_chunk(inputs: PrefillInputs, shape: Shape) -> Callable:
    vllm_kda = _import_vllm_fla("kda")

    def run():
        return vllm_kda.chunk_kda_with_fused_gate(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            raw_g=inputs.raw_gate,
            beta=inputs.beta,
            A_log=inputs.A_log,
            g_bias=inputs.dt_bias,
            scale=shape.key_dim**-0.5,
            initial_state=inputs.state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs.cu_seqlens,
        )

    return run


@KDA_PREFILL.register(
    "trtllm_kda_chunk",
    note="TRT-LLM-vendored FLA chunk pipeline + fused safe gate/beta",
)
def _trtllm_kda_chunk(inputs: PrefillInputs, shape: Shape) -> Callable:
    trt_kda = _import_trtllm_fla("chunk_kda")
    beta_logit = beta_logit_of(inputs)

    def run():
        return trt_kda.chunk_kda_with_fused_gate(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            raw_g=inputs.raw_gate,
            raw_beta=beta_logit.float(),
            A_log=inputs.A_log,
            g_bias=inputs.dt_bias,
            scale=shape.key_dim**-0.5,
            initial_state=inputs.state,
            initial_state_indices=inputs.state_indices,
            inplace_indexed_state_update=True,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs.cu_seqlens,
        )

    return run


@KDA_PREFILL.register(
    "trtllm_kda_cute",
    note="TRT-LLM in-house CuTe DSL Blackwell prefill (ln_prefill custom op, "
    "safe gate); production wrapper incl. l2norm + state gather/scatter",
)
def _trtllm_kda_cute(inputs: PrefillInputs, shape: Shape) -> Callable:
    """The `trid/kda-cute-int21-snapshots` branch's `cute` prefill backend
    (modules/mamba/flash_kda.py cute_kda_with_fused_gate -> KdaPrefillRunner,
    kernels under _torch/cute_dsl_kernels/blackwell/kimi_k3_ln/). Timed via
    the production wrapper, which normalizes q/k with the Triton l2norm and
    gathers/scatters the indexed fp32 state around the CuTe pipeline — the
    per-call glue the serving path pays too. SM100 + head_dim=128 + bf16."""
    # Import the custom op first: flash_kda.py's guarded relative import needs
    # the `custom_ops` stub package registered before it runs.
    _import_trtllm_module("custom_ops.cute_dsl_kimi_k3_custom_ops")
    trt_flash_kda = _import_trtllm_module("modules.mamba.flash_kda")
    if not trt_flash_kda.is_cute_kda_supported(
        shape.key_dim, torch.bfloat16, SAFE_GATE_LOWER_BOUND
    ):
        raise RuntimeError(
            "CuTe KDA needs SM100, head_dim=128, bf16 (is_cute_kda_supported)"
        )
    beta_logit = beta_logit_of(inputs)

    def run():
        # Final state is scattered back into inputs.state in place. raw_beta
        # must stay bf16 (production projection dtype): the wrapper casts the
        # activated beta back to raw_beta.dtype, and the CuTe kernel bakes a
        # bf16 layout — an fp32 beta silently corrupts half the heads (NaN).
        return trt_flash_kda.cute_kda_with_fused_gate(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            raw_g=inputs.raw_gate,
            raw_beta=beta_logit,
            A_log=inputs.A_log,
            g_bias=inputs.dt_bias,
            initial_state=inputs.state,
            initial_state_indices=inputs.state_indices,
            cu_seqlens=inputs.cu_seqlens,
            lower_bound=SAFE_GATE_LOWER_BOUND,
        )

    return run


def _b10_flashkda_triton_builder(
    inputs: PrefillInputs, shape: Shape, chunk: int
) -> Callable:
    """FlashKDA's algorithm (Neumann inverse, prep+carry split) in Triton; see
    kda/b10/b10_kda_prefill_triton.py. The canonical gate activation is fused
    into the prep kernel (raw gate in), matching what the other backends fuse."""
    from kda.b10.b10_kda_prefill_triton import kda_chunk_prefill

    batch = inputs.state.shape[0]
    tokens = inputs.q.shape[1]
    seq = tokens // batch
    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    q = inputs.q.view(batch, seq, H, K)
    k = inputs.k.view(batch, seq, H, K)
    v = inputs.v.view(batch, seq, H, V)
    raw_gate = inputs.raw_gate.view(batch, seq, H, K)
    beta = inputs.beta.float().view(batch, seq, H).contiguous()
    s0 = torch.zeros(batch, H, K, V, device=q.device, dtype=torch.float32)

    def run():
        return kda_chunk_prefill(
            q, k, v, raw_gate, beta, s0,
            chunk=chunk, A_log=inputs.A_log, dt_bias=inputs.dt_bias,
        )

    run()  # fail at build time if unsupported (e.g. seq not a chunk multiple)
    return run


@KDA_PREFILL.register(
    "b10_flashkda_triton_c16",
    note="b10 Triton port of FlashKDA's 2-kernel algorithm, C=16",
)
def _b10_flashkda_triton_c16(inputs: PrefillInputs, shape: Shape) -> Callable:
    return _b10_flashkda_triton_builder(inputs, shape, chunk=16)


@KDA_PREFILL.register(
    "b10_flashkda_triton_c64",
    note="same b10 Triton port, C=64",
)
def _b10_flashkda_triton_c64(inputs: PrefillInputs, shape: Shape) -> Callable:
    return _b10_flashkda_triton_builder(inputs, shape, chunk=64)


@KDA_PREFILL.register(
    "b10_kda_chunk_prefill",
    note="b10 promoted CuTeDSLGen chunk-prefill champion (INT21-port "
    "schedule); consumes pre-activated log-decay (canonical gate, activated "
    "untimed in the builder)",
)
def _b10_kda_chunk_prefill(inputs: PrefillInputs, shape: Shape) -> Callable:
    """b10's promoted CuTeDSLGen chunked prefill (a CuTeDSL reimplementation
    of the INT21 flashkda-ptx schedule; kda/b10/b10_kda_chunk_prefill_cutedsl.py).
    Contract: q/k/v [B,T,H,128] bf16, g fp32 log-decay (already activated),
    beta fp32, S0 [B,H,K,V] fp32 -> (o, S_T). Unlike the framework rows it does
    not fuse the raw-gate activation; that elementwise cook runs untimed here."""
    from kda.b10.b10_kda_chunk_prefill_cutedsl import kda_chunk_prefill

    batch = inputs.state.shape[0]
    tokens = inputs.q.shape[1]
    seq = tokens // batch
    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    q = inputs.q.view(batch, seq, H, K)
    k = inputs.k.view(batch, seq, H, K)
    v = inputs.v.view(batch, seq, H, V)
    g = activate_kda_gate(inputs.raw_gate, inputs.A_log, inputs.dt_bias).view(
        batch, seq, H, K
    )
    beta = inputs.beta.float().view(batch, seq, H).contiguous()
    s0 = torch.zeros(batch, H, K, V, device=q.device, dtype=torch.float32)

    def run():
        return kda_chunk_prefill(q, k, v, g, beta, s0)

    return run


@KDA_PREFILL.register(
    "b10_kda_recurrent",
    note="b10 retired spec-decode CuTeDSL kernel as short-prefill probe, seq<=1024",
)
def _b10_kda_recurrent(inputs: PrefillInputs, shape: Shape) -> Callable:
    """b10's CuTeDSL sequential recurrence as a SHORT-PREFILL backend
    (kda/b10/b10_kda_prefill_cutedsl.py).

    One state-resident kernel, no chunking/WY machinery: the state stays in
    registers through all T steps and is written once. Useless for MTP verify
    (commits in place, no rollback records); registered here to test the
    hypothesis that it beats chunked kernels at very short prefill, where
    their fixed multi-kernel/workspace overheads dominate. MEASURED VERDICT
    (results/bench_kda_short_prefill.csv, B200): it does NOT — it only ties
    flash_kda at S=32 (everything is launch-bound there, ~23-28 us GPU-only)
    and by S=64 the sequential T-loop (~0.7 us/token per chain at B=1) already
    loses 1.7x to chunk parallelism. The recurrent form only wins at T <= ~16,
    i.e. the decode/verify regime. Kept as a registered, seq<=1024-gated
    backend so the result stays reproducible."""
    cu = inputs.cu_seqlens
    seq_len = int(cu[1] - cu[0])
    if seq_len > 1024:
        raise RuntimeError("short-prefill backend: seq_len <= 1024 only")

    from kda.b10.b10_kda_prefill_cutedsl import kda_spec_decode

    batch = inputs.state.shape[0]
    total = inputs.q.shape[1]
    q = inputs.q.view(total, shape.qk_heads, shape.key_dim).contiguous()
    k = inputs.k.view_as(q).contiguous()
    v = inputs.v.view(total, shape.value_heads, shape.value_dim).contiguous()
    g = (
        activate_kda_gate(inputs.raw_gate, inputs.A_log, inputs.dt_bias)
        .view(total, shape.value_heads, shape.key_dim)
        .contiguous()
    )
    beta = inputs.beta.float().view(total, shape.value_heads).contiguous()
    cu32 = cu.to(torch.int32)
    scratch = torch.empty_like(inputs.state)

    def run():
        # prefill starts from S0 = 0; the kernel updates its state in place
        scratch.zero_()
        out = kda_spec_decode(q, k, v, g, beta, scratch, cu32)
        return out, scratch

    run()  # fail at build time if the kernel cannot serve this shape
    return run


@KDA_PREFILL.register(
    "flashinfer_cake_kda",
    note="FlashInfer CAKE-generated frozen SM100a BF16 recurrent KDA prefill "
    "(PR #4262 merged 2026-08-03, beta-TMA H=12 fix from #4351 merged "
    "2026-08-05, installed at flashinfer@38bf507); "
    "state is BF16 [B,H,K,K] — unavoidable semantic difference vs FP32 state "
    "in other backends; gate/l2norm/beta fused in kernel (safe gate, lower_bound=-5)",
)
def _flashinfer_cake_kda(inputs: PrefillInputs, shape: Shape) -> Callable:
    """FlashInfer CAKE recurrent KDA prefill (PR #4262, beta fix #4351).

    Dispatch contract (from kda_prefill.py eligibility check):
    - SM100a B200 / SM103a GB300 only
    - q/k/v/g BF16 [B, T, H, 128], beta BF16 [B, T, H]
    - A_log FP32 [H], dt_bias FP32 [H*K]
    - use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
      beta_is_logit=True, lower_bound finite & negative

    Semantic difference: CAKE state is BF16 [B, H, K, V] (in-place update
    inside kernel). Other backends use FP32 state. For correctness comparison
    the BF16 state output has higher quantisation error vs the FP32 reference;
    we use relaxed atol/rtol in the bench correctness pass.

    The builder reshapes packed [1, B*T, H, D] -> fixed-layout [B, T, H, D]
    so cu_seqlens is None and the kernel takes the fast fixed-layout path.
    Initial state is BF16 zeros (no FP32 state pool conversion).
    """
    from flashinfer.kda_prefill import (
        _flash_kda_prefill_is_eligible,
        _run_flash_kda_prefill,
    )

    batch = inputs.state.shape[0]
    total_tokens = inputs.q.shape[1]
    seq_len = total_tokens // batch
    H = shape.value_heads
    K = shape.key_dim
    V = shape.value_dim

    # Reshape from packed [1, B*T, H, D] to fixed [B, T, H, D]
    q = inputs.q.view(batch, seq_len, H, K).contiguous()
    k = inputs.k.view(batch, seq_len, H, K).contiguous()
    v = inputs.v.view(batch, seq_len, H, V).contiguous()
    g = inputs.raw_gate.view(batch, seq_len, H, K).contiguous()
    # beta_logit: logit of the pre-sigmoid beta the kernel will re-sigmoid
    beta_logit = beta_logit_of(inputs).view(batch, seq_len, H).to(torch.bfloat16).contiguous()
    # dt_bias as [H, K] FP32
    dt_bias_hk = inputs.dt_bias.view(H, K).contiguous()

    # BF16 initial state [B, H, V, K] — canonical KDA state layout
    # NOTE: BF16 precision is the unavoidable semantic difference vs FP32
    state_bf16 = torch.zeros(batch, H, V, K, device=q.device, dtype=torch.bfloat16)

    # Verify eligibility at build time (raises RuntimeError if dispatch fails)
    eligible = _flash_kda_prefill_is_eligible(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta_logit,
        A_log=inputs.A_log,
        dt_bias=dt_bias_hk,
        initial_state=state_bf16,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=SAFE_GATE_LOWER_BOUND,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=None,
        initial_state_source=None,
        initial_state_indices=None,
        beta_is_logit=True,
    )
    if not eligible:
        raise RuntimeError(
            "flashinfer_cake_kda: _flash_kda_prefill_is_eligible returned False — "
            "check device SM, dtypes, head_dim=128, and gate flags"
        )

    # Separate state buffer per call (kernel updates in-place; we reset to 0
    # at each timed invocation so the benchmark measures steady-state cost).
    state_buf = state_bf16.clone()

    def run():
        state_buf.zero_()
        out, final_state = _run_flash_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta_logit,
            A_log=inputs.A_log,
            dt_bias=dt_bias_hk,
            scale=None,          # defaults to 1/sqrt(128)
            initial_state=state_buf,
            output_final_state=True,
            lower_bound=float(SAFE_GATE_LOWER_BOUND),
            cu_seqlens=None,
            output=None,
            seq_order=None,
            prefill_workspace=None,
        )
        return out, final_state

    # Trigger JIT compilation eagerly at build time (first call compiles
    # the frozen CUDA kernel; subsequent calls run the cached module).
    run()
    return run


def get_kda_prefill_backends(
    inputs: PrefillInputs,
    shape: Shape,
    only: list[str] | None = None,
) -> tuple[dict[str, Callable], dict[str, str]]:
    return KDA_PREFILL.build(inputs, shape, only=only)


# ---------------------------------------------------------------------------
# SGLang module-level ``extend`` prefill (kernel objects behind SGLang's
# dispatch contract; used by bench_linear_attention.py)
# ---------------------------------------------------------------------------

KDA_SGLANG_EXTEND = BackendRegistry("SGLang KDA kernel-object extend prefill")


def _sglang_extend_builder(
    inputs: PrefillInputs,
    lower_bound: float | None,
    module_name: str,
    class_name: str,
) -> Callable:
    seq_lens = (inputs.cu_seqlens[1:] - inputs.cu_seqlens[:-1]).cpu().tolist()
    module = __import__(module_name, fromlist=[class_name])
    kernel = getattr(module, class_name)()

    def run():
        return kernel.extend(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.raw_gate,
            inputs.beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            lower_bound=lower_bound,
            ssm_states=inputs.state,
            cache_indices=inputs.state_indices,
            query_start_loc=inputs.cu_seqlens,
            extend_seq_lens_cpu=seq_lens,
        )

    return run


@KDA_SGLANG_EXTEND.register(
    "sglang_triton",
    note="SGLang TritonKDAKernel extend path (chunk_kda underneath)",
)
def _sglang_extend_triton(inputs: PrefillInputs, lower_bound) -> Callable:
    return _sglang_extend_builder(
        inputs,
        lower_bound,
        "sglang.srt.layers.attention.linear.kernels.kda_triton",
        "TritonKDAKernel",
    )


@KDA_SGLANG_EXTEND.register(
    "sglang_cutedsl",
    note="SGLang CuTeDSL kernel object extend path",
)
def _sglang_extend_cutedsl(inputs: PrefillInputs, lower_bound) -> Callable:
    return _sglang_extend_builder(
        inputs,
        lower_bound,
        "sglang.srt.layers.attention.linear.kernels.kda_cutedsl",
        "CuteDSLKDAKernel",
    )


@KDA_SGLANG_EXTEND.register(
    "sglang_flashkda",
    note="SGLang kernel object wrapping the SAME MoonshotAI flash_kda.fwd - not a separate impl",
)
def _sglang_extend_flashkda(inputs: PrefillInputs, lower_bound) -> Callable:
    seq_lens = (inputs.cu_seqlens[1:] - inputs.cu_seqlens[:-1]).cpu().tolist()
    if lower_bound is None or min(seq_lens) < 64 or max(seq_lens) > 2048:
        raise RuntimeError(
            "actual FlashKDA requires safe gate and 64 <= sequence length <= 2048; "
            "outside that range SGLang intentionally falls back to Triton"
        )
    return _sglang_extend_builder(
        inputs,
        lower_bound,
        "sglang.srt.layers.attention.linear.kernels.kda_flashkda",
        "FlashKDAKernel",
    )


def get_kda_sglang_extend_backends(
    inputs: PrefillInputs,
    *,
    lower_bound: float | None,
) -> tuple[dict[str, Callable], dict[str, str]]:
    return KDA_SGLANG_EXTEND.build(inputs, lower_bound)
