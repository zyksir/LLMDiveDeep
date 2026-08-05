"""REGISTER of every KDA one-token DECODE implementation.

This file is not a kernel — it only registers builders that map the shared
:class:`~kda.inputs.DecodeInputs` contract onto each framework's entry point.
The math they all implement is the naive recurrence in ``kda_attention.py``.
Benches consume these through :data:`KDA_DECODE` / :func:`get_kda_decode_backends`.

Rows prefixed ``b10_`` are kernels authored/generated in this repo (see
``kda/b10/``); everything else is the framework's own kernel called through
its public entry point.
"""

from __future__ import annotations

from typing import Callable

import torch

from frameworks import (
    _import_sglang_kernels,
    _import_trtllm_fla,
    _import_trtllm_module,
)
from kda.inputs import (  # noqa: F401
    DecodeInputs,
    make_conv_norm_inputs,
    make_decode_inputs,
    split_qkv,
)
from kda.kda_attention import (
    SAFE_GATE_LOWER_BOUND,
    activate_kda_gate,
    conv4_silu_reference,
    gated_rmsnorm_reference,
    kda_recurrent_reference,  # noqa: F401  (re-export for benches)
)
from linear_attention import BackendRegistry, Shape

KDA_DECODE = BackendRegistry("one-token KDA recurrence, matched inputs")


@KDA_DECODE.register(
    "torch_kda_reference",
    note="plain PyTorch full decode step (conv4+SiLU + gate activation + "
    "recurrence + gated RMSNorm, all from kda_attention.py); the correctness "
    "baseline every kernel row is checked against",
)
def _torch_kda_reference(inputs: DecodeInputs, shape: Shape) -> Callable:
    """The whole KDA.md §1.1 decode step in eager PyTorch, composed from the
    oracle functions in ``kda_attention.py``. Reading this builder top to
    bottom IS the pipeline every fused kernel implements: conv4+SiLU on the
    RAW packed q/k/v -> split -> gate/beta activation -> recurrence -> gated
    RMSNorm. Registered so the correctness table has an explicit zero-error
    baseline; too slow (dozens of eager launches) to be a useful latency row."""
    B = inputs.beta_logit.shape[0]
    Hq, H, K, V = shape.qk_heads, shape.value_heads, shape.key_dim, shape.value_dim
    cn = make_conv_norm_inputs(B, shape, device=str(inputs.mixed_qkv.device))
    mixed = inputs.mixed_qkv  # RAW pre-conv packed [q | k | v]
    state_k_first = inputs.state.transpose(-1, -2)  # shared pool is v-first

    def run():
        conv_out = conv4_silu_reference(mixed, cn.conv_weight, cn.conv_state)
        qc, kc, vc = torch.split(conv_out, [Hq * K, Hq * K, H * V], dim=-1)
        g = activate_kda_gate(
            inputs.raw_gate.view(B, 1, H, K), inputs.A_log, inputs.dt_bias
        )
        beta = torch.sigmoid(inputs.beta_logit.float()).view(B, 1, H)
        o, _ = kda_recurrent_reference(
            qc.view(B, 1, Hq, K),
            kc.view(B, 1, Hq, K),
            vc.view(B, 1, H, V),
            g,
            beta,
            initial_state=state_k_first,
        )
        return gated_rmsnorm_reference(o[:, 0], cn.z, cn.norm_weight)

    return run


@KDA_DECODE.register(
    "b10_kda_decode",
    note="b10 promoted CuTeDSLGen champion (kda/b10/b10_kda_decode_cutedsl.py)",
)
def _b10_kda_decode(inputs: DecodeInputs, shape: Shape) -> Callable:
    """b10's promoted CuTeDSL fused decode kernel as a black-box backend: map
    the packed DecodeInputs to the kernel's ``(q,k,v,g,beta,S)`` contract,
    then call the importable ``kda_decode_step`` API. Nothing kernel-specific
    leaks out; swapping in a new champion is just replacing that import."""
    from kda.b10.b10_kda_decode_cutedsl import kda_decode_step

    B = inputs.beta_logit.shape[0]
    H, Kd, Vd = shape.value_heads, shape.key_dim, shape.value_dim
    q, k, v = (t.reshape(B, H, -1).contiguous() for t in split_qkv(inputs, shape))
    g = activate_kda_gate(
        inputs.raw_gate.view(B, 1, H, Kd), inputs.A_log, inputs.dt_bias
    ).view(B, H, Kd)
    beta = torch.sigmoid(inputs.beta_logit.float()).view(B, H)
    # K-first copy of the (v-first) shared state pool, so correctness runs
    # exercise the same nonzero S0 as every other row.
    S = inputs.state.float().transpose(-1, -2).contiguous()

    def run():
        return kda_decode_step(q, k, v, g, beta, S)

    return run


@KDA_DECODE.register(
    "b10_kda_decode_gated",
    note="b10 CuTeDSL decode + fused gated RMSNorm epilogue (post-norm output, "
    "counterpart of trtllm_kda_fused_decode minus the conv)",
)
def _b10_kda_decode_gated(inputs: DecodeInputs, shape: Shape) -> Callable:
    """b10's gated variant: same recurrence core as ``b10_kda_decode`` with the
    sigmoid-gated RMSNorm fused into the epilogue, so the output is post-norm.
    Norm gate/weight come from the shared :func:`make_conv_norm_inputs`."""
    from kda.b10.b10_kda_decode_gated_cutedsl import kda_decode_gated

    B = inputs.beta_logit.shape[0]
    H, Kd = shape.value_heads, shape.key_dim
    q, k, v = (t.reshape(B, H, -1).contiguous() for t in split_qkv(inputs, shape))
    g = activate_kda_gate(
        inputs.raw_gate.view(B, 1, H, Kd), inputs.A_log, inputs.dt_bias
    ).view(B, H, Kd)
    beta = torch.sigmoid(inputs.beta_logit.float()).view(B, H)
    S = inputs.state.float().transpose(-1, -2).contiguous()
    cn = make_conv_norm_inputs(B, shape, device=str(q.device))

    def run():
        return kda_decode_gated(q, k, v, g, beta, S, cn.z, cn.norm_weight)

    return run


@KDA_DECODE.register(
    "b10_kda_decode_conv_gated",
    note="b10 CuTeDSL fully-fused layer step (conv4+SiLU + recurrence + gated "
    "RMSNorm in ONE kernel), counterpart of trtllm_kda_fused_decode / "
    "sglang_kda_fused_decode (the latter is HV=12-only; this works for ALL H). "
    "Retuned kernel (gen_kda_decode_conv_gated_h12_0730): 3 per-shape compile-time "
    "dispatches, ~95% stream ceiling; beats sglang_kda_fused_decode at EVERY batch "
    "incl B=32. Decay gate AND output gate z are pre-activated untimed here.",
)
def _b10_kda_decode_conv_gated(inputs: DecodeInputs, shape: Shape) -> Callable:
    """b10's conv-fused gated variant: same recurrence + gated-RMSNorm core as
    ``b10_kda_decode_gated`` plus the width-4 causal conv + SiLU on the packed
    pre-conv q/k/v fused in front, so it covers the same layer slice as
    ``sglang_kda_fused_decode`` / ``trtllm_kda_fused_decode``. Conv weights /
    conv state / output gate come from the shared
    :func:`make_conv_norm_inputs`; ``inputs.mixed_qkv`` is taken as RAW
    pre-conv."""
    from kda.b10.b10_kda_decode_conv_gated_cutedsl import kda_decode_conv_gated

    B = inputs.beta_logit.shape[0]
    H, Kd, Vd = shape.value_heads, shape.key_dim, shape.value_dim
    if shape.qk_heads != H or Kd != Vd or Kd != 128:
        raise RuntimeError("conv-fused decode requires equal heads and head_dim=128")
    device = inputs.mixed_qkv.device
    cn = make_conv_norm_inputs(B, shape, device=str(device))
    g = activate_kda_gate(
        inputs.raw_gate.view(B, 1, H, Kd), inputs.A_log, inputs.dt_bias
    ).view(B, H, Kd)
    beta = torch.sigmoid(inputs.beta_logit.float()).view(B, H)
    S = inputs.state.float().transpose(-1, -2).contiguous()
    mixed = inputs.mixed_qkv.contiguous()  # RAW pre-conv packed [q | k | v]
    raw_q, raw_k, raw_v = (
        x.view(B, H, Kd) for x in mixed.split(H * Kd, dim=-1)
    )
    if B > 1:
        assert not raw_q.is_contiguous()
        assert not raw_k.is_contiguous()
        assert not raw_v.is_contiguous()

    def run():
        return kda_decode_conv_gated(
            raw_q,
            raw_k,
            raw_v,
            cn.conv_weight,
            cn.conv_state,
            g,
            beta,
            S,
            cn.z,
            cn.norm_weight,
        )

    return run


@KDA_DECODE.register(
    "fla_kda_recurrent",
    note="upstream FLA Triton fused_recurrent (the original)",
)
def _fla_kda_recurrent(inputs: DecodeInputs, shape: Shape) -> Callable:
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    q, k, v = split_qkv(inputs, shape)
    raw_gate = inputs.raw_gate.view(1, -1, shape.value_heads, shape.key_dim)
    beta_logit = inputs.beta_logit.view(1, -1, shape.value_heads)

    def run():
        return fused_recurrent_kda_fwd(
            q,
            k,
            v,
            raw_gate,
            beta_logit,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            initial_state=inputs.state,
            scale=shape.key_dim**-0.5,
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=True,
            cu_seqlens=inputs.cu_seqlens,
            ssm_state_indices=inputs.state_indices,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )[0]

    return run


@KDA_DECODE.register(
    "sglang_kda_packed",
    note="SGLang serving fusion: packed QKV in, same recurrence as split",
)
def _sglang_kda_packed(inputs: DecodeInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.linear.kernels.kda_triton import (
        TritonKDAKernel,
    )

    kernel = TritonKDAKernel()

    def run():
        return kernel.packed_decode(
            inputs.mixed_qkv,
            inputs.raw_gate,
            inputs.beta_logit,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            scale=shape.key_dim**-0.5,
            ssm_states=inputs.state,
            cache_indices=inputs.state_indices,
            num_v_heads=shape.value_heads,
            head_v_dim=shape.value_dim,
        )

    return run


@KDA_DECODE.register(
    "sglang_kda_split",
    note="SGLang fork of FLA fused_recurrent (varlen T-loop + verify features)",
)
def _sglang_kda_split(inputs: DecodeInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.linear.kernels.kda_triton import (
        TritonKDAKernel,
    )

    kernel = TritonKDAKernel()
    q, k, v = split_qkv(inputs, shape)

    def run():
        return kernel.decode(
            q,
            k,
            v,
            inputs.raw_gate.unsqueeze(0),
            inputs.beta_logit.unsqueeze(0),
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            ssm_states=inputs.state,
            cache_indices=inputs.state_indices,
            query_start_loc=inputs.cu_seqlens,
        )

    return run


@KDA_DECODE.register(
    "sglang_kda_fused_decode",
    note="SGLang kimi-k3 fully-fused single-token decode (conv4+SiLU + gate "
    "activation + recurrence + gated RMSNorm in ONE JIT CUDA kernel, vendored "
    "NVIDIA x Moonshot K3 package); compiled for H=12, K=V=128 only",
)
def _sglang_kda_fused_decode(inputs: DecodeInputs, shape: Shape) -> Callable:
    """SGLang's kimi-k3 branch replaces the three-kernel decode chain
    (``causal_conv1d_update`` -> packed decode -> ``rms_norm_gated``) with one
    JIT CUDA kernel. Imported from the sglang-opensource checkout (the
    installed sglang wheel predates ``sglang.kernels``); conv weights /
    output gate / norm weight are synthetic, matching the other conv-fused
    rows, and ``inputs.mixed_qkv`` is taken as RAW pre-conv."""
    fused = _import_sglang_kernels("kernels.ops.attention.kda_fused_decode")

    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    if shape.qk_heads != H or (H, K, V) != (12, 128, 128):
        raise RuntimeError("sglang fused decode is compiled for H=12, K=V=128")
    if inputs.state.dtype != torch.float32:
        raise RuntimeError("sglang fused decode requires an FP32 state")
    B = inputs.state.shape[0]
    seg, packed = H * K, 3 * H * K
    device = inputs.mixed_qkv.device
    cn = make_conv_norm_inputs(B, shape, device=str(device))
    # The kernel wants fp32 [4, seg] per-segment transposed conv weights
    # (checkpoint dtype) and a time-major [B, 3, C] conv state; same values
    # as the other conv-fused rows, just re-laid-out.
    w_t = cn.conv_weight.float().t().contiguous()
    w_q_t, w_k_t, w_v_t = (
        w_t[:, :seg].contiguous(),
        w_t[:, seg : 2 * seg].contiguous(),
        w_t[:, 2 * seg :].contiguous(),
    )
    conv_bias = torch.zeros(packed, device=device, dtype=torch.float32)
    conv_state = cn.conv_state.transpose(1, 2).contiguous()
    onorm_g = cn.z.reshape(B, seg).contiguous()
    onorm_w = cn.norm_weight
    a = inputs.raw_gate.reshape(B, seg)
    b = inputs.beta_logit.reshape(B, H)
    if not fused.covered(
        inputs.mixed_qkv, a, b, conv_state, inputs.state, inputs.state_indices, onorm_g
    ):
        raise RuntimeError("inputs rejected by kda_fused_decode.covered()")

    def run():
        # lower_bound=None selects the canonical softplus gate, matching the
        # other sglang decode rows and the FLA reference.
        return fused.kda_fused_decode(
            inputs.mixed_qkv,
            a,
            b,
            conv_state,
            w_q_t,
            w_k_t,
            w_v_t,
            conv_bias,
            inputs.A_log,
            inputs.dt_bias,
            onorm_g,
            onorm_w,
            inputs.state,
            inputs.state_indices,
            scale=shape.key_dim**-0.5,
            onorm_eps=1e-5,
            lower_bound=None,
        )

    return run


# Disabled: 2-8x slower than every other row at all shapes (e.g. 1849us vs
# ~260us at H=96 B=128), so it only compresses the plot scale. Re-enable to
# re-measure.
# @KDA_DECODE.register(
#     "sglang_kda_cutedsl",
#     note="SGLang fused CUDA decode via CuTeDSL "
#     "(cutedsl_fused_sigmoid_gating_kda_update, SM90+)",
# )
def _sglang_kda_cutedsl(inputs: DecodeInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
        CuteDSLKDAKernel,
    )

    kernel = CuteDSLKDAKernel()
    q, k, v = split_qkv(inputs, shape)
    # The CuTeDSL kernel is compiled against a bf16 dt_bias (production
    # projection dtype); a fp32 dt_bias is silently misread as bf16 and blows
    # up the gate exp for a head subset (NaN).
    dt_bias = inputs.dt_bias.to(torch.bfloat16)

    def run():
        return kernel.decode(
            q,
            k,
            v,
            inputs.raw_gate.view(1, -1, shape.value_heads, shape.key_dim),
            inputs.beta_logit.unsqueeze(0),
            A_log=inputs.A_log,
            dt_bias=dt_bias,
            ssm_states=inputs.state,
            cache_indices=inputs.state_indices,
            query_start_loc=inputs.cu_seqlens,
        )

    return run


@KDA_DECODE.register(
    "sglang_kda_flashinfer",
    note="SGLang adapter over FlashInfer's KDA decode; unavailable here: this "
    "checkout lacks kernels/kda_flashinfer.py AND flashinfer 0.6.12 only "
    "ships the scalar-gate GDN decode (no channelwise KDA op)",
)
def _sglang_kda_flashinfer(inputs: DecodeInputs, shape: Shape) -> Callable:
    from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
        FlashInferKDAKernel,
    )

    kernel = FlashInferKDAKernel()
    q, k, v = split_qkv(inputs, shape)

    def run():
        return kernel.decode(
            q,
            k,
            v,
            inputs.raw_gate.unsqueeze(0),
            inputs.beta_logit.unsqueeze(0),
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            ssm_states=inputs.state,
            cache_indices=inputs.state_indices,
            query_start_loc=inputs.cu_seqlens,
        )

    return run


@KDA_DECODE.register(
    "vllm_kda_recurrent",
    note="vLLM-vendored FLA fused_recurrent, pre-activated gate",
)
def _vllm_kda_recurrent(inputs: DecodeInputs, shape: Shape) -> Callable:
    from frameworks import _import_vllm_fla

    vllm_kda = _import_vllm_fla("kda")
    batch = inputs.state.shape[0]
    q, k, v = split_qkv(inputs, shape)
    q = q.view(batch, 1, shape.qk_heads, shape.key_dim)
    k = k.view_as(q)
    v = v.view(batch, 1, shape.value_heads, shape.value_dim)
    raw_gate = inputs.raw_gate.view(batch, 1, shape.value_heads, shape.key_dim)
    log_decay = activate_kda_gate(raw_gate, inputs.A_log, inputs.dt_bias)
    beta = inputs.beta_logit.sigmoid().view(batch, 1, shape.value_heads)
    # vLLM reserves state slot 0; shift indices by one.
    state = torch.cat(
        [inputs.state.new_zeros(1, *inputs.state.shape[1:]), inputs.state]
    )
    state_indices = inputs.state_indices + 1

    def run():
        out, final_state = vllm_kda.fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=log_decay,
            beta=beta,
            scale=shape.key_dim**-0.5,
            initial_state=state,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            ssm_state_indices=state_indices,
        )
        return out, final_state[1:]

    return run


@KDA_DECODE.register(
    "trtllm_kda_split",
    note="TRT-LLM fused-Triton varlen recurrent "
    "(fused_sigmoid_gating_delta_rule_update), fork of the SGLang split kernel",
)
def _trtllm_kda_split(inputs: DecodeInputs, shape: Shape) -> Callable:
    trt_gate = _import_trtllm_fla("fused_sigmoid_gating_recurrent")
    batch = inputs.state.shape[0]
    q, k, v = split_qkv(inputs, shape)
    q = q.view(1, batch, shape.qk_heads, shape.key_dim)
    k = k.view_as(q)
    v = v.view(1, batch, shape.value_heads, shape.value_dim)
    a = inputs.raw_gate.view(1, batch, shape.value_heads, shape.key_dim)
    b = inputs.beta_logit.view(1, batch, shape.value_heads)

    def run():
        return trt_gate.fused_sigmoid_gating_delta_rule_update(
            A_log=inputs.A_log,
            a=a,
            dt_bias=inputs.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=inputs.state,
            initial_state_indices=inputs.state_indices,
            scale=shape.key_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs.cu_seqlens,
        )

    return run


@KDA_DECODE.register(
    "trtllm_kda_packed",
    note="TRT-LLM packed-QKV single-token Triton decode "
    "(fused_kda_packed_decode), counterpart of sglang_kda_packed",
)
def _trtllm_kda_packed(inputs: DecodeInputs, shape: Shape) -> Callable:
    trt_gate = _import_trtllm_fla("fused_sigmoid_gating_recurrent")
    batch = inputs.state.shape[0]
    a = inputs.raw_gate.reshape(batch, shape.value_heads * shape.key_dim)
    b = inputs.beta_logit.reshape(batch, shape.value_heads)

    def run():
        # lower_bound=None selects the canonical softplus gate, matching the
        # other canonical-gate decode rows and the FLA reference.
        return trt_gate.fused_kda_packed_decode(
            packed_qkv=inputs.mixed_qkv,
            a=a,
            b=b,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            initial_state_source=inputs.state,
            initial_state_indices=inputs.state_indices,
            scale=shape.key_dim**-0.5,
        )

    return run


@KDA_DECODE.register(
    "trtllm_kda_fused_decode",
    note="TRT-LLM Kimi K3 fully-fused single-token decode (conv4+SiLU + safe "
    "gate + recurrence + gated RMSNorm in ONE Triton kernel); covers more of "
    "the layer than the bare-recurrence rows",
)
def _trtllm_kda_fused_decode(inputs: DecodeInputs, shape: Shape) -> Callable:
    """The `trid/kda-cute-int21-snapshots` branch's specialized decode path
    (modules/mamba/fused_kda_decode.py). It consumes the pre-conv projected
    QKV and fuses the short conv + output norm, so its latency covers the
    full decode step. The kernel hard-codes the SAFE gate (lower_bound in
    [-5, 0)) — see DECODE_SAFE_GATE. Conv weights / output gate / norm
    weight come from the shared :func:`make_conv_norm_inputs`."""
    dec = _import_trtllm_module("modules.mamba.fused_kda_decode")
    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    if shape.qk_heads != H or K != V or K != 128:
        raise RuntimeError("fused KDA decode requires equal heads and head_dim=128")
    if inputs.state.dtype not in (torch.float16, torch.float32):
        raise RuntimeError("fused KDA decode requires an FP16 or FP32 state")
    batch = inputs.state.shape[0]
    device = inputs.mixed_qkv.device
    cn = make_conv_norm_inputs(batch, shape, device=str(device))
    output_gate = cn.z.reshape(batch, H * V).contiguous()
    norm_weight = cn.norm_weight.to(torch.bfloat16)
    raw_gate = inputs.raw_gate.view(1, batch, H, K)
    raw_beta = inputs.beta_logit.view(1, batch, H)

    def run():
        return dec.fused_kda_decode(
            projected_qkv=inputs.mixed_qkv,
            conv_weight=cn.conv_weight,
            conv_state=cn.conv_state,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            state_indices=inputs.state_indices,
            state=inputs.state,
            output_gate=output_gate,
            norm_weight=norm_weight,
            lower_bound=SAFE_GATE_LOWER_BOUND,
        )

    return run


# Pipeline coverage: how much of the full decode step (conv4+SiLU ->
# recurrence -> gated RMSNorm) each row's kernel fuses. The bench composes
# the missing stages from the torch references (conv4_silu_reference /
# gated_rmsnorm_reference) so EVERY row is correctness-checked on the same
# end-to-end pipeline.
DECODE_GATED = {"b10_kda_decode_gated"}  # recurrence + gated RMSNorm fused
DECODE_FUSED_LAYER = {  # conv + recurrence + gated RMSNorm, nothing composed
    "torch_kda_reference",  # eager, not one kernel — but covers the full step
    "sglang_kda_fused_decode",
    "trtllm_kda_fused_decode",
    "b10_kda_decode_conv_gated",
}

# Rows whose kernel hard-codes the SAFE gate (g = lower_bound * sigmoid(...))
# instead of the canonical softplus gate; the bench checks them against the
# safe-gate oracle.
DECODE_SAFE_GATE = {"trtllm_kda_fused_decode"}


def get_kda_decode_backends(
    inputs: DecodeInputs,
    shape: Shape,
) -> tuple[dict[str, Callable], dict[str, str]]:
    return KDA_DECODE.build(inputs, shape)
