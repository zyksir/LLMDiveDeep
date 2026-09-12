"""Explicit kernel adapters. Optional libraries import only when selected.

No backend silently falls back to another family. Preparation (layouts,
mask construction, varlen metadata) is outside the returned timed callable.
"""

from dataclasses import dataclass
from importlib import import_module

import torch
import torch.nn.functional as F

from .dense_attention import (
    attention_mask,
    matmul_attention,
    online_attention,
    validate_qkv,
)


BACKENDS = (
    "torch_eager",
    "torch_online",
    "torch_sdpa_math",
    "torch_sdpa_flash",
    "torch_sdpa_efficient",
    "torch_sdpa_cudnn",
    "torch_flex",
    "fa2",
    "fa3",
    "fa4",
    "sglang_fa3",
    "sglang_fa4",
    "flashinfer_fa2",
    "flashinfer_fa3",
)


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class PreparedAttention:
    run: object
    implementation: str
    boundary: str


def result_tensor(result):
    # FA versions can return Tensor or (Tensor, LSE). Never index Tensor[0].
    return result[0] if isinstance(result, (tuple, list)) else result


def optional_symbol(module, symbol):
    try:
        return getattr(import_module(module), symbol)
    except (ImportError, AttributeError) as exc:
        raise BackendUnavailable(f"{module}.{symbol}: {exc}") from exc


def prepare_attention(name, q, k, v, *, causal=True, window=None):
    validate_qkv(q, k, v)
    if name not in BACKENDS:
        raise ValueError(f"unknown backend: {name}")
    if window is not None and (window < 1 or not causal):
        raise ValueError("a positive window requires causal=True")
    b, nq, h, d = q.shape
    nk, hk = k.shape[1:3]
    scale = d**-0.5
    if name == "torch_eager":
        mask = attention_mask(nq, nk, device=q.device, causal=causal, window=window)
        return PreparedAttention(
            lambda: matmul_attention(q, k, v, mask=mask),
            "torch matmul/softmax (FP32)",
            "eager operators; GQA expansion included; mask prepared",
        )
    if name == "torch_online":
        return PreparedAttention(
            lambda: online_attention(q, k, v, causal=causal, window=window),
            "torch online-softmax teaching loop",
            "Python tile loop + torch operators; not a fused GPU kernel",
        )
    if name.startswith("torch_sdpa_"):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        enum = {
            "math": "MATH",
            "flash": "FLASH_ATTENTION",
            "efficient": "EFFICIENT_ATTENTION",
            "cudnn": "CUDNN_ATTENTION",
        }[name.removeprefix("torch_sdpa_")]
        backend = getattr(SDPBackend, enum)
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
        mask = None
        is_causal = causal and nq == nk
        if window is not None:
            mask = attention_mask(nq, nk, device=q.device, window=window)
            is_causal = False
        elif causal and 1 < nq < nk:
            from torch.nn.attention.bias import causal_lower_right

            mask = causal_lower_right(nq, nk)

        # Q=1 is the final token and sees all KV: no top-left causal flag.
        def run():
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(
                    qh,
                    kh,
                    vh,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=(h != hk),
                ).transpose(1, 2)

        return PreparedAttention(
            run,
            f"torch SDPA forced {enum}",
            "SDPA call; layout views and masks prepared; unsupported masks fail",
        )
    if name == "torch_flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        offset = nk - nq

        def keep(batch, head, qi, ki):
            valid = ki <= qi + offset if causal else qi >= 0
            return valid & (ki >= qi + offset - window + 1) if window else valid

        mask = create_block_mask(
            keep, B=None, H=None, Q_LEN=nq, KV_LEN=nk, device=str(q.device)
        )
        fn = torch.compile(flex_attention, dynamic=False)
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
        return PreparedAttention(
            lambda: fn(
                qh, kh, vh, block_mask=mask, scale=scale, enable_gqa=(h != hk)
            ).transpose(1, 2),
            "torch.compile FlexAttention",
            "compiled core; block mask prepared; first compile excluded",
        )
    qc, kc, vc = (x.contiguous() for x in (q, k, v))
    if name in ("fa2", "fa3", "fa4"):
        module = {
            "fa2": "flash_attn",
            "fa3": "flash_attn_interface",
            "fa4": "flash_attn.cute",
        }[name]
        fn = optional_symbol(module, "flash_attn_func")
        kwargs = dict(softmax_scale=scale, causal=causal)
        if window is not None:
            kwargs["window_size"] = (window - 1, 0)
        return PreparedAttention(
            lambda: result_tensor(fn(qc, kc, vc, **kwargs)),
            f"{module}.flash_attn_func",
            "contiguous BSHD core; no layout copies in timed call",
        )
    if name.startswith("sglang_fa"):
        version = int(name[-1])
        module = (
            "sgl_kernel.flash_attn"
            if version == 3
            else "sglang.kernels.ops.attention.flash_attention_v4"
        )
        if (
            version == 4
            and q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] == 12
        ):
            module += "_sm120"
        fn = optional_symbol(module, "flash_attn_varlen_func")
        cq = torch.arange(b + 1, dtype=torch.int32, device=q.device) * nq
        ck = torch.arange(b + 1, dtype=torch.int32, device=q.device) * nk
        qf, kf, vf = (x.flatten(0, 1) for x in (qc, kc, vc))
        kwargs = dict(
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=nq,
            max_seqlen_k=nk,
            softmax_scale=scale,
            causal=causal,
        )
        if window is not None:
            kwargs["window_size"] = (window - 1, 0)
        return PreparedAttention(
            lambda: result_tensor(fn(qf, kf, vf, **kwargs)).reshape_as(q),
            f"{module}.flash_attn_varlen_func",
            "SGLang kernel wrapper; varlen metadata prepared; no scheduler/layer",
        )
    if name.startswith("flashinfer_"):
        if b != 1:
            raise BackendUnavailable(
                "single-request FlashInfer adapter requires B=1; no serial batching substitute"
            )
        fn = optional_symbol("flashinfer", "single_prefill_with_kv_cache")
        implementation = name.removeprefix("flashinfer_")
        return PreparedAttention(
            lambda: result_tensor(
                fn(
                    qc[0],
                    kc[0],
                    vc[0],
                    causal=causal,
                    kv_layout="NHD",
                    sm_scale=scale,
                    window_left=-1 if window is None else window - 1,
                    backend=implementation,
                )
            ).unsqueeze(0),
            f"flashinfer.single_prefill_with_kv_cache backend={implementation}",
            "single-request core wrapper; JIT warmup excluded",
        )
    raise AssertionError("backend dispatch is incomplete")
