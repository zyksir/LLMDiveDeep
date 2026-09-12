"""Kimi-K3 top-16 routing via the unchanged SGLang ``RouteRadixKernel``.

Measured fastest K3 router at every batch size (2.08-2.50 us at decode vs the
Triton kernel's ~6 us; see ``routing_permutation_results.md``): 224 threads per
token, byte-wise radix COUNTING selection instead of serial top-k rounds. The
kernel body under ``sglang_radix/`` is vendored byte-for-byte from SGLang with
provenance and license beside it (``PROVENANCE.json`` carries a SHA-256 per
file); only the thin tvm-ffi adapter is local.

Built once per (source, arch, tvm-ffi) into a content-addressed cache at layer
init -- ~4 min cold, <1 s warm. Nothing compiles per shape or per request.

**The target arch is detected, never hardcoded.** ``common.arch`` is the single
place in this repo that decides compile targets, and three things here have to
agree with it or the build is wrong in a way that is easy to miss:

* ``-DSGL_CUDA_ARCH`` -- ``include/sgl_kernel/utils.cuh:122`` asserts
  ``__CUDA_ARCH__ == SGL_CUDA_ARCH``, so a stale value is a compile error
  (loud, which is why it is worth keeping that way).
* ``TVM_FFI_CUDA_ARCH_LIST`` -- decides which cubin is emitted. Building
  ``10.0a`` and running on a 10.3 device is *not* a static_assert; the
  arch-specific cubin simply has no image for the device at load time.
* the module name and the cache key -- ``tvm_ffi.cpp.build`` keys its output on
  name and source, NOT on arch, so two nodes of different arch sharing a cache
  (a baked image, an NFS home) would silently reuse the first one's ``.so``.
  Both the name and the hash carry the arch tag, exactly as
  ``common.arch.ext_name`` does for the torch extensions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from common import arch

from kimi_k3.config import NUM_EXPERTS, ROUTED_SCALING, TOP_K

_PKG = Path(__file__).resolve().parent / "sglang_radix"
_MODULE = None


def _cuda_arch_macro() -> int:
    """``__CUDA_ARCH__`` for the target arch: 1000 for sm_100a, 1030 for sm_103a."""
    major, minor = arch.sm_version()
    arch.suffix((major, minor))  # reject a target we have no flags for
    return major * 100 + minor * 10


def _cuda_flags() -> tuple[str, ...]:
    # No fast-math: the artifact is ID-exact against the FP32 oracle and
    # sigmoid rounding must stay bit-for-bit unchanged.
    return (
        "-std=c++20",
        "-O3",
        f"-DSGL_CUDA_ARCH={_cuda_arch_macro()}",
        "--expt-relaxed-constexpr",
    )


def _source_hash(flags: tuple[str, ...], arch_list: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(_PKG.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    digest.update(" ".join(flags).encode())
    # Not implied by the flags: TVM_FFI_CUDA_ARCH_LIST is what selects the
    # cubin, so it has to be part of the key too.
    digest.update(arch_list.encode())
    return digest.hexdigest()[:16]


def load_radix_module():
    """Build (cold) or load (warm) the radix routing extension for this device."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    import tvm_ffi
    from tvm_ffi.cpp import build

    flags = _cuda_flags()
    arch_list = arch.torch_arch_list()          # e.g. '10.3a'
    name = arch.ext_name("routing_radix")       # e.g. 'routing_radix_sm103a'
    cache = Path(
        os.environ.get(
            "TRTLLM_KIMI_K3_KERNEL_CACHE",
            Path.home() / ".cache" / "trtllm-kimi-k3",
        )
    ) / f"sglang_radix_{_source_hash(flags, arch_list)}_ffi{tvm_ffi.__version__}"
    artifact = cache / f"{name}.so"
    if not artifact.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        env = {"TVM_FFI_CUDA_ARCH_LIST": arch_list, "MAX_JOBS": "1"}
        saved = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            built = Path(build(
                name=name,
                cuda_files=str(_PKG / "adapter.cu"),
                extra_cflags=["-std=c++20", "-O3"],
                extra_cuda_cflags=list(flags),
                extra_include_paths=[str(_PKG / "include"), str(_PKG / "csrc")],
                build_directory=str(cache),
            ))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if built != artifact:
            artifact = built
    _MODULE = tvm_ffi.load_module(str(artifact))
    return _MODULE


def warmup() -> None:
    """Build/load the extension. Call from layer init, never per step."""
    load_radix_module()


def route_radix_into(
    logits: torch.Tensor,
    gate_bias_f32: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    sorted: bool = False,  # noqa: A002 - keyword is the kernel's own API
) -> None:
    """Route into caller-owned ``weights`` fp32 [B,16] / ``ids`` int32 [B,16].

    The allocation-free form, so a benchmark times the kernel and not
    ``torch.empty``.
    """
    load_radix_module().run(
        logits[:, :NUM_EXPERTS], gate_bias_f32, weights, ids,
        TOP_K, float(ROUTED_SCALING), True, ROUTED_SCALING != 1.0, sorted,
    )


def route_radix(
    logits: torch.Tensor,
    gate_bias_f32: torch.Tensor,
    *,
    fmt: str = "trtllm_gen",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ids int32 [B,16], scales)`` from BF16/FP32 logits ``[B, >=896]``.

    ``fmt="trtllm_gen"`` returns BF16 scales for the ``run_moe`` handoff;
    ``fmt="fp32"`` returns the kernel-native FP32 scales.
    """
    batch = logits.shape[0]
    weights = torch.empty(batch, TOP_K, device=logits.device,
                          dtype=torch.float32)
    ids = torch.empty(batch, TOP_K, device=logits.device, dtype=torch.int32)
    route_radix_into(logits, gate_bias_f32, weights, ids)
    if fmt == "trtllm_gen":
        return ids, weights.to(torch.bfloat16)
    if fmt == "fp32":
        return ids, weights
    raise ValueError(f"unsupported routing format {fmt!r}")


if __name__ == "__main__":
    flags = _cuda_flags()
    print(f"target        sm_{arch.suffix()}")
    print(f"SGL_CUDA_ARCH {_cuda_arch_macro()}")
    print(f"arch list     {arch.torch_arch_list()}")
    print(f"module name   {arch.ext_name('routing_radix')}")
    print(f"cache key     sglang_radix_"
          f"{_source_hash(flags, arch.torch_arch_list())}")
