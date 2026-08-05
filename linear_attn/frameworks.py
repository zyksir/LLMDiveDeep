"""Import shims for framework kernels (vLLM, TensorRT-LLM).

These load individual Triton/CuTe kernel modules straight out of the vLLM and
TRT-LLM checkouts in this workspace WITHOUT importing the frameworks' runtime
packages (whose ``__init__`` drags in engines, config stacks, and CUDA context
setup). Stub packages are registered for every ancestor so the leaf modules'
relative imports resolve against the checkout.

Used by both the KDA (``kda/``) and GDN (``gdn/``) backend registries.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import torch


def _ensure_nvtx_stub() -> None:
    if "nvtx" in sys.modules:
        return
    try:
        importlib.import_module("nvtx")
    except ImportError:
        nvtx = types.ModuleType("nvtx")

        class _Annotate:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __call__(self, fn):
                return fn

        nvtx.annotate = lambda *a, **k: _Annotate()
        sys.modules["nvtx"] = nvtx


def _import_sglang_kernels(dotted: str):
    """Import a leaf module from the sglang-opensource (kimi-k3 branch)
    checkout's ``sglang.kernels`` subtree.

    The venv's installed sglang wheel predates the ``sglang.kernels`` package
    entirely, so the subtree is grafted onto the installed package with stub
    packages (the ``_import_trtllm_module`` trick). The leaf modules'
    ``sglang.srt``/``sglang.utils`` imports still resolve against the
    installed wheel; stub specs carry a real ``origin`` so the JIT compiler's
    ``find_spec``-based csrc path resolution works.
    """

    root = (
        Path(__file__).resolve().parents[2]
        / "sglang-opensource"
        / "python"
        / "sglang"
    )
    if not root.is_dir():
        raise RuntimeError(f"sglang-opensource checkout not found at {root}")
    parts = dotted.split(".")
    assert parts[0] == "kernels", dotted
    name, path = "sglang", root
    for part in parts[:-1]:
        name, path = f"{name}.{part}", path / part
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            spec = importlib.machinery.ModuleSpec(
                name, None, origin=str(path / "__init__.py"), is_package=True
            )
            spec.submodule_search_locations = [str(path)]
            package.__spec__ = spec
            sys.modules[name] = package
    return importlib.import_module(f"sglang.{dotted}")


def _import_trtllm_module(dotted: str):
    """Import a TensorRT-LLM leaf module without loading the full runtime.

    ``dotted`` is the module path below ``tensorrt_llm._torch``, e.g.
    ``modules.fla.fused_recurrent`` or ``modules.mamba.flash_kda``. Stub
    packages are registered for every ancestor so relative imports inside the
    leaf modules resolve against the checkout without importing
    ``tensorrt_llm/__init__.py`` (which drags in the whole runtime).
    """

    workspace = Path(__file__).resolve().parents[2]
    root = next(
        path
        for name in ("TensorRT-LLM", "trt-llm")
        if (path := workspace / name / "tensorrt_llm").is_dir()
    )
    parts = ["_torch", *dotted.split(".")[:-1]]
    name, path = "tensorrt_llm", root
    packages = [(name, path)]
    for part in parts:
        name, path = f"{name}.{part}", path / part
        packages.append((name, path))
    for name, path in packages:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    # ``modules.mamba`` exports ``PAD_SLOT_ID`` from a tiny ``__init__.py``.
    # Do NOT runpy the root ``tensorrt_llm/__init__.py`` — that pulls the
    # full runtime. Just set the constant on the stub package.
    mamba_pkg = "tensorrt_llm._torch.modules.mamba"
    if mamba_pkg in sys.modules and not hasattr(sys.modules[mamba_pkg], "PAD_SLOT_ID"):
        sys.modules[mamba_pkg].PAD_SLOT_ID = -1

    # The KDA modules import these two runtime helpers; stub them so the
    # kernel import stays free of the full TensorRT-LLM runtime.
    if "tensorrt_llm._utils" not in sys.modules:
        shim = types.ModuleType("tensorrt_llm._utils")
        shim.is_flashinfer_gdn_supported_arch = lambda *a, **k: False
        # Enough for leaf Triton/FLA imports without the full runtime.
        shim.get_sm_version = lambda: (
            torch.cuda.get_device_capability()[0] * 10
            + torch.cuda.get_device_capability()[1]
            if torch.cuda.is_available()
            else 90
        )
        # default-bind: ``shim`` is rebound to the logger stub below, and a
        # late-binding lambda would call get_sm_version on the wrong module
        shim.is_sm_100f = lambda _s=shim: _s.get_sm_version() >= 100
        shim.mpi_rank = lambda: 0
        shim.mpi_world_size = lambda: 1
        shim.mpi_disabled = lambda: True
        sys.modules[shim.__name__] = shim
    else:
        shim = sys.modules["tensorrt_llm._utils"]
        if not hasattr(shim, "get_sm_version"):
            shim.get_sm_version = lambda: (
                torch.cuda.get_device_capability()[0] * 10
                + torch.cuda.get_device_capability()[1]
                if torch.cuda.is_available()
                else 90
            )
        if not hasattr(shim, "is_sm_100f"):
            shim.is_sm_100f = lambda _s=shim: _s.get_sm_version() >= 100
    if "tensorrt_llm.logger" not in sys.modules:
        shim = types.ModuleType("tensorrt_llm.logger")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *a, **k: None

        shim.logger = _Logger()
        sys.modules[shim.__name__] = shim
    _ensure_nvtx_stub()
    return importlib.import_module(f"tensorrt_llm._torch.{dotted}")


def _import_trtllm_fla(module: str):
    """Import a leaf module from TensorRT-LLM's vendored FLA package."""

    return _import_trtllm_module(f"modules.fla.{module}")


def _import_vllm_fla(module: str):
    """Import a vLLM FLA leaf module without loading the model/runtime stack."""

    root = Path(__file__).resolve().parents[2] / "vllm" / "vllm"
    packages = [
        ("vllm", root),
        ("vllm.model_executor", root / "model_executor"),
        ("vllm.model_executor.layers", root / "model_executor" / "layers"),
        ("vllm.model_executor.layers.fla", root / "model_executor" / "layers" / "fla"),
        (
            "vllm.model_executor.layers.fla.ops",
            root / "model_executor" / "layers" / "fla" / "ops",
        ),
        ("vllm.utils", root / "utils"),
    ]
    for name, path in packages:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package

    if "vllm.triton_utils" not in sys.modules:
        import triton
        import triton.language as tl
        import triton.language.extra.libdevice as tldevice

        shim = types.ModuleType("vllm.triton_utils")
        shim.triton, shim.tl, shim.tldevice = triton, tl, tldevice
        sys.modules[shim.__name__] = shim

    if "vllm.platforms" not in sys.modules:
        shim = types.ModuleType("vllm.platforms")
        shim.current_platform = types.SimpleNamespace(is_cuda_alike=lambda: True)
        sys.modules[shim.__name__] = shim

    if "vllm.utils.math_utils" not in sys.modules:
        import triton

        shim = types.ModuleType("vllm.utils.math_utils")
        shim.RCP_LN2 = 1.4426950408889634
        shim.cdiv = triton.cdiv
        shim.next_power_of_2 = triton.next_power_of_2
        sys.modules[shim.__name__] = shim

    if "vllm.model_executor.custom_op" not in sys.modules:
        shim = types.ModuleType("vllm.model_executor.custom_op")

        class CustomOp(torch.nn.Module):
            @classmethod
            def register(cls, _name):
                return lambda implementation: implementation

        shim.CustomOp = CustomOp
        sys.modules[shim.__name__] = shim

    return importlib.import_module(f"vllm.model_executor.layers.fla.ops.{module}")
