"""Target-architecture helpers for every JIT/AOT compile site in this repo.

One place decides what the kernels are built for, so a checkout runs on every
Blackwell datacenter part we deploy on instead of only the box it was tuned on:

    (10, 0) -> sm_100a   B200, GB200   (GB200 is Grace + the same GB100 die)
    (10, 3) -> sm_103a   B300, GB300

Both spellings are accepted by the CUDA 13.2 nvcc and the CuTeDSL 4.5.2 in the
1.3.0rc23 container (verified: `nvcc -gencode=arch=compute_103a,code=sm_103a`
compiles, and cutlass advertises sm_100a/sm_101a/sm_103a plus the `f` family
variants).

Why architecture-specific (``a``) and not family (``f``) targets: the CuTeDSL
GEMMs and the multimem kernels use arch-specific instructions (tcgen05 MMA,
multimem.ld_reduce), which the family variants do not expose. One cubin cannot
cover 10.0 and 10.3, so each site compiles for the device it is running on --
and every cache key therefore has to carry the arch (see ``ext_name`` and
``tag``), or a B200-built artifact gets loaded on a B300.

Set ``B10_FORCE_SM=10.3`` to exercise a target's flags on the wrong host (or
with no GPU at all); it only affects flag/name generation, never dispatch of
real work. Run ``python3 -m common.arch`` for a self-check that needs no GPU.
"""

from __future__ import annotations

import os
from typing import Iterable

# (major, minor) -> nvcc/CuTeDSL architecture suffix.
SUPPORTED: dict[tuple[int, int], str] = {
    (10, 0): "100a",  # B200 / GB200
    (10, 3): "103a",  # B300 / GB300
}

_FORCE_ENV = "B10_FORCE_SM"


class UnsupportedArch(RuntimeError):
    """Raised when a compile site is asked to target an arch we have no flags for."""


def sm_version() -> tuple[int, int]:
    """(major, minor) of the target arch, honouring ``B10_FORCE_SM``."""
    forced = os.environ.get(_FORCE_ENV)
    if forced:
        major, _, minor = forced.partition(".")
        return int(major), int(minor or 0)
    import torch

    return torch.cuda.get_device_capability()


def suffix(version: tuple[int, int] | None = None) -> str:
    """``'103a'`` for the target arch. Raises for anything we have no flags for."""
    version = version or sm_version()
    try:
        return SUPPORTED[version]
    except KeyError:
        known = ", ".join(f"sm_{value}" for value in sorted(SUPPORTED.values()))
        raise UnsupportedArch(
            f"compute capability {version[0]}.{version[1]} is not a target this "
            f"repo builds kernels for (have: {known}). Add it to "
            f"common.arch.SUPPORTED once its kernels are validated."
        ) from None


def tag(version: tuple[int, int] | None = None) -> str:
    """Short cache tag, e.g. ``'sm103a'``."""
    return f"sm{suffix(version)}"


def cutedsl_arch_option(version: tuple[int, int] | None = None) -> str:
    """``--gpu-arch`` fragment for ``cute.compile(..., options=...)``."""
    return f"--gpu-arch sm_{suffix(version)}"


def nvcc_gencode(version: tuple[int, int] | None = None) -> list[str]:
    """``-gencode`` flags for ``torch.utils.cpp_extension`` / raw nvcc."""
    value = suffix(version)
    return [f"-gencode=arch=compute_{value},code=sm_{value}"]


def torch_arch_list(version: tuple[int, int] | None = None) -> str:
    """Value for ``TORCH_CUDA_ARCH_LIST`` (torch wants ``'10.3a'``)."""
    version = version or sm_version()
    suffix(version)  # validate before handing the string to nvcc
    return f"{version[0]}.{version[1]}a"


def ext_name(base: str, version: tuple[int, int] | None = None) -> str:
    """Arch-qualified extension name.

    ``torch.utils.cpp_extension`` keys its build directory on the module name
    and the source hash -- NOT on the target arch -- so two nodes of different
    arch sharing a cache (a baked image, an NFS ``TORCH_EXTENSIONS_DIR``) would
    silently reuse the first one's cubin. Qualifying the name keys them apart.
    """
    return f"{base}_{tag(version)}"


def supports(*versions: Iterable[tuple[int, int]]) -> bool:
    """True when the target arch is one of ``versions``."""
    return sm_version() in set(versions)


def _selftest() -> None:
    """Exercise both targets without a GPU (``python3 -m common.arch``)."""
    original = os.environ.get(_FORCE_ENV)
    try:
        for forced, want_suffix in (("10.0", "100a"), ("10.3", "103a")):
            os.environ[_FORCE_ENV] = forced
            assert suffix() == want_suffix, suffix()
            assert tag() == f"sm{want_suffix}"
            assert cutedsl_arch_option() == f"--gpu-arch sm_{want_suffix}"
            assert nvcc_gencode() == [
                f"-gencode=arch=compute_{want_suffix},code=sm_{want_suffix}"]
            assert torch_arch_list() == f"{forced}a"
            assert ext_name("k3_comm_cuda") == f"k3_comm_cuda_sm{want_suffix}"
            print(f"sm {forced}: {cutedsl_arch_option()}, "
                  f"{nvcc_gencode()[0]}, ext=" + ext_name("k3_comm_cuda"))
        os.environ[_FORCE_ENV] = "9.0"
        try:
            suffix()
        except UnsupportedArch as error:
            print(f"sm 9.0 rejected as expected: {str(error)[:60]}...")
        else:
            raise AssertionError("sm 9.0 should not be a supported target")
    finally:
        if original is None:
            os.environ.pop(_FORCE_ENV, None)
        else:
            os.environ[_FORCE_ENV] = original
    print("common.arch selftest OK")


if __name__ == "__main__":
    _selftest()
