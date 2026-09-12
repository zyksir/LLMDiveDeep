"""Persist every JIT/compile cache the kimi_k3 benches touch.

Import this module (or call :func:`apply`) BEFORE importing cutlass /
flashinfer / triton / the block modules, and the first bench invocation pays
each compile once; every later process starts from warm disk caches.

    import kimi_k3.kernels.jit_cache_env  # applies on import
    # ... heavy imports / bench code ...

Lives at ``kernels/`` level (not inside ``sgl_adapters/``) because its cache
scope spans every kernel stack the benches touch — CuTe DSL, triton,
flashinfer, torch extensions — not just the vendored sglang JIT.

All caches default to subdirectories of ``<repo>/out/jit_cache`` (override the
root with ``LLMDD_JIT_CACHE_ROOT``; set ``LLMDD_JIT_CACHE_DISABLE=1`` to make
this module a no-op). The repo lives on /node-storage, which is bind-mounted
into the trt-k3 container at the same path, so the caches survive container
recreation — unlike the previous defaults under ``/root`` and ``/tmp``.

Knobs applied (each only when the variable is not already set):

``CUTE_DSL_CACHE_DIR`` and ``CUTE_EXPERIMENTAL_DSL_CACHE_DIR`` -> ``<root>/cute_dsl``
    nvidia-cutlass-dsl's on-disk cache of compiled MLIR modules (cubin
    embedded), keyed by a hash of the traced IR + env/compile options + DSL
    version. Saves the multi-second MLIR->PTX->cubin pipeline per kernel
    variant. Read at compile time, so setting it any time before the first
    compile works. The default is ``$TMPDIR/<user>/cutlass_python_cache``.
    Two variables because ``cute.compile`` and ``cute.experimental`` (the
    vendored TGV GEMM, KDA decode) are *separate DSL singletons* with separate
    env prefixes; cache files are DSL-name-prefixed so one directory serves
    both.

``CUTE_DSL_DUMP_DIR`` and ``CUTE_EXPERIMENTAL_DSL_DUMP_DIR`` -> ``<root>/cute_dsl_dump``
    The DSL folds every env knob into the cache hash, and DUMP_DIR defaults to
    the *current working directory* — without pinning it, running a bench from
    a different cwd silently invalidates every CuTe cache entry.

CuTe DSL patch (not an env var — see :func:`_patch_cute_dsl`):
    In nvidia-cutlass-dsl 4.5.0 the explicit-compile entry points the benches
    use (``cute.compile`` / ``cute.experimental.compile``) force
    ``no_cache=True`` ("Cache is disabled as user wants to compile only"), so
    the file cache above is never consulted for them. apply() re-enables the
    cache for those calls by wrapping ``BaseDSL.compile_and_cache`` — the same
    load/write machinery the supported ``@cute.jit`` direct-call path uses.
    The wrapper backs off whenever the user asked for uncached behaviour
    (``CUTE_DSL_NO_CACHE``, ``CUTE_DSL_DISABLE_FILE_CACHING``,
    ``CUTE_DSL_KEEP=ptx/cubin``). If cutlass is not imported yet, the patch is
    installed via an import hook so it lands whenever cutlass loads.

``SGLANG_JIT_CACHE_DIR`` -> ``<root>/sglang_jit``
    The vendored sgl_copied_kernels tvm-ffi build cache (content-addressed
    ``.so`` leaves; see ``sgl_copied_kernels/jit/utils/compile/cache.py``).
    Already persistent by design at ``~/.cache/sglang/jit``; pinned here so it
    survives container recreation. Saves ninja+nvcc per JIT module.

``FLASHINFER_WORKSPACE_BASE`` -> ``<root>/flashinfer``
    flashinfer resolves all of its dirs from this base at *import time*:
    JIT-compiled ops under ``<base>/.cache/flashinfer/<version>/<arch>/`` and
    downloaded trtllm-gen cubins under ``<base>/.cache/flashinfer/cubins``.
    Saves minutes of nvcc (moe_utils, trtllm_gen_fused_moe, ...) plus the
    cubin downloads. Must be set before flashinfer is imported; apply() warns
    and leaves it alone if it is too late. The pre-existing warm tree at
    ``~/.cache/flashinfer`` is copied in once: that carries the downloaded
    cubins over (no network on first run), but the *compiled* ops still
    rebuild once — flashinfer's ninja files embed absolute paths, so a
    relocated tree is stale by construction (measured: moe_utils 218 s once,
    0.09 s in every process after).

``TRITON_CACHE_DIR`` -> ``<root>/triton``
    Triton's compiled-kernel cache (used by the repo's triton kernels and any
    torch.compile-generated ones). Persists by default at ``~/.triton/cache``;
    pinned + seeded for durability.

``TORCH_EXTENSIONS_DIR`` -> ``<root>/torch_extensions``
    torch.utils.cpp_extension build dir (communication kernels such as
    ``k3_comm_cuda`` / ``b10_multimem_ar``). Persists by default at
    ``~/.cache/torch_extensions``; pinned + seeded.

``TORCHINDUCTOR_CACHE_DIR`` -> ``<root>/torchinductor``
    torch.compile / inductor artifacts. The default is ``/tmp/torchinductor_
    <user>`` which is genuinely ephemeral.

NOT persistable from here — TRT-LLM autotuner:
    tensorrt_llm._torch.autotuner has no env knob; persistence exists only as
    ``with autotune(cache_path=...)`` which loads/saves a per-rank cache file.
    Bench owners can pass ``cache_path=str(autotuner_cache_path("<bench>"))``
    to their existing ``autotune()`` calls to skip re-profiling on warm runs.

Everything here is idempotent and concurrency-safe: env vars are only
defaulted, the one-time seeding copies into a temp dir and publishes with an
atomic rename (8 mpi ranks applying simultaneously race harmlessly).
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import inspect
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Anchored to the REPO root (this file sits two levels down, at
# <repo>/kimi_k3/kernels/), so the resolved default stays
# <repo>/out/jit_cache regardless of where the module lives in the package.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "out" / "jit_cache"

_applied = False


def cache_root() -> Path:
    """The active cache root (honors ``LLMDD_JIT_CACHE_ROOT``)."""
    return Path(os.environ.get("LLMDD_JIT_CACHE_ROOT", str(_DEFAULT_ROOT)))


def autotuner_cache_path(name: str) -> Path:
    """A stable file path for ``tensorrt_llm...autotune(cache_path=...)``.

    The autotuner appends its own per-rank suffix; *name* should identify the
    bench/workload so different shape sets do not share a file.
    """
    directory = cache_root() / "trtllm_autotuner"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.json"


def apply() -> None:
    """Set all cache env vars and install the CuTe DSL cache patch. Idempotent."""
    global _applied
    if _applied or os.environ.get("LLMDD_JIT_CACHE_DISABLE") == "1":
        return
    _applied = True

    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)

    _default_env("CUTE_DSL_CACHE_DIR", root / "cute_dsl")
    _default_env("CUTE_EXPERIMENTAL_DSL_CACHE_DIR", root / "cute_dsl")
    _default_env("CUTE_DSL_DUMP_DIR", root / "cute_dsl_dump")
    _default_env("CUTE_EXPERIMENTAL_DSL_DUMP_DIR", root / "cute_dsl_dump")
    _default_env(
        "SGLANG_JIT_CACHE_DIR",
        root / "sglang_jit",
        seed_from=Path.home() / ".cache" / "sglang" / "jit",
    )
    _default_env(
        "FLASHINFER_WORKSPACE_BASE",
        root / "flashinfer",
        must_precede_import="flashinfer",
        # flashinfer derives <base>/.cache/flashinfer itself; seed that subtree
        # from the warm default. This carries the downloaded cubins over (no
        # network on first run); compiled ops still rebuild once because
        # flashinfer's ninja files embed absolute paths (see docstring).
        seed_from=Path.home() / ".cache" / "flashinfer",
        seed_subdir=Path(".cache") / "flashinfer",
    )
    _default_env(
        "TRITON_CACHE_DIR",
        root / "triton",
        seed_from=Path.home() / ".triton" / "cache",
    )
    _default_env(
        "TORCH_EXTENSIONS_DIR",
        root / "torch_extensions",
        seed_from=Path.home() / ".cache" / "torch_extensions",
    )
    _default_env("TORCHINDUCTOR_CACHE_DIR", root / "torchinductor")

    _enable_cute_dsl_file_cache()


def _default_env(
    name: str,
    target: Path,
    *,
    must_precede_import: str | None = None,
    seed_from: Path | None = None,
    seed_subdir: Path | None = None,
) -> None:
    if os.environ.get(name):
        return
    if must_precede_import and must_precede_import in sys.modules:
        logger.warning(
            "jit_cache_env: %s is already imported; %s stays unset and its "
            "caches keep their current location. Import kimi_k3."
            "kernels.jit_cache_env earlier to pin it.",
            must_precede_import,
            name,
        )
        return
    if seed_from is not None:
        _seed_once(seed_from, target / seed_subdir if seed_subdir else target)
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)


def _seed_once(src: Path, dst: Path) -> None:
    """Copy a warm cache tree into place, once, atomically.

    Publishing via ``os.rename`` makes concurrent ranks race harmlessly: one
    wins, the losers discard their copy. Failure is never fatal — the cache
    is then simply cold and the first run rebuilds it.
    """
    if dst.exists() or not src.is_dir():
        return
    staging = dst.with_name(f".{dst.name}.seed-{os.getpid()}")
    try:
        shutil.copytree(src, staging)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, dst)
        logger.info("jit_cache_env: seeded %s from %s", dst, src)
    except OSError as error:
        logger.warning("jit_cache_env: could not seed %s from %s: %s", dst, src, error)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# CuTe DSL: let explicit `cute.compile` use the DSL's own file cache
# ---------------------------------------------------------------------------

_CUTE_DSL_MODULE = "cutlass.base_dsl.dsl"


def _enable_cute_dsl_file_cache() -> None:
    module = sys.modules.get(_CUTE_DSL_MODULE)
    if module is not None:
        _patch_cute_dsl(module)
    else:
        sys.meta_path.insert(0, _PatchOnImport(_CUTE_DSL_MODULE, _patch_cute_dsl))


def _patch_cute_dsl(dsl_module) -> None:
    """Re-enable the DSL file cache for explicit-compile calls.

    ``cute.compile`` sets ``compile_only=True`` which (in nvidia-cutlass-dsl
    4.5.0) forces ``no_cache=True`` before ``BaseDSL.compile_and_cache`` runs,
    so compiled modules are neither looked up nor written to the file cache.
    The wrapper flips that back unless the user explicitly opted out of
    caching. Cache-hit results are rebuilt through the exact code path the
    supported ``@cute.jit`` direct-call flow uses (load bytecode, re-JIT the
    host engine around the embedded cubin), so behaviour on a hit is the
    DSL's own, not ours.
    """
    base_dsl = getattr(dsl_module, "BaseDSL", None)
    original = getattr(base_dsl, "compile_and_cache", None)
    if original is None:
        logger.warning(
            "jit_cache_env: %s has no BaseDSL.compile_and_cache; CuTe DSL "
            "compiles will not persist across processes.",
            _CUTE_DSL_MODULE,
        )
        return
    if getattr(original, "_llmdd_jit_cache_patch", False):
        return
    signature = inspect.signature(original)
    if "no_cache" not in signature.parameters:
        logger.warning(
            "jit_cache_env: BaseDSL.compile_and_cache has no `no_cache` "
            "parameter (DSL API changed); CuTe DSL compiles will not persist "
            "across processes.",
        )
        return

    @functools.wraps(original)
    def compile_and_cache(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        envar = self.envar
        user_opted_out = (
            getattr(envar, "no_cache", False)
            or getattr(envar, "disable_file_caching", False)
            or getattr(envar, "keep_ptx", False)
            or getattr(envar, "keep_cubin", False)
        )
        if bound.arguments.get("no_cache") and not user_opted_out:
            bound.arguments["no_cache"] = False
        return original(*bound.args, **bound.kwargs)

    compile_and_cache._llmdd_jit_cache_patch = True
    base_dsl.compile_and_cache = compile_and_cache
    logger.debug("jit_cache_env: CuTe DSL file cache enabled for cute.compile")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Run a callback on a module right after it is first executed."""

    def __init__(self, fullname: str, callback) -> None:
        self._fullname = fullname
        self._callback = callback
        self._finding = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._fullname or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._finding = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._callback, self)
        return spec


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, callback, finder: _PatchOnImport) -> None:
        self._inner = inner
        self._callback = callback
        self._finder = finder

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            pass
        self._callback(module)


apply()
