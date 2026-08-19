"""``cutedsl`` backend: our CuTeDSL tile-fused SINGLE kernels.

FLUX-style fusion — one launch does the whole boundary op.

``gemm_allreduce`` is a persistent tcgen05 GEMM whose EPILOGUE
TMA-stores partial tiles into torch symm-mem and reduces them in-kernel
via NVLS ``multimem.ld_reduce``/``st`` (see ``cutedsl_gemm_ar.py``;
measured 1.11–1.32x vs the two-kernel seq baseline at B >= 2048 on
B200 TP8, losing on small messages where the fixed epilogue / M-pad cost
dominates — autotune slots it accordingly).

``allreduce_norm_gemm`` is an M-SPLIT persistent tcgen05 GEMM with a
fused PROLOGUE: 128-row M-tiles are owned round-robin by rank; each
rank ``multimem.ld_reduce``-reduces + RMSNorms ONLY its own rows
(broadcasting the new residual via ``multimem.st``), GEMMs only its
own tiles (1/world of the FLOPs), and broadcasts finished y tiles to
every rank from the epilogue (see ``cutedsl_arng.py``; measured
1.03x/1.38x/1.62x vs the two-kernel seq baseline at B=2048/4096/8192
on B200 TP8, losing at small sizes — autotune slots it accordingly).

``allreduce_norm`` is the GEMM-free boundary op as ONE role-split
kernel: 16 producer blocks ``multimem.ld_reduce`` + residual-add +
``multimem.st``-broadcast their round-robin rows (and normalize them
producer-side), 112 consumer blocks poll per-row flags and normalize
the rest locally (see ``cutedsl_arnorm.py``; measured 1.08x/1.08x/
1.14x vs seq at B=2048/4096/8192 on B200 TP8, losing below B=2048
where the barrier pair + kernel floor dominates — autotune slots it
accordingly). CUDA-graph safe (device rounds + red.max row flags).

Engines are built lazily per shape key (symm rendezvous + kernel
compile — collective, reached only from autotune / explicit ``impl=``
paths).
"""

from __future__ import annotations

import torch

from ..context import Ctx

from ..kernels.cutedsl_arng import CuteDslARNG
from ..kernels.cutedsl_arnorm import _MAX_ROWS as _ARNORM_MAX_ROWS
from ..kernels.cutedsl_arnorm import CuteDslARNorm
from ..kernels.cutedsl_gemm_ar import CuteDslGemmAR


class CuteDslBackend:
    def __init__(self, ctx: Ctx, max_rows: int) -> None:
        if not torch.cuda.get_device_capability()[0] >= 10:
            raise RuntimeError("cutedsl kernels need sm100+")
        self.ctx = ctx
        self.max_rows = max_rows
        self._engines: dict[int, CuteDslGemmAR] = {}
        self._arng: dict[tuple[int, int], CuteDslARNG] = {}
        self._arnorm: dict[int, CuteDslARNorm] = {}

    def supports(self, op: str, tokens: int, n: int) -> bool:
        """Shape-only capacity gate. For ``gemm_allreduce`` ``n`` is the
        output width; for ``allreduce_norm_gemm`` / ``allreduce_norm``
        the dispatcher passes the AR-shaped operand's hidden dim (H)."""
        if tokens > self.max_rows:
            return False
        if op == "gemm_allreduce":
            return (tokens * n <= self.ctx.max_numel
                    and n % 128 == 0)
        if op == "allreduce_norm_gemm":
            return (tokens * n <= self.ctx.max_numel
                    and n % 256 == 0)
        if op == "allreduce_norm":
            return (tokens * n <= self.ctx.max_numel
                    and tokens <= _ARNORM_MAX_ROWS
                    and n % 1024 == 0)
        return False

    fits = supports

    def gemm_allreduce(self, x: torch.Tensor,
                       w: torch.Tensor) -> torch.Tensor:
        """Single-kernel allreduce(x @ w.T); returns a view of the
        engine's phase buffer, valid until its next call. First call
        per N builds the engine and compiles (collective)."""
        n = w.shape[0]
        eng = self._engines.get(n)
        if eng is None:
            eng = CuteDslGemmAR(
                self.ctx.group, max_m=self.max_rows,
                n=n, dtype=self.ctx.dtype)
            self._engines[n] = eng
        return eng.gemm_allreduce(x, w)

    def allreduce_norm(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-kernel rmsnorm(allreduce(x) + residual); returns
        (norm, new_residual) as views of internal buffers, valid until
        the engine's next call. ``residual`` is REQUIRED here (the
        dispatcher passes zeros for the residual-free form). First call
        per H builds the engine and compiles (collective)."""
        h = x.shape[1]
        eng = self._arnorm.get(h)
        if eng is None:
            eng = CuteDslARNorm(
                self.ctx.group,
                max_m=min(self.max_rows, _ARNORM_MAX_ROWS),
                h=h, dtype=self.ctx.dtype)
            self._arnorm[h] = eng
        return eng.allreduce_norm(x, gamma, eps, residual)

    def allreduce_norm_gemm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-kernel rmsnorm(allreduce(x) + residual) @ w.T; returns
        (y, new_residual) as views of internal buffers, valid until the
        engine's next call. First call per (H, N) builds the engine and
        compiles (collective)."""
        h = x.shape[1]
        n = w.shape[0]
        eng = self._arng.get((h, n))
        if eng is None:
            eng = CuteDslARNG(
                self.ctx.group,
                max_m=self.max_rows,
                h=h, n=n, dtype=self.ctx.dtype)
            self._arng[(h, n)] = eng
        return eng.allreduce_norm_gemm(x, residual, gamma, w, eps)
