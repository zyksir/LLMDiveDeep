"""kernel-bench: compare N semantically equivalent implementations of one op.

Three parts:

1. Time one function — ``bench_cuda``: us per call via CUDA-graph replay
   (eager fallback when not capturable).
2. Compare implementations — same canonical inputs (adapt is untimed), timed
   run closures, outputs normalized back; reports correctness vs a baseline
   and speedup vs that baseline. A failing impl becomes an ``error`` cell;
   the rest still run. Use ``check_impls`` / ``bench_impls`` (+
   ``BackendRegistry``) over plain closure dicts, or subclass ``KernelBench``.
3. Present results — rows are plain dicts: ``print_table`` / ``write_csv`` /
   ``plot_rows``, or all three in one call via ``report``.
"""

from __future__ import annotations

import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# 1. Time one function
# ---------------------------------------------------------------------------


def _try_graph_capture(fn: Callable, iters: int) -> "torch.cuda.CUDAGraph | None":
    """Capture `iters` calls of fn() into one CUDA graph; None if not capturable."""
    try:
        # warm once on a side stream so lazy init stays out of the capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.cuda.graph(graph):
                for _ in range(iters):
                    fn()
        return graph
    except Exception:
        torch.cuda.synchronize()
        return None


def bench_cuda(
    fn: Callable,
    warmup: int = 10,
    iters: int = 50,
    repeats: int = 5,
    use_graph: bool = True,
) -> float:
    """Median GPU-event-timed us per call (CUDA-graph replay; eager fallback).

    Closures that launch outside the capture stream (e.g. CuTeDSL drivers that
    cache a raw stream) capture an EMPTY graph without erroring; they must set
    ``fn.use_graph = False`` to force eager timing.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    use_graph = use_graph and getattr(fn, "use_graph", True)
    graph = _try_graph_capture(fn, iters) if use_graph else None

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iters):
                fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    del graph
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# 2. Compare implementations — functional API ({name: closure} -> dict rows)
# ---------------------------------------------------------------------------


class BackendRegistry:
    """Named builders (inputs -> run closure) for one contract; builders raise
    when unavailable and build() reports that instead of aborting."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.builders: dict[str, Callable] = {}
        self.notes: dict[str, str] = {}

    def register(self, name: str, note: str = "") -> Callable:
        """Decorator: register a builder under `name` (note = provenance)."""
        if name in self.builders:
            raise ValueError(f"backend {name!r} already registered")

        def decorator(builder: Callable) -> Callable:
            self.builders[name] = builder
            if note:
                self.notes[name] = note
            return builder

        return decorator

    def build(
        self, *args, only: "list[str] | None" = None, **kwargs
    ) -> tuple[dict[str, Callable], dict[str, str]]:
        """Build all (or `only`) backends -> (runners, unavailable reasons)."""
        runners: dict[str, Callable] = {}
        unavailable: dict[str, str] = {}
        for name, builder in self.builders.items():
            if only and name not in only:
                continue
            try:
                runners[name] = builder(*args, **kwargs)
            except Exception as exc:
                unavailable[name] = f"{type(exc).__name__}: {exc}"
        return runners, unavailable


def error_stats(
    out: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict:
    """THE shared error measure for all benches: ``max_abs`` + ``cosine`` for
    reporting, ``pass`` = elementwise ``|out - ref| <= atol + rtol * |ref|``
    (``torch.allclose``). The atol term covers near-zero elements, the rtol
    term scales with magnitude — for bf16 kernel outputs rtol has a hard
    floor of one bf16 ulp (2^-7 ~ 7.8e-3 relative)."""
    ref = ref.float()
    out = out.float().reshape_as(ref)
    return {
        "max_abs": (out - ref).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            out.flatten(), ref.flatten(), dim=0
        ).item(),
        "pass": torch.allclose(out, ref, atol=atol, rtol=rtol),
    }


def check_impls(
    runners: dict[str, Callable[[], Any]],
    reference: torch.Tensor,
    normalize: Callable[[str, Any], torch.Tensor],
    atol: float = 5e-3,
    rtol: float = 5e-3,
    notes: dict[str, str] | None = None,
) -> list[dict]:
    """Run each impl once, normalize its output, compare to `reference`."""
    rows: list[dict] = []
    for name, run in runners.items():
        row: dict[str, Any] = {"impl": name}
        try:
            row.update(
                error_stats(normalize(name, run()), reference, atol=atol, rtol=rtol)
            )
        except Exception as exc:
            torch.cuda.empty_cache()
            row["error"] = f"{type(exc).__name__}: {exc}"
        if notes and notes.get(name):
            row["note"] = notes[name]
        rows.append(row)
    return rows


def bench_impls(
    runners: dict[str, Callable[[], Any]],
    warmup: int,
    iters: int,
    repeats: int,
    row_extra: Callable[[str, float], dict] | None = None,
    notes: dict[str, str] | None = None,
    baseline: str | None = None,
) -> list[dict]:
    """Time each closure -> rows with latency_us (+ speedup vs `baseline`,
    + whatever row_extra(name, latency_us) returns)."""
    rows: list[dict] = []
    for name, run in runners.items():
        row: dict[str, Any] = {"impl": name}
        try:
            latency = bench_cuda(run, warmup, iters, repeats)
            row["latency_us"] = latency
            if row_extra:
                row.update(row_extra(name, latency))
        except Exception as exc:
            torch.cuda.empty_cache()
            row["error"] = f"{type(exc).__name__}: {exc}"
        if notes and notes.get(name):
            row.setdefault("note", notes[name])
        rows.append(row)
    if baseline is not None:
        base_us = next(
            (r["latency_us"] for r in rows
             if r["impl"] == baseline and "latency_us" in r),
            None,
        )
        if base_us:
            for row in rows:
                if "latency_us" in row:
                    row["speedup"] = base_us / row["latency_us"]
    return rows


# ---------------------------------------------------------------------------
# 2. Compare implementations — class API
# ---------------------------------------------------------------------------


def flatten_outputs(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten tensor / tuple / dict output(s) into {label: tensor}."""
    flat: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        flat[prefix or "out"] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_outputs(value, label))
    elif isinstance(obj, (tuple, list)):
        for index, value in enumerate(obj):
            label = f"{prefix}.{index}" if prefix else f"out{index}"
            flat.update(flatten_outputs(value, label))
    return flat


class KernelBench:
    """One contract, N implementations. Subclass per bench:

    required: make_inputs(), make_impls(inputs) -> {name: run closure}
    optional: baseline (reference impl name), normalize(name, raw),
              check_correctness(...), derive_metrics(name, us), notes
    use:      check() -> correctness rows, bench() -> latency/speedup rows
    """

    name: str = "bench"
    baseline: str | None = None
    atol: float = 5e-3
    rtol: float = 5e-3
    notes: dict[str, str] = {}

    def make_inputs(self) -> Any:
        """Build the canonical input set (must be deterministic/seeded)."""
        raise NotImplementedError

    def make_impls(self, inputs: Any) -> dict[str, Callable[[], Any]]:
        """Adapt inputs per impl (untimed) and return the run closures."""
        raise NotImplementedError

    def normalize(self, name: str, raw: Any) -> Any:
        """Map an impl's raw output(s) back to the canonical layout."""
        return raw

    def derive_metrics(self, name: str, latency_us: float) -> dict:
        """Extra row columns (tok/s, TFLOP/s, ...)."""
        return {}

    def check_correctness(self, name: str, outputs: Any, reference: Any) -> dict:
        """Worst-case error_stats over all output tensors."""
        refs = flatten_outputs(reference)
        outs = flatten_outputs(outputs)
        max_abs = 0.0
        cosine = 1.0
        passed = True
        for label, ref in refs.items():
            if label not in outs:
                return {"error": f"missing output '{label}'"}
            stats = error_stats(outs[label], ref, atol=self.atol, rtol=self.rtol)
            max_abs = max(max_abs, stats["max_abs"])
            cosine = min(cosine, stats["cosine"])
            passed &= stats["pass"]
        return {"max_abs": max_abs, "cosine": cosine, "pass": passed}

    def _run_once_fresh(self, name: str) -> Any:
        """Fresh inputs (in-place mutation can't leak between impls), one run."""
        inputs = self.make_inputs()
        impls = self.make_impls(inputs)
        if name not in impls:
            raise RuntimeError(f"impl {name!r} unavailable")
        return self.normalize(name, impls[name]())

    def check(self) -> list[dict]:
        """Compare every impl to the baseline on identical inputs."""
        if self.baseline is None:
            return []
        names = list(self.make_impls(self.make_inputs()))
        reference = self._run_once_fresh(self.baseline)
        rows: list[dict] = []
        for name in names:
            row: dict[str, Any] = {"impl": name}
            try:
                out = self._run_once_fresh(name)
                row.update(self.check_correctness(name, out, reference))
            except Exception as exc:
                torch.cuda.empty_cache()
                row["error"] = f"{type(exc).__name__}: {exc}"
            if self.notes.get(name):
                row.setdefault("note", self.notes[name])
            rows.append(row)
        return rows

    def bench(
        self,
        warmup: int = 10,
        iters: int = 50,
        repeats: int = 5,
        inputs: Any = None,
    ) -> list[dict]:
        """Time every impl -> rows with latency_us and speedup vs baseline."""
        inputs = self.make_inputs() if inputs is None else inputs
        return bench_impls(
            self.make_impls(inputs),
            warmup,
            iters,
            repeats,
            row_extra=self.derive_metrics,
            notes=self.notes,
            baseline=self.baseline,
        )


# ---------------------------------------------------------------------------
# 3. Present results — dict rows -> table / CSV / figure
# ---------------------------------------------------------------------------


def peak_dram_bytes_per_s() -> "float | None":
    """Theoretical peak DRAM bandwidth of device 0 (bytes/s): memory clock
    x bus width x 2 (DDR), from the CUDA device attributes. Matches the
    datasheet number (B200: 7.67e12 ~ "8 TB/s"). None if unavailable."""
    try:
        from cuda.bindings import driver as cuda_driver

        cuda_driver.cuInit(0)
        _, dev = cuda_driver.cuDeviceGet(0)
        attr = cuda_driver.CUdevice_attribute
        _, clock_khz = cuda_driver.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev
        )
        _, bus_bits = cuda_driver.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev
        )
        return clock_khz * 1e3 * (bus_bits / 8) * 2
    except Exception:
        return None


def sol_plot_kwargs(min_bytes: Callable[[float], float], xs) -> dict:
    """``plot_rows`` kwargs for the dashed SOL floor: the op's minimum DRAM
    traffic at each sweep point (``min_bytes(x)``, bytes — every input read
    once, every output written once) divided by the device's theoretical
    peak bandwidth, in us. Empty when the bandwidth query fails, so callers
    can always ``**`` it in."""
    peak = peak_dram_bytes_per_s()
    if peak is None:
        return {}
    return dict(
        sol=[(x, min_bytes(x) / peak * 1e6) for x in xs],
        sol_label=f"SOL (min bytes / {peak / 1e12:.2g} TB/s)",
    )


def print_section(title: str) -> None:
    """Print a boxed section header."""
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e5 or magnitude < 1e-3:
            return f"{value:.3e}"
        return f"{value:,.2f}" if magnitude >= 10 else f"{value:.4f}"
    return str(value)


def print_table(
    rows: list[dict],
    columns: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Print dict rows as an aligned table (columns = union of keys)."""
    if not rows:
        return
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    cells = [[_format_cell(row.get(col)) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(line[i]) for line in cells))
        for i, col in enumerate(columns)
    ]
    numeric = [
        all(
            isinstance(row.get(col), (int, float)) or row.get(col) is None
            for row in rows
        )
        for col in columns
    ]
    if title:
        print(f"\n{title}")
    header = "  ".join(
        col.rjust(w) if num else col.ljust(w)
        for col, w, num in zip(columns, widths, numeric)
    )
    print(f"  {header}")
    for line in cells:
        body = "  ".join(
            cell.rjust(w) if num else cell.ljust(w)
            for cell, w, num in zip(line, widths, numeric)
        )
        print(f"  {body}")


def write_csv(rows: list[dict], path) -> None:
    """Write dict rows to CSV (header = union of keys)."""
    import csv

    path = Path(path)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(
    rows: list[dict],
    *,
    title: str | None = None,
    columns: list[str] | None = None,
    table: bool = True,
    csv_path: "Path | str | None" = None,
    csv: bool = True,
    plot: dict | None = None,
) -> None:
    """Present one result set in a single call: print the table (unless
    ``table=False``), write the CSV (when ``csv_path`` is given and ``csv``),
    and save a figure (when ``plot`` is given: plot_rows kwargs like
    x/y/panel/suptitle; the figure lands at ``csv_path`` with a .png
    suffix, whether or not the CSV itself is written)."""
    if table:
        print_table(rows, columns=columns, title=title)
    if not rows or csv_path is None:
        return
    csv_path = Path(csv_path)
    if csv:
        write_csv(rows, csv_path)
        print(f"wrote {csv_path}")
    if plot is not None:
        figure = plot_rows(rows, csv_path.with_suffix(".png"), **plot)
        if figure is not None:
            print(f"wrote {figure}")


def plot_rows(
    rows: list[dict],
    figure_path: Path,
    x: str,
    y: str,
    series: str = "impl",
    panel: str | None = None,
    suptitle: str | None = None,
    logx: bool = True,
    logy: bool = False,
    sol: "list[tuple[float, float]] | None" = None,
    sol_label: str = "SOL",
):
    """Line figure: column `x` vs `y`, one line per `series` value, one
    subplot per `panel` value. Rows missing x or y are skipped. ``sol``
    (a list of (x, y) points) draws a dashed speed-of-light floor — e.g.
    minimum DRAM traffic / peak bandwidth — on every panel."""
    rows = [r for r in rows if r.get(x) is not None and r.get(y) is not None]
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        panels[str(r.get(panel, "")) if panel else ""].append(r)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.4), squeeze=False
    )
    for ax, (panel_label, panel_rows) in zip(axes[0], sorted(panels.items())):
        lines: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in panel_rows:
            lines[str(r.get(series))].append((float(r[x]), float(r[y])))
        for label, points in lines.items():
            points.sort()
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=label,
            )
        if sol:
            pts = sorted(sol)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=sol_label,
            )
        if logx:
            ax.set_xscale("log", base=2)
            xs = sorted({p[0] for pts in lines.values() for p in pts})
            ax.set_xticks(xs, [f"{v:g}" for v in xs])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if panel_label:
            ax.set_title(panel_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return figure_path
