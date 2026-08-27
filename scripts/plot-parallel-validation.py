#!/usr/bin/env python3
"""Plot measured against predicted miss-ratio curves for parallel loop nests.

One panel per (kernel, thread count). Each panel carries three curves:

  exact LRU            ground truth, from the sampler's measured reuse-distance
                       histogram -- no model of any kind
  Denning(measured RI) the sampler's measured reuse *intervals* pushed through
                       the same Denning recursion the analytical model uses
  CRI model            the analytical prediction

The middle curve is the one that makes the figure diagnostic rather than
decorative. Distance from it to `CRI model` is the error of the concurrent-
reuse-interval model; distance from it to `exact LRU` is the error of the
reuse-interval-to-reuse-distance conversion. A prediction can be wrong in either
place, and the two call for completely different fixes.

Usage:
    scripts/plot-parallel-validation.py                  # chunk=auto, block=1
    scripts/plot-parallel-validation.py --chunk 1 --block-size 8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

# Reuse the runners so the figure and the numeric report can never drift.
_spec = importlib.util.spec_from_file_location(
    "validate_parallel", REPO / "scripts" / "validate-parallel.py"
)
_validate = importlib.util.module_from_spec(_spec)
# Register before executing: the module defines a dataclass, and dataclasses
# resolves field types through `sys.modules[cls.__module__]`, which is absent for
# a module loaded straight off a path.
sys.modules["validate_parallel"] = _validate
_spec.loader.exec_module(_validate)

# Categorical slots 1-3 of the reference palette. Validated as a set for
# all-pairs separation and >= 3:1 contrast on a light surface; dash patterns
# carry the same distinction again so the figure survives print and CVD.
SERIES = [
    ("exact LRU (measured reuse distance)", "#2a78d6", "-", 2.0),
    ("Denning (measured reuse interval)", "#eb6834", (0, (5, 2)), 1.8),
    ("CRI model (prediction)", "#4a3aa7", (0, (1.5, 1.5)), 1.8),
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"


def collect(kernel: str, threads: int, chunk: str, block_size: int) -> dict:
    """Runs both tools and returns everything the panel needs."""
    measured = _validate.sample(kernel, threads, chunk, block_size)
    predicted = _validate.model(kernel, threads, chunk, block_size)

    curves = measured["miss_ratio_curves"]
    sizes = curves["cache_sizes"]
    curve = predicted["miss_ratio_curve"]
    model_curve = [
        _validate.step_lookup(curve["turning_points"], curve["miss_ratio"], size)
        for size in sizes
    ]
    parallel = predicted["parallel"]

    return {
        "sizes": sizes,
        "exact": curves["exact"],
        "from_ri": curves["from_reuse_interval"],
        "model": model_curve,
        "total_accesses": measured["total_accesses"],
        "cold_fraction": measured["cold_accesses"] / measured["total_accesses"],
        "resolved_chunk": parallel["chunk"],
        "sharing_degree": parallel["mean_sharing_degree"],
        "shared_portion": parallel["shared_portion"],
        "model_vs_ri": _validate.mean_absolute_error(model_curve, curves["from_reuse_interval"]),
        "conversion": _validate.mean_absolute_error(curves["from_reuse_interval"], curves["exact"]),
        "model_vs_exact": _validate.mean_absolute_error(model_curve, curves["exact"]),
    }


def draw_panel(axis, panel: dict, kernel: str, threads: int) -> None:
    sizes = panel["sizes"]
    # A cache of zero blocks is not meaningful on a log axis, and the curve is
    # flat at 1.0 there anyway.
    keep = [index for index, size in enumerate(sizes) if size >= 1.0]
    x = [sizes[index] for index in keep]

    for (label, color, dash, width), key in zip(SERIES, ("exact", "from_ri", "model")):
        values = panel[key]
        if values is None:
            continue
        y = [values[index] for index in keep]
        # No direct labels: the three curves coincide wherever the model is
        # right, which is most of the figure, so any label sits on top of the
        # others. The palette clears 3:1 contrast on this surface, so the
        # legend plus the dash patterns carry identity on their own.
        axis.plot(x, y, color=color, linestyle=dash, linewidth=width, label=label,
                  solid_capstyle="round", zorder=3)

    axis.set_xscale("log", base=2)
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlim(min(x), max(x))
    axis.grid(True, which="major", color="#e6e5e1", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d6d5d0")
    axis.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=3)

    axis.set_title(
        f"{kernel}   ·   T = {threads}",
        fontsize=9.5,
        fontweight="bold",
        color=INK_PRIMARY,
        loc="left",
        pad=6,
    )

    # Everything needed to reproduce and to judge the panel, in the panel.
    detail = (
        f"chunk {panel['resolved_chunk']}   accesses {panel['total_accesses']:,}\n"
        f"cold {panel['cold_fraction']:.3f}   shared reuse {panel['shared_portion']:.3f}\n"
        f"sharing degree {panel['sharing_degree']:.2f} of {threads}"
    )
    axis.text(
        0.015, 0.03, detail, transform=axis.transAxes, fontsize=6.6,
        color=INK_MUTED, va="bottom", ha="left", linespacing=1.5, zorder=4,
    )

    errors = (
        f"MAE model vs measured RI   {panel['model_vs_ri']:.4f}\n"
        f"MAE conversion              {panel['conversion']:.4f}\n"
        f"MAE model vs exact          {panel['model_vs_exact']:.4f}"
    )
    axis.text(
        0.985, 0.97, errors, transform=axis.transAxes, fontsize=6.6,
        color=INK_SECONDARY, va="top", ha="right", family="monospace",
        linespacing=1.5, zorder=4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                  edgecolor="#e6e5e1", linewidth=0.6),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernels", default=",".join(_validate.KERNELS))
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument("--chunk", default="auto",
                        help="`auto` is OpenMP's default schedule(static)")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--out", default=str(REPO / "results" / "parallel"))
    args = parser.parse_args()

    for binary in (_validate.PSAMPLE, _validate.ANALYZER):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    kernels = args.kernels.split(",")
    thread_counts = [int(value) for value in args.threads.split(",")]

    panels: dict[tuple[str, int], dict] = {}
    for kernel in kernels:
        for threads in thread_counts:
            print(f"  running {kernel} T={threads} ...", file=sys.stderr)
            panels[(kernel, threads)] = collect(kernel, threads, args.chunk, args.block_size)

    rows, cols = len(kernels), len(thread_counts)
    figure, axes = plt.subplots(rows, cols, figsize=(4.9 * cols, 3.15 * rows),
                                squeeze=False)
    figure.patch.set_facecolor(SURFACE)

    for row, kernel in enumerate(kernels):
        for col, threads in enumerate(thread_counts):
            axis = axes[row][col]
            axis.set_facecolor(SURFACE)
            draw_panel(axis, panels[(kernel, threads)], kernel, threads)
            if col == 0:
                axis.set_ylabel("miss ratio", fontsize=8.5, color=INK_SECONDARY)
            if row == rows - 1:
                axis.set_xlabel("cache size (blocks, log$_2$)", fontsize=8.5,
                                color=INK_SECONDARY)

    overall = [panel["model_vs_ri"] for panel in panels.values()]
    end_to_end = [panel["model_vs_exact"] for panel in panels.values()]
    figure.suptitle(
        "Parallel miss-ratio curves: measured against predicted",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, x=0.012, ha="left", y=0.995,
    )
    figure.text(
        0.012, 0.978,
        f"schedule(static, {args.chunk})   ·   block size {args.block_size} element(s)   ·   "
        f"fully-associative LRU   ·   uniform interleaving, seed 1   ·   "
        f"{len(panels)} configurations   ·   "
        f"mean MAE {sum(overall) / len(overall):.4f} (CRI model), "
        f"{sum(end_to_end) / len(end_to_end):.4f} (end to end)",
        fontsize=8.8, color=INK_SECONDARY, ha="left", va="top",
    )

    handles = [Line2D([0], [0], color=color, linestyle=dash, linewidth=width, label=label)
               for label, color, dash, width in SERIES]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.999),
                  ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.subplots_adjust(hspace=0.42, wspace=0.2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mrc-comparison-chunk-{args.chunk}-block-{args.block_size}"
    for extension in ("pdf", "png", "svg"):
        path = out_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=170, facecolor=SURFACE)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
