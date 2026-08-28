#!/usr/bin/env python3
"""Compare the three laws for a dilated reuse window, so the default can be chosen on evidence.

A reuse window that no other thread intercepts simply stretches. Three ways to
say by how much, all with mean `T*r`:

  nbd     `r + NB(r, 1/T)`, expanded term by term. Exact, but its support grows
          with `r`, so past the Theorem 3.1 bound it is abandoned for a point
          mass at `T*r` and the spread is lost.
  gamma   `r + Gamma(shape = r*(1 - 1/T), scale = T)`, the same law in
          continuous form, matching both moments. Fixed cost at any `r`.
  hybrid  the negative binomial below the bound, the Gamma above it.

The top panel is the decision: mean error per kernel under each law. The bottom
panels are why -- the two kernels where the laws actually diverge, drawn against
the measured curve. Everywhere else the choice is immaterial, which is itself
worth seeing.

Usage:
    scripts/plot-dilation-comparison.py
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_parallel", REPO / "scripts" / "validate-parallel.py"
)
_validate = importlib.util.module_from_spec(_spec)
sys.modules["validate_parallel"] = _validate
_spec.loader.exec_module(_validate)

# Categorical slots 1-3 of the reference palette: validated as a set for
# all-pairs separation and >= 3:1 contrast on this surface.
LAWS = [
    ("nbd", "#2a78d6", "-"),
    ("gamma", "#eb6834", (0, (5, 2))),
    ("hybrid", "#4a3aa7", (0, (1.5, 1.5))),
]
MEASURED = "#52514e"

INK = "#141a22"
INK_SOFT = "#52514e"
INK_FAINT = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

KERNELS = [
    "cell_private_long",
    "cell_short",
    "cell_shared_long",
    "matmul_128",
    "jacobi_2d",
    "syrk",
]
# The two kernels whose curves actually separate, and a configuration of each
# where the separation is visible.
DETAIL = [("cell_private_long", 8, 1), ("cell_short", 4, 1)]


def curves(kernel: str, threads: int, block_size: int) -> tuple[list, list, dict]:
    measured = _validate.sample(kernel, threads, "1", block_size)
    sizes = measured["miss_ratio_curves"]["cache_sizes"]
    reference = measured["miss_ratio_curves"]["from_reuse_interval"]
    predicted = {}
    for law, _, _ in LAWS:
        report = _validate.model(kernel, threads, "1", block_size, law)["miss_ratio_curve"]
        predicted[law] = [
            _validate.step_lookup(report["turning_points"], report["miss_ratio"], size)
            for size in sizes
        ]
    return sizes, reference, predicted


def style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d6d5d0")
    axis.tick_params(colors=INK_SOFT, labelsize=8.5, length=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", default="4,8,16")
    parser.add_argument("--block-sizes", default="1,8")
    parser.add_argument("--out", default=str(REPO / "results" / "parallel"))
    args = parser.parse_args()

    for binary in (_validate.PSAMPLE, _validate.ANALYZER):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    thread_counts = [int(v) for v in args.threads.split(",")]
    block_sizes = [int(v) for v in args.block_sizes.split(",")]

    errors: dict[str, dict[str, list[float]]] = {k: {law: [] for law, _, _ in LAWS} for k in KERNELS}
    for kernel in KERNELS:
        for threads in thread_counts:
            for block_size in block_sizes:
                print(f"  {kernel} T={threads} B={block_size} ...", file=sys.stderr)
                sizes, reference, predicted = curves(kernel, threads, block_size)
                for law, _, _ in LAWS:
                    errors[kernel][law].append(
                        _validate.mean_absolute_error(predicted[law], reference)
                    )

    figure = plt.figure(figsize=(15.5, 9.4))
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.16,
                               left=0.06, right=0.985, top=0.86, bottom=0.08)

    # ---- the decision: mean error per kernel, per law ----
    bars = figure.add_subplot(grid[0, :])
    bars.set_facecolor(SURFACE)
    groups = KERNELS + ["ALL KERNELS"]
    width = 0.26
    for index, (law, color, _) in enumerate(LAWS):
        heights = [statistics.mean(errors[k][law]) for k in KERNELS]
        heights.append(statistics.mean([e for k in KERNELS for e in errors[k][law]]))
        offsets = [i + (index - 1) * width for i in range(len(groups))]
        bars.bar(offsets, heights, width * 0.88, color=color, label=law, zorder=3)
        for x, h in zip(offsets, heights):
            bars.annotate(f"{h:.4f}", (x, h), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=7.2, color=INK_SOFT, zorder=4)
    best = {k: min(LAWS, key=lambda law: statistics.mean(errors[k][law[0]]))[0] for k in KERNELS}
    bars.set_xticks(range(len(groups)))
    bars.set_xticklabels(
        [f"{g}\n(best: {best[g]})" if g in best else f"{g}" for g in groups],
        fontsize=9, color=INK,
    )
    bars.set_ylabel("mean miss-ratio error vs measured reuse intervals", fontsize=9.5,
                    color=INK_SOFT)
    style(bars)
    bars.legend(frameon=False, fontsize=10, labelcolor=INK_SOFT, ncol=3,
                loc="upper left", title=None)
    bars.set_title(
        "Lower is better. The three laws are indistinguishable on four of six kernels; "
        "the choice is decided by the two on the left.",
        fontsize=10, color=INK_SOFT, loc="left", pad=10,
    )

    # ---- why: the two kernels that separate ----
    for column, (kernel, threads, block_size) in enumerate(DETAIL):
        axis = figure.add_subplot(grid[1, column])
        axis.set_facecolor(SURFACE)
        sizes, reference, predicted = curves(kernel, threads, block_size)
        keep = [i for i, s in enumerate(sizes) if s >= 1.0]
        x = [sizes[i] for i in keep]
        axis.plot(x, [reference[i] for i in keep], color=MEASURED, linewidth=3.0,
                  alpha=0.35, label="measured", zorder=2, solid_capstyle="round")
        for law, color, dash in LAWS:
            axis.plot(x, [predicted[law][i] for i in keep], color=color, linestyle=dash,
                      linewidth=1.8, label=law, zorder=3)
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.03, 1.05)
        axis.set_xlim(min(x), max(x))
        style(axis)
        axis.set_xlabel("cache size (blocks, log$_2$)", fontsize=9, color=INK_SOFT)
        if column == 0:
            axis.set_ylabel("miss ratio", fontsize=9.5, color=INK_SOFT)
        note = {
            "cell_private_long": "long private reuse: the negative binomial gives up\n"
                                 "past the bound and collapses to a point mass",
            "cell_short": "very short reuse: the window holds so few accesses\n"
                          "that the distribution's discreteness still shows",
        }[kernel]
        axis.set_title(f"{kernel}  ·  T = {threads}, block {block_size}", fontsize=10,
                       fontweight="bold", color=INK, loc="left", pad=6)
        axis.text(0.015, 0.06, note, transform=axis.transAxes, fontsize=8,
                  color=INK_FAINT, va="bottom", linespacing=1.5, zorder=4)
        rows = "\n".join(
            f"{law:<7}{_validate.mean_absolute_error(predicted[law], reference):.4f}"
            for law, _, _ in LAWS
        )
        axis.text(0.985, 0.96, rows, transform=axis.transAxes, fontsize=8,
                  family="monospace", color=INK_SOFT, va="top", ha="right",
                  linespacing=1.6, zorder=4,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                            edgecolor=GRID, linewidth=0.6))
        if column == 0:
            handles = [Line2D([0], [0], color=MEASURED, linewidth=3.0, alpha=0.35,
                              label="measured")]
            handles += [Line2D([0], [0], color=c, linestyle=d, linewidth=1.8, label=n)
                        for n, c, d in LAWS]
            axis.legend(handles=handles, frameon=False, fontsize=8.5,
                        labelcolor=INK_SOFT, loc="center left")

    figure.suptitle("Which law for a dilated reuse window?", fontsize=17, fontweight="bold",
                    color=INK, x=0.06, ha="left", y=0.975)
    figure.text(
        0.06, 0.925,
        f"schedule(static, 1)  ·  T in {{{args.threads}}}  ·  block sizes {args.block_sizes}  ·  "
        f"{len(thread_counts) * len(block_sizes)} configurations per kernel  ·  "
        "all three laws have mean T·r; they differ in how the spread is carried",
        fontsize=9.5, color=INK_SOFT, ha="left", va="top",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        path = out_dir / f"dilation-law-comparison.{extension}"
        figure.savefig(path, dpi=170, facecolor=SURFACE)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
