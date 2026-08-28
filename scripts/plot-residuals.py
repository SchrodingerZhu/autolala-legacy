#!/usr/bin/env python3
"""Show *where* along the cache-size axis a prediction is wrong, not just by how much.

A mean absolute error, and equally a shaded interval, collapses the whole curve
to one number. Neither can distinguish a prediction that is uniformly a little
low from one that tracks the truth over most of the range and then departs badly
across one band of cache sizes -- and those two call for completely different
work. Only a signed residual against cache size separates them.

Each column is a kernel. The upper row is the miss-ratio curves; the lower row
is `model - measured` at the same cache sizes, on a shared axis, with zero drawn.

  flat at zero        the model is right
  flat away from zero a uniform offset: the distribution is displaced, and a
                      scale factor would fix it
  sign change / lumps a curvature mismatch: the model has mass in the wrong
                      place, and no rescaling fixes it

Usage:
    scripts/plot-residuals.py --kernels matmul_128,syrk --threads 8
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
_spec = importlib.util.spec_from_file_location(
    "validate_parallel", REPO / "scripts" / "validate-parallel.py"
)
_validate = importlib.util.module_from_spec(_spec)
sys.modules["validate_parallel"] = _validate
_spec.loader.exec_module(_validate)

# Categorical slots 1-3, validated as a set.
STEADY = "#2a78d6"
COUPLED = "#eb6834"
MEASURED = "#4a3aa7"

INK = "#141a22"
INK_SOFT = "#52514e"
INK_FAINT = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d6d5d0")
    axis.tick_params(colors=INK_SOFT, labelsize=8, length=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernels", default="matmul_128,syrk,jacobi_2d,cell_shared_long")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk", default="1")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--out", default=str(REPO / "results" / "parallel"))
    args = parser.parse_args()

    for binary in (_validate.PSAMPLE, _validate.ANALYZER):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    kernels = args.kernels.split(",")
    figure, axes = plt.subplots(2, len(kernels), figsize=(4.5 * len(kernels), 7.2),
                                squeeze=False, sharex="col",
                                gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12,
                                             "wspace": 0.2})
    figure.patch.set_facecolor(SURFACE)

    for column, kernel in enumerate(kernels):
        print(f"  {kernel} T={args.threads} ...", file=sys.stderr)
        measured = _validate.sample(kernel, args.threads, args.chunk, args.block_size)
        predicted = _validate.model(kernel, args.threads, args.chunk, args.block_size)

        sizes = measured["miss_ratio_curves"]["cache_sizes"]
        reference = measured["miss_ratio_curves"]["from_reuse_interval"]
        series = {"steady": predicted["miss_ratio_curve"]}
        if "miss_ratio_curve_coupled" in predicted:
            series["coupled"] = predicted["miss_ratio_curve_coupled"]

        keep = [index for index, size in enumerate(sizes) if size >= 1.0]
        x = [sizes[index] for index in keep]
        truth = [reference[index] for index in keep]

        top, bottom = axes[0][column], axes[1][column]
        for axis in (top, bottom):
            axis.set_facecolor(SURFACE)
            style(axis)
            axis.set_xscale("log", base=2)
            axis.set_xlim(min(x), max(x))

        top.plot(x, truth, color=MEASURED, linewidth=2.6, alpha=0.45, zorder=2,
                 solid_capstyle="round")
        bottom.axhline(0.0, color=MEASURED, linewidth=1.6, alpha=0.6, zorder=2)

        worst = 0.0
        for name, colour, dash in (("steady", STEADY, "-"), ("coupled", COUPLED, (0, (5, 2)))):
            if name not in series:
                continue
            curve = series[name]
            values = [
                _validate.step_lookup(curve["turning_points"], curve["miss_ratio"], size)
                for size in x
            ]
            residual = [v - t for v, t in zip(values, truth)]
            top.plot(x, values, color=colour, linestyle=dash, linewidth=1.8, zorder=3)
            bottom.plot(x, residual, color=colour, linestyle=dash, linewidth=1.8, zorder=3)
            worst = max(worst, max(abs(r) for r in residual))

        top.set_ylim(-0.03, 1.05)
        limit = max(0.05, worst * 1.15)
        bottom.set_ylim(-limit, limit)
        top.set_title(f"{kernel}  ·  T = {args.threads}", fontsize=10.5, fontweight="bold",
                      color=INK, loc="left", pad=6)
        bottom.set_xlabel("cache size (blocks, log$_2$)", fontsize=9, color=INK_SOFT)
        if column == 0:
            top.set_ylabel("miss ratio", fontsize=9.5, color=INK_SOFT)
            bottom.set_ylabel("model − measured", fontsize=9.5, color=INK_SOFT)

        # Name the shape of the residual rather than leaving it to be eyeballed.
        # Peak size alone is not the useful fact; whether the error is spread
        # over the whole range or confined to one band decides what to fix.
        steady_curve = series["steady"]
        residual = [
            _validate.step_lookup(steady_curve["turning_points"], steady_curve["miss_ratio"], size)
            - t
            for size, t in zip(x, truth)
        ]
        threshold = 0.02
        offending = [index for index, r in enumerate(residual) if abs(r) > threshold]
        peak = max(abs(r) for r in residual)
        if not offending:
            summary = f"within {threshold:.2f} everywhere"
        else:
            width = len(offending) / len(residual)
            lo, hi = x[offending[0]], x[offending[-1]]
            positive = sum(1 for i in offending if residual[i] > 0)
            sign = ("model too high" if positive == len(offending)
                    else "model too low" if positive == 0
                    else "both signs")
            locality = "confined to" if width < 0.4 else "spread across"
            summary = (f"{locality} 2^{lo.bit_length() - 1 if isinstance(lo, int) else int(lo).bit_length() - 1}"
                       f"..2^{int(hi).bit_length() - 1}"
                       f" ({width:.0%} of range)\n{sign}")
        bottom.text(0.015, 0.05,
                    f"peak |residual| {peak:.3f}\n{summary}",
                    transform=bottom.transAxes, fontsize=7.6, color=INK_FAINT,
                    va="bottom", linespacing=1.5, zorder=4)
        # Shade the offending band so it is locatable at a glance.
        if offending:
            bottom.axvspan(x[offending[0]], x[offending[-1]], color=STEADY, alpha=0.08,
                           zorder=1)
            top.axvspan(x[offending[0]], x[offending[-1]], color=STEADY, alpha=0.08,
                        zorder=1)

    handles = [
        Line2D([0], [0], color=MEASURED, linewidth=2.6, alpha=0.45, label="measured"),
        Line2D([0], [0], color=STEADY, linewidth=1.8, label="steady state (racetrack)"),
        Line2D([0], [0], color=COUPLED, linewidth=1.8, linestyle=(0, (5, 2)),
               label="coupled (deterministic schedule)"),
    ]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                  ncol=3, frameon=False, fontsize=9.5, labelcolor=INK_SOFT)
    figure.suptitle("Where the prediction goes wrong, not just how much",
                    fontsize=15, fontweight="bold", color=INK, x=0.045, ha="left", y=0.985)
    figure.text(0.045, 0.945,
                f"schedule(static, {args.chunk})  ·  block size {args.block_size}  ·  "
                "residual is signed: flat means a displaced distribution, a sign change "
                "means mass in the wrong place",
                fontsize=9, color=INK_SOFT, ha="left", va="top")
    figure.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.075)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        path = out_dir / f"residuals-T{args.threads}.{extension}"
        figure.savefig(path, dpi=170, facecolor=SURFACE)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
