#!/usr/bin/env python3
"""Timing bar chart (paper Figure 3 shape) from the artifact's timings.csv.

Grouped horizontal bars per kernel, three methods, log-scaled time axis —
matching the paper's SALT / Barvinok / Cachegrind comparison.
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["salt", "barvinok", "cachegrind"]
LABELS = {"salt": "SALT (analytic)", "barvinok": "Barvinok (symbolic)",
          "cachegrind": "Cachegrind (simulation)"}
COLORS = {"salt": "#2ca02c", "barvinok": "#ff7f0e", "cachegrind": "#1f77b4"}


def main() -> None:
    csv_path, out_path = sys.argv[1], sys.argv[2]
    data: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(open(csv_path)):
        data.setdefault(row["kernel"], {})[row["method"]] = float(row["seconds"])

    kernels = list(data)
    y = np.arange(len(kernels))
    h = 0.25
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.9 * len(kernels)))
    for i, method in enumerate(METHODS):
        vals = [max(data[k].get(method, 0.0), 1e-4) for k in kernels]
        bars = ax.barh(y + (1 - i) * h, vals, height=h,
                       label=LABELS[method], color=COLORS[method])
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.4g}s",
                    va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(kernels)
    ax.set_xscale("log")
    ax.set_xlabel("time (seconds, log scale)")
    ax.set_title("Analysis time: SALT vs Barvinok vs Cachegrind")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
