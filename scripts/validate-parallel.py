#!/usr/bin/env python3
"""Validate the parallel CRI model against the reproducible sampler.

Runs `psample` (ground truth) and `analyzer ... --parallel-loop-depth` (model)
over a matrix of kernels, thread counts, chunk sizes and block sizes, and
reports where they disagree.

Three errors are reported per configuration, because a prediction can go wrong
in three independent places:

  model_vs_sampler_ri   the CRI model itself -- analytical miss-ratio curve
                        against the one the Denning recursion produces from the
                        *measured* reuse intervals.
  conversion            the reuse-interval-to-reuse-distance conversion --
                        measured intervals through Denning against the exact
                        LRU curve from measured reuse distances.
  model_vs_exact        the two combined, i.e. what a user actually sees.

Splitting them matters: a large `model_vs_exact` with a small
`model_vs_sampler_ri` means the CRI model is fine and the Denning conversion is
the problem, and no amount of work on Table 1 would help.

A fourth number, `reuse_mass_gap`, guards the step before any of them. It is the
difference between the fraction of accesses the model finds a reuse for and the
fraction the sampler measures. The CRI model can only redistribute the reuse it
is handed, so a large gap here means the *reuse-interval extraction* lost mass
before the model ran, and the three error columns would be measuring the wrong
thing. It should stay near zero; it exists to keep an extraction bug from being
misread as a modeling error, which is exactly what happened once during
development (a duplicated polyhedral dimension name silently dropped all
intra-block spatial reuse at `--block-size 8`).

Usage:
    scripts/validate-parallel.py [--threads 2,4,8] [--out results/parallel]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISC = REPO / "parallel-sampler" / "misc"
PSAMPLE = REPO / "target" / "release" / "psample"
ANALYZER = REPO / "target" / "release" / "analyzer"

# Every kernel parallelizes its outermost loop.
KERNELS = [
    "cell_private_long",
    "cell_shared_long",
    "cell_short",
    "matmul_128",
    "jacobi_2d",
    "syrk",
]


@dataclass
class Row:
    kernel: str
    threads: int
    chunk: str
    block_size: int
    total_accesses: int
    sampler_shared_fraction: float
    model_shared_fraction: float
    model_vs_sampler_ri: float
    conversion: float | None
    model_vs_exact: float | None
    reuse_mass_gap: float
    digest: str


def run(cmd: list[str], timeout: float | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-2000:]}"
        )
    return result.stdout


def sample(kernel: str, threads: int, chunk: str, block_size: int,
           reuse_distance: bool = True, timeout: float | None = None) -> dict:
    cmd = [
        str(PSAMPLE),
        "--input", str(MISC / f"{kernel}.mlir"),
        "--parallel-loop-depth", "0",
        "--threads", str(threads),
        "--block-size", str(block_size),
        "--seed", "1",
    ]
    # Exact reuse distance costs memory proportional to the working set and
    # feeds only the `exact` curve; callers comparing against measured reuse
    # *intervals* can skip it.
    if not reuse_distance:
        cmd.append("--no-reuse-distance")
    if chunk != "auto":
        cmd += ["--chunk", chunk]
    return json.loads(run(cmd, timeout=timeout))


def model(kernel: str, threads: int, chunk: str, block_size: int,
          dilation: str = "hybrid", timeout: float | None = None) -> dict:
    cmd = [
        str(ANALYZER),
        "--input", str(MISC / f"{kernel}.mlir"),
        "--json",
        "barvinok",
        # The repository's own prediction scripts run barvinok this way, and on
        # the larger PolyBench kernels it is the difference between two seconds
        # and not terminating.
        "--barvinok-arg=--approximation-method=scale",
        "--parallel-loop-depth", "0",
        "--threads", str(threads),
        "--block-size", str(block_size),
        "--dilation", dilation,
    ]
    if chunk != "auto":
        cmd += ["--chunk", chunk]
    # The analyzer prints a licence banner on stderr and the JSON document as
    # the final stdout line.
    return json.loads(run(cmd, timeout=timeout).strip().splitlines()[-1])


def model_scaled(base_report: Path, threads: int, chunk: str) -> dict:
    """Re-applies the CRI laws at `threads` from a report derived at another
    thread count, doing no polyhedral work.

    This is the parametric-in-T path, and it assumes each shared datum is
    touched by all T threads -- what PLUSS assumes. Its gap to a full
    re-derivation is what that assumption costs.
    """
    cmd = [
        str(ANALYZER), "--json", "barvinok",
        "--threads", str(threads),
        "--scale-from", str(base_report),
    ]
    if chunk != "auto":
        cmd += ["--chunk", chunk]
    return json.loads(run(cmd, timeout=timeout).strip().splitlines()[-1])


def step_lookup(turning_points: list[float], miss_ratio: list[float], size: float) -> float:
    """Miss ratio at `size`, mirroring `denning::MissRatioCurve::miss_ratio_at`."""
    chosen = 1.0
    for point, ratio in zip(turning_points, miss_ratio):
        if point <= size:
            chosen = ratio
        else:
            break
    return chosen


def mean_absolute_error(left: list[float], right: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return float("nan")
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def evaluate(kernel: str, threads: int, chunk: str, block_size: int) -> Row:
    measured = sample(kernel, threads, chunk, block_size)
    predicted = model(kernel, threads, chunk, block_size)

    curves = measured["miss_ratio_curves"]
    sizes = curves["cache_sizes"]
    from_ri = curves["from_reuse_interval"]
    exact = curves["exact"]

    curve = predicted["miss_ratio_curve"]
    model_curve = [
        step_lookup(curve["turning_points"], curve["miss_ratio"], size) for size in sizes
    ]

    joint_total = sum(row["count"] for row in measured["joint"]) or 1
    joint_shared = sum(
        row["count"] for row in measured["joint"] if row["sharing"] == "shared"
    )
    parallel = predicted["parallel"]
    reuse_mass = parallel["shared_portion"] + parallel["private_portion"]
    measured_reuse_mass = 1.0 - measured["cold_accesses"] / measured["total_accesses"]

    return Row(
        kernel=kernel,
        threads=threads,
        chunk=chunk,
        block_size=block_size,
        total_accesses=measured["total_accesses"],
        sampler_shared_fraction=joint_shared / joint_total,
        model_shared_fraction=parallel["shared_portion"] / reuse_mass if reuse_mass else 0.0,
        model_vs_sampler_ri=mean_absolute_error(model_curve, from_ri),
        conversion=mean_absolute_error(from_ri, exact) if exact else None,
        model_vs_exact=mean_absolute_error(model_curve, exact) if exact else None,
        reuse_mass_gap=abs(reuse_mass - measured_reuse_mass),
        digest=measured["digest"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument(
        "--chunks",
        default="auto,1",
        help="chunk sizes; `auto` is OpenMP's default schedule(static)",
    )
    parser.add_argument("--block-sizes", default="1,8")
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--out", default=str(REPO / "results" / "parallel"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="fail if any configuration's model_vs_sampler_ri exceeds this",
    )
    args = parser.parse_args()

    for binary in (PSAMPLE, ANALYZER):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    thread_counts = [int(value) for value in args.threads.split(",")]
    chunks = args.chunks.split(",")
    block_sizes = [int(value) for value in args.block_sizes.split(",")]
    kernels = args.kernels.split(",")

    rows: list[Row] = []
    failures: list[str] = []
    for kernel in kernels:
        for threads in thread_counts:
            for chunk in chunks:
                for block_size in block_sizes:
                    label = f"{kernel} T={threads} chunk={chunk} B={block_size}"
                    try:
                        row = evaluate(kernel, threads, chunk, block_size)
                    except Exception as error:  # noqa: BLE001 - reported, not raised
                        print(f"[skip] {label}: {error}", file=sys.stderr)
                        failures.append(label)
                        continue
                    rows.append(row)
                    flag = " EXTRACTION" if row.reuse_mass_gap > 0.02 else ""
                    print(
                        f"{label:<44} model-vs-RI {row.model_vs_sampler_ri:.4f}"
                        f"  conversion {row.conversion if row.conversion is None else format(row.conversion, '.4f')}"
                        f"  model-vs-exact {row.model_vs_exact if row.model_vs_exact is None else format(row.model_vs_exact, '.4f')}"
                        f"  mass-gap {row.reuse_mass_gap:.4f}{flag}"
                    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2)
    )

    if not rows:
        print("no configuration produced a result", file=sys.stderr)
        return 1

    worst = max(rows, key=lambda row: row.model_vs_sampler_ri)
    mean = sum(row.model_vs_sampler_ri for row in rows) / len(rows)
    print(
        f"\n{len(rows)} configuration(s): mean model-vs-RI error {mean:.4f}, "
        f"worst {worst.model_vs_sampler_ri:.4f} on {worst.kernel} T={worst.threads} "
        f"chunk={worst.chunk} B={worst.block_size}"
    )
    print(f"wrote {out_dir / 'validation.json'}")

    # Configurations whose reuse extraction lost mass are reported separately:
    # the CRI model never saw that reuse, so counting them against it would be
    # misattribution.
    blocked = [row for row in rows if row.reuse_mass_gap > 0.02]
    if blocked:
        print(
            f"\n{len(blocked)} configuration(s) lost reuse mass before the model ran, "
            f"so their error columns do not measure the CRI model:"
        )
        for row in blocked:
            print(
                f"  {row.kernel} T={row.threads} chunk={row.chunk} B={row.block_size}: "
                f"gap {row.reuse_mass_gap:.4f}"
            )
        sound = [row for row in rows if row.reuse_mass_gap <= 0.02]
        if sound:
            mean_sound = sum(row.model_vs_sampler_ri for row in sound) / len(sound)
            print(
                f"  over the {len(sound)} configuration(s) with sound extraction, "
                f"mean model-vs-RI error is {mean_sound:.4f}"
            )

    over = [
        row
        for row in rows
        if row.model_vs_sampler_ri > args.tolerance and row.reuse_mass_gap <= 0.02
    ]
    if over:
        print(f"\n{len(over)} configuration(s) exceed the {args.tolerance} tolerance:")
        for row in over:
            print(
                f"  {row.kernel} T={row.threads} chunk={row.chunk} B={row.block_size}: "
                f"{row.model_vs_sampler_ri:.4f}"
            )
    if failures:
        print(f"\n{len(failures)} configuration(s) failed to run:")
        for label in failures:
            print(f"  {label}")
    return 1 if over or failures else 0


if __name__ == "__main__":
    sys.exit(main())
