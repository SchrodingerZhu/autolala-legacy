#!/usr/bin/env python3
"""Validate the parallel CRI model per kernel category, and report per category.

Four categories:

  polybench        the constant-size PolyBench kernels in
                   analyzer/misc/polybench/const
  einsum           analyzer/misc/einsum/constant_global, lowered to affine MLIR
                   with cgeist
  fusion           the above, after `affine-loop-fusion`, keeping only kernels
                   the pass actually changes
  unroll-and-jam   the above, after `affine-loop-unroll-jam`, same rule

"Affected by" means the transformed IR differs from the untransformed IR once
both have been through the same canonicalization -- otherwise the analysis reads
the same program and the category would be padded with kernels the pass did not
really touch.

Each kernel is parallelized at its outermost loop and compared against the
sampler exactly as `validate-parallel.py` does, then aggregated by category.

Usage:
    scripts/validate-categories.py --threads 4,8 [--build-only]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_parallel", REPO / "scripts" / "validate-parallel.py"
)
_validate = importlib.util.module_from_spec(_spec)
sys.modules["validate_parallel"] = _validate
_spec.loader.exec_module(_validate)

POLYBENCH = REPO / "analyzer" / "misc" / "polybench" / "const"
EINSUM = REPO / "analyzer" / "misc" / "einsum" / "constant_global"

# Each transform is paired with the same pipeline minus the transforming pass, so
# the difference isolates the pass rather than the clean-up around it.
FUSION = "builtin.module(func.func(affine-loop-fusion,affine-loop-normalize),canonicalize)"
FUSION_REF = "builtin.module(func.func(affine-loop-normalize),canonicalize)"
UNROLL = ("builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=4}),"
          "canonicalize)")
UNROLL_REF = "builtin.module(canonicalize)"


def run_opt(pipeline: str, source: Path) -> str | None:
    result = subprocess.run(["mlir-opt", f"--pass-pipeline={pipeline}", str(source)],
                            capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.lstrip().startswith("module"):
        return None
    return result.stdout


def build_kernels(root: Path, polygeist: Path) -> dict[str, list[str]]:
    """Materializes the four categories under `root`; returns category -> names."""
    for category in ("polybench", "einsum", "fusion", "unroll-and-jam"):
        (root / category).mkdir(parents=True, exist_ok=True)

    for source in sorted(POLYBENCH.glob("const_*.mlir")):
        shutil.copy(source, root / "polybench" / f"{source.stem.removeprefix('const_')}.mlir")

    for source in sorted(EINSUM.glob("*.c")):
        lowered = subprocess.run(
            [str(polygeist / "cgeist"), str(source), "-S", "-raise-scf-to-affine"],
            capture_output=True, text=True)
        if lowered.returncode != 0:
            continue
        canonical = subprocess.run([str(polygeist / "polygeist-opt"), "--canonicalize"],
                                   input=lowered.stdout, capture_output=True, text=True)
        if canonical.returncode != 0:
            continue
        # cgeist stamps module attributes the analyzer's parser does not need.
        # Match the whole `module attributes {...} {` header: splitting on the
        # first brace would cut inside the attribute dictionary.
        text = re.sub(r"\Amodule attributes \{.*\} \{", "module {", canonical.stdout)
        (root / "einsum" / f"{source.stem}.mlir").write_text(text)

    for category, pipeline, baseline in (("fusion", FUSION, FUSION_REF),
                                        ("unroll-and-jam", UNROLL, UNROLL_REF)):
        for base in ("polybench", "einsum"):
            for source in sorted((root / base).glob("*.mlir")):
                transformed = run_opt(pipeline, source)
                reference = run_opt(baseline, source)
                if transformed is None or reference is None or transformed == reference:
                    continue
                (root / category / f"{base}_{source.stem}.mlir").write_text(transformed)

    return {
        category: sorted(path.stem for path in (root / category).glob("*.mlir"))
        for category in ("polybench", "einsum", "fusion", "unroll-and-jam")
    }


def evaluate(name: str, threads: int, chunk: str, block_size: int,
             timeout: float) -> float:
    # The comparison is against measured reuse *intervals*, so exact reuse
    # distance is never read here and its cost on the larger kernels is avoided.
    measured = _validate.sample(name, threads, chunk, block_size,
                                reuse_distance=False, timeout=timeout)
    predicted = _validate.model(name, threads, chunk, block_size, timeout=timeout)
    sizes = measured["miss_ratio_curves"]["cache_sizes"]
    reference = measured["miss_ratio_curves"]["from_reuse_interval"]
    curve = predicted["miss_ratio_curve"]
    model = [
        _validate.step_lookup(curve["turning_points"], curve["miss_ratio"], size)
        for size in sizes
    ]
    return _validate.mean_absolute_error(model, reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", default="4,8")
    parser.add_argument("--chunk", default="1")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--polygeist", default=str(Path.home() / "Documents/Polygeist/build/bin"))
    parser.add_argument("--root", default="")
    parser.add_argument("--out", default=str(REPO / "results" / "parallel"))
    parser.add_argument("--timeout", type=float, default=420.0,
                        help="seconds per tool invocation before a kernel is skipped")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else REPO / "target" / "category-kernels"
    catalogue = build_kernels(root, Path(args.polygeist))
    for category, names in catalogue.items():
        print(f"{category:<16} {len(names)} kernels", file=sys.stderr)
    if args.build_only:
        return 0

    # `sample`/`model` resolve names against MISC, so point it at the category root
    # and address kernels as `<category>/<name>`.
    _validate.MISC = root

    thread_counts = [int(value) for value in args.threads.split(",")]
    rows: list[dict] = []
    failures: dict[str, list[str]] = {}
    for category, names in catalogue.items():
        for name in names:
            for threads in thread_counts:
                label = f"{category}/{name} T={threads}"
                try:
                    error = evaluate(f"{category}/{name}", threads, args.chunk,
                                     args.block_size, args.timeout)
                except subprocess.TimeoutExpired:
                    failures.setdefault(category, []).append(
                        f"{name} T={threads}: timed out after {args.timeout:.0f}s")
                    print(f"  [slow] {label}", file=sys.stderr, flush=True)
                    continue
                except Exception as problem:  # noqa: BLE001 - collected, not raised
                    reason = str(problem).strip().splitlines()[-1][:110]
                    failures.setdefault(category, []).append(f"{name} T={threads}: {reason}")
                    print(f"  [skip] {label}", file=sys.stderr, flush=True)
                    continue
                rows.append({"category": category, "kernel": name, "threads": threads,
                             "error": error})
                print(f"  {label:<58} {error:.4f}", file=sys.stderr, flush=True)

    print("\n" + "=" * 72)
    print(f"{'category':<18}{'kernels':>9}{'runs':>7}{'mean':>10}{'median':>10}{'worst':>10}")
    print("-" * 72)
    for category in catalogue:
        errors = [row["error"] for row in rows if row["category"] == category]
        if not errors:
            print(f"{category:<18}{0:>9}{0:>7}{'-':>10}{'-':>10}{'-':>10}")
            continue
        kernels = len({row["kernel"] for row in rows if row["category"] == category})
        print(f"{category:<18}{kernels:>9}{len(errors):>7}"
              f"{statistics.mean(errors):>10.4f}{statistics.median(errors):>10.4f}"
              f"{max(errors):>10.4f}")
    if rows:
        errors = [row["error"] for row in rows]
        print("-" * 72)
        print(f"{'all':<18}{len({(r['category'], r['kernel']) for r in rows}):>9}"
              f"{len(errors):>7}{statistics.mean(errors):>10.4f}"
              f"{statistics.median(errors):>10.4f}{max(errors):>10.4f}")

    if failures:
        print("\nnot analyzable:")
        for category, reasons in failures.items():
            print(f"  {category}: {len(reasons)}")
            for reason in reasons[:4]:
                print(f"    {reason}")
            if len(reasons) > 4:
                print(f"    ... and {len(reasons) - 4} more")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "category-validation.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_dir / 'category-validation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
