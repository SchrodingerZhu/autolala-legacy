#!/usr/bin/env python3
"""Time the fully symbolic workflow, per kernel, into the same sqlite database.

Commands per kernel, pinned single-core by default or, with --cpu-threads all,
one kernel at a time with rayon free (see collect-timings.py for the modes):

  symbolic_derivation  analyzer -i sym_K.mlir --json -o sym.json barvinok
                         --block-size=8 --symbol-lowerbound=<b>... --infinite-repeat
                       One derivation, symbolic in the program parameters.
                       The lower bounds are the ones in
                       analyzer/misc/polybench/symbolic/command.txt.
  instantiation        mrc -i sym.json --symbol p0=<v> ... -c <cache bytes>
                         -b <block bytes> -a 1
                       Fixes the parameters, enumerates the pieces, runs the
                       Denning recursion and reads off the miss ratio and
                       count at one cache size: the per-query cost, with no
                       polyhedral work. Fully associative.
  instantiation_assoc  the same with -a 12: adds the set-associativity
                       conversion of the instantiated curve.

The parameter values are the lower bounds themselves; a zero bound becomes 16
and a parameter without a catalogue bound 256. Rows use category 'symbolic', threads 1 and
chunk '-', and keep the mrc output line in `detail`.

Usage:
    SYMBOLICA_LICENSE=... scripts/collect-symbolic-timings.py [--workers 48]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import sqlite3
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "collect_timings", REPO / "scripts" / "collect-timings.py"
)
_collect = importlib.util.module_from_spec(_spec)
sys.modules["collect_timings"] = _collect
_spec.loader.exec_module(_collect)

SYMBOLIC = REPO / "analyzer" / "misc" / "polybench" / "symbolic"


def read_catalogue() -> list[tuple[str, list[int]]]:
    kernels = []
    for line in (SYMBOLIC / "command.txt").read_text().splitlines():
        fields = line.split()
        if fields:
            kernels.append((fields[0], [int(v) for v in fields[1:]]))
    return kernels


def run_kernel(name: str, bounds: list[int], core: int, args) -> list[dict]:
    common = {"category": "symbolic", "kernel": name, "threads": 1, "chunk": "-",
              "block_size": args.block_size, "cpu_threads": args.cpu_threads}
    keep_dir = REPO / "target" / "symbolic"
    keep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    kept = keep_dir / f"{name}.json"
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "sym.json"
        if args.instantiation_only:
            # Re-time only the query, from the derivation kept by an earlier run.
            if not kept.exists():
                for step in ("instantiation", "instantiation_assoc"):
                    rows.append({**common, "step": step,
                                 "associativity": args.associativity, "seconds": 0.0,
                                 "status": "skipped", "detail": "no kept derivation"})
                return rows
            shutil.copyfile(kept, report)
        else:
            seconds, status, _, detail = _collect.timed([
                str(_collect.ANALYZER), "-i", str(SYMBOLIC / f"sym_{name}.mlir"),
                "--json", "-o", str(report), "barvinok", f"--block-size={args.block_size}",
                *[f"--symbol-lowerbound={b}" for b in bounds], "--infinite-repeat",
            ], core, args.timeout, args.cpu_threads)
            rows.append({**common, "step": "symbolic_derivation", "seconds": seconds,
                         "status": status, "detail": detail})
            if status != "ok":
                rows.append({**common, "step": "instantiation",
                             "associativity": args.associativity, "seconds": 0.0,
                             "status": "skipped", "detail": f"symbolic_derivation {status}"})
                return rows
            shutil.copyfile(report, kept)

        # One value per parameter the derivation actually has (the catalogue
        # may list fewer bounds than the kernel has symbols): the lower bound
        # where one is given and positive, 16 for a zero bound (a kernel
        # width, which must stay below the image size), 256 otherwise.
        params = json.loads(report.read_text())["symbolic"]["params"]
        values = []
        for i, _ in enumerate(params):
            bound = bounds[i] if i < len(bounds) else None
            values.append(bound if bound else (16 if bound == 0 else 256))
        for step, assoc in (("instantiation", 1), ("instantiation_assoc", args.associativity)):
            seconds, status, stdout, detail = _collect.timed([
                str(_collect.MRC), "-i", str(report),
                *[f"--symbol={name}={v}" for name, v in zip(params, values)],
                "-c", str(args.cache_bytes), "-b", str(args.block_bytes), "-a", str(assoc),
            ], core, args.timeout, args.cpu_threads)
            rows.append({**common, "step": step, "associativity": assoc,
                         "seconds": seconds, "status": status,
                         "detail": (stdout.strip() + " params=" + ",".join(map(str, values)))
                         if status == "ok" else detail})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--associativity", type=int, default=12)
    parser.add_argument("--cache-bytes", type=int, default=256 * 1024)
    parser.add_argument("--block-bytes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--database", default=str(REPO / "results" / "parallel" / "timings.db"))
    parser.add_argument("--only", default="", help="comma-separated kernel names")
    parser.add_argument("--instantiation-only", action="store_true",
                        help="skip the derivation; time mrc on target/symbolic/<kernel>.json")
    parser.add_argument("--cpu-threads", default="1",
                        help="1 = pinned single-core (default); 'all' = one kernel at a time "
                             "with rayon free to use every core")
    args = parser.parse_args()
    args.cpu_threads = (os.cpu_count() or 1) if args.cpu_threads == "all" else int(args.cpu_threads)
    if args.cpu_threads != 1:
        args.workers = 1

    if "SYMBOLICA_LICENSE" not in os.environ:
        print("SYMBOLICA_LICENSE is not set", file=sys.stderr)
        return 2
    for binary in (_collect.ANALYZER, _collect.MRC):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    kernels = read_catalogue()
    if args.only:
        wanted = set(args.only.split(","))
        kernels = [k for k in kernels if k[0] in wanted]

    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    _collect.migrate(connection)
    connection.executescript(_collect.SCHEMA)
    print(f"{len(kernels)} symbolic kernels, "
          + (f"{args.workers} pinned cores" if args.cpu_threads == 1
             else f"one at a time on {args.cpu_threads} cores"), file=sys.stderr)

    lock = threading.Lock()
    cores: queue.Queue[int] = queue.Queue()
    for core in range(args.first_core, args.first_core + args.workers):
        cores.put(core)

    def work(kernel: tuple[str, list[int]]) -> None:
        name, bounds = kernel
        core = cores.get()
        try:
            rows = run_kernel(name, bounds, core, args)
        finally:
            cores.put(core)
        with lock:
            for row in rows:
                connection.execute(
                    "INSERT OR REPLACE INTO timings VALUES "
                    "(:category,:kernel,:step,:threads,:chunk,:block_size,:cpu_threads,"
                    " :associativity,:seconds,:status,:detail,"
                    " :phase_derivation_s,:phase_scaling_s,:phase_mrc_s)",
                    {"associativity": None, "detail": None, "phase_derivation_s": None,
                     "phase_scaling_s": None, "phase_mrc_s": None, **row})
            connection.commit()
            summary = "  ".join(f"{r['step'][:11]}={r['seconds']:.2f}"
                                + ("" if r["status"] == "ok" else "!") for r in rows)
            print(f"  {name}  {summary}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, kernels))

    for step in ("symbolic_derivation", "instantiation", "instantiation_assoc"):
        rows = connection.execute(
            "SELECT status, seconds FROM timings WHERE category='symbolic' AND step=? "
            "AND cpu_threads=?", (step, args.cpu_threads)).fetchall()
        ok = sorted(s for status, s in rows if status == "ok")
        print(f"{step:20} n={len(rows)} ok={len(ok)} "
              + (f"median={ok[len(ok)//2]:.3f}s max={ok[-1]:.3f}s" if ok else ""),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
