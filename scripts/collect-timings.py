#!/usr/bin/env python3
"""Time the three commands of the parallel prediction workflow, per kernel, into sqlite.

The workflow is the one in `run-single-prediction.sh`, with parallelism added
the way commit a4f6a82 added it -- each stage its own command:

  derivation       analyzer -i K --json -o base.json barvinok
                     --barvinok-arg=--approximation-method=scale
                     -P 0 -T <base> -C <chunk> -b <block>
                   The symbolic derivation; varies by orders of magnitude across
                   kernels. The analyzer also reports its internal split, so the
                   Denning step (distribution to miss-ratio curve, which has no
                   standalone command) is recorded from that.
  thread_scaling   analyzer --json barvinok -T <T> -C <chunk> --scale-from base.json
                   Re-applies the CRI laws at another thread count with no
                   polyhedral work.
  associativity    assoc-conv -i base.json -o assoc.json -a 12
                   The 12-way conversion of the base curve.
  prediction       mrc -i assoc.json -c <cache bytes> -b <block bytes>
                   Loads the converted curve and reads off the miss ratio and
                   miss count at one cache size: the per-query cost once the
                   curve exists.

Two measurement modes, told apart by the `cpu_threads` column:

  --cpu-threads 1 (default)  every command is pinned to its own core with
        `taskset`, run at nice 19, with RAYON_NUM_THREADS=1 so the expansion
        and `compute_assoc`, both rayon-parallel, are timed sequentially. Only
        the collection runs 48-wide; no measurement ever shares a core.
  --cpu-threads all          one kernel at a time, unpinned, rayon free to use
        every core: the wall time of the workflow with nothing else running.

Runs that do not finish are kept with status 'timeout' and the budget as their
time, so the censoring is visible to whoever plots the distribution.

Requires SYMBOLICA_LICENSE in the environment: unlicensed Symbolica permits one
instance per machine, which rules out concurrent collection.

Usage:
    SYMBOLICA_LICENSE=... scripts/collect-timings.py [--scale-to 8,64] [--workers 48]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_categories", REPO / "scripts" / "validate-categories.py"
)
_categories = importlib.util.module_from_spec(_spec)
sys.modules["validate_categories"] = _categories
_spec.loader.exec_module(_categories)
ANALYZER = _categories._validate.ANALYZER
ASSOC_CONV = REPO / "target" / "release" / "assoc-conv"
MRC = REPO / "target" / "release" / "mrc"

SCHEMA = """
CREATE TABLE IF NOT EXISTS timings (
    category      TEXT    NOT NULL,
    kernel        TEXT    NOT NULL,
    step          TEXT    NOT NULL,   -- 'derivation' | 'thread_scaling' | 'associativity' | 'prediction'
    threads       INTEGER NOT NULL,   -- base T for derivation/associativity, target T for scaling
    chunk         TEXT    NOT NULL,
    block_size    INTEGER NOT NULL,
    cpu_threads   INTEGER NOT NULL,   -- cores the command was allowed: 1 = pinned sequential
    associativity INTEGER,            -- associativity rows only
    seconds       REAL    NOT NULL,   -- wall clock of the command; the budget on timeout
    status        TEXT    NOT NULL,   -- 'ok' | 'timeout' | 'error' | 'skipped'
    detail        TEXT,               -- error text; on prediction rows the mrc output line
                                      -- "<file>,<miss count>,<total accesses>,<miss ratio>,<cache bytes>"
    -- the analyzer's own split of the derivation command, ok rows only
    phase_derivation_s REAL,
    phase_scaling_s    REAL,
    phase_mrc_s        REAL,
    PRIMARY KEY (category, kernel, step, threads, chunk, block_size, cpu_threads)
);
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Adds `cpu_threads` to a database written before it existed; those rows
    were all pinned single-core, so they become cpu_threads = 1."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(timings)")}
    if not columns or "cpu_threads" in columns:
        return
    connection.executescript("ALTER TABLE timings RENAME TO timings_v1;")
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT INTO timings SELECT category, kernel, step, threads, chunk, block_size, 1, "
        "associativity, seconds, status, detail, phase_derivation_s, phase_scaling_s, "
        "phase_mrc_s FROM timings_v1")
    connection.executescript("DROP TABLE timings_v1;")
    connection.commit()


def timed(command: list[str], core: int, budget: float,
          cpu_threads: int) -> tuple[float, str, str, str | None]:
    """Runs one command: (seconds, status, stdout, detail).

    With `cpu_threads == 1` it is pinned to `core`, niced, and rayon is held to
    one thread so every phase is measured sequentially. With any other value
    it runs unpinned with rayon free to use the machine -- the collector then
    serializes kernels so nothing else competes.
    """
    env = dict(os.environ, SYMBOLICA_HIDE_BANNER="1")
    if cpu_threads == 1:
        wrapped = ["taskset", "-c", str(core), "nice", "-n", "19", *command]
        env["RAYON_NUM_THREADS"] = "1"
    else:
        wrapped = list(command)
        env.pop("RAYON_NUM_THREADS", None)
    started = time.monotonic()
    try:
        result = subprocess.run(wrapped, capture_output=True, text=True, timeout=budget, env=env)
    except subprocess.TimeoutExpired:
        return budget, "timeout", "", None
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        return elapsed, "error", "", (lines[-1][:200] if lines else f"exit {result.returncode}")
    return elapsed, "ok", result.stdout, None


def run_kernel(job: dict, core: int, args) -> list[dict]:
    """The three commands for one kernel, in workflow order; later ones need base.json."""
    common = {"category": job["category"], "kernel": job["kernel"], "chunk": args.chunk,
              "block_size": args.block_size, "cpu_threads": args.cpu_threads}
    # The base derivation lives in tmpfs while it is timed: the report can run
    # to hundreds of megabytes, and a disk write inside the measured command
    # would be charged to the derivation. It is copied out afterwards so
    # scaling and associativity can be re-timed without re-deriving.
    keep_dir = REPO / "target" / "category-kernels" / "base"
    keep_dir.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.TemporaryDirectory()
    base = Path(scratch.name) / "base.json"
    try:
        rows = timed_workflow(job, core, args, base, common)
    finally:
        if base.exists() and not args.prediction_only:
            shutil.copyfile(base, keep_dir / f"{job['category']}__{job['kernel']}__T{args.base_threads}.json")
        scratch.cleanup()
    return rows


def timed_workflow(job: dict, core: int, args, base: Path, common: dict) -> list[dict]:
    rows: list[dict] = []
    if args.prediction_only:
        # Only the query is timed: the kept base derivation is converted
        # untimed (unpinned, rayon free), then `mrc` is timed as usual.
        kept = REPO / "target" / "category-kernels" / "base" / \
            f"{job['category']}__{job['kernel']}__T{args.base_threads}.json"
        if not kept.exists():
            return [{**common, "step": "prediction", "threads": args.base_threads,
                     "associativity": args.associativity, "seconds": 0.0,
                     "status": "skipped", "detail": "no kept base derivation"}]
        assoc = base.with_name("assoc.json")
        # Untimed, so rayon may run -- but on the cores the measurements do
        # not use: 48 unpinned conversions would starve the pinned, niced
        # `mrc` runs, and single-threaded ones take up to 800 s each.
        spare = [c for c in range(os.cpu_count() or 1)
                 if not args.first_core <= c < args.first_core + args.workers]
        confine = (["taskset", "-c", ",".join(map(str, spare))] if spare else [])
        converted = subprocess.run(
            [*confine, str(ASSOC_CONV), "-i", str(kept), "-o", str(assoc),
             "-a", str(args.associativity)],
            capture_output=True,
            env=dict(os.environ, RAYON_NUM_THREADS=str(max(1, len(spare) // 8))))
        if converted.returncode != 0:
            # A base left behind by a censored derivation is truncated.
            return [{**common, "step": "prediction", "threads": args.base_threads,
                     "associativity": args.associativity, "seconds": 0.0,
                     "status": "skipped", "detail": "kept base derivation unusable"}]
        seconds, status, stdout, detail = timed([
            str(MRC), "-i", str(assoc), "-c", str(args.cache_bytes), "-b", str(args.block_bytes),
        ], core, args.timeout, args.cpu_threads)
        return [{**common, "step": "prediction", "threads": args.base_threads,
                 "associativity": args.associativity, "seconds": seconds, "status": status,
                 "detail": stdout.strip() if status == "ok" else detail}]

    seconds, status, _, detail = timed([
        str(ANALYZER), "-i", job["source"], "--json", "-o", str(base), "barvinok",
        "--barvinok-arg=--approximation-method=scale",
        "-P", "0", "-T", str(args.base_threads), "-C", args.chunk,
        "-b", str(args.block_size),
    ], core, args.timeout, args.cpu_threads)
    derivation = {**common, "step": "derivation", "threads": args.base_threads,
                  "seconds": seconds, "status": status, "detail": detail}
    if status == "ok":
        try:
            phases = json.loads(base.read_text())["timings"]
            derivation.update(phase_derivation_s=phases["derivation"],
                              phase_scaling_s=phases["parallel_scaling"],
                              phase_mrc_s=phases["mrc"])
        except (ValueError, KeyError, OSError) as problem:
            derivation.update(status="error", detail=f"unreadable base.json: {problem}")
    rows.append(derivation)

    if derivation["status"] != "ok":
        reason = f"derivation {derivation['status']}"
        for target in args.scale_to:
            rows.append({**common, "step": "thread_scaling", "threads": target,
                         "seconds": 0.0, "status": "skipped", "detail": reason})
        rows.append({**common, "step": "associativity", "threads": args.base_threads,
                     "associativity": args.associativity, "seconds": 0.0,
                     "status": "skipped", "detail": reason})
        rows.append({**common, "step": "prediction", "threads": args.base_threads,
                     "associativity": args.associativity, "seconds": 0.0,
                     "status": "skipped", "detail": reason})
        return rows

    for target in args.scale_to:
        seconds, status, _, detail = timed([
            str(ANALYZER), "--json", "barvinok", "-T", str(target), "-C", args.chunk,
            "--scale-from", str(base),
        ], core, args.timeout, args.cpu_threads)
        rows.append({**common, "step": "thread_scaling", "threads": target,
                     "seconds": seconds, "status": status, "detail": detail})

    assoc = base.with_name("assoc.json")
    seconds, status, _, detail = timed([
        str(ASSOC_CONV), "-i", str(base), "-o", str(assoc), "-a", str(args.associativity),
    ], core, args.timeout, args.cpu_threads)
    rows.append({**common, "step": "associativity", "threads": args.base_threads,
                 "associativity": args.associativity, "seconds": seconds,
                 "status": status, "detail": detail})

    if status != "ok":
        rows.append({**common, "step": "prediction", "threads": args.base_threads,
                     "associativity": args.associativity, "seconds": 0.0,
                     "status": "skipped", "detail": f"associativity {status}"})
        return rows
    seconds, status, stdout, detail = timed([
        str(MRC), "-i", str(assoc), "-c", str(args.cache_bytes), "-b", str(args.block_bytes),
    ], core, args.timeout, args.cpu_threads)
    rows.append({**common, "step": "prediction", "threads": args.base_threads,
                 "associativity": args.associativity, "seconds": seconds, "status": status,
                 "detail": stdout.strip() if status == "ok" else detail})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-threads", type=int, default=4)
    parser.add_argument("--scale-to", default="8,64",
                        help="thread counts the base derivation is rescaled to")
    parser.add_argument("--chunk", default="1")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--associativity", type=int, default=12)
    parser.add_argument("--cache-bytes", type=int, default=256 * 1024,
                        help="cache size the prediction step queries")
    parser.add_argument("--block-bytes", type=int, default=64,
                        help="bytes per block for the prediction step (8 doubles)")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--polygeist", default=str(Path.home() / "Documents/Polygeist/build/bin"))
    parser.add_argument("--database", default=str(REPO / "results" / "parallel" / "timings.db"))
    parser.add_argument("--cpu-threads", default="1",
                        help="1 = pinned single-core measurements (default); 'all' = one "
                             "kernel at a time with rayon free to use every core")
    parser.add_argument("--only-failed", action="store_true",
                        help="re-run only kernels with any non-ok row, keeping the rest")
    parser.add_argument("--prediction-only", action="store_true",
                        help="time only the mrc query, from the kept base derivations")
    args = parser.parse_args()
    args.scale_to = [int(value) for value in args.scale_to.split(",")]
    args.cpu_threads = (os.cpu_count() or 1) if args.cpu_threads == "all" else int(args.cpu_threads)
    if args.cpu_threads != 1:
        # Full-machine measurements must not overlap each other.
        args.workers = 1

    if "SYMBOLICA_LICENSE" not in os.environ:
        print("SYMBOLICA_LICENSE is not set; unlicensed Symbolica allows one instance per "
              "machine, so concurrent collection would abort", file=sys.stderr)
        return 2
    for binary in (ANALYZER, ASSOC_CONV, MRC):
        if not binary.exists():
            print(f"missing {binary}; run `cargo build --release` first", file=sys.stderr)
            return 2

    root = REPO / "target" / "category-kernels"
    catalogue = _categories.build_kernels(root, Path(args.polygeist))
    jobs = [{"category": category, "kernel": name,
             "source": str(root / category / f"{name}.mlir")}
            for category, names in catalogue.items() for name in names]

    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    migrate(connection)
    connection.executescript(SCHEMA)
    if args.only_failed:
        failed = {tuple(row) for row in connection.execute(
            "SELECT DISTINCT category, kernel FROM timings WHERE status != 'ok' "
            "AND cpu_threads = ?", (args.cpu_threads,))}
        jobs = [job for job in jobs if (job["category"], job["kernel"]) in failed]
    if args.cpu_threads == 1:
        print(f"{len(jobs)} kernels over {args.workers} cores "
              f"({args.first_core}-{args.first_core + args.workers - 1})", file=sys.stderr)
    else:
        print(f"{len(jobs)} kernels one at a time, {args.cpu_threads} cores each",
              file=sys.stderr)

    lock = threading.Lock()
    cores: queue.Queue[int] = queue.Queue()
    for core in range(args.first_core, args.first_core + args.workers):
        cores.put(core)

    def work(job: dict) -> None:
        core = cores.get()
        try:
            rows = run_kernel(job, core, args)
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
        summary = "  ".join(f"{r['step'][:5]}={r['seconds']:.2f}{'' if r['status']=='ok' else '!'}"
                            for r in rows)
        print(f"  {job['category']}/{job['kernel']}  {summary}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))

    print("\n" + "=" * 74)
    print(f"{'category':<16}{'step':<16}{'ok':>4}{'t/o':>5}{'err':>5}{'skip':>6}"
          f"{'median s':>11}{'max s':>10}")
    print("-" * 74)
    for category in catalogue:
        for step in ("derivation", "thread_scaling", "associativity", "prediction"):
            rows = connection.execute(
                "SELECT status, seconds FROM timings WHERE category=? AND step=? "
                "AND cpu_threads=?", (category, step, args.cpu_threads)).fetchall()
            ok = sorted(s for status, s in rows if status == "ok")
            count = lambda name: sum(status == name for status, _ in rows)  # noqa: E731
            if ok:
                mid = len(ok) // 2
                median = ok[mid] if len(ok) % 2 else (ok[mid - 1] + ok[mid]) / 2
                shown, worst = f"{median:.3f}", f"{ok[-1]:.3f}"
            else:
                shown, worst = "-", "-"
            print(f"{category:<16}{step:<16}{len(ok):>4}{count('timeout'):>5}"
                  f"{count('error'):>5}{count('skipped'):>6}{shown:>11}{worst:>10}")
    connection.close()
    print(f"\nwrote {database}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
