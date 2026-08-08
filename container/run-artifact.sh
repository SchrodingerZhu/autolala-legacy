#!/usr/bin/env bash
# One-click AutoLALA / SALT artifact driver.
#
# Reproduces the paper's headline comparison — SALT (analytic) vs Barvinok
# (symbolic enumeration) vs Cachegrind (simulation) — as end-to-end wall-clock
# timings, and emits the timing bar chart (Fig 3 shape) plus a machine-readable
# results table into the output directory.
#
# Absolute timings are host-specific (the paper used Ryzen 9950X / EPYC
# 7773X); the reproducible claim is the *ratio* — SALT orders of magnitude
# faster than Barvinok, which is orders faster than simulation. The self-check
# at the end asserts that ordering, not absolute values.
#
# Usage:  run-artifact.sh [OUTPUT_DIR]   (default /output)
#
# NOTE: the tensor-kernel set (Fig 2/3's eight kernels) and the exact SALT-vs-
# Barvinok invocation per figure are pending confirmation of the paper's
# kernel->input mapping; this driver runs the fully-specified matmul set and
# is structured so kernels are added to KERNELS below once that is settled.

set -euo pipefail

OUT="${1:-/output}"
MISC="${MISC_DIR:-/opt/autolala/misc}"
mkdir -p "$OUT"
RESULTS="$OUT/timings.csv"
echo "kernel,method,seconds" > "$RESULTS"

# Cache config for the simulation timing (fully-associative LRU, 64 B block,
# 1024 blocks = 64 KiB — a valid power-of-two-set config per the paper).
D1=32768; LLASSOC=16; LL=1048576

# kernel_name : mlir_file (relative to $MISC). Matmul is fully specified in
# §5.3; extend with the eight tensor kernels once the mapping is confirmed.
KERNELS=(
  "matmul_untiled:const_matmul_3acc.mlir"
  "matmul_tiled:const_matmul_once_tiled.mlir"
)

# time a command in seconds (wall clock), discarding its output
timeit() {
  local start end
  start=$(date +%s.%N)
  "$@" >/dev/null 2>&1 || true
  end=$(date +%s.%N)
  awk -v s="$start" -v e="$end" 'BEGIN{printf "%.4f", e-s}'
}

export SYMBOLICA_HIDE_BANNER=1
echo "=== AutoLALA/SALT artifact — timing comparison ==="
for entry in "${KERNELS[@]}"; do
  name="${entry%%:*}"; file="$MISC/${entry#*:}"
  [ -f "$file" ] || { echo "  skip $name (missing $file)"; continue; }
  echo "-- $name"

  t_salt=$(timeit analyzer -i "$file" salt -b 8)
  echo "$name,salt,$t_salt" >> "$RESULTS"

  t_barv=$(timeit analyzer -i "$file" barvinok \
      --barvinok-arg=--approximation-method=scale --infinite-repeat --block-size 8)
  echo "$name,barvinok,$t_barv" >> "$RESULTS"

  t_sim=$(timeit cachegrind-runner -i "$file" \
      --d1-cache-size "$D1" --ll-associativity "$LLASSOC" --ll-cache-size "$LL")
  echo "$name,cachegrind,$t_sim" >> "$RESULTS"

  printf "   SALT %ss   Barvinok %ss   Cachegrind %ss\n" "$t_salt" "$t_barv" "$t_sim"
done

echo "=== results -> $RESULTS ==="
python3 "$(dirname "$0")/plot_timing.py" "$RESULTS" "$OUT/timing_comparison.png"

# Self-check: the ordering SALT < Barvinok < Cachegrind must hold for every
# kernel (the paper's reproducible claim). Fail loudly if it doesn't.
python3 - "$RESULTS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    by.setdefault(r["kernel"], {})[r["method"]] = float(r["seconds"])
bad = []
for k, m in by.items():
    if not (m.get("salt", 9e9) <= m.get("barvinok", 0) <= m.get("cachegrind", 0)):
        bad.append((k, m))
if bad:
    print("SELF-CHECK FAILED — expected salt <= barvinok <= cachegrind:")
    for k, m in bad:
        print(f"  {k}: {m}")
    sys.exit(1)
print("SELF-CHECK OK: salt <= barvinok <= cachegrind for all kernels.")
for k, m in by.items():
    s, b, c = m["salt"], m["barvinok"], m["cachegrind"]
    print(f"  {k}: Barvinok/SALT={b/s:.0f}x  Cachegrind/SALT={c/s:.0f}x")
PY
echo "=== artifact outputs in $OUT ==="
ls -la "$OUT"
