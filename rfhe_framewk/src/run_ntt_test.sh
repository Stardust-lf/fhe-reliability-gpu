#!/usr/bin/env bash
# run_sweep.sh
# Sweep pbits=2..14 and append outputs to one file.
# Code comments in English.

set -euo pipefail

OUT="${1:-results_pbits_2_14_W24_N64_T1e7_seed42.txt}"
EXE="${2:-./four_step_ntt_protected}"

W=24
N=64
TRIALS=1000000
SEED=42

: > "$OUT"  # truncate output file
for P in $(seq 2 30); do
  echo "### pbits=${P} | $(date -Is)" >> "$OUT"
  "$EXE" --W "$W" --N "$N" --trials "$TRIALS" --seed "$SEED" --pbits "$P" >> "$OUT"
  echo >> "$OUT"
done

echo "done -> $OUT"

