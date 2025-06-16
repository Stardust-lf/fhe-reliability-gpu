#!/usr/bin/env bash
set -eEuo pipefail
trap 'echo "❌ Error on line $LINENO" >&2' ERR

echo "🕐 Starting dotprod simulations (50 runs per config)..."

# ensure data directory exists
mkdir -p ./data

# first sweep: bits_per_symbol = 1..16, num_symbols = 1
OUT1=./data/bits1-16_num1.txt
echo "### bits_per_symbol=1..16, num_symbols=1 (50 runs each)" > "$OUT1"
for b in {1..16}; do
  echo -e "\n=== bits_per_symbol=$b, num_symbols=1 ===" >> "$OUT1"
  for run in {1..50}; do
    printf "Run %2d: " "$run" >> "$OUT1"
    if ./build/dotprod_test "$b" 1 >> "$OUT1" 2>&1; then
      echo "OK" >> "$OUT1"
    else
      echo "FAIL" >> "$OUT1"
    fi
  done
done

# second sweep: bits_per_symbol = 1, num_symbols = 1..8
OUT2=./data/bits1_num1-8.txt
echo "### bits_per_symbol=1, num_symbols=1..8 (50 runs each)" > "$OUT2"
for n in {1..8}; do
  echo -e "\n=== bits_per_symbol=1, num_symbols=$n ===" >> "$OUT2"
  for run in {1..50}; do
    printf "Run %2d: " "$run" >> "$OUT2"
    if ./build/dotprod_test 1 "$n" >> "$OUT2" 2>&1; then
      echo "OK" >> "$OUT2"
    else
      echo "FAIL" >> "$OUT2"
    fi
  done
done

echo "✅ All 50× runs complete."
echo "  Results saved in:"
echo "    $OUT1"
echo "    $OUT2"
