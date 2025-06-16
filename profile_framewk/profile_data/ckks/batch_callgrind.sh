#!/usr/bin/env bash
set -euo pipefail

process_dir() {
  local PROFILE_DIR="$1"
  local OUTPUT_FILE="$2"
  : > "$OUTPUT_FILE"
  for f in "$PROFILE_DIR"/callgrind.out.*; do
    [[ -e "$f" ]] || continue
    base=$(basename "$f")
    # skip callgrind.out.<pid> and callgrind.out.<pid>.1
    if [[ "$base" =~ ^callgrind\.out\.[0-9]+$ ]] || [[ "$base" =~ ^callgrind\.out\.[0-9]+\.1$ ]]; then
      continue
    fi
    echo "=== $base ===" >> "$OUTPUT_FILE"
    callgrind_annotate --inclusive=no "$f" \
      >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  done
  local count
  count=$(ls -1 "$PROFILE_DIR"/callgrind.out.* 2>/dev/null \
            | grep -Ev '^.+\.out\.[0-9]+(\.1)?$' \
            | wc -l)
  echo "Processed $count files in '$PROFILE_DIR', output -> $OUTPUT_FILE"
}

if [[ $# -eq 1 && "$1" == "." ]]; then
  # scan every first‐level subdirectory
  for subdir in */; do
    [[ -d $subdir ]] || continue
    name=${subdir%/}
    process_dir "$PWD/$subdir" "$PWD/${name}.info"
  done
elif [[ $# -eq 2 ]]; then
  process_dir "$1" "$2"
else
  echo "Usage:"
  echo "  $0 <profile_dir> <output_file>"
  echo "  $0 ."
  echo "Examples:"
  echo "  $0 /path/to/profile_data/bgv/rotate_random rotate_random.txt"
  echo "  $0 .   # will process each subdir and create subdir.info"
  exit 1
fi
