#!/usr/bin/env bash
set -euo pipefail

process_dir() {
  local PROFILE_DIR="$1"
  local OUTPUT_FILE="$2"
  : > "$OUTPUT_FILE"

  # 1) 收集所有 callgrind.out.* 文件，排除 .out.<pid> 和 .out.<pid>.1
  # 2) 按最后一个“.”后的数字正序排序
  mapfile -t files < <(
    ls -1 "$PROFILE_DIR"/callgrind.out.* 2>/dev/null \
      | grep -Ev 'callgrind\.out\.[0-9]+(\.1)?$' \
      | sort -t'.' -k4,4n
  )

  # 3) 按排好序的列表依次处理
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    base=$(basename "$f")
    echo "=== $base ===" >> "$OUTPUT_FILE"
    callgrind_annotate --inclusive=no "$f" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  done

  echo "Processed ${#files[@]} files in '$PROFILE_DIR', output -> $OUTPUT_FILE"
}

if [[ $# -eq 1 && "$1" == "." ]]; then
  # 扫描每个一级子目录
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
  exit 1
fi
