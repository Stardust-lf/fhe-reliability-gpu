#!/bin/bash

# 如果已存在 exp_log.txt，就先删除
rm -f exp_log.txt

# 遍历 n = 2, 4, 8, …, 2048
for n in 2 4 8 16 32 64 128 256 512 1024 2048; do
  # 在日志里写入当前 n 的标识
  echo "===== Running n=$n (50 runs) =====" >> exp_log.txt
  # 对于每个 n，执行 50 次
  for i in $(seq 1 50); do
    echo "[n=$n | run $i] $(date '+%Y-%m-%d %H:%M:%S')" >> exp_log.txt
    ./ntt_test 12 32 $n >> exp_log.txt 2>&1
  done
  echo "" >> exp_log.txt
done
