#!/bin/bash

rm -f exp_log.txt

for n in 2 4 8 16 32 64 128 256 512 1024 2048; do
  echo "===== Running n=$n (50 runs) =====" >> exp_log.txt
  for i in $(seq 1 50); do
    echo "[n=$n | run $i] $(date '+%Y-%m-%d %H:%M:%S')" >> exp_log.txt
    ./ntt_test 12 32 $n >> exp_log.txt 2>&1
  done
  echo "" >> exp_log.txt
done
