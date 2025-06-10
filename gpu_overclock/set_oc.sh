#!/usr/bin/env bash
IDX=${1:-0}      # 默认索引 0
OFF=${2:-100}    # 默认 +100 MHz
echo "[*] Forcing Persistence Mode & raising PL"
sudo nvidia-smi -pm 1
MAX_PL=$(nvidia-smi -q -d POWER | awk '/Max Power Limit/ {print $5; exit}')
sudo nvidia-smi -pl $MAX_PL
echo "[*] Applying GPU clock offset +$OFF on index [$IDX]"
nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[$IDX]=$OFF"
