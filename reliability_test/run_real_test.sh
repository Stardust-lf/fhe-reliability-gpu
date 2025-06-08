#!/bin/bash
set -m
trap "echo 'Caught Ctrl+C, killing all...'; kill 0; exit 1" SIGINT

NUM_THREADS=8
RUNS=1000

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGDIR="log_${TIMESTAMP}"
mkdir -p "$LOGDIR"

for THREAD_ID in $(seq 1 $NUM_THREADS); do
    LOGFILE="${LOGDIR}/run_thread${THREAD_ID}.log"
    rm -f "$LOGFILE"

    (
        for ((i = 1; i <= RUNS; i++)); do
            echo "[Thread $THREAD_ID - Run $i] $(date '+%F %T')" >> "$LOGFILE"

            echo "[Thread $THREAD_ID - Run $i] GPU Frequencies (MHz):" >> "$LOGFILE"
            nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.gr \
                       --format=csv,noheader,nounits >> "$LOGFILE"

            ./ntt_test 17 1 1 >> "$LOGFILE" 2>&1
            RET=$?

            if [ $RET -ne 0 ]; then
                echo "[Thread $THREAD_ID - Run $i] *** ERROR: Exit code $RET ***" >> "$LOGFILE"
            fi

            echo "" >> "$LOGFILE"
        done
    ) &
done

wait
