#!/bin/bash
# v3: memory-safe rerun of the OOM-killed T3 (MET) jobs + full T4 (jet finding).
# Root cause of the v2 kills: EventShards decompressed every npz key (~37 GB)
# in every job; 5 concurrent loads exceeded node RAM and the kernel OOM killer
# took out 4 runs while the runner, which never checked exit codes, declared
# T3_DONE. Paired fix in train_event.py loads only task-relevant keys
# (tracks-only jobs ~4 GB; TC jobs ~35 GB).
# Policy here: tracks-only jobs run 4-wide; TC jobs at most 2-wide; every job
# must produce its metrics JSON or it is retried once, and the DONE markers
# are emitted only after counting the expected metrics files.
set -u
BASE=/storage/afarbin/jetreg
PY=/home/afarbin/.virtualenvs/CaloGraphNet/bin/python
DATA=$BASE/event_data
LOGS=$BASE/logs
CHAIN=$LOGS/t34_event_chain.log
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd $BASE/code

exec 9>$BASE/.t34_v3.lock
flock -n 9 || { echo "v3 already running" >> $CHAIN; exit 1; }

run_one() {  # task inputs seed gpu outdir
  local task=$1 inp=$2 seed=$3 gpu=$4 out=$5
  local m=$out/${task}_${inp}_seed${seed}_metrics.json
  local attempt
  for attempt in 1 2; do
    echo "[gpu$gpu] $task $inp seed $seed (attempt $attempt)" >> $CHAIN
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $PY -m dfm.jetreg.train_event \
      --task $task --inputs $inp --seed $seed --data-dir $DATA --out $out \
      > $LOGS/${task}_${inp}_s${seed}.log 2>&1
    [ -f "$m" ] && return 0
    echo "RETRY_NEEDED $task $inp seed $seed" >> $CHAIN
  done
  echo "JOB_FAILED $task $inp seed $seed" >> $CHAIN
  return 1
}

echo "=== v3 chain start $(date)" >> $CHAIN

# Phase A: tracks-only jobs, 4-wide (light after the lazy-key fix)
run_one met T 2 0 $BASE/results_t3 &
run_one jetfind T 0 1 $BASE/results_t4 &
run_one jetfind T 1 2 $BASE/results_t4 &
run_one jetfind T 2 3 $BASE/results_t4 &
wait

# Phase B1: MET TC jobs, 2-wide then 1
run_one met TC 0 0 $BASE/results_t3 &
run_one met TC 1 1 $BASE/results_t3 &
wait
run_one met TC 2 0 $BASE/results_t3
n=$(ls $BASE/results_t3/met_*_metrics.json 2>/dev/null | wc -l)
if [ "$n" -eq 6 ]; then echo "T3_DONE" >> $CHAIN; else echo "T3_INCOMPLETE n=$n" >> $CHAIN; fi

# Phase B2: jetfind TC jobs, 2-wide then 1
run_one jetfind TC 0 0 $BASE/results_t4 &
run_one jetfind TC 1 1 $BASE/results_t4 &
wait
run_one jetfind TC 2 0 $BASE/results_t4
n=$(ls $BASE/results_t4/jetfind_*_metrics.json 2>/dev/null | wc -l)
if [ "$n" -eq 6 ]; then echo "T4_DONE" >> $CHAIN; else echo "T4_INCOMPLETE n=$n" >> $CHAIN; fi

echo "T34_EVENT_COMPLETE" >> $CHAIN
