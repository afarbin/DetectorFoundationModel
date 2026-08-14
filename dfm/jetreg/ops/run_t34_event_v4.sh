#!/bin/bash
# v4: post-reboot recovery. The Aug 13 reboot killed the v3 chain during its
# final phase; everything except the three jetfind TC seeds had completed and
# is preserved (results_t3 6/6, results_t4 jetfind_T 3/3). This runner does
# ONLY the missing jobs, 2-wide then 1, with the same verification and gated
# markers as v3. Do NOT rerun v3 — its stale-metrics guard would delete and
# redo completed runs.
set -u
BASE=/storage/afarbin/jetreg
# venv relocated to local /test disk after the Aug 13 home wipe
PY=/test/afarbin/venvs/CaloGraphNet/bin/python
DATA=$BASE/event_data
LOGS=$BASE/logs
CHAIN=$LOGS/t34_event_chain.log
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd $BASE/code

exec 9>$BASE/.t34_v4.lock
flock -n 9 || { echo "v4 already running" >> $CHAIN; exit 1; }

run_one() {  # task inputs seed gpu outdir
  local task=$1 inp=$2 seed=$3 gpu=$4 out=$5
  local m=$out/${task}_${inp}_seed${seed}_metrics.json
  local attempt
  rm -f "$m"
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

echo "=== v4 recovery start $(date)" >> $CHAIN
run_one jetfind TC 0 0 $BASE/results_t4 &
run_one jetfind TC 1 1 $BASE/results_t4 &
wait
run_one jetfind TC 2 0 $BASE/results_t4

n4=$(ls $BASE/results_t4/jetfind_*_metrics.json 2>/dev/null | wc -l)
n3=$(ls $BASE/results_t3/met_*_metrics.json 2>/dev/null | wc -l)
if [ "$n4" -eq 6 ]; then echo "T4_DONE" >> $CHAIN; else echo "T4_INCOMPLETE n=$n4" >> $CHAIN; fi
if [ "$n3" -eq 6 ] && [ "$n4" -eq 6 ]; then
  echo "T34_EVENT_COMPLETE" >> $CHAIN
else
  echo "T34_EVENT_INCOMPLETE met=$n3 jetfind=$n4" >> $CHAIN
fi
