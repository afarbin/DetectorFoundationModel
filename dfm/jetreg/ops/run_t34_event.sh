#!/bin/bash
# T3 (MET) then T4 (jet finding), chained behind the event-dataset build.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd /storage/afarbin/jetreg/code
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/event_data
LOGS=/storage/afarbin/jetreg/logs

echo "waiting for event dataset..."
until grep -q "^done\." $LOGS/build_events.log 2>/dev/null; do sleep 120; done
echo "EVENT_BUILD_DONE"

run_queue() {  # $1 = task, $2 = results dir, $3 = marker
  local task=$1 res=$2 marker=$3
  mkdir -p $res
  local Q=/storage/afarbin/jetreg/${task}.queue
  cat > $Q <<JOBS
T 0
T 1
T 2
TC 0
TC 1
TC 2
JOBS
  touch $Q.lock
  worker() {
    local gpu=$1
    while true; do
      local job
      job=$(flock $Q.lock -c "head -n1 $Q; sed -i 1d $Q")
      [ -z "$job" ] && break
      set -- $job
      echo "[gpu$gpu] $task $1 seed $2"
      CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $PY -m dfm.jetreg.train_event \
        --task $task --inputs $1 --seed $2 --data-dir $DATA --out $res \
        > $LOGS/${task}_$1_s$2.log 2>&1
    done
  }
  for g in 0 1 2 3; do worker $g & done
  wait
  echo "$marker"
}

run_queue met /storage/afarbin/jetreg/results_t3 T3_DONE
run_queue jetfind /storage/afarbin/jetreg/results_t4 T4_DONE
echo "T34_EVENT_COMPLETE"
