#!/bin/bash
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd /storage/afarbin/jetreg/code
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/data_v2
RES=/storage/afarbin/jetreg/results_t2
LOGS=/storage/afarbin/jetreg/logs
mkdir -p $RES
Q=/storage/afarbin/jetreg/t2.queue
cat > $Q <<'JOBS'
TJC-graph+pan 0
TJC-graph+pan 1
TJC-graph+pan 2
TJC-graph+pan 0 --pan-lambda 0
TJC-graph+pan 1 --pan-lambda 0
TJC-graph+pan 2 --pan-lambda 0
JOBS
worker() {
  local gpu=$1
  while true; do
    local job
    job=$(flock $Q.lock -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    set -- $job
    local cfg=$1 seed=$2; shift 2
    local extra="$@"
    local sfx=""
    [[ "$extra" == *"lambda 0"* ]] && sfx="-tagonly"
    echo "[gpu$gpu] $cfg$sfx seed $seed"
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $PY -m dfm.jetreg.train \
      --data-dir $DATA --out $RES --config $cfg --seed $seed \
      --epochs 40 --patience 6 $extra \
      > $LOGS/t2_pan${sfx}_s${seed}.log 2>&1
  done
  echo "[gpu$gpu] done"
}
touch $Q.lock
for g in 0 1 2 3; do worker $g $Q & done
wait
$PY -m dfm.jetreg.evaluate aggregate --results $RES
echo "T2_COMPLETE"
