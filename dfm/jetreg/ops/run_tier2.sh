#!/bin/bash
# Tier 2: 6 configs x 3 seeds, queue-dispatched over 4 GPUs
cd /tmp/pycharm_project_9ac966d7
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/data
RES=/storage/afarbin/jetreg/results
LOGS=/storage/afarbin/jetreg/logs
Q=/storage/afarbin/jetreg/tier2.queue

cat > $Q <<'JOBS'
TC-graph 0
TC-graph 1
TC-graph 2
JC-graph 0
JC-graph 1
JC-graph 2
TJ 0
TJ 1
TJ 2
TC-set 0
TC-set 1
TC-set 2
C-graph+mu 0
C-graph+mu 1
C-graph+mu 2
TC-graph 0 --loss mae
TC-graph 1 --loss mae
TC-graph 2 --loss mae
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
    local suffix=""
    [[ "$extra" == *mae* ]] && suffix="-mae"
    echo "[gpu$gpu] $cfg$suffix seed $seed"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m dfm.jetreg.train \
      --data-dir $DATA --out $RES --config $cfg --seed $seed $extra \
      > $LOGS/train_${cfg}${suffix}_s${seed}.log 2>&1
  done
  echo "[gpu$gpu] queue empty"
}

touch $Q.lock
for g in 0 1 2 3; do worker $g & done
wait
echo "TIER2_COMPLETE"
$PY -m dfm.jetreg.evaluate aggregate --results $RES
