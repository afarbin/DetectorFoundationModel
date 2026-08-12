#!/bin/bash
# thread caps: torch/numpy OpenMP pools default to ALL 56 cores per
# process - stacked jobs starve the scheduler (the likely freeze cause)
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
# Tier 3 (full-data anchors + TJC) then Tier 4 (pretraining study), chained.
cd /storage/afarbin/jetreg/code
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/data
RES=/storage/afarbin/jetreg/results_full
LOGS=/storage/afarbin/jetreg/logs
PRE=/storage/afarbin/jetreg/pretrain_big/pretrain_graph_group_r0.15.pt
mkdir -p $RES

echo "waiting for ALL17_READY flag..."
until [ -f $DATA/ALL17_READY ]; do sleep 60; done
echo "all 17 shards ready"

Q=/storage/afarbin/jetreg/tier3.queue
cat > $Q <<'JOBS'
T 0
T 1
T 2
TJ 0
TJ 1
TJ 2
C-graph 0
C-graph 1
C-graph 2
TC-graph 0
TC-graph 1
TC-graph 2
TJC-graph 0
TJC-graph 1
TJC-graph 2
TJC-set 0
TJC-set 1
TJC-set 2
JOBS

worker() {
  local gpu=$1 q=$2 extra_common=$3
  while true; do
    local job
    job=$(flock $q.lock -c "head -n1 $q; sed -i 1d $q")
    [ -z "$job" ] && break
    set -- $job
    local cfg=$1 seed=$2; shift 2
    local extra="$@"
    local sfx=""
    [[ "$extra" == *mae* ]] && sfx="$sfx-mae"
    [[ "$extra" == *pretrained* ]] && sfx="$sfx-pre"
    [[ "$extra" == *freeze* ]] && sfx="$sfx-probe"
    [[ "$extra" == *train-frac* ]] && sfx="$sfx-frac"
    echo "[gpu$gpu] $cfg$sfx seed $seed"
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $PY -m dfm.jetreg.train \
      --data-dir $DATA --out $RES --config $cfg --seed $seed \
      --epochs 40 --patience 6 $extra \
      > $LOGS/t34_${cfg}${sfx}_s${seed}.log 2>&1
  done
  echo "[gpu$gpu] queue done"
}

touch $Q.lock
# GPU 1 belongs to Mohammad Ali - use 0, 2, 3
for g in 0 2 3; do worker $g $Q & done
wait
echo "TIER3_DONE"

echo "waiting for big pretraining checkpoint..."
until grep -q "done, best val" $LOGS/pretrain_big.log 2>/dev/null; do sleep 60; done

Q4=/storage/afarbin/jetreg/tier4.queue
cat > $Q4 <<JOBS
C-graph 0 --pretrained $PRE
C-graph 1 --pretrained $PRE
C-graph 2 --pretrained $PRE
TC-graph 0 --pretrained $PRE
TC-graph 1 --pretrained $PRE
TC-graph 2 --pretrained $PRE
C-graph 0 --pretrained $PRE --freeze-encoder
C-graph 1 --pretrained $PRE --freeze-encoder
C-graph 2 --pretrained $PRE --freeze-encoder
TC-graph 0 --train-frac 0.01
TC-graph 1 --train-frac 0.01
TC-graph 2 --train-frac 0.01
TC-graph 0 --train-frac 0.1
TC-graph 1 --train-frac 0.1
TC-graph 2 --train-frac 0.1
TC-graph 0 --pretrained $PRE --train-frac 0.01
TC-graph 1 --pretrained $PRE --train-frac 0.01
TC-graph 2 --pretrained $PRE --train-frac 0.01
TC-graph 0 --pretrained $PRE --train-frac 0.1
TC-graph 1 --pretrained $PRE --train-frac 0.1
TC-graph 2 --pretrained $PRE --train-frac 0.1
JOBS
touch $Q4.lock
for g in 0 2 3; do worker $g $Q4 & done
wait
echo "TIER4_DONE"
$PY -m dfm.jetreg.evaluate aggregate --results $RES
echo "TIER34_COMPLETE"
