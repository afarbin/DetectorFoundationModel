#!/bin/bash
# Phase 2 / T1: all-flavor dataset rebuild + flavor calibration matrix.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd /storage/afarbin/jetreg/code
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/data_v2
RES=/storage/afarbin/jetreg/results_t1
LOGS=/storage/afarbin/jetreg/logs
mkdir -p $DATA $RES

F=/storage/mxg1065/ttbar_100GB/user.bbullard.50733453._0000
FILES=""
for n in 11 12 13 14 15 16 17 18 19 20 23 25 28 29 30 31 32; do
  FILES="$FILES ${F}${n}.ntuple.root"
done
if [ ! -f $DATA/manifest.json ]; then
  echo "building all-flavor dataset (17 files)..."
  nice -n 10 $PY -m dfm.jetreg.build_dataset --files $FILES \
    --calo-ntuple /storage/mxg1065/input_data/ttbar_100events.root \
    --calo-processed /storage/mxg1065/processed_data/ttbar_1000 \
    --flavors all --out $DATA > $LOGS/build_v2.log 2>&1 || exit 1
fi
echo "BUILD_V2_DONE"

Q=/storage/afarbin/jetreg/t1.queue
cat > $Q <<'JOBS'
TJC-graph 0
TJC-graph 1
TJC-graph 2
TJC-graph+fc 0
TJC-graph+fc 1
TJC-graph+fc 2
TJC-graph 0 --flavor-select b
TJC-graph 1 --flavor-select b
TJC-graph 2 --flavor-select b
TJC-graph 0 --flavor-select c
TJC-graph 1 --flavor-select c
TJC-graph 2 --flavor-select c
TJC-graph 0 --flavor-select light
TJC-graph 1 --flavor-select light
TJC-graph 2 --flavor-select light
JOBS

worker() {
  local gpu=$1 q=$2
  while true; do
    local job
    job=$(flock $q.lock -c "head -n1 $q; sed -i 1d $q")
    [ -z "$job" ] && break
    set -- $job
    local cfg=$1 seed=$2; shift 2
    local extra="$@"
    local sfx=""
    [[ "$extra" == *"select b"* ]] && sfx="-b"
    [[ "$extra" == *"select c"* ]] && sfx="-c"
    [[ "$extra" == *"select light"* ]] && sfx="-l"
    echo "[gpu$gpu] $cfg$sfx seed $seed"
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $PY -m dfm.jetreg.train \
      --data-dir $DATA --out $RES --config $cfg --seed $seed \
      --epochs 40 --patience 6 $extra \
      > $LOGS/t1_$(echo $cfg | tr "+" "_")${sfx}_s${seed}.log 2>&1
  done
  echo "[gpu$gpu] done"
}

touch $Q.lock
for g in 0 1 2 3; do worker $g $Q & done
wait
$PY -m dfm.jetreg.evaluate aggregate --results $RES
echo "T1_COMPLETE"
