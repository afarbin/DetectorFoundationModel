#!/bin/bash
# Tier 1: 4 configs x 3 seeds distributed over 4 GPUs
cd /tmp/pycharm_project_9ac966d7
PY=~/.virtualenvs/CaloGraphNet/bin/python
DATA=/storage/afarbin/jetreg/data
OUT=/storage/afarbin/jetreg/results
LOGS=/storage/afarbin/jetreg/logs
CONFIGS=(J T C-graph C-set)
for g in 0 1 2 3; do
  (
    cfg=${CONFIGS[$g]}
    for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$g $PY -m dfm.jetreg.train \
        --data-dir $DATA --out $OUT --config $cfg --seed $seed \
        > $LOGS/train_${cfg}_s${seed}.log 2>&1
    done
    echo "GPU $g ($cfg) done"
  ) &
done
wait
echo "TIER1_COMPLETE"
$PY -m dfm.jetreg.evaluate aggregate --results $OUT
