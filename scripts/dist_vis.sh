#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

CONFIG=$1
CHECKPOINT=$2

python nusc_vis_pred.py $CONFIG $CHECKPOINT ${@:3}