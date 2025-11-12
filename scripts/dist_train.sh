#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

now=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./workdir/${now}/
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../train.py $CONFIG --launcher pytorch --timestamp ${now} --log-dir ./workdir/${now}/ ${@:3} 2>&1|tee ./workdir/${now}/$now.log
