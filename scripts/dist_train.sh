#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CONFIG=$1
GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}
PORT=${PORT:-$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")}

now="${DIST_TRAIN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
mkdir -p ./workdir/${now}/
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../train.py $CONFIG --launcher pytorch --timestamp ${now} --log-dir ./workdir/${now}/ ${@:3} 2>&1|tee ./workdir/${now}/$now.log
