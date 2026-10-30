#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

CONFIG=$1
GPUS=$2
PORT=${PORT:-$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../test.py $CONFIG --launcher pytorch ${@:3}
