#!/bin/bash
#SBATCH --job-name=bevf_eval
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/transfer/%x_%j.out
# usage: sbatch scripts/bevfusion_eval.sh <mode> [img_root]
#   smoke  : 8-sample forward sanity (GPU)
#   infer  : full inference, save outputs pkl, NO eval (GPU). optional img_root for transfer.
#   full   : full inference + nuScenes eval in one job (GPU; eval slow on ceph)
# metric mode (CPU, fast) is run via scripts/bevfusion_metric.sh on a saved pkl.
set -e
mkdir -p logs/transfer
MODE=${1:-smoke}
IMG_ROOT=${2:-}

REPO=/sfs/weka/scratch/tkg5kq/transfer/bevfusion
ENV=/scratch/tkg5kq/.conda/envs/bevfusion_env
CKPT=$REPO/pretrained/bevfusion-det.pth
CFG=$REPO/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
STAGE=/scratch/tkg5kq/transfer/nusc_root

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion_env
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$ENV
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd /home/tkg5kq/workdir/av_project/multifusion_mask
EXTRA=""
TAG="clean"
[ -n "$IMG_ROOT" ] && { EXTRA="--img-root $IMG_ROOT"; TAG=$(basename "$IMG_ROOT"); }

if [ "$MODE" = "full" ]; then
  echo "=== FULL eval (infer + nuScenes mAP/NDS) ==="
  python tools/bevfusion_eval.py "$CFG" "$CKPT" --eval bbox --data-root "$STAGE" \
      --out logs/transfer/outputs_${SLURM_JOB_ID}.pkl $EXTRA
elif [ "$MODE" = "infer" ]; then
  echo "=== INFER only (save outputs pkl, no eval) -> tag=$TAG ==="
  # deterministic pkl name (by img-root tag) so a dependent metric job can find it.
  # The bundled spconv-v1 randomly throws N=0 at CUDA init on Ampere; a fresh process
  # usually succeeds, so retry a few times within the job before giving up.
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "--- infer attempt $attempt ---"
    if python tools/bevfusion_eval.py "$CFG" "$CKPT" --no-eval \
        --out logs/transfer/outputs_${TAG}.pkl $EXTRA; then
      ok=1; break
    fi
    echo "--- attempt $attempt failed (likely spconv N=0); retrying ---"
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "INFER FAILED after retries"; exit 1; }
else
  echo "=== 8-sample SMOKE ==="
  python tools/bevfusion_eval.py "$CFG" "$CKPT" --limit 8 $EXTRA
fi
echo "EVAL_DONE"
