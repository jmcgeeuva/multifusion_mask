#!/bin/bash
#SBATCH --job-name=bevf_metric
#SBATCH -A eng_viva
#SBATCH -p standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/transfer/%x_%j.out
# usage: sbatch scripts/bevfusion_metric.sh <outputs.pkl>
# CPU-only official nuScenes mAP/NDS from a saved outputs pkl, using the local
# staged nuScenes root (fast table load ~45s vs ~>90min on ceph).
set -e
PKL=$1
REPO=/sfs/weka/scratch/tkg5kq/transfer/bevfusion
CKPT=$REPO/pretrained/bevfusion-det.pth
CFG=$REPO/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
STAGE=/scratch/tkg5kq/transfer/nusc_root

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion_env
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
cd /home/tkg5kq/workdir/av_project/multifusion_mask
python tools/bevfusion_eval.py "$CFG" "$CKPT" --eval bbox --from-pkl "$PKL" --data-root "$STAGE"
echo "METRIC_DONE"
