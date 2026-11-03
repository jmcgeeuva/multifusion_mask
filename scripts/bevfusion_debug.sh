#!/bin/bash
#SBATCH --job-name=bevf_dbg
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/transfer/%x_%j.out
set -e
REPO=/sfs/weka/scratch/tkg5kq/transfer/bevfusion
CKPT=$REPO/pretrained/bevfusion-det.pth
CFG=$REPO/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate bevfusion_env
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/scratch/tkg5kq/.conda/envs/bevfusion_env
cd /home/tkg5kq/workdir/av_project/multifusion_mask
python tools/bevfusion_debug_voxelize.py "$CFG" "$CKPT"
echo "DBG_DONE"
