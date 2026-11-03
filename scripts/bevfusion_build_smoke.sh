#!/bin/bash
#SBATCH --job-name=bevf_build
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/transfer/%x_%j.out
set -e
mkdir -p logs/transfer

REPO=/sfs/weka/scratch/tkg5kq/transfer/bevfusion
ENV=/scratch/tkg5kq/.conda/envs/bevfusion_env
CKPT=$REPO/pretrained/bevfusion-det.pth
CFG=$REPO/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion_env
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"
nvcc --version | grep release

echo "=== building BEVFusion CUDA ops (editable) ==="
cd "$REPO"
pip install -e . 2>&1 | tail -25

echo "=== import check: mmdet3d now = BEVFusion repo ==="
python -c "import mmdet3d; print(mmdet3d.__version__, mmdet3d.__file__)"
python -c "from mmdet3d.ops import bev_pool; print('bev_pool OK')"

echo "=== 8-sample smoke ==="
cd /home/tkg5kq/workdir/av_project/multifusion_mask
python tools/bevfusion_eval.py "$CFG" "$CKPT" --limit 8
echo "DONE"
