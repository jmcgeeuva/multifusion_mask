#!/bin/bash
#SBATCH --job-name=composite_tf
#SBATCH -A eng_viva
#SBATCH -p standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/transfer/%x_%j.out
# usage: sbatch scripts/composite_transfer.sh <bank.npy> <out_dir> [--random]
# Composite an IS-Fusion universal texture onto the full nuScenes val raw images
# for the BEVFusion transfer eval. Only vehicle-bearing camera images are written.
set -e
mkdir -p logs/transfer
BANK=$1
OUT=$2
EXTRA=${3:-}
INFOS=/scratch/tkg5kq/transfer/nusc_root/nuscenes_infos_val.pkl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion_env
cd /home/tkg5kq/workdir/av_project/multifusion_mask
python tools/composite_transfer.py --bank "$BANK" --out "$OUT" --infos "$INFOS" \
    --index ./nuscenes_masks_index.txt --vehicles car,truck,bus --workers 16 $EXTRA
echo "COMPOSITE_DONE"
