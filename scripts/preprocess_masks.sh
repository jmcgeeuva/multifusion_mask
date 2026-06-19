#!/bin/bash
#SBATCH --job-name=preprocess_masks
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=tkg5kq@virginia.edu
#SBATCH --array=0-5%6
#SBATCH --output=logs/preprocess_masks_%A_%a.out
#SBATCH --error=logs/preprocess_masks_%A_%a.err

CAMERAS=("CAM_FRONT" "CAM_BACK" "CAM_FRONT_LEFT" "CAM_FRONT_RIGHT" "CAM_BACK_LEFT" "CAM_BACK_RIGHT")
CAMERA=${CAMERAS[$SLURM_ARRAY_TASK_ID]}

MASK_PATH="/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks"
PARTS_DIR="./nuscenes_masks_index_parts"
mkdir -p "$PARTS_DIR"
mkdir -p logs

source /home/tkg5kq/.bashrc
source activate bound

echo "[$(date)] Array task $SLURM_ARRAY_TASK_ID — camera: $CAMERA"

python scripts/preprocess_masks.py \
    --mask-path "$MASK_PATH" \
    --scan-dir  "$MASK_PATH/train/samples/$CAMERA" \
    --output    "$PARTS_DIR/index_${SLURM_ARRAY_TASK_ID}.txt" \
    --threads   "$SLURM_NTASKS"

echo "[$(date)] Done. Wrote $PARTS_DIR/index_${SLURM_ARRAY_TASK_ID}.txt"

# After all 6 array tasks finish, merge with:
#   cat nuscenes_masks_index_parts/index_*.txt > nuscenes_masks_index.txt
