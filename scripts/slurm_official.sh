#!/bin/bash
#SBATCH --job-name=av_official
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/attack/official_%j.out
# Official nuScenes metrics (mAP/NDS/per-class AP) for IS-Fusion, clean or attacked.
# Env: BANK (universal bbox bank .npy; empty => clean baseline), MAX (0 = whole val), TAG.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
BANK=${BANK:-}
MAX=${MAX:-0}
TAG=${TAG:-run}
O=logs/attack/official_${TAG}_$SLURM_JOB_ID
ARGS="--mask all --cell 8 --bank 1 --inference-norm --place bbox --heldout 2 --official-eval --official-max $MAX"
[ -n "$BANK" ] && ARGS="$ARGS --load-bank $BANK"
echo "=== job $SLURM_JOB_ID TAG=$TAG BANK=${BANK:-<clean>} MAX=$MAX $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "$O" $ARGS
echo "=== official $TAG exit $?"
