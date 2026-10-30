#!/bin/bash
#SBATCH --job-name=av_multi
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multi_run/%j.out
# All-vehicle (or single-vehicle) camouflage training on 1 GPU (train_multi.py).
#   sbatch scripts/slurm_train_multi.sh [config] [extra train_multi.py args]
#   e.g. sbatch scripts/slurm_train_multi.sh config/classification/classification4.py --mask single
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=${1:-config/classification/classification4.py}; shift || true
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY train_multi.py "$C" --out "logs/multi_run/$SLURM_JOB_ID" "$@"
echo "=== exit $?"
