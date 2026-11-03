#!/bin/bash
#SBATCH --job-name=av_diag
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/attack/diag_%j.out
# Diagnostic: reproduce the universal-mode exit-120 with workers=0 + faulthandler so the real
# traceback prints in the main process instead of being swallowed in a dataloader worker.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
export PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1
C=config/classification/classification4.py
echo "=== diag $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY -u attack_multi.py "$C" --out "logs/attack/diag_${SLURM_JOB_ID}/fix_bbox" \
    --mask all --cell 8 --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --mode universal --place bbox --seed 1 --workers 0 --steps 10 --eval-every 1000
echo "=== diag exit $?"
