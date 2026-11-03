#!/bin/bash
#SBATCH --job-name=av_natsingle
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:30:00
#SBATCH --nice=0
#SBATCH --output=logs/attack/natsingle_%j.out
# Single-vehicle counterpart of natreguni/none: IDENTICAL universal recipe (bbox, inference-norm,
# bce vanish, w_tv=w_style=0) EXCEPT --mask single (vanish one target vehicle/scene) instead of all.
# Gives the matched single-vs-multi pair for the BEVFusion cross-model transfer thesis test.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
OUT=logs/attack/natsingle_${SLURM_JOB_ID}
echo "=== natreg single-mask universal -> $OUT ; $(hostname)"
$PY attack_multi.py "$C" --out "$OUT" \
    --mode universal --mask single --cell 8 --inference-norm --place bbox \
    --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --steps 1500 --eval-every 300 --heldout 40 \
    --w-tv 0 --w-style 0 --seed 0
echo "=== rc $?"
