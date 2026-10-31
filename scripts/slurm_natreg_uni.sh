#!/bin/bash
#SBATCH --job-name=av_natreguni
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:30:00
#SBATCH --nice=0
#SBATCH --array=0-2%3
#SBATCH --output=logs/attack/natreguni_%A_%a.out
# Universal (one texture, train/test disjoint) under naturalness reg -> the real trade-off AND a
# viewable texture (texture_*.png saved every eval). none / strong / very-strong reg.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
T=$SLURM_ARRAY_TASK_ID
NAMES=( none strong vstrong )
WTV=(   0    6      20 )
WST=(   0    35     120 )
OUT=logs/attack/natreguni_${SLURM_ARRAY_JOB_ID}/${NAMES[$T]}
echo "=== task $T ${NAMES[$T]} w_tv=${WTV[$T]} w_style=${WST[$T]} -> $OUT ; $(hostname)"
$PY attack_multi.py "$C" --out "$OUT" \
    --mode universal --mask all --cell 8 --inference-norm --place bbox \
    --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --steps 1500 --eval-every 300 --heldout 40 \
    --w-tv "${WTV[$T]}" --w-style "${WST[$T]}" --seed 0
echo "=== rc $?"
