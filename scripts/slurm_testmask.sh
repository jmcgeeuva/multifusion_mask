#!/bin/bash
#SBATCH --job-name=av_testmask
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --array=0-11%6
#SBATCH --output=logs/attack/testmask_%A_%a.out
# TEST-TIME mask-count sweep for HIDING: load a FIXED universal texture (--steps 0, no training)
# and vary how many held-out vehicles are painted at eval (--paint-n). Shows whether hiding
# strengthens as more vehicles are masked at test time, for no-reg vs naturalized textures.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
UNI=logs/attack/natreguni_22214923
BANKS=( none strong )          # no-reg vs naturalized universal texture
PN=( 1 2 3 4 5 0 )             # paint-n at test time (0 = all masked vehicles)
i=${SLURM_ARRAY_TASK_ID:-0}
b=$(( i / 6 )); p=$(( i % 6 ))
BK=${BANKS[$b]}; K=${PN[$p]}
O=logs/attack/testmask_${SLURM_ARRAY_JOB_ID}/${BK}/pn${K}
echo "=== task $i bank=$BK paint_n=$K -> $O ; $(hostname)"
$PY attack_multi.py "$C" --out "$O" --mode universal --mask all --cell 8 --place bbox \
    --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --steps 0 --heldout 40 --load-bank ${UNI}/${BK}/bank_last.npy --paint-n $K --fp-eval --seed 0
echo "=== testmask $BK pn${K} exit $?"
