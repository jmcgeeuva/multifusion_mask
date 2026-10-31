#!/bin/bash
#SBATCH --job-name=av_targets
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --array=0-2%3
#SBATCH --output=logs/attack/targets_%A_%a.out
# Task 2: how well does the texture attack train on non-vehicles vs vehicles vs both?
# Mask classes are COCO-named; the code maps person->pedestrian, bicycle, motorcycle to nuScenes GTs.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
T=$SLURM_ARRAY_TASK_ID
NAMES=( veh                 nonveh                       both )
CLS=(  "car,truck,bus"     "person,bicycle,motorcycle"  "car,truck,bus,person,bicycle,motorcycle" )
OUT=logs/attack/targets_${SLURM_ARRAY_JOB_ID}/${NAMES[$T]}
echo "=== task $T ${NAMES[$T]} classes='${CLS[$T]}' -> $OUT ; $(hostname)"
$PY attack_multi.py "$C" --out "$OUT" \
    --mode perscene --mask all --cell 8 --inference-norm --place bbox \
    --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --heldout 40 --perscene-steps 100 --fp-eval \
    --vehicles "${CLS[$T]}" --seed 0
echo "=== rc $?"
