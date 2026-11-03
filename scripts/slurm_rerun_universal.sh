#!/bin/bash
#SBATCH --job-name=av_rerunuf
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/attack/rerunuf_%A_%a.out
# One-off rerun of the universal-mode arms whose texture-save died when the home quota was full
# (eval.json had clean/random but no 'learned'). Each arm is written back into its ORIGINAL
# manifest directory so tools/paper_tables.py needs no edits. Recipes copied verbatim from
# slurm_universal_fix.sh (hiding universal) and slurm_baseline_ablation.sh (patch universal).
# python -u so logs flush (the unbuffered gap is what turned the real save error into a bare exit 120).
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py

# Shared recipes (must match the originals exactly).
HID_COMMON="--mask all --cell 8 --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 --eval-every 500"
PATCH_BASE="--mask all --cell 8 --place bbox --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --perscene-steps 100 --w-margin 0"

# (seed, out-dir-under-logs/attack, arm-name, full args) for each broken arm.
OUT=(   22261712_s1       22261712_s1        22261712_s1       22261714_s2       22261714_s2        22261714_s2       22261864_s1       22261865_s2)
SEEDS=( 1                 1                  1                 2                 2                  2                 1                 2)
NAMES=( fix_tile          fix_screen         fix_bbox          fix_tile          fix_screen         fix_bbox          patch_f20_uni     patch_f20_uni)
ARGS=(  "$HID_COMMON --mode universal --place tile"
        "$HID_COMMON --mode universal --place screen"
        "$HID_COMMON --mode universal --place bbox"
        "$HID_COMMON --mode universal --place tile"
        "$HID_COMMON --mode universal --place screen"
        "$HID_COMMON --mode universal --place bbox"
        "$PATCH_BASE --mode universal --attack patch --patch-frac 0.2 --eval-every 500"
        "$PATCH_BASE --mode universal --attack patch --patch-frac 0.2 --eval-every 500")

i=${SLURM_ARRAY_TASK_ID:-0}
S=${SEEDS[$i]}; O=${OUT[$i]}; N=${NAMES[$i]}
echo "=== rerun ${SLURM_ARRAY_JOB_ID}_${i} seed $S -> logs/attack/$O/$N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY -u attack_multi.py "$C" --out "logs/attack/$O/$N" --seed "$S" ${ARGS[$i]}
echo "=== $N (seed $S) exit $?"
