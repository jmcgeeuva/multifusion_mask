#!/bin/bash
#SBATCH --job-name=av_natreg
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --array=0-5%4
#SBATCH --output=logs/attack/natreg_%A_%a.out
# Task 1: does naturalness regularization (TV + VGG style) hurt the single-vehicle attack but
# spare/help the multi-vehicle attack?  Per-scene hiding, paint N random vehicles, sweep reg weight.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
T=$SLURM_ARRAY_TASK_ID
#            N=1 single           N=5 multi     (reg levels: none / moderate / strong)
NS=(  1      1      1      5      5      5   )
WTV=( 0      2      6      0      2      6   )
WST=( 0      12     35     0      12     35  )
N=${NS[$T]}; wtv=${WTV[$T]}; wst=${WST[$T]}
OUT=logs/attack/natreg_${SLURM_ARRAY_JOB_ID}/n${N}_tv${wtv}_st${wst}
echo "=== task $T N=$N w_tv=$wtv w_style=$wst -> $OUT ; $(hostname)"
$PY attack_multi.py "$C" --out "$OUT" \
    --mode perscene --mask all --cell 8 --inference-norm --place bbox \
    --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 \
    --heldout 40 --perscene-steps 100 --paint-n "$N" --fp-eval \
    --w-tv "$wtv" --w-style "$wst" --seed 0
echo "=== rc $?"
