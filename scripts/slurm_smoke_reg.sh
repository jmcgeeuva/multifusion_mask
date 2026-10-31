#!/bin/bash
#SBATCH --job-name=av_smoke_reg
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=00:40:00
#SBATCH --nice=0
#SBATCH --output=logs/attack/smoke_reg_%j.out
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
L=logs/attack/smoke_reg_$SLURM_JOB_ID
HIDE="--mode perscene --mask all --cell 8 --inference-norm --place bbox --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 --heldout 2 --perscene-steps 4"
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run(){ echo; echo "### $1"; shift; "$@"; echo "=== rc $?"; }

# Task 1: naturalness reg magnitude at a few weights (paint-n 1), + texture png to eyeball
run base_n1   $PY attack_multi.py "$C" --out $L/base_n1   $HIDE --paint-n 1
run reg_lo_n1 $PY attack_multi.py "$C" --out $L/reg_lo_n1 $HIDE --paint-n 1 --w-tv 0.1 --w-style 1.0
run reg_hi_n1 $PY attack_multi.py "$C" --out $L/reg_hi_n1 $HIDE --paint-n 1 --w-tv 0.3 --w-style 5.0
# Task 2: non-vehicle targeting (person->pedestrian)
run nonveh    $PY attack_multi.py "$C" --out $L/nonveh    $HIDE --vehicles person,bicycle,motorcycle
echo "=== ALL SMOKE DONE"
