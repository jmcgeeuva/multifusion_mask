#!/bin/bash
#SBATCH --job-name=av_unifix
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/attack/%A_%a.out
# Closing the universal gap. Shared by all: vanishing-only log-space loss, BatchNorm/Dropout on
# stored statistics, all-vehicle mask, 8 px cells. Varied: placement, accumulation/schedule, bank.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
COMMON="--mask all --cell 8 --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 --eval-every 500"
NAMES=(perseq_bbox fix_tile fix_screen fix_bbox fix_bbox_acc fix_bbox_acc_bank4)
ARGS=("--mode perseq --place bbox --seq-logs 6 --seq-steps 300 --perscene-lr 0.03 --grad-accum 2 --lr-sched cosine"
      "--mode universal --place tile"
      "--mode universal --place screen"
      "--mode universal --place bbox"
      "--mode universal --place bbox --grad-accum 4 --lr 0.03 --lr-sched cosine"
      "--mode universal --place bbox --grad-accum 4 --lr 0.03 --lr-sched cosine --bank 4")
i=${SLURM_ARRAY_TASK_ID:-0}; N=${NAMES[$i]}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i} $N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "logs/attack/${SLURM_ARRAY_JOB_ID}/$N" $COMMON ${ARGS[$i]}
echo "=== $N exit $?"
