#!/bin/bash
#SBATCH --job-name=av_losssweep
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/attack/%A_%a.out
# Loss-shape sweep, per-scene upper bound (all vehicles, shared texture, 4 px cells).
# The original loss with the same setup is ps_multi_shared_c4 in the main attack array.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
BASE="--mode perscene --mask all --bank 1 --cell 4"
NAMES=(loss_reweight loss_bce_topk_cap loss_bce_topk_vanishonly loss_bce_max_cap)
ARGS=("--w-vanish 4 --w-margin 0.25"
      "--vanish-form bce --k-dense 5 --k-prop 3 --margin-cap 2"
      "--vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0"
      "--vanish-form bce --margin-cap 2")
i=${SLURM_ARRAY_TASK_ID:-0}; N=${NAMES[$i]}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i} $N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "logs/attack/${SLURM_ARRAY_JOB_ID}/$N" $BASE ${ARGS[$i]}
echo "=== $N exit $?"
