#!/bin/bash
#SBATCH --job-name=av_region
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/attack/%A_%a.out
# Region-effect losses (LOSS_DESIGN.md), per-scene, one GPU each. All inference-norm + ghost metric.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
COMMON="--mode perscene --cell 8 --place bbox --inference-norm --fp-eval --vanish-form bce --k-dense 5 --k-prop 3 --perscene-steps 100"
NAMES=(ps_suppress_region ps_flatten_pullup ps_hull_saturate_multi ps_hull_saturate_single)
ARGS=("--mask all --region-mode suppress     --w-region 1.0 --w-margin 0"
      "--mask all --region-mode saturate     --w-region 1.0 --w-vanish 0.2 --w-margin 0"
      "--mask all --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --w-margin 0"
      "--mask single --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --w-margin 0")
i=${SLURM_ARRAY_TASK_ID:-0}; N=${NAMES[$i]}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i} $N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "logs/attack/${SLURM_ARRAY_JOB_ID}/$N" $COMMON ${ARGS[$i]}
echo "=== $N exit $?"
