#!/bin/bash
#SBATCH --job-name=av_baseabl
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%4
#SBATCH --output=logs/attack/%A_%a.out
# FiNE-style patch baseline + instance->scene fusion attribution ablation. 1 GPU each.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
BASE="--mask all --cell 8 --place bbox --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 --perscene-steps 100 --w-margin 0"
NAMES=(patch_f20_ps patch_f40_ps patch_f20_uni \
       abl_suppress_none abl_suppress_instance_fusion \
       abl_hull_none abl_hull_instance_fusion)
ARGS=("--mode perscene --attack patch --patch-frac 0.2" \
      "--mode perscene --attack patch --patch-frac 0.4" \
      "--mode universal --attack patch --patch-frac 0.2 --eval-every 500" \
      "--mode perscene --region-mode suppress --w-region 1.0 --fp-eval --ablate none" \
      "--mode perscene --region-mode suppress --w-region 1.0 --fp-eval --ablate instance_fusion" \
      "--mode perscene --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --fp-eval --ablate none" \
      "--mode perscene --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --fp-eval --ablate instance_fusion")
i=${SLURM_ARRAY_TASK_ID:-0}; N=${NAMES[$i]}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i} $N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "logs/attack/${SLURM_ARRAY_JOB_ID}/$N" $BASE ${ARGS[$i]}
echo "=== $N exit $?"
