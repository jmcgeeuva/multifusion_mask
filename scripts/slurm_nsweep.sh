#!/bin/bash
#SBATCH --job-name=av_nsweep
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%6
#SBATCH --output=logs/attack/%A_%a.out
# N-vehicle ghost dose-response: paint N randomly-chosen vehicles, per-scene hull_saturate, count ghosts.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
NS=(1 1 1 2 2 2 3 3 3 4 4 4 5 5 5)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}; N=${NS[$i]}; S=${SEEDS[$i]}
O=logs/attack/nsweep_${SLURM_ARRAY_JOB_ID}/n${N}_s${S}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i}  N=$N seed=$S  $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "$O" --mode perscene --mask all --cell 8 --place bbox \
    --inference-norm --fp-eval --vanish-form bce --k-dense 5 --k-prop 3 \
    --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --w-margin 0 \
    --perscene-steps 100 --heldout 40 --paint-n $N --seed $S
echo "=== nsweep n${N}_s${S} exit $?"
