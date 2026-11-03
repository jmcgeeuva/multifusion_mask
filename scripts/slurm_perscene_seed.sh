#!/bin/bash
#SBATCH --job-name=av_psseed
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/attack/psseed_%j.out
# Seed replica of ONLY the per-scene upper-bound arm (ps_multi_bneval) that feeds the hiding table,
# so that row can be reported as mean+-std. Same recipe as slurm_bneval_check.sh's ps_multi_bneval.
#   SEED=1 sbatch scripts/slurm_perscene_seed.sh
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
O=logs/attack/psseed_${SLURM_JOB_ID}_s${SEED:-0}
echo "=== job $SLURM_JOB_ID seed ${SEED:-0} $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "$O/ps_multi_bneval" \
    --mode perscene --mask all --bank 1 --cell 4 --inference-norm --seed "${SEED:-0}"
echo "=== ps_multi_bneval exit $?"
