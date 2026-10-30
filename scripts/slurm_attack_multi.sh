#!/bin/bash
#SBATCH --job-name=av_attack
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-6%3
#SBATCH --output=logs/attack/%A_%a.out
# Targeted camouflage arms (attack_multi.py), one GPU each, at most 3 at once.
#   universal: multi shared c8 | multi shared c4 | multi bank4 c4 | single shared c8
#   per-scene upper bound: multi shared c4 | single shared c4 | multi bank4 c4
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
NAMES=(uni_multi_shared_c8 uni_multi_shared_c4 uni_multi_bank4_c4 uni_single_shared_c8 ps_multi_shared_c4 ps_single_shared_c4 ps_multi_bank4_c4)
ARGS=("--mode universal --mask all --bank 1 --cell 8"
      "--mode universal --mask all --bank 1 --cell 4"
      "--mode universal --mask all --bank 4 --cell 4"
      "--mode universal --mask single --bank 1 --cell 8"
      "--mode perscene --mask all --bank 1 --cell 4"
      "--mode perscene --mask single --bank 1 --cell 4"
      "--mode perscene --mask all --bank 4 --cell 4")
i=${SLURM_ARRAY_TASK_ID:-0}
N=${NAMES[$i]}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i} $N $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "logs/attack/${SLURM_ARRAY_JOB_ID}/$N" ${ARGS[$i]}
echo "=== $N exit $?"
