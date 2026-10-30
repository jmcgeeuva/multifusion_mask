#!/bin/bash
#SBATCH --job-name=av_gradcheck
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grad_check/%j.out
# Gradient-break + camera-vs-LiDAR check for the camouflage attack (grad_check.py). One GPU.
#   sbatch scripts/slurm_grad_check.sh                       # one config per loss family
#   RUNS="name|config|extra args;name2|config|args" sbatch scripts/slurm_grad_check.sh
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
RUNS=${RUNS:-"classification4|config/classification/classification4.py|;bbox2|config/bbox/bbox2.py|;heatmap1|config/heatmap/heatmap1.py|"}
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
i=0
IFS=';' read -ra LIST <<< "$RUNS"
for R in "${LIST[@]}"; do
  IFS='|' read -r N C A <<< "$R"
  echo; echo "################ $N: $C $A"
  MASTER_PORT=$((29600 + (SLURM_JOB_ID + i) % 1000)) \
    $PY grad_check.py "$C" --out "logs/grad_check/$SLURM_JOB_ID/$N" $A
  echo "=== $N exit $?"; i=$((i+1))
done
