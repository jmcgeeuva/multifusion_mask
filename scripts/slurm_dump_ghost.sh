#!/bin/bash
#SBATCH --job-name=av_ghostdump
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=02:00:00
#SBATCH --nice=500000
#SBATCH --output=logs/attack/ghostdump_%j.out
# Re-dump the GHOST attack at the full 100-step strength (the 60-step dump made ~0 ghosts;
# the 100-step arms make ~9 in-hull ghosts/scene).  Dump 24 scenes so fig1 can pick the
# ghost-richest one.  Matches arms 22190966/67/69/70 (mask=all hull_saturate).  CPU render
# via tools/fig1_ghosts.py afterward.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
O=figs_data; mkdir -p $O
L=logs/attack/ghostdump_$SLURM_JOB_ID
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PY attack_multi.py "$C" --out $L \
    --mask all --cell 8 --bank 1 --inference-norm --place bbox \
    --mode perscene --perscene-steps 100 \
    --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --w-margin 0 \
    --vanish-form bce --k-dense 5 --k-prop 3 \
    --fp-eval --fp-thresh 0.3 \
    --heldout 24 --dump-scenes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --dump $O/ghosts_strong.pkl
echo "=== rc $? ; ALL GHOST DUMP DONE"
