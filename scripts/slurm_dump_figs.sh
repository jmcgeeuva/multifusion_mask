#!/bin/bash
#SBATCH --job-name=av_dumpfig
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/attack/dumpfig_%j.out
# Produce figure bundles for all 3 paper figures over a shared held-out subset (scenes 0-9),
# so fig3 can compare baselines on one common scene. CPU render scripts consume these.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
O=figs_data; mkdir -p $O
L=logs/attack/dumpfig_$SLURM_JOB_ID
U=logs/attack/22125951            # universal banks (seed 0)
COMMON="--mask all --cell 8 --bank 1 --inference-norm --heldout 10 --dump-scenes 0,1,2,3,4,5,6,7,8,9"
PS="--mode perscene --perscene-steps 60 --place bbox"
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run() { echo; echo "### $1"; shift; "$@"; echo "=== rc $?"; }

# --- figure 1 (ghosts) + figure 2 (hiding): per-scene attacks ---
run ghost  $PY attack_multi.py "$C" --out $L/ghost $COMMON $PS \
    --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --fp-eval --dump $O/ghosts.pkl
run hide   $PY attack_multi.py "$C" --out $L/hide  $COMMON $PS \
    --vanish-form bce --k-dense 5 --k-prop 3 --w-margin 0 --dump $O/hide.pkl

# --- figure 3 (baselines on a shared scene) ---
run full   $PY attack_multi.py "$C" --out $L/full   $COMMON --mode universal --place bbox   --steps 0 --load-bank $U/fix_bbox/bank_last.npy   --dump $O/full.pkl
run tile   $PY attack_multi.py "$C" --out $L/tile   $COMMON --mode universal --place tile   --steps 0 --load-bank $U/fix_tile/bank_last.npy   --dump $O/tile.pkl
run screen $PY attack_multi.py "$C" --out $L/screen $COMMON --mode universal --place screen --steps 0 --load-bank $U/fix_screen/bank_last.npy --dump $O/screen.pkl
run random $PY attack_multi.py "$C" --out $L/random $COMMON --mode universal --place bbox   --steps 0 --dump $O/random.pkl
run patch20 $PY attack_multi.py "$C" --out $L/patch20 $COMMON $PS --attack patch --patch-frac 0.2 --dump $O/patch20.pkl
run patch40 $PY attack_multi.py "$C" --out $L/patch40 $COMMON $PS --attack patch --patch-frac 0.4 --dump $O/patch40.pkl
echo "=== ALL DUMPS DONE"
