#!/bin/bash
#SBATCH --job-name=av_unighost
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/attack/unighost_%A_%a.out
# UNIVERSAL ghost-fabrication texture: train ONE hull-saturate texture on the training split,
# to then evaluate on full val (a universal/at-scale fabrication claim, vs the existing per-scene
# ghost result). Recipe = the canonical universal HIDING arm (slurm_universal_fix.sh:fix_bbox)
# with the objective swapped to fabrication: region-mode hull_saturate, w-vanish 0, fp-eval on.
# This combo (universal + hull_saturate) is UNVALIDATED, so each task runs a tiny smoke first and
# only proceeds to the full train if the smoke produced a finite ghost count.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py

SEEDS=(0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
S=${SEEDS[$i]}
JOB=${SLURM_ARRAY_JOB_ID:-local}
ROOT=logs/attack/unighost_${JOB}_s${S}

# Ghost-fabrication objective (shared by smoke and full run).
GHOST="--mask all --cell 8 --place bbox --inference-norm --vanish-form bce --k-dense 5 --k-prop 3 \
       --w-margin 0 --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --fp-eval"

echo "=== unighost ${JOB}_${i} seed $S $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- smoke: 6 steps, 2 held-out frames; must emit eval.json with false_positives ---
SMOKE=$ROOT/smoke
$PY -u attack_multi.py "$C" --out "$SMOKE" --seed "$S" --mode universal $GHOST \
    --steps 6 --heldout 2 --eval-every 3
if ! grep -q "false_positives" "$SMOKE/eval.json" 2>/dev/null; then
  echo "=== SMOKE FAILED (no false_positives in $SMOKE/eval.json) -- aborting full run"
  exit 1
fi
echo "=== smoke ok -> full universal ghost train"

# --- full universal ghost training (default 3000 steps, matches universal hiding) ---
$PY -u attack_multi.py "$C" --out "$ROOT/uni_hull_saturate" --seed "$S" --mode universal $GHOST \
    --eval-every 500
echo "=== uni_hull_saturate (seed $S) exit $?"
