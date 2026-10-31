#!/bin/bash
#SBATCH --job-name=av_nsweepext
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --array=0-29%8
#SBATCH --output=logs/attack/nsweepext_%A_%a.out
# Extra ghost dose-response lines for fig4, same config as scripts/slurm_nsweep.sh:
#   tasks  0-14 : "+naturalization"  = identical sweep + TV/VGG style reg (w_tv 6, w_style 35)
#   tasks 15-29 : "+nonveh"          = paint set expanded to vehicles + non-vehicles
# So the original fig4 lines read as "vehicles only" and these two overlay on top.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
NS=(1 1 1 2 2 2 3 3 3 4 4 4 5 5 5)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -lt 15 ]; then
  COND=natreg; j=$i
  EXTRA="--w-tv 6 --w-style 35"
  VEH="car,truck,bus"
else
  COND=nonveh; j=$((i-15))
  EXTRA=""
  VEH="car,truck,bus,person,bicycle,motorcycle"
fi
N=${NS[$j]}; S=${SEEDS[$j]}
O=logs/attack/nsweepext_${SLURM_ARRAY_JOB_ID}/${COND}/n${N}_s${S}
echo "=== job ${SLURM_ARRAY_JOB_ID}_${i}  cond=$COND N=$N seed=$S veh='$VEH' extra='$EXTRA'  $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY attack_multi.py "$C" --out "$O" --mode perscene --mask all --cell 8 --place bbox \
    --inference-norm --fp-eval --vanish-form bce --k-dense 5 --k-prop 3 \
    --region-mode hull_saturate --w-region 1.0 --w-vanish 0 --w-margin 0 \
    --perscene-steps 100 --heldout 40 --paint-n $N --vehicles "$VEH" $EXTRA --seed $S
echo "=== nsweepext $COND n${N}_s${S} exit $?"
