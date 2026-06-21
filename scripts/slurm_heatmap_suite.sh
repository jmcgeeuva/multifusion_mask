#!/bin/bash
#SBATCH --job-name=ablation_heatmap
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=tkg5kq@virginia.edu
#SBATCH --array=0-6%7
#SBATCH --exclude=udc-aw37-1,udc-aw37-6,udc-aw37-10,udc-aw37-14,udc-aw38-2,udc-aw38-6,udc-aw38-10,udc-ba11-3,udc-ba11-7,udc-ba11-11,udc-ba11-15,udc-ba11-19,udc-ba11-23,udc-ba11-27,udc-an25-20,udc-an25-24,udc-an38-1,udc-an38-5,udc-an38-9,udc-an38-13,udc-an38-17,udc-an38-25,udc-an38-29,udc-an38-33,udc-an40-13,udc-an40-17,udc-an40-25,udc-an40-29

# Heatmap (GIADLoss) ablation suite
# heatmap0: baseline (original gaussian)
# heatmap1: reverse gaussian
# heatmap2: attention diffusion
# heatmap3: entropy
# heatmap4: contrast
# heatmap5: veiling luminance
# heatmap6: ring loss

CONFIGS=(
    config/heatmap/heatmap0.py
    config/heatmap/heatmap1.py
    config/heatmap/heatmap2.py
    config/heatmap/heatmap3.py
    config/heatmap/heatmap4.py
    config/heatmap/heatmap5.py
    config/heatmap/heatmap6.py
)

NAMES=(
    heatmap0
    heatmap1
    heatmap2
    heatmap3
    heatmap4
    heatmap5
    heatmap6
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
WORK_DIR="work_dirs/heatmap/${NAME}"

mkdir -p "${WORK_DIR}"

# Log file setup (mirrors slurm_test.sh pattern)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    now=$(date +"%y%m%d")
    logpath="${LOG_DIR}/${SLURM_ARRAY_JOB_ID}/logs-${now}-${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$logpath"
    logfile="${logpath}/${SLURM_ARRAY_TASK_ID}.out"
else
    now=$(date +"%y%m%d-%H%M%S")
    mkdir -p "${LOG_DIR}/${now}"
    logfile="${LOG_DIR}/${now}/logs-${now}.out"
fi

source /home/tkg5kq/.bashrc >> "${logfile}" 2>&1
conda activate ssiai_adv >> "${logfile}" 2>&1

echo "======================================================" | tee -a "${logfile}"
echo "Heatmap ablation suite — task ${SLURM_ARRAY_TASK_ID}" | tee -a "${logfile}"
echo "Config   : ${CONFIG}"                                  | tee -a "${logfile}"
echo "Work dir : ${WORK_DIR}"                                | tee -a "${logfile}"
echo "Host     : $(hostname)"                                | tee -a "${logfile}"
echo "======================================================" | tee -a "${logfile}"

# Use a unique port per array task to avoid torchrun conflicts on shared nodes
export PORT=$((29500 + SLURM_ARRAY_TASK_ID))

source activate ssiai_adv
bash scripts/dist_train.sh "${CONFIG}" 2 --work-dir "${WORK_DIR}" >> "${logfile}" 2>&1

echo "Done: $(date)" | tee -a "${logfile}"
