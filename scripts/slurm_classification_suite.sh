#!/bin/bash
#SBATCH --job-name=ablation_cls
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=tkg5kq@virginia.edu
#SBATCH --array=0-5%6

# Classification (ClassAttackLoss) ablation suite
# classification0: baseline (original focal)
# classification1: reverse focal
# classification2: complement loss
# classification3: uniform confusion
# classification4: margin confusion
# classification5: hard wrong class

CONFIGS=(
    config/classification/classification0.py
    config/classification/classification1.py
    config/classification/classification2.py
    config/classification/classification3.py
    config/classification/classification4.py
    config/classification/classification5.py
)

NAMES=(
    classification0
    classification1
    classification2
    classification3
    classification4
    classification5
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
WORK_DIR="work_dirs/classification/${NAME}"

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
echo "Classification ablation suite — task ${SLURM_ARRAY_TASK_ID}" | tee -a "${logfile}"
echo "Config   : ${CONFIG}"                                         | tee -a "${logfile}"
echo "Work dir : ${WORK_DIR}"                                       | tee -a "${logfile}"
echo "Host     : $(hostname)"                                       | tee -a "${logfile}"
echo "======================================================" | tee -a "${logfile}"

# Use a unique port per array task to avoid torchrun conflicts on shared nodes
export PORT=$((29510 + SLURM_ARRAY_TASK_ID))

source activate ssiai_adv
bash scripts/dist_train.sh "${CONFIG}" 4 --work-dir "${WORK_DIR}" >> "${logfile}" 2>&1

echo "Done: $(date)" | tee -a "${logfile}"
