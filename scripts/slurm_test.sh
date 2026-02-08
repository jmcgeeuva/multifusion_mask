#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20 # number of cores/processors
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=tkg5kq@virginia.edu

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    mkdir -p "$LOG_DIR/$SLURM_ARRAY_JOB_ID"
else
    now=$(date +"%y%m%d-%H%M%S")
    mkdir -p "$LOG_DIR/$now"
fi

# configure log file path
# Check if SLURM_ARRAY_JOB_ID is set and not empty
if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    now=$(date +"%y%m%d")
    logpath="${LOG_DIR}/$SLURM_ARRAY_JOB_ID/logs-$now-${SLURM_ARRAY_JOB_ID}"
    mkdir -p $logpath
    logfile="$logpath/${SLURM_ARRAY_TASK_ID}.out"
else
    # Use the last argument as the log file if SLURM_ARRAY_JOB_ID is not set
    now=$(date +"%y%m%d-%H%M%S")
    logfile="${LOG_DIR}/$now/logs-$now.out"
fi

source /home/tkg5kq/.bashrc > "${logfile}" 2>&1
source activate bound >> "${logfile}" 2>&1

echo "Running $@ with ID ${SLURM_ARRAY_TASK_ID} ..."

if [ -f $1 ]; then
  config=$1
  echo "Running job with:"
  echo "  ARG1 = $ARG1"
else
  echo "need a config file"
  exit
fi

bash scripts/dist_test.sh ${config} 4 --eval bbox 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log >> "${logfile}" 2>&1

sleep 45
