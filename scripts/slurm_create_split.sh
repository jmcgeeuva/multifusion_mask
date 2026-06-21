#!/bin/bash
#SBATCH --job-name=nusc_split
#SBATCH --time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH -A eng_viva
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=tkg5kq@virginia.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch scripts/slurm_create_split.sh --split-percent 10
#   sbatch scripts/slurm_create_split.sh --split-percent 5 --pkl ./data/nuscenes/nuscenes_infos_train.pkl
#
# All arguments after the script name are forwarded to create_diverse_split.py.
# The --num-workers flag defaults to $SLURM_CPUS_PER_TASK so it fills the
# allocation automatically.
# ---------------------------------------------------------------------------

set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
now=$(date +"%y%m%d-%H%M%S")
logfile="${LOG_DIR}/create_split_${now}.out"

exec > >(tee -a "$logfile") 2>&1

echo "=== nusc_split job ==================================="
echo "Date        : $(date)"
echo "Host        : $(hostname)"
echo "SLURM job   : ${SLURM_JOB_ID:-<local>}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-1}"
echo "Args        : $*"
echo "======================================================"

source /home/tkg5kq/.bashrc
conda activate ssiai_adv

cd "$(dirname "$(dirname "$(realpath "$0")")")"   # repo root
echo "Working dir : $(pwd)"

python scripts/create_diverse_split.py \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    "$@"

echo ""
echo "Done: $(date)"
