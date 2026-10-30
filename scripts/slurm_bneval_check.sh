#!/bin/bash
#SBATCH --job-name=av_bneval
#SBATCH -A eng_viva
#SBATCH -p gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/attack/bneval_%j.out
# Is the per-scene collapse a train-mode BatchNorm artifact? One GPU, runs in sequence.
cd "$SLURM_SUBMIT_DIR" || exit 1
source /home/tkg5kq/.bashrc >/dev/null 2>&1
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
export CUDA_HOME=/home/tkg5kq/.conda/envs/ssiai_adv
C=config/classification/classification4.py
O=logs/attack/bneval_$SLURM_JOB_ID
U=logs/attack/22118064
echo "=== job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
run() { N=$1; shift; echo; echo "################ $N"; $PY attack_multi.py $C --out $O/$N "$@"; echo "=== $N exit $?"; }
run ps_single_bneval    --mode perscene --mask single --bank 1 --cell 4 --inference-norm
run ps_single_trainnorm --mode perscene --mask single --bank 1 --cell 4
run rescore_uni_multi_shared_c8_bneval  --mode universal --mask all    --bank 1 --cell 8 --steps 0 --inference-norm --load-bank $U/uni_multi_shared_c8/bank_last.npy
run rescore_uni_multi_shared_c4_bneval  --mode universal --mask all    --bank 1 --cell 4 --steps 0 --inference-norm --load-bank $U/uni_multi_shared_c4/bank_last.npy
run rescore_uni_multi_bank4_c4_bneval   --mode universal --mask all    --bank 4 --cell 4 --steps 0 --inference-norm --load-bank $U/uni_multi_bank4_c4/bank_last.npy
run rescore_uni_single_shared_c8_bneval --mode universal --mask single --bank 1 --cell 8 --steps 0 --inference-norm --load-bank $U/uni_single_shared_c8/bank_last.npy
run ps_multi_bneval     --mode perscene --mask all --bank 1 --cell 4 --inference-norm
