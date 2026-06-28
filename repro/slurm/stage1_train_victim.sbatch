#!/usr/bin/env bash
#SBATCH --job-name=bw_stage1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=12:00:00
## Pass your allocation at submit time:  sbatch --account=<ACCT> ...
#
# Stage 1 — train the victim continual learner on tasks 0..8 (the 10th task is
# left for poisoning). OPTIONAL for CIFAR-100 / miniImageNet: the authors already
# provide these checkpoints on HuggingFace, so you normally skip this and go
# straight to stage 3. Run this only to retrain from scratch or for tinyImageNet.
#
# Submit from the repo root:   sbatch repro/slurm/stage1_train_victim.sbatch
set -eo pipefail   # NOTE: no '-u' — conda's activation scripts reference unset $PS1 in batch jobs

# ===== EDIT ==================================================================
ENV_NAME="brainwash"
EXPERIMENT="split_cifar100"  # split_cifar100 | split_mini_imagenet | split_tiny_imagenet
APPROACH="ewc"               # ewc | mas | rwalk | afec_ewc
LAMB="500000"                # see per-method table in REPRODUCE.md
LAMB_EMP="100"               # afec_ewc only
# ============================================================================

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
export PYTHONUNBUFFERED=1   # stream python stdout to the .out log in real time (no buffering)
mkdir -p logs

AFEC_ARGS=""
[[ "${APPROACH}" == "afec_ewc" ]] && AFEC_ARGS="--lamb_emp ${LAMB_EMP}"

CUDA_VISIBLE_DEVICES=0 python main_baselines.py \
    --experiment "${EXPERIMENT}" \
    --approach "${APPROACH}" \
    --lasttask 9 \
    --tasknum 10 \
    --nepochs 20 \
    --batch-size 16 \
    --lamb "${LAMB}" \
    ${AFEC_ARGS} \
    --clip 100. \
    --lr 0.01

echo "Stage 1 done. A <approach>_*.pkl victim checkpoint was written to the repo root."
