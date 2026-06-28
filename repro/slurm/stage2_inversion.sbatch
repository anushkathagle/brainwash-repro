#!/usr/bin/env bash
#SBATCH --job-name=bw_stage2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=06:00:00
## Pass your allocation at submit time:  sbatch --account=<ACCT> ...
#
# Stage 2 — model-inversion attack: reconstruct approximate samples of tasks 0..8
# from the victim's weights. OPTIONAL for CIFAR-100 / miniImageNet (the authors
# provide inverted_data on HuggingFace). Run only to regenerate, or for tinyImageNet.
#
# Submit from the repo root:   sbatch repro/slurm/stage2_inversion.sbatch
set -eo pipefail   # NOTE: no '-u' — conda's activation scripts reference unset $PS1 in batch jobs

# ===== EDIT ==================================================================
ENV_NAME="brainwash"
PKL="REPLACE_WITH_victim_checkpoint.pkl"
SAVE_DIR="inv_data_ewc_cifar100_alltasks_resnet"  # becomes the --distill_folder for stage 3
# ============================================================================

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
export PYTHONUNBUFFERED=1   # stream python stdout to the .out log in real time (no buffering)
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python main_inv.py \
    --pretrained_model_add="${PKL}" \
    --num_samples=128 \
    --save_dir="${SAVE_DIR}" \
    --task_lst=0,1,2,3,4,5,6,7,8 \
    --save_every=1000 \
    --batch_reg \
    --init_acc \
    --n_iters=10000

echo "Stage 2 done. Inverted .npz files are in ${SAVE_DIR}/ — pass it as --distill_folder in stage 3."
