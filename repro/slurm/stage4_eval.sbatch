#!/usr/bin/env bash
#SBATCH --job-name=bw_stage4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=02:00:00
## Pass your allocation at submit time:  sbatch --account=<ACCT> ...
## Use 'sla-prio' partition instead of 'standard' only if you hold an SLA allocation.
#
# Stage 4 — evaluate the noise: resume the victim's continual learning on the
# poisoned final task, then report BWT + accuracy (the numbers in Table 1).
# Writes acc_mat_*.npy and prints "After BWT / After avg acc / Last task acc".
#
# Submit from the repo root:   sbatch repro/slurm/stage4_eval.sbatch
set -eo pipefail   # NOTE: no '-u' — conda's activation scripts reference unset $PS1 in batch jobs

# ===== EDIT: match the method/lamb of the checkpoint you attacked ============
ENV_NAME="brainwash"
EXPERIMENT="split_cifar100"  # split_cifar100 | split_mini_imagenet | split_tiny_imagenet
APPROACH="ewc"               # ewc | mas | rwalk | afec_ewc
LAMB="500000"                # EWC=5e5(cifar)/1e4(mini), MAS=10(cifar)/5(mini), RWALK=1(cifar)/5(mini), AFEC=5e5/5e3
LAMB_EMP="100"               # only used by afec_ewc
NOISE_PKL="REPLACE_WITH_noise_*.pkl_FROM_STAGE_3"
# Variants for the Table 1 columns:
#   ours noise   -> flags below as-is (--addnoise, no --uniform)
#   uniform      -> add  --uniform   (keeps --addnoise)
#   clean        -> remove --addnoise entirely
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
    --lr 0.01 \
    --clip 100. \
    --lamb "${LAMB}" \
    ${AFEC_ARGS} \
    --checkpoint "${NOISE_PKL}" \
    --init_acc \
    --addnoise

echo "Stage 4 done. See the printed 'After BWT' / 'Last task acc' and the acc_mat_*.npy file."
