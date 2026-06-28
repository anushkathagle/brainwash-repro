#!/usr/bin/env bash
#SBATCH --job-name=bw_stage3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=08:00:00
## --- Roar Collab: full A100 on the 'standard' partition (p-gc-3001..3040). ---
## Use 'sla-prio' instead if your allocation is a paid SLA one (faster scheduling).
## Pass your allocation account at submit time:  sbatch --account=<ACCT> ...
## Find it with:  sacctmgr -n show assoc user=$USER format=account%30
## ---------------------------------------------------------------------------
#
# Stage 3 — train the BrainWash noise (the bilevel optimization). REQUIRED:
# the HuggingFace repo does NOT ship trained noise, so you must produce it here.
# Output: a noise_*.pkl that stage 4 consumes via --checkpoint.
#
# Submit from the repo root:   sbatch repro/slurm/stage3_brainwash.sbatch
set -eo pipefail   # NOTE: no '-u' — conda's activation scripts reference unset $PS1 in batch jobs

# ===== EDIT: the experiment you want to reproduce ============================
ENV_NAME="brainwash"
METHOD="ewc"                 # ewc | mas | rwalk | afec_ewc
DATASET_DIR="cifar100_files" # cifar100_files | miniImagenet_files
PKL="repro/artifacts/${DATASET_DIR}/cl_methods/ewc/ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_optim_name_sgd_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32____.pkl"
INV_DIR="repro/artifacts/${DATASET_DIR}/inverted_data/inv_data_ewc_cifar100_alltasks_resnet"
MODE="reckless"              # reckless | cautious
DELTA="0.3"                  # 0.3 -> Table 1 eps=0.3 column ; 0.1 -> eps=0.1 column
SEED="0"
EXTRA="${MODE}_eps${DELTA}_${METHOD}"
# For cautious mode, the paper uses eta (w_cur); add:  --w_cur 1
# ============================================================================

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
export PYTHONUNBUFFERED=1   # stream python stdout to the .out log in real time (no buffering)
mkdir -p logs

nvidia-smi || true
python - <<'PY'
import torch; print("CUDA avail:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

EXTRA_ARGS=""
[[ "${MODE}" == "cautious" ]] && EXTRA_ARGS="--w_cur 1"

CUDA_VISIBLE_DEVICES=0 python main_brainwash.py \
    --extra_desc="${EXTRA}" \
    --pretrained_model_add="${PKL}" \
    --mode="${MODE}" \
    --target_task_for_eval=0 \
    --delta="${DELTA}" \
    --seed="${SEED}" \
    --eval_every=10 \
    --distill_folder="${INV_DIR}" \
    --init_acc \
    --noise_norm=inf \
    --cont_learner_lr=0.001 \
    --n_epochs=5000 \
    --save_every=100 \
    ${EXTRA_ARGS}

echo "Stage 3 done. The noise_*.pkl in the repo root is the input to stage 4."
