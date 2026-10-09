#!/usr/bin/env bash
#SBATCH --job-name=pvq_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# Train one PureVQ-GAN purifier (defense layer 1). Env-driven; see repro/defense/README_purevqgan.md.
#   sbatch --account=<ACCT> --export=ALL,PVQ_TAG=cifar10_K512 repro/defense/train_purifier.sbatch
#   sbatch --account=<ACCT> --export=ALL,PVQ_TAG=cifar10_K64,K=64 repro/defense/train_purifier.sbatch
#   sbatch --account=<ACCT> --export=ALL,PVQ_TAG=stl10_84_K512,PVQ_DATASET=stl10,IMG_SIZE=84,LATENT=21 \
#          repro/defense/train_purifier.sbatch
# Required: PVQ_TAG   Optional: PVQ_DATASET(cifar10) IMG_SIZE(32) LATENT(8) K(512) BASE_CH(128)
#   NUM_RES_BLOCKS(1) GAN_WEIGHT(0.1) EPOCHS(100) BATCH(256) LR(4e-4) PVQ_SEED(0) EXTRA_ARGS("", e.g. "--restart_dead_codes 0")
#   PVQ_ROOT(purifiers)  -> output in $PVQ_ROOT/$PVQ_TAG/purifier.pt  (resumable: re-submit to continue)
set -eo pipefail

: "${PVQ_TAG:?set PVQ_TAG (e.g. cifar10_K512)}"
: "${PVQ_DATASET:=cifar10}"
: "${IMG_SIZE:=32}"
: "${LATENT:=8}"
: "${K:=512}"
: "${BASE_CH:=128}"
: "${NUM_RES_BLOCKS:=1}"
: "${GAN_WEIGHT:=0.1}"
: "${EPOCHS:=100}"
: "${BATCH:=256}"
: "${LR:=4e-4}"
: "${PVQ_SEED:=0}"
: "${EXTRA_ARGS:=}"
: "${PVQ_ROOT:=purifiers}"

: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"   # Roar Collab base anaconda
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
: "${ENV_PREFIX:=$(conda env list 2>/dev/null | awk '/\/brainwash$/{print $NF; exit}')}"
conda activate "${ENV_PREFIX:-brainwash}"
python -c "import torch" >/dev/null 2>&1 || { echo "FATAL: torch unavailable after conda activate '${ENV_PREFIX:-brainwash}'"; exit 99; }
export PYTHONUNBUFFERED=1
mkdir -p logs "${PVQ_ROOT}"

OUT="${PVQ_ROOT}/${PVQ_TAG}"
if [[ -f "${OUT}/purifier.pt" ]]; then echo "[pvq] ${OUT}/purifier.pt exists - nothing to do"; exit 0; fi
echo "[pvq] tag=${PVQ_TAG} data=${PVQ_DATASET} ${IMG_SIZE}->${LATENT} K=${K} base_ch=${BASE_CH} lambda=${GAN_WEIGHT} epochs=${EPOCHS}"

python -m purevqgan.train \
    --dataset "${PVQ_DATASET}" --img_size "${IMG_SIZE}" --latent_size "${LATENT}" \
    --num_codes "${K}" --base_ch "${BASE_CH}" --num_res_blocks "${NUM_RES_BLOCKS}" \
    --gan_weight "${GAN_WEIGHT}" --epochs "${EPOCHS}" --batch_size "${BATCH}" --lr "${LR}" \
    --seed "${PVQ_SEED}" --amp --resume --out "${OUT}" ${EXTRA_ARGS}
