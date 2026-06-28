#!/usr/bin/env bash
#SBATCH --job-name=bw_s3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_3g:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=24:00:00   # miniImageNet noise is ~4x slower than CIFAR (~23h for 5000 epochs)
#
# Parameterized STAGE 3 — train the BrainWash noise (bilevel optimization).
# Stage 3 is CL-method-agnostic: it only needs the victim checkpoint + inverted data,
# so no --approach/--lamb here (those are read from the pkl / used only at eval time).
#
# Driven by env vars (pass via submit_sweep.sh, or by hand):
#   sbatch --account=<ACCT> --export=ALL,TAG=...,PKL=...,INVDIR=...,MODE=...,DELTA=... stage3.sbatch
# Required: TAG PKL INVDIR MODE DELTA      Optional: SEED(0) NEPOCHS(5000) ENV_NAME(brainwash)
#
# Output: noise pkl(s) moved to repro/sweep/noise/$TAG/  (stage 4 reads the newest from there).
set -eo pipefail

: "${ENV_NAME:=brainwash}"
: "${SEED:=0}"
: "${NEPOCHS:=5000}"
: "${SAVE_EVERY:=2500}"   # snapshot interval; noise is written straight to BW_OUT_DIR (-> /storage/work)
: "${TAG:?set TAG (e.g. cifar100_ewc_reckless_eps0.3)}"
: "${PKL:?set PKL (victim checkpoint path)}"
: "${INVDIR:?set INVDIR (inverted-data folder)}"
: "${MODE:?set MODE (reckless|cautious)}"
: "${DELTA:?set DELTA (0.1|0.3)}"

# robust conda activation (batch name-resolution can silently no-op -> "No module named torch")
: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"   # Roar Collab base anaconda
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
# auto-detect the 'brainwash' env path (portable across users); override with ENV_PREFIX=/path/to/envs/brainwash
: "${ENV_PREFIX:=$(conda env list 2>/dev/null | awk '/\/brainwash$/{print $NF; exit}')}"
conda activate "${ENV_PREFIX:-brainwash}"
python -c "import torch" >/dev/null 2>&1 || { echo "FATAL: torch unavailable after conda activate '${ENV_PREFIX:-brainwash}' (set ENV_PREFIX=/path/to/envs/brainwash)"; exit 99; }
export PYTHONUNBUFFERED=1
mkdir -p logs "repro/sweep/noise/${TAG}"

nvidia-smi || true
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# sanity: inputs exist
[[ -f "${PKL}" ]]    || { echo "MISSING checkpoint: ${PKL}"; exit 2; }
[[ -d "${INVDIR}" ]] || { echo "MISSING inverted-data dir: ${INVDIR}"; exit 2; }

EXTRA_ARGS=""
[[ "${MODE}" == "cautious" ]] && EXTRA_ARGS="--w_cur 1"

echo "[stage3] TAG=${TAG} MODE=${MODE} DELTA=${DELTA} SEED=${SEED} NEPOCHS=${NEPOCHS}"
echo "[stage3] PKL=${PKL}"
echo "[stage3] INVDIR=${INVDIR}"

export BW_OUT_DIR="repro/sweep/noise/${TAG}"   # main_brainwash writes noise straight here (-> /storage/work via symlink), never home CWD
CUDA_VISIBLE_DEVICES=0 python main_brainwash.py \
    --extra_desc="${TAG}" \
    --pretrained_model_add="${PKL}" \
    --mode="${MODE}" \
    --target_task_for_eval=0 \
    --delta="${DELTA}" \
    --seed="${SEED}" \
    --eval_every=10 \
    --distill_folder="${INVDIR}" \
    --init_acc \
    --noise_norm=inf \
    --cont_learner_lr=0.001 \
    --n_epochs="${NEPOCHS}" \
    --save_every="${SAVE_EVERY}" \
    ${EXTRA_ARGS}

# main_brainwash wrote noise straight into the dir (BW_OUT_DIR); keep only the best (newest) snapshot
ls -t "repro/sweep/noise/${TAG}"/*.pkl 2>/dev/null | tail -n +2 | xargs -r rm -f
n=$(ls "repro/sweep/noise/${TAG}"/*.pkl 2>/dev/null | wc -l)
echo "[stage3] done. noise in repro/sweep/noise/${TAG}/ (kept ${n})"
ls -lht "repro/sweep/noise/${TAG}/" 2>/dev/null | head
[[ "${n}" -gt 0 ]] || { echo "WARNING: no noise pkl produced in repro/sweep/noise/${TAG}/"; exit 3; }
