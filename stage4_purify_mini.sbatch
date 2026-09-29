#!/usr/bin/env bash
#SBATCH --job-name=bw_s4_purify
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# Layer-1a DiffPure STAGE 4 -- same shape/variable naming as repro/sweep/stage4.sbatch,
# but runs purify_mini_layer1a.py (main_baselines.py + a purification hook) instead of
# main_baselines.py directly. Submit with sbatch, exactly like stage3/stage4 -- never
# salloc for a real run; salloc is fine only for the sub-10-minute smoke test.
#
#   sbatch --account=<ACCT> --export=ALL,EVAL_TAG=...,VARIANT=...,NOISE_TAG=...,SIGMA_STAR=... stage4_purify.sbatch
#
# Required: EVAL_TAG VARIANT(clean|uniform|ours) NOISE_TAG
# Optional: EXPERIMENT(split_mini_imagenet) APPROACH(ewc) LAMB(10000) LAMB_EMP(-)
#           LASTTASK(9) TASKNUM(10) NEPOCHS(20) SEED(0) SIGMA_STAR(0.5)
#           PURIFY_CHUNK(8) PURIFY_CKPT_EVERY(1) FP32(0|1) GD_PATH ENV_PREFIX
#
# ============================================================================
# RESUMABILITY -- this is the reason this script exists as a separate submission
# from stage4.sbatch's plain eval path. Purification alone can take hours (full
# ancestral DDPM sampling at 256x256, no step-skipping), which is longer than a
# short/contended salloc-style GPU slot, and longer than this script's own
# --time budget can comfortably guarantee on a busy queue.
#
# --purify_cache_dir is set below by default (OUT_DIR/cache), which turns on
# BOTH result caching AND checkpointed resumability in purify_mini_layer1a.py:
# progress is saved to disk after every chunk (see PURIFY_CKPT_EVERY). If this
# job is killed for ANY reason -- hits --time, the node is reclaimed, whatever --
# just re-submit the EXACT SAME sbatch command again. It will skip every chunk
# already done and continue from the last checkpoint, not from image 0.
#
# Practical use: if you can only reliably get ~3h GPU slots, submit this
# repeatedly (same EVAL_TAG/NOISE_TAG/SIGMA_STAR) until the log shows
# "After BWT" -- each submission makes real forward progress on the same task.
# ============================================================================
set -eo pipefail

: "${EVAL_TAG:?set EVAL_TAG (e.g. mini_ewc_reckless_eps0.3_purify_sigma0.5)}"
: "${VARIANT:?set VARIANT (clean|uniform|ours)}"
: "${NOISE_TAG:?set NOISE_TAG (dir under repro/sweep/noise/, e.g. mini_ewc_reckless_eps0.3)}"
: "${EXPERIMENT:=split_mini_imagenet}"
: "${APPROACH:=ewc}"
: "${LAMB:=10000}"
: "${LAMB_EMP:=-}"
: "${LASTTASK:=9}"
: "${TASKNUM:=10}"
: "${NEPOCHS:=20}"
: "${SEED:=0}"
: "${SIGMA_STAR:=0.5}"
: "${PURIFY_RESIZE:=256}"
: "${PURIFY_CHUNK:=8}"
: "${PURIFY_CKPT_EVERY:=1}"
: "${FP32:=0}"
: "${PURIFY_CKPT:=/storage/work/axt5884/projects/myworkflow/brainwash-repro/repro/artifacts/diffusion/256x256_diffusion_uncond.pt}"
: "${OUT_DIR:=/storage/work/axt5884/projects/myworkflow/brainwash-repro/repro/artifacts/miniImagenet_files/purify_results}"
: "${PURIFY_CACHE_DIR:=${OUT_DIR}/cache}"   # on by default -- this is what makes resumability actually happen
: "${RESULTS_DIR:=repro/sweep/results_purify}"   # separate from results/ so purify logs never collide with undefended stage4.sbatch logs
: "${GD_PATH:=/storage/home/axt5884/guided-diffusion}"

# --- conda activation (same pattern as repro/sweep/stage3.sbatch / stage4.sbatch) ---
: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
: "${ENV_PREFIX:=/storage/home/axt5884/work/conda/myworkflow}"
conda activate "${ENV_PREFIX}"
python -c "import torch" >/dev/null 2>&1 || { echo "FATAL: torch unavailable after activating ${ENV_PREFIX}"; exit 99; }

export PYTHONPATH="${GD_PATH}:${PYTHONPATH:-}"
python -c "import guided_diffusion" >/dev/null 2>&1 || {
  echo "FATAL: cannot import guided_diffusion from GD_PATH=${GD_PATH}"
  echo "  git clone https://github.com/openai/guided-diffusion.git ${GD_PATH}"
  exit 98
}

export PYTHONUNBUFFERED=1
mkdir -p logs "${OUT_DIR}" "${PURIFY_CACHE_DIR}" "${RESULTS_DIR}"

nvidia-smi || true
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# locate the noise pkl (newest = best/final for that config) -- same lookup as stage4.sbatch
NOISE_PKL=$(ls -t "repro/sweep/noise/${NOISE_TAG}"/*.pkl 2>/dev/null | head -1 || true)
[[ -n "${NOISE_PKL}" && -f "${NOISE_PKL}" ]] || { echo "MISSING noise pkl in repro/sweep/noise/${NOISE_TAG}/ (run stage 3 first)"; exit 2; }
[[ -f "${PURIFY_CKPT}" ]] || { echo "MISSING purifier ckpt: ${PURIFY_CKPT}"; exit 2; }

case "${VARIANT}" in
  clean)   NOISE_ARGS="" ;;
  uniform) NOISE_ARGS="--addnoise --uniform" ;;
  ours)    NOISE_ARGS="--addnoise" ;;
  *) echo "bad VARIANT=${VARIANT} (clean|uniform|ours)"; exit 2 ;;
esac

EMP_ARGS=""
[[ "${APPROACH}" == "afec_ewc" && "${LAMB_EMP}" != "-" ]] && EMP_ARGS="--lamb_emp ${LAMB_EMP}"

FP32_ARGS=""
[[ "${FP32}" == "1" ]] && FP32_ARGS="--purify_fp32"

echo "[stage4_purify] EVAL_TAG=${EVAL_TAG} VARIANT=${VARIANT} SIGMA_STAR=${SIGMA_STAR} SEED=${SEED} FP32=${FP32}"
echo "[stage4_purify] NOISE_PKL=${NOISE_PKL}"
echo "[stage4_purify] OUT_DIR=${OUT_DIR}"
echo "[stage4_purify] PURIFY_CACHE_DIR=${PURIFY_CACHE_DIR}  (resumability lives here -- do not delete between resubmits)"

CUDA_VISIBLE_DEVICES=0 python purify_mini_layer1a.py \
    --experiment "${EXPERIMENT}" \
    --approach "${APPROACH}" \
    --lasttask "${LASTTASK}" \
    --tasknum "${TASKNUM}" \
    --nepochs "${NEPOCHS}" \
    --batch-size 16 \
    --lr 0.01 \
    --clip 100. \
    --lamb "${LAMB}" \
    --seed "${SEED}" \
    ${EMP_ARGS} \
    --checkpoint "${NOISE_PKL}" \
    --init_acc \
    --out_dir "${OUT_DIR}" \
    --purify --sigma_star "${SIGMA_STAR}" \
    --purify_ckpt "${PURIFY_CKPT}" \
    --purify_resize "${PURIFY_RESIZE}" \
    --purify_chunk "${PURIFY_CHUNK}" \
    --purify_cache_dir "${PURIFY_CACHE_DIR}" \
    --purify_ckpt_every "${PURIFY_CKPT_EVERY}" \
    ${FP32_ARGS} \
    ${NOISE_ARGS} 2>&1 | tee -a "${RESULTS_DIR}/${EVAL_TAG}.log"

echo "[stage4_purify] done -> ${RESULTS_DIR}/${EVAL_TAG}.log, ${OUT_DIR}/summary.tsv"
