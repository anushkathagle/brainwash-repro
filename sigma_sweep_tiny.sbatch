#!/usr/bin/env bash
#SBATCH --job-name=bw_sweep_tiny
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Training-free sigma_star sweep + seen/unseen memorization check for the
# Tiny-ImageNet ImageNet-64 purifier (sigma_sweep_tiny.py). Run this BEFORE any
# stage4_purify_tiny.sbatch job.
# COST (measured, A100, fp32, full 4000-step schedule): ~2 ms per image per reverse
# step, and each sigma purifies 3 x N_IMAGES images (clean, perturbed, val). With
# N_IMAGES=256: sigma 0.1 ~6 min, 0.25 ~15, 0.5 ~29, 1.0 ~50 -> ~100 min for the
# default list. Split it with SIGMAS="..." if needed; each SIGMAS list gets its own
# TSV, and rows are saved after every sigma.
#
#   sbatch --account=<ACCT> sigma_sweep_tiny.sbatch
#   sbatch --account=<ACCT> --export=ALL,NOISE_TAG=tiny_ewc_reckless_eps0.3 sigma_sweep_tiny.sbatch
#
# Optional: SIGMAS("0 0.1 0.25 0.5 1.0") N_IMAGES(256) TASKNUM(20) TASK(TASKNUM-1)
#           NOISE_TAG (use the real stage-3 poison for survival; else a random-sign probe)
#           PROBE_EPS(0.0313725) RESPACE("") BACKEND(auto) FP16(0) TINY_NPZ PURIFY_CKPT
#           OUT_DIR ID_PATH ENV_PREFIX
set -eo pipefail
# All default paths below are relative to the repo root -- submit from there.
[[ -f sigma_sweep_tiny.py ]] || { echo "FATAL: run sbatch from the repo root (no ./sigma_sweep_tiny.py in $PWD)"; exit 1; }

: "${SIGMAS:=0 0.1 0.25 0.5 1.0}"
: "${N_IMAGES:=256}"
: "${TASKNUM:=20}"
: "${TASK:=$((TASKNUM - 1))}"
: "${NOISE_TAG:=}"
: "${PROBE_EPS:=0.0313725}"
: "${RESPACE:=}"
: "${BACKEND:=auto}"
: "${FP16:=0}"
: "${SEED:=0}"
: "${TINY_NPZ:=./data/tiny_imagenet.npz}"
: "${PURIFY_CKPT:=repro/artifacts/diffusion/imagenet64_uncond_100M_1500K.pt}"
: "${OUT_DIR:=repro/artifacts/tinyImagenet_files/sigma_sweep}"
: "${ID_PATH:=/storage/home/axt5884/improved-diffusion}"

# Fail fast (before conda) on missing inputs. Run ./check_tiny_setup.sh on the
# login node BEFORE submitting -- these checks only run once the job leaves the queue.
[[ -f "${PURIFY_CKPT}" ]] || { echo "MISSING purifier ckpt: ${PURIFY_CKPT}"; echo "  wget -P $(dirname "${PURIFY_CKPT}") https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt"; exit 2; }
[[ -f "${TINY_NPZ}" ]] || { echo "MISSING ${TINY_NPZ} (python repro/prepare_tiny_imagenet.py)"; exit 2; }

: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
: "${ENV_PREFIX:=/storage/home/axt5884/work/conda/myworkflow}"
conda activate "${ENV_PREFIX}"
export PYTHONPATH="${ID_PATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p logs "${OUT_DIR}"


EXTRA=()
TAG="probe"
if [[ -n "${NOISE_TAG}" ]]; then
  NOISE_PKL=$(ls -t "repro/sweep/noise/${NOISE_TAG}"/*.pkl 2>/dev/null | head -1 || true)
  [[ -n "${NOISE_PKL}" ]] || { echo "MISSING noise pkl in repro/sweep/noise/${NOISE_TAG}/"; exit 2; }
  EXTRA+=(--noise_pkl "${NOISE_PKL}"); TAG="${NOISE_TAG}"
fi
[[ -n "${RESPACE}" ]] && { EXTRA+=(--respace "${RESPACE}"); TAG="${TAG}_sp${RESPACE}"; }
[[ "${FP16}" == "1" ]] && EXTRA+=(--fp16)

python sigma_sweep_tiny.py \
    --purify_ckpt "${PURIFY_CKPT}" --tiny_npz "${TINY_NPZ}" \
    --tasknum "${TASKNUM}" --task "${TASK}" --sigmas ${SIGMAS} \
    --n_images "${N_IMAGES}" --probe_eps "${PROBE_EPS}" --backend "${BACKEND}" --seed "${SEED}" \
    --out "${OUT_DIR}/sweep_${TAG}_s${SIGMAS// /_}.tsv" "${EXTRA[@]}"
