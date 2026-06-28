#!/usr/bin/env bash
#SBATCH --job-name=bw_s4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_3g:1
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Parameterized STAGE 4 — evaluate noise by resuming CL on the poisoned task; report BWT/Acc.
#
# Driven by env vars (pass via submit_sweep.sh):
#   sbatch --account=<ACCT> --export=ALL,EVAL_TAG=...,VARIANT=...,EXPERIMENT=...,APPROACH=...,
#          LAMB=...,NOISE_TAG=...[,LAMB_EMP=,LASTTASK=,TASKNUM=] stage4.sbatch
# Required: EVAL_TAG VARIANT(clean|uniform|ours) EXPERIMENT APPROACH LAMB NOISE_TAG
# Optional: LAMB_EMP(-) LASTTASK(9) TASKNUM(10) NEPOCHS(20) ENV_NAME(brainwash)
#
# NOISE_TAG = which repro/sweep/noise/<tag>/ dir to read the noise pkl from. clean/uniform still
# need a pkl (for pretrained_ckpt + delta), so point them at a reckless run of matching dataset/method/eps.
set -eo pipefail

: "${ENV_NAME:=brainwash}"
: "${LAMB_EMP:=-}"
: "${LASTTASK:=9}"
: "${TASKNUM:=10}"
: "${NEPOCHS:=20}"
: "${SEED:=0}"
: "${EVAL_TAG:?set EVAL_TAG (e.g. cifar100_ewc_uniform_eps0.3)}"
: "${VARIANT:?set VARIANT (clean|uniform|ours)}"
: "${EXPERIMENT:?set EXPERIMENT (split_cifar100|split_mini_imagenet|split_tiny_imagenet)}"
: "${APPROACH:?set APPROACH (ewc|mas|rwalk|afec_ewc)}"
: "${LAMB:?set LAMB}"
: "${NOISE_TAG:?set NOISE_TAG (noise dir under repro/sweep/noise/)}"

# robust conda activation (batch name-resolution can silently no-op -> "No module named torch")
: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"   # Roar Collab base anaconda
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
# auto-detect the 'brainwash' env path (portable across users); override with ENV_PREFIX=/path/to/envs/brainwash
: "${ENV_PREFIX:=$(conda env list 2>/dev/null | awk '/\/brainwash$/{print $NF; exit}')}"
conda activate "${ENV_PREFIX:-brainwash}"
python -c "import torch" >/dev/null 2>&1 || { echo "FATAL: torch unavailable after conda activate '${ENV_PREFIX:-brainwash}' (set ENV_PREFIX=/path/to/envs/brainwash)"; exit 99; }
export PYTHONUNBUFFERED=1
mkdir -p logs repro/sweep/results

# locate the noise pkl (newest = best/final for that config)
NOISE_PKL=$(ls -t "repro/sweep/noise/${NOISE_TAG}"/*.pkl 2>/dev/null | head -1 || true)
[[ -n "${NOISE_PKL}" && -f "${NOISE_PKL}" ]] || { echo "MISSING noise pkl in repro/sweep/noise/${NOISE_TAG}/ (run stage 3 first)"; exit 2; }

# variant -> noise flags
case "${VARIANT}" in
  clean)   NOISE_ARGS="" ;;
  uniform) NOISE_ARGS="--addnoise --uniform" ;;
  ours)    NOISE_ARGS="--addnoise" ;;
  *) echo "bad VARIANT=${VARIANT}"; exit 2 ;;
esac

# afec_ewc needs lamb_emp
EMP_ARGS=""
[[ "${APPROACH}" == "afec_ewc" && "${LAMB_EMP}" != "-" ]] && EMP_ARGS="--lamb_emp ${LAMB_EMP}"

echo "[stage4] EVAL_TAG=${EVAL_TAG} VARIANT=${VARIANT} APPROACH=${APPROACH} LAMB=${LAMB} ${EMP_ARGS}"
echo "[stage4] NOISE_PKL=${NOISE_PKL}"

CUDA_VISIBLE_DEVICES=0 python main_baselines.py \
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
    ${NOISE_ARGS} 2>&1 | tee "repro/sweep/results/${EVAL_TAG}.log"

echo "[stage4] done -> repro/sweep/results/${EVAL_TAG}.log (and acc_mat_*.npy in repo root)"
