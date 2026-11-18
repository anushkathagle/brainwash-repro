#!/usr/bin/env bash
#SBATCH --job-name=bw_s4_purify_tiny
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
# Layer-1a DiffPure STAGE 4 for TINY-IMAGENET -- same shape/variable naming as
# stage4_purify_mini.sbatch, but runs purify_tiny_layer1a.py with the native-
# resolution improved-diffusion ImageNet-64 UNCONDITIONAL purifier (no resize).
#
#   sbatch --account=<ACCT> --time=06:00:00 --export=ALL,VARIANT=clean,NOISE_TAG=<noise dir> stage4_purify_tiny.sbatch
#
# Required: VARIANT(clean|uniform|ours) NOISE_TAG
# Read from the noise pkl automatically (the values the victim was trained with;
# the training code takes lamb/lamb_emp from the pkl anyway): APPROACH LAMB LAMB_EMP.
#   Setting one of them by hand is allowed only if it AGREES with the pkl.
# EVAL_TAG defaults to <NOISE_TAG>_<VARIANT>_purify64_sigma<SIGMA_STAR>[_sp<RESPACE>].
# Optional: EXPERIMENT(split_tiny_imagenet)
#           LASTTASK(19) TASKNUM(20) NEPOCHS(20) SEED(0) SIGMA_STAR(0.5)
#           RESPACE("" = full 4000 steps | 1000) PURIFY_CHUNK(128) PURIFY_CKPT_EVERY(1)
#           FP16(0|1) BACKEND(auto|improved|guided) TINY_NPZ(./data/tiny_imagenet.npz)
#           ID_PATH(improved-diffusion clone) ENV_PREFIX
#
# Run sigma_sweep_tiny.sbatch and an Arm D (VARIANT=clean) job FIRST; fix SIGMA_STAR
# from those, never from the attacked arm.
#
# RESUMABILITY: identical mechanism to stage4_purify_mini.sbatch -- PURIFY_CACHE_DIR
# holds per-chunk progress; if the job dies, re-submit the EXACT same command.
set -eo pipefail
# All default paths below are relative to the repo root -- submit from there.
[[ -f purify_tiny_layer1a.py ]] || { echo "FATAL: run sbatch from the repo root (no ./purify_tiny_layer1a.py in $PWD)"; exit 1; }

: "${VARIANT:?set VARIANT (clean|uniform|ours)}"
: "${NOISE_TAG:?set NOISE_TAG (dir under repro/sweep/noise/, e.g. tiny_ewc_reckless_eps0.3)}"
: "${EXPERIMENT:=split_tiny_imagenet}"
: "${LASTTASK:=19}"
: "${TASKNUM:=20}"
: "${NEPOCHS:=20}"
: "${SEED:=0}"
: "${SIGMA_STAR:=0.5}"
: "${RESPACE:=}"
: "${PURIFY_CHUNK:=128}"
: "${PURIFY_CKPT_EVERY:=1}"
: "${FP16:=0}"
: "${BACKEND:=auto}"
: "${TINY_NPZ:=./data/tiny_imagenet.npz}"
: "${PURIFY_CKPT:=repro/artifacts/diffusion/imagenet64_uncond_100M_1500K.pt}"
: "${OUT_DIR:=repro/artifacts/tinyImagenet_files/purify_results}"
: "${PURIFY_CACHE_DIR:=${OUT_DIR}/cache}"   # on by default -- this is what makes resumability actually happen
: "${RESULTS_DIR:=repro/sweep/results_purify}"   # separate from results/ so purify logs never collide with undefended stage4.sbatch logs
: "${ID_PATH:=/storage/home/axt5884/improved-diffusion}"

# --- conda activation (same pattern as repro/sweep/stage3.sbatch / stage4.sbatch) ---
# Fail fast (before conda) on missing inputs -- see ./check_tiny_setup.sh.
[[ -f "${PURIFY_CKPT}" ]] || { echo "MISSING purifier ckpt: ${PURIFY_CKPT}"; echo "  wget -P $(dirname "${PURIFY_CKPT}") https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt"; exit 2; }
[[ -f "${TINY_NPZ}" ]] || { echo "MISSING ${TINY_NPZ} (python repro/prepare_tiny_imagenet.py)"; exit 2; }

: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true; eval "$(conda shell.bash hook)"; fi
: "${ENV_PREFIX:=/storage/home/axt5884/work/conda/myworkflow}"
conda activate "${ENV_PREFIX}"
echo "[stage4_purify_tiny] conda env: ${CONDA_PREFIX:-?}  (ENV_PREFIX=${ENV_PREFIX})  python: $(command -v python)  account: ${SLURM_JOB_ACCOUNT:-?}"
python -c "import torch" >/dev/null 2>&1 || { echo "FATAL: torch unavailable after activating ${ENV_PREFIX}"; exit 99; }

export PYTHONPATH="${ID_PATH}:${PYTHONPATH:-}"
# BACKEND=guided relies on guided_diffusion already being on PYTHONPATH (as for mini).
BACKEND_MOD=improved_diffusion; [[ "${BACKEND}" == "guided" ]] && BACKEND_MOD=guided_diffusion
python -c "import ${BACKEND_MOD}.script_util" >/dev/null 2>&1 || {
  echo "FATAL: cannot import ${BACKEND_MOD} (ID_PATH=${ID_PATH})"
  echo "  git clone https://github.com/openai/improved-diffusion.git ${ID_PATH}"
  exit 98
}

export PYTHONUNBUFFERED=1
mkdir -p logs "${OUT_DIR}" "${PURIFY_CACHE_DIR}" "${RESULTS_DIR}"

nvidia-smi || true
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# locate the noise pkl (newest = best/final for that config) -- same lookup as stage4.sbatch
NOISE_PKL=$(ls -t "repro/sweep/noise/${NOISE_TAG}"/*.pkl 2>/dev/null | head -1 || true)
[[ -n "${NOISE_PKL}" && -f "${NOISE_PKL}" ]] || { echo "MISSING noise pkl in repro/sweep/noise/${NOISE_TAG}/ (run stage 3 first)"; exit 2; }

# --- APPROACH / LAMB / LAMB_EMP from the noise pkl (single source of truth) ---
read -r PKL_APPROACH PKL_LAMB PKL_LAMB_EMP < <(python - "${NOISE_PKL}" <<'PY'
import pickle, sys
a = pickle.load(open(sys.argv[1], 'rb'))['pretrained_ckpt']['cont_method_args']
print(a.get('method', a.get('approach')), a['lamb'], a.get('lamb_emp', '-'))
PY
)
[[ -n "${PKL_APPROACH}" && -n "${PKL_LAMB}" ]] || { echo "FATAL: could not read approach/lamb from ${NOISE_PKL}"; exit 2; }
for pair in "APPROACH:${PKL_APPROACH}" "LAMB:${PKL_LAMB}"; do
  k=${pair%%:*}; v=${pair#*:}
  if [[ -n "${!k:-}" ]] && ! python -c "import sys; a,b=sys.argv[1:]; sys.exit(0 if a==b or (a.replace('.','',1).replace('e','',1).lstrip('-').isdigit() and float(a)==float(b)) else 1)" "${!k}" "$v"; then
    echo "FATAL: you set ${k}=${!k} but the noise pkl says ${v} -- wrong NOISE_TAG?"; exit 2
  fi
done
APPROACH="${PKL_APPROACH}"; LAMB="${PKL_LAMB}"; LAMB_EMP="${PKL_LAMB_EMP}"
_sp=""; [[ -n "${RESPACE}" ]] && _sp="_sp${RESPACE}"
: "${EVAL_TAG:=${NOISE_TAG}_${VARIANT}_purify64_sigma${SIGMA_STAR}${_sp}}"
echo "[stage4_purify_tiny] from pkl: APPROACH=${APPROACH} LAMB=${LAMB} LAMB_EMP=${LAMB_EMP}"

case "${VARIANT}" in
  clean)   NOISE_ARGS="" ;;
  uniform) NOISE_ARGS="--addnoise --uniform" ;;
  ours)    NOISE_ARGS="--addnoise" ;;
  *) echo "bad VARIANT=${VARIANT} (clean|uniform|ours)"; exit 2 ;;
esac

EMP_ARGS=""
[[ "${APPROACH}" == "afec_ewc" && "${LAMB_EMP}" != "-" ]] && EMP_ARGS="--lamb_emp ${LAMB_EMP}"

FP16_ARGS=""
[[ "${FP16}" == "1" ]] && FP16_ARGS="--purify_fp16"
RESPACE_ARGS=""
[[ -n "${RESPACE}" ]] && RESPACE_ARGS="--purify_respace ${RESPACE}"

echo "[stage4_purify_tiny] EVAL_TAG=${EVAL_TAG} VARIANT=${VARIANT} SIGMA_STAR=${SIGMA_STAR} RESPACE=${RESPACE:-none} SEED=${SEED} FP16=${FP16}"
echo "[stage4_purify_tiny] NOISE_PKL=${NOISE_PKL}"
echo "[stage4_purify_tiny] OUT_DIR=${OUT_DIR}"
echo "[stage4_purify_tiny] PURIFY_CACHE_DIR=${PURIFY_CACHE_DIR}  (resumability lives here -- do not delete between resubmits)"

CUDA_VISIBLE_DEVICES=0 python purify_tiny_layer1a.py \
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
    --purify_backend "${BACKEND}" \
    --tiny_npz "${TINY_NPZ}" \
    ${RESPACE_ARGS} \
    --purify_chunk "${PURIFY_CHUNK}" \
    --purify_cache_dir "${PURIFY_CACHE_DIR}" \
    --purify_ckpt_every "${PURIFY_CKPT_EVERY}" \
    ${FP16_ARGS} \
    ${NOISE_ARGS} 2>&1 | tee -a "${RESULTS_DIR}/${EVAL_TAG}.log"

echo "[stage4_purify_tiny] done -> ${RESULTS_DIR}/${EVAL_TAG}.log, ${OUT_DIR}/summary.tsv"
