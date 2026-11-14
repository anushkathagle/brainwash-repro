#!/usr/bin/env bash
# Preflight for the Tiny-ImageNet purification runs. Run on the LOGIN NODE from the
# repo root, BEFORE sbatch -- takes seconds, needs no GPU, and catches every missing
# input that would otherwise only surface after the job has waited in the queue.
#
#   ./check_tiny_setup.sh                      # checks for the sigma sweep
#   NOISE_TAG=tiny_ewc_reckless_eps0.3 ./check_tiny_setup.sh   # + stage-3 noise for stage 4
#
# Uses the same variables/defaults as sigma_sweep_tiny.sbatch / stage4_purify_tiny.sbatch.
: "${TINY_NPZ:=./data/tiny_imagenet.npz}"
: "${PURIFY_CKPT:=repro/artifacts/diffusion/imagenet64_uncond_100M_1500K.pt}"
: "${ID_PATH:=/storage/home/axt5884/improved-diffusion}"
: "${ENV_PREFIX:=/storage/home/axt5884/work/conda/myworkflow}"
: "${CONDA_SH:=/swst/apps/anaconda3/2021.05_gcc-8.5.0/etc/profile.d/conda.sh}"
: "${NOISE_TAG:=}"
bad=0
ok()   { echo "  [ok]   $*"; }
fail() { echo "  [MISSING] $*"; bad=1; }

echo "Tiny-ImageNet purification preflight (cwd: $PWD)"
[[ -f sigma_sweep_tiny.py && -f purify_tiny_layer1a.py ]] && ok "repo root, branch $(git branch --show-current 2>/dev/null)" \
  || fail "not in the repo root (cd to brainwash-tiny first)"
[[ -d logs ]] && ok "logs/ exists" || fail "logs/ -- SLURM will not create it; jobs die with no log. Fix: mkdir -p logs"
if [[ -f "${PURIFY_CKPT}" ]]; then
  sz=$(stat -c %s "${PURIFY_CKPT}")
  (( sz > 400000000 )) && ok "checkpoint ${PURIFY_CKPT} ($((sz/1000000)) MB)" \
    || fail "checkpoint ${PURIFY_CKPT} is only $((sz/1000000)) MB -- partial download? (expect ~480 MB)"
else
  fail "checkpoint ${PURIFY_CKPT}
           wget -P $(dirname "${PURIFY_CKPT}") https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt"
fi
[[ -f "${TINY_NPZ}" ]] && ok "data ${TINY_NPZ} -> $(readlink -f "${TINY_NPZ}")" \
  || fail "data ${TINY_NPZ} (symlink the old folder's copy, or: python repro/prepare_tiny_imagenet.py)"
[[ -f "${ID_PATH}/improved_diffusion/script_util.py" ]] && ok "improved-diffusion code at ${ID_PATH}" \
  || fail "improved-diffusion code at ${ID_PATH}  (git clone https://github.com/openai/improved-diffusion.git ${ID_PATH})"
if [[ -n "${NOISE_TAG}" ]]; then
  compgen -G "repro/sweep/noise/${NOISE_TAG}/*.pkl" >/dev/null && ok "noise pkl in repro/sweep/noise/${NOISE_TAG}/" \
    || fail "noise pkl in repro/sweep/noise/${NOISE_TAG}/ (run stage 3 first)"
fi

# Python-level check in the job's env: imports + builds the UNet on CPU (no GPU, no weights needed).
if [[ -f "${CONDA_SH}" ]]; then source "${CONDA_SH}"; else module load anaconda3 2>/dev/null || true; eval "$(conda shell.bash hook 2>/dev/null)"; fi
if conda activate "${ENV_PREFIX}" 2>/dev/null; then
  PYTHONPATH="${ID_PATH}:${PYTHONPATH:-}" python - <<'PY' && ok "env ${ENV_PREFIX}: torch + improved_diffusion import, UNet builds" || fail "python check in ${ENV_PREFIX} (see error above)"
import torch, diffpure_imagenet64 as d
from improved_diffusion.script_util import create_model_and_diffusion
m, diff = create_model_and_diffusion(**d.IMAGENET64_UNCOND_CONFIG)
assert diff.num_timesteps == 4000 and diff.rescale_timesteps
print(f"           torch {torch.__version__}, UNet {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")
PY
else
  fail "conda env ${ENV_PREFIX}"
fi

echo
(( bad )) && { echo "NOT READY -- fix the [MISSING] items, then re-run this."; exit 1; }
echo "READY -- submit from this folder:  sbatch --account=<ACCT> sigma_sweep_tiny.sbatch"
