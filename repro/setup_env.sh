#!/usr/bin/env bash
# Create the BrainWash reproduction conda env on Penn State Roar Collab.
#
# Run this on a Roar Collab *login/interactive* node (it needs internet).
#   bash repro/setup_env.sh
#
# Notes:
#   * Roar Collab uses Lmod ("module") + conda. The exact anaconda module name
#     can vary; `module spider anaconda` will list options. Adjust the line below
#     if `anaconda3` is not found.
#   * If your compute nodes are offline, run BOTH this script and
#     repro/download_artifacts.py from a login node first, then submit jobs.
set -euo pipefail

ENV_NAME="${1:-brainwash}"

# --- 1. Load conda -----------------------------------------------------------
# Try common module names; fall back to whatever `conda` is already on PATH.
module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found. Run 'module spider anaconda' and load the right module." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

# --- 2. Create env -----------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Conda env '${ENV_NAME}' already exists — reusing it."
else
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# --- 3. Install deps (authors' CUDA 11.3 stack) ------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install --upgrade pip
pip install -r "${REPO_ROOT}/repro/requirements-cu113.txt"

# --- 4. Sanity check ---------------------------------------------------------
python - <<'PY'
import torch, torchvision
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("CUDA build  :", torch.version.cuda)
print("CUDA avail  :", torch.cuda.is_available(), "(False is normal on a login node)")
PY

echo
echo "Done. Activate later with:  conda activate ${ENV_NAME}"
