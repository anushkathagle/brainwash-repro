#!/usr/bin/env bash
# Place the downloaded raw datasets where the BrainWash code expects them.
#
# Run from the repo root, after repro/download_artifacts.py:
#   bash repro/setup_data.sh
#
# Expected final layout:
#   ~/data/miniImagenet/{train_x,train_y,test_x,test_y}.npy   (split_mini_imagenet)
#   ~/data/tiny-imagenet-200/{train,val,...}                  (split_tiny_imagenet, optional)
#   ./data/cifar-100-python/...                               (split_cifar100)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${REPO_ROOT}/repro/artifacts/datasets"
DATA_HOME="${HOME}/data"
mkdir -p "${DATA_HOME}"

# --- miniImageNet ------------------------------------------------------------
if [[ -f "${ART}/miniImagenet.zip" ]]; then
    echo "Unzipping miniImagenet.zip -> ${DATA_HOME} ..."
    unzip -o -q "${ART}/miniImagenet.zip" -d "${DATA_HOME}"
    # The code reads ~/data/miniImagenet/train_x.npy etc. Locate them if nested.
    if [[ ! -f "${DATA_HOME}/miniImagenet/train_x.npy" ]]; then
        found="$(find "${DATA_HOME}" -name train_x.npy -print -quit || true)"
        if [[ -n "${found}" ]]; then
            echo "  NOTE: train_x.npy found at: ${found}"
            echo "  Ensure ~/data/miniImagenet/ contains train_x.npy/train_y.npy/test_x.npy/test_y.npy"
            echo "  (move/symlink the containing folder to ~/data/miniImagenet if needed)."
        else
            echo "  WARNING: train_x.npy not found after unzip — inspect the archive layout." >&2
        fi
    else
        echo "  OK: ~/data/miniImagenet/train_x.npy present."
    fi
else
    echo "Skipping miniImageNet (datasets/miniImagenet.zip not downloaded)."
fi

# --- tinyImageNet (optional; only needed for the 20-split tinyImagenet table) -
if [[ -f "${ART}/tiny-imagenet-200.zip" ]]; then
    echo "Unzipping tiny-imagenet-200.zip -> ${DATA_HOME} ..."
    unzip -o -q "${ART}/tiny-imagenet-200.zip" -d "${DATA_HOME}"
    echo "  Tiny ImageNet unzipped. Before training, build the cached npz with:"
    echo "    python repro/prepare_tiny_imagenet.py"
else
    echo "Skipping tinyImageNet (datasets/tiny-imagenet-200.zip not downloaded)."
fi

# --- CIFAR-100 ---------------------------------------------------------------
echo "Preparing CIFAR-100 under ./data ..."
( cd "${REPO_ROOT}" && python repro/prepare_cifar100.py )

echo
echo "Data setup complete."
