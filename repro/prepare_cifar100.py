#!/usr/bin/env python3
"""Pre-download CIFAR-100 into ./data (relative to the repo root).

Why this is needed:
  * approaches/data_utils.py (stage 1) calls CIFAR100(..., download=True).
  * The root data_utils.py used by stages 2-4 calls CIFAR100(root='./data', download=False).
    If you start from a *provided* checkpoint and skip stage 1, the data must already exist.

Why it pulls from a HuggingFace mirror:
  The official host www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz currently returns
  HTTP 403 Forbidden to programmatic clients (verified June 2026, all user-agents/protocols),
  which breaks torchvision's default download. We instead fetch the *identical* tarball from
  the HF mirror `nakroy/cifar100-python` (md5 verified == torchvision's expected
  eb9058c3a382ffc7106e4002c42a8d85), drop it in ./data, and let torchvision verify + extract
  it offline. HuggingFace is reachable from Roar Collab, so this works on the cluster.

Run from the repo root:
  python repro/prepare_cifar100.py
"""
import hashlib
import os
import shutil

from torchvision.datasets import CIFAR100

EXPECTED_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
HF_MIRROR = ("nakroy/cifar100-python", "cifar-100-python.tar.gz")

data_root = os.path.join(os.getcwd(), "data")
os.makedirs(data_root, exist_ok=True)
tar_path = os.path.join(data_root, "cifar-100-python.tar.gz")


def _valid_tar(path):
    if not os.path.exists(path):
        return False
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == EXPECTED_MD5


if not _valid_tar(tar_path):
    print("Fetching cifar-100-python.tar.gz from HuggingFace mirror "
          "(official Toronto host returns 403) ...")
    from huggingface_hub import hf_hub_download
    cache = os.path.join(data_root, ".hfcache")
    p = hf_hub_download(HF_MIRROR[0], HF_MIRROR[1], repo_type="dataset", cache_dir=cache)
    shutil.copyfile(os.path.realpath(p), tar_path)
    shutil.rmtree(cache, ignore_errors=True)
    assert _valid_tar(tar_path), "Downloaded CIFAR-100 tar failed md5 check."

print(f"Extracting/loading CIFAR-100 from {tar_path} (offline) ...")
CIFAR100(root=data_root, train=True, download=True)   # local tar present -> no network
CIFAR100(root=data_root, train=False, download=True)
print("CIFAR-100 ready under ./data. Launch the pipeline from the repo root so './data' resolves here.")
