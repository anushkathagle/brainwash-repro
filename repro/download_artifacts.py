#!/usr/bin/env python3
"""Download BrainWash artifacts (victim checkpoints, inverted samples, raw datasets)
from the authors' HuggingFace dataset repo.

Repo: https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24

What's actually in the repo (verified June 2026):
  cifar100_files/     194 MB  4 victim checkpoints (afec_ewc, ewc, mas, rwalk)
                              + 36 inverted-sample .npz (4 methods x 9 tasks)
  miniImagenet_files/ 528 MB  same layout for miniImageNet
  datasets/           1.3 GB  miniImagenet.zip + tiny-imagenet-200.zip (raw data)

NOT in the repo (you must produce these yourself by running the pipeline):
  * The trained BrainWash noise (stage 3 output).
  * Any tinyImageNet checkpoints / inverted data (only the raw zip is provided).
  * CIFAR-100 raw data (torchvision downloads it automatically; see prepare_cifar100.py).

Why this script downloads to a cache and then copies:
  The checkpoint filenames are ~180 chars. HuggingFace's `local_dir` download mode
  stages each file as `<name>.<64-char-sha>.incomplete`, which exceeds the 255-char
  filename limit on many cluster filesystems (OSError 36, "File name too long").
  Downloading into the hash-named HF cache avoids the long staging names; we then copy
  the files into a clean repro/artifacts/<path> layout where the final name fits.

Examples:
  # Just CIFAR-100 checkpoints + inverted data (smallest, fully reproducible benchmark)
  python repro/download_artifacts.py --what cifar100

  # CIFAR-100 + miniImageNet model artifacts + the raw dataset zips
  python repro/download_artifacts.py --what cifar100 miniimagenet datasets

  # Everything
  python repro/download_artifacts.py --what all
"""
import argparse
import os
import shutil
import sys

REPO_ID = "mintlabvandy/BrainWash-CVPR24"

PATTERNS = {
    "cifar100": ["cifar100_files/**"],
    "miniimagenet": ["miniImagenet_files/**"],
    "datasets": ["datasets/**"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["cifar100"],
                    choices=list(PATTERNS) + ["all"],
                    help="Which artifact groups to download (default: cifar100).")
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "artifacts"),
                    help="Local directory to place files into (default: repro/artifacts).")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Keep the temporary HF cache instead of deleting it after copy.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    if "all" in args.what:
        allow = None  # whole repo
    else:
        allow = [p for w in args.what for p in PATTERNS[w]]

    dest = os.path.abspath(args.dest)
    cache_dir = os.path.join(dest, ".hfcache")
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading {args.what} from {REPO_ID} into a temp cache ...")
    snap = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=cache_dir,       # hash-named blobs -> short staging names, no Errno 36
        allow_patterns=allow,
    )

    print(f"Copying files into clean layout under {dest} ...")
    n = 0
    for root, _, files in os.walk(snap):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, snap)
            dst = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.realpath(src), dst)  # resolve symlink -> real blob
            n += 1
    print(f"Copied {n} files.")

    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    print("\nDone. Next: run  bash repro/setup_data.sh  to place datasets under ~/data,")
    print("and see REPRODUCE.md for the per-stage commands.")


if __name__ == "__main__":
    main()
