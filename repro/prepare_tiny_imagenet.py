#!/usr/bin/env python3
"""(Optional) Build the cached tiny_imagenet.npz that the 20-split tinyImageNet
experiments load.

Only needed if you want to reproduce the tinyImageNet rows of Table 1. The
HuggingFace repo provides the raw tiny-imagenet-200.zip but NO tinyImageNet
checkpoints or inverted data, so tinyImageNet must be run through all 4 stages
from scratch.

Steps performed:
  1. Reorganize ~/data/tiny-imagenet-200/val images into per-class subfolders
     (the loader expects val/images/<wnid>/*.JPEG).
  2. Read every train/val image and cache a single ./data/tiny_imagenet.npz that
     stages 1-4 load via dataset_file='data/tiny_imagenet.npz'.

Run from the repo root:
  python repro/prepare_tiny_imagenet.py
"""
import os
import numpy as np
from data_utils import create_tinyimangenet_val_img_folder, generate_split_tiny_imagenet_tasks

root_add = os.path.join(os.path.expanduser("~"), "data", "tiny-imagenet-200")
npz_out = os.path.join(os.getcwd(), "data", "tiny_imagenet.npz")
os.makedirs(os.path.dirname(npz_out), exist_ok=True)

if not os.path.isdir(root_add):
    raise SystemExit(f"{root_add} not found. Run repro/setup_data.sh first.")

print("1/2 Reorganizing val images into class subfolders ...")
create_tinyimangenet_val_img_folder(root_add)

print(f"2/2 Caching all images into {npz_out} (this reads ~110k images, takes a few minutes) ...")
generate_split_tiny_imagenet_tasks(
    task_num=20,
    rnd_order=False,
    order=np.arange(200),
    save_data=True,
    dataset_file=npz_out,
    root_add=root_add,
)
print("Done. data/tiny_imagenet.npz is ready.")
