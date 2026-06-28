# Reproducing BrainWash (CVPR 2024) on Penn State Roar Collab (A100)

This guide reproduces the BrainWash poisoning attack results (Table 1 of the paper)
using the authors' published code + checkpoints + inverted samples. It targets the
**Penn State ICDS Roar Collab** cluster with **A100 GPUs** and SLURM.

> **What BrainWash actually is.** Despite the "forgetting in LLMs" motivation, BrainWash
> is a *computer-vision* attack on **regularization-based continual learning**. It trains a
> ResNet-18 on a sequence of image-classification tasks (CIFAR-100 / miniImageNet /
> tinyImageNet) and crafts additive noise on the final task that maximizes forgetting of
> earlier tasks. No language models are involved.

---

## 0. Read this first — what "identical results" realistically means

You asked for 100% identical numbers. Two facts bound that:

1. **You must run on NVIDIA/CUDA.** The code is hard-wired to CUDA (`main_baselines.py`
   calls `sys.exit()` if CUDA is missing and sets `torch.cuda.FloatTensor` as the default
   tensor type). The A100s are exactly the right substrate. Your M1 Mac cannot run this
   faithfully and is only useful for reading/editing the code.
2. **Bit-for-bit identical is not guaranteed even on A100s.** The authors did not publish
   their exact CUDA/cuDNN/PyTorch versions or GPU model, and cuDNN has nondeterminism.
   With the matched stack below + the authors' checkpoints + fixed seeds, you should land
   **within the paper's reported precision (~0.1 in BWT/Acc)**. That is what reproduction
   means here.

What is provided vs. what you must compute:

| Stage | Artifact on HuggingFace? | Action |
|------|---------------------------|--------|
| 1. Train victim (tasks 0–8) | ✅ CIFAR-100 & miniImageNet checkpoints | skip — use provided `.pkl` |
| 2. Model inversion          | ✅ CIFAR-100 & miniImageNet inverted `.npz` | skip — use provided folder |
| 3. **Train BrainWash noise**| ❌ not provided | **you run this (the heavy step)** |
| 4. **Evaluate noise → Table 1** | ❌ | **you run this** |

Not reproducible from provided artifacts: **tinyImageNet** (only the raw dataset is
shipped — no checkpoints/inverted data, so run all 4 stages), and **ANCL** (the 5th CL
method in the paper is *not in this repo*; only EWC / MAS / RWALK / AFEC-EWC are).

---

## 1. Environment (run once, on a login/interactive node with internet)

```bash
cd <repo-root>            # the folder containing main_baselines.py
bash repro/setup_env.sh   # creates conda env "brainwash": py3.10 + torch1.12.1+cu113
```

This uses the **authors' stack** (Python 3.10 + PyTorch 1.12.1 + CUDA 11.3). On that stack
the original code runs **with zero source changes** and minimal numerical drift. (CUDA 11.3
fully supports the A100 / sm_80.) The repo's only modern-PyTorch incompatibility — the
`from torch._six import inf` line in `approaches/utils.py` — has been wrapped in a
try/except so the code also imports on PyTorch ≥ 1.13 if you ever need a newer stack.

If `module load anaconda3` fails, run `module spider anaconda` and edit the module name in
`repro/setup_env.sh`.

## 2. Download artifacts (login node)

```bash
# CIFAR-100 only (smallest, fully reproducible — recommended to start):
python repro/download_artifacts.py --what cifar100

# Add miniImageNet model artifacts + raw dataset zips when ready:
python repro/download_artifacts.py --what cifar100 miniimagenet datasets
```

Sizes: cifar100_files ≈ 194 MB, miniImagenet_files ≈ 528 MB, datasets ≈ 1.3 GB.
Everything lands under `repro/artifacts/`.

## 3. Place datasets where the code expects them

```bash
bash repro/setup_data.sh   # CIFAR-100 -> ./data ; miniImagenet/tiny -> ~/data
```

- CIFAR-100: auto-downloaded to `./data` (needed because stages 2–4 load it with
  `download=False`).
- miniImageNet: unzipped to `~/data/miniImagenet/{train_x,train_y,test_x,test_y}.npy`.
- tinyImageNet (optional): unzipped to `~/data/tiny-imagenet-200/`; then build the cache
  with `python repro/prepare_tiny_imagenet.py`.

> If Roar Collab compute nodes are offline, do steps 1–3 on a login node first.

---

## 4. Reproduce a Table-1 cell (worked example: CIFAR-100 / EWC / reckless / ε=0.3)

Edit the two SLURM scripts' top "EDIT" blocks if needed, then submit from the repo root.

**Stage 3 — train the noise (~hours on one A100):**
```bash
mkdir -p logs
sbatch repro/slurm/stage3_brainwash.sbatch
```
The script's `--init_acc` will print the loaded victim's per-task accuracy. For the EWC
CIFAR-100 checkpoint you should see approximately:
```
init acc task 0..8 ≈ 54.9  57.0  67.1  62.6  66.0  66.1  63.7  61.1  59.6   (mean ≈ 62.0%)
```
Matching these confirms the checkpoint + data are wired correctly. Stage 3 writes a
`noise_ewc_*.pkl` to the repo root; the noise lives under its `latest_noise` / `noise` keys.

**Stage 4 — evaluate the noise → BWT + accuracy:**
```bash
# put the noise_*.pkl path into repro/slurm/stage4_eval.sbatch (NOISE_PKL=...), then:
sbatch repro/slurm/stage4_eval.sbatch
```
This resumes continual learning on the poisoned 10th task and prints
`After BWT : ... After avg acc : ... Last task acc : ...` and writes `acc_mat_*.npy`.
Compare `After BWT (After avg acc)` to the paper's Table 1 **Reckless, ε=0.3** cell for
CIFAR-100/EWC: **−29.1 (25.5)**.

### The three Table-1 variants per cell
Each (dataset, method, ε) cell has three numbers; all use the **same** stage-4 script,
only the flags change:
- **ours** (the attack): `--addnoise` (default in the script).
- **uniform** baseline: add `--uniform` (keep `--addnoise`).
- **clean** baseline: remove `--addnoise`.

### Sweeping the table
Vary these and re-run stages 3→4:
- **Method** → `ewc | mas | rwalk | afec_ewc` (set matching checkpoint, inverted folder, and λ below).
- **ε** → `--delta 0.3` or `--delta 0.1` in stage 3.
- **Mode** → `--mode reckless` or `--mode cautious` (cautious adds `--w_cur 1`, the paper's η).
- **Dataset** → swap `cifar100_files`↔`miniImagenet_files` and `--experiment split_cifar100`↔`split_mini_imagenet`.

### Per-method regularization λ (must match the checkpoint, used in stage 4)

| Method    | CIFAR-100 λ | miniImageNet λ |
|-----------|-------------|----------------|
| EWC       | 500000 (5e5)| 10000 (1e4)    |
| MAS       | 10          | 5              |
| RWALK     | 1           | 5              |
| AFEC-EWC  | 5e5, lamb_emp=100 | 5000, lamb_emp=1 |

(These come directly from the provided checkpoint filenames. Stage 4 also auto-reads λ
from the checkpoint's `cont_method_args`, so the value mainly matters for retraining a
victim from scratch in stage 1.)

---

## 5. From-scratch / tinyImageNet (optional)

To regenerate stages 1–2 yourself (required for tinyImageNet, optional otherwise):
```bash
sbatch repro/slurm/stage1_train_victim.sbatch   # -> victim <approach>_*.pkl
sbatch repro/slurm/stage2_inversion.sbatch      # -> inverted_data folder
```
Then feed those into stages 3–4 as above.

---

## 6. File map of what was added for reproduction

```
repro/
  requirements-cu113.txt     # pinned authors' stack (torch 1.12.1+cu113, numpy<2, ...)
  setup_env.sh               # build the conda env on Roar Collab
  download_artifacts.py      # pull checkpoints/inverted-data/datasets from HuggingFace
  setup_data.sh              # place datasets under ./data and ~/data
  prepare_cifar100.py        # pre-download CIFAR-100 to ./data (for stages 2-4)
  prepare_tiny_imagenet.py   # build data/tiny_imagenet.npz (tinyImageNet only)
  slurm/
    stage1_train_victim.sbatch
    stage2_inversion.sbatch
    stage3_brainwash.sbatch  # REQUIRED to produce the noise
    stage4_eval.sbatch       # REQUIRED to get Table-1 numbers
REPRODUCE.md                 # this file
```

### Changes to the original source (3 small, surgical patches; all CUDA logic untouched)
1. `approaches/utils.py` — wrapped `from torch._six import inf` in try/except (compat with PyTorch ≥ 1.13).
2. `main_brainwash.py` — `model_save_dict.setdefault('optim_name', 'sgd')` after the checkpoint load
   (the AFEC checkpoint omits this key; its victim was trained with SGD like the others).
3. `main_brainwash.py` — noise is written to `$BW_OUT_DIR` (default `.`) instead of always the CWD,
   so the sweep can save the ~140–850 MB noise pkls straight to large scratch (`/storage/work`) and
   never fill the 16 GB home quota.

For the **automated full-table workflow** (recommended), use the sweep harness in `repro/sweep/`
instead of editing these SLURM scripts by hand — see `repro/sweep/README_sweep.md`. The manual
single-cell flow above is kept as a conceptual walkthrough.
