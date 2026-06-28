# BrainWash — Reproduction & Evaluation Harness

Reproduction of **"BrainWash: A Poisoning Attack to Forget in Continual Learning"** (Abbasi et al.,
CVPR 2024) on the **Penn State ICDS Roar Collab** cluster (A100 GPUs, SLURM), plus a parameterized
harness to run the full attack and to **evaluate defenses against it**.

This is **step 1** of the thesis (faithfully reproduce the attack). It is the baseline the defense
work builds on. Original paper: https://arxiv.org/abs/2311.11995 · Authors' code: https://github.com/mint-vu/Brainwash

> **What BrainWash actually is:** despite the "forgetting in LLMs" framing, it is a *computer-vision*
> poisoning attack on **regularization-based continual learning**. A ResNet-18 learns a sequence of
> image-classification tasks; the attacker crafts ℓ∞-bounded additive noise on the *final* task that
> maximizes catastrophic forgetting of earlier tasks. No language models are involved.

---

## TL;DR — what was reproduced

We reproduced **Table 1** of the paper for the 4 methods that ship in the released code
(**EWC, MAS, RWALK, AFEC-EWC**) on **CIFAR-100** and **miniImageNet**:

| dataset | EWC | AFEC | MAS | RWALK | total OK |
|---|---|---|---|---|---|
| CIFAR-100 (5000 ep; MAS/RWALK 3-seed) | 7/7 | 6/7 | 3/7 | 6/7 | **22/28** |
| miniImageNet (2500 ep; single-seed) | 6/7 | 4/7 | 5/7 | 6/7 | **21/28** |

**43 / 56 cells within tolerance (±2 BWT, ±3 Acc).** The headline metric — **BWT (forgetting) — reproduces
tightly on essentially every cell**; the deviations are almost all in *last-task accuracy*, concentrated in
the high-variance MAS method and `cautious`-mode cells, plus **two demonstrable anomalies in the paper's own
CIFAR-MAS numbers** (where our values are the self-consistent ones). Full analysis:
- [`repro/sweep/RESULTS_cifar100.md`](repro/sweep/RESULTS_cifar100.md)
- [`repro/sweep/RESULTS_mini.md`](repro/sweep/RESULTS_mini.md) (ends with the combined verdict)

**Out of scope:** **ANCL** (the 5th method in the paper — *not implemented* in the released code) and
**tinyImageNet** (only the raw dataset is published; would need from-scratch victim training + inversion).

---

## ⚠️ Before you run — set these for YOUR account

The harness runs on Roar Collab. Three things are per-user:

| What | How to set | Default / note |
|---|---|---|
| **SLURM account** | pass `ACCT=<your_account>` to `submit_sweep.sh` | we used `gvresearch_cr_default` (the group allocation) |
| **conda env path** | nothing — auto-detected from `conda env list` | override with `ENV_PREFIX=/path/to/envs/brainwash` if needed |
| **scratch for data+noise** | symlink `~/data` and `repro/sweep/noise` to your `/storage/work/<id>/...` (see Gotchas) | home is only **16 GB**; data+noise must live on `/storage/work` |

`CONDA_SH` (`/swst/apps/anaconda3/.../conda.sh`) is the cluster-wide base anaconda and is the same for everyone.

---

## Repository layout

```
README.md                 ← you are here (handoff guide)
UPSTREAM_README.md        ← the original authors' README (4-stage overview, run commands)
REPRODUCE.md              ← conceptual single-cell walkthrough (manual workflow)
LICENSE                   ← original license

main_brainwash.py         ← stage 3: train the BrainWash noise  (PATCHED, see below)
main_baselines.py         ← stages 1 & 4: train victim / evaluate noise
main_inv.py               ← stage 2: model inversion
approaches/, *.py         ← authors' CL methods + utils (approaches/utils.py PATCHED)

repro/                    ← everything we added for reproduction
  setup_env.sh              build the conda env (py3.10 + torch 1.12.1+cu113)
  requirements-cu113.txt    pinned authors' stack
  download_artifacts.py     pull checkpoints / inverted data / datasets from HuggingFace
  setup_data.sh             place datasets where the code expects them
  prepare_cifar100.py       pre-download CIFAR-100
  prepare_tiny_imagenet.py  build tinyImageNet cache (only if you do tiny)
  slurm/                    hand-editable per-stage SLURM scripts (manual workflow)
  sweep/                    ★ the automated full-Table-1 harness ★
    configs.tsv               per (dataset, method): λ, checkpoint path, inverted-data dir
    targets.tsv               the paper's Table-1 numbers (for auto-comparison)
    stage3.sbatch             parameterized noise training (env-driven)
    stage4.sbatch             parameterized evaluation (clean/uniform/ours)
    submit_sweep.sh           orchestrator: submits all stage-3 + dependent stage-4 jobs
    collect_results.py        scans results, prints reproduced-vs-paper with OK/VAR/XX flags
    README_sweep.md           detailed harness docs
    RESULTS_cifar100.md       CIFAR-100 results + verdict
    RESULTS_mini.md           miniImageNet results + combined verdict
```

Large artifacts (checkpoints, noise `.pkl`, datasets, `.npy`) are **not** in git — they're on
HuggingFace / generated on the cluster (see `.gitignore`).

---

## Quickstart (Roar Collab)

```bash
# 0. clone + build env (login node, has internet)
git clone <this-repo-url> brainwash && cd brainwash
bash repro/setup_env.sh
module load anaconda3; eval "$(conda shell.bash hook)"; conda activate brainwash

# 1. get artifacts (checkpoints + inverted data) + datasets
python repro/download_artifacts.py --what cifar100 miniimagenet datasets
bash repro/setup_data.sh          # CIFAR-100 -> ./data ; miniImagenet -> ~/data/miniImagenet

# 2. run the whole CIFAR-100 table (one command; ~a day on A100s)
ACCT=<your_account> GRES=gpu:a100:1 repro/sweep/submit_sweep.sh cifar100
#    multi-seed the noisy methods (optional, used for the published CIFAR numbers):
SEEDS="1 2" ACCT=<your_account> GRES=gpu:a100:1 repro/sweep/submit_sweep.sh cifar100 mas
SEEDS="1 2" ACCT=<your_account> GRES=gpu:a100:1 repro/sweep/submit_sweep.sh cifar100 rwalk

# 3. collect + compare to the paper (re-run anytime; updates as jobs finish)
python repro/sweep/collect_results.py
python repro/sweep/collect_results.py --csv brainwash_repro.csv
```

`DRY_RUN=1 ... submit_sweep.sh cifar100` prints the full plan without submitting. miniImageNet is the
same but read **Gotchas → miniImageNet** first (it's slower and needs `NEPOCHS3=2500`). Full details:
[`repro/sweep/README_sweep.md`](repro/sweep/README_sweep.md).

---

## Patches to the upstream code (the only source changes)

Everything else we added is additive under `repro/`. The three in-place patches:

1. **`approaches/utils.py`** — `from torch._six import inf` wrapped in try/except (PyTorch ≥1.13 compat).
2. **`main_brainwash.py`** — `model_save_dict.setdefault('optim_name', 'sgd')` after loading the checkpoint.
   *Why:* the AFEC checkpoint is the only one missing the `optim_name` key → `KeyError` without this.
3. **`main_brainwash.py`** — noise is saved to `$BW_OUT_DIR` (default `.`) rather than always the CWD,
   so the sweep writes the large noise `.pkl`s straight to `/storage/work` instead of the 16 GB home.

All CUDA logic is untouched. `git log` shows the full history (upstream commits + our reproduction work).

---

## Cluster gotchas (READ THIS — these cost us the most time)

- **conda in batch jobs:** `conda activate brainwash` *by name* silently no-ops in some SLURM shells →
  jobs die with `No module named torch`. The sweep scripts fix this by `source`-ing conda by absolute
  path and activating by full env **prefix** (auto-detected), with a fail-fast `import torch` guard.
- **16 GB home quota is the recurring trap.** Datasets and noise must go on `/storage/work` (110 TB):
  ```bash
  mkdir -p /storage/work/$USER/bw_data /storage/work/$USER/bw_noise
  rm -rf ~/data && ln -s /storage/work/$USER/bw_data ~/data
  rm -rf repro/sweep/noise && ln -s /storage/work/$USER/bw_noise repro/sweep/noise
  ```
  The code follows the symlinks transparently. (`BW_OUT_DIR` already routes new noise here.)
- **GPU choice:** use full A100 (`GRES=gpu:a100:1`) for batch sweeps — there are ~80 of them vs only ~7
  MIG slices, so throughput is far higher. Results are identical on either.
- **miniImageNet is ~4× slower** than CIFAR (84×84 vs 32×32): 5000 epochs ≈ 23 h (over typical walls).
  We run mini at **2500 epochs** (`NEPOCHS3=2500`) — a documented compute trade-off; reckless/uniform/clean
  cells confirm 2500 ep matches the paper's 5000-ep numbers. `stage3.sbatch` uses a 24 h wall for this.
- **`#SBATCH` directive changes only take effect on re-deploy** (re-clone/rsync) — an in-place edit to the
  conda block does **not** carry a `--time` change into already-running scripts.
- Don't double-submit; the sweep is **resumable** (configs whose `noise/<tag>/` has a `.pkl`, or whose
  `results/<tag>.log` has `After BWT`, are skipped), so re-running `submit_sweep.sh` only fills gaps.

---

## Status & how to extend (the defense)

**Done:** attack reproduced + a one-command harness that produces the 43-cell baseline and auto-compares
to the paper. **Next (the thesis contribution):** a defense that the victim applies when learning the
poisoned task T, evaluated by dropping it into **stage 4** and measuring how much BWT/Acc is restored vs
this baseline — same harness, one new knob. Natural intervention points: sanitizing/denoising the
incoming task-T data, poison-aware regularization, or anomaly-detection/rejection before consolidation.

## Credits & license
Original method, code, and released checkpoints/inverted data by Ali Abbasi, Parsa Nooralinejad, Hamed
Pirsiavash, Soheil Kolouri (MINT-VU). This repo forks their code; reproduction harness + docs added for
the thesis. License unchanged — see `LICENSE` and `UPSTREAM_README.md`.
