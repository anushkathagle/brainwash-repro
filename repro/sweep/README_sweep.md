# BrainWash Table-1 sweep harness

Automates reproducing the paper's Table 1: for each (dataset, method) it trains the BrainWash
noise (stage 3) for both attack modes × both ε, then evaluates every column (clean / uniform /
cautious / reckless) and compares against the paper.

## What it covers (and what it can't)
- **4 methods**: `ewc`, `mas`, `rwalk`, `afec_ewc`. (The repo does **not** implement **ANCL**, so
  that Table-1 row is not reproducible here.)
- **CIFAR-100 + miniImageNet**: ready to go — victim checkpoints + inverted data exist on HF.
- **tinyImageNet**: only raw data is published — needs from-scratch victim training (stage 1) +
  model inversion (stage 2) for all 4 methods first. Treat as a separate, heavier follow-on.

That's **56 cells** (2 datasets × 4 methods × 7 columns) from ready artifacts.

## Files
| file | role |
|---|---|
| `configs.tsv`  | per (dataset, method): experiment, lasttask, λ, λ_emp, checkpoint path, inverted-data dir |
| `targets.tsv`  | paper Table-1 BWT/Acc (transcribed from the PDF — sanity-check before trusting any cell) |
| `stage3.sbatch`| parameterized noise training (env-driven); writes `noise/<tag>/*.pkl` |
| `stage4.sbatch`| parameterized eval (clean/uniform/ours); writes `results/<eval_tag>.log` + `acc_mat_*.npy` |
| `submit_sweep.sh` | orchestrator: submits all stage-3 + dependent stage-4 jobs (resumable, dry-runnable) |
| `collect_results.py` | scans `results/*.log`, prints reproduced vs paper with deltas + match flags |

## 0. Deploy (from your Mac, in the Thesis dir)
Push the updated `repro/` (fixed downloader + this harness + unbuffered sbatch) to the cluster:
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='repro/artifacts' \
  --exclude='repro/sweep/noise' --exclude='repro/sweep/results' --exclude='.DS_Store' \
  ./Brainwash mfr5933@submit.hpc.psu.edu:~/
```

## 1. CIFAR-100 (ready now)
On the cluster:
```bash
cd ~/Brainwash
# (optional) seed the already-trained reckless eps0.3 noise so stage 3 skips it:
mkdir -p repro/sweep/noise/cifar100_ewc_reckless_eps0.3
mv noise_*min_acc_target_36.pkl repro/sweep/noise/cifar100_ewc_reckless_eps0.3/ 2>/dev/null || true

# preview the full plan WITHOUT submitting:
DRY_RUN=1 ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh cifar100 | less

# launch everything for CIFAR-100 (stage-3 noise runs + dependent evals), MIG slices:
ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh cifar100
```
Monitor and collect:
```bash
squeue -u $USER
python repro/sweep/collect_results.py            # reproduced vs paper, updates as jobs finish
python repro/sweep/collect_results.py --csv cifar100_table1.csv
```

## 2. miniImageNet (needs data first; validate one cell before the full launch)
```bash
cd ~/Brainwash
python repro/download_artifacts.py --what miniimagenet datasets   # checkpoints+inversions + raw zips
bash repro/setup_data.sh                                          # extracts miniImageNet under ~/data

# validate ONE cell end-to-end before committing the whole dataset:
ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh mini ewc
python repro/sweep/collect_results.py            # check mini/ewc/reckless lands near paper
# then the rest:
ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh mini
```

## Run-order tips
- **Two-phase (most robust for big runs):** `ONLY=stage3 ... submit_sweep.sh` first; once all noise
  is trained, `ONLY=stage4 ... submit_sweep.sh` (evals find the noise dirs, no SLURM deps to track).
- **One-shot:** the default (`ONLY=both`) wires stage-4 to its stage-3 via `--dependency=afterok`.
  If a stage-3 fails, its dependents show `DependencyNeverSatisfied` — fix and re-run (it's resumable:
  configs whose `noise/<tag>/` already has a pkl are skipped).
- **Full A100 instead of MIG:** prepend `GRES=gpu:a100:1` (≈2.5× faster/epoch, may queue longer).

## Multi-seed verification (variance)
High-variance methods (MAS, RWALK) won't match a single-seed paper cell exactly. Run several seeds
and let the collector report mean ± std and whether the paper value lands inside our spread:
```bash
# adds seeds 1 and 2 (seed 0 = the runs you already did; skipped automatically)
SEEDS="1 2" GRES=gpu:a100:1 ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh cifar100 mas
SEEDS="1 2" GRES=gpu:a100:1 ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh cifar100 rwalk
python repro/sweep/collect_results.py        # now shows n=3, mean±std, OK/VAR/XX
```
Flags: **OK** = our mean within tol of paper; **VAR** = paper inside our mean±std (reproduced, just
noisy); **XX** = paper outside our spread (a real discrepancy — e.g. an anomalous paper cell).
seed 0 keeps the original un-suffixed tags; seeds≥1 use `_s<seed>`, and finished evals are skipped,
so this is safe to run alongside / after the seed-0 sweep without duplicating work.

## Compute budget (rough)
- Stage 3: 32 noise runs (2 datasets × 4 methods × 2 modes × 2 ε) × ~3–5 h each on a MIG slice
  ≈ **~130 MIG-GPU-hours**. Highly parallel — submit all, let SLURM spread them; wall-clock ~1–2 days.
- Stage 4: 56 evals × ~10–30 min ≈ **~15 GPU-hours**.
- This draws on your `gvresearch_cr_default` credits — budget accordingly.

## Reading the comparison
`collect_results.py` flags `OK` when |ΔBWT|≤2 and |ΔAcc|≤3 (run-to-run nondeterminism across
GPUs/library versions is normal — our validated EWC/CIFAR-100 reckless cell came in at −28.9 vs −29.1).
`~` = within 2× tolerance, `XX` = off (investigate λ, the noise pkl picked, or a transcribed target).
