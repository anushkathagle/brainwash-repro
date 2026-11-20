# Layer 2: FRIENDS against BrainWash

FRIENDS (Liu, Yang & Mirzasoleiman, NeurIPS 2022) applied while the victim continual learner trains on the
poisoned last task. The attacker's code, `main_baselines.py` and `approaches/*.py` are not modified: everything
Layer-2 lives in this folder.

| File | What it is |
|---|---|
| `friends.py` | `generate_friendly_noise` (port of the official function; V1 checks it is bit-identical), the random noise `RandomNoise`, and `FriendsDefense` (when and how the noise is applied) |
| `friends_training.py` | the Layer-2 learners for EWC / RWALK / AFEC: copies of each method's `train()` / `train_epoch()` plus marked `# >>> LAYER2` lines |
| `friends_layer2.py` | the stage-4 script (a fork of `main_baselines.py`); all options are flags, see its header |
| `validate_friends.py` | checks V1, V2, V4, V5, V6, V7 |
| `colab_helpers.py` | the run loop and result tables shared by the notebooks |
| `notebooks/` | the Colab notebooks (below) |

## Notebooks

Open in Colab: *File > Open notebook > GitHub*, repository `anushkathagle/brainwash-repro`, branch `layer2/Friendly`
(private repo: tick *Include private repos* and authorize). Each notebook clones the branch, or pulls its newest
commit when re-run, so a change pushed here is picked up by re-running the "Code" cell.

| Notebook | What it runs |
|---|---|
| `1_validation_checks.ipynb` | V1 + V4 (must PASS after any code change), V2 + V5, V6, V7 |
| `2_option_a_features.ipynb` | option (a): friendly noise preserves the shared 1024-d features (`--friendly_target features`) |
| `3_option_b_warmup2.ipynb` | option (b): the paper's Algorithm 2 (warm-up 2, eta preserves head T); eta-optimizer settings; BatchNorm control |
| `4_option_c_warmup0.ipynb` | option (c): eta from the checkpoint (warm-up 0) |

Results: one Drive folder per attack strength, `layer2_runs/layer2_defense_eps<EPS>`, shared by notebooks 2-4
(`A_clean` / `B_poison` run once; an arm with a finished log is skipped, so a disconnected Colab is simply re-run).
`code_version.txt` in that folder records which commit of this branch produced the runs.

## Main flags of `friends_layer2.py`

| Flag | Meaning |
|---|---|
| `--addnoise` | poison the last task (without it: the clean run) |
| `--friends`, `--components both / friendly_only / random_only` | FRIENDS and its ablation |
| `--zeta`, `--mu_rand` | cap of the friendly noise / size of the random noise, in units of 1/255 |
| `--warmup_epochs` | epochs before eta is computed (2 = option b, 0 = option c) |
| `--friendly_target head / features` | what eta must leave unchanged (head = the paper; features = option a) |
| `--friendly_lambda`, `--friendly_lr`, `--friendly_momentum`, `--friendly_epochs` | the eta optimizer (official 1 / 100 / 0.9 / 30) |
| `--random_after_warmup`, `--head_warmup_epochs`, `--bn_mode eval / nostats`, `--log_old_acc` | schedule, BatchNorm and logging options |
| `--purify --sigma_star 0.5 --purify_ckpt ...` | Layer 1 (DiffPure, `diffpure_edm.py` in the repo root) before Layer 2 |

The method name in `summary_layer2.tsv` records every non-default setting (e.g.
`ours_friends-both_z32_mu32_w2_lr10_mom0`), so different settings never overwrite each other.

## Notes

* `friends.py` is the official FRIENDS optimizer (clamp inside the computation, as in the official code). A
  variant that clamps in place after each step was tried on Colab (`gemini_fix1`); on CPU it gives bit-identical
  noise to the official code even with half the pixels at the cap, and on Colab the same results to 4 decimals.
* All runs are seed 0 (there are no victim checkpoints for other seeds yet).
