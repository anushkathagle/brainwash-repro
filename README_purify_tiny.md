# Layer 1a on Tiny-ImageNet: native-resolution diffusion purification

This is the third dataset of the Layer 1a defense, set up as a **controlled test of the
resolution explanation** from miniImageNet:

| Dataset | Purifier | Resolution | Result |
|---|---|---|---|
| CIFAR-100 | EDM, fine-tuned, in-domain (`diffpure_edm.py`) | native 32 | strong recovery, no clean cost (measured) |
| miniImageNet | guided-diffusion ImageNet-256 uncond (`diffpure_guided.py`) | 84→256→84 | acts like a blur: partial recovery, real cost (measured) |
| Tiny-ImageNet | improved-diffusion **ImageNet-64 uncond** (`diffpure_imagenet64.py`) | **native 64** | prediction: like CIFAR-100 |

## Files (all new; nothing on `main` or in the existing purify files was modified)

| File | Role |
|---|---|
| `diffpure_imagenet64.py` | Loads `imagenet64_uncond_100M_1500K.pt`. Reuses `diffpure_guided.py`'s `sigma_to_t` / `purify_in_chunks` / `cache_key` unchanged. |
| `purify_tiny_layer1a.py` | Fork of `main_baselines.py` with purification hooks, mirroring `purify_mini_layer1a.py` (all additions are marked `# >>> LAYER1A`). |
| `sigma_sweep_tiny.py` | Training-free σ\* sweep: distortion, poison survival, cost, and the seen/unseen memorization check. |
| `sigma_sweep_tiny.sbatch` | SLURM wrapper for the sweep. |
| `stage4_purify_tiny.sbatch` | SLURM wrapper for one purify arm (resumable, same variables as `stage4_purify_mini.sbatch`). |

## Setup

```bash
git clone https://github.com/openai/improved-diffusion.git /storage/home/axt5884/improved-diffusion
```

```bash
wget -P repro/artifacts/diffusion https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt
```

```bash
python repro/prepare_tiny_imagenet.py   # builds ./data/tiny_imagenet.npz
```

Use the **unconditional L_hybrid** checkpoint above. Don't use `imagenet64_cond_270M_250K.pt`, because the
purifier must not rely on the poisoned set's labels. Don't use the `_vlb_` variant either.

Model settings, from the improved-diffusion README plus that package's defaults:
`image_size=64, num_channels=128, num_res_blocks=3, learn_sigma=True, diffusion_steps=4000,
noise_schedule=cosine`, and **`rescale_timesteps=True`**. The last one is improved-diffusion's
default but not guided-diffusion's. If it is wrong, the model is given the wrong timestep and returns
plausible-looking garbage without raising an error. The loader sets it explicitly and refuses to
run without it.

**Backend.** `improved_diffusion` is used when it's importable. Otherwise `guided_diffusion` is
used, with flags that build the same UNet. Both were checked here: the improved-diffusion
state dict loads into the guided-diffusion UNet strictly, and the forward outputs match to
2e-8. A key mismatch is always fatal.

## σ\* on this schedule

Nearest-step match on the cosine 4000-step schedule. Always check the `resolved sigma` log line.

| σ\* | steps (T=4000) | steps (`--purify_respace 1000`) |
|---|---|---|
| 0.1 | 226 | 57 |
| 0.25 | 598 | 150 |
| **0.5** (default, = CIFAR-100) | **1159** | 290 |
| 1.0 | 1984 | 497 |

The full schedule is the cleaner choice. Respacing is about 4× faster if time is short. It is part
of the cache key and the output tag (`_sp1000`), so the two settings never get mixed up.

σ\* = 0 is an **exact identity** at 64×64, because there is nothing to resize. miniImageNet's
"Arm E" (resize only) therefore doesn't exist for Tiny-ImageNet: it's the same as Arm B.

## Run order: cheap checks before any training

**0. Prerequisite.** Tiny-ImageNet has no released victim or inversion artifacts, so stages 1–3
must be run first (`REPRODUCE.md` §5). That produces `repro/sweep/noise/<NOISE_TAG>/*.pkl` and
the noise pkls (the approach, `lamb` and `lamb_emp` are read from them automatically). The sweep in step 1 can run before that, using its random-sign probe.

**1. σ\* sweep plus the memorization check** (minutes):
```bash
sbatch --account=<ACCT> sigma_sweep_tiny.sbatch
```
After stage 3, repeat it with the real poison:
```bash
sbatch --account=<ACCT> --export=ALL,NOISE_TAG=tiny_ewc_reckless_eps0.3 sigma_sweep_tiny.sbatch
```
Output: `.../tinyImagenet_files/sigma_sweep/sweep_<tag>.tsv`, plus `_samples.pt` for visual checks.
* `survival` and `aligned` show how much of the poison is left after purification (1 = all of it).
  **If survival falls steeply as σ\* grows, as on CIFAR-100, rather than staying flat as on
  miniImageNet, native resolution is working.** You'll know this before spending any training compute.
* `psnr_seen` / `psnr_unseen` / `psnr_gap_seen_minus_unseen`: the memorization check. Seen means
  the task's training images, which come from ImageNet train and so were likely in the purifier's
  training data. Unseen means Tiny-ImageNet val. A large positive gap, especially at high σ\*, means the
  purifier is recalling images it has seen.
  **First confirm that Tiny-ImageNet's val images are not in ImageNet's training set.** This is
  not verified here. If they are, this comparison doesn't separate seen from unseen images.
* `sec_per_img` × 5,000 (one task: 10 classes × 500 images) gives the purification time per arm.
  Use it to choose `--time`, `RESPACE` and `PURIFY_CHUNK`.

Choose σ\* by protection, distortion and cost, the same rule as on miniImageNet. Keep 0.5 for
comparability with CIFAR-100 unless the sweep gives a clear reason not to.

**2. Arm D (clean + purify):** the purifier's own cost, expected to be about 0, as on CIFAR-100.
```bash
sbatch --account=<ACCT> --time=06:00:00 --export=ALL,VARIANT=clean,NOISE_TAG=tiny_ewc_reckless_eps0.3 stage4_purify_tiny.sbatch
```
APPROACH, LAMB and LAMB_EMP come from the noise pkl; EVAL_TAG defaults to
`<NOISE_TAG>_<VARIANT>_purify64_sigma<SIGMA_STAR>`.
```bash
```

**3. Attacked arms** at the fixed σ\*: `VARIANT=ours` (reckless or cautious `NOISE_TAG`) and
`VARIANT=uniform`. The undefended Arms A and B come from the normal `repro/sweep/stage4.sbatch`.

If a job dies, resubmit the same command. Purification resumes from `PURIFY_CACHE_DIR`.
Results go to `repro/sweep/results_purify/<EVAL_TAG>.log` and
`.../tinyImagenet_files/purify_results/summary.tsv`. The `method` column reads like
`ours_purify64_sigma0.5`, and the `resize` column holds `64` (native) or `64sp1000`.

## Not included

* No tiny rows were added to `repro/sweep/configs.tsv`, and there is no tiny version of
  `submit_sweep_purify.sh`. Tiny has no checkpoints until stages 1–3 have run.
  Once they have, submit the arms by hand as above, or add a tiny configs file.
* Not run on the real checkpoint here: the download host was blocked in the setup sandbox.
  The pipeline was checked on CPU with random weights in the same architecture: both backends,
  the σ\*→step table above, an exact identity at σ\*=0, deterministic seeded purification, and an
  end-to-end `sigma_sweep_tiny.py` run on a synthetic `.npz` and noise pkl. On the cluster, the
  first log line to check is `[diffpure_imagenet64] loaded ... (121M params, ... T=4000 ...)`.
