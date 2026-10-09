# Defense layer 1 — PureVQ-GAN input purification (thesis RQ1)

**PureVQ-GAN** (Branch et al., 2025, [arXiv:2509.25792](https://arxiv.org/abs/2509.25792)) has no
public code, so this is a from-scratch re-implementation built from Sec. 3 and Appendix B of the
paper. It plugs into the BrainWash stage-4 evaluation. Every sample of the current (poisoned) task
goes through the purifier, `x̂ = G(x)`, before the continual-learning update. Nothing else in the
pipeline changes, so a defended run differs from its undefended baseline only by the purifier.

```
purevqgan/                     the method (importable package)
  model.py                       VQ-VAE generator (encoder / codebook / decoder) + PatchGAN discriminator
  train.py                       python -m purevqgan.train      — train a purifier
  purify.py                      python -m purevqgan.purify     — load/apply + perturbation diagnostics
  data.py                        purifier training-data sources (cifar10, stl10, npy:, folder:, bw_poisoned:)
main_baselines.py              + --purifier / --purify_passes / --purify_tag  (stage-4 hook)
repro/sweep/stage4.sbatch      + optional PURIFIER / PURIFY_PASSES / RESULTS_DIR env vars
repro/defense/
  train_purifier.sbatch          train one purifier on an A100
  submit_purevqgan.sh            defended stage-4 sweep (reuses the attack sweep's noise)
  collect_defense.py             defended vs undefended table (+ paired t-test across seeds)
```

## 1. What the paper specifies vs. what we had to choose

| Item | Paper | This implementation |
|---|---|---|
| Generator | VQ-VAE: `E → nearest codebook vector (eq. 1) → D` | same |
| Codebook | K = 512, d = 256 | `--num_codes 512 --z_ch 256` |
| Latent | 32×32 → 8×8 | `--img_size 32 --latent_size 8` (any power-of-two ratio) |
| Encoder | "4 residual blocks with stride-2 convolutions" | 1 res-block + stride-2 conv per level × 2 levels, then 2 res-blocks at 8×8 (= 4 res-blocks) |
| Decoder | mirror, transposed convs | mirror, `ConvTranspose2d(4, stride 2)` |
| Norm | GroupNorm | GroupNorm (32 groups) |
| Discriminator | PatchGAN, 3 layers, spectral norm | taming-style `NLayerDiscriminator(n_layers=3)` + spectral norm |
| Loss | `L_rec + L_VQ + λ·L_GAN`, β = 0.25, λ = 0.1 | same; `L_VQ` exactly eq. 2 (straight-through) |
| Optimiser | Adam, lr 4e-4, batch 256, 100 epochs | same |
| Inference | one forward pass (eq. 4, optional more) | `--purify_passes 1` (default) |

Where the paper says nothing, we chose the following. Each choice is marked `[choice]` in the code.
State these in the thesis methods section:

1. **Adam betas (0.5, 0.9).** These are the VQGAN/taming defaults. The paper doesn't give betas.
2. **Generator adversarial term** uses the standard *non-saturating* form `−log D(x̂)` of eq. 3.
   λ is fixed at 0.1. We don't use taming's adaptive weight because the paper says λ is fixed.
3. **No discriminator warm-up by default** (`--disc_start_epoch 0`). Raise it if training goes unstable.
4. **Width.** `--base_ch 128` gives a ~12M-parameter G. The paper's headline results use a
   **300M** model, and its ablation shows clean accuracy improving up to ~50M with diminishing returns
   after that. Scale with `--base_ch` / `--num_res_blocks` / `--n_mid`. Measured G sizes:
   `64/1/2` 3.5M · `128/1/2` 12.3M · `256/1/2` 46.5M · `256/2/2` 58.3M · `384/2/4` 172M ·
   `512/3/4` 351M. The script prints the count. Report the size you use.
5. **Codebook init and dead-code restart (the one real deviation).** Codes start uniform in ±1/K
   (van den Oord) with no EMA. Every 200 steps, codes that weren't used are re-seeded from random
   encoder outputs (`--restart_dead_codes 1`, the default). The paper doesn't mention this, but
   without it the codebook **collapses**. In our sanity run on natural-image 32×32 patches
   (3.5M-param G, batch 64, CPU):

   | | epoch 0 codes used | epoch 0 val PSNR | epoch 1 |
   |---|---|---|---|
   | plain VQ (paper as written) | 20 / 512 | 21.3 dB | 28 / 512, 22.4 dB |
   | + dead-code restart (default) | 475 / 512 | 25.0 dB | – |

   The paper reports 500–600 codes in use for K = 2048, so the authors evidently avoided collapse
   somehow. Use `--restart_dead_codes 0` for the literal version, and state this choice in the thesis.
6. **Random horizontal flips** during purifier training.
7. **Reconstruction MSE is computed in [-1, 1] space.** That is a constant ×4 vs. [0, 1] and only
   rescales λ's effective weight slightly.

### Purifier training data (an important design decision)
* **Paper protocol:** the purifier is trained on the *same poisoned training set* the classifier uses.
  For an ablation, use `--dataset bw_poisoned:<noise.pkl> --bw_experiment split_cifar100`.
* **Your proposal (Sec. 4.4):** "a fixed prior trained only on generic natural images, with no access
  to the victim's task data". Defaults that respect this:
  * split CIFAR-100 → `--dataset cifar10` (32×32, same source collection, **class-disjoint** from CIFAR-100)
  * split miniImageNet → `--dataset stl10 --img_size 84 --latent_size 21` (100k unlabeled
    ImageNet-derived images. 84 = 21·4 keeps the same 4× spatial compression as CIFAR)
  * split TinyImageNet → `--img_size 64 --latent_size 16` with `stl10` or another generic set

  STL-10 comes from ImageNet, so it may share *classes* with miniImageNet, but not the victim's images.
  Note that in the thesis.

### A note on the paper's "PSNR > 40 dB"
An 8×8 latent with K = 512 stores 64 × 9 = 576 bits per image. Near-lossless 40 dB reconstruction
of CIFAR images through that bottleneck is far above what VQ-GANs at this compression usually
reach (mid-20s dB is more typical). **Don't treat 40 dB as a target you must hit.** It also pulls
against the defense: a purifier that reconstructs everything losslessly would also pass the
perturbation through. That tension is exactly what the RQ1 strength sweep measures.

## 2. Workflow on Roar Collab

The existing `brainwash` env (py3.10, torch 1.12.1) is enough. No new dependencies.
Purifiers go in `purifiers/<tag>/` (git-ignored). Symlink that to `/storage/work` like the noise.

```bash
# ---- 1. train purifiers (≈ 1 A100 each; resumable — just re-submit) ----
sbatch --account=$ACCT --export=ALL,PVQ_TAG=cifar10_K512 repro/defense/train_purifier.sbatch
# RQ1 strength axis = codebook capacity:
for K in 64 128 256 1024; do
  sbatch --account=$ACCT --export=ALL,PVQ_TAG=cifar10_K$K,K=$K repro/defense/train_purifier.sbatch
done
# miniImageNet purifier:
sbatch --account=$ACCT --export=ALL,PVQ_TAG=stl10_84_K512,PVQ_DATASET=stl10,IMG_SIZE=84,LATENT=21 \
       repro/defense/train_purifier.sbatch
# check: purifiers/<tag>/train_log.jsonl (val_psnr, codes_used, perplexity), samples_eXXX.png (top: input, bottom: recon)

# ---- 2. cheap pre-screen: how much BrainWash perturbation survives each purifier (seconds, no CL) ----
python -m purevqgan.purify --purifier purifiers/cifar10_K512/purifier.pt --experiment split_cifar100 \
    --noise_pkl "$(ls -t repro/sweep/noise/cifar100_ewc_cautious_eps0.1/*.pkl | head -1)"

# ---- 3. defended stage-4 evaluations (reuses the attack sweep's noise) ----
DRY_RUN=1 ACCT=$ACCT PURIFIERS="purifiers/cifar10_K512/purifier.pt" repro/defense/submit_purevqgan.sh cifar100 ewc
ACCT=$ACCT PURIFIERS="$(ls purifiers/cifar10_K*/purifier.pt)" MODES="clean cautious" \
    repro/defense/submit_purevqgan.sh cifar100
# 5 seeds (proposal 4.7) — needs the attack noise for seeds 1-4 first (repro/sweep with SEEDS="1 2 3 4"):
SEEDS="0 1 2 3 4" ACCT=$ACCT PURIFIERS=purifiers/cifar10_K512/purifier.pt repro/defense/submit_purevqgan.sh cifar100

# ---- 4. results ----
python repro/defense/collect_defense.py --csv defense_pvq.csv
```

Running one by hand is just the usual stage-4 command plus the purifier flags:
```bash
python main_baselines.py --experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 \
  --batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --checkpoint <noise.pkl> --init_acc --addnoise \
  --purifier purifiers/cifar10_K512/purifier.pt --purify_passes 1
```

### What each run logs
`Purification report : ...` is printed before training on task T:

| key | meaning |
|---|---|
| `psnr_clean` | PSNR(P(x), x): how much the purifier distorts clean data (the utility cost) |
| `survival_l2` | ‖P(x+δ) − P(x)‖ / ‖δ‖ on poisoned samples. 1 = δ passes through, 0 = δ erased |
| `code_flip` | fraction of latent positions whose codebook index δ changes (VQ-specific) |
| `psnr_poisoned` | PSNR(P(x+δ), x): how close the purified poison is to the clean image |

`acc_mat_*_<variant>_pure-<tag>.npy` is saved next to the undefended `acc_mat_*` files, so nothing
gets overwritten.

### Mapping to the proposal's RQ1 design
* **Clean (no attack) + purifier** is the utility-cost row. Use it for the "within one seed-std of
  clean accuracy" operating-point rule (Sec. 4.7).
* **Strength sweep** = codebook capacity K (and optionally `--latent_size`/width). For each K,
  `collect_defense.py` gives BWT and Acc. The operating point is the K with the lowest forgetting
  whose clean-purified accuracy stays within the margin.
* **Primary threat:** the cautious attacker (`MODES="clean cautious"`). The 2 % / 5 % injection-rate
  conditions come from stage 3 (the attack), not from this code.

## 3. Scope and caveats (worth stating in the thesis)
* **Threat model is non-adaptive.** The BrainWash noise is crafted without knowledge of the purifier.
  An adaptive attacker who differentiates through G (it's a single forward pass, so this is feasible)
  is out of scope here and would be a natural limitation/future-work item.
* Only the **current task's training inputs** are purified. Test sets, the validation set used for
  LR scheduling, and earlier tasks are untouched, so BWT stays comparable with the undefended
  baseline. (The PureVQ-GAN paper also purifies test images at inference. That isn't done here,
  because it would change the accuracy matrix of tasks the checkpoint already learned.)
* Verified in this repo: a short natural-image sanity run (table above) shows it learns. The code was also smoke-tested on CPU (shapes for 32/64/84 px, training loop,
  checkpoint save/load, the stage-4 hook, the sweep dry-run, the collector). **It has not yet been
  trained at full scale on CIFAR-10 or run on the cluster.** Check `train_log.jsonl` on the first
  real run before launching sweeps.
