# Defense layer 1 (alternative) — PureGen-EBM purification against BrainWash

Paper: Pooladzandi, Jiang, Bhat, Pottie, *PureEBM: Universal Poison Purification via Mid-Run Dynamics of Energy-Based
Models* (arXiv:2405.19376). Code + released weights: https://github.com/SunayBhat1/PureGen_PoisonDefense.

**What is ours vs. theirs.** We don't copy their code: the repo's README says CC BY-ND 4.0 while its LICENSE file says
MIT, so we play safe. The EBM network class is imported at runtime from a clone of their repo (`models/EBMs.py`). The
pretrained weights `SunayBhat1/puregen-ebm-cinic10-imagenet` (trained on CINIC-10's ImageNet images, so no victim data)
are loaded from Hugging Face. The Langevin loop in `puregen_ebm/ebm.py` is written from the paper (Eq. 9 / Alg. 1)
with their defaults: inputs in [-1, 1], step eps 1.25e-2, temperature 1e-4, 150 steps, output rounded to 8-bit.
Their code targets TPUs (`torch_xla`); ours is plain PyTorch.

| piece | file |
|---|---|
| load EBM, Langevin purification, energies, survival report | `puregen_ebm/ebm.py` |
| step 1: energy go/no-go check (minutes) | `python -m puregen_ebm.energy_check ...` |
| step 2: stage-4 hook | `main_baselines.py --ebm_purify --ebm_steps N [--ebm_path ...] [--puregen_repo ...]` |
| Colab notebook (steps 1 + 2) | `repro/defense/ebm_brainwash_colab.ipynb` |
| cluster | `repro/sweep/stage4.sbatch` with `EBM_STEPS=150 PUREGEN_REPO=/path/to/clone` (+ `EBM_PATH=<local dir>` if compute nodes are offline) |

Tested here: the module, the stage-4 hook and the energy check, using a randomly initialised EBM of the authors'
architecture loaded through the same `from_pretrained` path. **Not tested here:** the released weights, because my
sandbox can't reach Hugging Face, and a real CUDA stage-4 run. The first Colab cell that loads the EBM checks the first.
