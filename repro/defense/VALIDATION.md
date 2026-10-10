# Validating PureVQ-GAN in the paper's own setting (do this BEFORE BrainWash)

> **Fast path:** `repro/defense/validate_narcissus_colab.ipynb` runs rungs 1-4 for Narcissus end to end on Colab, using a compact
> re-implementation of the attack (`narcissus_lite.py`). Its undefended-PSR gate (rung 2) is what justifies trusting that attack.

Goal: separate "my code is wrong" from "the method doesn't work on BrainWash". We reproduce the
paper's CIFAR-10 setting first. **Write the thresholds below into your notes before running**, so a
miss can't be explained away afterwards.

Paper setting (Sec. 4, Table 1): CIFAR-10, ResNet-18, 1 % poison budget, ε = 8/255.
Attacks: Gradient Matching (GM), Narcissus (NS), Bullseye Polytope (BP). The purifier is trained on the
**same poisoned training set** the classifier uses (Sec. 3.3), then a ResNet-18 is trained on the
purified set. Paper numbers to compare against (undefended → PureVQ-GAN):

| attack | PSR undefended | PSR paper | nat. acc paper |
|---|---|---|---|
| GM | 44.0 | 0.00 | 92.8 |
| NS | 43.95 | 1.64 | 94.61 |
| BP | 46.0 | 0.00 | 91.43 |

## The harness
`repro/defense/validate_paper.py` takes a poisoned dataset `.npz` (format in its docstring) and
reports natural accuracy and PSR for any of the three attacks. It has two stages: `purifier` (health
of the purifier on clean data) and `classify` (train ResNet-18, report acc and PSR). It has been
smoke-tested on random data only. **It has not been run on real poisons.**

## Getting the poisons (needs internet, so use Colab; I couldn't fetch these from my sandbox)
Use the attackers' official code; don't re-implement the attacks, or you add a second unfaithful piece.
* **GM (Witches' Brew):** `JonasGeiping/poisoning-gradient-matching`. CIFAR-10, ResNet-18, `eps=8`,
  `budget=0.01`. Export the poisoned train set, the targets, and the adversarial labels.
* **Narcissus:** `reds-lab/Narcissus`. Export the poisoned train set and the optimized trigger
  (multiplied by the test-time scale, as an additive array).
* **BP:** `ucsb-seclab/BullseyePoison`, fine-tuning setting as in the paper.

Convert each to the `.npz` format. Also save `x_train_clean` if you can: it enables the
perturbation-survival diagnostic. **Check that your poisons reproduce the "undefended" PSR in the
table above before testing any defense.** If your attack doesn't succeed undefended, the experiment
is meaningless.

## The ladder (stop at the first failing rung and fix it there)
| # | Check | Command | Pass if (decide before looking) |
|---|---|---|---|
| 0 | Purifier trains healthily | `train.py --dataset npy:<x_train>` | `codes_used` ≥ 200 of 512, val PSNR rises, samples look like inputs |
| 1 | Clean-reference classifier | `--stage classify --clean_train_npz clean.npz` | nat. acc ≥ 92 % |
| 2 | Attack works undefended | `--stage classify --tag undefended` | PSR within ±10 points of the table |
| 3 | Purifier clean fidelity | `--stage purifier` | report PSNR; **do not require 40 dB** (see below) |
| 4 | Defended classifier | `--stage classify --purifier ...` | PSR ≤ 10 (paper claims ≈ 0–1.6) and nat. acc ≥ 88 % |
| 5 | Ablation: no GAN | train with `--gan_weight 0` | paper says blurrier + lower acc; check direction |
| 6 | Ablation: K = 64 | `--num_codes 64` | paper says PSR still ≈ 0 |

## How to read the outcomes
* **Pass at 4:** the implementation is faithful enough. A BrainWash failure is then about the
  method or the threat, not a bug.
* **Fail at 0–2:** an engineering problem (training, data conversion, attack). Not evidence about the method.
* **Rung 4 fails but 0–2 pass:** ambiguous. Try, in this order, each being something the paper
  leaves unspecified: larger G (`--base_ch 256 --num_res_blocks 2`, ~58M; the paper's headline is
  300M), longer training, `--purify_test`, and two passes. Record each. If PSR stays high after
  these, report it honestly as a failure to replicate. That is a legitimate thesis finding, not a
  failure of the thesis.
* **Known paper concerns** (so you aren't surprised): the "PSNR > 40 dB" claim looks implausible for
  an 8×8 latent; two of its baselines (EPIC, FRIENDS) cite unrelated papers; every cell reaching
  0 % PSR with 1 %-budget attacks is unusually clean; and there is no seed count or error bar for PureVQ-GAN's own row
  in the main text. None of this means the method fails; it means your replication target may be softer than printed.

## Protocol details to fix and report
Classifier training recipe (here: SGD, OneCycle, 40 epochs, standard crop/flip; the paper gives none),
number of seeds (use ≥ 3 per cell and report mean ± std), whether test inputs are purified
(`--purify_test`; Sec. 3.3 describes inference-time purification, Sec. 4 says "trained on purified
datasets"), and the PSR definition per attack (GM/BP: fraction of targets classified as the adversarial
label; NS: fraction of non-target-class test images flipped to the target class when the trigger is added).
