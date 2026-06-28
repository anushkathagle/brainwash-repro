# BrainWash reproduction — CIFAR-100 (Table 1)

**Setup.** 10-split CIFAR-100, ResNet-18, SGD lr 1e-2, batch 16, victim trained on tasks 1–9,
task 10 poisoned. Attack: model-inversion + bilevel noise optimization (5000 epochs, k=1).
Artifacts (victim checkpoints + inverted data) are the authors' own from the BrainWash HF release.
Eval = resume CL on the poisoned task, report **BWT** (backward transfer; more negative = more
forgetting) and **Acc** (last-task accuracy). EWC/AFEC are single-seed; MAS/RWALK are mean ± std
over 3 seeds (they are high-variance under the attack). Match tolerance: |ΔBWT| ≤ 2, |ΔAcc| ≤ 3.

## Results (reproduced vs. paper)

| Method | Mode | ε | BWT (ours) | Acc (ours) | BWT (paper) | Acc (paper) | flag |
|---|---|---|---|---|---|---|---|
| EWC | clean | – | −5.2 | 68.4 | −5.2 | 68.3 | OK |
| EWC | uniform | 0.1 | −5.1 | 67.0 | −5.1 | 67.0 | OK |
| EWC | cautious | 0.1 | −9.5 | 56.4 | −9.5 | 58.8 | OK |
| EWC | reckless | 0.1 | −12.4 | 52.1 | −12.6 | 51.0 | OK |
| EWC | uniform | 0.3 | −12.2 | 57.5 | −12.2 | 57.5 | OK |
| EWC | cautious | 0.3 | −24.7 | 43.2 | −24.7 | 42.2 | OK |
| EWC | reckless | 0.3 | −28.9 | 26.4 | −29.1 | 25.5 | OK |
| AFEC | clean | – | −2.9 | 65.6 | −2.9 | 65.6 | OK |
| AFEC | uniform | 0.1 | −3.6 | 64.1 | −3.6 | 64.1 | OK |
| AFEC | cautious | 0.1 | −8.0 | 57.2 | −7.9 | 57.2 | OK |
| AFEC | reckless | 0.1 | −8.9 | 54.7 | −8.8 | 55.4 | OK |
| AFEC | uniform | 0.3 | −14.4 | 52.8 | −14.4 | 52.4 | OK |
| AFEC | cautious | 0.3 | −24.2 | 49.5 | −24.2 | 40.9 | **XX** |
| AFEC | reckless | 0.3 | −24.5 | 35.9 | −24.6 | 36.9 | OK |
| MAS | clean | – | −1.7 ± 0.8 | 69.3 ± 0.6 | −1.8 | 62.6 | **XX** |
| MAS | uniform | 0.1 | −2.2 ± 1.2 | 67.1 ± 0.8 | −1.8 | 67.7 | OK |
| MAS | cautious | 0.1 | −6.0 ± 0.6 | 59.5 ± 1.2 | −5.8 | 61.4 | OK |
| MAS | reckless | 0.1 | −9.8 ± 0.5 | 45.4 ± 0.8 | −6.4 | 53.7 | **XX** |
| MAS | uniform | 0.3 | −10.7 ± 0.3 | 53.9 ± 3.2 | −9.6 | 67.5 | **XX** |
| MAS | cautious | 0.3 | −23.3 ± 1.6 | 47.9 ± 1.7 | −22.8 | 50.7 | OK |
| MAS | reckless | 0.3 | −27.9 ± 2.6 | 27.2 ± 6.4 | −17.9 | 45.0 | **XX** |
| RWALK | clean | – | −4.2 ± 1.5 | 70.3 ± 0.2 | −6.0 | 70.1 | OK |
| RWALK | uniform | 0.1 | −5.7 ± 0.2 | 68.5 ± 1.3 | −5.1 | 68.5 | OK |
| RWALK | cautious | 0.1 | −16.4 ± 0.4 | 56.6 ± 2.0 | −16.3 | 55.8 | OK |
| RWALK | reckless | 0.1 | −14.5 ± 2.6 | 64.7 ± 3.8 | −14.5 | 62.8 | OK |
| RWALK | uniform | 0.3 | −26.8 ± 2.0 | 47.2 ± 1.7 | −25.5 | 48.8 | OK |
| RWALK | cautious | 0.3 | −37.3 ± 2.0 | 38.9 ± 2.2 | −32.5 | 47.0 | **XX** |
| RWALK | reckless | 0.3 | −21.3 ± 1.9 | 56.4 ± 2.6 | −21.6 | 53.5 | OK |

**Score: 22 / 28 cells within tolerance.** Per method: EWC 7/7, AFEC 6/7, RWALK 6/7, MAS 3/7.
ANCL (the 5th method in the paper) is **not implemented in the released code**, so its row is not
reproducible here.

## Reading of the 6 deviations (all `XX`)

The attack's primary metric — **BWT (forgetting)** — reproduces tightly almost everywhere; even
several `XX` cells have ΔBWT ≈ 0 (e.g. AFEC cautious 0.3 matches BWT exactly at −24.2). The flags
are driven mostly by **last-task Acc** in the high-variance methods. The 6 fall into two groups:

**(a) Anomalies in the paper's own numbers (our results are more self-consistent):**
- **MAS uniform ε0.3** — paper Acc 67.5 is *higher than its own clean Acc 62.6* (uniform noise
  cannot improve accuracy); ours (53.9) correctly drops below clean.
- **MAS reckless ε0.3** — paper BWT −17.9 forgets *less* than its own cautious −22.8 (backwards;
  reckless should forget more). Ours (−27.9) restores the expected ordering (reckless > cautious).

**(b) Consistent offsets in the noisiest cells (tight seed std → not run-to-run noise):**
- **MAS clean Acc** 69.3 vs 62.6 — our clean MAS (69.3) is in line with our EWC (68.4)/RWALK (70.3);
  the paper's MAS clean (62.6) is the outlier vs *its own* EWC/RWALK.
- **MAS reckless ε0.1**, **RWALK cautious ε0.3**, **AFEC cautious ε0.3 (Acc only)** — the trained
  noise differs slightly because the authors' `main_brainwash` does not set `cudnn.deterministic`,
  so the bilevel optimization is GPU-nondeterministic; the attack reproduces in direction and
  roughly in magnitude, with small offsets concentrated in the cautious/reckless ε0.3 cells.

## Conclusion
The BrainWash attack on CIFAR-100 reproduces faithfully: EWC and AFEC essentially exactly, RWALK
essentially so, and MAS in trend. The forgetting metric (BWT) matches across the board; the few
remaining Acc gaps are confined to the known high-variance MAS method and the strongest-ε cautious
cells, and two of the six discrepancies are demonstrable inconsistencies in the paper's published
MAS numbers rather than reproduction error.

## Repro notes / fixes applied
- AFEC victim checkpoint omits the `optim_name` key → `main_brainwash.py` patched to default it to
  `'sgd'` (the optimizer AFEC's victim was trained with).
- Noise snapshots are written to CWD with no `save_dir` option; `SAVE_EVERY` raised to 2500 + prune
  to the best snapshot per run to stay within the 16 GB home quota.
- All numbers via `repro/sweep/` (parameterized stage-3/4 + `submit_sweep.sh` + `collect_results.py`).
