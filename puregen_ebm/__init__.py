"""PureGen-EBM purification (Pooladzandi, Jiang, Bhat, Pottie 2024; arXiv:2405.19376) for the BrainWash defense study.

We do NOT copy the authors' code. The EBM network class is imported at runtime from a clone of
https://github.com/SunayBhat1/PureGen_PoisonDefense (models/EBMs.py), the released pretrained weights are
loaded from Hugging Face, and the Langevin purification loop is written here from the paper (Eq. 9 / Alg. 1)
with the authors' default constants (step eps = 1.25e-2, temperature 1e-4, 150 steps, inputs in [-1, 1]).
"""
from .ebm import DEFAULT_EBM, auroc, energies, langevin_purify, load_ebm, purify_with_report
