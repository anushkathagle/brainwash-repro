"""PureVQ-GAN (Branch et al., 2025, arXiv:2509.25792) re-implementation for the BrainWash defense study.

No official code was released; this follows Sec. 3 + Appendix B of the paper. Places where the paper
is silent and we had to choose are marked `# [choice]` in the source and listed in
repro/defense/README_purevqgan.md.
"""
from .model import VQGAN, PatchDiscriminator, VectorQuantizer, build_vqgan
from .purify import load_purifier, purify_tensor
