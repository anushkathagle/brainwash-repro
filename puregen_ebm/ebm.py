"""Load a pretrained PureGen EBM and purify images with mid-run Langevin dynamics.

    x_{k+1} = x_k - (eps^2 / 2) * grad_x [ E(x_k) / T ] + eps * N(0, I)        (paper Eq. 9)

with x in [-1, 1]. Defaults match the authors' code (scripts/PureGen.py): eps = 1.25e-2, T = 1e-4, 150 steps.
"""
import importlib.util
import os

import torch
import torch.nn.functional as F

DEFAULT_EBM = 'SunayBhat1/puregen-ebm-cinic10-imagenet'   # trained on the ImageNet part of CINIC-10 (generic images)


def _ebm_class(puregen_repo):
    repo = puregen_repo or os.environ.get('PUREGEN_REPO')
    if not repo:
        raise ValueError('pass --puregen_repo (or set PUREGEN_REPO) = path to a clone of SunayBhat1/PureGen_PoisonDefense')
    path = os.path.join(repo, 'models', 'EBMs.py')
    spec = importlib.util.spec_from_file_location('puregen_EBMs', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EBMSNGAN32


def load_ebm(ebm_path=DEFAULT_EBM, puregen_repo=None, device=None):
    """ebm_path: Hugging Face id or a local directory produced by save_pretrained / huggingface-cli download."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    EBM = _ebm_class(puregen_repo)
    ebm = EBM.from_pretrained(ebm_path)
    ebm = ebm.to(device).eval()
    for p in ebm.parameters():
        p.requires_grad_(False)
    return ebm


def langevin_purify(ebm, x01, steps=150, temp=1e-4, eps=1.25e-2, init_noise=0.0, batch_size=500, quantize=True):
    """x01: (N,3,H,W) float in [0,1] on any device. Returns purified images in [0,1] on x01's device/dtype.

    quantize=True rounds to 8-bit like the authors' pipeline (they save purified data as PIL images).
    init_noise: std of Gaussian noise added in [-1, 1] units BEFORE the chain (DiffPure-like knob; the authors' code
    has this hook but fixes it at 0, so large values are outside what the EBM was trained for).
    """
    dev = next(ebm.parameters()).device
    out = torch.empty_like(x01)
    with torch.enable_grad():
        for i in range(0, len(x01), batch_size):
            x = (x01[i:i + batch_size].to(dev).float() * 2 - 1).detach()
            if init_noise > 0:
                x = x + init_noise * torch.randn_like(x)
            for _ in range(steps):
                x.requires_grad_(True)
                e = ebm(x).sum() / temp
                g, = torch.autograd.grad(e, x)
                x = (x.detach() - (eps ** 2 / 2) * g + eps * torch.randn_like(x)).detach()
            y = ((x + 1) / 2).clamp(0, 1)
            if quantize:
                y = (y * 255).round() / 255
            out[i:i + batch_size] = y.to(x01.device, x01.dtype)
    return out


@torch.no_grad()
def energies(ebm, x01, batch_size=1000):
    dev = next(ebm.parameters()).device
    return torch.cat([ebm(x01[i:i + batch_size].to(dev).float() * 2 - 1).flatten().cpu()
                      for i in range(0, len(x01), batch_size)])


def _psnr(a, b):
    mse = F.mse_loss(a.float(), b.float(), reduction='none').flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


def auroc(neg, pos):
    """P(score_pos > score_neg); 0.5 = indistinguishable. Rank-based, no sklearn needed."""
    s = torch.cat([neg, pos])
    r = torch.empty_like(s)
    r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    return ((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def purify_with_report(ebm, x_clean, x_pois, steps, **kw):
    """Purify the (possibly poisoned) set and measure how much perturbation survives. Returns (P(x_pois), report).

    Computed on the poisoned rows (where x_pois != x_clean), mirroring purevqgan.purify.perturbation_report:
      survival_l2   = ||P(x+d) - P(x)|| / ||d||   (1 = perturbation passes through, ~0 = erased)
      psnr_clean    = PSNR(P(x), x)               (utility cost)
      psnr_poisoned = PSNR(P(x+d), x)             (how close the purified poison gets to the clean image)
    Clean and poisoned sets are purified with the same seed and batch layout, so the Langevin noise is identical
    and the difference isolates the perturbation.
    """
    xc, xp = x_clean.float().cpu(), x_pois.float().cpu()
    d = xp - xc
    rows = d.flatten(1).abs().amax(1) > 0
    torch.manual_seed(0)
    pp = langevin_purify(ebm, x_pois, steps, **kw)
    rep = {'steps': steps, 'n_total': len(xc), 'n_poisoned': int(rows.sum())}
    if rows.any():
        torch.manual_seed(0)
        pc = langevin_purify(ebm, x_clean, steps, **kw).float().cpu()
        ppc = pp.float().cpu()
        rep['psnr_clean'] = _psnr(pc, xc)
        rep['input_psnr_poisoned'] = _psnr(xp[rows], xc[rows])
        rep['psnr_poisoned'] = _psnr(ppc[rows], xc[rows])
        num = (ppc[rows] - pc[rows]).flatten(1).norm(dim=1)
        rep['survival_l2'] = (num / d[rows].flatten(1).norm(dim=1).clamp_min(1e-12)).mean().item()
    else:
        rep['psnr_clean'] = _psnr(pp.float().cpu(), xc)
    return pp, rep
