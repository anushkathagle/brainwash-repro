"""Apply a trained PureVQ-GAN purifier, and measure how much of a BrainWash perturbation survives it.

Library use (this is what main_baselines.py calls):
    G = load_purifier('purifiers/cifar10_K512/purifier.pt')
    x_pure = purify_tensor(G, x, passes=1)

Stand-alone diagnostic (no CL training; seconds on a GPU) -- useful to pre-screen sweep points:
    python -m purevqgan.purify --purifier purifiers/cifar10_K512/purifier.pt \
        --experiment split_cifar100 --noise_pkl repro/sweep/noise/cifar100_ewc_cautious_eps0.1/<file>.pkl
"""
import argparse
import json
import pickle as pkl

import torch
import torch.nn.functional as F

from .model import VQGAN, _torch_load


def load_purifier(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = _torch_load(path, map_location='cpu')
    G = VQGAN(**ck['config'])
    G.load_state_dict(ck['G'])
    return G.to(device).eval()


@torch.no_grad()
def purify_tensor(G, x, passes=1, batch_size=500):
    """x: float tensor (N,3,H,W) in [0,1] on any device. Returns a purified tensor on x's device/dtype."""
    dev = next(G.parameters()).device
    out = torch.empty_like(x)
    for i in range(0, len(x), batch_size):
        out[i:i + batch_size] = G.purify(x[i:i + batch_size].to(dev).float(), passes=passes).to(x.device, x.dtype)
    return out


@torch.no_grad()
def code_indices(G, x, batch_size=500):
    dev = next(G.parameters()).device
    idx = []
    for i in range(0, len(x), batch_size):
        z = G.encode(x[i:i + batch_size].to(dev).float())
        idx.append(G.quantizer.nearest(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()).view(len(z), -1).cpu())
    return torch.cat(idx)


def _psnr(a, b):
    mse = F.mse_loss(a.float(), b.float(), reduction='none').flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def perturbation_report(G, x_clean, x_poisoned, passes=1):
    """How much of delta = x_poisoned - x_clean survives purification (computed on the poisoned rows only).

    survival_l2   = ||P(x+d) - P(x)||_2 / ||d||_2     (1 = perturbation fully passed through, 0 = erased)
    code_flip     = fraction of latent positions whose codebook index changes because of d
    psnr_clean    = PSNR(P(x), x)        -> purifier distortion on clean data (utility cost)
    psnr_poisoned = PSNR(P(x+d), x)      -> how close the purified poison gets to the clean image
    """
    x_clean, x_poisoned = x_clean.float().cpu(), x_poisoned.float().cpu()
    d = x_poisoned - x_clean
    rows = d.flatten(1).abs().amax(1) > 0
    rep = {'n_total': len(x_clean), 'n_poisoned': int(rows.sum()),
           'input_psnr_poisoned': _psnr(x_poisoned[rows], x_clean[rows]) if rows.any() else None}
    p_clean = purify_tensor(G, x_clean, passes)
    rep['psnr_clean'] = _psnr(p_clean, x_clean)
    if rows.any():
        xc, xp, pc = x_clean[rows], x_poisoned[rows], p_clean[rows]
        pp = purify_tensor(G, xp, passes)
        num = (pp - pc).flatten(1).norm(dim=1)
        den = d[rows].flatten(1).norm(dim=1).clamp_min(1e-12)
        rep['survival_l2'] = (num / den).mean().item()
        rep['residual_linf'] = (pp - pc).flatten(1).abs().amax(1).mean().item()
        rep['psnr_poisoned'] = _psnr(pp, xc)
        rep['code_flip'] = (code_indices(G, xp) != code_indices(G, xc)).float().mean().item()
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--purifier', required=True)
    ap.add_argument('--experiment', required=True, choices=['split_cifar100', 'split_mini_imagenet', 'split_tiny_imagenet'])
    ap.add_argument('--noise_pkl', required=True)
    ap.add_argument('--passes', type=int, default=1)
    ap.add_argument('--tasknum', type=int, default=10)
    ap.add_argument('--json', help='also write the report here')
    a = ap.parse_args()

    from .data import load_brainwash_split
    ck = pkl.load(open(a.noise_pkl, 'rb'))
    t = ck['pretrained_ckpt']['task_num']
    x = load_brainwash_split(a.experiment, a.tasknum)[t]['train']['x']
    x = x[torch.as_tensor(ck['rnd_idx_train']).cpu()]
    xp = torch.clamp(x + torch.as_tensor(ck['latest_noise']).cpu(), 0, 1)
    rep = perturbation_report(load_purifier(a.purifier), x, xp, a.passes)
    rep.update(purifier=a.purifier, noise_pkl=a.noise_pkl, passes=a.passes)
    print(json.dumps(rep, indent=2))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(rep, f, indent=2)


if __name__ == '__main__':
    main()
