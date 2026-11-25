"""Matched-distortion controls for the PureVQ-GAN result (is it the VQ bottleneck, or would any comparable distortion do?).

Each control is a plain, training-free image transform whose strength is tuned so that it distorts CLEAN task-T images
exactly as much as the purifier does (same mean PSNR(T(x), x)). The controls then go through the same stage-4 arms as
PureVQ-GAN (clean + T, poisoned + T). If they recover about as much forgetting, the effect is "BrainWash noise is fragile
to input transforms"; if PureVQ-GAN clearly beats them at equal clean distortion, the bottleneck gets the credit.

Transforms (param = strength knob, calibrated by `calibrate`):
  blur    Gaussian blur, reflect padding                    param = sigma in pixels      (continuous -> bisection)
  jpeg    JPEG re-encoding with PIL (default 4:2:0 chroma)  param = quality 1..95        (integer   -> grid search)
  downup  area-downsample to s x s, bilinear back up        param = s, the small side    (integer   -> grid search)

Library use (this is what purify_layer1b.py --ctrl_transform calls):
    x_t = apply_transform(x, 'blur', 1.1)
    rep = control_report(x_clean, x_poisoned, 'blur', 1.1)

CLI (seconds, no CL training): calibrate all three to a target PSNR on the clean task-T images of a noise pkl and print
the same diagnostics the purifier prints:
    python repro/defense/matched_controls.py --experiment split_cifar100 --noise_pkl <noise.pkl> \
        --target_psnr 25.23 --json ctrl_calib.json
"""
import argparse
import io
import json
import math
import os
import pickle as pkl
import sys

import numpy as np
import torch
import torch.nn.functional as F

KINDS = ('blur', 'jpeg', 'downup')


# ------------------------------------------------------------------------------------------------ transforms
def _gaussian_blur(x, sigma):
    if sigma <= 0:
        return x.clone()
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).to(x.dtype)
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), kh, groups=c)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), kv, groups=c)


def _jpeg(x, quality):
    from PIL import Image
    q = int(round(quality))
    arr = (x.detach().float().clamp(0, 1).cpu() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    out = np.empty_like(arr)
    for i, im in enumerate(arr):
        buf = io.BytesIO()
        Image.fromarray(im).save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out[i] = np.array(Image.open(buf).convert('RGB'))
    return torch.from_numpy(out).permute(0, 3, 1, 2).to(x.device, x.dtype) / 255


def _downup(x, size):
    s = int(round(size))
    h, w = x.shape[-2:]
    if s >= min(h, w):
        return x.clone()
    small = F.adaptive_avg_pool2d(x, (s, s))                       # area downsampling (anti-aliased)
    return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)


def apply_transform(x, kind, param, batch_size=1000):
    """x: float (N,3,H,W) in [0,1]. Returns the transformed images on x's device/dtype, clamped to [0,1]."""
    fn = {'blur': _gaussian_blur, 'jpeg': _jpeg, 'downup': _downup}[kind]
    out = torch.empty_like(x)
    for i in range(0, len(x), batch_size):
        out[i:i + batch_size] = fn(x[i:i + batch_size].float(), param).to(x.dtype).clamp(0, 1)
    return out


# ------------------------------------------------------------------------------------------------ metrics
def psnr(a, b):
    """Mean per-image PSNR in dB for images in [0,1] (same definition as purevqgan.purify._psnr)."""
    mse = F.mse_loss(a.float(), b.float(), reduction='none').flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


def calibrate(x_clean, kind, target_psnr):
    """Strength param whose mean PSNR(T(x), x) on x_clean is closest to target_psnr. Returns (param, achieved_psnr)."""
    x = x_clean.float()
    if kind == 'blur':                                            # PSNR decreases monotonically with sigma
        lo, hi = 0.05, 6.0
        if psnr(apply_transform(x, kind, lo), x) < target_psnr:
            return lo, psnr(apply_transform(x, kind, lo), x)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if psnr(apply_transform(x, kind, mid), x) > target_psnr:
                lo = mid
            else:
                hi = mid
        p = round(0.5 * (lo + hi), 4)
        return p, psnr(apply_transform(x, kind, p), x)
    grid = range(1, 96) if kind == 'jpeg' else range(2, min(x.shape[-2:]))
    best = min(((abs(psnr(apply_transform(x, kind, g), x) - target_psnr), g) for g in grid))
    return best[1], psnr(apply_transform(x, kind, best[1]), x)


@torch.no_grad()
def control_report(x_clean, x_poisoned, kind, param):
    """Same keys as purevqgan.purify.perturbation_report (minus code_flip), so the tables line up.

    survival_l2   = ||T(x+d) - T(x)||_2 / ||d||_2    (1 = perturbation passes through, 0 = erased)
    psnr_clean    = PSNR(T(x), x)                    (utility cost; this is what calibration matches)
    psnr_poisoned = PSNR(T(x+d), x)
    """
    x_clean, x_poisoned = x_clean.float().cpu(), x_poisoned.float().cpu()
    d = x_poisoned - x_clean
    rows = d.flatten(1).abs().amax(1) > 0
    rep = {'n_total': len(x_clean), 'n_poisoned': int(rows.sum()),
           'input_psnr_poisoned': psnr(x_poisoned[rows], x_clean[rows]) if rows.any() else None,
           'ctrl': kind, 'ctrl_param': float(param)}
    t_clean = apply_transform(x_clean, kind, param)
    rep['psnr_clean'] = psnr(t_clean, x_clean)
    if rows.any():
        tc = t_clean[rows]
        tp = apply_transform(x_poisoned[rows], kind, param)
        rep['survival_l2'] = ((tp - tc).flatten(1).norm(dim=1) / d[rows].flatten(1).norm(dim=1).clamp_min(1e-12)).mean().item()
        rep['residual_linf'] = (tp - tc).flatten(1).abs().amax(1).mean().item()
        rep['psnr_poisoned'] = psnr(tp, x_clean[rows])
    return rep


# ------------------------------------------------------------------------------------------------ CLI
def _load_task(experiment, noise_pkl, tasknum):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from purevqgan.data import load_brainwash_split
    ck = pkl.load(open(noise_pkl, 'rb'))
    t = ck['pretrained_ckpt']['task_num']
    x = load_brainwash_split(experiment, tasknum)[t]['train']['x']
    x = x[torch.as_tensor(ck['rnd_idx_train']).cpu()]              # same order as stage 4 builds it
    return x, torch.clamp(x + torch.as_tensor(ck['latest_noise']).cpu(), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=['split_cifar100', 'split_mini_imagenet', 'split_tiny_imagenet'])
    ap.add_argument('--noise_pkl', required=True)
    ap.add_argument('--target_psnr', type=float, required=True, help="the purifier's psnr_clean (e.g. 25.23 for A_c10_K512_L8)")
    ap.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    ap.add_argument('--params', nargs='+', type=float, default=None,
                    help='skip calibration and use these params (one per --kinds), e.g. to report on another noise file')
    ap.add_argument('--tasknum', type=int, default=10)
    ap.add_argument('--json')
    a = ap.parse_args()

    x, xp = _load_task(a.experiment, a.noise_pkl, a.tasknum)
    res = {'target_psnr': a.target_psnr, 'noise_pkl': a.noise_pkl, 'controls': {}}
    for i, kind in enumerate(a.kinds):
        if a.params:
            param = a.params[i]
        else:
            param, got = calibrate(x, kind, a.target_psnr)
            print(f'[calibrate] {kind}: param={param} -> clean PSNR {got:.2f} dB (target {a.target_psnr:.2f})')
        res['controls'][kind] = control_report(x, xp, kind, param)
        print(f'[report] {kind}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in res['controls'][kind].items()}))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
