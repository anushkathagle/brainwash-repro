"""
sigma_sweep_tiny.py

Cheap, training-free checks for the Tiny-ImageNet (64x64) purifier, run BEFORE
any purify_tiny_layer1a.py training run. Nothing here trains a ResNet; it only
runs the purifier on a sample of one task's images and measures:

  1. DISTORTION   -- how far purification moves CLEAN images:
                     mean L2 and PSNR of P(x) vs x.  (Proxy for clean cost;
                     Arm D measures the real cost afterwards.)
  2. SURVIVAL     -- how much of a perturbation delta survives purification,
                     using the SAME diffusion noise for x and x+delta:
                        survival  = ||P(x+d) - P(x)||_2 / ||d||_2
                        aligned   = <P(x+d) - P(x), d> / ||d||_2^2
                     (1.0 = untouched, 0 = removed). With --noise_pkl this is
                     the real BrainWash poison from stage 3; without it, a
                     random-sign +/-eps probe (attack-free; labelled as such).
  3. COST         -- seconds per image at each sigma_star.
  4. MEMORIZATION -- the ImageNet-64 model was trained on ImageNet train, and
                     Tiny-ImageNet's train images come from ImageNet train, so
                     the purifier has likely SEEN them. Distortion is reported
                     separately for the task's train images ("seen") and its
                     val images ("unseen"). If seen images are reconstructed
                     clearly better than unseen ones, especially at large
                     sigma_star, the purifier is partly recalling them, which
                     inflates both clean accuracy and poison removal.
                     CAVEAT: this only separates seen/unseen if Tiny-ImageNet's
                     val images are NOT in ImageNet's train set -- confirm that
                     before interpreting the gap (see README_purify_tiny.md).

Reading the result (the decision rule used for miniImageNet, applied here):
  * If survival falls steeply with sigma_star (as on CIFAR-100) instead of
    staying flat (as on miniImageNet via the 256 model), native resolution is
    doing its job.
  * Choose sigma_star from protection (low survival), distortion and cost.
    Default for comparability is 0.5 (CIFAR-100); keep it unless the sweep
    shows a clear reason not to. Never pick it from attacked-arm accuracy.

Usage (repo root; GPU strongly recommended):
    python sigma_sweep_tiny.py \
        --purify_ckpt /path/to/imagenet64_uncond_100M_1500K.pt \
        --tiny_npz ./data/tiny_imagenet.npz --tasknum 20 --task 19 \
        --sigmas 0 0.1 0.25 0.5 1.0 --n_images 256 \
        [--noise_pkl repro/sweep/noise/<tag>/<file>.pkl] \
        --out repro/artifacts/tinyImagenet_files/sigma_sweep/sweep.tsv
"""

import argparse
import os
import pickle as pkl
import sys
import time

import numpy as np
import torch

from diffpure_imagenet64 import load_imagenet64_net, purify_in_chunks, sigma_to_t


def _psnr(a, b):
    mse = (a - b).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse))  # per image, data range 1


def _purify(net, x, sigma, args):
    # Same seed for every call at a given sigma -> identical diffusion noise for
    # x and x+delta (same shape, same chunk order), so their difference isolates
    # what happens to delta.
    return purify_in_chunks(net, x, sigma, chunk_size=args.chunk, device=args.device,
                            seed=args.seed, verbose=False)


def load_task(args):
    from approaches.data_utils import generate_split_tiny_imagenet_tasks
    root_add = os.path.join(os.path.expanduser('~'), 'data', 'tiny-imagenet-200')
    data, _, _, _ = generate_split_tiny_imagenet_tasks(
        task_num=args.tasknum, rnd_order=False, save_data=False,
        dataset_file=args.tiny_npz, order=np.arange(200), root_add=root_add)
    return data[args.task]['train']['x'].float(), data[args.task]['test']['x'].float()


def build_delta(xtrain, args):
    """Return (x_seen_clean, delta, source) for the survival measurement."""
    g = torch.Generator().manual_seed(args.seed)
    if args.noise_pkl:
        ck = pkl.load(open(args.noise_pkl, 'rb'))
        perm = ck['rnd_idx_train']
        xp = xtrain[perm]
        noise = ck['latest_noise'].detach().float().cpu()
        if noise.shape != xp.shape:
            sys.exit(f'noise shape {tuple(noise.shape)} != task train shape {tuple(xp.shape)} '
                     f'-- wrong --task/--tasknum for this pkl?')
        delta = (xp + noise).clamp(0, 1) - xp          # effective (post-clamp) poison
        idx = torch.nonzero(delta.flatten(1).abs().sum(1) > 0).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:args.n_images]]
        src = f'brainwash poison ({os.path.basename(args.noise_pkl)}, delta={ck.get("delta", "?")})'
        return xp[idx], delta[idx], src
    idx = torch.randperm(len(xtrain), generator=g)[:args.n_images]
    x = xtrain[idx]
    sign = torch.randint(0, 2, x.shape, generator=g).float() * 2 - 1
    delta = (x + args.probe_eps * sign).clamp(0, 1) - x
    return x, delta, f'random-sign probe (eps={args.probe_eps:.4f}; NOT the real poison)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--purify_ckpt', required=True)
    ap.add_argument('--tiny_npz', default='./data/tiny_imagenet.npz')
    ap.add_argument('--tasknum', type=int, default=20)
    ap.add_argument('--task', type=int, default=None, help='task index (default: last = the poisoned task)')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument('--n_images', type=int, default=256)
    ap.add_argument('--noise_pkl', default=None, help='stage-3 BrainWash noise pkl for real-poison survival')
    ap.add_argument('--probe_eps', type=float, default=8 / 255,
                    help='L-inf size of the random-sign probe when no --noise_pkl is given')
    ap.add_argument('--respace', default='', help="'' = full 4000 steps; e.g. 1000")
    ap.add_argument('--backend', default='auto', choices=['auto', 'improved', 'guided'])
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='repro/artifacts/tinyImagenet_files/sigma_sweep/sweep.tsv')
    ap.add_argument('--save_samples', type=int, default=8,
                    help='save the first N images (clean/purified per sigma) to a .pt next to --out')
    args = ap.parse_args()
    if args.task is None:
        args.task = args.tasknum - 1

    torch.manual_seed(args.seed)
    print(f'[sweep] loading Tiny-ImageNet task {args.task}/{args.tasknum} from {args.tiny_npz}')
    xtrain, xval = load_task(args)
    assert xtrain.shape[-1] == 64, f'expected 64x64 Tiny-ImageNet, got {tuple(xtrain.shape)}'

    x_seen, delta, delta_src = build_delta(xtrain, args)
    g = torch.Generator().manual_seed(args.seed + 1)
    x_unseen = xval[torch.randperm(len(xval), generator=g)[:args.n_images]]
    print(f'[sweep] {len(x_seen)} seen (train) + {len(x_unseen)} unseen (val) images; '
          f'perturbation: {delta_src}; ||d||_inf={delta.abs().max():.4f}')

    net = load_imagenet64_net(args.purify_ckpt, device=args.device, use_fp16=args.fp16,
                              respace=args.respace, backend=args.backend)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, samples = [], {'clean': x_seen[:args.save_samples], 'poisoned': (x_seen + delta)[:args.save_samples]}
    d_norm2 = delta.flatten(1).pow(2).sum(1).clamp_min(1e-12)

    for sigma in args.sigmas:
        steps, resolved = sigma_to_t(net, sigma)
        t0 = time.time()
        p_seen = _purify(net, x_seen, sigma, args)
        sec_per_img = (time.time() - t0) / len(x_seen)
        p_pois = _purify(net, (x_seen + delta).clamp(0, 1), sigma, args)
        p_unseen = _purify(net, x_unseen, sigma, args)

        resid = (p_pois - p_seen).flatten(1)
        survival = (resid.pow(2).sum(1) / d_norm2).sqrt()
        aligned = (resid * delta.flatten(1)).sum(1) / d_norm2

        l2_seen = (p_seen - x_seen).flatten(1).norm(dim=1)
        l2_unseen = (p_unseen - x_unseen).flatten(1).norm(dim=1)
        psnr_seen, psnr_unseen = _psnr(p_seen, x_seen), _psnr(p_unseen, x_unseen)

        row = dict(sigma_star=sigma, resolved_sigma=round(resolved, 4), steps=steps,
                   T=net['diffusion'].num_timesteps,
                   survival=survival.mean().item(), aligned=aligned.mean().item(),
                   l2_seen=l2_seen.mean().item(), l2_unseen=l2_unseen.mean().item(),
                   psnr_seen=psnr_seen.mean().item(), psnr_unseen=psnr_unseen.mean().item(),
                   psnr_gap_seen_minus_unseen=(psnr_seen.mean() - psnr_unseen.mean()).item(),
                   sec_per_img=sec_per_img)
        rows.append(row)
        samples[f'sigma{sigma}'] = p_seen[:args.save_samples]
        samples[f'sigma{sigma}_poisoned'] = p_pois[:args.save_samples]
        print(f'[sweep] sigma*={sigma:<5} (resolved {resolved:.3f}, {steps:4d}/{row["T"]} steps) '
              f'survival={row["survival"]:.3f} aligned={row["aligned"]:.3f} | '
              f'PSNR seen={row["psnr_seen"]:.2f} unseen={row["psnr_unseen"]:.2f} '
              f'(gap {row["psnr_gap_seen_minus_unseen"]:+.2f} dB) | {sec_per_img*1e3:.1f} ms/img', flush=True)

        # Rewrite the TSV after EVERY sigma, so a cancelled / timed-out job keeps
        # the rows it already finished (large sigma_star is the slow part).
        cols = list(rows[0].keys())
        with open(args.out, 'w') as fh:
            fh.write(f'# task={args.task}/{args.tasknum} n={len(x_seen)} perturbation={delta_src} '
                     f'respace={args.respace or "none"} backend={net["backend"]} seed={args.seed}\n')
            fh.write('\t'.join(cols) + '\n')
            for r in rows:
                fh.write('\t'.join(f'{r[c]:.6g}' if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
        print(f'[sweep] wrote {len(rows)} row(s) -> {args.out}', flush=True)
    if args.save_samples > 0:
        sp = os.path.splitext(args.out)[0] + '_samples.pt'
        torch.save(samples, sp)
        print(f'[sweep] wrote sample images -> {sp}')

    # A loud sanity check: sigma_star=0 must be an identity at native 64x64.
    for r in rows:
        if r['sigma_star'] == 0 and (abs(r['survival'] - 1) > 1e-4 or r['l2_seen'] > 1e-4):
            print('[sweep] WARNING: sigma_star=0 was not an identity -- something resized the input.')


if __name__ == '__main__':
    main()
