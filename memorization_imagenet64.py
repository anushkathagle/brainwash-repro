"""
memorization_imagenet64.py

Instance-memorization check for the ImageNet-64 purifier on its OWN data split:
    SEEN   = ImageNet-64 train images (the model was trained on these)
    UNSEEN = ImageNet-64 val images   (held out from training)
Same design as the CIFAR-100 check (fine-tune train vs test), but here seen/unseen
is certain by construction -- no need to know where Tiny-ImageNet's images came from.

Two measurements, per image, at each sigma_star (same seed => same diffusion noise):
  1. RECONSTRUCTION  -- purify, compare to the original (MSE, PSNR). This is the
                        quantity the defense depends on.
  2. DENOISING LOSS  -- the model's training loss ||eps_hat - eps||^2 at the same
                        timesteps. The most sensitive memorization signal (the basis
                        of membership-inference attacks on diffusion models): seen
                        images get lower loss if the model memorized them.
Reported: seen mean, unseen mean, gap (seen - unseen) with a bootstrap 95% CI, and
whether the gap lies inside a pre-set equivalence margin.

How it transfers to Tiny-ImageNet: Tiny's train images are ImageNet train photos
(re-resized), so if the model shows no instance memorization even on its exact
training images, it cannot be recalling Tiny's versions of them either.

Usage (GPU):
    python memorization_imagenet64.py --purify_ckpt repro/artifacts/diffusion/imagenet64_uncond_100M_1500K.pt \
        --seen_parquet  /storage/work/<id>/imagenet64_hf/data/train-00000-of-00022.parquet \
        --unseen_parquet /storage/work/<id>/imagenet64_hf/data/val-00000-of-00001.parquet \
        --n 1000 --sigmas 0.25 0.5 1.0 --out repro/artifacts/tinyImagenet_files/memorization/imagenet64
"""

import argparse
import io
import os
import time

import numpy as np
import torch

from diffpure_imagenet64 import load_imagenet64_net, purify_in_chunks, sigma_to_t


def load_parquet_images(path, n, seed):
    """Random sample of n images (as [n,3,64,64] float in [0,1]) from an HF parquet file."""
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path, columns=['image'])
    idx = np.sort(np.random.default_rng(seed).choice(t.num_rows, size=min(n, t.num_rows), replace=False))
    col = t.column('image').take(idx).to_pylist()
    xs = []
    for im in col:
        raw = im['bytes'] if isinstance(im, dict) else im
        a = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB').resize((64, 64)), dtype=np.float32) / 255.
        xs.append(a.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(xs)), idx


def boot_ci(a, b, n_boot=2000, seed=0):
    """95% CI of mean(a) - mean(b) (independent samples)."""
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


@torch.no_grad()
def denoise_loss(net, x01, steps, chunk, device, seed):
    """Per-image epsilon-prediction MSE at timestep steps-1, fixed noise per image index."""
    model, diff = net['model'], net['diffusion']
    out = []
    g = torch.Generator().manual_seed(seed)
    noise_all = torch.randn(x01.shape, generator=g)
    for lo in range(0, len(x01), chunk):
        x = (x01[lo:lo + chunk] * 2 - 1).to(device)
        eps = noise_all[lo:lo + chunk].to(device)
        t = torch.full((len(x),), steps - 1, device=device, dtype=torch.long)
        xt = diff.q_sample(x, t, noise=eps)
        if hasattr(diff, '_wrap_model'):      # respaced: map t to the original timestep + rescale
            pred = diff._wrap_model(model)(xt, t)
        else:
            pred = model(xt, diff._scale_timesteps(t))
        pred = pred[:, :3] if pred.shape[1] == 6 else pred   # learn_sigma: first 3 channels = eps
        out.append((pred - eps).pow(2).flatten(1).mean(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--purify_ckpt', required=True)
    ap.add_argument('--seen_parquet', required=True, help='ImageNet-64 TRAIN parquet shard')
    ap.add_argument('--unseen_parquet', required=True, help='ImageNet-64 VAL parquet')
    ap.add_argument('--n', type=int, default=1000, help='images per side')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    ap.add_argument('--respace', default='')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--margin_db', type=float, default=0.5,
                    help='equivalence margin for the PSNR gap, fixed BEFORE looking at results')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='repro/artifacts/tinyImagenet_files/memorization/imagenet64')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    xs, _ = load_parquet_images(args.seen_parquet, args.n, args.seed)
    xu, _ = load_parquet_images(args.unseen_parquet, args.n, args.seed + 1)
    print(f'[mem] seen (train) {tuple(xs.shape)}  unseen (val) {tuple(xu.shape)}', flush=True)
    net = load_imagenet64_net(args.purify_ckpt, device=args.device, use_fp16=args.fp16, respace=args.respace)

    rows = []
    for sigma in args.sigmas:
        steps, resolved = sigma_to_t(net, sigma)
        t0 = time.time()
        res = {}
        for name, x in (('seen', xs), ('unseen', xu)):
            p = purify_in_chunks(net, x, sigma, chunk_size=args.chunk, device=args.device,
                                 seed=args.seed, verbose=False)
            mse = (p - x).pow(2).flatten(1).mean(1).numpy()
            res[name] = dict(psnr=10 * np.log10(1 / np.maximum(mse, 1e-12)), mse=mse,
                             loss=denoise_loss(net, x, steps, args.chunk, args.device, args.seed))
        r = dict(sigma_star=sigma, resolved_sigma=round(resolved, 4), steps=steps)
        for m in ('psnr', 'mse', 'loss'):
            a, b = res['seen'][m], res['unseen'][m]
            lo, hi = boot_ci(a, b, seed=args.seed)
            r.update({f'{m}_seen': a.mean(), f'{m}_unseen': b.mean(), f'{m}_gap': a.mean() - b.mean(),
                      f'{m}_gap_lo': lo, f'{m}_gap_hi': hi})
        r['psnr_equivalent'] = int(-args.margin_db < r['psnr_gap_lo'] and r['psnr_gap_hi'] < args.margin_db)
        r['minutes'] = (time.time() - t0) / 60
        rows.append(r)
        print(f"[mem] sigma*={sigma} ({steps} steps): PSNR seen {r['psnr_seen']:.2f} unseen {r['psnr_unseen']:.2f} "
              f"gap {r['psnr_gap']:+.3f} dB [{r['psnr_gap_lo']:+.3f}, {r['psnr_gap_hi']:+.3f}] "
              f"{'within' if r['psnr_equivalent'] else 'NOT within'} +/-{args.margin_db} dB | "
              f"denoise loss seen {r['loss_seen']:.4f} unseen {r['loss_unseen']:.4f} "
              f"gap {r['loss_gap']:+.4f} [{r['loss_gap_lo']:+.4f}, {r['loss_gap_hi']:+.4f}] "
              f"| {r['minutes']:.1f} min", flush=True)
        cols = list(rows[0].keys())
        with open(args.out + '.tsv', 'w') as fh:
            fh.write(f'# seen={os.path.basename(args.seen_parquet)} unseen={os.path.basename(args.unseen_parquet)} '
                     f'n={args.n} seed={args.seed} margin_db={args.margin_db}\n' + '\t'.join(cols) + '\n')
            for q in rows:
                fh.write('\t'.join(f'{q[c]:.6g}' if isinstance(q[c], float) else str(q[c]) for c in cols) + '\n')
    print(f'[mem] wrote {args.out}.tsv')
    print('[mem] read: memorization => seen PSNR clearly HIGHER and seen denoise loss clearly LOWER '
          'than unseen (CI excludes 0 on that side), growing with sigma*.')


if __name__ == '__main__':
    main()
