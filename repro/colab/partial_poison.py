"""Partial-poisoning wrapper around main_baselines.py (stage 4). No existing file is modified.

Poisons only a random fraction of the last task's training samples, reusing the full-poison
noise from a stage-3 checkpoint -- no stage-3 re-run. It writes a derived checkpoint pkl and
points --checkpoint at it, then hands off to the unmodified main_baselines.main():
  * ours    (--addnoise)           : latest_noise is zeroed on the un-poisoned samples.
  * uniform (--addnoise --uniform) : sets inj_data_idx, which main_baselines already supports
                                     ("uniform noise only on the injected data").
  * clean   (no --addnoise)        : checkpoint untouched (noise is not used).

Usage (from the repo root; every normal main_baselines.py flag is accepted unchanged):
  python repro/partial_poison.py --poison_frac 0.05 --poison_seed 0 \
      --experiment split_cifar100 --approach ewc ... --checkpoint <noise.pkl> --addnoise
"""
import argparse
import os
import pickle as pkl
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # so main_baselines / approaches / utils import as when run from the repo root

import torch


def build_partial_checkpoint(src, dst, frac, seed, uniform):
    with open(src, 'rb') as f:
        ck = pkl.load(f)
    n = len(ck['rnd_idx_train'])
    k = int(round(frac * n))
    # indices are in the permuted (rnd_idx_train) space -- the space main_baselines applies noise in
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k]
    if uniform:
        ck['inj_data_idx'] = idx
    else:
        mask = torch.zeros(n, *([1] * (ck['latest_noise'].dim() - 1)), dtype=ck['latest_noise'].dtype)
        mask[idx] = 1
        ck['latest_noise'] = ck['latest_noise'] * mask
        if 'noise' in ck and torch.is_tensor(ck['noise']):
            ck['noise'] = ck['noise'] * mask
    ck['poison_frac'], ck['poison_seed'], ck['poison_n'] = frac, seed, k
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        pkl.dump(ck, f)
    print(f'[partial_poison] poisoning {k}/{n} samples ({frac:.1%}), seed={seed}, '
          f'{"uniform" if uniform else "learned"} noise -> {dst}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--poison_frac', type=float, required=True, help='fraction of last-task samples to poison, e.g. 0.02 / 0.05')
    p.add_argument('--poison_seed', type=int, default=0, help='which random subset gets poisoned')
    p.add_argument('--poison_dir', type=str, default=os.path.join(ROOT, 'repro/sweep/noise_partial'),
                   help='where the derived (masked) checkpoint is written')
    own, rest = p.parse_known_args()
    assert 0.0 < own.poison_frac <= 1.0, '--poison_frac must be in (0, 1]'

    from approaches.arguments import get_args   # imported here so build_partial_checkpoint() can be
    import main_baselines                       # used on its own (e.g. from a Colab notebook)

    sys.argv = [sys.argv[0]] + rest          # the original parser never sees our flags
    args = get_args()
    if args.checkpoint is not None and args.addnoise:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        kind = 'uniform' if args.uniform else 'ours'
        dst = os.path.join(own.poison_dir, f'{stem}_{kind}_p{own.poison_frac}_ps{own.poison_seed}.pkl')
        build_partial_checkpoint(args.checkpoint, dst, own.poison_frac, own.poison_seed, args.uniform)
        args.checkpoint = dst                # acc_mat_*.npy name now carries the fraction/seed too
    main_baselines.main(args)
