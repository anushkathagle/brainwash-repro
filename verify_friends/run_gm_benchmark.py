"""
run_gm_benchmark.py

Reproduce FRIENDS (Liu, Yang & Mirzasoleiman, NeurIPS 2022) Table 3, row "Gradient Matching (xi=8),
from scratch" with the OFFICIAL code (https://github.com/tianyu139/friendly-noise), before trusting
our own port in layer2_friends/friends.py.

Poisons: Geiping's benchmark_v1_cml/bm_base_ResNet18 (folders 0..99, each with poisons.pickle,
base_indices.pickle, target.pickle). Expected from the paper (Table 3, 80 epochs, ResNet-18, CIFAR-10):

    undefended   poison acc ~50%   test acc ~93.5%
    FRIENDS-B    poison acc  ~0%   test acc ~91.5%

Two passes, so the expensive defended arm only runs where the attack worked:
    1. undefended on every index in --indices
    2. FRIENDS-B on the indices where the undefended run flipped the target (--all_defended: on all)

Usage (from anywhere):
    python verify_friends/run_gm_benchmark.py --friends_repo /content/friendly-noise \
        --bm_dir /content/bm_base_ResNet18 --indices 0 1 2 3 4 5 6 7 8 9 --out /content/drive/MyDrive/fn_gm
    python verify_friends/run_gm_benchmark.py ... --dry_run      # print the commands only
    python verify_friends/run_gm_benchmark.py ... --check        # poison count / Linf check first

Finished runs (a result.csv exists) are skipped, so a Colab disconnect costs at most one run.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

FRIENDS_FLAGS = ['--noise_type', 'friendly', 'bernoulli', '--friendly_begin_epoch', '5',
                 '--noise_eps', '16', '--friendly_clamp', '16']


def run_dir(out, arm, idx, seed):
    return os.path.join(out, arm, f'idx{idx}_s{seed}')


def read_result(path):
    """Last data row of the official result.csv (it is appended to, with a header every time)."""
    rows = [r for r in csv.DictReader(open(path)) if r['epoch'] != 'epoch']
    return rows[-1] if rows else None


def flipped(row):
    return str(row['poison_acc']).lower() == 'true'


def command(args, arm, idx):
    cmd = [args.python, 'main.py', '--poisons_path', os.path.join(args.bm_dir, str(idx)),
           '--dataset_index', str(idx), '--epochs', str(args.epochs), '--seed', str(args.seed),
           '--out', run_dir(args.out, arm, idx, args.seed), '--no-progress']
    if arm == 'friends':
        cmd += FRIENDS_FLAGS
    return cmd + args.extra


def check_poisons(args):
    import pickle
    import numpy as np
    import torchvision
    ds = torchvision.datasets.CIFAR10(os.path.join(args.friends_repo, 'data'), train=True, download=True)
    for idx in args.indices:
        d = os.path.join(args.bm_dir, str(idx))
        poisons = pickle.load(open(os.path.join(d, 'poisons.pickle'), 'rb'))
        base = pickle.load(open(os.path.join(d, 'base_indices.pickle'), 'rb'))
        target = pickle.load(open(os.path.join(d, 'target.pickle'), 'rb'))
        clean = np.stack([np.asarray(ds[int(i)][0]) for i in base]).astype(int)
        pois = np.stack([np.asarray(p[0]) for p in poisons]).astype(int)
        print(f'[check] idx {idx}: {len(poisons)} poisons, {len(base)} indices, '
              f'Linf {np.abs(pois - clean).max()}/255, poison label {poisons[0][1]}, target class {target[1]}')


def launch(args, arm, idx):
    rd = run_dir(args.out, arm, idx, args.seed)
    csv_path = None
    if os.path.isdir(rd):
        for root, _, files in os.walk(rd):
            if 'result.csv' in files:
                csv_path = os.path.join(root, 'result.csv')
    if csv_path:
        print(f'[skip] {arm} idx {idx}: done')
        return read_result(csv_path)
    cmd = command(args, arm, idx)
    print(f'[run ] {arm} idx {idx}: {" ".join(cmd)}', flush=True)
    if args.dry_run:
        return None
    os.makedirs(rd, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(rd, 'stdout.log'), 'w') as log:
        rc = subprocess.run(cmd, cwd=args.friends_repo, stdout=log, stderr=subprocess.STDOUT).returncode
    for root, _, files in os.walk(rd):
        if 'result.csv' in files:
            row = read_result(os.path.join(root, 'result.csv'))
            print(f'[done] {arm} idx {idx} ({time.time() - t0:.0f}s): poison_acc={row["poison_acc"]} '
                  f'test_acc={float(row["acc"]):.2f}', flush=True)
            return row
    sys.exit(f'[FAIL] {arm} idx {idx} (exit {rc}); see {rd}/stdout.log')


def summarize(results):
    print('\n arm         n   target flipped   mean test acc')
    for arm, rows in results.items():
        rows = [r for r in rows.values() if r]
        if not rows:
            continue
        k = sum(flipped(r) for r in rows)
        acc = sum(float(r['acc']) for r in rows) / len(rows)
        print(f' {arm:<10} {len(rows):>3}   {k:>3}/{len(rows):<3} ({100 * k / len(rows):4.0f}%)   {acc:6.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--friends_repo', required=True, help='clone of tianyu139/friendly-noise (main.py is run from here)')
    p.add_argument('--bm_dir', required=True, help='.../bm_base_ResNet18')
    p.add_argument('--out', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=80, help='80 = paper Table 3; 40 is faster but a deviation')
    p.add_argument('--all_defended', action='store_true', help='also run FRIENDS where the undefended run did not flip the target')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--check', action='store_true', help='verify poison count and Linf of the chosen folders, then continue')
    p.add_argument('extra', nargs='*', help='extra flags passed to main.py (after --)')
    args = p.parse_args()
    args.friends_repo = os.path.abspath(args.friends_repo)
    args.bm_dir = os.path.abspath(args.bm_dir)
    args.out = os.path.abspath(args.out)

    if args.check:
        check_poisons(args)

    results = {'undefended': {}, 'friends': {}}
    for idx in args.indices:
        results['undefended'][idx] = launch(args, 'undefended', idx)
    for idx in args.indices:
        und = results['undefended'][idx]
        if und is not None and not flipped(und) and not args.all_defended:
            print(f'[skip] friends idx {idx}: undefended run did not flip the target')
            continue
        results['friends'][idx] = launch(args, 'friends', idx)
    summarize(results)


if __name__ == '__main__':
    main()
