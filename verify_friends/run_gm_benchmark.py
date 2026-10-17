"""
run_gm_benchmark.py

Reproduce FRIENDS (Liu, Yang & Mirzasoleiman, NeurIPS 2022) Table 3, row "Gradient Matching (xi=8),
from scratch" with the OFFICIAL code (https://github.com/tianyu139/friendly-noise), and then with OUR
PORT (layer2_friends/friends.py) swapped into the same training loop.

Poisons: Geiping's benchmark_v1_cml/bm_base_ResNet18 (folders 0..99, each with poisons.pickle,
base_indices.pickle, target.pickle). Expected from the paper (Table 3, 80 epochs, ResNet-18, CIFAR-10):

    undefended   poison acc ~50%   test acc ~93.5%
    FRIENDS-B    poison acc  ~0%   test acc ~91.5%

Arms (a --arms list; undefended always runs first):
    undefended   official main.py, no defense
    friends      official main.py, official friendly noise + Bernoulli noise   (FRIENDS-B)
    port         main_port.py = official main.py with ONLY two things replaced by the port in
                 friends.py: generate_friendly_noise and the Bernoulli random noise (RandomNoise).
                 main_port.py is generated from the official main.py by this script, which fails
                 if the official file does not look as expected.

The defended arms only run on indices where the undefended run flipped the target (--all_defended: all).

Usage:
    python verify_friends/run_gm_benchmark.py --friends_repo /content/content/friendly-noise \
        --bm_dir <bm_base_ResNet18> --out <results dir> --indices 1 2 --arms friends port \
        --port_dir /content/brainwash-repro/layer2_friends
    ... --dry_run      print the commands only (also writes main_port.py)
    ... --check        poison count / Linf check first
    ... -- --friendly_epochs 2 --friendly_begin_epoch 1 --epochs 3    smoke test (extra flags go to main.py)

The port arm needs --port_dir (the folder that contains friends.py). By default the port's random noise
is NOT clamped to [0,1], like the official BernoulliNoise (its clamp line comes after a return and never
runs); --port_clamp01 turns the port's clamp on.

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

OFFICIAL_IMPORT = 'from friendly_noise import generate_friendly_noise, UniformNoise, GaussianNoise, BernoulliNoise\n'

# Inserted into main_port.py right after the official import above. It shadows the two names the
# official main() uses (generate_friendly_noise, BernoulliNoise) with versions that call the port.
SHIM = '''
# >>> PORT SHIM (written by verify_friends/run_gm_benchmark.py; everything else in this file is the official main.py)
import os as _os
import sys as _sys
_sys.path.insert(0, _os.environ['FRIENDS_PORT_DIR'])
import friends as _port                       # the port: layer2_friends/friends.py
_PORT_CLAMP01 = _os.environ.get('FRIENDS_PORT_CLAMP01', '0') == '1'


class _Heads(torch.nn.Module):
    """Official model: normalized input, one logits tensor. Port: raw [0,1] input, list of heads (task 0)."""
    def __init__(self, net, normalize):
        super().__init__()
        self.net, self.normalize = net, normalize

    def forward(self, x):
        return [self.net(self.normalize(x))]


def generate_friendly_noise(model, trainloader, device, friendly_epochs, mu, friendly_lr,
                            friendly_momentum=0.9, nesterov=True, friendly_steps=None,
                            clamp_min=-32 / 255, clamp_max=32 / 255, return_preds=False, loss_fn='KL'):
    assert not return_preds and clamp_min == -clamp_max, 'port shim: symmetric clamp, no preds'
    ds = trainloader.dataset
    normalize, ds.transform = ds.transform, None          # un-normalized [0,1] images, as the official code does
    xs = torch.zeros(len(ds), 3, 32, 32)
    for inputs, _, _, idx in trainloader:
        xs[idx] = inputs
    ds.transform = normalize
    return _port.generate_friendly_noise(
        _Heads(model, normalize), xs, task=0, target='head', loss_fn=loss_fn,
        friendly_epochs=friendly_epochs, lam=mu, friendly_lr=friendly_lr,
        friendly_momentum=friendly_momentum, nesterov=nesterov, clamp=clamp_max,
        batch_size=trainloader.batch_size, device=device)


class BernoulliNoise(object):
    """The port's RandomNoise as a per-sample transform. Each DataLoader worker (and epoch) seeds its own
    generator from torch.initial_seed(), so the workers do not repeat each other's noise."""
    def __init__(self, eps):
        self.eps, self._rn = eps, None

    def __call__(self, tensor):
        if self._rn is None:
            self._rn = _port.RandomNoise(self.eps, 'bernoulli', seed=torch.initial_seed() % 2 ** 31,
                                         clamp01=_PORT_CLAMP01)
        return self._rn(tensor)
# <<< PORT SHIM
'''


def write_main_port(friends_repo):
    src = open(os.path.join(friends_repo, 'main.py')).read()
    if src.count(OFFICIAL_IMPORT) != 1:
        sys.exit('main.py does not contain the expected friendly_noise import line exactly once; '
                 'was the official repo modified?')
    dst = os.path.join(friends_repo, 'main_port.py')
    with open(dst, 'w') as f:
        f.write(src.replace(OFFICIAL_IMPORT, OFFICIAL_IMPORT + SHIM))
    return dst


def run_dir(out, arm, idx, seed):
    return os.path.join(out, arm, f'idx{idx}_s{seed}')


def read_result(path):
    """Last data row of the official result.csv (it is appended to, with a header every time)."""
    rows = [r for r in csv.DictReader(open(path)) if r['epoch'] != 'epoch']
    return rows[-1] if rows else None


def flipped(row):
    return str(row['poison_acc']).lower() == 'true'


def command(args, arm, idx):
    cmd = [args.python, 'main_port.py' if arm == 'port' else 'main.py',
           '--poisons_path', os.path.join(args.bm_dir, str(idx)),
           '--dataset_index', str(idx), '--epochs', str(args.epochs), '--seed', str(args.seed),
           '--out', run_dir(args.out, arm, idx, args.seed), '--no-progress']
    if arm in ('friends', 'port'):
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


def find_result(rd):
    for root, _, files in os.walk(rd):
        if 'result.csv' in files:
            return os.path.join(root, 'result.csv')
    return None


def launch(args, arm, idx):
    rd = run_dir(args.out, arm, idx, args.seed)
    csv_path = find_result(rd) if os.path.isdir(rd) else None
    if csv_path:
        print(f'[skip] {arm} idx {idx}: done')
        return read_result(csv_path)
    cmd = command(args, arm, idx)
    print(f'[run ] {arm} idx {idx}: {" ".join(cmd)}', flush=True)
    if args.dry_run:
        return None
    os.makedirs(rd, exist_ok=True)
    env = dict(os.environ, FRIENDS_PORT_CLAMP01='1' if args.port_clamp01 else '0')
    if args.port_dir:
        env['FRIENDS_PORT_DIR'] = args.port_dir
    t0 = time.time()
    with open(os.path.join(rd, 'stdout.log'), 'w') as log:
        rc = subprocess.run(cmd, cwd=args.friends_repo, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    csv_path = find_result(rd)
    if csv_path:
        row = read_result(csv_path)
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
    print('(defended arms ran only on indices where the undefended run flipped the target, unless --all_defended)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--friends_repo', required=True, help='clone of tianyu139/friendly-noise (main.py is run from here)')
    p.add_argument('--bm_dir', required=True, help='.../bm_base_ResNet18')
    p.add_argument('--out', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--arms', nargs='+', choices=['friends', 'port'], default=['friends'],
                   help='defended arms to run after the undefended one')
    p.add_argument('--port_dir', default=None, help='folder with friends.py (needed for --arms port)')
    p.add_argument('--port_clamp01', action='store_true', help="clamp the port's random noise to [0,1] (official: no clamp)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=80, help='80 = paper Table 3; 40 is faster but a deviation')
    p.add_argument('--all_defended', action='store_true', help='also run defended arms where the undefended run did not flip the target')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--check', action='store_true', help='verify poison count and Linf of the chosen folders, then continue')
    p.add_argument('extra', nargs='*', help='extra flags passed to main.py (after --)')
    args = p.parse_args()
    args.friends_repo = os.path.abspath(args.friends_repo)
    args.bm_dir = os.path.abspath(args.bm_dir)
    args.out = os.path.abspath(args.out)
    if 'port' in args.arms:
        if not args.port_dir or not os.path.isfile(os.path.join(args.port_dir, 'friends.py')):
            sys.exit('--arms port needs --port_dir <folder containing friends.py>')
        args.port_dir = os.path.abspath(args.port_dir)
        print(f'[port] wrote {write_main_port(args.friends_repo)}  (random noise clamp01={args.port_clamp01})')

    if args.check:
        check_poisons(args)

    results = {'undefended': {}}
    for idx in args.indices:
        results['undefended'][idx] = launch(args, 'undefended', idx)
    for arm in args.arms:
        results[arm] = {}
        for idx in args.indices:
            und = results['undefended'][idx]
            if und is not None and not flipped(und) and not args.all_defended:
                print(f'[skip] {arm} idx {idx}: undefended run did not flip the target')
                continue
            results[arm][idx] = launch(args, arm, idx)
    summarize(results)


if __name__ == '__main__':
    main()
