"""Step 1 go/no-go: can the pretrained PureGen EBM "see" BrainWash noise, and how much does Langevin purification remove?

No continual-learning training happens here; it takes minutes on a GPU.

  python -m puregen_ebm.energy_check --puregen_repo /content/PureGen_PoisonDefense --experiment split_cifar100 \
      --noise_pkl <dir-or-file of cifar100_ewc_cautious_eps0.1> [<another pkl> ...] --steps 50 100 150 300

Per noise pkl it reports:
  * EBM energy of clean task-T images vs the same images + BrainWash noise vs + uniform noise of the same eps (control)
  * AUROC(poisoned energy > clean energy) plus the same for uniform noise as a control (both ~1.0 at these eps, so
    detection is NOT the go/no-go; the verdict uses survival and clean PSNR at the largest step count)
  * per Langevin step count: psnr_clean (utility cost), psnr_poisoned, survival_l2 (fraction of the perturbation left)
and writes <out>/<tag>.json plus an energy histogram <out>/<tag>_energy.png.
"""
import argparse
import glob
import json
import os
import pickle as pkl

import torch

from .ebm import DEFAULT_EBM, auroc, energies, load_ebm, purify_with_report


def load_task(experiment, noise_pkl, tasknum=10):
    from purevqgan.data import load_brainwash_split
    ck = pkl.load(open(noise_pkl, 'rb'))
    t = ck['pretrained_ckpt']['task_num']
    x = load_brainwash_split(experiment, tasknum)[t]['train']['x']
    x = x[torch.as_tensor(ck['rnd_idx_train']).cpu()].float()
    noise = torch.as_tensor(ck['latest_noise']).cpu().float()
    return x, torch.clamp(x + noise, 0, 1), float(ck['delta'])


def run_check(ebm, x, xp, delta, steps_list, max_images=2000, seed=0):
    """x, xp: clean / poisoned task images in [0,1]. Returns (report dict, energy arrays for plotting)."""
    g = torch.Generator().manual_seed(seed)
    rows = (xp - x).flatten(1).abs().amax(1) > 0
    idx = rows.nonzero().flatten()
    idx = idx[torch.randperm(len(idx), generator=g)[:max_images]]
    xc, xpo = x[idx], xp[idx]
    xu = torch.clamp(xc + (torch.rand(xc.shape, generator=g) * 2 - 1) * delta, 0, 1)
    ec, ep, eu = energies(ebm, xc), energies(ebm, xpo), energies(ebm, xu)
    rep = {'n_images': len(idx), 'n_poisoned_total': int(rows.sum()), 'delta': delta,
           'energy_clean': [ec.mean().item(), ec.std().item()],
           'energy_poisoned': [ep.mean().item(), ep.std().item()],
           'energy_uniform': [eu.mean().item(), eu.std().item()],
           'auroc_poisoned_vs_clean': auroc(ec, ep),
           'auroc_uniform_vs_clean': auroc(ec, eu),
           'purify': []}
    curves = {'clean': ec, 'BrainWash': ep, 'uniform': eu}
    for s in steps_list:
        pp, r = purify_with_report(ebm, xc, xpo, s)
        r['energy_purified_poisoned'] = energies(ebm, pp).mean().item()
        rep['purify'].append(r)
        curves[f'purified {s} steps'] = energies(ebm, pp)
        print('  ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return rep, curves


def verdict(rep):
    """Decision uses PURIFICATION results, not detection: detection AUROC is ~1.0 for ANY noise of this size
    (compare auroc_uniform_vs_clean), so it says nothing about whether Langevin can remove the perturbation."""
    last = rep['purify'][-1]
    surv, q, steps = last['survival_l2'], last['psnr_clean'], last['steps']
    gain = last['psnr_poisoned'] - last['input_psnr_poisoned']
    if surv <= 0.5 and q >= 25:
        return f'GO: at {steps} steps survival {surv:.2f} <= 0.5 with clean PSNR {q:.1f} dB -> run step 2'
    if surv <= 0.75 and q >= 25:
        return f'MAYBE: survival {surv:.2f} at {steps} steps; run one stage-4 point before a sweep'
    return (f'STOP: at {steps} steps survival is {surv:.2f} (perturbation mostly intact), purified poison is only '
            f'{gain:+.1f} dB closer to clean, and clean PSNR is {q:.1f} dB. Detection AUROC is not evidence of removal.')


def save_hist(curves, path, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(7, 4))
    for k, v in curves.items():
        plt.hist(v.numpy(), bins=60, alpha=0.45, label=k, density=True)
    plt.xlabel('EBM energy (lower = more natural)'); plt.ylabel('density'); plt.title(title, fontsize=9)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--puregen_repo', default=os.environ.get('PUREGEN_REPO'))
    ap.add_argument('--ebm_path', default=DEFAULT_EBM)
    ap.add_argument('--experiment', default='split_cifar100',
                    choices=['split_cifar100', 'split_mini_imagenet', 'split_tiny_imagenet'])
    ap.add_argument('--noise_pkl', nargs='+', required=True, help='pkl files or noise directories (newest pkl used)')
    ap.add_argument('--steps', type=int, nargs='+', default=[50, 100, 150, 300])
    ap.add_argument('--max_images', type=int, default=2000)
    ap.add_argument('--tasknum', type=int, default=10)
    ap.add_argument('--out', default='repro/defense/ebm_check')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ebm = load_ebm(a.ebm_path, a.puregen_repo)
    for p in a.noise_pkl:
        if os.path.isdir(p):
            p = max(glob.glob(os.path.join(p, '*.pkl')), key=os.path.getmtime)
        tag = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        print(f'=== {tag}  ({p})', flush=True)
        x, xp, delta = load_task(a.experiment, p, a.tasknum)
        rep, curves = run_check(ebm, x, xp, delta, a.steps, a.max_images)
        rep.update(noise_pkl=p, ebm_path=a.ebm_path, verdict=verdict(rep))
        print(json.dumps({k: v for k, v in rep.items() if k != 'purify'}, indent=1))
        with open(os.path.join(a.out, f'{tag}.json'), 'w') as f:
            json.dump(rep, f, indent=2)
        save_hist(curves, os.path.join(a.out, f'{tag}_energy.png'), f'{tag}  AUROC={rep["auroc_poisoned_vs_clean"]:.2f}')


if __name__ == '__main__':
    main()
