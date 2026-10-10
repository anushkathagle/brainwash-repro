"""Helpers shared by the defense notebooks: run BrainWash stage 4 (resumable), parse results, print tables.

recovery % = (BWT_defended - BWT_undefended) / (BWT_clean - BWT_undefended) * 100
             0 % = no help, 100 % = forgetting back to the no-attack level (can exceed 100 or go negative).
"""
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_defense import parse_log  # noqa: E402

# split CIFAR-100 victims (repro/sweep/configs.tsv). The checkpoint's own lambda is used at eval time anyway.
METHODS = {'ewc': dict(lamb=500000), 'mas': dict(lamb=10), 'rwalk': dict(lamb=1), 'afec_ewc': dict(lamb=500000, lamb_emp=100)}


def sh(cmd, log=None):
    """Run a shell command, stream its output, optionally tee to a log file. Raises on failure."""
    print('$', cmd, flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    f = open(log, 'w') if log else None
    lines = []
    for line in p.stdout:
        lines.append(line)
        if f:
            f.write(line); f.flush()
        if not line.startswith(('Epoch', '| Epoch')):   # keep the notebook readable
            print(line, end='', flush=True)
    if f:
        f.close()
    if p.wait():
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}\n' + ''.join(lines[-40:]))
    print(f'[{time.time() - t0:.0f}s]', flush=True)
    return ''.join(lines)


def find_pkl(noise_root, tag):
    pk = glob.glob(os.path.join(noise_root, tag, '*.pkl'))
    if not pk:
        raise FileNotFoundError(f'no noise pkl in {os.path.join(noise_root, tag)} - copy it from the cluster first')
    return max(pk, key=os.path.getmtime)


def stage4(name, noise_pkl, variant, method, results_dir, defense_args='', seed=0, experiment='split_cifar100'):
    """variant: clean | uniform | ours. Skips if results_dir/name.log already has a result. Returns parsed dict."""
    os.makedirs(results_dir, exist_ok=True)
    log = os.path.join(results_dir, f'{name}.log')
    if os.path.exists(log) and parse_log(log):
        print(f'[skip] {name} (done)')
        return {'name': name, **parse_log(log)}
    m = METHODS[method]
    noise = {'clean': '', 'uniform': '--addnoise --uniform', 'ours': '--addnoise'}[variant]
    emp = f"--lamb_emp {m['lamb_emp']}" if 'lamb_emp' in m else ''
    cmd = (f'python main_baselines.py --experiment {experiment} --approach {method} --lasttask 9 --tasknum 10 '
           f'--nepochs 20 --batch-size 16 --lr 0.01 --clip 100. --lamb {m["lamb"]} {emp} --seed {seed} '
           f'--checkpoint "{noise_pkl}" --init_acc {noise} {defense_args}')
    sh(cmd, log)
    r = parse_log(log)
    if r is None:
        raise RuntimeError(f'no "After BWT" line in {log}')
    return {'name': name, **r}


def recovery(bwt_def, bwt_undef, bwt_clean):
    gap = bwt_clean - bwt_undef
    return 100 * (bwt_def - bwt_undef) / gap if abs(gap) > 1e-9 else float('nan')


def table(rows, undef=None, clean=None):
    """rows: list of dicts from stage4(); undef/clean: reference dicts for the recovery column."""
    print(f"{'run':52} {'BWT':>7} {'last-task Acc':>13} {'avg Acc':>8} {'recovery':>9} {'surv':>6} {'PSNRc':>6}")
    for r in rows:
        rec = recovery(r['bwt'], undef['bwt'], clean['bwt']) if (undef and clean and 'clean' not in r['name']) else float('nan')
        print(f"{r['name']:52} {r['bwt']:7.2f} {r['acc']:13.2f} {r['avg']:8.2f} {rec:8.1f}% "
              f"{r.get('survival_l2', float('nan')):6.2f} {r.get('psnr_clean', float('nan')):6.1f}")
    print('BWT more negative = more forgetting. recovery = share of the attack-induced BWT loss that the defense '
          'removed. surv = fraction of perturbation surviving purification; PSNRc = purifier distortion on clean data.')
