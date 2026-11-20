"""
colab_helpers.py

Shared pieces of the Layer-2 Colab notebooks (layer2_defense/notebooks/*.ipynb), kept here so that every
notebook runs the same loop and builds the same tables:

    base_cmd(...)          the stage-4 command (friends_layer2.py, CIFAR-100 / EWC settings of repro/sweep/configs.tsv)
    run_arms(...)          runs a list of (name, flags) arms; arms with a finished log in results_dir are skipped,
                           so a disconnected Colab can simply be re-run
    record_code_version()  appends the git commit of this checkout to results_dir/code_version.txt
    results_table(...)     summary_layer2.tsv as a table, with "recovered %" and "last-task change vs the attack"
    per_epoch_table(...)   old-task accuracy after every epoch (arms run with --log_old_acc)
"""
import os
import re
import glob
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_cmd(noise_pkl, results_dir, seed=0, lamb=500000, approach='ewc', extra=''):
    """friends_layer2.py with the stage-4 settings used throughout (CIFAR-100, task 9 poisoned, 20 epochs)."""
    return (f'python layer2_defense/friends_layer2.py --experiment split_cifar100 --approach {approach} '
            f'--lasttask 9 --tasknum 10 --nepochs 20 --batch-size 16 --lr 0.01 --clip 100. --lamb {lamb} '
            f'--seed {seed} --checkpoint "{noise_pkl}" --init_acc --log_old_acc --out_dir "{results_dir}" {extra}').strip()


def record_code_version(results_dir):
    """Write which commit of the code produced the runs in this folder (one line per Colab session)."""
    try:
        rev = subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%h %ad %s', '--date=short'],
                             capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(['git', '-C', REPO, 'rev-parse', '--abbrev-ref', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
    except Exception as e:  # git missing: still run
        rev, branch = f'unknown ({e})', '?'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'code_version.txt'), 'a') as fh:
        fh.write(f'{time.strftime("%Y-%m-%d %H:%M")}  {branch}  {rev}\n')
    print(f'code: {branch} @ {rev}')
    return rev


def run_arms(arms, cmd, results_dir, seed=0, stop_after_fails=3):
    """Run each (name, flags) arm; logs go to results_dir/<name>_s<seed>.log. Finished arms are skipped."""
    assert len({n for n, _ in arms}) == len(arms), 'duplicate arm names'
    os.makedirs(results_dir, exist_ok=True)
    fails = 0
    for name, flags in arms:
        log = os.path.join(results_dir, f'{name}_s{seed}.log')
        if os.path.exists(log) and 'appended summary row' in open(log).read():
            print(f'[skip] {name}: already done')
            continue
        t0 = time.time()
        print(f'[run ] {name} ...', flush=True)
        subprocess.run(f'cd "{REPO}" && {cmd} {flags} 2>&1 | grep -v Warning > "{log}"', shell=True)
        text = open(log).read()
        done = 'appended summary row' in text
        print(f'[{"done" if done else "FAIL"}] {name}  ({time.time() - t0:.0f}s)', flush=True)
        for line in text.splitlines():
            if line.startswith(('[LAYER2] friendly noise computed', '[LAYER1]')):
                print('       ', line)
        if done:
            fails = 0
        else:
            print('\n'.join(text.splitlines()[-12:]))
            fails += 1
            if fails >= stop_after_fails:  # something systematic (e.g. Drive unmounted): stop instead of burning hours
                raise SystemExit(f'{stop_after_fails} arms failed in a row: stopping')
    print('ALL DONE')


def results_table(results_dir, seed=0):
    """summary_layer2.tsv of one folder (one attack eps), in %:
       recovered_%    = share of the BWT gap between the attack ('ours') and the clean run closed by the arm
       last_vs_attack = last-task accuracy minus the attack's (negative = what the recovery costs on the new task)"""
    import pandas as pd
    df = pd.read_csv(os.path.join(results_dir, 'summary_layer2.tsv'), sep='\t')
    df = df[df.seed == seed].copy()
    df['eps'] = df.noise_pkl.str.extract(r'eps([0-9]+\.?[0-9]*)')[0].fillna('?')   # attack eps from the noise file name
    df = df.drop_duplicates(['eps', 'method'], keep='last')
    for c in ['bwt_after', 'avg_acc_after', 'last_task_acc']:
        df[c] = 100 * df[c]
    out = []
    for eps, d in df.groupby('eps'):
        d = d.copy()
        if {'clean', 'ours'} <= set(d.method):
            b_clean = d.loc[d.method == 'clean', 'bwt_after'].iloc[0]
            b_att = d.loc[d.method == 'ours', 'bwt_after'].iloc[0]
            l_att = d.loc[d.method == 'ours', 'last_task_acc'].iloc[0]
            gap = b_clean - b_att
            d['recovered_%'] = 100 * (d.bwt_after - b_att) / gap if abs(gap) > 1e-9 else float('nan')
            d['last_vs_attack'] = d.last_task_acc - l_att
        out.append(d)
    df = pd.concat(out)
    for c in ['bwt_after', 'avg_acc_after', 'last_task_acc', 'recovered_%', 'last_vs_attack']:
        if c in df:
            df[c] = df[c].round(1)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_colwidth', 70)
    cols = ['eps', 'method', 'bwt_after', 'recovered_%', 'avg_acc_after', 'last_task_acc', 'last_vs_attack',
            'old_acc_at_eta', 'eta_mean_over_zeta', 'eta_frac_at_bound']
    return df[[c for c in cols if c in df]].reset_index(drop=True)


def per_epoch_table(results_dir, seed=0):
    """Old-task accuracy after every epoch (rows = epochs, columns = arms); arms run with --log_old_acc."""
    import pandas as pd
    rows = {}
    for log in sorted(glob.glob(os.path.join(results_dir, f'*_s{seed}.log'))):
        name = os.path.basename(log)[:-len(f'_s{seed}.log')]
        pts = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'\[LAYER2\] epoch (\d+): old-task acc ([0-9.]+)', open(log).read())}
        if pts:
            rows[name] = pts
    return pd.DataFrame(rows).sort_index() if rows else 'no arm in this folder was run with --log_old_acc'
