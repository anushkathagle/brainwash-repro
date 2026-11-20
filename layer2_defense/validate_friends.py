"""
validate_friends.py

Minimal validation of the FRIENDS port (layer2_defense/friends.py) plus the
diagnostic that decides between friendly-noise variants (a) and (b). It needs
no Layer-2 training script: it loads a BrainWash checkpoint, the last task's
(poisoned) training data and the CL method exactly as stage 4 does, and then
runs three checks.

  V1  port == official code.  Our generate_friendly_noise and the official one
      (friendly-noise/friendly_noise.py) are run on the same model, batch and
      RNG seed. On CPU the noise must match to float tolerance (it does bit for
      bit). The optimization is chaotic -- a 1e-7 change in the official code's
      input already flips some pixels between +zeta and -zeta -- so on GPU ours vs
      official is judged against that 1e-7 sensitivity. CIFAR-100 only (the
      official function hard-codes 3x32x32).

  V2  invariants + the (a)/(b)/(c) diagnostic. Friendly noise is generated for
      every task-T training image under three variants:
          (c) 'as proposed'  KL on head T at the start-of-task checkpoint
          (a) 'features'     relative MSE on the 1024-d embedding, same checkpoint
          (b) 'warm-up'      KL on head T after --warmup_epochs of training on
                             task T with random noise only (paper's Alg. 2)
      For each it reports how far the optimization moved the noise (mean|eta|/zeta,
      fraction at the bound, vs the U(-8/255,8/255) initialization) and how
      "friendly" it is: how much it changes features / head-T predictions /
      old-task predictions, relative to a control with the SAME per-pixel
      magnitudes but random signs. For (b) it also reports how much old-task
      accuracy is already lost during the warm-up.

  V4  random noise r and the other Layer-2 options (BatchNorm control, head-only epochs, epoch log).
      The Layer-2 training step (friends_training.py: the
      original step + one marked line) adds noise to every training batch, of
      both AFEC networks, never during testing, with fresh draws each batch;
      and with mu=0 it trains to exactly the same weights as the original step.

  V5  does eta change what the model LEARNS from the poisoned images?
      (V2 only checks that eta is gentle on the model; V5 checks it does
      something to the poison.)
        part 1: for each option, compare the "lesson" (the direction a training
                step would push the shared body) on poisoned images with and
                without eta, against same-size random noise.
        part 2: train task T for --lesson_epochs on clean / poisoned /
                poisoned+eta / poisoned+same-size-random images (random noise r
                off) and measure old-task accuracy: how much of the poison's
                extra damage does eta remove? The clean images are used only as
                this reference -- the defense never sees them.

  V6  how much of the forgetting is BatchNorm statistics? The model's BatchNorm layers keep
      running averages of their inputs; they are not weights, so EWC/RWALK/AFEC do not
      protect them. Train task T for --lesson_epochs on clean / poisoned / poisoned + random
      noise r, then test the old tasks as they are and with the checkpoint's BatchNorm
      statistics put back. Needs no friendly noise.

  V7  gradient signature. The poison's signature in the training signal (what the training step
      pushes the shared body to do on the poisoned images, minus on the same images without
      poison) is compared with and without noise added to both. cosine ~1 = the noise does not
      touch what the poison encodes. Needs no friendly-noise training.

Run from anywhere; paths are resolved before the script chdirs to the repo root
(the data loaders read ./data). Example (CIFAR-100, EWC, cautious eps=0.3):

    python layer2_defense/validate_friends.py \
        --noise_pkl /content/drive/MyDrive/bw/noise_ewc_wcur_1.0_..._.pkl \
        --official_repo /content/friendly-noise \
        --out layer2_defense/results/v_cifar_ewc_cautious03.json
"""

import argparse
import contextlib
import inspect
import io
import json
import math
import os
import pickle as pkl
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group('inputs')
    src.add_argument('--noise_pkl', default=None,
                     help='stage-3 noise pkl (contains the victim checkpoint). Preferred: lets V2 run on the poisoned task.')
    src.add_argument('--ckpt', default=None,
                     help='victim checkpoint pkl (repro/artifacts/.../cl_methods/...). Used only if --noise_pkl is not given; '
                          'V2 then runs on CLEAN task-T data.')
    src.add_argument('--variant', choices=['poison', 'clean'], default=None,
                     help='task-T data to use (default: poison if --noise_pkl is given, else clean)')
    src.add_argument('--official_repo', default=None,
                     help='clone of github.com/tianyu139/friendly-noise (needed for V1)')
    src.add_argument('--mini_root', default=os.path.join(os.path.expanduser('~'), 'data', 'miniImagenet'))

    run = p.add_argument_group('what to run')
    run.add_argument('--checks', nargs='+', default=['v1', 'v2', 'v4', 'v5'], choices=['v1', 'v2', 'v4', 'v5', 'v6', 'v7'],
                     help='v5 needs the friendly noise from v2, so v5 also runs v2')
    run.add_argument('--v2_variants', nargs='+', default=['c', 'a', 'b'], choices=['a', 'b', 'c'])
    run.add_argument('--out', default=None, help='JSON results file (default: layer2_defense/results/...)')

    fr = p.add_argument_group('FRIENDS hyper-parameters (official defaults; zeta/mu in 1/255 units)')
    fr.add_argument('--zeta', type=float, default=16, help='friendly-noise L-inf clamp, x/255 (paper: 16)')
    fr.add_argument('--mu_rand', type=float, default=16, help='random-noise magnitude, x/255 (paper: 16)')
    fr.add_argument('--noise_kind', default='bernoulli', choices=['bernoulli', 'uniform', 'gaussian'])
    fr.add_argument('--friendly_epochs', type=int, default=30, help='official main.py default (paper text says 20)')
    fr.add_argument('--friendly_lr', type=float, default=100.)
    fr.add_argument('--friendly_lambda', type=float, default=1.,
                    help='lambda: weight of the "make the friendly noise large" term (official code: friendly_mu)')
    fr.add_argument('--friendly_batch', type=int, default=128)
    fr.add_argument('--warmup_epochs', type=int, default=2, help='variant (b): epochs of random-only training first')

    st = p.add_argument_group('stage-4 settings (match repro/sweep/stage4.sbatch)')
    st.add_argument('--lasttask', type=int, default=9)
    st.add_argument('--tasknum', type=int, default=10)
    st.add_argument('--lr', type=float, default=0.01)
    st.add_argument('--batch-size', dest='batch_size', type=int, default=16)
    st.add_argument('--clip', type=float, default=100.)
    st.add_argument('--seed', type=int, default=0)

    sz = p.add_argument_group('sizes')
    sz.add_argument('--v1_n', type=int, default=256, help='images used for the GPU part of V1 (2 batches of 128)')
    sz.add_argument('--v1_n_cpu', type=int, default=128, help='images used for the exact CPU part of V1')
    sz.add_argument('--v2_n', type=int, default=0, help='images used for V2 (0 = the whole task)')
    sz.add_argument('--v4_n', type=int, default=160, help='images used for V4 (10 steps at batch 16)')
    sz.add_argument('--lesson_grad_n', type=int, default=512, help='V5 part 1: images used for the lesson similarity')
    sz.add_argument('--lesson_epochs', type=int, default=2, help='V5 part 2 and V6: epochs of real training per run')
    sz.add_argument('--v6_mus', type=float, nargs='+', default=[16, 64], help='V6: random-noise sizes (x/255) to try')
    sz.add_argument('--v7_mus', type=float, nargs='+', default=[16, 64], help='V7: random-noise sizes (x/255) to try')
    sz.add_argument('--v7_n', type=int, default=512, help='V7: images used')
    return p.parse_args()


args = parse_args()
for _k in ('noise_pkl', 'ckpt', 'official_repo', 'out', 'mini_root'):
    if getattr(args, _k):
        setattr(args, _k, os.path.abspath(os.path.expanduser(getattr(args, _k))))
if not (args.noise_pkl or args.ckpt):
    sys.exit('give --noise_pkl (preferred) or --ckpt')
if args.variant is None:
    args.variant = 'poison' if args.noise_pkl else 'clean'
if args.variant == 'poison' and not args.noise_pkl:
    sys.exit('--variant poison needs --noise_pkl')

ZETA = args.zeta / 255
MU_RAND = args.mu_rand / 255
INIT_MAG = 8 / 255  # hard-coded in the official generator
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    sys.exit('CUDA is required (Colab: Runtime > Change runtime type > GPU)')


# ----------------------------------------------------------------------------
# checkpoint first: it tells us the CL method and dataset, which approaches/*
# need in sys.argv at import time (they call get_args() on import)
# ----------------------------------------------------------------------------

noise_ck = None
t0 = time.time()
if args.noise_pkl:
    with open(args.noise_pkl, 'rb') as fh:
        noise_ck = pkl.load(fh)
    pre = noise_ck['pretrained_ckpt']
else:
    with open(args.ckpt, 'rb') as fh:
        pre = pkl.load(fh)
approach_name = pre['cont_method_args']['method']
experiment = pre['dataset']
print(f'[load] checkpoint: method={approach_name} dataset={experiment} ({time.time() - t0:.0f}s)')

sys.argv = [sys.argv[0], '--experiment', experiment, '--approach', approach_name, '--optim', 'sgd',
            '--lasttask', str(args.lasttask), '--tasknum', str(args.tasknum)]
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from approaches.arguments import get_args  # noqa: E402
import approaches.utils as appr_utils  # noqa: E402
from resnet import ResNet18  # noqa: E402
from friends import generate_friendly_noise, RandomNoise, FriendsDefense, model_output  # noqa: E402
from friends_training import TRAINERS  # noqa: E402

if approach_name == 'ewc':
    from approaches import ewc as approach_mod
elif approach_name == 'rwalk':
    from approaches import rwalk as approach_mod
elif approach_name == 'afec_ewc':
    from approaches import afec_ewc as approach_mod
elif approach_name == 'mas':
    from approaches import mas as approach_mod
else:
    sys.exit(f'unsupported method {approach_name}')
IS_AFEC = 'afec' in approach_name
CL_ARGS = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# data (same calls as main_baselines.py)
# ----------------------------------------------------------------------------

def load_data():
    if experiment == 'split_cifar100':
        from approaches.data_utils import generate_split_cifar100_tasks
        data, _, _, _ = generate_split_cifar100_tasks(args.tasknum, args.seed, rnd_order=False, order=np.arange(100))
    elif experiment == 'split_mini_imagenet':
        from approaches.data_utils import generate_split_mini_imagenet_tasks
        data, _, _, _ = generate_split_mini_imagenet_tasks(args.mini_root, task_num=args.tasknum,
                                                           rnd_order=False, order=np.arange(100))
    else:
        sys.exit(f'{experiment}: not wired up in this validation script yet')
    return data


def task_train_data(data, t):
    """xtrain/ytrain for task t exactly as main_baselines.py builds them, plus the
    same images WITHOUT the poison (same order) -- the clean copy is only used by
    V5 as a reference; the defense itself never sees it."""
    xtrain = data[t]['train']['x'].clone()
    ytrain = data[t]['train']['y'].clone()
    if args.variant == 'poison':
        all_noise = noise_ck['latest_noise'].detach().cpu()
        prm = noise_ck['rnd_idx_train']
        xtrain = xtrain[prm]
        ytrain = ytrain[prm]
        xclean = xtrain.clone()
        xtrain = torch.clamp(xtrain + all_noise, 0, 1)
        return xtrain, ytrain, xclean
    return xtrain, ytrain, xtrain


print('[load] data ...')
DATA = load_data()
T = args.lasttask
XTRAIN, YTRAIN, XCLEAN = task_train_data(DATA, T)
XVALID, YVALID = DATA[T]['test']['x'].clone(), DATA[T]['test']['y'].clone()
print(f'[load] task {T} train: {tuple(XTRAIN.shape)} ({args.variant}), test: {tuple(XVALID.shape)}')


# ----------------------------------------------------------------------------
# CL learner (same construction as main_baselines.py)
# ----------------------------------------------------------------------------

def build_appr(layer2=False):
    """layer2=False: the original approaches/<method>.py learner.
    layer2=True: the Layer-2 learner from friends_training.py (same, plus the
    marked random-noise line in its training step; no noise until one is set)."""
    ncls = DATA['ncla'] // args.tasknum
    net = ResNet18(args.tasknum, ncls, nf=32, include_head=True).to(DEVICE)
    net_emp = ResNet18(args.tasknum, ncls, nf=32, include_head=True).to(DEVICE)
    cargs = pre['cont_method_args']
    if layer2 and approach_name not in TRAINERS:
        sys.exit(f'no Layer-2 trainer for {approach_name}')
    Appr = TRAINERS[approach_name] if layer2 else approach_mod.Appr
    with contextlib.redirect_stdout(io.StringIO()):
        if IS_AFEC:
            appr = Appr(net, sbatch=args.batch_size, lamb=cargs['lamb'], lamb_emp=cargs['lamb_emp'],
                        lr=args.lr, nepochs=1, args=CL_ARGS, log_name='', empty_net=net_emp,
                        clipgrad=args.clip)
        else:
            appr = Appr(net, lamb=cargs['lamb'], sbatch=args.batch_size, lr=args.lr, nepochs=1,
                                     args=CL_ARGS, log_name='', clipgrad=args.clip)
        appr.load_model(pre['model'])
        if IS_AFEC:
            appr.load_emp_model(cargs['model_emp'])
    appr.add_emp = 0  # AFEC sets this inside Appr.train(); needed when calling its epochs directly
    return appr


def train_one_epoch(appr, x, y, t, e):
    """One epoch of task-t training, as Appr.train runs it (minus eval/print)."""
    if IS_AFEC:
        appr.train_emp_epoch(t, x, y, e)
        appr.model_emp_tmp = deepcopy(appr.model_emp)
        appr.model_emp_tmp.train()
        appr_utils.freeze_model(appr.model_emp_tmp)
        with contextlib.redirect_stderr(io.StringIO()):
            appr.fisher_emp, _ = appr_utils.fisher_matrix_diag_emp(t, x, y, appr.model_emp, appr.criterion)
    if len(inspect.signature(appr.train_epoch).parameters) >= 4:
        appr.train_epoch(t, x, y, e)
    else:  # rwalk.train_epoch(t, x, y)
        appr.train_epoch(t, x, y)


def fresh_optimizers(appr):
    appr.optimizer = appr._get_optimizer(args.lr)
    if IS_AFEC:
        appr.optimizer_emp = appr._get_optimizer_emp(args.lr)


def task_accs(appr, upto):
    out = []
    for u in range(upto + 1):
        _, acc = appr.eval(u, DATA[u]['test']['x'], DATA[u]['test']['y'])
        out.append(100 * float(acc))
    return out


# ----------------------------------------------------------------------------
# V1
# ----------------------------------------------------------------------------

class _HeadOnly(torch.nn.Module):
    """model(x)[t] as a plain module -- what the official generator expects."""

    def __init__(self, model, t):
        super().__init__()
        self.model, self.t = model, t

    def forward(self, x):
        return self.model(x)[self.t]


class _TensorDataset:
    transform = staticmethod(lambda z: z)  # BrainWash uses raw [0,1] pixels: no Normalize

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Loader:
    """Minimal stand-in for the official DataLoader: in-order batches of
    (inputs, target, poison_flag, index) and NO RNG consumption (a real
    DataLoader draws a base seed, which would shift the eta initialization)."""

    def __init__(self, x, y, bs):
        self.x, self.y, self.bs = x, y, bs
        self.dataset = _TensorDataset(len(x))

    def __len__(self):
        return math.ceil(len(self.x) / self.bs)

    def __iter__(self):
        for lo in range(0, len(self.x), self.bs):
            hi = min(lo + self.bs, len(self.x))
            yield self.x[lo:hi], self.y[lo:hi], torch.zeros(hi - lo), torch.arange(lo, hi)


def check_v1(appr):
    print('\n' + '=' * 78 + '\nV1  port == official generate_friendly_noise\n' + '=' * 78)
    if XTRAIN.shape[-1] != 32:
        print('  skipped: the official function hard-codes 3x32x32 -- run V1 on CIFAR-100')
        return {'skipped': True}
    if not args.official_repo:
        print('  skipped: pass --official_repo /path/to/friendly-noise')
        return {'skipped': True}
    sys.path.insert(0, args.official_repo)
    import friendly_noise as official

    def run_official(model, x, y, device, loss):
        torch.manual_seed(1234)
        with contextlib.redirect_stdout(io.StringIO()):
            return official.generate_friendly_noise(
                _HeadOnly(model, T), _Loader(x, y, args.friendly_batch), device,
                friendly_epochs=args.friendly_epochs, mu=args.friendly_lambda, friendly_lr=args.friendly_lr,
                clamp_min=-ZETA, clamp_max=ZETA, loss_fn=loss).cpu().float()

    def run_ours(model, x, device, loss):
        torch.manual_seed(1234)
        return generate_friendly_noise(model, x, task=T, target='head', loss_fn=loss,
                                       friendly_epochs=args.friendly_epochs, lam=args.friendly_lambda,
                                       friendly_lr=args.friendly_lr, clamp=ZETA,
                                       batch_size=args.friendly_batch, device=device)

    def diff_stats(u, v):
        d = (u - v).abs()
        return {'max': d.max().item(), 'mean': d.mean().item(), 'frac_gt_1e-3': (d > 1e-3).float().mean().item()}

    def fmt(d):
        return f"max {d['max']:.2e}  mean {d['mean']:.2e}  pixels>1e-3 {100 * d['frac_gt_1e-3']:.2f}%"

    # (1) CPU, exact. CPU kernels are deterministic, so a faithful port must match
    #     the official code bit-for-bit (up to float rounding). This is the real test.
    #     The official function hard-codes inputs.cuda(); it is made a no-op here
    #     only for the duration of this comparison.
    n_cpu = min(args.v1_n_cpu, len(XTRAIN))
    xc, yc = XTRAIN[:n_cpu].cpu(), YTRAIN[:n_cpu].cpu()
    model_cpu = deepcopy(appr.model).cpu()
    real_cuda = torch.Tensor.cuda
    results = {'cpu': {}, 'gpu': {}}
    print(f'  (1) CPU, {n_cpu} images -- deterministic, must match up to float rounding:')
    try:
        torch.Tensor.cuda = lambda self, *a, **k: self
        for loss in ('KL', 'MSE'):
            t0 = time.time()
            d = diff_stats(run_official(model_cpu, xc, yc, torch.device('cpu'), loss),
                           run_ours(model_cpu, xc, torch.device('cpu'), loss))
            ok = d['frac_gt_1e-3'] <= 1e-3  # <=0.1% of pixels; rounding-level drift can still flip a few
            results['cpu'][loss] = dict(d, passed=ok)
            print(f'      loss={loss:3s}  ours vs official: {fmt(d)}   {"PASS" if ok else "FAIL"}  [{time.time() - t0:.0f}s]')
    finally:
        torch.Tensor.cuda = real_cuda
    del model_cpu

    # (2) On the GPU. The noise optimization is extremely sensitive: changing the
    #     official code's INPUT by 1e-7 (float-rounding size) already flips some pixels
    #     between +zeta and -zeta. So the yardstick is: ours vs official must differ
    #     no more than official differs from itself under that 1e-7 nudge.
    n = min(args.v1_n, len(XTRAIN))
    x, y = XTRAIN[:n].cpu(), YTRAIN[:n].cpu()
    x_nudged = (x + 1e-7 * torch.randn_like(x)).clamp(0, 1)
    model = appr.model
    print(f'  (2) on {DEVICE.type.upper()}, {n} images -- ours vs official, and for scale how much the official '
          f'code moves when its inputs change by 1e-7:')
    for loss in ('KL', 'MSE'):
        ref = run_official(model, x, y, DEVICE, loss)
        ours = run_ours(model, x, DEVICE, loss)
        ref_nudged = run_official(model, x_nudged, y, DEVICE, loss)
        d, d_nudge = diff_stats(ref, ours), diff_stats(ref, ref_nudged)
        ok = d['frac_gt_1e-3'] <= max(d_nudge['frac_gt_1e-3'], 1e-4) or d['max'] <= 1e-5
        results['gpu'][loss] = {'ours_vs_official': d, 'official_vs_official_inputs_nudged_1e-7': d_nudge,
                                'mean_abs_over_zeta_ours': (ours.abs().mean() / ZETA).item(),
                                'mean_abs_over_zeta_official': (ref.abs().mean() / ZETA).item(), 'passed': ok}
        print(f'      loss={loss:3s}  ours vs official:                     {fmt(d)}   {"PASS" if ok else "FAIL"}')
        print(f'                official vs itself (inputs +1e-7): {fmt(d_nudge)}')
        print(f'                mean|eta|/zeta  ours {results["gpu"][loss]["mean_abs_over_zeta_ours"]:.3f}  '
              f'official {results["gpu"][loss]["mean_abs_over_zeta_official"]:.3f}')
    return results


# ----------------------------------------------------------------------------
# V2
# ----------------------------------------------------------------------------

def noise_stats(eta, trace):
    a = eta.abs()
    first = np.mean([b['first'][0] for b in trace])
    last = np.mean([b['last'][0] for b in trace])
    return {
        'linf': a.max().item(),
        'linf_over_zeta': a.max().item() / ZETA,
        'mean_abs_over_zeta': a.mean().item() / ZETA,
        'frac_at_bound': (a >= 0.99 * ZETA).float().mean().item(),
        'init_mean_abs_over_zeta': (INIT_MAG / 2) / ZETA,
        'emp_risk_first': float(first),
        'emp_risk_last': float(last),
        'constraint_first': float(np.mean([b['first'][1] for b in trace])),
        'constraint_last': float(np.mean([b['last'][1] for b in trace])),
    }


@torch.no_grad()
def response(model, x, eta, t, bs=500):
    """How much x -> clamp(x+eta) changes the model: features, head t, old heads."""
    model.eval()
    agree = kl_t = feat = kl_old = 0.0
    n = len(x)
    for lo in range(0, n, bs):
        xb = x[lo:lo + bs].to(DEVICE)
        xp = torch.clamp(xb + eta[lo:lo + bs].to(DEVICE), 0, 1)
        f0 = model_output(model, xb, 'features', t)
        fp = model_output(model, xp, 'features', t)
        feat += ((fp - f0).norm(dim=1) / f0.norm(dim=1).clamp_min(1e-12)).sum().item()
        l0, lp = model.heads[t](f0), model.heads[t](fp)
        agree += (l0.argmax(1) == lp.argmax(1)).float().sum().item()
        kl_t += F.kl_div(F.log_softmax(lp, 1), F.log_softmax(l0, 1), log_target=True, reduction='sum').item()
        if t > 0:
            ks = [F.kl_div(F.log_softmax(model.heads[u](fp), 1), F.log_softmax(model.heads[u](f0), 1),
                           log_target=True, reduction='sum').item() for u in range(t)]
            kl_old += float(np.mean(ks))
    return {'feat_rel_change': feat / n, 'headT_agree': agree / n, 'headT_kl': kl_t / n, 'oldheads_kl': kl_old / n}


def controls(eta, seed=0):
    g = torch.Generator().manual_seed(seed)
    sign = (torch.rand(eta.shape, generator=g) > 0.5).float() * 2 - 1
    return {
        'sign_shuffled': eta.abs() * sign,           # same per-pixel magnitudes, random direction
        'bernoulli_zeta': sign * ZETA,               # every pixel at +-zeta
    }


ETAS = {}          # variant -> friendly noise eta (kept for V5)
START_MODELS = {}  # variant -> learner the noise was computed with (V5 trains from it)


def evaluate_variant(name, model, x, target, loss_fn):
    t0 = time.time()
    eta, trace = generate_friendly_noise(model, x, task=T, target=target, loss_fn=loss_fn,
                                         friendly_epochs=args.friendly_epochs, lam=args.friendly_lambda,
                                         friendly_lr=args.friendly_lr, clamp=ZETA,
                                         batch_size=args.friendly_batch, device=DEVICE, return_trace=True)
    gen_s = time.time() - t0
    ETAS[name] = eta
    res = {'target': target, 'loss': loss_fn, 'gen_seconds': gen_s, 'noise': noise_stats(eta, trace)}
    res['response'] = {'friendly': response(model, x, eta, T)}
    for cname, c in controls(eta).items():
        res['response'][cname] = response(model, x, c, T)
    fr, sh = res['response']['friendly'], res['response']['sign_shuffled']
    res['friendliness_ratio'] = {  # < 1 means the optimized directions are gentler than random ones
        k: (fr[k] / sh[k] if sh[k] > 0 else float('nan')) for k in ('feat_rel_change', 'headT_kl', 'oldheads_kl')}
    return res


def print_variant(tag, r):
    ns, rf = r['noise'], r['response']
    print(f'\n  ({tag})  target={r["target"]} loss={r["loss"]}   [{r["gen_seconds"]:.0f}s]')
    print(f'      noise   : mean|eta|/zeta={ns["mean_abs_over_zeta"]:.3f} (init {ns["init_mean_abs_over_zeta"]:.3f})  '
          f'at-bound={100 * ns["frac_at_bound"]:.1f}%  Linf/zeta={ns["linf_over_zeta"]:.3f}')
    print(f'      objective: emp_risk {ns["emp_risk_first"]:.4g} -> {ns["emp_risk_last"]:.4g}   '
          f'constraint {ns["constraint_first"]:.3g} -> {ns["constraint_last"]:.3g}')
    print(f'      {"":15s} {"feat chg":>9s} {"headT agree":>12s} {"headT KL":>10s} {"old-heads KL":>13s}')
    for k in ('friendly', 'sign_shuffled', 'bernoulli_zeta'):
        v = rf[k]
        print(f'      {k:15s} {v["feat_rel_change"]:9.4f} {100 * v["headT_agree"]:11.1f}% '
              f'{v["headT_kl"]:10.4g} {v["oldheads_kl"]:13.4g}')
    fr = r['friendliness_ratio']
    print(f'      friendliness ratio (friendly / sign-shuffled; <1 = gentler than random): '
          f'feat {fr["feat_rel_change"]:.3f}  headT-KL {fr["headT_kl"]:.3f}  old-KL {fr["oldheads_kl"]:.3f}')


def check_v2(appr):
    print('\n' + '=' * 78 + '\nV2  invariants + (a)/(b)/(c) diagnostic\n' + '=' * 78)
    n = args.v2_n or len(XTRAIN)
    x, y = XTRAIN[:n].cpu(), YTRAIN[:n].cpu()
    print(f'  {n} task-{T} training images ({args.variant}); zeta={args.zeta}/255, '
          f'{args.friendly_epochs} steps, lr={args.friendly_lr}, lambda={args.friendly_lambda}')
    out = {}
    ckpt_model = appr.model
    START_MODELS['c'] = START_MODELS['a'] = appr

    if 'c' in args.v2_variants:
        out['c'] = evaluate_variant('c', ckpt_model, x, 'head', 'KL')
        print_variant('c: as proposed -- KL, head T, checkpoint', out['c'])
    if 'a' in args.v2_variants:
        out['a'] = evaluate_variant('a', ckpt_model, x, 'features', 'MSE_rel')
        print_variant('a: features -- relative MSE, checkpoint', out['a'])

    if 'b' in args.v2_variants:
        accs_before = task_accs(appr, T)
        wa = build_appr(layer2=True)  # warm up a separate copy so the checkpoint model stays intact
        wa.random_noise = RandomNoise(MU_RAND, args.noise_kind, seed=args.seed)
        if IS_AFEC:
            wa.random_noise_emp = RandomNoise(MU_RAND, args.noise_kind, seed=args.seed + 1)
        fresh_optimizers(wa)
        t0 = time.time()
        np.random.seed(args.seed)
        for e in range(args.warmup_epochs):
            with contextlib.redirect_stderr(io.StringIO()):
                train_one_epoch(wa, x, y, T, e)
        wa.random_noise = TRAINERS[approach_name].random_noise          # back to "no noise"
        if IS_AFEC:
            wa.random_noise_emp = TRAINERS[approach_name].random_noise_emp
        START_MODELS['b'] = wa
        warm_s = time.time() - t0
        accs_after = task_accs(wa, T)
        drop = float(np.mean([accs_after[u] - accs_before[u] for u in range(T)]))
        out['b'] = evaluate_variant('b', wa.model, x, 'head', 'KL')
        out['b']['warmup'] = {'epochs': args.warmup_epochs, 'seconds': warm_s,
                              'old_task_acc_before': accs_before[:T], 'old_task_acc_after': accs_after[:T],
                              'mean_old_task_change': drop,
                              'taskT_test_acc_before': accs_before[T], 'taskT_test_acc_after': accs_after[T]}
        print_variant(f'b: warm-up {args.warmup_epochs} ep (random noise only) -- KL, head T', out['b'])
        print(f'      warm-up snapshot (mid-training, NOT a final result): old-task acc '
              f'{np.mean(accs_before[:T]):.1f}% -> {np.mean(accs_after[:T]):.1f}% ({drop:+.1f} pts); '
              f'task-{T} test acc {accs_before[T]:.1f}% -> {accs_after[T]:.1f}%')
    return out


# ----------------------------------------------------------------------------
# V4
# ----------------------------------------------------------------------------

def run_epoch_on_subset(x, y, layer2, mu, track=False):
    """Fresh learner from the checkpoint, one epoch on (x, y).
    layer2=False: original trainer. layer2=True: Layer-2 trainer with random noise of size mu."""
    appr = build_appr(layer2=layer2)
    fresh_optimizers(appr)
    noises = {}
    if layer2 and mu is not None:
        appr.random_noise = noises['model'] = RandomNoise(mu, args.noise_kind, seed=args.seed, track=track)
        if IS_AFEC:
            appr.random_noise_emp = noises['model_emp'] = RandomNoise(mu, args.noise_kind, seed=args.seed + 1,
                                                                      track=track)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with contextlib.redirect_stderr(io.StringIO()):
        train_one_epoch(appr, x, y, T, 0)
    return appr, noises


def max_param_diff(m1, m2):
    return max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))


def check_v4():
    print('\n' + '=' * 78 + '\nV4  random noise r in the Layer-2 training step\n' + '=' * 78)
    n = min(args.v4_n, len(XTRAIN))
    x, y = XTRAIN[:n].cpu(), YTRAIN[:n].cpu()
    steps = math.ceil(n / args.batch_size)
    res, ok_all = {}, True

    def report(key, ok, text):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f'  {text}  {"PASS" if ok else "FAIL"}')

    appr, nz = run_epoch_on_subset(x, y, True, MU_RAND, track=True)
    expected = {'model': steps}
    if IS_AFEC:
        expected['model_emp'] = steps
    for k, e in expected.items():
        got = nz[k].calls
        res[f'{k}_noisy_batches'] = {'got': got, 'expected': e, 'pass': got == e}
        report(f'{k}_noisy_batches', got == e, f'{k}: batches that got random noise in 1 epoch = {got} (expected {e})')

    before = nz['model'].calls
    appr.eval(T, XVALID, YVALID)
    appr.eval(0, DATA[0]['test']['x'], DATA[0]['test']['y'])
    ok = nz['model'].calls == before
    res['no_noise_in_eval'] = ok
    report('no_noise_in_eval', ok, 'testing (eval) adds no noise')

    cs = nz['model'].checksums
    ok = len(cs) >= 2 and len(set(cs)) == len(cs)
    res['fresh_draws'] = {'checksums': cs, 'pass': ok}
    report('fresh_draws', ok, f'fresh noise every batch (first {len(cs)} draws all different)')

    # The Layer-2 training step is a copy of the original plus one line. With mu = 0
    # that line does nothing, so training must give exactly the original weights.
    o1, _ = run_epoch_on_subset(x, y, False, None)
    o2, _ = run_epoch_on_subset(x, y, False, None)
    l0, _ = run_epoch_on_subset(x, y, True, 0.0)
    pairs = [('model', lambda a: a.model)] + ([('model_emp', lambda a: a.model_emp)] if IS_AFEC else [])
    for name, get in pairs:
        floor = max_param_diff(get(o1), get(o2))
        d0 = max_param_diff(get(o1), get(l0))
        dmu = max_param_diff(get(o1), get(appr))
        ok = d0 <= floor
        res[f'copy_identical_{name}'] = {'diff_original_vs_original': floor, 'diff_original_vs_layer2_mu0': d0,
                                         'diff_original_vs_layer2_mu': dmu, 'pass': ok}
        report(f'copy_identical_{name}', ok,
               f'{name}: Layer-2 training step with mu=0 == original (max weight diff {d0:.2e}, '
               f'original-vs-original {floor:.2e})')
        print(f'      (sanity: with mu={args.mu_rand}/255 the weights do change: max diff {dmu:.2e})')

    # train(): the Layer-2 copy of the whole task training, with no defense attached,
    # must give exactly the weights of the original train().
    def full_train(layer2, defense=None):
        ap = build_appr(layer2=layer2)
        ap.nepochs = 2
        if defense is not None:
            defense.attach(ap)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ap.train(T, x, y, XVALID, YVALID, DATA, None, None)
        return ap

    t1, t2, t3 = full_train(False), full_train(False), full_train(True)
    for name, get in pairs:
        floor = max_param_diff(get(t1), get(t2))
        d0 = max_param_diff(get(t1), get(t3))
        ok = d0 <= floor
        res[f'train_copy_identical_{name}'] = {'diff_original_vs_original': floor, 'diff_original_vs_layer2': d0, 'pass': ok}
        report(f'train_copy_identical_{name}', ok,
               f'{name}: Layer-2 train() with no defense == original train() (max weight diff {d0:.2e}, '
               f'original-vs-original {floor:.2e})')

    # the defense in train(): eta computed ONCE, at the warm-up epoch, within its cap;
    # random noise on every step of every epoch.
    fd = FriendsDefense(zeta255=args.zeta, mu255=args.mu_rand, kind=args.noise_kind, warmup_epochs=1,
                        friendly_epochs=min(args.friendly_epochs, 5), friendly_lr=args.friendly_lr,
                        friendly_lambda=args.friendly_lambda, seed=args.seed, device=DEVICE)
    ap = full_train(True, fd)
    ok = fd.log['calls'] == 1 and fd.log.get('epoch') == 1
    report('eta_once_at_warmup', ok, f'friendly noise computed {fd.log["calls"]} time(s), at epoch {fd.log.get("epoch")} '
                                    f'(expected once, at epoch 1)')
    ok = fd.log.get('linf_over_zeta', 9) <= 1 + 1e-6
    report('eta_within_cap', ok, f'friendly noise within its cap: Linf/zeta = {fd.log.get("linf_over_zeta", float("nan")):.4f}')
    exp_calls = 2 * steps
    for k, obj in [('model', ap.random_noise)] + ([('model_emp', ap.random_noise_emp)] if IS_AFEC else []):
        report(f'{k}_noise_all_epochs', obj.calls == exp_calls,
               f'{k}: random noise on every step of both epochs = {obj.calls} batches (expected {exp_calls})')
    # random noise held back until epoch 1 (--random_after_warmup): on for epoch 1 only
    fd2 = FriendsDefense(zeta255=args.zeta, mu255=args.mu_rand, kind=args.noise_kind, warmup_epochs=1,
                         friendly_epochs=min(args.friendly_epochs, 5), friendly_lr=args.friendly_lr,
                         friendly_lambda=args.friendly_lambda, random_start_epoch=1, seed=args.seed, device=DEVICE)
    ap2 = full_train(True, fd2)
    report('random_from_epoch1', ap2.random_noise.calls == steps,
           f'random noise held back until epoch 1: {ap2.random_noise.calls} noisy batches (expected {steps}, epoch 1 only)')
    # BatchNorm control, head-only epochs, per-epoch log: one epoch of train() on the Layer-2 learner
    def bn_state(model):
        return [(m_.running_mean.clone(), m_.running_var.clone()) for m_ in model.modules()
                if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm)]

    def bn_diff(m1, m2):
        return max(max((a_ - c_).abs().max().item(), (b_ - d_).abs().max().item())
                   for (a_, b_), (c_, d_) in zip(bn_state(m1), bn_state(m2)))

    def body_diff(m1, m2):
        return max((p1 - p2).abs().max().item() for (n_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters())
                   if 'heads' not in n_)

    def head_diff(m1, m2):
        return max((p1 - p2).abs().max().item() for (n_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters())
                   if f'heads.{T}.' in n_)

    def one_epoch(**kw):
        start_ap, ap = build_appr(layer2=True), build_appr(layer2=True)
        FriendsDefense(use_friendly=False, use_random=False, warmup_epochs=0, seed=args.seed, device=DEVICE,
                       **kw).attach(ap)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            ap.train(T, x, y, XVALID, YVALID, DATA, None, None)   # nepochs = 1
        return start_ap, ap, out.getvalue()

    s0, a0, _ = one_epoch()
    report('bn_normal_updates', bn_diff(s0.model, a0.model) > 0,
           'normal training updates the BatchNorm statistics (sanity: they change)')
    for mode in ('eval', 'nostats'):
        s1, a1, _ = one_epoch(bn_mode=mode)
        d_bn, d_w = bn_diff(s1.model, a1.model), body_diff(s1.model, a1.model)
        report(f'bn_{mode}', d_bn == 0 and d_w > 0,
               f"bn_mode='{mode}': BatchNorm statistics unchanged ({d_bn:.1e}), weights still train ({d_w:.1e})")
    s2, a2, _ = one_epoch(head_warmup_epochs=1)   # 1 epoch, all of it head-only
    ok = body_diff(s2.model, a2.model) == 0 and bn_diff(s2.model, a2.model) == 0 and head_diff(s2.model, a2.model) > 0
    if IS_AFEC:
        ok = ok and body_diff(s2.model_emp, a2.model_emp) == 0 and bn_diff(s2.model_emp, a2.model_emp) == 0
    report('head_only', ok, f'head-only epoch: body and BatchNorm exactly unchanged ({body_diff(s2.model, a2.model):.1e}), '
                            f'new head trained ({head_diff(s2.model, a2.model):.1e})')
    _, _, text = one_epoch(log_old_acc=True)
    nlines = text.count('[LAYER2] epoch 0:')
    report('epoch_log', nlines == 1, f'per-epoch accuracy log: {nlines} line(s) for 1 epoch (expected 1)')
    res['pass'] = ok_all
    return res


# ----------------------------------------------------------------------------
# V5
# ----------------------------------------------------------------------------

def body_gradient(model, xb, yb):
    """The 'lesson' of one batch: how the training loss wants to move the shared
    body's weights (heads excluded). Only the input-dependent part of the
    victim's loss (cross-entropy on head T) -- the CL penalty does not depend
    on the images, so it is left out."""
    loss = F.cross_entropy(model(xb)[T], yb)
    params = [p for n_, p in model.named_parameters() if 'heads' not in n_]
    grads = torch.autograd.grad(loss, params)
    return torch.cat([g.flatten() for g in grads])


def lesson_similarity(start_model, x, y, eta, ctrl, n, bs=128):
    m = deepcopy(start_model)
    m.train()
    cos_eta, cos_ctrl = [], []
    for lo in range(0, min(n, len(x)), bs):
        xb, yb = x[lo:lo + bs].to(DEVICE), y[lo:lo + bs].to(DEVICE)
        g0 = body_gradient(m, xb, yb)
        ge = body_gradient(m, torch.clamp(xb + eta[lo:lo + bs].to(DEVICE), 0, 1), yb)
        gc = body_gradient(m, torch.clamp(xb + ctrl[lo:lo + bs].to(DEVICE), 0, 1), yb)
        cos_eta.append(F.cosine_similarity(g0, ge, dim=0).item())
        cos_ctrl.append(F.cosine_similarity(g0, gc, dim=0).item())
    return float(np.mean(cos_eta)), float(np.mean(cos_ctrl))


@torch.no_grad()
def old_task_eval(model):
    """Plain accuracy and cross-entropy on the test sets (no CL penalty in the loss)."""
    model.eval()
    accs, losses = [], []
    for u in range(T + 1):
        xt, yt = DATA[u]['test']['x'], DATA[u]['test']['y']
        correct = loss = 0.0
        for lo in range(0, len(xt), 500):
            out = model(xt[lo:lo + 500].to(DEVICE))[u]
            tgt = yt[lo:lo + 500].to(DEVICE)
            loss += F.cross_entropy(out, tgt, reduction='sum').item()
            correct += (out.argmax(1) == tgt).float().sum().item()
        accs.append(100 * correct / len(xt))
        losses.append(loss / len(xt))
    return {'old_acc': float(np.mean(accs[:T])), 'old_loss': float(np.mean(losses[:T])), 'taskT_acc': accs[T]}


def short_training(start_appr, x_in, y):
    appr = deepcopy(start_appr)
    fresh_optimizers(appr)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for e in range(args.lesson_epochs):
        with contextlib.redirect_stderr(io.StringIO()):
            train_one_epoch(appr, x_in, y, T, e)
    return old_task_eval(appr.model)


def check_v5():
    print('\n' + '=' * 78 + '\nV5  does the friendly noise change what the model LEARNS from the poisoned images?\n'
          + '=' * 78)
    if args.variant != 'poison':
        print('  skipped: needs the poisoned data (--noise_pkl)')
        return {'skipped': True}
    n = args.v2_n or len(XTRAIN)
    x, y, xc = XTRAIN[:n].cpu(), YTRAIN[:n].cpu(), XCLEAN[:n].cpu()
    out = {}

    print(f'\n  Part 1 - "lesson" similarity. For {min(args.lesson_grad_n, n)} poisoned images: does adding the noise change')
    print('  the direction training would push the shared body?  1.00 = same lesson, lower = more scrambled.')
    print(f'      {"option":8s} {"eta":>8s} {"same-size random":>18s}')
    for v in [v for v in ('c', 'a', 'b') if v in ETAS]:
        ctrl = controls(ETAS[v])['sign_shuffled']
        ce, cc = lesson_similarity(START_MODELS[v].model, x, y, ETAS[v], ctrl, args.lesson_grad_n)
        out.setdefault(v, {})['lesson_cos'] = {'eta': ce, 'same_size_random': cc}
        print(f'      ({v})      {ce:8.3f} {cc:18.3f}')

    print(f'\n  Part 2 - real training: {args.lesson_epochs} epoch(s) on task {T}, then accuracy on old tasks 0-{T - 1}.')
    print('  (random noise r is OFF here, to see what eta alone does)')
    starts = {}
    for v in [v for v in ('c', 'a', 'b') if v in ETAS]:
        key = 'warm-up model (b)' if v == 'b' else 'checkpoint'
        starts.setdefault(key, []).append(v)
    t_all = time.time()
    for start_name, vs in starts.items():
        start = START_MODELS[vs[0]]
        base = old_task_eval(start.model)
        res = {'before': base, 'clean': short_training(start, xc, y), 'poison': short_training(start, x, y)}
        for v in vs:
            res[f'eta_{v}'] = short_training(start, torch.clamp(x + ETAS[v], 0, 1), y)
            res[f'ctrl_{v}'] = short_training(start, torch.clamp(x + controls(ETAS[v])['sign_shuffled'], 0, 1), y)

        print(f'\n    start model: {start_name}')
        print(f'      {"trained on":36s} {"old-task acc":>12s} {"change":>8s} {"task-" + str(T) + " acc":>11s}')
        labels = [('before', 'nothing (before training)'), ('clean', 'clean images (reference only)'),
                  ('poison', 'poisoned images')]
        for v in vs:
            labels += [(f'eta_{v}', f'poisoned + eta ({v})'), (f'ctrl_{v}', f'poisoned + same-size random ({v})')]
        for k, lab in labels:
            r = res[k]
            ch = '' if k == 'before' else f'{r["old_acc"] - base["old_acc"]:+8.1f}'
            print(f'      {lab:36s} {r["old_acc"]:11.1f}% {ch:>8s} {r["taskT_acc"]:10.1f}%')

        damage = res['clean']['old_acc'] - res['poison']['old_acc']
        print(f'      -> the poison\'s extra damage (clean minus poisoned) = {damage:.1f} pts')
        judgeable = damage >= 1.0
        if not judgeable:
            print(f'      -> under 1 point: too little poison damage after {args.lesson_epochs} epoch(s) to judge; '
                  f're-run with more --lesson_epochs')
        for v in vs:
            fe = (res[f'eta_{v}']['old_acc'] - res['poison']['old_acc']) / damage if damage > 0 else float('nan')
            fc = (res[f'ctrl_{v}']['old_acc'] - res['poison']['old_acc']) / damage if damage > 0 else float('nan')
            out.setdefault(v, {})['training'] = {k: res[k] for k in ('before', 'clean', 'poison', f'eta_{v}', f'ctrl_{v}')}
            out[v]['damage_removed'] = {'eta': fe, 'same_size_random': fc, 'poison_damage_pts': damage}
            if judgeable:
                print(f'      -> ({v}) eta removes {100 * fe:.0f}% of that damage; '
                      f'same-size random removes {100 * fc:.0f}%   (100% = as if no poison, 0% = no help)')
    print(f'\n  [{time.time() - t_all:.0f}s]')
    return out



# ----------------------------------------------------------------------------
# V6
# ----------------------------------------------------------------------------

def train_variant(x_in, y, mu255=None):
    """Fresh Layer-2 learner from the checkpoint, --lesson_epochs of training on x_in,
    optionally with random noise r of size mu255/255 on every step."""
    ap = build_appr(layer2=True)
    fresh_optimizers(ap)
    if mu255:
        ap.random_noise = RandomNoise(mu255 / 255, args.noise_kind, seed=args.seed)
        if IS_AFEC:
            ap.random_noise_emp = RandomNoise(mu255 / 255, args.noise_kind, seed=args.seed + 1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for e in range(args.lesson_epochs):
        with contextlib.redirect_stderr(io.StringIO()):
            train_one_epoch(ap, x_in, y, T, e)
    return ap


def with_checkpoint_bn(model, start_model):
    """Copy of `model` whose BatchNorm running statistics are put back to the checkpoint's."""
    m = deepcopy(model)
    for a, b in zip(m.modules(), start_model.modules()):
        if isinstance(a, torch.nn.modules.batchnorm._BatchNorm):
            a.running_mean.copy_(b.running_mean)
            a.running_var.copy_(b.running_var)
            a.num_batches_tracked.copy_(b.num_batches_tracked)
    return m


def check_v6():
    print('\n' + '=' * 78 + '\nV6  how much of the forgetting is BatchNorm statistics, and what does random noise do?\n'
          + '=' * 78)
    if args.variant != 'poison':
        print('  skipped: needs the poisoned data (--noise_pkl)')
        return {'skipped': True}
    n = args.v2_n or len(XTRAIN)
    x, y, xc = XTRAIN[:n].cpu(), YTRAIN[:n].cpu(), XCLEAN[:n].cpu()
    start = build_appr(layer2=True)
    base = old_task_eval(start.model)
    print(f'  The model keeps running averages of its inputs (BatchNorm statistics). They are not weights, so the')
    print(f'  continual-learning penalty (EWC etc.) does not protect them. Train task {T} for {args.lesson_epochs} epoch(s),')
    print(f'  then test old tasks 0-{T - 1} twice: as the model is, and with those statistics put back to the checkpoint\'s.')
    print(f'  Checkpoint: old-task acc {base["old_acc"]:.1f}%.   (random noise r: Bernoulli, off unless stated)\n')
    print(f'      {"trained on":34s} {"old acc":>8s} {"with checkpoint BN":>19s} {"gain":>7s}  {"task-" + str(T) + " acc":>10s}')
    runs = [('clean images (reference only)', xc, None), ('poisoned images', x, None)]
    runs += [(f'poisoned + random noise r, mu={m:g}/255', x, m) for m in args.v6_mus]
    out = {'checkpoint': base}
    for name, x_in, mu in runs:
        ap = train_variant(x_in, y, mu)
        r1 = old_task_eval(ap.model)
        r2 = old_task_eval(with_checkpoint_bn(ap.model, start.model))
        out[name] = {'as_is': r1, 'checkpoint_bn': r2}
        print(f'      {name:34s} {r1["old_acc"]:7.1f}% {r2["old_acc"]:18.1f}% {r2["old_acc"] - r1["old_acc"]:+6.1f}  {r1["taskT_acc"]:9.1f}%')
    p = out['poisoned images']
    lost = base['old_acc'] - p['as_is']['old_acc']
    back = p['checkpoint_bn']['old_acc'] - p['as_is']['old_acc']
    if lost > 1:
        print(f'\n  Poison: {lost:.1f} points of old-task accuracy lost; restoring only the BatchNorm statistics gives back '
              f'{back:.1f} ({100 * back / lost:.0f}%).')
    return out



# ----------------------------------------------------------------------------
# V7
# ----------------------------------------------------------------------------

def check_v7():
    print('\n' + '=' * 78 + '\nV7  does the noise touch what the poison encodes in the training signal?\n' + '=' * 78)
    if args.variant != 'poison':
        print('  skipped: needs the poisoned data (--noise_pkl)')
        return {'skipped': True}
    n = min(args.v7_n, len(XTRAIN))
    bs = 128
    xc, xp, y = XCLEAN[:n].cpu(), XTRAIN[:n].cpu(), YTRAIN[:n].cpu()
    start = build_appr(layer2=True)
    m = deepcopy(start.model)
    m.train()
    gen = torch.Generator().manual_seed(args.seed)

    def bern(mu255):
        return ((torch.rand(xc.shape, generator=gen) > 0.5).float() * 2 - 1) * (mu255 / 255)

    eta = generate_friendly_noise(start.model, xp, task=T, target='head', loss_fn='KL',
                                  friendly_epochs=args.friendly_epochs, lam=args.friendly_lambda,
                                  friendly_lr=args.friendly_lr, clamp=ZETA, batch_size=args.friendly_batch,
                                  device=DEVICE)
    noises = {f'random noise r, mu={mu:g}/255': bern(mu) for mu in args.v7_mus}
    noises['friendly noise eta (from the checkpoint)'] = eta
    noises[f'eta + random noise r, mu={args.mu_rand:g}/255'] = eta + bern(args.mu_rand)

    delta = xp - xc
    sign = (torch.rand(delta.shape, generator=gen) > 0.5).float() * 2 - 1
    xp_other = torch.clamp(xc + delta.abs() * sign, 0, 1)          # an unrelated perturbation of the same size

    def grad(xb, yb):
        return body_gradient(m, xb.to(DEVICE), yb.to(DEVICE))

    cos = {k: [] for k in noises}
    ratio = {k: [] for k in noises}
    cos_other, norm_ratio_other = [], []
    for lo in range(0, n, bs):
        sl = slice(lo, lo + bs)
        g_c, g_p, yb = grad(xc[sl], y[sl]), grad(xp[sl], y[sl]), y[sl]
        d_clean = g_p - g_c                                      # the poison's gradient signature
        cos_other.append(F.cosine_similarity(d_clean, grad(xp_other[sl], yb) - g_c, dim=0).item())
        for k, nz in noises.items():
            d_noisy = grad(torch.clamp(xp[sl] + nz[sl], 0, 1), yb) - grad(torch.clamp(xc[sl] + nz[sl], 0, 1), yb)
            cos[k].append(F.cosine_similarity(d_clean, d_noisy, dim=0).item())
            ratio[k].append((d_noisy.norm() / d_clean.norm()).item())
    print(f'  Signature = (training signal on the poisoned images) - (training signal on the same images without poison),')
    print(f'  for the shared body, on {n} images. The same noise is added to both. cosine 1.00 = the poison\'s signature')
    print(f'  is untouched by the noise; the first row is the reference for "unrelated".\n')
    print(f'      {"noise added to both":44s} {"cosine":>7s} {"size ratio":>11s}')
    ref = float(np.mean(cos_other))
    print(f'      {"(reference) unrelated perturbation, same size":44s} {ref:7.3f} {"":>11s}')
    out = {'unrelated_reference_cos': ref}
    for k in noises:
        out[k] = {'cos': float(np.mean(cos[k])), 'size_ratio': float(np.mean(ratio[k]))}
        print(f'      {k:44s} {out[k]["cos"]:7.3f} {out[k]["size_ratio"]:11.2f}')
    return out


# ----------------------------------------------------------------------------

def main():
    results = {'config': {k: v for k, v in vars(args).items()}, 'method': approach_name,
               'experiment': experiment, 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}

    appr = build_appr()
    accs = task_accs(appr, T - 1)
    print(f'[load] checkpoint old-task acc: {" ".join(f"{a:.1f}" for a in accs)}  (mean {np.mean(accs):.1f}%)')
    results['checkpoint_old_task_acc'] = accs

    if 'v1' in args.checks:
        results['v1'] = check_v1(appr)
    if 'v2' in args.checks or 'v5' in args.checks:
        results['v2'] = check_v2(appr)
    if 'v4' in args.checks:
        results['v4'] = check_v4()
    if 'v5' in args.checks:
        results['v5'] = check_v5()
    if 'v6' in args.checks:
        results['v6'] = check_v6()
    if 'v7' in args.checks:
        results['v7'] = check_v7()

    out = args.out or os.path.join(HERE, 'results', f'validation_{experiment}_{approach_name}_{args.variant}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\n[done] results -> {out}')


if __name__ == '__main__':
    main()