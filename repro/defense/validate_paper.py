#!/usr/bin/env python3
"""Validate the PureVQ-GAN re-implementation in the PAPER's own setting (CIFAR-10, ResNet-18, GM/BP/NS poisons),
before using it against BrainWash.  Reports natural accuracy and poison success rate (PSR).

Input: a poisoned-dataset .npz (build it from the official attack code, see repro/defense/VALIDATION.md):
  x_train (N,32,32,3 uint8, POISONED)   y_train (N,)     x_test (M,32,32,3 uint8)   y_test (M,)
  attack  'gm' | 'bp' | 'ns'
  gm/bp : target_x (T,32,32,3 uint8)  target_y (T,) true labels   adv_y (T,) adversarial labels
  ns    : trigger (32,32,3 float, ADDITIVE in [0,1] units, already multiplied by the test-time scale)  target_class (int)

Stages (each prints one JSON line, so it can be grepped / compared):
  --stage purifier : purifier health on CLEAN test images (PSNR, codes used). No classifier needed.
  --stage classify : train ResNet-18 on the (optionally purified) train set; report acc + PSR.

Examples
  python repro/defense/validate_paper.py --stage purifier --purifier P/purifier.pt --poison_npz ns.npz
  python repro/defense/validate_paper.py --stage classify --poison_npz ns.npz --tag undefended
  python repro/defense/validate_paper.py --stage classify --poison_npz ns.npz --purifier P/purifier.pt --tag pvq
  python repro/defense/validate_paper.py --stage classify --poison_npz ns.npz --clean_train_npz clean.npz --tag clean_ref
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from purevqgan import load_purifier, purify_tensor  # noqa: E402
from purevqgan.purify import code_indices  # noqa: E402

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def to_t(x):  # uint8 NHWC -> float NCHW in [0,1]
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255


def resnet18_cifar(num_classes=10):
    import torchvision
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # standard CIFAR stem
    m.maxpool = nn.Identity()
    return m


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = resnet18_cifar()
        self.register_buffer('mean', MEAN.clone()); self.register_buffer('std', STD.clone())

    def forward(self, x01):
        return self.m((x01 - self.mean) / self.std)


@torch.no_grad()
def predict(net, x, dev, bs=1000):
    net.eval()
    return torch.cat([net(x[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(x), bs)])


def train_classifier(x, y, epochs, bs, lr, dev, seed):
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb, yb = x[idx].to(dev), y[idx].to(dev)
            flip = torch.rand(len(xb), device=dev) < 0.5                      # standard CIFAR augmentation
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            pad = F.pad(xb, (4, 4, 4, 4), mode='reflect')
            dx, dy = np.random.randint(0, 9, 2)
            xb = pad[:, :, dy:dy + 32, dx:dx + 32]
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return net


def psr(net, d, dev, tf=lambda x: x):
    if str(d['attack']) in ('gm', 'bp'):
        pred = predict(net, tf(to_t(d['target_x'])), dev)
        adv = torch.from_numpy(d['adv_y']).long()
        return {'psr': 100 * (pred == adv).float().mean().item(), 'n_targets': len(adv)}
    tc = int(d['target_class'])
    keep = d['y_test'] != tc
    x = (to_t(d['x_test'][keep]) + torch.from_numpy(d['trigger']).permute(2, 0, 1).float()).clamp(0, 1)
    return {'psr': 100 * (predict(net, tf(x), dev) == tc).float().mean().item(), 'n_targets': int(keep.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['purifier', 'classify'])
    ap.add_argument('--poison_npz', required=True)
    ap.add_argument('--clean_train_npz', help='npz with x_train/y_train: clean reference run (no attack, no defense)')
    ap.add_argument('--purifier')
    ap.add_argument('--purify_test', action='store_true', help='also purify test/target/trigger inputs (paper Sec. 3.3 inference)')
    ap.add_argument('--passes', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--out', default='repro/defense/results_paper')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = dict(np.load(a.poison_npz, allow_pickle=True))
    res = {'tag': a.tag, 'stage': a.stage, 'attack': str(d['attack']), 'seed': a.seed}

    G = load_purifier(a.purifier, dev) if a.purifier else None
    if a.stage == 'purifier':
        assert G is not None, '--purifier required'
        xt = to_t(d['x_test'])
        p = purify_tensor(G, xt, a.passes)
        mse = F.mse_loss(p, xt, reduction='none').flatten(1).mean(1)
        idx = code_indices(G, xt)
        res.update(psnr_clean_test=(10 * torch.log10(1 / mse.clamp_min(1e-10))).mean().item(),
                   codes_used=int(idx.unique().numel()), num_codes=G.quantizer.num_codes,
                   params_M=sum(q.numel() for q in G.parameters()) / 1e6)
        xtr, xp = to_t(d['x_train']), None
        if 'x_train_clean' in d:   # if the attack code kept the clean version, measure perturbation survival
            from purevqgan.purify import perturbation_report
            res['perturbation'] = perturbation_report(G, to_t(d['x_train_clean']), xtr, a.passes)
    else:
        if a.clean_train_npz:
            c = np.load(a.clean_train_npz)
            xtr, ytr = to_t(c['x_train']), torch.from_numpy(c['y_train']).long()
        else:
            xtr, ytr = to_t(d['x_train']), torch.from_numpy(d['y_train']).long()
        if G is not None:
            t0 = time.time(); xtr = purify_tensor(G, xtr, a.passes)
            res['purify_sec'] = round(time.time() - t0, 1)
        t0 = time.time()
        net = train_classifier(xtr, ytr, a.epochs, a.batch_size, a.lr, dev, a.seed)
        tf = (lambda x: purify_tensor(G, x, a.passes)) if (G is not None and a.purify_test) else (lambda x: x)
        xte = tf(to_t(d['x_test']))
        res['purify_test'] = bool(G is not None and a.purify_test)
        res['nat_acc'] = 100 * (predict(net, xte, dev) == torch.from_numpy(d['y_test']).long()).float().mean().item()
        res.update(psr(net, d, dev, tf))
        res['train_sec'] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}_{res['attack']}_{a.stage}_s{a.seed}.json"), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
