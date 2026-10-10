"""Compact re-implementation of the Narcissus clean-label backdoor (Zeng et al., 2022, arXiv:2204.05255)
for CIFAR-10, written to produce the poisoned-dataset .npz that validate_paper.py consumes.

This is OUR implementation of the published algorithm, not the authors' code. It is only trustworthy
if the undefended PSR gate passes (VALIDATION.md rung 2). Algorithm:
  1. Attacker holds a small set of target-class images (= the poison budget) and a public out-of-distribution
     set (POOD). Train a surrogate ResNet-18 on POOD (its own labels) + target images (one extra label).
  2. Synthesize a trigger: optimize an additive L_inf-bounded noise so the surrogate classifies POOD images +
     noise as the target label (sign-gradient steps).
  3. Clean-label poisoning: add the trigger to the budgeted target-class training images (labels unchanged).
  4. Test time: add trigger * multiplier to non-target-class test images; PSR = fraction classified as target.
Deviations from the paper's code (state them): POOD = CIFAR-100 train (the authors use Tiny-ImageNet for CIFAR-10);
no trigger refinement stage; fixed surrogate/trigger schedules below.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_paper import Net, to_t  # noqa: E402


def load_cifar(root='./data'):
    import torchvision
    tr10 = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te10 = torchvision.datasets.CIFAR10(root, train=False, download=True)
    tr100 = torchvision.datasets.CIFAR100(root, train=True, download=True)
    return dict(x_train=tr10.data, y_train=np.array(tr10.targets), x_test=te10.data, y_test=np.array(te10.targets),
                x_pood=tr100.data, y_pood=np.array(tr100.targets))


def fake_cifar(n_train=600, n_test=200, n_pood=400, seed=0):
    r = np.random.default_rng(seed)
    im = lambda k: r.integers(0, 256, (k, 32, 32, 3), dtype=np.uint8)
    return dict(x_train=im(n_train), y_train=np.arange(n_train) % 10, x_test=im(n_test), y_test=np.arange(n_test) % 10,
                x_pood=im(n_pood), y_pood=np.arange(n_pood) % 20)


def _train(net, x, y, epochs, bs, lr, dev):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb, yb = x[idx].to(dev), y[idx].to(dev)
            flip = torch.rand(len(xb), device=dev) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        print(f'  [surrogate] epoch {ep + 1}/{epochs} loss {loss.item():.3f}', flush=True)


def build_narcissus(data, target_class=2, budget=500, eps=16 / 255, test_multiplier=3.0, surrogate_epochs=30,
                    trigger_steps=2000, trigger_bs=256, seed=0, dev=None):
    """Returns the dict to np.savez for validate_paper.py (attack='ns')."""
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ytr = data['y_train']
    tgt_idx = np.where(ytr == target_class)[0]
    poison_idx = rng.choice(tgt_idx, size=min(budget, len(tgt_idx)), replace=False)

    # --- 1. surrogate on POOD + the attacker's target-class images ---
    n_pood_cls = int(data['y_pood'].max()) + 1
    xs = torch.cat([to_t(data['x_pood']), to_t(data['x_train'][poison_idx])])
    ys = torch.cat([torch.from_numpy(data['y_pood']).long(),
                    torch.full((len(poison_idx),), n_pood_cls, dtype=torch.long)])   # extra "target" label
    sur = Net().to(dev)
    sur.m.fc = torch.nn.Linear(sur.m.fc.in_features, n_pood_cls + 1).to(dev)
    print(f'[narcissus] surrogate: {len(xs)} images, {n_pood_cls + 1} labels')
    _train(sur, xs, ys, surrogate_epochs, 128, 0.1, dev)
    sur.eval()
    for p in sur.parameters():
        p.requires_grad_(False)

    # --- 2. trigger synthesis: min_noise CE(f(x_pood + noise), target), |noise|_inf <= eps ---
    pood = to_t(data['x_pood'])
    noise = torch.zeros(1, 3, 32, 32, device=dev)
    tgt_lbl = torch.full((trigger_bs,), n_pood_cls, device=dev, dtype=torch.long)
    for s in range(trigger_steps):
        step = eps / 10 * (0.1 + 0.9 * (1 - s / trigger_steps))      # decaying step size
        xb = pood[torch.randint(0, len(pood), (trigger_bs,))].to(dev)
        noise.requires_grad_(True)
        loss = F.cross_entropy(sur((xb + noise).clamp(0, 1)), tgt_lbl[:len(xb)])
        g, = torch.autograd.grad(loss, noise)
        noise = (noise.detach() - step * g.sign()).clamp(-eps, eps)
        if s % 250 == 0 or s == trigger_steps - 1:
            with torch.no_grad():
                acc = (sur((xb + noise).clamp(0, 1)).argmax(1) == n_pood_cls).float().mean().item()
            print(f'  [trigger] step {s} loss {loss.item():.3f}  surrogate-target-rate {acc:.2f}', flush=True)
    trigger = noise.detach().cpu()[0].permute(1, 2, 0).numpy().astype(np.float32)   # (32,32,3), |.|<=eps

    # --- 3. clean-label poisoning ---
    x_clean = data['x_train'].copy()
    x_pois = x_clean.copy()
    x_pois[poison_idx] = (np.clip(x_clean[poison_idx].astype(np.float32) / 255 + trigger, 0, 1) * 255).round().astype(np.uint8)
    return dict(attack='ns', x_train=x_pois, x_train_clean=x_clean, y_train=ytr, x_test=data['x_test'],
                y_test=data['y_test'], trigger=(trigger * test_multiplier).astype(np.float32),
                target_class=target_class, poison_idx=poison_idx, eps=eps, test_multiplier=test_multiplier)
