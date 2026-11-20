"""
friends_p1.py  --  P1: friendly noise that DISTURBS the shared features while keeping head T's prediction.

A variant of FRIENDS (not the paper's method; friends.py is left untouched and remains the official port).

Why: BrainWash (cautious) keeps the new task learnable (head T) and makes the shared body drift, which is what
the old heads read. The only defender-side fact used here is that in a multi-head model forgetting can only
come through the shared body (old heads never get gradients from the new task); nothing about the attack's
procedure is used.

FRIENDS' friendly noise:   minimise  KL(head T(x+eta) || head T(x))  -  lambda * mean(eta^2)
P1 noise:                  minimise  KL(head T(x+eta) || head T(x))  -  beta * relMSE(phi(x+eta), phi(x))
                                                                       -  lambda * mean(eta^2)      (lambda default 0)
phi = the shared 1024-d embedding every head reads; relMSE = MSE / mean(phi(x)^2) (scale-free).
So eta keeps head T's prediction (KL term) while pushing the shared features as far as possible (beta term),
within |eta| <= zeta per pixel. The optimizer is the same as FRIENDS' (SGD, lr 100, Nesterov momentum 0.9,
30 steps, init U(+-8/255), clamp inside the computation, final clamp).

Logged per run (printed and in defense.log): mean|eta|/zeta, share at the cap, head-T prediction agreement
with and without eta (%), and the relative feature change -- check that agreement stays high (~99%) and the
feature change is clearly larger than option (b)'s.
"""
import time

import torch
import torch.nn.functional as F

from friends import FriendsDefense, model_output


def generate_feature_disrupting_noise(model, x, *, task, beta=1.0, lam=0.0, friendly_epochs=30, friendly_lr=100.,
                                      friendly_momentum=0.9, clamp=16 / 255, init_mag_255=8, batch_size=128,
                                      device='cuda'):
    """P1 noise for every image in x ([N,C,H,W] in [0,1]). Returns (eta on CPU in [-clamp, clamp], stats)."""
    kl_fn = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    was_training = model.training
    model.eval()
    n = x.shape[0]
    noise = torch.zeros(x.shape, device='cpu')
    agree, feat_chg, kl_end = 0., 0., 0.

    for lo in range(0, n, batch_size):
        inputs = x[lo:lo + batch_size].to(device)
        init = (torch.rand(*(inputs.shape)) - 0.5) * 2 * init_mag_255 / 255
        eta = init.to(device).requires_grad_(True)
        opt = torch.optim.SGD([eta], lr=friendly_lr, momentum=friendly_momentum, nesterov=friendly_momentum > 0)
        with torch.no_grad():
            logp0 = F.log_softmax(model_output(model, inputs, 'head', task), dim=1)
            f0 = model_output(model, inputs, 'features', task)
            scale = f0.pow(2).mean().clamp_min(1e-12)
        for _ in range(friendly_epochs):
            eta_c = torch.clamp(eta, -clamp, clamp)
            xp = torch.clamp(inputs + eta_c, 0, 1)
            kl = kl_fn(F.log_softmax(model_output(model, xp, 'head', task), dim=1), logp0)
            feat = F.mse_loss(model_output(model, xp, 'features', task), f0) / scale
            loss = kl - beta * feat - lam * torch.mean(torch.square(eta_c))
            opt.zero_grad()
            loss.backward()
            model.zero_grad()
            opt.step()
        with torch.no_grad():
            eta_c = torch.clamp(eta, -clamp, clamp)
            xp = torch.clamp(inputs + eta_c, 0, 1)
            lp = F.log_softmax(model_output(model, xp, 'head', task), dim=1)
            fp = model_output(model, xp, 'features', task)
            b = inputs.shape[0]
            agree += (lp.argmax(1) == logp0.argmax(1)).float().sum().item()
            feat_chg += ((fp - f0).pow(2).sum(1) / f0.pow(2).sum(1).clamp_min(1e-12)).sqrt().sum().item()
            kl_end += kl_fn(lp, logp0).item() * b
        noise[lo:lo + batch_size] = eta.detach().cpu()

    noise = torch.clamp(noise, -clamp, clamp)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return noise, {'headT_agreement_pct': 100 * agree / n, 'feature_change_rel': feat_chg / n, 'headT_kl': kl_end / n}


class FriendsDefenseP1(FriendsDefense):
    """FriendsDefense with the friendly noise replaced by P1 noise (everything else -- warm-up, random noise,
    BatchNorm and schedule options -- is inherited unchanged)."""

    def __init__(self, *a, p1_beta=1.0, p1_lambda=0.0, **k):
        super().__init__(*a, **k)
        self.p1_beta, self.p1_lambda = p1_beta, p1_lambda

    def add_friendly_noise(self, appr, xtrain, t, data, epoch):
        t0 = time.time()
        old = []
        for u in range(t):
            xt, yt = data[u]['test']['x'].cuda(), data[u]['test']['y'].cuda()
            old.append(100 * float(appr.eval(u, xt, yt)[1]))
        eta, st = generate_feature_disrupting_noise(
            appr.model, xtrain, task=t, beta=self.p1_beta, lam=self.p1_lambda, friendly_epochs=self.friendly_epochs,
            friendly_lr=self.friendly_lr, friendly_momentum=self.friendly_momentum, clamp=self.zeta,
            batch_size=self.friendly_batch, device=self.device)
        x_new = torch.clamp(xtrain + eta.to(xtrain.device), 0, 1)
        a = eta.abs()
        self.log = {'calls': self.log['calls'] + 1, 'epoch': epoch, 'seconds': time.time() - t0,
                    'old_task_acc_at_eta': float(sum(old) / len(old)) if old else float('nan'),
                    'eta_mean_over_zeta': a.mean().item() / self.zeta,
                    'eta_frac_at_bound': (a >= 0.99 * self.zeta).float().mean().item(),
                    'linf_over_zeta': a.max().item() / self.zeta,
                    'input_change_linf': (x_new - xtrain).abs().max().item(), **st}
        L = self.log
        print(f"[LAYER2] friendly noise computed at epoch {epoch} (P1: keep head T, disturb features; beta={self.p1_beta}, "
              f"lambda={self.p1_lambda}): mean|eta|/zeta={L['eta_mean_over_zeta']:.3f}  "
              f"at-bound={100 * L['eta_frac_at_bound']:.1f}%  head-T agreement={L['headT_agreement_pct']:.1f}%  "
              f"feature change={L['feature_change_rel']:.3f}  old-task acc at that moment={L['old_task_acc_at_eta']:.1f}%  "
              f"[{L['seconds']:.0f}s]")
        return x_new
