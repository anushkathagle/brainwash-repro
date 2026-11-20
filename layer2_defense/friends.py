"""
friends.py

FRIENDS (Liu, Yang & Mirzasoleiman, NeurIPS 2022) building blocks for the
Layer-2 defense against BrainWash. Two pieces:

  generate_friendly_noise(...)  the friendly component eta_i (computed once per task)
  RandomNoise(...)              the random component r (fresh every training step)

generate_friendly_noise is a PORT of the function with the same name in the
official repo (https://github.com/tianyu139/friendly-noise, file friendly_noise.py).
For a given model, input batch and RNG state it performs the identical
computation -- validate_friends.py --checks v1 asserts that numerically. The only
changes are generalizations the BrainWash pipeline needs:

  * no hard-coded 3x32x32 / 10-class buffers  (miniImageNet is 84x84, tiny is 64x64)
  * takes a tensor instead of a DataLoader     (stage 4 holds xtrain as one tensor)
  * multi-head model: `target` picks what must be preserved
        'head'     -> logits of head `task` (the paper's objective; loss 'KL' or 'MSE')
        'features' -> the shared 1024-d embedding before the heads   (variant (a))
  * one extra loss, 'MSE_rel' = MSE / mean(target^2): MSE made scale-free so the
    same lam (lambda) can be used on features as on logits.
  * no input normalization: the BrainWash victim consumes raw [0,1] pixels, so
    the official `transform` (Normalize) is the identity here.

Notation (thesis): eta = friendly noise (the official code and the FRIENDS paper
call it eps / epsilon; renamed because epsilon is the BrainWash poison budget
0.1 / 0.3 here), zeta = its per-pixel cap, lam = lambda, the weight of the
"make eta large" term (the official code calls it mu, which clashes with the
random-noise size mu), mu = random-noise size.

Faithfully kept from the official code (do not "fix" these without saying so in
the thesis -- they are part of what FRIENDS is):
  * objective: emp_risk - lam * mean(eta_clamped^2), SGD + Nesterov momentum 0.9
  * KL is KLDivLoss(log p(x+eta), log p(x)) = KL( p(x) || p(x+eta) ), i.e. the
    reverse direction of the paper's Eq. 3
  * init eta ~ U(-8/255, 8/255) per batch, drawn from the global torch RNG
  * the MultiStepLR scheduler is created but never stepped in the official loop,
    so the learning rate is constant; we keep it constant
  * returned noise is the unclamped iterate, clamped to [-clamp, clamp] at the end
"""

import time

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# model outputs
# ----------------------------------------------------------------------------

def model_output(model, x, target, task):
    """What the friendly noise must preserve, for the BrainWash multi-head ResNet."""
    if target == 'head':
        return model(x)[task]
    if target == 'features':
        # resnet.ResNet returns the pooled embedding when include_head is False
        old = model.include_head
        model.include_head = False
        try:
            return model(x)
        finally:
            model.include_head = old
    raise ValueError(f'unknown target {target!r}')


# ----------------------------------------------------------------------------
# friendly component
# ----------------------------------------------------------------------------

def generate_friendly_noise(model, x, *, task=-1, target='head', loss_fn='KL',
                            friendly_epochs=30, lam=1.0, friendly_lr=100.,
                            friendly_momentum=0.9, nesterov=True,
                            clamp=16 / 255, init_mag_255=8,
                            batch_size=128, device='cuda', verbose=False,
                            return_trace=False):
    """Friendly noise for every image in x ([N,C,H,W] in [0,1]).

    Returns a CPU tensor shaped like x, in [-clamp, clamp]. With
    return_trace=True also returns per-batch (first, last) values of the
    empirical-risk and constraint terms, which V2 uses to check that the
    optimization actually moved.
    """
    if loss_fn == 'KL':
        criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    elif loss_fn in ('MSE', 'MSE_rel'):
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f'unknown loss_fn {loss_fn!r}')

    was_training = model.training
    model.eval()

    n = x.shape[0]
    friendly_noise = torch.zeros(x.shape, device='cpu')
    trace = []

    for batch_idx, lo in enumerate(range(0, n, batch_size)):
        inputs = x[lo:lo + batch_size].to(device)
        # same draw as the official code: global RNG, shape of the batch
        # same arithmetic, in the same order, as the official `(torch.rand(...) - 0.5) * 2 * 8/255`:
        # a differently-rounded start (e.g. times a pre-computed 8/255) is enough to send the
        # (chaotic) optimization to a different result
        init = (torch.rand(*(inputs.shape)) - 0.5) * 2 * init_mag_255 / 255
        eta = init.to(device).requires_grad_(True)
        optimizer = torch.optim.SGD([eta], lr=friendly_lr, momentum=friendly_momentum,
                                    nesterov=nesterov)

        output_original = model_output(model, inputs, target, task)
        if loss_fn == 'KL':
            output_original = F.log_softmax(output_original, dim=1).detach()
        else:
            output_original = output_original.detach()
        scale = output_original.pow(2).mean().clamp_min(1e-12) if loss_fn == 'MSE_rel' else 1.0

        first = None
        for friendly_epoch in range(friendly_epochs):
            eta_clamp = torch.clamp(eta, -clamp, clamp)
            perturbed = torch.clamp(inputs + eta_clamp, 0, 1)
            output_perturb = model_output(model, perturbed, target, task)
            if loss_fn == 'KL':
                output_perturb = F.log_softmax(output_perturb, dim=1)

            emp_risk = criterion(output_perturb, output_original) / scale
            constraint = torch.mean(torch.square(eta_clamp))
            loss = emp_risk - lam * constraint

            optimizer.zero_grad()
            loss.backward()
            model.zero_grad()
            optimizer.step()

            if first is None:
                first = (emp_risk.item(), constraint.item())
            if verbose:
                print(f'  friendly batch {batch_idx} epoch {friendly_epoch}: '
                      f'emp_risk={emp_risk.item():.4g} constraint={constraint.item():.4g}',
                      end='\r', flush=True)

        trace.append({'first': first, 'last': (emp_risk.item(), constraint.item())})
        friendly_noise[lo:lo + batch_size] = eta.detach().cpu()

    if verbose:
        print()
    friendly_noise = torch.clamp(friendly_noise, -clamp, clamp)
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    if return_trace:
        return friendly_noise, trace
    return friendly_noise


# ----------------------------------------------------------------------------
# random component
# ----------------------------------------------------------------------------

def sample_random_noise(shape, mu, kind, generator, device, dtype=torch.float32):
    """Bernoulli {-mu,+mu}, Uniform [-mu,mu] or Gaussian N(0, mu^2) -- the paper's
    three FRIENDS-B / -U / -G options."""
    if kind == 'bernoulli':
        r = torch.rand(shape, generator=generator, device=device, dtype=dtype)
        return ((r > 0.5).to(dtype) * 2 - 1) * mu
    if kind == 'uniform':
        r = torch.rand(shape, generator=generator, device=device, dtype=dtype)
        return (r * 2 - 1) * mu
    if kind == 'gaussian':
        return torch.randn(shape, generator=generator, device=device, dtype=dtype) * mu
    raise ValueError(f'unknown random noise kind {kind!r}')


class RandomNoise:
    """The random component r: fresh random noise for every training batch.

    Called explicitly, on one marked line inside the Layer-2 copy of the
    victim's training step (layer2_defense/friends_training.py):

        images = self.random_noise(images)

    Each call draws new noise (Bernoulli: every pixel gets +mu or -mu), adds it
    and clips the result to [0, 1]. It has its own random-number generator, so
    the batch order of training (numpy shuffle) is exactly the same as in the
    undefended run. mu = 0 returns the images untouched.

    calls counts how many batches were noised; with track=True, checksums holds
    the noise sum of the first few calls (V4 uses it to check draws are fresh).
    """

    def __init__(self, mu, kind='bernoulli', seed=0, clamp01=True, track=False):
        self.mu, self.kind, self.seed, self.clamp01, self.track = mu, kind, seed, clamp01, track
        self.calls = 0
        self.checksums = []
        self._gen = None
        self.enabled = True  # FriendsDefense can hold the noise back until its start epoch

    def __call__(self, images):
        if self.mu == 0 or not self.enabled:
            return images
        if self._gen is None:
            self._gen = torch.Generator(device=images.device)
            self._gen.manual_seed(self.seed)
        noise = sample_random_noise(images.shape, self.mu, self.kind, self._gen, images.device, images.dtype)
        out = images + noise
        if self.clamp01:
            out = torch.clamp(out, 0, 1)
        self.calls += 1
        if self.track and len(self.checksums) < 4:
            self.checksums.append(float(noise.double().sum()))
        return out


# ----------------------------------------------------------------------------
# the defense: FRIENDS as the paper's Algorithm 2, for one continual-learning task
# ----------------------------------------------------------------------------

class FriendsDefense:
    """FRIENDS settings for one task, and the one step that needs the model.

    Paper's Algorithm 2 on the task being learned (the poisoned last task):
      * epochs 0 .. warmup_epochs-1 : train on x, plus fresh random noise r every step
      * at epoch == warmup_epochs   : compute the friendly noise eta ONCE, from the
                                      CURRENT model (head T, KL), for every training
                                      image; from here on the training images are x + eta
      * remaining epochs            : train on x + eta, plus fresh random noise r

    The three Layer-2 trainers (friends_training.py) call on_epoch_start() from one marked
    block at the top of their epoch loop; attach() gives them r. random_start_epoch > 0 holds
    r back until that epoch (e.g. = warmup_epochs: a plain warm-up, then eta and r together).

    use_friendly / use_random switch the two components off for the ablation
    ("friendly only" / "random only"; the paper matches the budgets: zeta=32 for
    friendly only, mu=32 for random only -- set them on the command line).

    Three extra options, all independent of the noise (use_friendly / use_random may both
    be False: then this object only carries these options):
      * bn_mode          'normal' | 'eval' | 'nostats'. BatchNorm control for the whole task:
                         'eval'    BatchNorm normalizes with its stored (checkpoint) statistics
                                   and never updates them;
                         'nostats' BatchNorm still normalizes with each batch's statistics (as the
                                   attacker's crafting model does) but never updates the stored ones.
      * head_warmup_epochs  the first epochs train ONLY the new task's head: the body (including
                         BatchNorm statistics) is frozen, so the poison cannot change it. Then
                         everything unfreezes. With warmup_epochs >= head_warmup_epochs, the friendly
                         noise is computed from a model whose new head has been trained and whose
                         body is still the checkpoint's.
      * log_old_acc      print old-task / current-task test accuracy after every epoch.

    log is filled when eta is computed (printed, and written to summary_layer2.tsv).
    """

    def __init__(self, zeta255=16, mu255=16, kind='bernoulli', warmup_epochs=2, friendly_epochs=30,
                 friendly_lr=100., friendly_lambda=1., friendly_batch=128, use_friendly=True,
                 use_random=True, random_start_epoch=0, bn_mode='normal', head_warmup_epochs=0,
                 log_old_acc=False, seed=0, device='cuda', friendly_target='head', friendly_loss=None,
                 friendly_momentum=0.9):
        self.zeta, self.mu = zeta255 / 255, mu255 / 255
        # what the friendly noise must leave unchanged: 'head' = the logits of the new task's head, KL (the paper,
        # options b and c); 'features' = the shared 1024-d embedding every head reads, relative MSE (option a)
        if friendly_target not in ('head', 'features'):
            raise ValueError(f'unknown friendly_target {friendly_target!r}')
        if friendly_loss is None:
            friendly_loss = 'KL' if friendly_target == 'head' else 'MSE_rel'
        if friendly_target == 'features' and friendly_loss == 'KL':
            raise ValueError('KL is for logits; use MSE or MSE_rel with friendly_target="features"')
        self.friendly_target, self.friendly_loss = friendly_target, friendly_loss
        self.friendly_momentum = friendly_momentum   # SGD momentum of the eta optimizer (official: 0.9)
        self.kind, self.warmup_epochs = kind, warmup_epochs
        self.friendly_epochs, self.friendly_lr, self.friendly_lambda = friendly_epochs, friendly_lr, friendly_lambda
        self.friendly_batch, self.use_friendly, self.use_random = friendly_batch, use_friendly, use_random
        self.random_start_epoch = random_start_epoch  # 0 = from the first epoch (the paper)
        self.bn_mode, self.head_warmup_epochs, self.log_old_acc = bn_mode, head_warmup_epochs, log_old_acc
        if bn_mode not in ('normal', 'eval', 'nostats'):
            raise ValueError(f'unknown bn_mode {bn_mode!r}')
        if head_warmup_epochs > 0 and use_friendly and head_warmup_epochs > warmup_epochs:
            raise ValueError('head_warmup_epochs must be <= warmup_epochs: the friendly noise is computed at '
                             'epoch warmup_epochs, after the head-only epochs')
        self.head_only_active = False
        self._bn_momentum = {}
        self.seed, self.device = seed, device
        self.log = {'calls': 0}

    def attach(self, appr):
        """Hand r (and this object) to a Layer-2 learner."""
        appr.friends = self
        if self.use_random:
            appr.random_noise = RandomNoise(self.mu, self.kind, seed=self.seed)
            if hasattr(appr, 'random_noise_emp'):                  # AFEC's second network
                appr.random_noise_emp = RandomNoise(self.mu, self.kind, seed=self.seed + 1)
            if self.random_start_epoch > 0:                        # held back until its start epoch
                appr.random_noise.enabled = False
                if hasattr(appr, 'random_noise_emp'):
                    appr.random_noise_emp.enabled = False

    def on_epoch_start(self, appr, xtrain, t, data, epoch):
        """Called at the top of every epoch by the Layer-2 train(): switches the random noise on at its
        start epoch (if it was held back) and computes the friendly noise once, after the warm-up."""
        if self.head_warmup_epochs > 0:
            if epoch == 0:
                self._set_body_trainable(appr, False)
                self.head_only_active = True
                print(f'[LAYER2] epochs 0..{self.head_warmup_epochs - 1}: only the new head trains '
                      f'(body and BatchNorm statistics frozen)')
            if epoch == self.head_warmup_epochs:
                self._set_body_trainable(appr, True)
                self.head_only_active = False
                print(f'[LAYER2] epoch {epoch}: body unfrozen')
        if self.use_random and self.random_start_epoch > 0 and epoch == self.random_start_epoch:
            appr.random_noise.enabled = True
            if hasattr(appr, 'random_noise_emp'):
                appr.random_noise_emp.enabled = True
            print(f'[LAYER2] random noise switched on at epoch {epoch}')
        if self.use_friendly and epoch == self.warmup_epochs:
            xtrain = self.add_friendly_noise(appr, xtrain, t, data, epoch)
        return xtrain

    @staticmethod
    def _set_body_trainable(appr, trainable):
        """Freeze / unfreeze everything except the heads, on the learner's network(s)."""
        for net in (appr.model, getattr(appr, 'model_emp', None)):
            if net is None:
                continue
            for n, p in net.named_parameters():
                if 'heads' not in n:
                    p.requires_grad_(trainable)

    def set_bn_mode(self, net):
        """Called right after net.train() at the top of every epoch of training: applies the BatchNorm
        control (and keeps BatchNorm fixed during head-only epochs)."""
        freeze = self.bn_mode in ('eval', 'nostats') or self.head_only_active
        use_stored = self.bn_mode == 'eval' or self.head_only_active
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                self._bn_momentum.setdefault(id(m), m.momentum)
                # momentum 0: the stored statistics are never updated, not even by the Fisher pass
                # that every method runs over the training data when the task ends
                m.momentum = 0.0 if freeze else self._bn_momentum[id(m)]
                if use_stored:
                    m.eval()

    def on_epoch_end(self, appr, t, data, epoch):
        """Called at the end of every epoch by the Layer-2 train(): optional accuracy log."""
        if not self.log_old_acc:
            return
        old = []
        for u in range(t):
            xt, yt = data[u]['test']['x'].cuda(), data[u]['test']['y'].cuda()
            old.append(100 * float(appr.eval(u, xt, yt)[1]))
        xt, yt = data[t]['test']['x'].cuda(), data[t]['test']['y'].cuda()
        cur = 100 * float(appr.eval(t, xt, yt)[1])
        print(f"[LAYER2] epoch {epoch}: old-task acc {sum(old) / max(len(old), 1):.2f}  task-{t} acc {cur:.2f}")

    def add_friendly_noise(self, appr, xtrain, t, data, epoch):
        """eta from the model as it is NOW; returns clamp(xtrain + eta, 0, 1)."""
        t0 = time.time()
        old = []
        for u in range(t):  # how much of the old tasks is already gone when eta is computed
            xt, yt = data[u]['test']['x'].cuda(), data[u]['test']['y'].cuda()
            old.append(100 * float(appr.eval(u, xt, yt)[1]))
        eta = generate_friendly_noise(appr.model, xtrain, task=t, target=self.friendly_target, loss_fn=self.friendly_loss,
                                      friendly_epochs=self.friendly_epochs, lam=self.friendly_lambda,
                                      friendly_lr=self.friendly_lr, friendly_momentum=self.friendly_momentum,
                                      nesterov=self.friendly_momentum > 0,   # official: Nesterov; momentum 0 = plain SGD
                                      clamp=self.zeta,
                                      batch_size=self.friendly_batch, device=self.device)
        x_new = torch.clamp(xtrain + eta.to(xtrain.device), 0, 1)
        a = eta.abs()
        self.log = {'calls': self.log['calls'] + 1, 'epoch': epoch, 'seconds': time.time() - t0,
                    'old_task_acc_at_eta': float(sum(old) / len(old)) if old else float('nan'),
                    'eta_mean_over_zeta': a.mean().item() / self.zeta,
                    'eta_frac_at_bound': (a >= 0.99 * self.zeta).float().mean().item(),
                    'linf_over_zeta': a.max().item() / self.zeta,
                    'input_change_linf': (x_new - xtrain).abs().max().item()}
        L = self.log
        print(f"[LAYER2] friendly noise computed at epoch {epoch} (after {epoch} warm-up epoch(s); "
              f"preserves {self.friendly_target}, loss {self.friendly_loss}): "
              f"mean|eta|/zeta={L['eta_mean_over_zeta']:.3f}  at-bound={100 * L['eta_frac_at_bound']:.1f}%  "
              f"Linf/zeta={L['linf_over_zeta']:.3f}  old-task acc at that moment={L['old_task_acc_at_eta']:.1f}%  "
              f"[{L['seconds']:.0f}s]")
        return x_new