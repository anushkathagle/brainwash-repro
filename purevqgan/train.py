"""Train a PureVQ-GAN purifier.

Objective (Sec. 3.2):  L_G = L_rec + L_VQ + lambda * L_GAN,   L_D = standard GAN discriminator loss.
Defaults = Appendix B.1: Adam lr 4e-4, batch 256, 100 epochs, beta 0.25, lambda 0.1, K 512, d 256,
32x32 -> 8x8 latent, 3-layer spectral-norm PatchGAN.

Examples
  # CIFAR-100 victim, purifier trained on generic natural images (CIFAR-10), paper hyper-parameters
  python -m purevqgan.train --dataset cifar10 --img_size 32 --latent_size 8 --out purifiers/cifar10_K512
  # codebook-capacity sweep point
  python -m purevqgan.train --dataset cifar10 --num_codes 64 --out purifiers/cifar10_K64
  # miniImageNet victim (84x84 -> 21x21 latent)
  python -m purevqgan.train --dataset stl10 --img_size 84 --latent_size 21 --out purifiers/stl10_84_K512
  # smoke test on CPU
  python -m purevqgan.train --dataset fake --epochs 1 --batch_size 16 --base_ch 32 --out /tmp/pvq_smoke
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_images
from .model import VQGAN, PatchDiscriminator, _torch_load, count_params


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10', help="see purevqgan/data.py (e.g. cifar10, stl10, npy:x.npy, 'a+b')")
    p.add_argument('--data_root', default='./data')
    p.add_argument('--bw_experiment', default=None, help='for --dataset bw_poisoned:<pkl>')
    p.add_argument('--out', required=True, help='output directory')
    # architecture
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--latent_size', type=int, default=8)
    p.add_argument('--num_codes', type=int, default=512, help='codebook size K')
    p.add_argument('--z_ch', type=int, default=256, help='latent / code dimension d')
    p.add_argument('--base_ch', type=int, default=128, help='width; main knob for the model-size ablation')
    p.add_argument('--ch_mult', type=int, nargs='+', default=None, help='per-downsample width multipliers')
    p.add_argument('--num_res_blocks', type=int, default=1, help='res blocks per resolution level')
    p.add_argument('--n_mid', type=int, default=2, help='res blocks at latent resolution')
    p.add_argument('--disc_ch', type=int, default=64)
    p.add_argument('--disc_layers', type=int, default=3)
    # objective / optimisation
    p.add_argument('--beta', type=float, default=0.25, help='commitment weight')
    p.add_argument('--gan_weight', type=float, default=0.1, help='lambda; 0 = VQ-VAE-only ablation')
    p.add_argument('--disc_start_epoch', type=int, default=0, help='[choice] epochs of pure VQ-VAE warm-up')
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--disc_lr', type=float, default=None, help='defaults to --lr')
    p.add_argument('--adam_betas', type=float, nargs=2, default=(0.5, 0.9), help='[choice] taming-VQGAN betas')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--hflip', type=int, default=1, help='[choice] random horizontal flips')
    p.add_argument('--restart_dead_codes', type=int, default=1,
                   help='[choice] not in the paper: re-seed codes unused in the last --restart_every steps from '
                        'encoder outputs (1 = on). Without it the codebook collapses to ~20-30 of 512 codes.')
    p.add_argument('--restart_every', type=int, default=200, help='steps between dead-code restarts')
    p.add_argument('--amp', action='store_true', help='bf16 autocast (A100)')
    p.add_argument('--val_size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if present')
    p.add_argument('--max_steps', type=int, default=0, help='stop early after N steps (debug)')
    return p.parse_args(argv)


def psnr(a01, b01):
    mse = F.mse_loss(a01.float(), b01.float(), reduction='none').flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def evaluate(G, xval, device, bs=500):
    G.eval()
    ps, used = [], torch.zeros(G.quantizer.num_codes, dtype=torch.long)
    for i in range(0, len(xval), bs):
        x = xval[i:i + bs].to(device).float() / 255
        rec, _, idx, _ = G(x)
        ps.append(psnr(((rec.float() + 1) / 2).clamp(0, 1), x) * len(x))
        used += torch.bincount(idx.flatten().cpu(), minlength=len(used))
    G.train()
    p = used.float() / used.sum()
    perplexity = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    return sum(ps) / len(xval), int((used > 0).sum()), perplexity


def save_samples(G, xval, path, device, n=32):
    try:
        from torchvision.utils import save_image
    except ImportError:
        return
    x = xval[:n].to(device).float() / 255
    rec = G.purify(x)
    save_image(torch.cat([x, rec]), path, nrow=n // 2 if n > 8 else n)


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_all = load_images(args.dataset, args.img_size, root=args.data_root, bw_experiment=args.bw_experiment)
    perm = torch.randperm(len(x_all))
    n_val = min(args.val_size, len(x_all) // 10)
    xval, xtr = x_all[perm[:n_val]], x_all[perm[n_val:]]
    print(f'[data] train {tuple(xtr.shape)}  val {tuple(xval.shape)}')

    G = VQGAN(img_size=args.img_size, latent_size=args.latent_size, base_ch=args.base_ch,
              ch_mult=tuple(args.ch_mult) if args.ch_mult else None, num_res_blocks=args.num_res_blocks,
              n_mid=args.n_mid, z_ch=args.z_ch, num_codes=args.num_codes, beta=args.beta).to(device)
    D = PatchDiscriminator(ndf=args.disc_ch, n_layers=args.disc_layers).to(device)
    print(f'[model] G params {count_params(G) / 1e6:.2f}M  D params {count_params(D) / 1e6:.2f}M  config {G.config}')
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=tuple(args.adam_betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.disc_lr or args.lr, betas=tuple(args.adam_betas))

    start_epoch, step = 0, 0
    last_path = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = _torch_load(last_path, map_location=device)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D'])
        optG.load_state_dict(ck['optG']); optD.load_state_dict(ck['optD'])
        start_epoch, step = ck['epoch'] + 1, ck['step']
        print(f'[resume] from epoch {start_epoch}')

    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(args.out, 'train_log.jsonl'), 'a')
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if (args.amp and device == 'cuda') else torch.autocast('cpu', enabled=False)
    steps_per_epoch = math.ceil(len(xtr) / args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        use_gan = args.gan_weight > 0 and epoch >= args.disc_start_epoch
        order = torch.randperm(len(xtr))
        agg = {'rec': 0., 'vq': 0., 'g_adv': 0., 'd': 0.}
        window = torch.zeros(args.num_codes, dtype=torch.long, device=device)
        restarted = 0
        for it in range(steps_per_epoch):
            x = xtr[order[it * args.batch_size:(it + 1) * args.batch_size]].to(device, non_blocking=True).float() / 255
            if args.hflip:
                flip = torch.rand(len(x), device=device) < 0.5
                x = torch.where(flip[:, None, None, None], x.flip(3), x)
            x_pm = x * 2 - 1

            # ---- generator: L_rec + L_VQ + lambda * L_GAN ----
            for p in D.parameters():
                p.requires_grad_(False)
            with amp:
                rec, vq_loss, idx, z_e = G(x)
                rec_loss = F.mse_loss(rec.float(), x_pm)
                g_adv = F.softplus(-D(rec)).mean() if use_gan else torch.zeros((), device=device)  # non-saturating -log D(x^)
                loss_g = rec_loss + vq_loss + args.gan_weight * g_adv
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()
            window += torch.bincount(idx.flatten(), minlength=args.num_codes)
            if args.restart_dead_codes and (step + 1) % args.restart_every == 0:
                restarted += G.quantizer.restart_dead_codes(window, z_e.detach())
                window.zero_()

            # ---- discriminator: max E[log D(x)] + E[log(1 - D(x^))] ----
            d_loss = torch.zeros((), device=device)
            if use_gan:
                for p in D.parameters():
                    p.requires_grad_(True)
                with amp:
                    d_loss = F.softplus(-D(x_pm)).mean() + F.softplus(D(rec.detach())).mean()
                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()

            for k, v in (('rec', rec_loss), ('vq', vq_loss), ('g_adv', g_adv), ('d', d_loss)):
                agg[k] += v.item() / steps_per_epoch
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        val_psnr, used, ppl = evaluate(G, xval, device)
        rec_ = dict(epoch=epoch, step=step, **{k: round(v, 5) for k, v in agg.items()}, val_psnr=round(val_psnr, 3),
                    codes_used=used, perplexity=round(ppl, 1), restarted=restarted, gan=use_gan,
                    sec=round(time.time() - t0, 1))
        print('[epoch] ' + json.dumps(rec_), flush=True)
        log.write(json.dumps(rec_) + '\n'); log.flush()

        torch.save(dict(G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(),
                        epoch=epoch, step=step, config=G.config, args=vars(args)), last_path)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_samples(G, xval, os.path.join(args.out, f'samples_e{epoch:03d}.png'), device)
        if args.max_steps and step >= args.max_steps:
            break

    final = os.path.join(args.out, 'purifier.pt')
    torch.save(dict(G=G.state_dict(), config=G.config, args=vars(args),
                    val_psnr=val_psnr, codes_used=used), final)
    print(f'[done] purifier -> {final}  (val PSNR {val_psnr:.2f} dB, {used}/{args.num_codes} codes used)')


if __name__ == '__main__':
    main()
