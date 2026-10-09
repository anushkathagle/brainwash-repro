"""PureVQ-GAN networks: VQ-VAE generator G = D(Q(E(x))) and a PatchGAN discriminator D_adv.

Paper spec (Sec. 3.1, App. B.2):
  * encoder: 4 residual blocks with stride-2 convs, 32x32 -> 8x8 latent, d = 256 latent channels
  * codebook: K = 512 vectors of dim d, nearest-neighbour quantization (eq. 1)
  * decoder mirrors the encoder with transposed convolutions
  * discriminator: PatchGAN, 3 layers, spectral normalization
  * GroupNorm everywhere instead of BatchNorm

Images enter/leave the public API in [0, 1] (the BrainWash data range). Internally the networks
work in [-1, 1].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _gn(ch):
    # [choice] 32 groups (VQGAN/taming default); fall back to fewer groups for narrow layers.
    g = 32
    while ch % g:
        g //= 2
    return nn.GroupNorm(g, ch, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            _gn(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, 1, 1),
            _gn(cout), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(cout, cout, 3, 1, 1),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)


class Encoder(nn.Module):
    """conv_in -> [ResBlock x nrb, stride-2 conv] x n_down -> ResBlock x n_mid -> GN/SiLU/conv -> z_e."""

    def __init__(self, base_ch=128, ch_mult=(1, 2), num_res_blocks=1, n_mid=2, z_ch=256, in_ch=3):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        layers = [nn.Conv2d(in_ch, chs[0], 3, 1, 1)]
        cur = chs[0]
        for i, ch in enumerate(chs):
            for _ in range(num_res_blocks):
                layers.append(ResBlock(cur, ch))
                cur = ch
            layers.append(nn.Conv2d(cur, cur, 4, 2, 1))          # stride-2 downsample
        for _ in range(n_mid):
            layers.append(ResBlock(cur, cur))
        layers += [_gn(cur), nn.SiLU(), nn.Conv2d(cur, z_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of Encoder; upsampling with transposed convolutions (App. B.2)."""

    def __init__(self, base_ch=128, ch_mult=(1, 2), num_res_blocks=1, n_mid=2, z_ch=256, out_ch=3):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        cur = chs[-1]
        layers = [nn.Conv2d(z_ch, cur, 3, 1, 1)]
        for _ in range(n_mid):
            layers.append(ResBlock(cur, cur))
        for ch in reversed(chs):
            layers.append(nn.ConvTranspose2d(cur, cur, 4, 2, 1))  # stride-2 upsample
            for _ in range(num_res_blocks):
                layers.append(ResBlock(cur, ch))
                cur = ch
        layers += [_gn(cur), nn.SiLU(), nn.Conv2d(cur, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook lookup (eq. 1) with the VQ-VAE loss (eq. 2) and straight-through grads.

    L_VQ = beta * ||z_e - sg[e_q]||^2 + ||sg[z_e] - e_q||^2
    """

    def __init__(self, num_codes=512, dim=256, beta=0.25):
        super().__init__()
        self.num_codes, self.dim, self.beta = num_codes, dim, beta
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)  # van den Oord init

    def nearest(self, z_flat):
        e = self.codebook.weight
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ e.t()
             + e.pow(2).sum(1)[None, :])
        return d.argmin(1)

    def forward(self, z):
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c).float()
        idx = self.nearest(z_flat)
        z_q = self.codebook(idx).view(b, h, w, c).permute(0, 3, 1, 2).to(z.dtype)
        loss = self.beta * F.mse_loss(z, z_q.detach()) + F.mse_loss(z.detach(), z_q)
        z_q = z + (z_q - z).detach()                                    # straight-through estimator
        return z_q, loss, idx.view(b, h, w)

    @torch.no_grad()
    def restart_dead_codes(self, usage_counts, z_e):
        """Not in the paper (on by default in train.py): re-seed unused codes from random encoder outputs."""
        dead = (usage_counts == 0).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.dim).float()
        pick = torch.randint(0, z_flat.shape[0], (len(dead),), device=z_flat.device)
        self.codebook.weight.data[dead] = z_flat[pick].to(self.codebook.weight.dtype)
        return len(dead)


class VQGAN(nn.Module):
    """Generator G of PureVQ-GAN. `purify(x)` is the single-forward-pass purifier (eq. 4, T=1)."""

    def __init__(self, img_size=32, latent_size=8, base_ch=128, ch_mult=None, num_res_blocks=1, n_mid=2,
                 z_ch=256, num_codes=512, beta=0.25):
        super().__init__()
        n_down = int(round(math.log2(img_size / latent_size)))
        if 2 ** n_down * latent_size != img_size:
            raise ValueError(f'img_size {img_size} / latent_size {latent_size} must be a power of two')
        if ch_mult is None:
            ch_mult = tuple([1] + [2] * (n_down - 1))   # 32->8: (1, 2)
        if len(ch_mult) != n_down:
            raise ValueError(f'ch_mult needs {n_down} entries for {img_size}->{latent_size}, got {ch_mult}')
        self.config = dict(img_size=img_size, latent_size=latent_size, base_ch=base_ch, ch_mult=tuple(ch_mult),
                           num_res_blocks=num_res_blocks, n_mid=n_mid, z_ch=z_ch, num_codes=num_codes, beta=beta)
        self.encoder = Encoder(base_ch, ch_mult, num_res_blocks, n_mid, z_ch)
        self.quantizer = VectorQuantizer(num_codes, z_ch, beta)
        self.decoder = Decoder(base_ch, ch_mult, num_res_blocks, n_mid, z_ch)

    def encode(self, x01):
        return self.encoder(x01 * 2 - 1)

    def forward(self, x01):
        """Returns (recon in [-1,1] space, vq_loss, code indices, z_e). Used for training."""
        z_e = self.encode(x01)
        z_q, vq_loss, idx = self.quantizer(z_e)
        return self.decoder(z_q), vq_loss, idx, z_e

    @torch.no_grad()
    def purify(self, x01, passes=1):
        """x^{t+1} = G(x^{t}), x^{0} = x  (eq. 4). Input/output in [0, 1]."""
        x = x01
        for _ in range(passes):
            rec, _, _, _ = self.forward(x)
            x = ((rec.float() + 1) / 2).clamp(0, 1)
        return x


class PatchDiscriminator(nn.Module):
    """PatchGAN (pix2pix / taming NLayerDiscriminator) with spectral norm, n_layers=3 (App. B.1/B.2).

    Outputs a map of real/fake logits; no norm layers besides spectral norm.
    """

    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        sn = spectral_norm
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)), nn.LeakyReLU(0.2, True),
                   sn(nn.Conv2d(ndf * mult, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x_pm1):
        return self.net(x_pm1)


def _torch_load(path, map_location='cpu'):
    # torch>=2.6 defaults to weights_only=True (our checkpoints hold plain dicts/tuples); torch 1.12 lacks the kwarg.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_vqgan(config):
    return VQGAN(**config)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
