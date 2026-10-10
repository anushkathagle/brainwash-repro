"""Training-data sources for the purifier. Everything is returned as a uint8 tensor (N, 3, H, W) on CPU.

Spec strings for --dataset:
  cifar10                generic natural images, class-disjoint from CIFAR-100   (default for split_cifar100)
  cifar100               (victim's own distribution -- only for ablations)
  stl10                  STL-10 'unlabeled' split (100k ImageNet-derived 96x96 images; resized)
  npy:/path/x.npy        (N,H,W,3) or (N,3,H,W) uint8 array; a .npz is read via its 'x_train' key
  folder:/path/to/dir    any directory tree of .png/.jpg/.jpeg images
  bw_poisoned:/path.pkl  the poisoned BrainWash task-T training set from a stage-3 noise pkl, i.e. the
                         PureVQ-GAN paper's own protocol (purifier trained on the poisoned training set).
                         Needs --bw_experiment.
  fake                   random images (smoke tests only)
Multiple specs can be joined with '+', e.g. 'cifar10+stl10'.
"""
import glob
import os
import pickle as pkl
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_uint8_nchw(x):
    x = torch.from_numpy(np.array(x)) if not torch.is_tensor(x) else x
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    if x.dtype != torch.uint8:
        x = (x.float().clamp(0, 1) * 255).round().to(torch.uint8) if x.is_floating_point() else x.to(torch.uint8)
    return x.contiguous()


def resize_uint8(x, size, chunk=4096):
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    out = []
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk].float() / 255
        xi = F.interpolate(xi, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        out.append((xi.clamp(0, 1) * 255).round().to(torch.uint8))
    return torch.cat(out)


def _torchvision(name, root):
    import torchvision
    if name == 'cifar10':
        return _to_uint8_nchw(torchvision.datasets.CIFAR10(root, train=True, download=True).data)
    if name == 'cifar100':
        return _to_uint8_nchw(torchvision.datasets.CIFAR100(root, train=True, download=True).data)
    if name == 'stl10':
        return torch.from_numpy(torchvision.datasets.STL10(root, split='unlabeled', download=True).data)
    raise ValueError(name)


def _folder(path):
    from PIL import Image
    files = sorted(f for ext in ('png', 'jpg', 'jpeg', 'JPEG', 'PNG')
                   for f in glob.glob(os.path.join(path, '**', f'*.{ext}'), recursive=True))
    if not files:
        raise FileNotFoundError(f'no images under {path}')
    imgs = [np.array(Image.open(f).convert('RGB')) for f in files]
    sizes = {im.shape for im in imgs}
    if len(sizes) > 1:  # mixed sizes: resize each to the first one
        h, w, _ = imgs[0].shape
        imgs = [np.array(Image.fromarray(im).resize((w, h), Image.BILINEAR)) for im in imgs]
    return _to_uint8_nchw(np.stack(imgs))


def load_brainwash_task(experiment, noise_pkl, lasttask=None, tasknum=10):
    """Rebuild the poisoned task-T training set exactly as main_baselines.py does for VARIANT=ours."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    ck = pkl.load(open(noise_pkl, 'rb'))
    lasttask = ck['pretrained_ckpt']['task_num'] if lasttask is None else lasttask
    data = load_brainwash_split(experiment, tasknum)
    x = data[lasttask]['train']['x'].clone()
    noise = torch.as_tensor(ck['latest_noise']).cpu()
    x = torch.clamp(x[torch.as_tensor(ck['rnd_idx_train']).cpu()] + noise, 0, 1)
    return _to_uint8_nchw(x)


def load_brainwash_split(experiment, tasknum=10):
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from approaches import data_utils as du
    order = np.arange(200 if experiment == 'split_tiny_imagenet' else 100)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)   # the authors' loaders use paths relative to the repo root (./data)
    try:
        if experiment == 'split_cifar100':
            data, *_ = du.generate_split_cifar100_tasks(tasknum, 0, rnd_order=False, order=order)
        elif experiment == 'split_mini_imagenet':
            root = os.path.join(os.path.expanduser('~'), 'data', 'miniImagenet')
            data, *_ = du.generate_split_mini_imagenet_tasks(root, task_num=tasknum, rnd_order=False, order=order)
        elif experiment == 'split_tiny_imagenet':
            data, *_ = du.generate_split_tiny_imagenet_tasks(
                task_num=tasknum, rnd_order=False, save_data=False, dataset_file='./data/tiny_imagenet.npz',
                order=order, root_add=os.path.join(os.path.expanduser('~'), 'data', 'tiny-imagenet-200'))
        else:
            raise ValueError(experiment)
    finally:
        os.chdir(cwd)
    return data


def load_images(spec, img_size, root='./data', bw_experiment=None, n_fake=512):
    parts = []
    for s in spec.split('+'):
        kind, _, arg = s.partition(':')
        if kind in ('cifar10', 'cifar100', 'stl10'):
            x = _torchvision(kind, root)
        elif kind == 'npy':
            arr = np.load(arg) if arg.endswith('.npz') else np.load(arg, mmap_mode='r')
            if arg.endswith('.npz'):
                arr = arr['x_train']          # poisoned-dataset npz from repro/defense (key: x_train)
            x = _to_uint8_nchw(arr)
        elif kind == 'folder':
            x = _folder(arg)
        elif kind == 'bw_poisoned':
            if not bw_experiment:
                raise ValueError('bw_poisoned needs --bw_experiment')
            x = load_brainwash_task(bw_experiment, arg)
        elif kind == 'fake':
            x = torch.randint(0, 256, (n_fake, 3, img_size, img_size), dtype=torch.uint8)
        else:
            raise ValueError(f'unknown dataset spec {s!r}')
        parts.append(resize_uint8(x, img_size))
        print(f'[data] {s}: {tuple(parts[-1].shape)}')
    return torch.cat(parts)
