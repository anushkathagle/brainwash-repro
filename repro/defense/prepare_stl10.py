#!/usr/bin/env python3
"""Download STL-10 once and save it as ONE compact uint8 .npy for purifier training (python -m purevqgan.train --dataset npy:<file>).

Why: `--dataset stl10` re-downloads ~2.5 GB every Colab session, and Roar compute nodes have no internet. This writes a single
(N, 3, S, S) uint8 array at the purifier's resolution (84 for miniImageNet, 64 for Tiny-ImageNet), plus a preview image and a
.json with its sha256 so you can check the copy you move to Roar.

  # Colab (writes the .npy to Drive; the raw download stays on the local disk and is deleted afterwards)
  python repro/defense/prepare_stl10.py --img_size 84 --out /content/drive/MyDrive/datasets/stl10_unlabeled_84.npy
  # 50k-image subset for a first run
  python repro/defense/prepare_stl10.py --img_size 84 --max_images 50000 --out /content/drive/MyDrive/datasets/stl10_unl50k_84.npy
  # Roar: run on a LOGIN node (compute nodes are offline), write to /storage/work
  python repro/defense/prepare_stl10.py --img_size 84 --cache_dir /storage/work/$USER/stl10_cache \
      --out /storage/work/$USER/datasets/stl10_unlabeled_84.npy
  # already have stl10_binary.tar.gz somewhere? skip the download:
  python repro/defense/prepare_stl10.py --tgz /path/stl10_binary.tar.gz --img_size 84 --out ...

Splits: 'unlabeled' (100,000 images, default), 'train' (5,000 labeled), 'test' (8,000 labeled). Join with '+'.
Caveat for the thesis: STL-10 is drawn from ImageNet, so it may share classes with miniImageNet / Tiny-ImageNet (its unlabeled
split has no labels, so they cannot be filtered out). It is a generic natural-image prior, not a guaranteed class-disjoint one.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from purevqgan.data import resize_uint8  # noqa: E402  (same antialiased resize the trainer uses)

URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
MEMBERS = {'unlabeled': 'stl10_binary/unlabeled_X.bin', 'train': 'stl10_binary/train_X.bin', 'test': 'stl10_binary/test_X.bin'}
IMG_BYTES = 3 * 96 * 96


def download(url, dst):
    """Download with resume (HTTP Range) and a progress line."""
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:          # "range not satisfiable": the file is already complete
            print(f'  already downloaded ({have / 1e9:.2f} GB)')
            return
        raise
    with r:
        if have and r.status != 206:        # server ignored Range: start over
            have = 0
        total = have + int(r.headers.get('Content-Length', 0))
        if have and total and have >= total:
            return
        mode = 'ab' if have else 'wb'
        t0, got = time.time(), have
        with open(dst, mode) as f:
            while True:
                buf = r.read(1 << 22)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                print(f'\r  downloading {got / 1e9:.2f}/{total / 1e9:.2f} GB  ({rate:.1f} MB/s)', end='', flush=True)
    print()


def extract(tgz, member, out_dir):
    path = os.path.join(out_dir, os.path.basename(member))
    if os.path.exists(path):
        return path
    print(f'  extracting {member} ...', flush=True)
    try:
        tf = tarfile.open(tgz, 'r:gz')
    except (tarfile.ReadError, EOFError) as e:
        raise SystemExit(f'{tgz} looks incomplete or corrupt ({e}); delete it and re-run to download again')
    with tf:
        src = tf.extractfile(tf.getmember(member))
        with open(path + '.part', 'wb') as f:
            shutil.copyfileobj(src, f, length=1 << 22)
    os.replace(path + '.part', path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='output .npy path (e.g. on Drive or /storage/work)')
    ap.add_argument('--img_size', type=int, default=84, help='84 for miniImageNet, 64 for Tiny-ImageNet, 96 = native')
    ap.add_argument('--split', default='unlabeled', help="unlabeled | train | test, joined with '+'")
    ap.add_argument('--max_images', type=int, default=0, help='random subset of this size (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache_dir', default='/tmp/stl10_cache', help='LOCAL disk for the raw download (~5.5 GB while running)')
    ap.add_argument('--keep_cache', action='store_true', help='keep the raw download for a second resolution')
    ap.add_argument('--tgz', default=None, help='use this stl10_binary.tar.gz instead of downloading')
    ap.add_argument('--url', default=URL)
    a = ap.parse_args()

    os.makedirs(a.cache_dir, exist_ok=True)
    tgz = a.tgz or os.path.join(a.cache_dir, 'stl10_binary.tar.gz')
    if not a.tgz:
        print(f'[stl10] downloading {a.url} -> {tgz}')
        download(a.url, tgz)

    # memory-map each raw .bin; STL-10 stores every image column-major, hence the (0, 1, 3, 2) transpose (as torchvision does)
    raws = []
    for s in a.split.split('+'):
        b = extract(tgz, MEMBERS[s], a.cache_dir)
        n = os.path.getsize(b) // IMG_BYTES
        raws.append(np.memmap(b, dtype=np.uint8, mode='r', shape=(n, 3, 96, 96)))
        print(f'[stl10] {s}: {n} images')
    index = [(r, i) for r in range(len(raws)) for i in range(len(raws[r]))]
    if a.max_images and a.max_images < len(index):
        rng = np.random.default_rng(a.seed)
        index = [index[k] for k in sorted(rng.choice(len(index), a.max_images, replace=False))]
    N, S = len(index), a.img_size
    print(f'[stl10] writing {N} images at {S}x{S} (~{N * 3 * S * S / 1e9:.2f} GB)')

    # write locally first (Drive/FUSE is slow for random writes), then copy to --out
    tmp = os.path.join(a.cache_dir, 'out.npy')
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(N, 3, S, S))
    CH = 2048
    for k in range(0, N, CH):
        chunk = np.stack([raws[r][i] for r, i in index[k:k + CH]]).transpose(0, 1, 3, 2)
        out[k:k + len(chunk)] = resize_uint8(torch.from_numpy(np.ascontiguousarray(chunk)), S).numpy()
        print(f'\r  {min(k + CH, N)}/{N}', end='', flush=True)
    print()
    out.flush()
    del out

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f'[stl10] copying to {a.out}')
    shutil.copyfile(tmp, a.out)
    os.remove(tmp)
    meta = dict(source=a.url if not a.tgz else a.tgz, split=a.split, n_images=N, shape=[N, 3, S, S], dtype='uint8',
                layout='NCHW', max_images=a.max_images, seed=a.seed, sha256=sha256(a.out))
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    try:
        from torchvision.utils import save_image
        x = torch.from_numpy(np.load(a.out, mmap_mode='r')[:64].copy()).float() / 255
        save_image(x, os.path.splitext(a.out)[0] + '_preview.png', nrow=8)
    except Exception as e:  # preview is optional
        print(f'[stl10] (no preview: {e})')
    if not a.keep_cache and not a.tgz:
        shutil.rmtree(a.cache_dir, ignore_errors=True)
    print(f'[stl10] done: {a.out}\n         sha256 {meta["sha256"]}\n'
          f'         train with: python -m purevqgan.train --dataset npy:{a.out} --img_size {S} ...')


if __name__ == '__main__':
    main()
