"""
check_tiny_val_provenance.py

Is Tiny-ImageNet's VAL folder (= tst_data in data/tiny_imagenet.npz, the "unseen"
side of the memorization check) made of ImageNet *validation* images? The
ImageNet-64 purifier was trained on ImageNet TRAIN only, so:

    Tiny val images found in ImageNet val   -> unseen -> the train-vs-val
                                               memorization check is valid (as on CIFAR)
    Tiny val images found in ImageNet train -> seen   -> the check compares seen vs
                                               seen and cannot detect memorization

Only one task's classes are needed (final task, 20-task split = classes 190-199):
500 Tiny val images vs ImageNet's images of the same 10 classes. CPU only, minutes.

HOW MATCHING WORKS
    Every image (Tiny and ImageNet) is reduced to a 16x16 grayscale thumbnail,
    mean-subtracted and normalized; two images match if the correlation of the
    thumbnails is near 1. Each ImageNet image is tried both squashed to a square
    and center-cropped, since the exact way Tiny-ImageNet was resized is not
    documented. Each Tiny image is compared only with ImageNet images of its
    own class. Reported per Tiny image: the best correlation, and the gap to the
    second-best (a true duplicate stands out far above every other image of the
    class). A contact sheet PNG shows Tiny | best match side by side -- LOOK AT
    IT; the numbers are a summary, your eyes are the evidence.

REFERENCE (ImageNet) FORMATS ACCEPTED -- give exactly one:
    --ref_dir DIR            DIR/<wnid>/**/*.JPEG   (ILSVRC train layout, or a val
                             folder already sorted into class subfolders)
    --ref_flat_dir DIR --ref_labels_csv CSV
                             flat ILSVRC2012 val folder (ILSVRC2012_val_000xxxxx.JPEG)
                             + Kaggle's LOC_val_solution.csv (ImageId,PredictionString)
                             or any CSV/TXT whose lines are "<image id or file>,<wnid> ..."
    --ref_npz NPZ --ref_map MAP
                             downsampled ImageNet (Imagenet64_val/val_data.npz, keys
                             data (N,12288) uint8 + labels 1..1000) + map_clsloc.txt
                             ("<wnid> <label> <name>" per line)

USAGE
    # 1) main check: Tiny val vs ImageNet VAL
    python check_tiny_val_provenance.py --tiny_split val --ref_flat_dir /path/ILSVRC2012/val \
        --ref_labels_csv /path/LOC_val_solution.csv --out repro/artifacts/tinyImagenet_files/provenance/val_vs_inval
    # 2) only if (1) finds few matches: Tiny val vs ImageNet TRAIN of the same classes
    python check_tiny_val_provenance.py --tiny_split val --ref_dir /path/ILSVRC2012/train \
        --out .../val_vs_intrain
    # optional control for (2): Tiny train vs ImageNet train (should match ~all)
    python check_tiny_val_provenance.py --tiny_split train --n_tiny 500 --ref_dir /path/ILSVRC2012/train \
        --out .../train_vs_intrain
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image

# Final task of the 20-task split (sorted wnids 191-200 = classes 190-199).
FINAL_TASK_WNIDS = ['n07871810', 'n07873807', 'n07875152', 'n07920052', 'n09193705',
                    'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677']
THUMB = 16
IMG_EXT = ('.jpeg', '.jpg', '.png')


# ---------------------------------------------------------------- descriptors
def _thumb(img64):
    """img64: PIL RGB 64x64 -> unit-norm, zero-mean 16x16 grayscale vector."""
    g = np.asarray(img64.convert('L').resize((THUMB, THUMB), Image.BOX), dtype=np.float64).ravel()
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-8 else g


def _variants(img):
    """ImageNet image -> 64x64 renderings to try (squash, center-crop)."""
    img = img.convert('RGB')
    w, h = img.size
    out = [img.resize((64, 64), Image.BOX)]
    s = min(w, h)
    if w != h:
        l, t = (w - s) // 2, (h - s) // 2
        out.append(img.crop((l, t, l + s, t + s)).resize((64, 64), Image.BOX))
    return out


# ---------------------------------------------------------------- loaders
def load_tiny(root, split, wnids, n_per_class):
    out = {}
    for w in wnids:
        if split == 'train':
            d = os.path.join(root, 'train', w, 'images')
        else:
            d = os.path.join(root, 'val', 'images', w)  # layout after prepare_tiny_imagenet.py
            if not os.path.isdir(d):
                d = None
        if d is None or not os.path.isdir(d):
            files = _tiny_val_from_annotations(root, w)
        else:
            files = sorted(f for f in glob.glob(os.path.join(d, '*')) if f.lower().endswith(IMG_EXT))
        if n_per_class:
            files = files[:n_per_class]
        out[w] = [(f, Image.open(f).convert('RGB')) for f in files]
    return out


def _tiny_val_from_annotations(root, wnid):
    ann = os.path.join(root, 'val', 'val_annotations.txt')
    if not os.path.isfile(ann):
        sys.exit(f'cannot find Tiny val images for {wnid} under {root}/val')
    with open(ann) as fh:
        names = [l.split('\t')[0] for l in fh if l.split('\t')[1] == wnid]
    return [os.path.join(root, 'val', 'images', n) for n in names]


def load_ref(args, wnids):
    """-> {wnid: [(name, [PIL 64x64 variants])]}"""
    ref = {w: [] for w in wnids}
    if args.ref_dir:
        for w in wnids:
            files = [f for f in glob.glob(os.path.join(args.ref_dir, w, '**', '*'), recursive=True)
                     if f.lower().endswith(IMG_EXT)]
            for f in sorted(files):
                ref[w].append((os.path.basename(f), _variants(Image.open(f))))
    elif args.ref_flat_dir:
        want = set(wnids)
        lab = {}
        with open(args.ref_labels_csv) as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or row[0].lower() in ('imageid', 'image_id', 'filename'):
                    continue
                wn = row[1].split()[0]
                if wn in want:
                    lab[os.path.splitext(os.path.basename(row[0]))[0]] = wn
        for stem, w in sorted(lab.items()):
            hits = [p for p in glob.glob(os.path.join(args.ref_flat_dir, stem + '.*'))
                    if p.lower().endswith(IMG_EXT)]
            if hits:
                ref[w].append((os.path.basename(hits[0]), _variants(Image.open(hits[0]))))
    elif args.ref_npz:
        lab_of = {}
        with open(args.ref_map) as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 2:
                    lab_of[p[0]] = int(p[1])
        d = np.load(args.ref_npz)
        data, labels = d['data'], np.asarray(d['labels']).ravel()
        for w in wnids:
            for i in np.nonzero(labels == lab_of[w])[0]:
                arr = data[i].reshape(3, 64, 64).transpose(1, 2, 0)
                ref[w].append((f'npz#{i}', [Image.fromarray(arr.astype(np.uint8))]))
    else:
        sys.exit('give one of --ref_dir, --ref_flat_dir + --ref_labels_csv, --ref_npz + --ref_map')
    return ref


# ---------------------------------------------------------------- main
def contact_sheet(pairs, path, cols=8):
    """pairs: [(tiny PIL, ref PIL, score)] -> PNG grid of [tiny|ref] tiles."""
    if not pairs:
        return
    rows = (len(pairs) + cols - 1) // cols
    tile_w, tile_h = 2 * 64 + 4, 64 + 14
    sheet = Image.new('RGB', (cols * (tile_w + 6), rows * (tile_h + 6)), 'white')
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)
    except Exception:
        draw = None
    for k, (a, b, s) in enumerate(pairs):
        x, y = (k % cols) * (tile_w + 6), (k // cols) * (tile_h + 6)
        sheet.paste(a.resize((64, 64)), (x, y))
        sheet.paste(b.resize((64, 64)), (x + 68, y))
        if draw:
            draw.text((x, y + 64), f'{s:.3f}', fill='black')
    sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiny_root', default=os.path.expanduser('~/data/tiny-imagenet-200'))
    ap.add_argument('--tiny_split', choices=['val', 'train'], default='val')
    ap.add_argument('--wnids', nargs='+', default=FINAL_TASK_WNIDS)
    ap.add_argument('--n_tiny', type=int, default=0, help='Tiny images per class (0 = all)')
    ap.add_argument('--ref_dir')
    ap.add_argument('--ref_flat_dir')
    ap.add_argument('--ref_labels_csv')
    ap.add_argument('--ref_npz')
    ap.add_argument('--ref_map')
    ap.add_argument('--dup_thresh', type=float, default=0.95,
                    help='best correlation counted as a duplicate (also needs a clear gap to 2nd best)')
    ap.add_argument('--gap_thresh', type=float, default=0.10)
    ap.add_argument('--out', default='repro/artifacts/tinyImagenet_files/provenance/check')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    tiny = load_tiny(args.tiny_root, args.tiny_split, args.wnids, args.n_tiny)
    ref = load_ref(args, args.wnids)
    print(f'Tiny {args.tiny_split}: {sum(map(len, tiny.values()))} images; reference: '
          f'{sum(map(len, ref.values()))} images over {len(args.wnids)} classes')

    rows, sheet_pairs = [], []
    for w in args.wnids:
        if not ref[w]:
            print(f'  WARNING: no reference images for {w} -- check the reference path / labels')
            continue
        # one row per (ref image, variant); remember which ref image each row belongs to
        R, owner, rimg = [], [], []
        for j, (_, vs) in enumerate(ref[w]):
            for v in vs:
                R.append(_thumb(v)); owner.append(j); rimg.append(v)
        R, owner = np.stack(R), np.asarray(owner)
        for fname, timg in tiny[w]:
            sims = R @ _thumb(timg)
            per_ref = np.full(len(ref[w]), -1.0)
            np.maximum.at(per_ref, owner, sims)
            order = np.argsort(per_ref)[::-1]
            best, second = per_ref[order[0]], (per_ref[order[1]] if len(order) > 1 else -1.0)
            bi = int(np.argmax(sims))
            dup = best >= args.dup_thresh and best - second >= args.gap_thresh
            rows.append((w, os.path.basename(fname), ref[w][order[0]][0], best, second, dup))
            sheet_pairs.append((timg, rimg[bi], best))
        n_w = sum(1 for r in rows if r[0] == w)
        d_w = sum(1 for r in rows if r[0] == w and r[5])
        print(f'  {w}: {d_w}/{n_w} duplicates (ref images: {len(ref[w])})')

    if not rows:
        sys.exit('nothing compared')
    best = np.array([r[3] for r in rows]); second = np.array([r[4] for r in rows])
    frac = np.mean([r[5] for r in rows])
    with open(args.out + '.tsv', 'w') as fh:
        fh.write('wnid\ttiny_file\tbest_ref\tbest_corr\tsecond_corr\tduplicate\n')
        for r in rows:
            fh.write(f'{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\t{int(r[5])}\n')
    idx = np.argsort(best)
    pick = list(idx[::-1][:24]) + list(idx[:24])  # best 24 and worst 24 matches
    contact_sheet([sheet_pairs[i] for i in pick], args.out + '_sheet.png')

    print(f'\nduplicates: {frac*100:.1f}% of {len(rows)} Tiny {args.tiny_split} images '
          f'(best corr >= {args.dup_thresh} and >= {args.gap_thresh} above 2nd best)')
    print(f'best corr median {np.median(best):.3f} | 2nd-best median {np.median(second):.3f}')
    print(f'wrote {args.out}.tsv and {args.out}_sheet.png (top row block = best 24, then worst 24)')
    if frac > 0.9:
        verdict = 'MATCHED: these Tiny images are in this reference set.'
    elif frac < 0.1:
        verdict = ('NOT MATCHED: not in this reference set -- IF the control (Tiny train vs ImageNet '
                   'train) matches, the method works and the absence is real.')
    else:
        verdict = 'PARTIAL / UNCLEAR: inspect the sheet; resizing may differ from what was tried.'
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
