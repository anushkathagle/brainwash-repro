#!/usr/bin/env python3
"""Collect PureVQ-GAN-defended BrainWash results and compare them to the undefended baseline.

Reads  repro/defense/results/<base>__pvq-<ptag>.log   (defended, from submit_purevqgan.sh)
and    repro/sweep/results/<base>.log                (undefended, from the attack sweep)
where <base> = <ds>_<method>_<mode>[_eps<e>][_s<seed>].

Per (dataset, method, mode, eps, purifier) it reports mean +/- std over seeds of:
  BWT / last-task Acc  undefended vs defended, dBWT = defended - undefended (positive = less forgetting),
  avg-acc, and the purification diagnostics printed by main_baselines.py (survival_l2, code_flip,
  psnr_clean). With >= 2 paired seeds and scipy available it adds a paired t-test p-value on BWT.

Usage:
  python repro/defense/collect_defense.py
  python repro/defense/collect_defense.py --purifier cifar10_K512 --csv defense.csv
"""
import argparse
import csv
import glob
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_DIR = os.path.join(HERE, "results")
BASE_DIR = os.path.join(HERE, "..", "sweep", "results")

BASE_RE = re.compile(
    r"^(?P<ds>cifar100|mini|tiny)_(?P<m>afec_ewc|ewc|mas|rwalk)_"
    r"(?P<mode>clean|uniform|cautious|reckless)(?:_eps(?P<eps>0\.\d+))?(?:_s(?P<seed>\d+))?$"
)
AFTER_RE = re.compile(
    r"After BWT\s*:\s*(-?[\d.eE+-]+)\s+After avg acc\s*:\s*(-?[\d.eE+-]+)\s+Last task acc\s*:\s*(-?[\d.eE+-]+)"
)
REPORT_RE = re.compile(r"Purification report : (.*)")


def parse_log(path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = AFTER_RE.search(line)
            if m:
                out["bwt"], out["avg"], out["acc"] = (float(g) * 100 for g in m.groups())
            m = REPORT_RE.search(line)
            if m:
                for kv in m.group(1).split():
                    k, _, v = kv.partition("=")
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out if "bwt" in out else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "   -       "
    if len(xs) == 1:
        return f"{xs[0]:6.1f}      "
    return f"{statistics.mean(xs):6.1f} ±{statistics.stdev(xs):4.1f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def paired_p(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return None
    return float(ttest_rel([p[1] for p in pairs], [p[0] for p in pairs]).pvalue)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--purifier", help="only this purifier tag")
    ap.add_argument("--csv")
    args = ap.parse_args()

    groups = {}
    for path in sorted(glob.glob(os.path.join(DEF_DIR, "*.log"))):
        name = os.path.basename(path)[:-4]
        base, sep, ptag = name.partition("__pvq-")
        m = BASE_RE.match(base)
        if not sep or not m or (args.purifier and ptag != args.purifier):
            continue
        key = (m["ds"], m["m"], m["mode"], m["eps"] or "-", ptag)
        g = groups.setdefault(key, {"seeds": [], "und": [], "def": [], "incomplete": 0})
        d = parse_log(path)
        if d is None:
            g["incomplete"] += 1
            continue
        bpath = os.path.join(BASE_DIR, base + ".log")
        g["seeds"].append(int(m["seed"] or 0))
        g["def"].append(d)
        g["und"].append(parse_log(bpath) if os.path.exists(bpath) else None)

    hdr = (f"{'ds':8} {'method':8} {'mode':8} {'eps':4} {'purifier':16} {'n':>2} | "
           f"{'BWT undef':>12} {'BWT def':>12} {'dBWT':>6} {'p':>6} | {'Acc undef':>12} {'Acc def':>12} | "
           f"{'surv':>5} {'flip':>5} {'PSNRc':>5}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for key in sorted(groups):
        g = groups[key]
        ds, meth, mode, eps, ptag = key
        und_bwt = [u["bwt"] if u else None for u in g["und"]]
        und_acc = [u["acc"] if u else None for u in g["und"]]
        def_bwt = [d["bwt"] for d in g["def"]]
        def_acc = [d["acc"] for d in g["def"]]
        mb, mu = mean(def_bwt), mean(und_bwt)
        dbwt = (mb - mu) if (mb is not None and mu is not None) else None
        p = paired_p(und_bwt, def_bwt)
        surv, flip, psnrc = (mean([d.get(k) for d in g["def"]]) for k in ("survival_l2", "code_flip", "psnr_clean"))
        inc = f"  (+{g['incomplete']} running/failed)" if g["incomplete"] else ""
        print(f"{ds:8} {meth:8} {mode:8} {eps:4} {ptag:16} {len(g['def']):>2} | "
              f"{ms(und_bwt):>12} {ms(def_bwt):>12} {dbwt if dbwt is not None else float('nan'):6.1f} "
              f"{p if p is not None else float('nan'):6.3f} | {ms(und_acc):>12} {ms(def_acc):>12} | "
              f"{surv if surv is not None else float('nan'):5.2f} {flip if flip is not None else float('nan'):5.2f} "
              f"{psnrc if psnrc is not None else float('nan'):5.1f}{inc}")
        rows.append(dict(dataset=ds, method=meth, mode=mode, eps=eps, purifier=ptag, n=len(g["def"]),
                         seeds=" ".join(map(str, g["seeds"])),
                         bwt_undef=mu, bwt_def=mb, dbwt=dbwt, p_bwt=p,
                         acc_undef=mean(und_acc), acc_def=mean(def_acc),
                         avg_acc_def=mean([d["avg"] for d in g["def"]]),
                         survival_l2=surv, code_flip=flip, psnr_clean=psnrc))
    if not rows:
        print(f"(no defended results under {DEF_DIR})")
    print("\ndBWT > 0 = purification reduced forgetting. surv = ||P(x+d)-P(x)||/||d|| on poisoned samples "
          "(0 = perturbation erased), flip = fraction of latent codes changed by d, PSNRc = PSNR(P(x), x) in dB.")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
