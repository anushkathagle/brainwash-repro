#!/usr/bin/env python3
"""Collect BrainWash sweep results (multi-seed aware) and compare to the paper's Table 1.

Groups repro/sweep/results/*.log by (dataset, method, mode, eps), aggregating across seeds
(filenames may end in _s<seed>; an un-suffixed file is seed 0). Reports mean +/- std and flags
whether the paper's value falls inside our seed spread.

Flags:
  OK  = our mean is within tolerance of the paper (bwt +/-2, acc +/-3 by default)
  VAR = mean is off, but the paper value lies inside our mean +/- std (+tol) -> consistent w/ variance
  XX  = paper value lies outside our seed spread -> a genuine discrepancy worth a look

Usage:
  python repro/sweep/collect_results.py
  python repro/sweep/collect_results.py --csv table1.csv --bwt-tol 2 --acc-tol 3
"""
import argparse
import csv
import glob
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TARGETS_TSV = os.path.join(HERE, "targets.tsv")

TAG_RE = re.compile(
    r"^(?P<ds>cifar100|mini|tiny)_(?P<m>afec_ewc|ewc|mas|rwalk)_"
    r"(?P<mode>clean|uniform|cautious|reckless)(?:_eps(?P<eps>0\.\d))?(?:_s(?P<seed>\d+))?$"
)
AFTER_RE = re.compile(
    r"After BWT\s*:\s*(-?[\d.eE+-]+)\s+After avg acc\s*:\s*(-?[\d.eE+-]+)\s+Last task acc\s*:\s*(-?[\d.eE+-]+)"
)


def load_targets():
    t = {}
    if os.path.exists(TARGETS_TSV):
        with open(TARGETS_TSV) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                ds, m, mode, eps, bwt, acc = line.split("\t")
                t[(ds, m, mode, eps)] = (float(bwt), float(acc))
    return t


def parse_result(path):
    bwt = acc = None
    with open(path, errors="ignore") as f:
        for line in f:
            mm = AFTER_RE.search(line)
            if mm:
                bwt = float(mm.group(1)) * 100.0
                acc = float(mm.group(3)) * 100.0   # last-task acc = the paper's 'Acc'
    return (bwt, acc) if bwt is not None else None


def fmt(mean, sd, n):
    return f"{mean:6.1f} ±{sd:4.1f}" if n >= 2 else f"{mean:6.1f}      "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--bwt-tol", type=float, default=2.0)
    ap.add_argument("--acc-tol", type=float, default=3.0)
    args = ap.parse_args()

    targets = load_targets()
    groups = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.log"))):
        m = TAG_RE.match(os.path.basename(path)[:-4])
        if not m:
            continue
        key = (m["ds"], m["m"], m["mode"], m["eps"] or "-")
        g = groups.setdefault(key, {"bwt": [], "acc": [], "seeds": [], "incomplete": 0})
        res = parse_result(path)
        if res is None:
            g["incomplete"] += 1
        else:
            g["bwt"].append(res[0]); g["acc"].append(res[1]); g["seeds"].append(m["seed"] or "0")

    order = {"clean": 0, "uniform": 1, "cautious": 2, "reckless": 3}
    keys = sorted(groups, key=lambda k: (k[0], k[1], k[3], order.get(k[2], 9)))

    hdr = (f"{'dataset':8} {'method':8} {'mode':8} {'eps':4} {'n':>2} | {'BWT mean+/-sd':>12} | "
           f"{'Acc mean+/-sd':>12} | {'paperB':>7} {'paperA':>7} | {'dBWT':>6} {'dAcc':>6} | flag")
    print(hdr); print("-" * len(hdr))
    n_ok = n_var = n_xx = 0
    rows_csv = []
    for key in keys:
        ds, method, mode, eps = key
        g = groups[key]
        n = len(g["bwt"])
        if n == 0:
            print(f"{ds:8} {method:8} {mode:8} {eps:4} {0:>2} | INCOMPLETE ({g['incomplete']} log(s), no 'After BWT')")
            continue
        mb, ma = statistics.mean(g["bwt"]), statistics.mean(g["acc"])
        sb = statistics.stdev(g["bwt"]) if n >= 2 else 0.0
        sa = statistics.stdev(g["acc"]) if n >= 2 else 0.0
        tgt = targets.get(key)
        if tgt:
            pb, pa = tgt
            db, da = mb - pb, ma - pa
            if abs(db) <= args.bwt_tol and abs(da) <= args.acc_tol:
                flag = "OK"; n_ok += 1
            elif n >= 2 and (mb - sb - args.bwt_tol) <= pb <= (mb + sb + args.bwt_tol) \
                    and (ma - sa - args.acc_tol) <= pa <= (ma + sa + args.acc_tol):
                flag = "VAR"; n_var += 1
            else:
                flag = "XX"; n_xx += 1
            print(f"{ds:8} {method:8} {mode:8} {eps:4} {n:>2} | {fmt(mb, sb, n)} | {fmt(ma, sa, n)} | "
                  f"{pb:7.1f} {pa:7.1f} | {db:+6.1f} {da:+6.1f} | {flag}")
            rows_csv.append([ds, method, mode, eps, n, round(mb, 2), round(sb, 2), round(ma, 2), round(sa, 2), pb, pa])
        else:
            print(f"{ds:8} {method:8} {mode:8} {eps:4} {n:>2} | {fmt(mb, sb, n)} | {fmt(ma, sa, n)} | {'(no target)':>17}")
            rows_csv.append([ds, method, mode, eps, n, round(mb, 2), round(sb, 2), round(ma, 2), round(sa, 2), "", ""])

    print("-" * len(hdr))
    print(f"Groups: {n_ok} OK, {n_var} within-seed-variance (paper inside our spread), {n_xx} off (XX).")
    print("flags: OK = our mean within tol | VAR = paper inside our mean+/-sd | XX = paper outside our spread")

    seen = set(groups)
    missing = sorted(k for k in targets if k not in seen and k[0] in {"cifar100", "mini"})
    if missing:
        print(f"Not yet run ({len(missing)}): " + ", ".join("/".join(x for x in k if x != "-") for k in missing))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method", "mode", "eps", "n_seeds", "bwt_mean", "bwt_sd", "acc_mean", "acc_sd", "paper_bwt", "paper_acc"])
            w.writerows(rows_csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
