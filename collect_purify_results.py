#!/usr/bin/env python3
"""
collect_purify_results.py

Build TSV tables from the stage-4 log files -- undefended runs (repro/sweep/results/)
and purification runs (repro/sweep/results_purify/) -- so arms can be compared
side by side without opening logs.

Writes two files:
  <out>_runs.tsv     one row per log: dataset prefix, method, mode, eps, seed, arm,
                     sigma*, after BWT / after avg acc / last-task acc (in %),
                     before BWT / before avg acc, and the log path
  <out>_summary.tsv  mean / std / n over seeds for each (prefix, method, mode, eps, arm)

Arms:
  A          undefended clean         (results/<p>_<m>_clean[_sN].log)
  B          undefended poison        (results/<p>_<m>_<mode>_eps<e>[_sN].log)
  B_uniform  undefended uniform noise (results/<p>_<m>_uniform_eps<e>[_sN].log)
  D          clean + purify           (results_purify/<noise tag>_clean_purify64_sigma<s>.log)
  C          poison + purify          (results_purify/<noise tag>_ours_purify64_sigma<s>.log)
  C_uniform  uniform + purify         (results_purify/<noise tag>_uniform_purify64_sigma<s>.log)
Logs without a final "After BWT" line (unfinished / crashed) are listed as incomplete.

Usage (repo root):
  python collect_purify_results.py
  python collect_purify_results.py --prefix tiny20 --out repro/sweep/tiny20_results
  python collect_purify_results.py --dirs repro/sweep/results /path/to/other/results_purify
"""
import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

AFTER_RE = re.compile(r"After BWT\s*:\s*(\S+)\s+After avg acc\s*:\s*(\S+)\s+Last task acc\s*:\s*(\S+)")
BEFORE_RE = re.compile(r"Before BWT\s*:\s*(\S+)\s+Before avg acc\s*:\s*(\S+)")
NAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)_(?P<method>afec_ewc|ewc|mas|rwalk)_"
    r"(?P<mode>clean|uniform|cautious|reckless)(?:_eps(?P<eps>[0-9.]+))?(?:_s(?P<seed>\d+))?"
    r"(?:_(?P<variant>clean|ours|uniform)_purify(?P<res>64)?_sigma(?P<sigma>[0-9.]+?)(?:_sp(?P<sp>\d+))?"
    r"|_purify_sigma(?P<sigma_old>[0-9.]+))?$"
)
METRICS = ["after_bwt", "after_avg_acc", "last_task_acc", "before_bwt", "before_avg_acc"]


def parse_log(path):
    vals = {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = AFTER_RE.search(line)
            if m:
                vals.update(after_bwt=float(m[1]) * 100, after_avg_acc=float(m[2]) * 100,
                            last_task_acc=float(m[3]) * 100)
            m = BEFORE_RE.search(line)
            if m:
                vals.update(before_bwt=float(m[1]) * 100, before_avg_acc=float(m[2]) * 100)
    return vals


def classify(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    variant, sigma = d["variant"], d["sigma"] or d["sigma_old"]
    mode = d["mode"]
    if variant is None and d["sigma_old"] is None:          # undefended
        arm = {"clean": "A", "uniform": "B_uniform"}.get(mode, "B")
        eps = "" if mode == "clean" else (d["eps"] or "")
    else:                                                    # purified
        variant = variant or ("clean" if mode == "clean" else "uniform" if mode == "uniform" else "ours")
        arm = {"clean": "D", "ours": "C", "uniform": "C_uniform"}[variant]
        if variant == "clean":
            mode, eps = "clean", ""                          # Arm D does not depend on the pkl's mode/eps
        elif variant == "uniform":
            mode, eps = "uniform", d["eps"] or ""
        else:
            eps = d["eps"] or ""
    return dict(prefix=d["prefix"], method=d["method"], mode=mode, eps=eps,
                seed=int(d["seed"] or 0), arm=arm, sigma=sigma or "-",
                respace=d["sp"] or "-")


def fmt(x):
    return "" if x is None else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=["repro/sweep/results", "repro/sweep/results_purify"])
    ap.add_argument("--prefix", default=None, help="only this dataset prefix, e.g. tiny20")
    ap.add_argument("--method", default=None)
    ap.add_argument("--out", default="repro/sweep/purify_results")
    args = ap.parse_args()

    runs, skipped, incomplete = [], [], []
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.log"))):
            name = os.path.basename(path)[:-4]
            info = classify(name)
            if info is None:
                skipped.append(path); continue
            if args.prefix and info["prefix"] != args.prefix:
                continue
            if args.method and info["method"] != args.method:
                continue
            vals = parse_log(path)
            if "after_bwt" not in vals:
                incomplete.append(path); continue
            runs.append({**info, **vals, "log": path})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    key = lambda r: (r["prefix"], r["method"], r["mode"], r["eps"], r["arm"], r["sigma"], r["respace"])
    order = {"A": 0, "B": 1, "B_uniform": 2, "D": 3, "C": 4, "C_uniform": 5}
    runs.sort(key=lambda r: (r["prefix"], r["method"], order[r["arm"]], r["mode"], r["eps"], r["seed"]))

    cols = ["prefix", "method", "arm", "mode", "eps", "seed", "sigma", "respace"] + METRICS + ["log"]
    with open(args.out + "_runs.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in runs:
            fh.write("\t".join(fmt(r.get(c)) if c in METRICS else str(r[c]) for c in cols) + "\n")

    groups = defaultdict(list)
    for r in runs:
        groups[key(r)].append(r)
    scols = ["prefix", "method", "arm", "mode", "eps", "sigma", "respace", "n", "seeds"]
    for m in METRICS[:3]:
        scols += [f"{m}_mean", f"{m}_std"]
    with open(args.out + "_summary.tsv", "w") as fh:
        fh.write("\t".join(scols) + "\n")
        for k in sorted(groups, key=lambda k: (k[0], k[1], order[k[4]], k[2], k[3])):
            g = groups[k]
            row = list(k[:2]) + [k[4], k[2], k[3], k[5], k[6], str(len(g)),
                                 ",".join(str(r["seed"]) for r in sorted(g, key=lambda r: r["seed"]))]
            for m in METRICS[:3]:
                xs = [r[m] for r in g]
                row += [fmt(statistics.mean(xs)), fmt(statistics.stdev(xs)) if len(xs) > 1 else ""]
            fh.write("\t".join(row) + "\n")

    print(f"{len(runs)} finished runs -> {args.out}_runs.tsv")
    print(f"{len(groups)} groups (mean/std over seeds) -> {args.out}_summary.tsv")
    if incomplete:
        print(f"{len(incomplete)} incomplete logs (no 'After BWT'):")
        for p in incomplete:
            print("   ", p)
    if skipped:
        print(f"{len(skipped)} logs with unrecognized names were ignored (e.g. {os.path.basename(skipped[0])})")


if __name__ == "__main__":
    main()
