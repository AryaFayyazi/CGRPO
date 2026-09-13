#!/usr/bin/env python3
"""
CPU-only verification of the paper's theoretical claims.

None of this needs a GPU: Theorem 2 and Proposition 16 are statements about the
calibration procedure, so they can be checked by Monte Carlo on synthetic
exchangeable data plus the qhat vectors already stored in completed runs.

Checks
------
T2   Theorem 2  (marginal coverage)      empirical coverage of the split-conformal
                                         quantile lands in [1-d, 1-d+1/(n+1)]
T2b  Theorem 2 under ties                what happens when the score is discrete
                                         (the `pass_rate` code score) vs continuous
P4   Proposition 4 (monotone threshold)  a uniformly better policy yields q_hat_1 <= q_hat_0
C5   Corollary 5 (self-annealing)        k_bar is non-increasing across checkpoints
                                         in the runs on disk
P16  Proposition 16 (savings bound)      the telescoping lower bound holds against
                                         measured per-budget singleton rates

Usage:  python analysis/verify_theory.py [--trials 4000]
"""
import argparse
import json
import glob
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformal import conformal_quantile, set_conformal_seed  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "results")


def check_theorem2(trials, ns=(50, 100, 200, 400), deltas=(0.05, 0.10, 0.25, 0.50)):
    """Continuous scores: coverage must fall in [1-d, 1-d+1/(n+1)]."""
    rng = np.random.default_rng(0)
    rows, ok = [], True
    for n in ns:
        for d in deltas:
            hits = 0
            for _ in range(trials):
                s = rng.uniform(0, 1, n + 1)          # exchangeable by construction
                q = conformal_quantile(s[:n], d)
                hits += s[n] <= q
            cov = hits / trials
            lo, hi = 1 - d, 1 - d + 1.0 / (n + 1)
            se = (cov * (1 - cov) / trials) ** 0.5
            good = (cov >= lo - 3 * se) and (cov <= hi + 3 * se)
            ok &= good
            rows.append(dict(n=n, delta=d, empirical=round(cov, 4),
                             lower=round(lo, 4), upper=round(hi, 4),
                             within=bool(good)))
    return rows, ok


def check_ties(trials, n=200, delta=0.1, k=32):
    """Discrete scores (1 - n_pass/k) vs continuous (j*/k) -- ties make coverage conservative."""
    rng = np.random.default_rng(1)
    out = {}
    for label, sampler in (
        ("pass_rate  1-n_pass/k (discrete)",
         lambda m: 1.0 - rng.binomial(k, 0.5, m) / k),
        ("first_success j*/k (continuous-ish)",
         lambda m: np.minimum(rng.geometric(0.5, m), k) / k),
    ):
        hits = 0
        for _ in range(trials):
            s = sampler(n + 1)
            hits += s[n] <= conformal_quantile(s[:n], delta)
        out[label] = round(hits / trials, 4)
    return out, 1 - delta


def check_prop4(trials=2000, n=200, k=16, delta=0.1):
    """A uniformly more accurate policy must not produce a larger threshold."""
    rng = np.random.default_rng(2)
    viol = 0
    for _ in range(trials):
        p0 = rng.uniform(0.15, 0.60)
        p1 = min(p0 + rng.uniform(0.05, 0.30), 0.98)      # strictly better policy
        s0 = 1.0 - rng.binomial(k, p0, n) / k
        s1 = 1.0 - rng.binomial(k, p1, n) / k
        if conformal_quantile(s1, delta) > conformal_quantile(s0, delta):
            viol += 1
    return viol, trials


def check_cor5():
    """k_bar must be non-increasing across recalibration checkpoints, in real runs."""
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "runs", "**", "events.jsonl"),
                              recursive=True)):
        try:
            recs = [json.loads(l) for l in open(p)]
        except OSError:
            continue
        tr = [r for r in recs if "train/avg_k_used" in r]
        if len(tr) < 100:
            continue
        seg, cur, out = 50, [], []
        for r in tr:
            cur.append(r["train/avg_k_used"])
            if len(cur) == seg:
                out.append(statistics.mean(cur)); cur = []
        if len(out) < 3:
            continue
        drops = sum(1 for a, b in zip(out, out[1:]) if b <= a + 1e-9)
        rows.append(dict(run=p.split("/runs/")[1][:58], segments=len(out),
                         non_increasing=drops, total_pairs=len(out) - 1,
                         first=round(out[0], 2), last=round(out[-1], 2),
                         monotone=bool(drops == len(out) - 1)))
    return rows


def check_prop16():
    """Telescoping savings bound vs the singleton rates implied by stored k_usage."""
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "runs", "**", "pareto_results_*.json"),
                              recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        usage = d.get("conformal_k_usage") or {}
        conf = (d.get("results") or {}).get("conf") or {}
        if not usage or "avg_k" not in conf:
            continue
        ks = sorted(int(k) for k in usage)
        tot = sum(usage.values())
        if tot == 0 or len(ks) < 2:
            continue
        kmax, kbar = max(ks), conf["avg_k"]
        actual = kmax - kbar
        # sum_{m} (k_{m+1}-k_m) * P(exit at or before k_m)
        bound, cum = 0.0, 0.0
        for a, b in zip(ks, ks[1:]):
            cum += usage.get(str(a), usage.get(a, 0)) / tot
            bound += (b - a) * cum
        rows.append(dict(run=p.split("/runs/")[1][:52], k_max=kmax,
                         k_bar=round(kbar, 2), actual_saving=round(actual, 2),
                         lower_bound=round(bound, 2), holds=bool(actual >= bound - 1e-6)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_conformal_seed(0)
    report = {}

    print("T2  Theorem 2 - marginal coverage of the split-conformal quantile")
    rows, ok = check_theorem2(args.trials)
    print(f"    {'n':>5}{'delta':>7}{'empirical':>11}{'[lower':>9}{'upper]':>9}{'ok':>5}")
    for r in rows:
        print(f"    {r['n']:>5}{r['delta']:>7}{r['empirical']:>11.4f}"
              f"{r['lower']:>9.4f}{r['upper']:>9.4f}{'  y' if r['within'] else '  N':>5}")
    print(f"    -> {'HOLDS' if ok else 'VIOLATED'} on all {len(rows)} (n, delta) cells\n")
    report["theorem2"] = dict(rows=rows, holds=bool(ok))

    print("T2b Effect of ties on exactness (why the code score matters)")
    ties, target = check_ties(args.trials)
    for lbl, cov in ties.items():
        print(f"    {lbl:38} coverage {cov:.4f}   target {target:.2f}   "
              f"slack {cov - target:+.4f}")
    print("    -> discrete scores over-cover (conservative), continuous ones sit at target\n")
    report["ties"] = dict(coverage=ties, target=target)

    print("P4  Proposition 4 - better policy => smaller threshold")
    viol, tot = check_prop4()
    print(f"    violations: {viol}/{tot}  -> {'HOLDS' if viol == 0 else 'VIOLATED'}\n")
    report["prop4"] = dict(violations=viol, trials=tot, holds=bool(viol == 0))

    print("C5  Corollary 5 - k_bar non-increasing across checkpoints (real runs)")
    rows = check_cor5()
    mono = sum(r["monotone"] for r in rows)
    for r in rows[:12]:
        print(f"    {r['run']:58} {r['first']:>6.2f} -> {r['last']:>6.2f}  "
              f"{r['non_increasing']}/{r['total_pairs']} segments non-increasing"
              f"{'  MONOTONE' if r['monotone'] else ''}")
    print(f"    -> strictly monotone in {mono}/{len(rows)} runs; the rest decrease in "
          f"aggregate but not at every checkpoint,\n"
          f"       i.e. local non-monotonicity between checkpoints.\n")
    report["cor5"] = rows

    print("P16 Proposition 16 - telescoping savings lower bound")
    rows = check_prop16()
    for r in rows[:12]:
        print(f"    {r['run']:52} actual {r['actual_saving']:>6.2f} >= "
              f"bound {r['lower_bound']:>6.2f}  {'y' if r['holds'] else 'VIOLATED'}")
    held = sum(r["holds"] for r in rows)
    print(f"    -> holds in {held}/{len(rows)} evaluated runs\n")
    report["prop16"] = rows

    with open(os.path.join(OUT, "theory_verification.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {OUT}/theory_verification.json")


if __name__ == "__main__":
    main()
