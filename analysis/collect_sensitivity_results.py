#!/usr/bin/env python3
"""
Collect the sensitivity experiments into tables.

study  split-delta vs shipped delta selection: per-checkpoint coverage.
       (the coverage audit itself lives in
        scripts/analyze_coverage_over_training.py)
study  stronger base: Llama-3.1-8B C-GRPO vs fixed G=8 / G=16, against the
       Qwen2.5-7B arms, to show whether savings shrink as base accuracy rises.
study  sensitivity to Delta_cal: accuracy, k_bar and calibration overhead at
       Delta_cal in {25, 50 (existing v3), 100, inf}.

Safe to run while jobs are still going; missing arms print as PENDING.

Usage:  python scripts/collect_r2_results.py
"""
import json
import glob
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "analysis", "results")
BATCH = 4
STEPS = 300

V3_GSM8K = os.path.join(REPO, "runs", "conformal_grpo",
                        "QWEN2.5-7b_gsm8k_main_v3_seed0",
                        "QWEN2.5-7b_deltaauto_seed0_20260419_141159")


def pareto(run_root):
    hits = glob.glob(os.path.join(run_root, "**", "pareto_results_*.json"),
                     recursive=True)
    if not hits:
        return None
    with open(sorted(hits, key=os.path.getmtime)[-1]) as fh:
        return json.load(fh)


def acc(p, k):
    try:
        return 100.0 * p["results"][str(k)]["accuracy"]
    except (KeyError, TypeError):
        return None


def events(run_root):
    hits = glob.glob(os.path.join(run_root, "**", "events.jsonl"), recursive=True)
    if not hits:
        return None
    with open(sorted(hits, key=os.path.getmtime)[-1]) as fh:
        return [json.loads(l) for l in fh]


def train_stats(run_root):
    """(train rollouts, mean k_bar, n_recalibrations, calibration rollouts)."""
    rows = events(run_root)
    if not rows:
        return None
    tr = [r for r in rows if "train/avg_k_used" in r]
    if not tr:
        return None
    boot = [r for r in rows if r.get("event") == "calibration_done"]
    kmax = max(int(k) for k in boot[0]["qhats"]) if boot else None
    n_recal = len([r for r in rows if r.get("event") == "recalibration"])
    return dict(
        rollouts=BATCH * sum(r["train/avg_k_used"] for r in tr),
        kbar=statistics.mean(r["train/avg_k_used"] for r in tr),
        steps=len(tr), n_recal=n_recal, k_max=kmax,
    )


def fmt(v, w=7, p=1):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def row(label, run_root, extra=None):
    p, s = pareto(run_root), train_stats(run_root)
    return dict(arm=label, status="done" if p else "PENDING",
                p1=acc(p, 1), p8=acc(p, 8), p16=acc(p, 16), p32=acc(p, 32),
                conf=acc(p, "conf"),
                kbar=s["kbar"] if s else None,
                rollouts=s["rollouts"] if s else None,
                n_recal=s["n_recal"] if s else None,
                steps=s["steps"] if s else None,
                **(extra or {}))


def show(title, rows, cols=("kbar", "rollouts", "p1", "p8", "p16", "p32", "conf")):
    print(f"\n{title}")
    print("=" * 104)
    hdr = f"{'arm':34}" + "".join(f"{c:>11}" for c in cols) + f"{'status':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['arm']:34}"
        for c in cols:
            v = r.get(c)
            line += (f"{v:>11,.0f}" if c == "rollouts" and isinstance(v, (int, float))
                     else fmt(v, 11))
        print(line + f"{r['status']:>10}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    R = os.path.join(REPO, "runs", "ablations")
    allrows = {}

    # ---- study: recalibration-period sensitivity -------------------------
    q5 = [
        row("Delta_cal = 25", os.path.join(R, "cgrpo_dcal25")),
        row("Delta_cal = 50 (v3, existing)", V3_GSM8K),
        row("Delta_cal = 100", os.path.join(R, "cgrpo_dcal100")),
        row("Delta_cal = inf (no recal)", os.path.join(R, "cgrpo_dcalINF")),
    ]
    show("study: sensitivity to re-calibration period (GSM8K, Qwen2.5-7B)", q5,
         cols=("kbar", "n_recal", "rollouts", "p1", "p16", "conf"))
    print("  n_recal x n_cal x K_max = calibration rollouts; n_cal=200, K_max=32 in all arms.")
    for r in q5:
        if r["n_recal"] is not None:
            r["cal_rollouts"] = (r["n_recal"] + 1) * 200 * 32
    allrows["q5_recal_sensitivity"] = q5

    # ---- study: stronger base -------------------------------------------
    q2 = [
        row("Qwen2.5-7B  C-GRPO", V3_GSM8K),
        row("Qwen2.5-7B  GRPO G=8", os.path.join(R, "grpo_G8")),
        row("Qwen2.5-7B  GRPO G=16", os.path.join(R, "grpo_G16")),
        row("Llama-3.1-8B C-GRPO", os.path.join(R, "llama_cgrpo")),
        row("Llama-3.1-8B GRPO G=8", os.path.join(R, "llama_grpo_G8")),
        row("Llama-3.1-8B GRPO G=16", os.path.join(R, "llama_grpo_G16")),
    ]
    show("study: does the advantage survive a stronger base? (GSM8K)", q2)
    allrows["q2_stronger_base"] = q2

    # ---- study: split-delta ---------------------------------------------
    q1 = [
        row("shipped delta_auto (v3)", V3_GSM8K),
        row("split-delta (disjoint halves)", os.path.join(R, "cgrpo_splitdelta")),
    ]
    show("study: split-delta calibration (coverage detail in the audit script)", q1)
    allrows["q1_split_delta"] = q1
    print("\n  run  python scripts/analyze_coverage_over_training.py")
    print("  for the per-checkpoint nominal-vs-empirical coverage comparison.")

    with open(os.path.join(OUT_DIR, "r2_results.json"), "w") as fh:
        json.dump(allrows, fh, indent=2)
    print(f"\nwrote {OUT_DIR}/r2_results.json")


if __name__ == "__main__":
    main()
