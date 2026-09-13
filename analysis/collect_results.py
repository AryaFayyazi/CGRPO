#!/usr/bin/env python3
"""
Collect the experiments into the tables that answer Q1, Q2 and Q4.

Q1  accuracy-vs-training-compute frontier for fixed-G GRPO, G in {2,4,8,16,32,64}
    against C-GRPO, all trained identically except the rollout-allocation rule.
Q2  matched-compute comparison: C-GRPO's measured training-rollout total is
    located on the fixed-G frontier by linear interpolation between the two
    bracketing arms, so "same compute, which is more accurate?" is answered
    directly rather than by comparing against G=32 only.
Q4  long-horizon (600-step) C-GRPO trajectory vs the 300-step run.

Safe to run while the chain is still going; missing arms are reported as
pending rather than silently dropped.

Usage:  python scripts/collect_results.py
"""
import json
import glob
import os
import re
import csv
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "analysis", "results")
BATCH = 4
STEPS = 300

# The C-GRPO arm the fixed-G sweep is matched against.
CGRPO_300 = os.path.join(
    REPO, "runs", "conformal_grpo", "QWEN2.5-7b_gsm8k_main_v3_seed0",
    "QWEN2.5-7b_deltaauto_seed0_20260419_141159")
CGRPO_LONG_GLOB = os.path.join(REPO, "runs", "ablations", "cgrpo_long600", "**")

FIXED_G = [2, 4, 8, 16, 32, 64]


def load_pareto(run_dir):
    hits = glob.glob(os.path.join(run_dir, "**", "pareto_results_*.json"),
                     recursive=True)
    if not hits:
        return None
    with open(sorted(hits, key=os.path.getmtime)[-1]) as fh:
        return json.load(fh)


def train_rollouts_from_events(run_dir):
    """Actual rollouts consumed by gradient steps (not calibration/eval)."""
    hits = glob.glob(os.path.join(run_dir, "**", "events.jsonl"), recursive=True)
    if not hits:
        return None, None
    with open(sorted(hits, key=os.path.getmtime)[-1]) as fh:
        rows = [json.loads(line) for line in fh]
    tr = [r for r in rows if "train/avg_k_used" in r]
    if not tr:
        return None, None
    return BATCH * sum(r["train/avg_k_used"] for r in tr), len(tr)


def wallclock_from_logs(pattern):
    """Parse '>>> TRAIN_WALLCLOCK_SECONDS <tag>: N' markers emitted by the slurm scripts."""
    out = {}
    for path in glob.glob(os.path.join(REPO, "ablations", "logs", pattern)):
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = re.search(r">>> TRAIN_WALLCLOCK_SECONDS ([^:]+): (\d+)", line)
                if m:
                    out[m.group(1).strip()] = int(m.group(2))
    return out


def acc(pareto, key):
    try:
        return 100.0 * pareto["results"][str(key)]["accuracy"]
    except (KeyError, TypeError):
        return None


def fmt(v, w=6, p=1):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wall = wallclock_from_logs("fixedG_*.log")

    rows = []

    # ---- fixed-G arms ----------------------------------------------------
    for G in FIXED_G:
        run_dir = os.path.join(REPO, "runs", "ablations", f"grpo_G{G}")
        pareto = load_pareto(run_dir)
        roll, nsteps = train_rollouts_from_events(run_dir)
        rows.append(dict(
            arm=f"GRPO G={G}", kind="fixed", G=G,
            train_rollouts=roll if roll else STEPS * BATCH * G,
            train_rollouts_measured=roll is not None,
            steps=nsteps or STEPS,
            train_h=(wall.get(f"G={G}") or 0) / 3600.0 or None,
            p1=acc(pareto, 1), p8=acc(pareto, 8),
            p16=acc(pareto, 16), p32=acc(pareto, 32),
            conf=acc(pareto, "conf"),
            status="done" if pareto else "PENDING",
        ))

    # ---- C-GRPO 300-step (existing headline arm) -------------------------
    p_c = load_pareto(CGRPO_300)
    roll_c, steps_c = train_rollouts_from_events(CGRPO_300)
    rows.append(dict(
        arm="C-GRPO (300 steps)", kind="conformal", G=None,
        train_rollouts=roll_c, train_rollouts_measured=True, steps=steps_c,
        train_h=None,
        p1=acc(p_c, 1), p8=acc(p_c, 8), p16=acc(p_c, 16), p32=acc(p_c, 32),
        conf=acc(p_c, "conf"),
        status="done" if p_c else "PENDING",
    ))

    # ---- C-GRPO 600-step (Q4) -------------------------------------------
    long_dirs = [d for d in glob.glob(CGRPO_LONG_GLOB) if os.path.isdir(d)]
    p_l = load_pareto(os.path.join(REPO, "runs", "ablations", "cgrpo_long600"))
    roll_l, steps_l = train_rollouts_from_events(
        os.path.join(REPO, "runs", "ablations", "cgrpo_long600"))
    rows.append(dict(
        arm="C-GRPO (600 steps)", kind="conformal", G=None,
        train_rollouts=roll_l, train_rollouts_measured=roll_l is not None,
        steps=steps_l, train_h=None,
        p1=acc(p_l, 1), p8=acc(p_l, 8), p16=acc(p_l, 16), p32=acc(p_l, 32),
        conf=acc(p_l, "conf"),
        status="done" if p_l else "PENDING",
    ))

    # ---- Q1 table --------------------------------------------------------
    print("\nQ1: accuracy vs TRAINING compute (GSM8K, Qwen2.5-7B, matched config)")
    print("=" * 104)
    print(f"{'arm':22}{'train rollouts':>15}{'steps':>7}{'train h':>9}"
          f"{'p@1':>7}{'p@8':>7}{'p@16':>7}{'p@32':>7}{'conf':>7}{'status':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['arm']:22}{r['train_rollouts'] or 0:>15,}{r['steps'] or 0:>7}"
              f"{fmt(r['train_h'], 9, 2)}{fmt(r['p1'], 7)}{fmt(r['p8'], 7)}"
              f"{fmt(r['p16'], 7)}{fmt(r['p32'], 7)}{fmt(r['conf'], 7)}"
              f"{r['status']:>10}")

    # ---- Q2 matched-compute interpolation --------------------------------
    print("\nQ2: matched-compute comparison")
    print("=" * 104)
    done = [r for r in rows if r["kind"] == "fixed" and r["status"] == "done"
            and r["p1"] is not None]
    cg = next((r for r in rows if r["arm"].startswith("C-GRPO (300")), None)
    if cg and cg["p1"] is not None and len(done) >= 2 and cg["train_rollouts"]:
        target = cg["train_rollouts"]
        done.sort(key=lambda r: r["train_rollouts"])
        lo = max((r for r in done if r["train_rollouts"] <= target),
                 key=lambda r: r["train_rollouts"], default=None)
        hi = min((r for r in done if r["train_rollouts"] >= target),
                 key=lambda r: r["train_rollouts"], default=None)
        print(f"  C-GRPO consumed {target:,.0f} training rollouts "
              f"(= fixed G_eff {target / (STEPS * BATCH):.1f})")
        if lo and hi and lo is not hi:
            for metric in ("p1", "p8", "p16", "p32"):
                w = (target - lo["train_rollouts"]) / (hi["train_rollouts"] - lo["train_rollouts"])
                interp = lo[metric] + w * (hi[metric] - lo[metric])
                delta = cg[metric] - interp
                print(f"    {metric:>4}:  fixed-G frontier @ matched compute = {interp:5.2f}"
                      f"   C-GRPO = {cg[metric]:5.2f}   delta = {delta:+5.2f} pp"
                      f"   (interp between G={lo['G']} and G={hi['G']})")
        else:
            print("    need one fixed-G arm on each side of the matched budget; "
                  "waiting on more arms.")
    else:
        print("  pending: needs C-GRPO plus >=2 completed fixed-G arms.")

    # ---- Q4 trajectory ---------------------------------------------------
    print("\nQ4: long-horizon trajectory (600 steps)")
    print("=" * 104)
    ev = glob.glob(os.path.join(REPO, "runs", "ablations", "cgrpo_long600",
                                "**", "events.jsonl"), recursive=True)
    if ev:
        with open(sorted(ev, key=os.path.getmtime)[-1]) as fh:
            rws = [json.loads(line) for line in fh]
        tr = [r for r in rws if "train/avg_k_used" in r]
        print(f"{'steps':>12}{'avg k_used':>12}{'hard rate':>12}{'reward':>10}{'KL':>10}")
        for lo in range(0, 600, 50):
            seg = [r for r in tr if lo < r["step"] <= lo + 50]
            if not seg:
                continue
            print(f"{f'{lo+1}-{lo+50}':>12}"
                  f"{statistics.mean(r['train/avg_k_used'] for r in seg):>12.2f}"
                  f"{statistics.mean(r['train/hard_rate'] for r in seg):>12.3f}"
                  f"{statistics.mean(r['train/avg_reward'] for r in seg):>10.3f}"
                  f"{statistics.mean(r['train/avg_kl'] for r in seg):>10.5f}")
        for r in rws:
            if "eval/conf_acc" in r:
                print(f"    eval @ step {r['step']:>4}: conf_acc={r['eval/conf_acc']:.4f} "
                      f"greedy={r['eval/greedy_acc']:.4f} avg_k={r['eval/avg_k']:.2f}")
    else:
        print("  pending.")

    with open(os.path.join(OUT_DIR, "accuracy_compute_frontier.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(OUT_DIR, "accuracy_compute_frontier.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_DIR}/accuracy_compute_frontier.{{json,csv}}")


if __name__ == "__main__":
    main()
