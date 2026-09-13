#!/usr/bin/env python3
"""
Coverage audit: does marginal coverage hold ACROSS re-calibration checkpoints?

The coverage theorem (Thm 2) is stated for a FIXED policy and a fresh
calibration draw. The algorithm actually:
  (a) reuses the same calibration PROMPTS D_cal at every checkpoint, and
  (b) selects delta_auto from those same scores before computing q_hat
      from them -- the calibration data is used twice.
(b) is the sharper problem: split-conformal exactness requires the level to
be fixed independently of the calibration scores. Choosing delta from them
turns q_hat into a data-dependent order statistic and voids exactness.

WHAT THIS MEASURES -- read before quoting any number from it.

train.py's conf_coverage is the fraction of held-out eval prompts whose gold
answer lies in C_k at the ADAPTIVELY SELECTED k_used. That is a POST-SELECTION
quantity: k_used is the smallest k making |C_k| = 1, so the stopping rule
deliberately picks the k at which the set is smallest, which biases coverage
DOWNWARD. Theorem 2 is stated per FIXED k and therefore does not govern it.

So an under-coverage here is NOT by itself a refutation of Theorem 2. What it
does show is that the operational guarantee -- coverage at the budget the
method actually stops at -- is weaker than the nominal 1-delta, which is
precisely the "does the guarantee compose over training?" gap R2 raises.

Appendix A reports this same post-selection quantity against the nominal
target, so the audit is the right check on the PAPER'S CLAIM even though it is
not a direct test of the theorem. For the per-fixed-k quantity the theorem does
cover, see results["theorem2_coverage_per_k"] in the pareto_results_*.json
emitted by eval_pareto.py (added for these runs).

Nominal target at each eval = 1 - delta in force at that moment. delta is
reconstructed as:
    delta(0)              from the 'calibration_done' event
    delta(t) after recal  from the 'delta' field if the run logs it, else
                          parsed from "delta updated: X -> Y" stdout lines
Evals precede the recalibration at the same step (samples -> EVAL ->
recalibration), so an eval at step N is governed by the delta set at the
PREVIOUS checkpoint.

Outputs analysis/results/coverage_over_training.{json,csv}.

Usage:  python scripts/analyze_coverage_over_training.py
"""
import json
import csv
import glob
import math
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "analysis", "results")
NCAL_DEFAULT = 200          # finite-sample tolerance claimed in Appendix A

# eval-split sizes, from the launcher scripts; used for the binomial-noise test
N_EVAL = {
    "QWEN2.5-7b_gsm8k_main_v3_seed0": 400, "QWEN2.5-7b_algebra_seed0": 200,
    "QWEN2.5-7b_bbh_mix6_seed0": 300, "QWEN2.5-7b_gpqa_diamond_seed0": 66,
    "QWEN2.5-7b_math500_seed0": 166, "QWEN2.5-7b_mbpp_full_seed0": 400,
    "QWEN2.5-7b_openai_humaneval_seed0": 54, "gemma3_4b_mbpp_full_seed0": 400,
    "llama3.1-8b_gpqa_diamond_seed0": 66, "llama3.1-8b_mbpp_full_seed0": 400,
    "llama3.1-8b_gsm8k_main_seed0": 400,
    # runs
    "QWEN2.5-7b_gsm8k_seed0": 400, "llama3.1-8b_gsm8k_seed0": 400,
}


def delta_trajectory_from_stdout(run_name):
    """Recover {step: delta_after_recal} by parsing the slurm stdout log."""
    candidates = (glob.glob(os.path.join(REPO, "*.log"))
                  + glob.glob(os.path.join(REPO, "ablations", "logs", "*.log")))
    for path in candidates:
        try:
            with open(path, errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        if run_name not in text:
            continue
        traj, step = {}, None
        for line in text.splitlines():
            m = re.search(r"Re-calibrating conformal qhats at step (\d+)", line)
            if m:
                step = int(m.group(1))
                continue
            m = re.search(r"updated:\s*([0-9.]+)\s*(?:->|→)\s*([0-9.]+)", line)
            if m and step is not None:
                traj[step] = float(m.group(2))
        return traj, os.path.basename(path)
    return {}, None


def analyse(events_path):
    with open(events_path) as fh:
        rows = [json.loads(line) for line in fh]

    boot = [r for r in rows if r.get("event") == "calibration_done"]
    evals = [r for r in rows if "eval/conf_coverage" in r]
    recals = [r for r in rows if r.get("event") == "recalibration"]
    if not boot or not evals:
        return None

    run_dir = os.path.dirname(events_path)
    run_name = os.path.basename(run_dir)
    delta0 = boot[0].get("delta")
    if delta0 is None:
        return None

    traj = {r["step"]: r["delta"] for r in recals if "delta" in r}
    source = "events.jsonl"
    if not traj:
        traj, logfile = delta_trajectory_from_stdout(run_name)
        source = f"stdout:{logfile}" if logfile else "delta0-only"

    recal_steps = sorted(r["step"] for r in recals)

    def delta_in_force(step):
        d = delta0
        for s in recal_steps:
            if s < step:
                d = traj.get(s, d)
        return d

    dataset = (events_path.split("/runs/")[1].split("/")[1]
               if "/runs/" in events_path else "?")
    parent = os.path.basename(os.path.dirname(run_dir))
    n_eval = N_EVAL.get(parent) or N_EVAL.get(dataset)

    points = []
    for e in sorted(evals, key=lambda r: r["step"]):
        d = delta_in_force(e["step"])
        nominal, emp = 1.0 - d, e["eval/conf_coverage"]
        slack = 100 * (emp - nominal)
        z = None
        if n_eval:
            se = math.sqrt(max(nominal * (1 - nominal), 1e-9) / n_eval) * 100
            z = round(slack / se, 2) if se > 0 else None
        points.append(dict(
            step=e["step"], delta=round(d, 4),
            nominal_pct=round(100 * nominal, 2),
            empirical_pct=round(100 * emp, 2),
            slack_pp=round(slack, 2), z=z,
            violation=bool(emp < nominal),
            significant=bool(z is not None and z < -2),
            avg_k=round(e.get("eval/avg_k", float("nan")), 2),
        ))
    return dict(run=run_name, dataset=parent, delta_source=source,
                delta0=delta0, n_eval=n_eval, points=points)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = []
    for path in sorted(glob.glob(os.path.join(REPO, "runs", "**", "events.jsonl"),
                                 recursive=True)):
        if "/conformal_grpo/" not in path and "/scripts/" not in path:
            continue
        try:
            rec = analyse(path)
        except Exception as exc:            # noqa: BLE001
            print(f"  skip {path}: {exc}")
            continue
        if rec and rec["points"]:
            out.append(rec)

    tol = 100.0 / (NCAL_DEFAULT + 1)
    n_pts = n_viol = n_beyond = n_sig = 0
    for rec in out:
        print(f"\n{rec['dataset']}  ({rec['run']})")
        print(f"  delta source: {rec['delta_source']}   delta(0)={rec['delta0']:.3f}"
              f"   n_eval={rec['n_eval']}")
        print(f"  {'step':>6}{'delta':>8}{'nominal%':>10}{'empirical%':>12}"
              f"{'slack pp':>10}{'z':>8}{'verdict':>12}")
        for p in rec["points"]:
            n_pts += 1
            n_viol += p["violation"]
            n_beyond += (-p["slack_pp"] > tol)
            n_sig += p["significant"]
            verdict = ("OK" if not p["violation"]
                       else ("UNDER*" if p["significant"] else "under"))
            zs = f"{p['z']:>8.2f}" if p["z"] is not None else f"{'--':>8}"
            print(f"  {p['step']:>6}{p['delta']:>8.3f}{p['nominal_pct']:>10.2f}"
                  f"{p['empirical_pct']:>12.2f}{p['slack_pp']:>10.2f}{zs}{verdict:>12}")

    print("\n" + "=" * 78)
    print("quantity: coverage at the ADAPTIVELY SELECTED k_used (post-selection);")
    print("this is what Appendix A reports, but NOT the fixed-k object of Thm 2.")
    print(f"checkpoints audited                                : {n_pts}")
    print(f"under-covering (empirical < nominal)               : {n_viol}"
          f"  ({100.0 * n_viol / max(n_pts, 1):.0f}%)")
    print(f"beyond App. A tolerance +/-1/(n_cal+1) = {tol:.2f} pp     : {n_beyond}")
    print(f"under-covering by >2 binomial SE (not eval noise)   : {n_sig}"
          f"  ({100.0 * n_sig / max(n_pts, 1):.0f}%)")
    print("\n* = under-covers by more than two binomial standard errors,")
    print("  i.e. not attributable to finite eval-set noise.")

    flat = [dict(dataset=r["dataset"], run=r["run"],
                 delta_source=r["delta_source"], n_eval=r["n_eval"], **p)
            for r in out for p in r["points"]]
    with open(os.path.join(OUT_DIR, "coverage_over_training.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    if flat:
        with open(os.path.join(OUT_DIR, "coverage_over_training.csv"),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    print(f"\nwrote {OUT_DIR}/coverage_over_training.{{json,csv}}")


if __name__ == "__main__":
    main()
