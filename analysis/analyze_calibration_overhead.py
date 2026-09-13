#!/usr/bin/env python3
"""
Systems accounting: end-to-end systems accounting including re-calibration overhead.

Decomposes every completed C-GRPO run's wall clock into three disjoint parts
using consecutive event timestamps in events.jsonl:

    training      time attributable to gradient steps
    calibration   time inside CALIBRATE() at each re-calibration checkpoint
    evaluation    time inside the in-loop evaluate() call  (MEASUREMENT, not
                  method -- a production run would not pay this)

The event stream per checkpoint is ordered:
    ... train(step N) -> 'samples' -> [eval block] -> EVAL row
                      -> [calibration block] -> 'recalibration' -> train(N+1)
so the two blocks are separable by timestamp differences.

It then compares C-GRPO's true method cost (training + ALL calibration,
including the step-0 bootstrap) against a fixed-G=K_max GRPO run costed at
the *same run's own measured* training-rollout rate, and checks the
predicted crossover condition

    n_cal / delta_cal  <  B * s * (1 - kbar / K_max)          (*)

where B is the prompt batch size and s = r_cal / r_train is the measured
throughput advantage of calibration generation (pure batched inference, no
backward pass, no optimizer step, no KL term) over training rollouts.

Outputs: analysis/results/calibration_overhead.{json,csv} + a summary table.

Usage:  python scripts/analyze_calibration_overhead.py
"""
import json
import csv
import glob
import os
import statistics
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "analysis", "results")
BATCH = 4  # prompts per gradient step (cfg.batch_size) for every run below

# n_cal is not recorded in events.jsonl; it is fixed per launcher script.
# Mapping taken from the `N_CAL=` line of the corresponding run_*.slurm.
NCAL_BY_RUNDIR = OrderedDict([
    ("gsm8k_main_v3", 200),   # run_qwen_gsm8k_v3.slurm
    ("bbh_mix6", 200),        # run_qwen_bbh_v3.slurm
    ("algebra_seed0", 100),   # run_qwen_algebra_v3.slurm
    ("math500", 100),         # run_qwen_math500_v3.slurm / llama / gemma
    ("hendrycks_math_algebra", 100),
    ("gpqa_diamond", 40),     # run_*_gpqa_v3.slurm
    ("mbpp_full", 40),        # run_*_mbpp_v3.slurm
    ("openai_humaneval", 40),  # run_qwen_humaneval_v3.slurm
])

MIN_STEPS = 250  # only analyse runs that got far enough to be meaningful


def ncal_for(path):
    for key, val in NCAL_BY_RUNDIR.items():
        if key in path:
            return val
    return None


def label_of(row):
    return "EVAL" if "eval/avg_k" in row else (row.get("event") or "train")


def analyse(path):
    with open(path) as fh:
        rows = [json.loads(line) for line in fh]

    train_rows = [r for r in rows if "train/avg_k_used" in r]
    if len(train_rows) < MIN_STEPS:
        return None

    n_cal = ncal_for(path)
    boot = [r for r in rows if r.get("event") == "calibration_done"]
    if n_cal is None or not boot:
        return None
    k_max = max(int(k) for k in boot[0]["qhats"])

    # --- decompose the timeline into disjoint blocks ----------------------
    stream = sorted((r["ts"], label_of(r)) for r in rows)
    train_s = cal_s = eval_s = 0.0
    n_recal = 0
    for i in range(1, len(stream)):
        dt = stream[i][0] - stream[i - 1][0]
        prev, cur = stream[i - 1][1], stream[i][1]
        if prev == "samples" and cur == "EVAL":
            eval_s += dt
        elif prev == "EVAL" and cur == "recalibration":
            cal_s += dt
            n_recal += 1
        elif cur == "train":
            train_s += dt
    if n_recal == 0 or train_s <= 0 or cal_s <= 0:
        return None

    kbar = statistics.mean(r["train/avg_k_used"] for r in train_rows)
    steps = len(train_rows)

    # --- rollout accounting ----------------------------------------------
    roll_train = BATCH * sum(r["train/avg_k_used"] for r in train_rows)
    # every calibration draws n_cal * k_max completions; +1 for the step-0
    # bootstrap calibration, which happens before the first training step
    roll_cal = (n_recal + 1) * n_cal * k_max

    r_train = roll_train / train_s              # training rollouts / s
    r_cal = roll_cal / (cal_s * (n_recal + 1) / n_recal)
    speedup = r_cal / r_train

    cal_h = cal_s * (n_recal + 1) / n_recal / 3600.0   # incl. bootstrap
    train_h = train_s / 3600.0
    cgrpo_h = train_h + cal_h

    # fixed-G=K_max GRPO costed at this run's own measured training rate
    fixed_h = (steps * BATCH * k_max) / r_train / 3600.0
    saving = 100.0 * (1.0 - cgrpo_h / fixed_h)

    # --- crossover condition (*) -----------------------------------------
    delta_cal = 50  # recalibrate_every for every v3 launcher
    lhs = n_cal / delta_cal
    rhs = BATCH * speedup * (1.0 - kbar / k_max)
    predicted_win = lhs < rhs

    return dict(
        run=os.path.relpath(os.path.dirname(path), os.path.join(REPO, "runs")),
        dataset=path.split("/runs/conformal_grpo/")[1].split("/")[0],
        steps=steps, n_cal=n_cal, k_max=k_max, kbar=round(kbar, 2),
        n_recal=n_recal,
        train_h=round(train_h, 3), cal_h=round(cal_h, 3),
        eval_h=round(eval_h := eval_s / 3600.0, 3),
        cgrpo_h=round(cgrpo_h, 3), fixed_Kmax_h=round(fixed_h, 3),
        saving_pct=round(saving, 1),
        rollouts_train=int(roll_train), rollouts_cal=int(roll_cal),
        r_train_per_s=round(r_train, 3), r_cal_per_s=round(r_cal, 3),
        cal_speedup=round(speedup, 2),
        cal_over_train_pct=round(100.0 * cal_h / train_h, 1),
        crossover_lhs=round(lhs, 3), crossover_rhs=round(rhs, 3),
        predicted_win=predicted_win, actual_win=bool(saving > 0),
        crossover_correct=bool(predicted_win == (saving > 0)),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    pattern = os.path.join(REPO, "runs", "conformal_grpo", "**", "events.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            rec = analyse(path)
        except Exception as exc:          # noqa: BLE001 - keep going on odd runs
            print(f"  skip {path}: {exc}")
            continue
        if rec:
            results.append(rec)

    results.sort(key=lambda r: -r["saving_pct"])

    hdr = (f"{'dataset':32}{'ncal':>5}{'K':>4}{'kbar':>6}{'train_h':>8}"
           f"{'cal_h':>7}{'eval_h':>8}{'C-GRPO':>8}{'G=Kmax':>8}{'save':>7}{'xover':>7}")
    print("\nQ3: end-to-end cost including ALL re-calibration overhead")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['dataset'][:32]:32}{r['n_cal']:>5}{r['k_max']:>4}{r['kbar']:>6.1f}"
              f"{r['train_h']:>8.2f}{r['cal_h']:>7.2f}{r['eval_h']:>8.2f}"
              f"{r['cgrpo_h']:>8.2f}{r['fixed_Kmax_h']:>8.2f}{r['saving_pct']:>6.0f}%"
              f"{'  ok' if r['crossover_correct'] else '  MISS':>7}")

    savings = [r["saving_pct"] for r in results]
    speedups = [r["cal_speedup"] for r in results]
    correct = sum(r["crossover_correct"] for r in results)
    print("-" * len(hdr))
    print(f"runs analysed            : {len(results)}")
    print(f"saving vs fixed G=K_max  : median {statistics.median(savings):.0f}%  "
          f"range {min(savings):.0f}% .. {max(savings):.0f}%")
    print(f"calibration speedup s    : median {statistics.median(speedups):.2f}x  "
          f"range {min(speedups):.2f} .. {max(speedups):.2f}")
    print(f"crossover condition (*)  : {correct}/{len(results)} runs predicted correctly")
    print("\ncalibration cost as % of training wall clock, by n_cal:")
    for nc in sorted({r["n_cal"] for r in results}):
        grp = [r["cal_over_train_pct"] for r in results if r["n_cal"] == nc]
        print(f"   n_cal={nc:>4}: median {statistics.median(grp):>5.0f}%   (n={len(grp)})")

    with open(os.path.join(OUT_DIR, "calibration_overhead.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    if results:
        with open(os.path.join(OUT_DIR, "calibration_overhead.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"\nwrote {OUT_DIR}/calibration_overhead.{{json,csv}}")


if __name__ == "__main__":
    main()
