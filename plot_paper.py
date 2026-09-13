#!/usr/bin/env python3
"""
plot_paper.py  –  Generate NeurIPS-spotlight-quality figures for C-GRPO.

Usage (standalone):
    python plot_paper.py --run-dir runs/conformal_grpo/QWEN2.5-7b_openai_humaneval_seed0 \
                         --dataset humaneval

Usage (called from SLURM after eval_pareto):
    python plot_paper.py \
        --run-dir     runs/conformal_grpo/QWEN2.5-7b_openai_humaneval_seed0 \
        --pareto-json <path>/pareto_results_openai_openai_humaneval.json \
        --baseline-grpo  runs/baseline/grpo/QWEN2.5-7b_humaneval_seed0 \
        --baseline-aero  runs/baseline/aero/QWEN2.5-7b_humaneval_seed0 \
        --baseline-gdro  runs/baseline/gdro/QWEN2.5-7b_humaneval_seed0 \
        --out-dir     figures/humaneval

Figures generated:
    fig1_adaptive_k.pdf       Avg-k over training steps (key claim figure)
    fig2_accuracy_curves.pdf  Greedy + conf accuracy during training
    fig3_pareto_frontier.pdf  Accuracy-vs-compute Pareto frontier (main paper fig)
    fig4_k_distribution.pdf   k=2/16 usage breakdown over training
    fig5_qhat_evolution.pdf   Conformal threshold qhats over training
    fig6_token_savings.pdf    Total token cost per step (C-GRPO vs fixed k baselines)
    fig7_reward_kl.pdf        Reward + KL divergence during training
    fig8_passk_curves.pdf     pass@k oracle curves for all methods + C-GRPO op-point
    fig9_coverage_calib.pdf   Conformal coverage vs nominal (1-δ) — guarantee check
    fig10_efficiency_curve.pdf avg-k + accuracy dual-axis over training steps
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── matplotlib setup (non-interactive for SLURM) ─────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
})

# ── colour palette (colourblind-safe) ─────────────────────────────────────────
C = {
    "cgrpo":   "#2166AC",   # blue
    "grpo":    "#D6604D",   # red
    "aero":    "#F4A582",   # salmon
    "gdro":    "#92C5DE",   # light blue
    "fixed_k": "#4DAC26",   # green  (oracle fixed-k curve)
    "conf":    "#762A83",   # purple (conformal op-point)
    "k2":      "#1A9850",
    "k4":      "#91CF60",
    "k8":      "#FFFFBF",
    "k16":     "#FC8D59",
    "k32":     "#D73027",
}
FIXED_K_COLORS = ["#2166AC", "#4DAC26", "#FDAE61", "#F46D43", "#D73027",
                  "#A50026", "#762A83"]


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_events(run_dir: Path) -> list[dict]:
    """Load events.jsonl from the most-recent non-empty run subdir (searches recursively)."""
    run_dir = Path(run_dir)
    # Sort by mtime newest first, skip empty files
    candidates = sorted(
        (p for p in run_dir.rglob("events.jsonl") if p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return []
    best = candidates[0]
    rows = []
    with open(best) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _load_pareto(json_path: Optional[str]) -> dict:
    if not json_path or not Path(json_path).exists():
        return {}
    with open(json_path) as f:
        return json.load(f)


def _split_rows(events: list[dict]):
    """Return (train_rows, eval_rows, calib_rows)."""
    train = [r for r in events if "train/avg_reward" in r]
    evals = [r for r in events if "eval/greedy_acc" in r]
    calib = [r for r in events if r.get("event") == "calibration_done"]
    return train, evals, calib


def _ewma(vals: list[float], alpha: float = 0.1) -> list[float]:
    out, s = [], None
    for v in vals:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


# ── Figure 1: Adaptive k over training ───────────────────────────────────────

def fig1_adaptive_k(train_rows: list[dict], k_values: list[int],
                    out: Path, dataset: str) -> None:
    steps = [r["step"] for r in train_rows if "train/avg_k_used" in r]
    k_raw = [r["train/avg_k_used"] for r in train_rows if "train/avg_k_used" in r]
    if not steps:
        print("  [fig1] No avg_k_used data — skipping.")
        return

    k_smooth = _ewma(k_raw, alpha=0.15)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(steps, k_raw,    color=C["cgrpo"], alpha=0.25, linewidth=0.8, label="_raw")
    ax.plot(steps, k_smooth, color=C["cgrpo"], linewidth=2.2, label="C-GRPO (smoothed)")

    # Horizontal baselines
    palette = ["#D73027", "#FC8D59", "#FDAE61", "#91CF60", "#4DAC26"]
    for i, k in enumerate(sorted(k_values, reverse=True)):
        ax.axhline(k, linestyle="--", color=palette[i % len(palette)],
                   linewidth=1.0, alpha=0.8, label=f"fixed k={k}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg. rollouts k per problem")
    ax.set_title(f"Adaptive Rollout Budget During Training  [{dataset}]")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_xlim(left=0)

    # Annotate final value
    if k_smooth:
        ax.annotate(f"k={k_smooth[-1]:.1f}",
                    xy=(steps[-1], k_smooth[-1]),
                    xytext=(-30, 8), textcoords="offset points",
                    fontsize=8, color=C["cgrpo"],
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=C["cgrpo"]))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig1] → {out}")


# ── Figure 2: Accuracy curves ─────────────────────────────────────────────────

def fig2_accuracy_curves(eval_rows: list[dict],
                         baseline_evals: dict[str, list[dict]],
                         out: Path, dataset: str) -> None:
    if not eval_rows:
        print("  [fig2] No eval data — skipping.")
        return

    steps  = [r["step"]            for r in eval_rows]
    greedy = [r["eval/greedy_acc"] for r in eval_rows]
    conf   = [r.get("eval/conf_acc") for r in eval_rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(steps, greedy, color=C["cgrpo"], linestyle="-",  linewidth=2.0,
            label="C-GRPO pass@1 (greedy)")
    if any(v is not None for v in conf):
        conf_clean = [v if v is not None else float("nan") for v in conf]
        ax.plot(steps, conf_clean, color=C["conf"], linestyle="--", linewidth=2.0,
                label="C-GRPO conf-acc")

    # Baseline flat lines omitted

    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy During Training  [{dataset}]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", framealpha=0.8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig2] → {out}")


# ── Figure 3: Pareto frontier ─────────────────────────────────────────────────

def fig3_pareto_frontier(pareto_data: dict,
                         baseline_evals: dict[str, list[dict]],
                         out: Path, dataset: str) -> None:
    results = pareto_data.get("results", {})
    if not results:
        print("  [fig3] No pareto results — skipping.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Fixed-k oracle curve
    fixed_ks, fixed_accs = [], []
    for k_str in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 9999):
        if not k_str.isdigit():
            continue
        k = int(k_str)
        row = results[k_str]
        if row.get("accuracy") is not None:
            fixed_ks.append(float(row.get("avg_k", k)))
            fixed_accs.append(row["accuracy"])

    if fixed_ks:
        ax.plot(fixed_ks, fixed_accs, color=C["fixed_k"], linestyle="-",
                linewidth=1.5, marker="o", markersize=5, label="Fixed-k oracle",
                alpha=0.85, zorder=2)
        for k_val, k_avg, acc in zip(
                sorted([int(x) for x in results if x.isdigit()]),
                fixed_ks, fixed_accs):
            ax.annotate(f"k={k_val}", (k_avg, acc),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=C["fixed_k"])

    # C-GRPO conformal operating point
    conf_row = results.get("conf")
    if conf_row:
        cx = conf_row.get("avg_k", 0)
        cy = conf_row.get("accuracy", 0)
        ax.scatter([cx], [cy], s=120, color=C["cgrpo"], zorder=5,
                   marker="*", label=f"C-GRPO (adaptive) k={cx:.1f}, acc={cy:.3f}")

    # Baseline operating points (fixed k=k_max, greedy acc)
    bl_markers = {"grpo": ("s", C["grpo"]), "aero": ("^", C["aero"]),
                  "gdro": ("D", C["gdro"])}
    k_max_used = pareto_data.get("k_max_generated", 32)
    for name, bl_rows in baseline_evals.items():
        bl_eval = [r for r in bl_rows if "eval/greedy_acc" in r]
        if bl_eval:
            final_acc = bl_eval[-1]["eval/greedy_acc"]
            n_rollouts_rows = [r for r in bl_rows if "train/n_rollouts" in r]
            k_used = (n_rollouts_rows[-1]["train/n_rollouts"]
                      if n_rollouts_rows else k_max_used)
            mk, col = bl_markers.get(name, ("o", "#888888"))
            ax.scatter([k_used], [final_acc], s=80, color=col, zorder=4,
                       marker=mk, label=f"{name.upper()} k={k_used} acc={final_acc:.3f}")

    ax.set_xlabel("Average rollouts k per problem  (↓ better)")
    ax.set_ylabel("Accuracy  (↑ better)")
    ax.set_title(f"Accuracy–Compute Pareto Frontier  [{dataset}]")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig3] → {out}")


# ── Figure 4: k distribution stacked area ────────────────────────────────────

def fig4_k_distribution(eval_rows: list[dict], out: Path, dataset: str) -> None:
    rows = [r for r in eval_rows if "eval/k_2_pct" in r or "eval/k_16_pct" in r]
    if not rows:
        print("  [fig4] No k distribution data in eval rows — skipping.")
        return

    steps   = [r["step"] for r in rows]
    k2_pct  = [r.get("eval/k_2_pct",  0.0) for r in rows]
    k16_pct = [r.get("eval/k_16_pct", 0.0) for r in rows]
    # Mid = everything that's neither k=2 nor k=16 (k=4,8,16 intermediate)
    mid_pct = [max(0.0, 1.0 - k2 - k16) for k2, k16 in zip(k2_pct, k16_pct)]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.stackplot(steps,
                 k2_pct, mid_pct, k16_pct,
                 labels=["k=2 (easy)", "k=4–8 (medium)", "k=16–32 (hard)"],
                 colors=[C["k2"], C["k8"], C["k16"]],
                 alpha=0.85)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of problems")
    ax.set_title(f"Distribution of Conformal k over Training  [{dataset}]")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(left=min(steps) if steps else 0)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig4] → {out}")


# ── Figure 5: qhat evolution ──────────────────────────────────────────────────

def fig5_qhat_evolution(calib_rows: list[dict], out: Path, dataset: str) -> None:
    if not calib_rows:
        print("  [fig5] No calibration events — skipping.")
        return

    steps_by_k: dict[str, list] = {}
    qhats_by_k: dict[str, list] = {}
    for row in calib_rows:
        s = row.get("step", 0)
        qhats = row.get("qhats", {})
        for k, q in qhats.items():
            steps_by_k.setdefault(k, []).append(s)
            qhats_by_k.setdefault(k, []).append(q)

    if not steps_by_k:
        print("  [fig5] qhats dict empty — skipping.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    palette = ["#2166AC", "#4DAC26", "#FDAE61", "#D73027", "#762A83"]
    for i, k in enumerate(sorted(steps_by_k.keys(), key=int)):
        ss = steps_by_k[k]
        qq = qhats_by_k[k]
        ax.plot(ss, qq, color=palette[i % len(palette)],
                marker="o", markersize=4, label=f"q̂(k={k})")

    ax.axhline(1.0, linestyle=":", color="#888888", linewidth=1.0,
               label="qhat=1.0 (inactive)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Conformal threshold q̂(k)")
    ax.set_title(f"Calibration Thresholds q̂(k) over Training  [{dataset}]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig5] → {out}")


# ── Figure 6: Cumulative rollout budget (efficiency over training) ────────────
# Shows TOTAL number of rollouts consumed through step t.
# C-GRPO starts at slope=k_max then flattens as qhats tighten.
# Static-k methods are straight lines.  The area between C-GRPO and k=k_max is
# the total rollout savings — the primary efficiency claim of the paper.

def fig6_token_savings(train_rows: list[dict],
                       baseline_evals_train: dict[str, list[dict]],
                       out: Path, dataset: str) -> None:
    # Use avg_k_used if available (direct rollout count), else fall back to
    # token_cost normalized by a rough per-completion token estimate.
    k_rows  = [r for r in train_rows if "train/avg_k_used" in r]
    tc_rows = [r for r in train_rows if "train/token_cost" in r]

    if not k_rows and not tc_rows:
        print("  [fig6] No rollout/cost data — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # ── Left panel: instantaneous avg_k per step ────────────────────────────
    ax = axes[0]
    if k_rows:
        steps_k = [r["step"] for r in k_rows]
        k_vals  = [r["train/avg_k_used"] for r in k_rows]
        k_max   = max(k_vals) if k_vals else 32
        ax.plot(steps_k, k_vals, color=C["cgrpo"], alpha=0.25, linewidth=0.8)
        ax.plot(steps_k, _ewma(k_vals, 0.08), color=C["cgrpo"], linewidth=2.2,
                label="C-GRPO (adaptive k)")
        # Reference lines for fixed-k methods
        for kref, col, ls in [(k_max, "#555555", "--"), (k_max // 2, "#aaaaaa", ":"),
                               (2, "#cccccc", ":")]:
            ax.axhline(kref, color=col, linestyle=ls, linewidth=1.2, alpha=0.7,
                       label=f"fixed k={kref}")
        # GRPO/AERO/GDRO reference lines omitted from this panel
    ax.set_xlabel("Training step"); ax.set_ylabel("Rollouts per step (avg k)")
    ax.set_title("(a) Instantaneous rollout cost")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    # ── Right panel: cumulative rollout budget ────────────────────────────────
    ax = axes[1]
    if k_rows:
        steps_k = [r["step"] for r in k_rows]
        k_vals  = [r["train/avg_k_used"] for r in k_rows]
        k_max   = max(k_vals) if k_vals else 32
        cum_k = [sum(k_vals[:i+1]) for i in range(len(k_vals))]
        ax.plot(steps_k, cum_k, color=C["cgrpo"], linewidth=2.5,
                label="C-GRPO (adaptive)", zorder=5)
        # Static-k budget references as straight lines
        for kref, col, ls, alpha in [
            (k_max,       "#d62728", "--", 0.8),
            (k_max // 4,  "#aaaaaa", ":", 0.6),
            (2,           "#cccccc", ":", 0.5)]:
            cum_static = [kref * (i+1) for i in range(len(steps_k))]
            ax.plot(steps_k, cum_static, color=col, linestyle=ls, linewidth=1.2,
                    alpha=alpha, label=f"static k={kref}")
        # Shade the savings region between adaptive and k_max
        cum_kmax = [k_max * (i+1) for i in range(len(steps_k))]
        ax.fill_between(steps_k, cum_k, cum_kmax, alpha=0.13, color=C["cgrpo"],
                        label="rollout savings")
        total_saved = cum_kmax[-1] - cum_k[-1] if cum_k else 0
        ax.text(0.97, 0.06, f"saved {total_saved:.0f} rollouts\n({100*total_saved/max(cum_kmax[-1],1):.1f}% vs k={k_max})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=C["cgrpo"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("Training step"); ax.set_ylabel("Cumulative rollouts")
    ax.set_title("(b) Cumulative rollout budget")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    fig.suptitle(f"Rollout Efficiency During Training  [{dataset}]", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig6] → {out}")


# ── Figure 7: Reward + KL ─────────────────────────────────────────────────────

def fig7_reward_kl(train_rows: list[dict], out: Path, dataset: str) -> None:
    steps   = [r["step"]               for r in train_rows if "train/avg_reward" in r]
    rewards = [r["train/avg_reward"]   for r in train_rows if "train/avg_reward" in r]
    kls     = [r.get("train/avg_kl", 0) for r in train_rows if "train/avg_reward" in r]

    if not steps:
        print("  [fig7] No reward data — skipping.")
        return

    rew_smooth = _ewma(rewards, alpha=0.1)
    kl_smooth  = _ewma(kls,     alpha=0.1)

    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
    ax2 = ax1.twinx()

    ax1.plot(steps, rewards,    color=C["cgrpo"], alpha=0.2, linewidth=0.6)
    ax1.plot(steps, rew_smooth, color=C["cgrpo"], linewidth=2.0, label="Reward")
    ax2.plot(steps, kl_smooth,  color="#D6604D", linewidth=1.5, linestyle="--",
             label="KL divergence")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Avg. reward",    color=C["cgrpo"])
    ax2.set_ylabel("KL divergence",  color="#D6604D")
    ax1.tick_params(axis="y", labelcolor=C["cgrpo"])
    ax2.tick_params(axis="y", labelcolor="#D6604D")
    ax1.set_title(f"Reward & KL Divergence  [{dataset}]")
    ax1.set_xlim(left=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig7] → {out}")


# ── Figure 8: pass@k curves ───────────────────────────────────────────────────

def fig8_passk_curves(own_pareto: dict,
                      baseline_paretos: dict[str, dict],
                      out: Path, dataset: str) -> None:
    own_results = own_pareto.get("results", {})
    if not own_results:
        print("  [fig8] No pareto results — skipping.")
        return

    FIXED_KS = [1, 2, 4, 8, 16, 32, 64]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    def _plot_method_curve(results: dict, label: str, color: str,
                           linestyle: str = "-", zorder: int = 2) -> None:
        ks, accs = [], []
        for k in FIXED_KS:
            row = results.get(str(k))
            if row and row.get("accuracy") is not None:
                ks.append(float(row.get("avg_k", k)))
                accs.append(row["accuracy"])
        if ks:
            ax.plot(ks, accs, color=color, linestyle=linestyle,
                    linewidth=1.8, marker="o", markersize=4,
                    label=label, zorder=zorder)

    _plot_method_curve(own_results, "C-GRPO (fixed-k oracle)", C["cgrpo"],
                       zorder=4)

    bl_colors = {"grpo": C["grpo"], "aero": C["aero"], "gdro": C["gdro"]}
    bl_ls     = {"grpo": "--", "aero": ":", "gdro": "-."}
    for name, bl_pareto in baseline_paretos.items():
        bl_res = bl_pareto.get("results", {})
        if bl_res:
            _plot_method_curve(bl_res, name.upper(),
                               bl_colors.get(name, "#888888"),
                               bl_ls.get(name, "--"))

    # C-GRPO adaptive operating point
    conf_row = own_results.get("conf")
    if conf_row:
        cx = conf_row.get("avg_k", 0)
        cy = conf_row.get("accuracy", 0)
        ax.scatter([cx], [cy], s=180, color=C["cgrpo"], zorder=6, marker="*",
                   label=f"C-GRPO adaptive (k={cx:.1f})")
        ax.annotate(f"C-GRPO\nk={cx:.1f}", (cx, cy),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=C["cgrpo"])

    ax.set_xlabel("k  (rollouts per problem)")
    ax.set_ylabel("Accuracy (pass@k)")
    ax.set_title(f"pass@k Curves — All Methods  [{dataset}]")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig8] → {out}")


# ── Combined summary figure ───────────────────────────────────────────────────

def fig_summary(train_rows, eval_rows, calib_rows, k_values,
                pareto_data, baseline_events, out: Path, dataset: str) -> None:
    """2×3 overview panel — all key metrics in one publication-ready figure."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"C-GRPO Training & Evaluation Summary  [{dataset}]",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── panel (0,0): adaptive k ──────────────────────────────────────────
    ax = axes[0, 0]
    steps_k = [r["step"] for r in train_rows if "train/avg_k_used" in r]
    k_raw   = [r["train/avg_k_used"] for r in train_rows if "train/avg_k_used" in r]
    if steps_k:
        ax.plot(steps_k, k_raw, color=C["cgrpo"], alpha=0.2, linewidth=0.6)
        ax.plot(steps_k, _ewma(k_raw, 0.15), color=C["cgrpo"], linewidth=2.0)
        for i, k in enumerate(sorted(k_values)):
            ax.axhline(k, linestyle="--", color=FIXED_K_COLORS[i % len(FIXED_K_COLORS)],
                       linewidth=0.8, alpha=0.75)
    ax.set_xlabel("Step"); ax.set_ylabel("Avg-k")
    ax.set_title("(a) Adaptive k")

    # ── panel (0,1): accuracy curves ─────────────────────────────────────
    ax = axes[0, 1]
    if eval_rows:
        esteps  = [r["step"]            for r in eval_rows]
        greedy  = [r["eval/greedy_acc"] for r in eval_rows]
        conf_a  = [r.get("eval/conf_acc") for r in eval_rows]
        ax.plot(esteps, greedy, color=C["cgrpo"], linewidth=1.8, label="pass@1")
        if any(v is not None for v in conf_a):
            ax.plot(esteps, [v if v else float("nan") for v in conf_a],
                    color=C["conf"], linestyle="--", linewidth=1.8, label="conf-acc")
    for name, bl_rows in baseline_events.items():
        bl_e = [r for r in bl_rows if "eval/greedy_acc" in r]
        if bl_e:
            ax.axhline(bl_e[-1]["eval/greedy_acc"],
                       linestyle=":", color=bl_colors_map(name), linewidth=1.2)
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("(b) Accuracy")
    ax.legend(fontsize=7)

    # ── panel (0,2): pareto frontier ─────────────────────────────────────
    ax = axes[0, 2]
    res = pareto_data.get("results", {})
    fks, faccs = [], []
    for k in sorted([int(x) for x in res if x.isdigit()]):
        row = res.get(str(k))
        if row and row.get("accuracy") is not None:
            fks.append(float(row.get("avg_k", k)))
            faccs.append(row["accuracy"])
    if fks:
        ax.plot(fks, faccs, color=C["fixed_k"], marker="o", markersize=4,
                linewidth=1.5, label="Fixed-k oracle")
    conf_r = res.get("conf")
    if conf_r:
        ax.scatter([conf_r["avg_k"]], [conf_r["accuracy"]], s=120,
                   color=C["cgrpo"], marker="*", zorder=5, label="C-GRPO (adaptive)")
    ax.set_xlabel("Avg-k"); ax.set_ylabel("Accuracy")
    ax.set_title("(c) Pareto frontier")
    ax.legend(fontsize=7)

    # ── panel (1,0): reward ───────────────────────────────────────────────
    ax = axes[1, 0]
    steps_r = [r["step"] for r in train_rows if "train/avg_reward" in r]
    rews    = [r["train/avg_reward"] for r in train_rows if "train/avg_reward" in r]
    if steps_r:
        ax.plot(steps_r, rews, color=C["cgrpo"], alpha=0.2, linewidth=0.6)
        ax.plot(steps_r, _ewma(rews, 0.1), color=C["cgrpo"], linewidth=1.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_title("(d) Avg. reward")

    # ── panel (1,1): qhats ────────────────────────────────────────────────
    ax = axes[1, 1]
    steps_by_k: dict = {}
    qhats_by_k: dict = {}
    for row in calib_rows:
        s = row.get("step", 0)
        for k, q in row.get("qhats", {}).items():
            steps_by_k.setdefault(k, []).append(s)
            qhats_by_k.setdefault(k, []).append(q)
    for i, k in enumerate(sorted(steps_by_k, key=int)):
        ax.plot(steps_by_k[k], qhats_by_k[k],
                color=FIXED_K_COLORS[i % len(FIXED_K_COLORS)],
                marker="o", markersize=3, label=f"k={k}")
    ax.axhline(1.0, linestyle=":", color="#aaa", linewidth=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("q̂(k)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(e) Conformal thresholds")
    ax.legend(fontsize=7)

    # ── panel (1,2): cumulative rollout budget ────────────────────────────
    ax = axes[1, 2]
    k_rows_s = [r for r in train_rows if "train/avg_k_used" in r]
    if k_rows_s:
        steps_k2 = [r["step"] for r in k_rows_s]
        kvals2   = [r["train/avg_k_used"] for r in k_rows_s]
        k_max2   = max(kvals2) if kvals2 else 32
        cum_k2   = [sum(kvals2[:i+1]) for i in range(len(kvals2))]
        ax.plot(steps_k2, cum_k2, color=C["cgrpo"], linewidth=2.0, label="C-GRPO")
        cum_max2 = [k_max2*(i+1) for i in range(len(steps_k2))]
        ax.plot(steps_k2, cum_max2, color="#d62728", linestyle="--",
                linewidth=1.2, alpha=0.7, label=f"static k={k_max2}")
        ax.fill_between(steps_k2, cum_k2, cum_max2, alpha=0.12, color=C["cgrpo"])
    ax.set_xlabel("Step"); ax.set_ylabel("Cum. rollouts")
    ax.set_title("(f) Rollout budget")
    ax.legend(fontsize=7)

    for _ax in axes.flat:
        _ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [summary] → {out}")


# ── Figure 9: Coverage calibration — does the guarantee hold? ────────────────
# For each eval checkpoint, plot the *nominal* coverage (1 − δ) on the x-axis
# and the *empirical* conformal coverage (eval/conf_coverage) on the y-axis.
# A perfectly calibrated predictor sits on the diagonal y = x.
# Points above: over-covered (conservative / safe).
# Points below: under-covered (violation — should never happen in theory).

def fig9_coverage_calibration(eval_rows: list[dict], calib_rows: list[dict],
                               out: Path, dataset: str) -> None:
    # Build a step → delta map from calibration events (initial + recalibrations)
    step_delta: dict[int, float] = {}
    for row in calib_rows:
        s = row.get("step", 0)
        d = row.get("delta")
        if d is not None:
            step_delta[s] = float(d)

    if not step_delta:
        print("  [fig9] No calibration delta data — skipping.")
        return

    cov_rows = [r for r in eval_rows if "eval/conf_coverage" in r]
    if not cov_rows:
        print("  [fig9] No eval/conf_coverage data — skipping.")
        return

    sorted_cal_steps = sorted(step_delta.keys())

    def _nearest_delta(step: int) -> float:
        best = sorted_cal_steps[0]
        for s in sorted_cal_steps:
            if s <= step:
                best = s
            else:
                break
        return step_delta[best]

    nominal: list[float] = []
    empirical: list[float] = []
    steps_seq: list[int] = []
    for row in cov_rows:
        step = row["step"]
        delta = _nearest_delta(step)
        nominal.append(1.0 - delta)
        empirical.append(float(row["eval/conf_coverage"]))
        steps_seq.append(step)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # ── Left: scatter nominal vs empirical ───────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(nominal, empirical,
                    c=steps_seq, cmap="viridis", s=55, alpha=0.85, zorder=4)
    lim_lo = max(0.0, min(nominal) - 0.05)
    lim_hi = min(1.0, max(nominal) + 0.05)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            "k--", linewidth=1.2, alpha=0.6, label="Perfect calibration (y=x)")
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                    alpha=0.07, color=C["cgrpo"], label="Over-covered (safe)")
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_lo, lim_lo],
                    alpha=0.07, color="#d62728", label="Under-covered (violation)")
    plt.colorbar(sc, ax=ax, label="Training step")
    ax.set_xlabel("Nominal coverage  (1 − δ)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Coverage calibration scatter")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, min(1.02, lim_hi + 0.05))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # ── Right: coverage & nominal over training steps ────────────────────────
    ax = axes[1]
    ax.plot(steps_seq, empirical, color=C["cgrpo"], linewidth=2.2,
            label="Empirical coverage")
    ax.plot(steps_seq, nominal, color="#d62728", linestyle="--",
            linewidth=1.5, label="Nominal (1 − δ)")
    ax.fill_between(steps_seq, nominal, empirical,
                    where=[e >= n for e, n in zip(empirical, nominal)],
                    alpha=0.12, color=C["cgrpo"], label="Excess coverage")
    ax.fill_between(steps_seq, nominal, empirical,
                    where=[e < n for e, n in zip(empirical, nominal)],
                    alpha=0.18, color="#d62728", label="Coverage violation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Coverage")
    ax.set_title("(b) Coverage vs nominal over training")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, framealpha=0.85)

    fig.suptitle(f"Conformal Coverage Calibration  [{dataset}]",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig9] → {out}")


# ── Figure 10: Efficiency curve — avg-k ↓ as accuracy ↑ ─────────────────────
# Dual-axis: left y-axis = avg_k (rollout budget per problem, dense training
# signal), right y-axis = accuracy (eval checkpoints).
# The key narrative: C-GRPO improves the policy (accuracy ↑) AND tightens the
# conformal thresholds (avg_k ↓) simultaneously.  This is the core contribution
# compressed into a single publication-ready figure.

def fig10_efficiency_curve(train_rows: list[dict], eval_rows: list[dict],
                            out: Path, dataset: str) -> None:
    k_rows   = [r for r in train_rows if "train/avg_k_used" in r]
    acc_rows = [r for r in eval_rows  if "eval/greedy_acc" in r]

    if not k_rows and not acc_rows:
        print("  [fig10] Insufficient data — skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    # ── avg_k on left axis (dense per-step signal) ───────────────────────────
    if k_rows:
        steps_k = [r["step"] for r in k_rows]
        k_vals  = [r["train/avg_k_used"] for r in k_rows]
        k_max   = max(k_vals)
        ax1.plot(steps_k, k_vals, color=C["cgrpo"], alpha=0.18, linewidth=0.7)
        ax1.plot(steps_k, _ewma(k_vals, 0.07), color=C["cgrpo"],
                 linewidth=2.5, label=f"Avg. rollouts k  (↓ budget)")
        ax1.axhline(k_max, color="#aaaaaa", linestyle=":",
                    linewidth=1.0, label=f"Fixed k={k_max} (no adaptation)")
        ax1.set_ylim(0, k_max * 1.15)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Avg. rollouts per problem  k", color=C["cgrpo"])
    ax1.tick_params(axis="y", labelcolor=C["cgrpo"])
    ax1.set_xlim(left=0)

    # ── accuracy on right axis (eval checkpoints) ────────────────────────────
    if acc_rows:
        esteps  = [r["step"]             for r in acc_rows]
        greedy  = [r["eval/greedy_acc"]  for r in acc_rows]
        conf_a  = [r.get("eval/conf_acc", r["eval/greedy_acc"]) for r in acc_rows]
        avg_k_e = [r.get("eval/avg_k")   for r in acc_rows]

        ax2.plot(esteps, greedy, color=C["grpo"], linestyle="--",
                 linewidth=2.0, marker="s", markersize=5,
                 label="Greedy acc.  (pass@1)")
        ax2.plot(esteps, conf_a, color=C["conf"], linestyle="-.",
                 linewidth=2.0, marker="^", markersize=5,
                 label="Conformal acc.  (adaptive k)")
        # Annotate the final eval avg_k value
        if avg_k_e and avg_k_e[-1] is not None:
            ax2.annotate(f"eval avg k={avg_k_e[-1]:.1f}",
                         xy=(esteps[-1], conf_a[-1]),
                         xytext=(-50, 12), textcoords="offset points",
                         fontsize=8, color=C["conf"],
                         arrowprops=dict(arrowstyle="->", color=C["conf"], lw=1.0))

    ax2.set_ylabel("Accuracy", color=C["grpo"])
    ax2.tick_params(axis="y", labelcolor=C["grpo"])
    ax2.set_ylim(0, 1.05)

    # Unified legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=8, framealpha=0.88)

    ax1.set_title(
        f"Efficiency Curve: Rollout Budget ↓ while Accuracy ↑  [{dataset}]",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig10] → {out}")


def bl_colors_map(name: str) -> str:
    return {"grpo": C["grpo"], "aero": C["aero"], "gdro": C["gdro"]}.get(name, "#888888")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Generate paper figures for C-GRPO.")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Root dir of C-GRPO run (contains events.jsonl or subdir with it)")
    p.add_argument("--dataset", default="humaneval",
                   help="Dataset name for figure titles")
    p.add_argument("--pareto-json", type=str, default=None,
                   help="Path to pareto_results_*.json (C-GRPO)")
    p.add_argument("--baseline-grpo", type=Path, default=None,
                   help="Baseline GRPO run dir (contains events.jsonl)")
    p.add_argument("--baseline-aero", type=Path, default=None)
    p.add_argument("--baseline-gdro", type=Path, default=None)
    p.add_argument("--baseline-pareto-grpo", type=str, default=None,
                   help="Path to pareto_results JSON for GRPO baseline")
    p.add_argument("--baseline-pareto-aero", type=str, default=None)
    p.add_argument("--baseline-pareto-gdro", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for figures (default: <run-dir>/figures)")
    p.add_argument("--k-values", type=str, default="2,4,8,16,32",
                   help="Comma-separated k values used in training")
    p.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"],
                   help="Output figure format")
    return p.parse_args()


def main():
    args = _parse()

    out_dir = args.out_dir or (args.run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating figures → {out_dir}")

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    fmt = args.fmt

    # ── Load C-GRPO events ─────────────────────────────────────────────
    events = _load_events(args.run_dir)
    if not events:
        print(f"  WARNING: No events.jsonl found under {args.run_dir}. "
              f"Figures will be empty.", file=sys.stderr)
    train_rows, eval_rows, calib_rows = _split_rows(events)
    print(f"  Events loaded: {len(train_rows)} train, {len(eval_rows)} eval, "
          f"{len(calib_rows)} calibrations.")

    # ── Load pareto ────────────────────────────────────────────────────
    # Auto-discover pareto JSON under run-dir if not supplied
    if not args.pareto_json:
        candidates = sorted(args.run_dir.rglob("pareto_results_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            args.pareto_json = str(candidates[0])
            print(f"  Auto-found pareto JSON: {args.pareto_json}")
    pareto_data = _load_pareto(args.pareto_json)

    # ── Load baseline events ───────────────────────────────────────────
    baseline_events: dict[str, list[dict]] = {}
    for name, bdir in [("grpo", args.baseline_grpo),
                       ("aero", args.baseline_aero),
                       ("gdro", args.baseline_gdro)]:
        if bdir:
            bl_evs = _load_events(bdir)
            if bl_evs:
                baseline_events[name] = bl_evs
                print(f"  Baseline {name}: {len(bl_evs)} events loaded.")

    # ── Load baseline pareto JSONs ─────────────────────────────────────
    baseline_paretos: dict[str, dict] = {}
    for name, bp in [("grpo", args.baseline_pareto_grpo),
                     ("aero", args.baseline_pareto_aero),
                     ("gdro", args.baseline_pareto_gdro)]:
        if bp:
            pd2 = _load_pareto(bp)
            if pd2:
                baseline_paretos[name] = pd2

    # ── Generate all figures ───────────────────────────────────────────
    def O(stem): return out_dir / f"{stem}.{fmt}"

    fig1_adaptive_k(train_rows, k_values, O("fig1_adaptive_k"), args.dataset)
    fig2_accuracy_curves(eval_rows, baseline_events, O("fig2_accuracy_curves"), args.dataset)
    fig3_pareto_frontier(pareto_data, baseline_events, O("fig3_pareto_frontier"), args.dataset)
    fig4_k_distribution(eval_rows, O("fig4_k_distribution"), args.dataset)
    fig5_qhat_evolution(calib_rows, O("fig5_qhat_evolution"), args.dataset)
    fig6_token_savings(train_rows, baseline_events, O("fig6_token_savings"), args.dataset)
    fig7_reward_kl(train_rows, O("fig7_reward_kl"), args.dataset)
    fig8_passk_curves(pareto_data, baseline_paretos, O("fig8_passk_curves"), args.dataset)
    fig9_coverage_calibration(eval_rows, calib_rows, O("fig9_coverage_calib"), args.dataset)
    fig10_efficiency_curve(train_rows, eval_rows, O("fig10_efficiency_curve"), args.dataset)
    fig_summary(train_rows, eval_rows, calib_rows, k_values,
                pareto_data, baseline_events, O("fig0_summary"), args.dataset)

    print(f"\n  ✓ All figures saved to {out_dir}/")
    # List files generated
    for f in sorted(out_dir.glob(f"*.{fmt}")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
