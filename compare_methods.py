#!/usr/bin/env python3
"""
compare_methods.py  –  Compare all training methods on a given dataset.

Reads  runs/baseline/<method>/QWEN2.5-7b_<ds>_seed0/*/state.json
       runs/baseline/<method>/QWEN2.5-7b_<ds>_seed0/*/events.jsonl
       runs/conformal_grpo/QWEN2.5-7b_<ds>_seed0/*/state.json        (C-GRPO, if present)
       runs/conformal_grpo/QWEN2.5-7b_<ds>_seed0/*/events.jsonl

Usage:
    python compare_methods.py --dataset gsm8k
    python compare_methods.py --dataset algebra --model QWEN2.5-7b --seed 0
    python compare_methods.py --dataset all          # iterate every known dataset
    python compare_methods.py --dataset gsm8k --save # also write .txt and .json files
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── constants ───────────────────────────────────────────────────────────────

RUNS_ROOT = Path(__file__).parent / "runs"

BASELINE_METHODS = [
    "grpo",
    "aero",
    "gdro",
    "gdro_prompt",
    "reinforce_ada",
    "reinforce_est",
    "greso",
]

METHOD_DISPLAY = {
    "grpo":          "GRPO",
    "aero":          "AERO",
    "gdro":          "GDRO",
    "gdro_prompt":   "GDRO-Prompt",
    "reinforce_ada": "Reinforce-Ada",
    "reinforce_est": "Reinforce-Est",
    "greso":         "GRESO",
    "c_grpo":        "C-GRPO ★",
}

# Map CLI dataset name → directory slug(s) used inside the runs/ tree.
# A list means "try each in order and take the first that exists".
#
# IMPORTANT: baseline/ and conformal_grpo/ use DIFFERENT naming conventions:
#   baseline:        {model}_{slug}_seed{N}           e.g. QWEN2.5-7b_gsm8k_seed0
#   conformal_grpo:  {model}_{slug}_{config}_seed{N}  e.g. QWEN2.5-7b_gsm8k_main_seed0
#
# DS_SLUGS        → used for baseline/ runs
# DS_SLUGS_CGRPO  → used for conformal_grpo/ runs

DS_SLUGS = {
    "gsm8k":     "gsm8k",
    "algebra":   "hendrycks_math_algebra",
    "humaneval": ["humaneval", "openai_humaneval", "human_eval"],
    "mbpp":      ["mbpp_full", "mbpp_sanitized", "mbpp"],
    "gpqa":      ["gpqa_diamond", "gpqa"],
    "math500":   ["HuggingFaceH4_MATH-500", "math500", "MATH-500"],
}

DS_SLUGS_CGRPO = {
    "gsm8k":     "gsm8k_main",
    "algebra":   "hendrycks_math_algebra",
    "humaneval": ["humaneval", "openai_humaneval", "human_eval"],
    "mbpp":      ["mbpp_full", "mbpp_sanitized", "mbpp"],
    "gpqa":      ["gpqa_diamond", "gpqa"],
    "math500":   ["math500", "HuggingFaceH4_MATH-500", "MATH-500"],
}

# Known caveats per dataset — displayed in the output
DS_CAVEATS = {
    "gsm8k": (
        "⚠  COMPARISON CAVEAT: existing baselines used max_new_tokens=512 (avg_tok≈511) and "
        "n_eval=200, while C-GRPO uses 256 tokens and n_eval=400. "
        "Baselines scores are likely inflated — re-run baselines with MAX_TOK=256 N_EVAL=400 for a fair comparison."
    ),
}

# Note printed before every accuracy table
_CONFORMAL_EVAL_NOTE = (
    "ℹ  CONFORMAL EVAL NOTE: ALL methods are evaluated with conformal prediction (post-hoc, delta=0.1).\n"
    "   Only C-GRPO is TRAINED with conformal feedback — baselines use conformal at eval time only.\n"
    "   'Conf-acc' = fraction of eval examples where the correct answer is in the conformal set.\n"
    "   'Avg-k' shows set efficiency: lower is better. C-GRPO is trained to minimise k."
)

ALL_DATASETS = list(DS_SLUGS.keys())

# ── helpers ──────────────────────────────────────────────────────────────────

def _find_run_dir(root: Path, slug: str, model: str, seed: int) -> Optional[Path]:
    """Return the most-recent run sub-directory that contains a state.json."""
    pattern = str(root / f"{model}_{slug}_seed{seed}" / "*" / "state.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    # pick the latest by mtime
    best = max(matches, key=os.path.getmtime)
    return Path(best).parent


def _resolve_slug(base: Path, ds: str, model: str, seed: int):
    """Return (slug_used, run_dir) or (None, None) if not found."""
    slugs = DS_SLUGS.get(ds)
    if slugs is None:
        return None, None
    if isinstance(slugs, str):
        slugs = [slugs]
    for slug in slugs:
        rd = _find_run_dir(base, slug, model, seed)
        if rd:
            return slug, rd
    return slugs[0], None


def _load_state(run_dir: Path) -> dict:
    p = run_dir / "state.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _tail_mean(rows: list[dict], key: str, n: int = 50) -> Optional[float]:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return None
    return sum(vals[-n:]) / len(vals[-n:])


def _load_pareto_results(run_dir: Optional[Path]) -> dict:
    """Load pareto_results_*.json from the run directory tree.

    Returns the inner ``results`` dict  {k_str: {accuracy, avg_k, ...}}
    plus a top-level "conf" key if conformal results are present.
    Returns {} if no file is found.
    """
    if run_dir is None:
        return {}
    # Search the whole run-dir subtree (pareto json is written next to the ckpt)
    candidates = sorted(run_dir.rglob("pareto_results_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {}
    try:
        with open(candidates[0]) as f:
            data = json.load(f)
        return data.get("results", {})
    except Exception:
        return {}


def _collect(method_key: str, run_dir: Optional[Path]) -> dict:
    """Gather all metrics for one method/run."""
    out = {"method": method_key, "run_dir": str(run_dir) if run_dir else None}
    if run_dir is None:
        return out

    state  = _load_state(run_dir)
    events = _load_events(run_dir)

    # ── 1. Eval accuracy (from state.json or latest eval/ event) ─────────
    # Source 1: state.json "final_eval" (baselines)
    # Source 2: last eval/* row in events.jsonl (C-GRPO / any run without final_eval)
    fe = state.get("final_eval", {})
    if not fe.get("greedy_acc"):
        eval_rows = [r for r in events if "eval/greedy_acc" in r]
        if eval_rows:
            last = eval_rows[-1]
            fe = {
                "greedy_acc":     last.get("eval/greedy_acc"),
                "conf_acc":       last.get("eval/conf_acc"),
                "conf_coverage":  last.get("eval/conf_coverage"),
                "avg_k":          last.get("eval/avg_k"),
                "avg_new_tokens": last.get("eval/avg_new_tokens"),
                "k_2_pct":        last.get("eval/k_2_pct"),
                "k_16_pct":       last.get("eval/k_16_pct"),
                "step":           last.get("step"),
            }
    out["greedy_acc"]     = fe.get("greedy_acc")
    out["conf_acc"]       = fe.get("conf_acc")       # post-hoc for baselines
    out["conf_coverage"]  = fe.get("conf_coverage")
    out["avg_k_eval"]     = fe.get("avg_k")
    out["avg_new_tokens"] = fe.get("avg_new_tokens")
    out["eval_step"]      = fe.get("step", state.get("step"))
    # k2/k16 usage percentages — from explicit fields (C-GRPO) or k_usage dict (baselines)
    k2_pct  = fe.get("k_2_pct")
    k16_pct = fe.get("k_16_pct")
    if k2_pct is None or k16_pct is None:
        ku = fe.get("k_usage") or {}
        total_ku = sum(ku.values()) if ku else 0
        if total_ku > 0:
            k2_pct  = ku.get("2", 0)  / total_ku
            k16_pct = ku.get("16", 0) / total_ku
    out["k2_pct"]  = k2_pct
    out["k16_pct"] = k16_pct

    # ── 2. Training rows ──────────────────────────────────────────────────
    train_rows = [r for r in events if "train/avg_reward" in r]

    # reward / KL / loss (last 50 steps)
    out["final_reward"]   = _tail_mean(train_rows, "train/avg_reward")
    out["final_kl"]       = _tail_mean(train_rows, "train/avg_kl")
    out["final_loss_ema"] = _tail_mean(train_rows, "train/loss_ema")

    # ── 3. Rollouts per step ──────────────────────────────────────────────
    # Baselines: fixed n_rollouts logged each step
    # C-GRPO:    variable avg_k_used (the conformal set size used to generate)
    baseline_n_rollouts = [r["train/n_rollouts"] for r in train_rows
                           if "train/n_rollouts" in r]
    cgrpo_k_used        = [r["train/avg_k_used"]  for r in train_rows
                           if "train/avg_k_used"  in r]

    if baseline_n_rollouts:
        out["avg_rollouts"]       = sum(baseline_n_rollouts) / len(baseline_n_rollouts)
        out["rollout_type"]       = "fixed"
        out["avg_k_used_train"]   = None
        out["k_used_first50"]     = None
        out["k_used_last50"]      = None
        out["hard_rate"]          = None
    elif cgrpo_k_used:
        out["avg_rollouts"]       = sum(cgrpo_k_used) / len(cgrpo_k_used)
        out["rollout_type"]       = "adaptive"
        out["avg_k_used_train"]   = out["avg_rollouts"]
        n = len(cgrpo_k_used)
        h = max(1, min(50, n // 4))
        out["k_used_first50"]     = sum(cgrpo_k_used[:h])  / h
        out["k_used_last50"]      = sum(cgrpo_k_used[-h:]) / h
        hard_vals = [r.get("train/hard_rate", 0) for r in train_rows
                     if "train/hard_rate" in r]
        out["hard_rate"] = sum(hard_vals[-50:]) / len(hard_vals[-50:]) if hard_vals else None
    else:
        out["avg_rollouts"] = out["rollout_type"] = None
        out["avg_k_used_train"] = out["k_used_first50"] = out["k_used_last50"] = None
        out["hard_rate"] = None

    # ── 4. Compute cost ───────────────────────────────────────────────────
    tok_rows = [r for r in events if "train/token_cost" in r]
    if tok_rows:
        total_tokens          = sum(r["train/token_cost"] for r in tok_rows)
        out["total_tokens"]   = total_tokens
        out["total_tokens_M"] = total_tokens / 1e6
        out["avg_tok_step"]   = total_tokens / len(tok_rows)
        out["total_steps"]    = len(tok_rows)

        tps = [r["perf/tokens_per_s"] for r in tok_rows
               if r.get("perf/tokens_per_s") is not None]
        out["avg_tok_s"] = sum(tps) / len(tps) if tps else None

        sps = [r["perf/steps_per_s"] for r in tok_rows
               if r.get("perf/steps_per_s") is not None]
        out["avg_steps_s"] = sum(sps) / len(sps) if sps else None

        ts_vals = sorted([(r.get("step", 0), r["ts"]) for r in events if "ts" in r],
                         key=lambda x: x[0])
        if len(ts_vals) >= 2:
            wall_s = ts_vals[-1][1] - ts_vals[0][1]
            out["train_time_s"]   = wall_s
            out["train_time_h"]   = wall_s / 3600
            out["train_time_str"] = _fmt_time(wall_s)
        else:
            out["train_time_s"] = out["train_time_h"] = None
            out["train_time_str"] = "N/A"
    else:
        out["total_tokens"] = out["total_tokens_M"] = None
        out["avg_tok_step"] = out["total_steps"]    = None
        out["avg_tok_s"]    = out["avg_steps_s"]    = None
        out["train_time_s"] = out["train_time_h"]   = None
        out["train_time_str"] = "N/A"

    # ── 5. Derived efficiency ─────────────────────────────────────────────
    out["greedy_per_tok_M"] = (out["greedy_acc"] / out["total_tokens_M"]
                               if out.get("greedy_acc") and out.get("total_tokens_M") else None)

    # ── 6. Pass@k from pareto_results JSON ───────────────────────────────
    pareto = _load_pareto_results(run_dir)
    for _k in [1, 2, 4, 8, 16, 32, 64]:
        v = pareto.get(str(_k))
        out[f"pass_at_{_k}"] = v.get("accuracy") if v else None
    conf_v = pareto.get("conf")
    out["pareto_conf_acc"] = conf_v.get("accuracy") if conf_v else None
    out["pareto_avg_k"]    = conf_v.get("avg_k")    if conf_v else None
    out["has_pareto"]      = bool(pareto)

    return out


def _fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _f(v: Optional[float], fmt: str = ".3f") -> str:
    return f"{v:{fmt}}" if v is not None else "—"


def _best_idx(rows: list[dict], key: str, higher_is_better: bool = True) -> set[int]:
    vals = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
    if not vals:
        return set()
    best_val = max(v for _, v in vals) if higher_is_better else min(v for _, v in vals)
    return {i for i, v in vals if v == best_val}


# ── table printer ─────────────────────────────────────────────────────────────

def _hline(widths: list[int], char: str = "─", cross: str = "┼") -> str:
    return cross + cross.join(char * (w + 2) for w in widths) + cross


def _row(cells: list[str], widths: list[int], sep: str = "│") -> str:
    parts = [f" {c:<{w}} " for c, w in zip(cells, widths)]
    return sep + sep.join(parts) + sep


def _print_table(title: str, headers: list[str], rows: list[list[str]],
                 bold_cols: dict[int, set[int]] | None = None) -> list[str]:
    """Print a Unicode box table; return list of output lines."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    lines = []
    # top border
    lines.append("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐")
    # header
    lines.append(_row(headers, widths))
    lines.append("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")
    # data rows
    for ri, row in enumerate(rows):
        cells = []
        for ci, cell in enumerate(row):
            mark = bold_cols.get(ci, set()) if bold_cols else set()
            cells.append(f"*{cell}*" if ri in mark else cell)
        lines.append(_row(cells, widths))
    # bottom border
    lines.append("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")

    print(f"\n  {title}")
    for l in lines:
        print("  " + l)
    return lines


# ── main comparison function ──────────────────────────────────────────────────

def compare(dataset: str, model: str = "QWEN2.5-7b", seed: int = 0,
            save: bool = False, out_dir: Optional[Path] = None) -> dict:
    """Collect + print comparison for one dataset.  Returns raw data dict."""

    print(f"\n{'='*72}")
    print(f"  COMPARISON TABLE  |  dataset={dataset}  model={model}  seed={seed}")
    print(f"{'='*72}")

    all_data: list[dict] = []

    # ── baseline methods ────────────────────────────────────────────────
    base_root = RUNS_ROOT / "baseline"
    for meth in BASELINE_METHODS:
        slug, rd = _resolve_slug(base_root / meth, dataset, model, seed)
        d = _collect(meth, rd)
        all_data.append(d)
        status = rd.name if rd else "NOT FOUND"
        print(f"  [baseline/{meth:15s}]  {status}")

    # ── C-GRPO ──────────────────────────────────────────────────────────
    cgrpo_root = RUNS_ROOT / "conformal_grpo"
    cgrpo_slugs = DS_SLUGS_CGRPO.get(dataset, DS_SLUGS.get(dataset))
    if isinstance(cgrpo_slugs, str):
        cgrpo_slugs = [cgrpo_slugs]
    rd = None
    for sl in (cgrpo_slugs or []):
        rd = _find_run_dir(cgrpo_root, sl, model, seed)
        if rd:
            break
    d = _collect("c_grpo", rd)
    all_data.append(d)
    status = rd.name if rd else "NOT FOUND"
    print(f"  [conformal_grpo/c_grpo    ]  {status}")

    # ── check we have anything ──────────────────────────────────────────
    found = [d for d in all_data if d.get("greedy_acc") is not None]
    if not found:
        print("\n  ⚠  No results found for this dataset/model/seed combination.")
        return {"dataset": dataset, "model": model, "seed": seed, "results": all_data}

    # ── print caveats ───────────────────────────────────────────────────
    if dataset in DS_CAVEATS:
        print(f"\n  {DS_CAVEATS[dataset]}")

    # ── print conformal eval note ────────────────────────────────────────
    print()
    for line in _CONFORMAL_EVAL_NOTE.split("\n"):
        print(f"  {line}")

    # helper: display name
    def name(d: dict) -> str:
        return METHOD_DISPLAY.get(d["method"], d["method"])

    # ── TABLE 1: ACCURACY (greedy only — fair direct comparison) ─────────
    print("\n  ── MODEL ACCURACY ──────────────────────────────────────────────")
    print("  Greedy accuracy = eval the trained model with a single greedy decode.")
    print("  This is the only metric that is directly comparable across all methods.")
    hdrs1 = ["Method", "pass@1", "Avg Tokens", "Steps", "Rollout Type"]
    rows1 = []
    for d in all_data:
        rtype = d.get("rollout_type") or "—"
        rows1.append([
            name(d),
            _pct(d.get("greedy_acc")),
            _f(d.get("avg_new_tokens"), ".0f"),
            str(d.get("eval_step") or "—"),
            rtype,
        ])
    best1 = {1: _best_idx(all_data, "greedy_acc")}
    _print_table("1. pass@1 ACCURACY  (trained model, greedy single-pass decode)", hdrs1, rows1, best1)

    # ── TABLE 2: TRAINING EFFICIENCY ─────────────────────────────────────
    print("\n  ── TRAINING EFFICIENCY ─────────────────────────────────────────")
    print("  Rollouts/step: baselines use a fixed budget; C-GRPO uses adaptive k.")
    print("  C-GRPO 'k-first' vs 'k-last' shows whether the model learned to need")
    print("  fewer rollouts over time (key claim: avg_k decreases during training).")
    hdrs2 = ["Method", "Rollouts/step", "k-first", "k-last",
             "Hard-rate", "Tok/step", "Tok/s", "Wall-time", "Total Tok (M)"]
    rows2 = []
    for d in all_data:
        rows2.append([
            name(d),
            _f(d.get("avg_rollouts"), ".2f"),
            _f(d.get("k_used_first50"), ".2f"),
            _f(d.get("k_used_last50"),  ".2f"),
            _pct(d.get("hard_rate"))   if d.get("hard_rate") is not None else "—",
            _f(d.get("avg_tok_step"),  ".0f"),
            _f(d.get("avg_tok_s"),     ".1f"),
            d.get("train_time_str") or "—",
            _f(d.get("total_tokens_M"), ".2f"),
        ])
    best2 = {
        1: _best_idx(all_data, "avg_rollouts",   higher_is_better=False),
        5: _best_idx(all_data, "avg_tok_step",   higher_is_better=False),
        6: _best_idx(all_data, "avg_tok_s",      higher_is_better=True),
        7: _best_idx(all_data, "train_time_h",   higher_is_better=False),
        8: _best_idx(all_data, "total_tokens_M", higher_is_better=False),
    }
    _print_table("2. TRAINING EFFICIENCY", hdrs2, rows2, best2)

    # ── TABLE 3: CONFORMAL PREDICTION QUALITY ────────────────────────────
    print("\n  ── CONFORMAL PREDICTION QUALITY ────────────────────────────────")
    print("  C-GRPO is *trained* to minimise k (the conformal set size).")
    print("  Baselines: conformal is applied post-hoc to the frozen trained model.")
    print("  'Conf-acc' = P(correct answer ∈ conformal set) at δ=0.1.")
    print("  Lower avg-k = model is more confident/consistent = cheaper deployment.")
    hdrs3 = ["Method", "Conf-acc", "Coverage", "Avg-k", "k=2 use%", "k=16 use%", "Trained-for-k?"]
    rows3 = []
    for d in all_data:
        trained = "YES ✓" if d["method"] == "c_grpo" else "no (post-hoc)"
        rows3.append([
            name(d),
            _pct(d.get("conf_acc")),
            _pct(d.get("conf_coverage")),
            _f(d.get("avg_k_eval"), ".2f"),
            _pct(d.get("k2_pct")),
            _pct(d.get("k16_pct")),
            trained,
        ])
    best3 = {
        1: _best_idx(all_data, "conf_acc"),
        2: _best_idx(all_data, "conf_coverage"),
        3: _best_idx(all_data, "avg_k_eval", higher_is_better=False),
        4: _best_idx(all_data, "k2_pct",     higher_is_better=True),
        5: _best_idx(all_data, "k16_pct",    higher_is_better=False),
    }
    _print_table("3. CONFORMAL QUALITY  (C-GRPO trained / baselines post-hoc)",
                 hdrs3, rows3, best3)

    # ── TABLE 4: TRAINING DYNAMICS ────────────────────────────────────────
    hdrs4 = ["Method", "Final-Reward", "Final-KL", "Loss-EMA", "Greedy/tok-M"]
    rows4 = []
    for d in all_data:
        rows4.append([
            name(d),
            _f(d.get("final_reward")),
            _f(d.get("final_kl"), ".5f"),
            _f(d.get("final_loss_ema")),
            _f(d.get("greedy_per_tok_M"), ".4f"),
        ])
    best4 = {
        1: _best_idx(all_data, "final_reward"),
        2: _best_idx(all_data, "final_kl",          higher_is_better=False),
        4: _best_idx(all_data, "greedy_per_tok_M",  higher_is_better=True),
    }
    _print_table("4. TRAINING DYNAMICS & COMPUTE EFFICIENCY", hdrs4, rows4, best4)

    # ── TABLE 5: PASS@K COMPARISON ───────────────────────────────────────
    if any(d.get("has_pareto") for d in all_data):
        print("\n  ── PASS@K FIXED-K vs ADAPTIVE ──────────────────────────────────────")
        print("  Fixed-k oracle: best accuracy achievable by always generating exactly k samples.")
        print("  pass@1 = greedy decode;  pass@k (k>1) = any-correct over k independent samples.")
        print("  C-GRPO adaptive-k: conformal set size per query; acc = same target, lower avg k.")
        hdrs5 = ["Method", "pass@1", "pass@2", "pass@4", "pass@8",
                 "pass@16", "pass@32", "pass@64", "C-GRPO-acc", "Avg-k"]
        rows5 = []
        for d in all_data:
            rows5.append([
                name(d),
                _pct(d.get("pass_at_1")),
                _pct(d.get("pass_at_2")),
                _pct(d.get("pass_at_4")),
                _pct(d.get("pass_at_8")),
                _pct(d.get("pass_at_16")),
                _pct(d.get("pass_at_32")),
                _pct(d.get("pass_at_64")),
                _pct(d.get("pareto_conf_acc")),
                _f(d.get("pareto_avg_k"), ".2f"),
            ])
        best5 = {
            1: _best_idx(all_data, "pass_at_1"),
            2: _best_idx(all_data, "pass_at_2"),
            3: _best_idx(all_data, "pass_at_4"),
            4: _best_idx(all_data, "pass_at_8"),
            5: _best_idx(all_data, "pass_at_16"),
            6: _best_idx(all_data, "pass_at_32"),
            7: _best_idx(all_data, "pass_at_64"),
            8: _best_idx(all_data, "pareto_conf_acc"),
            9: _best_idx(all_data, "pareto_avg_k", higher_is_better=False),
        }
        _print_table("5. PASS@K COMPARISON  (fixed-k oracle   vs   C-GRPO adaptive)",
                     hdrs5, rows5, best5)

    # ── SUMMARY ────────────────────────────────────────────────────────────
    print("\n  6. SUMMARY  (* = best in column)")
    metrics_sum = [
        ("greedy_acc",      "pass@1 acc",             True,  _pct),
        ("avg_rollouts",    "Avg rollouts/step (↓)",  False, lambda v: _f(v, ".2f")),
        ("k_used_last50",   "k-last 50 steps (↓)",   False, lambda v: _f(v, ".2f")),
        ("avg_tok_step",    "Tok/step (↓)",           False, lambda v: _f(v, ".0f")),
        ("avg_tok_s",       "Throughput tok/s (↑)",   True,  lambda v: _f(v, ".1f")),
        ("train_time_h",    "Wall-time h (↓)",        False, lambda v: _f(v, ".2f")),
        ("total_tokens_M",  "Total tok M (↓)",        False, lambda v: _f(v, ".2f")),
        ("greedy_per_tok_M","pass@1/tok-M (↑)",       True,  lambda v: _f(v, ".4f")),
        ("conf_acc",        "Conf-acc (post-hoc ↑)",  True,  _pct),
        ("avg_k_eval",      "Eval avg-k (↓)",         False, lambda v: _f(v, ".2f")),
        ("pareto_conf_acc", "Pareto-conf-acc (↑)",    True,  _pct),
        ("pareto_avg_k",    "Pareto avg-k (↓)",       False, lambda v: _f(v, ".2f")),
    ]
    for key, label, hib, fmt in metrics_sum:
        vals = [(d, d.get(key)) for d in all_data if d.get(key) is not None]
        if not vals:
            continue
        best_v   = max(v for _, v in vals) if hib else min(v for _, v in vals)
        winners  = [name(d) for d, v in vals if v == best_v]
        print(f"    {label:<30s}  ★ {', '.join(winners):<22s}  ({fmt(best_v)})")

    # ── optional save ──────────────────────────────────────────────────
    result = {"dataset": dataset, "model": model, "seed": seed, "results": all_data}
    if save:
        od = out_dir or Path(".")
        od.mkdir(parents=True, exist_ok=True)
        stem = f"compare_{dataset}_{model}_seed{seed}"
        json_path = od / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Saved → {json_path}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare GRPO-variant methods on a dataset."
    )
    parser.add_argument(
        "--dataset", default="gsm8k",
        help=f"Dataset key or 'all'.  Choices: {ALL_DATASETS} | all"
    )
    parser.add_argument("--model",  default="QWEN2.5-7b")
    parser.add_argument("--seed",   default=0, type=int)
    parser.add_argument("--save",   action="store_true",
                        help="Save results to compare_<ds>_<model>_seed<N>.json")
    parser.add_argument("--out-dir", default=".", type=Path,
                        help="Directory for saved output files (default: .)")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        if ds not in DS_SLUGS:
            print(f"  Unknown dataset '{ds}'.  Known: {ALL_DATASETS}")
            sys.exit(1)
        compare(ds, model=args.model, seed=args.seed,
                save=args.save, out_dir=args.out_dir)

    print()


if __name__ == "__main__":
    main()
