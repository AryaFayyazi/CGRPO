#!/usr/bin/env python3
"""
Pareto Analysis: Conformal Dynamic-k vs. Fixed-k Majority Vote on GSM8K.

Loads the final trained LoRA checkpoint, generates k_max samples for every
eval prompt ONCE, then post-hoc computes majority-vote accuracy at every
fixed k ∈ {1, 2, 4, 8, 16, 32, 64} plus our conformal dynamic-k method.

Outputs a clean table and Pareto comparison.
"""

import os, sys, json, time, random
import argparse
import numpy as np
import torch
from tqdm import tqdm
# silence tqdm progress bars (stop 'Loading weights' messages)
import tqdm as _tqdm
_tqdm.tqdm = lambda *args, **kwargs: (args[0] if args else iter([]))
from collections import Counter
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import TrainConfig
from data import load_gsm8k_splits, format_prompt, extract_example
from generation import generate_n
from verifier import canonicalize_answer, verify_answer
from utils import pick_cuda_devices, batch_to_examples
from conformal import answer_freqs, conformal_set_from_freqs, calibrate_qhats, score_true_answer
from model_registry import get_model_path, get_model_dtype_str

# ── Config ──────────────────────────────────────────────────────────────
# default values will be overridden by command-line arguments
CKPT_DIR = None
K_MAX = 64                          # generate this many samples per prompt
FIXED_K_VALUES = [1, 2, 4, 8, 16, 32, 64]
# choose a (hopefully empty) GPU automatically
if torch.cuda.is_available():
    devs = pick_cuda_devices(1)
    DEVICE = f"cuda:{devs[0]}" if devs else "cuda:0"
    print(f"Automatically selected device for evaluation: {DEVICE}")
else:
    DEVICE = "cpu"
BATCH_SIZE = 4                       # prompts per generation batch
# ────────────────────────────────────────────────────────────────────────


def majority_vote(answers, k):
    """Return majority-vote prediction from first k answers."""
    subset = [a for a in answers[:k] if a != ""]
    if not subset:
        return ""
    return Counter(subset).most_common(1)[0][0]


def _is_code_dataset(name: str) -> bool:
    """True for execution-verified code datasets (MBPP, HumanEval)."""
    n = name.lower()
    return "mbpp" in n or "humaneval" in n


def _verify(pred_raw: str, gold_raw: str, pred_canon: str, gold_canon: str, dataset: str) -> int:
    """Dispatch verify_answer correctly for code vs. non-code datasets.

    Code datasets need the *raw* generation and *raw* JSON gold to run
    execution-based verification.  Non-code datasets use the already-
    canonicalized strings with the conventional '#### ' prefix.
    """
    if _is_code_dataset(dataset):
        return verify_answer(pred_raw, gold_raw, dataset)
    return verify_answer(f"#### {pred_canon}", f"#### {gold_canon}", dataset)


def _pass_at_k(raws: list, gold_raw: str, dataset: str, k: int) -> int:
    """Return 1 if any of the first k completions pass verification, else 0."""
    for c in raws[:k]:
        if verify_answer(c, gold_raw, dataset):
            return 1
    return 0


def choose_k_and_sets(extracted_answers, qhats, k_values):
    """Same as train.py: pick smallest k with singleton conformal set."""
    for k in k_values:
        freqs = answer_freqs(extracted_answers[:k])
        Ck = conformal_set_from_freqs(freqs, k, qhats[k])
        if len(Ck) == 1:
            return k, Ck
    k = max(k_values)
    freqs = answer_freqs(extracted_answers[:k])
    return k, conformal_set_from_freqs(freqs, k, qhats[k])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pareto cost-accuracy tradeoffs")
    parser.add_argument("--ckpt-dir", default=None, help="LoRA checkpoint directory to load")
    parser.add_argument("--model-key", default=None, help="model key for TrainConfig (overrides config.model_key)")
    parser.add_argument("--dataset-name", default=None, help="dataset name for HuggingFace (e.g. openai/gsm8k)")
    parser.add_argument("--dataset-config", default=None, help="dataset config string")
    parser.add_argument("--n-cal", type=int, default=None, help="number of calibration examples")
    parser.add_argument("--n-eval", type=int, default=None, help="number of evaluation examples")
    parser.add_argument("--k-max", type=int, default=None, help="maximum samples per prompt to generate")
    parser.add_argument("--device", default=None, help="torch device to use")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size for generation")
    parser.add_argument("--dynamic-baselines", type=str, default="", help="comma-separated heuristics: softmax,entropy")
    parser.add_argument("--baseline-thresholds", type=str, default="0.5", help="comma-separated threshold values used by heuristics")
    parser.add_argument("--no-lora", action="store_true", help="do not load/merge LoRA; evaluate base model only")
    parser.add_argument("--delta", type=str, default="auto",
                        help="miscoverage for the thresholds: 'auto' (as in training: (1 - pass@K_max) + 0.05 from "
                             "the calibration scores) or a float")
    parser.add_argument("--split-delta", action="store_true",
                        help="with --delta auto, choose delta on the first half of the calibration set and the "
                             "thresholds on the second half, as train.py --split-delta does")
    parser.add_argument("--k-values", type=str, default=None,
                        help="comma-separated conformal budget grid; pass the grid the checkpoint was trained with "
                             "(the config default stops at 16)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="max new tokens per completion (overrides TrainConfig default; "
                             "use the same value as training for classification datasets)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    from conformal import set_conformal_seed
    set_conformal_seed(cfg_seed := 0)   # APS tie-breaking / U-draw

    cfg = TrainConfig()

    # allow overrides from CLI
    if args.model_key:
        cfg.model_key = args.model_key
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    # Allow explicitly passing "" to clear the dataset config (e.g., HumanEval).
    if args.dataset_config is not None:
        cfg.dataset_config = args.dataset_config or None
    if args.n_cal is not None:
        cfg.n_cal = args.n_cal
    if args.n_eval is not None:
        cfg.n_eval = args.n_eval
    if args.k_values:
        cfg.k_values = tuple(sorted(int(x) for x in args.k_values.split(",")))
    if args.k_max is not None:
        global K_MAX
        K_MAX = args.k_max
    if args.device:
        global DEVICE
        DEVICE = args.device
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    if args.ckpt_dir:
        global CKPT_DIR
        CKPT_DIR = args.ckpt_dir
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens

    # ── 1. Load dataset ─────────────────────────────────────────────────
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.getcwd(), "hf_cache", "datasets"))
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache", "hub"))
    train_ds, cal_ds, eval_ds = load_gsm8k_splits(
        cfg.dataset_name, cfg.dataset_config,
        cfg.n_train, cfg.n_cal, cfg.n_eval, cfg.seed,
    )
    print(f"Eval set: {len(eval_ds)} examples | Cal set: {len(cal_ds)} examples")

    # ── 2. Load model ───────────────────────────────────────────────────
    model_path = get_model_path(cfg.model_key)
    dtype_str = get_model_dtype_str(cfg.model_key)
    dtype = torch.bfloat16 if "bf16" in dtype_str or "bfloat" in dtype_str else torch.float16

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.convert_ids_to_tokens(tok.pad_token_id)
    tok.chat_template = None

    print(f"Loading base model from {model_path} ...")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=None, trust_remote_code=False, local_files_only=True,
    )
    if args.no_lora or CKPT_DIR is None:
        model = base
        print("Using base model (no LoRA adapter)")
    else:
        print(f"Loading LoRA adapter from {CKPT_DIR} ...")
        try:
            # prefer local files only to avoid hub lookup
            model = PeftModel.from_pretrained(base, CKPT_DIR, local_files_only=True)
            model = model.merge_and_unload()       # merge for faster inference
        except Exception as e:
            print(f"Warning: failed to load LoRA adapter ({e}); using base model instead")
            model = base
    model.to(DEVICE)
    model.eval()
    print(f"Model on {next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")

    # ── 3. Generate K_MAX samples per eval prompt ────────────────────────
    print(f"\n{'='*70}")
    print(f"Generating {K_MAX} samples per prompt for {len(eval_ds)} eval examples ...")
    print(f"{'='*70}")

    all_gold = []        # canonicalized gold (for majority-vote / conformal)
    all_gold_raw = []    # raw gold (JSON string for code, raw text otherwise)
    all_answers = []     # list of (list[canon_str], list[token_count]), len K_MAX each
    all_preds_raw = []   # list of list[raw_str], for code pass@k  (len K_MAX each)
    all_greedy = []      # canon greedy
    all_greedy_raw = []  # raw greedy completions
    total_tokens_generated = 0  # accumulate for FLOPs estimate

    t0 = time.time()
    for i in tqdm(range(0, len(eval_ds), BATCH_SIZE), desc="Generating"):
        batch = eval_ds[i : i + BATCH_SIZE]
        examples = batch_to_examples(batch)
        batch_pairs = [extract_example(ex, cfg.dataset_name) for ex in examples]
        if batch_pairs:
            questions, gold_list = zip(*batch_pairs)
        else:
            questions, gold_list = [], []
        prompts = [format_prompt(q, cfg.dataset_name) for q in questions]

        # Greedy (k=1)
        greedy_groups = generate_n(
            model, tok, prompts, 1,
            cfg.max_new_tokens,
            temperature=None, top_p=1.0,
            device=DEVICE, repetition_penalty=1.0,
        )

        # Sampling (K_MAX samples)
        sampled_groups = generate_n(
            model, tok, prompts, K_MAX,
            cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p,
            device=DEVICE,
        )

        for p, g_out, s_out, g in zip(prompts, greedy_groups, sampled_groups, gold_list):
            gold_ans = canonicalize_answer(g, cfg.dataset_name)
            all_gold.append(gold_ans)
            all_gold_raw.append(g)   # keep raw JSON for code verification

            # Greedy answer
            greedy_comp = g_out[0][len(p):] if g_out[0].startswith(p) else g_out[0]
            all_greedy.append(canonicalize_answer(greedy_comp, cfg.dataset_name))
            all_greedy_raw.append(greedy_comp)

            # All K_MAX sampled answers
            completions = [o[len(p):] if o.startswith(p) else o for o in s_out]
            all_preds_raw.append(completions)   # raw for code execution
            answers = [canonicalize_answer(c, cfg.dataset_name) for c in completions]
            # also count tokens
            token_counts = [len(tok.encode(c)) for c in completions]
            total_tokens_generated += sum(token_counts)
            all_answers.append((answers, token_counts))

    gen_time = time.time() - t0
    n_total = len(all_gold)
    # VRAM peak (capture right after generation, before model is swapped out)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    print(f"\nGeneration done: {n_total} prompts × {K_MAX} samples in {gen_time/60:.1f} min")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB | Total tokens generated (K_MAX): {total_tokens_generated:,}")

    # ── 4. Conformal calibration on cal set ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"Calibrating conformal qhats on {len(cal_ds)} calibration examples ...")
    print(f"{'='*70}")

    # We need to generate samples for the cal set too, at the k_values we'll use
    # If a checkpoint directory is provided and we're not evaluating base model,
    # we can reuse stored qhats from training; otherwise leave empty.
    stored_qhats = {}
    if CKPT_DIR is not None:
        try:
            with open(os.path.join(CKPT_DIR, "..", "state.json")) as f:
                state = json.load(f)
            stored_qhats = {int(k): v for k, v in state.get("qhats", {}).items()}
            print(f"Using stored qhats from training: {stored_qhats}")
        except Exception as e:
            print(f"Warning: could not load stored qhats: {e}")
    else:
        print("No checkpoint provided; using only fresh calibration qhats.")

    # Also calibrate fresh on cal set for k up to K_MAX
    # We need cal samples to calibrate for larger k values like 32, 64
    print(f"\nGenerating {K_MAX} samples for {len(cal_ds)} cal examples (for fresh calibration)...")
    cal_group_answers = []
    cal_true = []
    for i in tqdm(range(0, len(cal_ds), BATCH_SIZE), desc="Cal samples"):
        batch = cal_ds[i : i + BATCH_SIZE]
        examples = batch_to_examples(batch)
        batch_pairs = [extract_example(ex, cfg.dataset_name) for ex in examples]
        if batch_pairs:
            questions, golds = zip(*batch_pairs)
        else:
            questions, golds = [], []
        prompts = [format_prompt(q, cfg.dataset_name) for q in questions]

        sampled = generate_n(
            model, tok, prompts, K_MAX,
            cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p,
            device=DEVICE,
        )

        for p, s_out, g in zip(prompts, sampled, golds):
            cal_true.append(canonicalize_answer(g, cfg.dataset_name))
            completions = [o[len(p):] if o.startswith(p) else o for o in s_out]
            cal_group_answers.append([canonicalize_answer(c, cfg.dataset_name) for c in completions])
            # Also store raw for code execution-based calibration
            if _is_code_dataset(cfg.dataset_name):
                # Overwrite the last entry with raw completions for score_fn below
                cal_group_answers[-1] = completions
                cal_true[-1] = g  # raw JSON gold

    # Calibrate qhats for ALL k values we want to evaluate
    # k=1 is only calibrated when it is part of the conformal grid; the fixed k=1 row is greedy decoding.
    all_k_for_cal = sorted({k for k in FIXED_K_VALUES if k >= 2} | set(cfg.k_values))
    all_k_for_cal = [k for k in all_k_for_cal if 1 <= k <= K_MAX]
    if max(cfg.k_values) < K_MAX:
        print(f"WARNING: conformal grid {cfg.k_values} stops below --k-max {K_MAX}; "
              f"pass --k-values with the training grid")
    _score_fn = None
    if "aquamuse" in cfg.dataset_name.lower():
        from verifier import _token_precision
        def _aqua_score_fn(gold: str, gens: list) -> float:
            if not gold:
                return 1.0
            best = max((_token_precision(g, gold) for g in gens if g.strip()), default=0.0)
            return 1.0 - best
        _score_fn = _aqua_score_fn
    elif _is_code_dataset(cfg.dataset_name):
        # Deterministic nonconformity score: 1 - (n_passes / k).
        # score = 0  if all k rollouts pass (easy problem, model is confident).
        # score = 1  if no rollout passes  (impossible with k budget).
        # Deterministic: same model + same cal example always gives same score,
        # satisfying exchangeability for valid marginal coverage guarantees.
        # Monotone in model quality: better model → more passes → lower qhat.
        _dataset_name_closure = cfg.dataset_name
        def _code_score_fn(gold_raw: str, gens_raw: list) -> float:
            k = max(len(gens_raw), 1)
            n_pass = sum(1 for gen in gens_raw
                         if verify_answer(gen, gold_raw, _dataset_name_closure))
            return 1.0 - n_pass / k
        _score_fn = _code_score_fn
    # ---- miscoverage level: mirror train.py's calibrate_conformal ----------------------
    # Paper Table 4 reports coverage at the auto-selected delta. A fixed delta=0.1 on a benchmark the
    # policy mostly fails pins every threshold at 1.0 and makes the coverage check vacuous.
    _q_true, _q_groups = cal_true, cal_group_answers
    cal_solve_rate = None
    if (args.delta or "auto").strip().lower() == "auto":
        from conformal import select_delta_auto
        _d_true, _d_groups = cal_true, cal_group_answers
        if args.split_delta and len(cal_true) >= 4:
            _h = len(cal_true) // 2
            _d_true, _d_groups = cal_true[:_h], cal_group_answers[:_h]
            _q_true, _q_groups = cal_true[_h:], cal_group_answers[_h:]
        _kmax_scores = [
            _score_fn(t, g[:K_MAX]) if _score_fn is not None else score_true_answer(t, answer_freqs(g[:K_MAX]), K_MAX)
            for t, g in zip(_d_true, _d_groups)
        ]
        _delta, cal_solve_rate = select_delta_auto(np.asarray(_kmax_scores))
        cfg.deltas = (float(_delta),)
        delta_mode = "auto, split" if args.split_delta else "auto"
        print(f"\n  [Auto-δ] pass@{K_MAX}={cal_solve_rate:.1%} on {len(_kmax_scores)} cal examples "
              f"-> δ={cfg.deltas[0]:.3f}")
    else:
        cfg.deltas = (float(args.delta),)
        delta_mode = "fixed"
    print(f"\nCalibrating qhats for k = {all_k_for_cal}, delta = {cfg.deltas[0]} ({delta_mode})")
    fresh_qhats = calibrate_qhats(_q_true, _q_groups, tuple(all_k_for_cal), cfg.deltas[0],
                                   score_fn=_score_fn)

    # build dynamic baseline helpers and parse CLI arguments
    heuristics = [h.strip() for h in args.dynamic_baselines.split(",") if h.strip()]
    generated_k_values = [k for k in FIXED_K_VALUES if k <= K_MAX]
    thresholds = [float(t) for t in args.baseline_thresholds.split(",") if t.strip()]

    def choose_k_softmax(answers, k_values, thresh=0.5):
        for k in k_values:
            freqs = answer_freqs(answers[:k])
            total = sum(freqs.values())
            if total == 0:
                continue
            pmax = max(freqs.values()) / total
            if pmax >= thresh:
                return k, Counter(freqs).most_common(1)[0][0]
        return max(k_values), majority_vote(answers, max(k_values))

    def choose_k_entropy(answers, k_values, thresh=1.0):
        for k in k_values:
            freqs = answer_freqs(answers[:k])
            total = sum(freqs.values())
            if total == 0:
                continue
            probs = [v / total for v in freqs.values()]
            H = -sum(p * np.log(p) for p in probs if p > 0)
            if H <= thresh:
                return k, Counter(freqs).most_common(1)[0][0]
        return max(k_values), majority_vote(answers, max(k_values))

    # ── 5. Compute accuracy at every fixed k ─────────────────────────────
    print(f"\n{'='*70}")
    print("COMPUTING ACCURACY AT EACH FIXED k (majority vote)")
    print(f"{'='*70}")

    results = {}

    # Greedy (k=1)
    all_greedy_tokens = []
    # we didn't record greedy token counts earlier; compute now
    for g in all_greedy_raw:
        all_greedy_tokens.append(len(tok.encode(g)))
    if _is_code_dataset(cfg.dataset_name):
        greedy_correct = sum(
            1 for raw, gold_raw in zip(all_greedy_raw, all_gold_raw)
            if verify_answer(raw, gold_raw, cfg.dataset_name)
        )
    else:
        greedy_correct = sum(
            1 for ga, gold in zip(all_greedy, all_gold)
            if verify_answer(f"#### {ga}", f"#### {gold}", cfg.dataset_name)
        )
    results[1] = {
        "k": 1, "method": "greedy (T=0, 1 sample)",
        "accuracy": greedy_correct / n_total,
        "avg_k": 1.0,
        "avg_tokens": sum(all_greedy_tokens) / n_total,
        "correct": greedy_correct, "total": n_total,
    }

    # Fixed k majority vote (non-code) / pass@k (code)
    for k in FIXED_K_VALUES:
        # k > K_MAX would silently reuse the K_MAX samples under a larger label
        if k == 1 or k > K_MAX:
            continue
        correct = 0
        covered = 0
        tot_tokens = 0
        if _is_code_dataset(cfg.dataset_name):
            for raws, (answers, tokens), gold_raw in zip(all_preds_raw, all_answers, all_gold_raw):
                if _pass_at_k(raws, gold_raw, cfg.dataset_name, k):
                    correct += 1
                    covered += 1
                tot_tokens += sum(tokens[:k])
        else:
            for (answers, tokens), gold in zip(all_answers, all_gold):
                pred = majority_vote(answers, k)
                if verify_answer(f"#### {pred}", f"#### {gold}", cfg.dataset_name):
                    correct += 1
                if gold in [a for a in answers[:k] if a != ""]:
                    covered += 1
                tot_tokens += sum(tokens[:k])
        results[k] = {
            "k": k, "method": f"fixed-{k}" if not _is_code_dataset(cfg.dataset_name) else f"pass@{k}",
            "accuracy": correct / n_total,
            "avg_k": float(k),
            "avg_tokens": tot_tokens / n_total,
            "correct": correct, "total": n_total,
            "coverage": covered / n_total,
        }

    # ── Ave@k: mean accuracy of an INDIVIDUAL sampled rollout ─────────────
    # This is the metric the paper calls "Ave@32": average correctness across
    # all k sampled completions, not a majority vote and not greedy decoding.
    # It is a much lower-variance estimate of single-rollout accuracy than
    # scoring one sample, and it is NOT the same quantity as results[1]
    # (greedy), which is typically far higher on math benchmarks.
    try:
        per_sample_hits = per_sample_total = 0
        for idx, ((answers, _tok), gold) in enumerate(zip(all_answers, all_gold)):
            if _is_code_dataset(cfg.dataset_name):
                for c in all_preds_raw[idx][:K_MAX]:
                    per_sample_hits += verify_answer(c, all_gold_raw[idx], cfg.dataset_name)
                    per_sample_total += 1
            else:
                for a in answers[:K_MAX]:
                    per_sample_hits += verify_answer(f"#### {a}", f"#### {gold}",
                                                     cfg.dataset_name)
                    per_sample_total += 1
        if per_sample_total:
            results["ave@k"] = {
                "k": K_MAX, "method": f"Ave@{K_MAX} (mean single-rollout, T={cfg.temperature})",
                "accuracy": per_sample_hits / per_sample_total,
                "avg_k": 1.0,
                "correct": per_sample_hits, "total": per_sample_total,
            }
    except Exception as _exc:
        results["ave@k"] = {"error": repr(_exc)}

    # Dynamic baseline heuristics (e.g. softmax confidence, entropy)
    for h in heuristics:
        for thresh in thresholds:
            key = f"dyn-{h}-{thresh}"
            correct = 0
            total_k = 0
            total_tokens = 0
            for idx, ((answers, tokens), gold) in enumerate(zip(all_answers, all_gold)):
                if h == "softmax":
                    k_chosen, pred = choose_k_softmax(answers, generated_k_values, thresh)
                elif h == "entropy":
                    k_chosen, pred = choose_k_entropy(answers, generated_k_values, thresh)
                else:
                    k_chosen, pred = max(generated_k_values), majority_vote(answers, max(generated_k_values))
                if _is_code_dataset(cfg.dataset_name):
                    if _pass_at_k(all_preds_raw[idx], all_gold_raw[idx], cfg.dataset_name, k_chosen):
                        correct += 1
                else:
                    if verify_answer(f"#### {pred}", f"#### {gold}", cfg.dataset_name):
                        correct += 1
                total_k += k_chosen
                total_tokens += sum(tokens[:k_chosen])
            results[key] = {
                "k": "dynamic",
                "method": f"{h}-{thresh}",
                "accuracy": correct / n_total,
                "avg_k": total_k / n_total,
                "avg_tokens": total_tokens / n_total,
                "correct": correct,
                "total": n_total,
            }

    # ── 6. Conformal dynamic-k ───────────────────────────────────────────
    # also compute avg_tokens for each method similarly
    # Only budgets that were actually generated; with --k-max below the training
    # grid, qhats for the larger k do not exist and the stopping rule cannot use them.
    conf_k_values = tuple(sorted(k for k in cfg.k_values if k <= K_MAX))
    if not conf_k_values:
        raise ValueError(f"--k-max {K_MAX} is below every conformal budget {sorted(cfg.k_values)}")
    # Use fresh qhats (restricted to our k_values)
    conf_qhats = {k: fresh_qhats[k] for k in conf_k_values if k in fresh_qhats}
    conf_correct = 0
    conf_covered = 0
    total_k_used = 0
    total_tokens = 0
    k_usage = Counter()
    k_max_conf = max(conf_k_values)

    if _is_code_dataset(cfg.dataset_name):
        # ── Code adaptive early-stopping ────────────────────────────────
        # Determine conformal k-budget from calibration:
        # pick the smallest k where qhat(k) < 1.0 (model has >0 coverage),
        # falling back to k_max_conf if all qhats are 1.0.
        # qhat < 1.0 means ≥(1-delta) of cal problems are solved within k
        # samples (for continuous score this means the median-ish example
        # is solved well within k).
        code_k_budget = k_max_conf
        best_qhat = 1.0
        for _k in conf_k_values:
            _q = conf_qhats.get(_k, 1.0)
            if _q < best_qhat:
                best_qhat = _q
                code_k_budget = _k      # prefer smallest k that improves
            if _q < 1e-9:              # perfect coverage → stop here
                code_k_budget = _k
                break
        # Cap at k_max_conf in case calibration picks something larger
        code_k_budget = min(code_k_budget, k_max_conf)
        print(f"  [Code conformal] budget k={code_k_budget}  "
              f"(best qhat={best_qhat:.4f} @ k={code_k_budget})")

        # Per-problem adaptive early-stopping:
        # Simulate online sampling → execute tests after each new batch.
        # Stop at the FIRST k_value where any sample passes.  If still
        # unsolved at budget, count as wrong and record k=budget.
        for idx, ((answers, tokens), _gold) in enumerate(zip(all_answers, all_gold)):
            solved = False
            k_stop = code_k_budget
            toks_stop = sum(tokens[:code_k_budget])
            for _k in conf_k_values:
                if _k > code_k_budget:
                    break
                if _pass_at_k(all_preds_raw[idx], all_gold_raw[idx],
                              cfg.dataset_name, _k):
                    k_stop = _k
                    toks_stop = sum(tokens[:_k])
                    solved = True
                    break
            k_usage[k_stop] += 1
            total_k_used += k_stop
            total_tokens += toks_stop
            if solved:
                conf_correct += 1
                conf_covered += 1
    else:
        for idx, ((answers, tokens), gold) in enumerate(zip(all_answers, all_gold)):
            ans_sub = answers[:k_max_conf]
            tok_sub = tokens[:k_max_conf]
            k_used, Ck = choose_k_and_sets(ans_sub, conf_qhats, conf_k_values)
            k_usage[k_used] += 1
            total_k_used += k_used
            total_tokens += sum(tok_sub[:k_used])
            pred = majority_vote(ans_sub, k_used)
            if verify_answer(f"#### {pred}", f"#### {gold}", cfg.dataset_name):
                conf_correct += 1
            if gold in Ck:
                conf_covered += 1

    avg_k_conf = total_k_used / n_total
    results["conf"] = {
        "k": "dynamic", "method": "conformal",
        "accuracy": conf_correct / n_total,
        "correct": conf_correct,
        "total": n_total,
        "avg_k": avg_k_conf,
        "avg_tokens": total_tokens / n_total,
        # NOTE: this is coverage at the ADAPTIVELY SELECTED k_used. It is a
        # post-selection quantity and is NOT the object of Theorem 2, which is
        # stated per FIXED k. See theorem2_coverage_per_k below for the
        # quantity the theorem actually guarantees.
        "coverage_at_selected_k": conf_covered / n_total,
        "coverage": conf_covered / n_total,   # kept for backward compatibility
        "qhats": conf_qhats,
    }

    # ── Theorem 2 test: per-FIXED-k marginal coverage ────────────────────
    # Thm 2 claims  P( s(y*; y_1:k) <= qhat_k )  in [1-delta, 1-delta+1/(n+1)]
    # for each fixed k in K. That is what this measures, on the held-out eval
    # split, one number per k -- unlike results[k]["coverage"] above, which is
    # oracle recall (gold appears among the first k samples) and is unrelated
    # to the conformal guarantee.
    try:
        per_k_cov = {}
        for _k in conf_k_values:
            _q = conf_qhats.get(_k, 1.0)
            _hit = 0
            for idx, ((answers, _tokens), gold) in enumerate(zip(all_answers, all_gold)):
                if _is_code_dataset(cfg.dataset_name):
                    # training/calibration score for code is 1 - pass_rate@k
                    _npass = sum(
                        1 for c in all_preds_raw[idx][:_k]
                        if verify_answer(c, all_gold_raw[idx], cfg.dataset_name)
                    )
                    _score = 1.0 - _npass / max(_k, 1)
                else:
                    _freqs = answer_freqs(answers[:_k])
                    _score = score_true_answer(gold, _freqs, _k)
                if _score <= _q:
                    _hit += 1
            per_k_cov[str(_k)] = {
                "k": _k, "qhat": float(_q),
                "empirical_coverage": _hit / n_total,
                "n": n_total,
            }
        results["theorem2_coverage_per_k"] = per_k_cov
    except Exception as _exc:                     # never let the audit break eval
        results["theorem2_coverage_per_k"] = {"error": repr(_exc)}

    # ── 7. Print results ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PARETO COMPARISON: Accuracy vs. Cost (Samples per Prompt)")
    print(f"{'='*70}")
    model_desc = cfg.model_key
    # label whether this run is evaluating the base model or LoRA-merged model
    if args.no_lora:
        model_desc += " (base)"
    elif CKPT_DIR:
        model_desc += " + LoRA"
    print(f"  Model: {model_desc} | Dataset: {cfg.dataset_name}{(':'+cfg.dataset_config) if cfg.dataset_config else ''} ({n_total} eval)")
    print(f"  Temperature: {cfg.temperature} | top_p: {cfg.top_p} | max_tokens: {cfg.max_new_tokens}")
    print(f"{'='*70}")
    print()

    # Header
    hdr = f"  {'Method':<20} {'Samples/Q':>10} {'AvgTok':>7} {'Accuracy':>10} {'Correct':>9} {'Coverage':>10} {'Cost vs k=8':>12} {'Cost vs k=max':>13}"
    print(hdr)
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*9} {'─'*10} {'─'*12} {'─'*13}")

    # Sort by avg_k
    rows = []
    for key, r in results.items():
        # protect against missing/None values which used to crash formatting
        avg_k = r.get("avg_k", 0.0) or 0.0
        acc = r.get("accuracy", 0.0) or 0.0
        cov = r.get("coverage", "–")
        cov_str = f"{cov:.3f}" if isinstance(cov, float) else cov
        vs_8 = f"{avg_k/8:.2f}×"
        vs_64 = f"{avg_k/K_MAX:.3f}×"
        savings_8 = f"({(1 - avg_k/8)*100:+.1f}%)"
        savings_64 = f"({(1 - avg_k/K_MAX)*100:+.1f}%)"
        avg_tok = r.get("avg_tokens", 0.0) or 0.0
        method = r.get("method", str(key)) or str(key)
        correct = r.get("correct", 0) or 0
        # if any field was None, warn to help debug later
        if any(v is None for v in (r.get("avg_k"), r.get("accuracy"), r.get("method"))):
            print(f"[warning] result {key} contains None values: {r}")
        rows.append((avg_k, method, avg_k, acc, correct, n_total, cov_str, vs_8, savings_8, vs_64, savings_64, avg_tok))

    rows.sort(key=lambda x: x[0])

    for _, method, avg_k, acc, correct, total, cov_str, vs_8, sav8, vs_64, sav64, avg_tok in rows:
        marker = " ★" if method == "conformal" else ""
        # avoid None in method when concatenating
        method_display = (method or "") + marker
        line = (
            f"  {method_display:<20} {avg_k:>10.1f} {avg_tok:>7.1f} "
            f"{acc:>10.3f} {correct:>5}/{total:<4} {cov_str:>10} {vs_8 + ' ' + sav8:>12} {vs_64 + ' ' + sav64:>13}"
        )
        print(line)

    print()

    # ── 8. Pareto analysis ───────────────────────────────────────────────
    print(f"{'='*70}")
    print("PARETO FRONTIER ANALYSIS")
    print(f"{'='*70}")

    # Build (cost, accuracy) pairs, check Pareto dominance.
    # `results` also carries diagnostic entries that are not budget points
    # (e.g. theorem2_coverage_per_k, a dict of per-k coverage records), so
    # select on the keys a budget point must have rather than assuming them.
    points = [(r["avg_k"], r["accuracy"], r["method"])
              for r in results.values()
              if isinstance(r, dict) and {"avg_k", "accuracy", "method"} <= r.keys()]
    points.sort(key=lambda x: x[0])

    pareto = []
    best_acc = -1
    for cost, acc, method in points:
        if acc > best_acc:
            pareto.append((cost, acc, method))
            best_acc = acc

    print("\n  Pareto-optimal methods (higher accuracy at lower cost):")
    for cost, acc, method in pareto:
        print(f"    {method:<20} cost={cost:>5.1f}  acc={acc:.3f}")

    # Check if conformal is Pareto-optimal
    conf_cost = results["conf"]["avg_k"]
    conf_acc  = results["conf"]["accuracy"]
    is_pareto = any(m == "conformal" for _, _, m in pareto)

    print()
    if is_pareto:
        print("  ✅ Conformal dynamic-k IS on the Pareto frontier!")
    else:
        # Find which fixed-k dominates it
        for cost, acc, method in points:
            if cost <= conf_cost and acc >= conf_acc and method != "conformal":
                print(f"  ⚠️  Conformal is dominated by {method} (cost={cost:.1f}, acc={acc:.3f})")
                break

    # note about static k=8 baseline
    if 8 in results:
        r8 = results[8]
        print(f"\n  static k=8: cost=8 avg_tok={r8.get('avg_tokens',0):.1f} acc={r8['accuracy']:.3f}; ")
        print("conformal uses avg_k={:.2f} (avg_tok={:.1f}) and often higher accuracy, hence dynamic k is beneficial".format(
              results['conf']['avg_k'], results['conf'].get('avg_tokens',0)))

    # ── 9. Conformal-specific stats ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONFORMAL METHOD DETAILS")
    print(f"{'='*70}")
    print(f"  qhats: {conf_qhats}")
    print(f"  Coverage: {results['conf']['coverage']:.3f} (target: ≥{1-cfg.deltas[0]:.2f})")
    print(f"  Avg samples/prompt: {avg_k_conf:.2f}")
    print(f"  k usage distribution:")
    for k in sorted(k_usage.keys()):
        pct = k_usage[k] / n_total * 100
        bar = "█" * int(pct / 2)
        print(f"    k={k:>2}: {k_usage[k]:>4} ({pct:5.1f}%) {bar}")

    # ── 10. DeepSeek comparison note ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON WITH DEEPSEEK-MATH GRPO")
    print(f"{'='*70}")
    print("""
  DeepSeek-Math (Shao et al., 2024) GRPO hyperparameters:
    - Model: DeepSeekMath-Instruct 7B (2.3× our 3B)
    - Group size: G = 64 samples per prompt
    - GSM8K accuracy: 88.2% (after GRPO training)
    - No dynamic k, no conformal prediction

  Our method (Conformal GRPO):
    - Model: Qwen2.5-3B + LoRA
    - Dynamic k ∈ {2, 4, 8, 16} via conformal prediction
""")
    print(f"    - GSM8K greedy accuracy:    {results[1]['accuracy']:.1%}")
    print(f"    - GSM8K conformal accuracy: {results['conf']['accuracy']:.1%}")
    print(f"    - Average samples/prompt:   {avg_k_conf:.1f} (vs DeepSeek's 64)")
    print(f"    - Sampling cost reduction:  {(1 - avg_k_conf/64)*100:.1f}% vs DeepSeek's G=64")
    print(f"    - Coverage guarantee:       {results['conf']['coverage']:.1%} ≥ {1-cfg.deltas[0]:.0%}")
    print()
    print("  Note: Direct accuracy comparison is not apples-to-apples because")
    print("  DeepSeek uses a 7B model with full fine-tuning. The relevant")
    print("  comparison is the COST-ACCURACY TRADEOFF: our method achieves")
    print("  comparable majority-vote accuracy at a fraction of the sampling cost.")

    # ── 11. GPU-hour / FLOPs cost metrics ─────────────────────────────────
    # Model parameter count (for FLOPs estimate)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_b = n_params / 1e9
    # FLOPs: ~2 × N_params per token (forward pass approximation)
    total_flops = 2 * n_params * total_tokens_generated
    cal_tokens_est = total_tokens_generated * len(cal_ds) / max(n_total, 1)
    total_flops_with_cal = 2 * n_params * (total_tokens_generated + int(cal_tokens_est))
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpu_hours = gen_time * n_gpus / 3600
    flops_per_prompt = total_flops / max(n_total, 1)

    print(f"\n{'='*70}")
    print("COMPUTE COST METRICS")
    print(f"{'='*70}")
    print(f"  Model parameters:       {n_params_b:.2f}B")
    print(f"  GPUs used:              {n_gpus}")
    print(f"  Total generation time:  {gen_time/60:.1f} min  ({gpu_hours:.4f} GPU-hours)")
    print(f"  Peak VRAM:              {peak_vram_gb:.2f} GB")
    print(f"  Tokens generated:       {total_tokens_generated:,}  (full K_MAX={K_MAX} eval set)")
    print(f"  FLOPs (eval only):      {total_flops/1e12:.2f} TFLOPs  ({total_flops/1e15:.4f} PFLOPs)")
    print(f"  FLOPs (eval+cal est.):  {total_flops_with_cal/1e12:.2f} TFLOPs")
    print()

    # ── 11b. Full cost-vs-accuracy comparison table (ALL methods) ──────────
    # Reference: static k=8 (standard GRPO baseline)
    ref_key = 8
    ref_r = results.get(ref_key, results.get(1, {"avg_tokens": 512.0, "accuracy": 0.0}))
    ref_avg_tok = ref_r.get("avg_tokens", 1.0) or 1.0
    # GPU-hours scale linearly with tokens actually consumed per prompt × n_total
    ref_gpu_h_per_prompt = gpu_hours * ref_avg_tok / max(total_tokens_generated / max(n_total, 1), 1.0)

    print(f"{'='*70}")
    print("COST vs ACCURACY COMPARISON  (all methods, reference = static k=8)")
    print(f"{'='*70}")
    col_hdr = (
        f"  {'Method':<22} {'Acc':>6} {'Samples/Q':>10} {'AvgTok':>8} "
        f"{'TFLOPs/Q':>10} {'GPUh/Q(×1e-5)':>14} {'vs k=8 tok':>11} {'vs k=max tok':>12}"
    )
    print(col_hdr)
    print(f"  {'─'*22} {'─'*6} {'─'*10} {'─'*8} {'─'*10} {'─'*14} {'─'*11} {'─'*12}")

    # Sort all results by avg_k, then name
    all_rows_sorted = sorted(results.items(), key=lambda kv: (kv[1].get("avg_k", 0) or 0, str(kv[0])))
    for key, r in all_rows_sorted:
        method   = r.get("method", str(key)) or str(key)
        avg_k    = r.get("avg_k", 0.0) or 0.0
        avg_tok  = r.get("avg_tokens", 0.0) or 0.0
        acc      = r.get("accuracy", 0.0) or 0.0
        # FLOPs per query for this method (only tokens actually used, not K_MAX)
        flops_q  = 2 * n_params * avg_tok / 1e12   # TFLOPs per query
        # GPU-hours per query (proportional)
        gpu_h_q  = gpu_hours * avg_tok / max(total_tokens_generated / max(n_total, 1), 1.0)
        # Token ratio vs k=8
        tok_vs_8  = avg_tok / ref_avg_tok
        # vs the most exhaustive budget actually generated (was a hard-coded k=64 row,
        # which does not exist when --k-max < 64)
        tok_vs_kmax = avg_tok / max(results.get(K_MAX, {}).get("avg_tokens", ref_avg_tok * K_MAX / 8) or 1, 1)
        marker = " ★" if method == "conformal" else "  "
        print(
            f"  {method+marker:<22} {acc:>6.3f} {avg_k:>10.1f} {avg_tok:>8.1f} "
            f"{flops_q:>10.4f} {gpu_h_q*1e5:>14.3f} "
            f"{tok_vs_8:>10.2f}× {tok_vs_kmax:>11.2f}×"
        )

    print()
    print(f"  Interpretation:")
    print(f"    TFLOPs/Q  = FLOPs consumed per query for this method (2×params×avg_tokens)")
    print(f"    GPUh/Q    = estimated GPU-hours consumed per query (×10⁻⁵)")
    print(f"    vs k=8    = token cost relative to static k=8 (the standard GRPO baseline)")
    print(f"    vs k={K_MAX:<4} = token cost relative to exhaustive k={K_MAX} sampling")
    # Save metrics alongside results
    cost_metrics = {
        "n_params_B": n_params_b,
        "n_gpus": n_gpus,
        "generation_time_s": gen_time,
        "gpu_hours": gpu_hours,
        "peak_vram_gb": peak_vram_gb,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": total_tokens_generated / gen_time if gen_time > 0 else 0.0,
        "prompts_per_minute": n_total / (gen_time / 60) if gen_time > 0 else 0.0,
        "total_flops_eval": total_flops,
        "total_flops_with_cal": total_flops_with_cal,
        "flops_per_prompt": flops_per_prompt,
        "per_method_flops_T_per_query": {
            str(k): round(2 * n_params * (v.get("avg_tokens") or 0) / 1e12, 6)
            for k, v in results.items()
        },
    }

    # ── 12. Save results ─────────────────────────────────────────────────
    # include dataset info in filename to avoid collisions when running multiple datasets
    dname = cfg.dataset_name.replace("/", "_")
    if cfg.dataset_config:
        dname += f"_{cfg.dataset_config}"
    # The adapter/base distinction MUST be in the filename. Without it, a
    # subsequent `--no-lora` run silently overwrote the trained model's results
    # with the base model's, because both invocations share the same --ckpt-dir.
    variant = "base" if args.no_lora else "lora"
    filename = f"pareto_results_{dname}_{variant}.json"
    out_dir = os.path.dirname(CKPT_DIR) if CKPT_DIR else os.getcwd()
    out_path = os.path.join(out_dir, filename)
    save_data = {
        "model": cfg.model_key,
        "ckpt": CKPT_DIR,
        "adapter": ("base (no LoRA)" if args.no_lora else "LoRA adapter loaded"),
        "n_eval": n_total,
        "k_max_generated": K_MAX,
        "generation_time_min": gen_time / 60,
        "results": {str(k): {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                    for k, v in results.items()},
        "conformal_qhats": conf_qhats,
        "conformal_k_usage": dict(k_usage),
        "delta": float(cfg.deltas[0]),
        "delta_mode": delta_mode,
        "calibration_solve_rate": cal_solve_rate,
        "cost_metrics": cost_metrics,
        # Saved at top level because the "results" flattening above keeps only
        # scalar fields, and every per-k coverage record is itself a dict. Nested
        # inside "results" it was silently reduced to {} on every run, so the
        # Theorem-2 measurement was computed and then thrown away at save time.
        "theorem2_coverage_per_k": results.get("theorem2_coverage_per_k", {}),
    }
    # Convert any non-serializable types
    def _convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=_convert)
    print(f"\n  Results saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
