import os, time, json, math, random
from tqdm import tqdm
# disable all tqdm progress bars globally (e.g. weight loading)
import tqdm as _tqdm
_tqdm.tqdm = lambda *args, **kwargs: (args[0] if args else iter([]))

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from logger import make_loggers
from config import TrainConfig
from utils import set_seed, pick_cuda_devices, batch_to_examples, total_free_gpu_mib
from data import load_gsm8k_splits, format_prompt, extract_example
from generation import generate_n, count_new_tokens
from verifier import canonicalize_answer, verify_answer
from conformal import (calibrate_qhats, answer_freqs,
                       conformal_set_from_freqs, set_conformal_seed)
from grpo_loss import completion_logprob, grpo_objective, per_token_kl
from model_registry import get_model_path, get_model_dtype_str


# -------------------------
# Utils: dirs, checkpoints
# -------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def save_checkpoint(pi, opt, step: int, out_dir: str, tokenizer=None, qhats=None):
    """
    Save:
      - LoRA adapters (pi.save_pretrained)
      - optimizer state_dict
      - meta.json with step + optional qhats
      - tokenizer (optional)
    """
    ensure_dir(out_dir)
    ckpt_dir = os.path.join(out_dir, f"ckpt_step_{step:06d}")
    ensure_dir(ckpt_dir)

    # Save LoRA adapters
    pi.save_pretrained(ckpt_dir)

    # Save optimizer state
    torch.save(opt.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    # Save tokenizer if requested
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    meta = {"step": int(step), "ts": time.time()}
    if qhats is not None:
        meta["qhats"] = {str(k): float(v) for k, v in qhats.items()}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def summarize_list(vals):
    if not vals:
        return {"n": 0}
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
    }


# -------------------------
# Model loading
# -------------------------
def _resolve_dtype(dtype_str: str, prefer_bf16: bool) -> torch.dtype:
    if dtype_str.lower() in ("bf16", "bfloat16"):
        if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_str.lower() in ("fp16", "float16"):
        return torch.float16
    if dtype_str.lower() in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16 if (prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

def _disk_model_size_mib(model_path: str) -> float:
    """Return total checkpoint size on disk in MiB as a proxy for model weight size."""
    import glob
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for fpath in glob.glob(os.path.join(model_path, pattern)):
            total += os.path.getsize(fpath)
    return total / (1024 * 1024)


def _needs_shared_ref(model_path: str, dtype) -> bool:
    """Return True when loading two separate model copies would exhaust GPU memory.

    We estimate the on-disk model size (≈weight size in memory) and compare it
    to the total free VRAM across all visible GPUs.  We need 2× the model size
    (policy + reference) plus headroom for activations, optimizer state, etc.
    If that exceeds 80 % of total free VRAM we fall back to shared-ref mode.

    On single-GPU nodes we always use shared-ref mode regardless of memory, to
    avoid the brittle CPU-load → .to(device) pattern that can fail when the GPU
    has a lingering CUDA context from a previous job.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() <= 1:
        return True
    size_mib = _disk_model_size_mib(model_path)
    if size_mib == 0:          # can't estimate — assume it fits
        return False
    # bf16 / fp16 weights are roughly 1× the raw file size; fp32 ≈ 2×
    bytes_per_elem = 4 if dtype == torch.float32 else 2
    file_factor = bytes_per_elem / 2  # safetensors store bf16 as 2 bytes/elem
    mem_per_copy_mib = size_mib * file_factor
    free_list = total_free_gpu_mib()
    total_free_mib = sum(free_list) if free_list else 0
    needed_mib = mem_per_copy_mib * 2  # pi + ref
    return needed_mib > total_free_mib * 0.80


def build_models(cfg, pi_device, ref_device):
    model_path = get_model_path(cfg.model_key)
    dtype_str = get_model_dtype_str(cfg.model_key)
    dtype = _resolve_dtype(dtype_str, cfg.use_bf16)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
        tok.pad_token = tok.convert_ids_to_tokens(tok.pad_token_id)
    tok.chat_template = None
    if hasattr(tok, 'apply_chat_template'):
        tok.apply_chat_template = lambda x, **kwargs: x

    # LoRA config (shared between both loading paths)
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    full_ft = bool(getattr(cfg, 'full_finetune', False))
    if full_ft:
        print('  FULL-PARAMETER fine-tuning: LoRA disabled, all weights trainable.')
        print('  NOTE: a 7-8B model in bf16 with AdamW needs ~14x params in optimiser\n'
              '        state; use >=2 GPUs, --optim-8bit, or gradient checkpointing.')
    shared_ref = _needs_shared_ref(model_path, dtype)
    if full_ft and shared_ref:
        # Adapter toggling needs LoRA, so full fine-tuning always holds a separate frozen
        # reference. On a single GPU both copies can share the device when they fit:
        # ~2x weights (policy + reference) + 1x gradients + 2x AdamW state, in bf16, ≈ 8x the
        # bf16 checkpoint size once activations are allowed for.
        size_mib = _disk_model_size_mib(model_path)
        free_list = total_free_gpu_mib()
        single_gpu = torch.cuda.is_available() and torch.cuda.device_count() == 1
        if single_gpu and size_mib and free_list and size_mib * 8 <= free_list[0] * 0.9:
            print(f"  Full fine-tuning on one GPU: separate frozen reference on the same device "
                  f"(model {size_mib} MiB, free {free_list[0]} MiB).")
            shared_ref = False
        else:
            raise RuntimeError(
                'full_finetune=True requires a separate frozen reference model, which does not\n'
                f'fit here (model {size_mib} MiB on disk; need ~8x that free on one GPU).\n'
                'Use >=2 visible GPUs, a smaller model, or LoRA.')

    if shared_ref:
        # Load one copy with device_map="auto", use adapter-off for ref.
        # (torch is imported at module level; a local import here would make `torch`
        # function-local and break every earlier use in this function.)
        n_gpus = torch.cuda.device_count()
        free_list = total_free_gpu_mib()
        model_size_mib = _disk_model_size_mib(model_path)
        # Cap each GPU at 3× model size (weights + LoRA + grad buffers) so at
        # least half the VRAM stays free for KV-cache and activations during
        # calibration/generation.  Never exceed 70% of free memory.
        per_gpu_cap = max(model_size_mib * 3, 8192)  # at least 8 GiB cap
        max_memory = {
            i: f"{int(min(per_gpu_cap, free_list[i] * 0.70))}MiB"
            for i in range(min(n_gpus, len(free_list)))
        }
        print(f"  Shared-ref mode (device_map=auto). model_disk={model_size_mib}MiB")
        print(f"  max_memory per GPU: {max_memory}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto",
            max_memory=max_memory, trust_remote_code=False, local_files_only=True
        )
        if hasattr(base.generation_config, 'chat_format'):
            base.generation_config.chat_format = None
        base.gradient_checkpointing_enable()
        if full_ft:
            pi = base
            for _p in pi.parameters():
                _p.requires_grad = True
        else:
            pi = get_peft_model(base, lora)
        pi.train()
        first_dev = next(pi.parameters()).device
        print(f"  pi  on {first_dev} (device_map=auto across all GPUs)")
        return tok, pi, None, True   # ref=None signals shared-ref mode

    # ── Standard path: two separate model copies on different GPUs ─────────
    print(f"Loading base model to {pi_device}, ref model to {ref_device}...")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=None, trust_remote_code=False, local_files_only=True
    )
    ref  = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=None, trust_remote_code=False, local_files_only=True
    )
    if hasattr(base.generation_config, 'chat_format'):
        base.generation_config.chat_format = None
    if hasattr(ref.generation_config, 'chat_format'):
        ref.generation_config.chat_format = None
    base.gradient_checkpointing_enable()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    if full_ft:
        pi = base
        for _p in pi.parameters():
            _p.requires_grad = True
    else:
        pi = get_peft_model(base, lora)
    pi.train()

    try:
        pi.to(pi_device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            raise RuntimeError(
                f"Unable to move policy model to {pi_device} (OOM). "
                "Your selected GPU may not have enough free memory. "
                "Consider using a smaller model or setting CUDA_VISIBLE_DEVICES."
            ) from e
        raise
    ref.to(ref_device)
    print(f"  pi  on {next(pi.parameters()).device}")
    print(f"  ref on {next(ref.parameters()).device}")
    return tok, pi, ref, False



# -------------------------
# Phase-level wall-clock accounting
# -------------------------
class PhaseTimer:
    """Accumulate wall-clock per training phase.

    Needed to report an end-to-end breakdown (training generation vs.
    forward/backward vs. calibration) rather than a single total, so the
    efficiency claim can be audited phase by phase.
    """

    def __init__(self):
        self.totals = {}

    def __call__(self, name):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            t0 = time.time()
            try:
                yield
            finally:
                self.totals[name] = self.totals.get(name, 0.0) + (time.time() - t0)
        return _ctx()

    def as_dict(self, prefix="phase_h/"):
        return {f"{prefix}{k}": v / 3600.0 for k, v in self.totals.items()}


# -------------------------
# Conformal calibration
# -------------------------
def calibrate_conformal(cfg, tok, ref, cal_ds, delta: float, device=None, shared_ref: bool = False, auto_delta: bool = False):
    """Calibrate conformal qhats. `device` = the device where `ref` lives.

    If ``shared_ref=True``, ``ref`` is actually the policy model (PeftModel).
    Adapters are temporarily disabled so the base weights act as the reference.

    For CODE datasets (HumanEval, MBPP): APS scoring cannot work because the
    gold reference code and model-generated code are never identical strings
    even when functionally equivalent.  Instead we use an execution-based
    nonconformity score:
        score = (index of first passing rollout) / k    if any pass
        score = 1.0                                     if no rollout passes
    This is the APS analog for binary-correctness tasks: it measures how many
    rollouts must be drawn before finding a correct solution (normalized to [0,1]).
    We therefore keep cal_truth as the raw JSON blob (with test metadata) and
    keep the generated completions as raw strings so the execution harness works.
    """
    if device is None:
        device = next(ref.parameters()).device
    k_max = max(cfg.k_values)

    # Determine whether this dataset uses execution-based scoring (code tasks).
    _ds_lower = cfg.dataset_name.lower()
    _is_code  = ("humaneval" in _ds_lower) or ("mbpp" in _ds_lower)

    # convert examples to generic (text, answer)
    cal_pairs = [extract_example(ex, cfg.dataset_name) for ex in cal_ds]
    cal_prompts = [format_prompt(text, cfg.dataset_name) for text, _ in cal_pairs]

    # For code datasets we must keep the raw JSON gold (contains test harness).
    # For all others, canonicalize so that the APS string-matching works.
    from verifier import canonicalize_answer
    if _is_code:
        cal_truth = [ans for _, ans in cal_pairs]          # raw JSON
    else:
        cal_truth = [canonicalize_answer(ans, cfg.dataset_name) for _, ans in cal_pairs]

    all_groups = []
    bs = cfg.batch_size
    for i in tqdm(range(0, len(cal_prompts), bs), desc=f"Calibrating (delta={delta})"):
        batch_prompts = cal_prompts[i:i+bs]
        if shared_ref:
            if hasattr(ref, 'disable_adapter_layers'):
                ref.disable_adapter_layers()
        try:
            groups = generate_n(
                model=ref,
                tokenizer=tok,
                prompts=batch_prompts,
                n=k_max,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                device=device,
            )
        finally:
            if shared_ref:
                if hasattr(ref, 'enable_adapter_layers'):
                    ref.enable_adapter_layers()

        groups_ans = []
        for bi, outs in enumerate(groups):
            prompt = batch_prompts[bi]
            # Strip the prompt echo from each generated string
            completions = [o[len(prompt):] if o.startswith(prompt) else o for o in outs]
            if _is_code:
                # Keep raw completions — the execution score_fn needs them
                groups_ans.append(completions)
            else:
                groups_ans.append([canonicalize_answer(c, cfg.dataset_name) for c in completions])
        all_groups.extend(groups_ans)

    # ---- Choose / build the appropriate nonconformity score function --------

    score_fn = None

    _code_score_kind = str(getattr(cfg, "code_score", "first_success")).lower()

    def _score_from_passes(passes: list) -> float:
        """Map a per-completion pass/fail list to a nonconformity score."""
        k = max(len(passes), 1)
        if _code_score_kind == "pass_rate":
            return 1.0 - sum(passes) / k
        for j, ok in enumerate(passes, start=1):   # paper Eq. (5): j*/k
            if ok:
                return j / k
        return 1.0


    # Execution-based nonconformity score for code. TWO variants exist:
    #
    #   "first_success" (DEFAULT, and what the paper's Eq. (5) defines):
    #       s = j*/k, where j* is the 1-based position of the first passing
    #       completion, or 1.0 if none passes. Order-dependent by design -- the
    #       ordering reflects the sequential generation process, and the
    #       geometric distribution of j* is what makes the score continuous in
    #       the local pass rate (needed for EXACT rather than conservative
    #       coverage).
    #
    #   "pass_rate":
    #       s = 1 - n_pass/k. Permutation-invariant and lower variance, but
    #       supported on the discrete grid {0, 1/k, ..., 1}, so ties occur with
    #       positive probability and coverage becomes conservative rather than
    #       exact.
    #
    #   Select with cfg.code_score / --code-score. Both are reported so the
    #   choice can be justified empirically instead of asserted.
    if "humaneval" in _ds_lower:
        import json as _json
        from verifier import _run_humaneval_code
        def _humaneval_score_fn(gold_json: str, gens: list) -> float:
            try:
                d = _json.loads(gold_json)
                prompt_he = d.get("prompt", "")
                test_fn   = d.get("test_fn", "")
                entry_pt  = d.get("entry_point", "")
            except Exception:
                return 1.0
            passes = [bool(_run_humaneval_code(prompt_he, gen, test_fn, entry_pt))
                      for gen in gens]
            return _score_from_passes(passes)
        score_fn = _humaneval_score_fn

    # MBPP: deterministic execution-based nonconformity score (same principle).
    elif "mbpp" in _ds_lower:
        import json as _json
        from verifier import _run_code_with_tests, _extract_code_block
        def _mbpp_score_fn(gold_json: str, gens: list) -> float:
            try:
                d     = _json.loads(gold_json)
                tests = d.get("tests", [])
                setup = d.get("setup", "")
            except Exception:
                return 1.0
            passes = [bool(_run_code_with_tests(_extract_code_block(gen), tests, setup))
                      for gen in gens]
            return _score_from_passes(passes)
        score_fn = _mbpp_score_fn

    # AQuaMUSE: abstractive QA — 1 - max token-precision because
    # correct concise answers have very low recall against the long gold passage.
    elif "aquamuse" in _ds_lower:
        from verifier import _token_precision
        def _aqua_score_fn(gold: str, gens: list) -> float:
            if not gold:
                return 1.0
            best = max((_token_precision(g, gold) for g in gens if g.strip()), default=0.0)
            return 1.0 - best
        score_fn = _aqua_score_fn

    # ---- Auto-select δ if requested (data-driven, no manual tuning) -------
    # Compute scores at k_max to measure pass@k_max on the calibration set,
    # then set δ = (1 - pass@k_max) + safety_margin.  This makes the coverage
    # target honest (equal to what the model can actually achieve) without any
    # per-dataset manual tuning.
    # Split-conformal exactness requires the miscoverage level δ to be fixed
    # INDEPENDENTLY of the scores used to compute q_hat.  Selecting δ_auto from
    # the same calibration scores uses the data twice and voids the guarantee.
    # With split_delta_calibration=True we partition D_cal: half A selects δ,
    # half B computes q_hat, restoring exactness at the cost of an effective
    # calibration size of n_cal/2 (finite-sample slack 2/(n_cal+2)).
    split_delta = bool(getattr(cfg, "split_delta_calibration", False))
    n_all = len(cal_truth)
    if split_delta and auto_delta and n_all >= 4:
        n_a = n_all // 2
        delta_truth, delta_groups = cal_truth[:n_a], all_groups[:n_a]
        qhat_truth, qhat_groups = cal_truth[n_a:], all_groups[n_a:]
        print(f"  [split-δ] D_cal partitioned: {n_a} examples select δ, "
              f"{n_all - n_a} calibrate q̂ (disjoint)")
    else:
        delta_truth, delta_groups = cal_truth, all_groups
        qhat_truth, qhat_groups = cal_truth, all_groups

    if auto_delta:
        import numpy as _np
        from conformal import select_delta_auto, score_true_answer as _sts
        _k_max_scores = []
        for _truth, _group in zip(delta_truth, delta_groups):
            if score_fn is not None:
                _k_max_scores.append(score_fn(_truth, _group[:k_max]))
            else:
                _freqs = answer_freqs(_group[:k_max])
                _k_max_scores.append(_sts(_truth, _freqs, k_max))
        delta, _frac_solvable = select_delta_auto(_np.asarray(_k_max_scores))
        print(f"\n  [Auto-δ] pass@{k_max}={_frac_solvable:.1%} on {len(_k_max_scores)} cal examples "
              f"→ δ={delta:.3f}  (target coverage ≥{1-delta:.1%})\n")

    qhats = calibrate_qhats(qhat_truth, qhat_groups, cfg.k_values, delta,
                             score_fn=score_fn)
    return qhats, delta

def choose_k_and_sets(extracted_answers, qhats, k_values):
    for k in k_values:
        freqs = answer_freqs(extracted_answers[:k])
        Ck = conformal_set_from_freqs(freqs, k, qhats[k])
        if len(Ck) == 1:
            return k, Ck
    k = max(k_values)
    freqs = answer_freqs(extracted_answers[:k])
    return k, conformal_set_from_freqs(freqs, k, qhats[k])


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(cfg, tok, pi, eval_ds, qhats, device=None):
    """Evaluate with greedy + conformal. `device` = the device where `pi` lives."""
    if device is None:
        device = next(pi.parameters()).device
    pi.eval()
    
    # --- Greedy baseline (single generation per prompt) ---
    greedy_correct = 0
    total = 0
    total_new_tokens = 0
    
    # --- Conformal evaluation (sampling, majority vote within conformal set) ---
    conf_correct = 0
    conf_covered = 0   # did conformal set contain the true answer?
    total_k = 0
    k_usage_counts = {}  # how often each k is used

    bs = cfg.batch_size
    k_values = tuple(sorted(cfg.k_values))
    k_max = max(k_values)

    for i in tqdm(range(0, len(eval_ds), bs), desc="Eval"):
        batch = eval_ds[i:i+bs]
        examples = batch_to_examples(batch)
        batch_pairs = [extract_example(ex, cfg.dataset_name) for ex in examples]
        if batch_pairs:
            questions, gold_list = zip(*batch_pairs)
        else:
            questions, gold_list = [], []
        prompts = [format_prompt(q, cfg.dataset_name) for q in questions]

        # --- Greedy pass (1 sample, no sampling) ---
        greedy_groups = generate_n(
            pi, tok, prompts, 1,
            cfg.max_new_tokens,
            temperature=None,
            top_p=1.0,
            device=device,
            repetition_penalty=1.0,
        )
        
        # --- Sampling pass (k_max samples with temperature) ---
        sampled_groups = generate_n(
            pi, tok, prompts, k_max,
            cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            device=device,
        )

        from verifier import canonicalize_answer, verify_answer
        _ds_lower_eval = cfg.dataset_name.lower()
        _is_code_eval  = ("humaneval" in _ds_lower_eval) or ("mbpp" in _ds_lower_eval)
        for p, greedy_outs, sampled_outs, g in zip(prompts, greedy_groups, sampled_groups, gold_list):
            # canonicalized true answer (math or string depending on dataset)
            gold_ans = canonicalize_answer(g, cfg.dataset_name)
            
            # -- Greedy accuracy --
            greedy_comp = greedy_outs[0][len(p):] if greedy_outs[0].startswith(p) else greedy_outs[0]
            if verify_answer(greedy_comp, g, cfg.dataset_name):
                greedy_correct += 1

            # -- Conformal accuracy (with sampling) --
            completions = [o[len(p):] if o.startswith(p) else o for o in sampled_outs]

            if _is_code_eval:
                # Execution-based conformal k-selection, matching training logic.
                # qhats use score = 1 - pass_rate, so we apply the same comparison
                # here to get meaningful conf_acc / avg_k in training eval logs.
                pass_results_eval = [verify_answer(c, g, cfg.dataset_name) for c in completions]
                k_used = max(k_values)
                for _k in sorted(k_values):
                    _qhat_k = qhats.get(_k, 1.0)
                    if _qhat_k >= 1.0:
                        # Same fix as training loop: skip degenerate qhat=1.0 levels.
                        continue
                    _score = 1.0 - sum(pass_results_eval[:_k]) / _k
                    if _score <= _qhat_k:
                        k_used = _k
                        break
                k_usage_counts[k_used] = k_usage_counts.get(k_used, 0) + 1
                if any(pass_results_eval[:k_used]):
                    conf_correct += 1
                    conf_covered += 1
            else:
                answers = [canonicalize_answer(c, cfg.dataset_name) for c in completions]
                k_used, Ck = choose_k_and_sets(answers, qhats, k_values)
                k_usage_counts[k_used] = k_usage_counts.get(k_used, 0) + 1

                # Majority vote among k_used samples
                freqs = answer_freqs(answers[:k_used])
                pred_ans = freqs.most_common(1)[0][0] if len(freqs) else ""
                if verify_answer(pred_ans, g, cfg.dataset_name):
                    conf_correct += 1
                
                # Conformal coverage: is the true answer in the conformal set?
                if gold_ans in Ck:
                    conf_covered += 1

            total += 1
            total_k += k_used
            total_new_tokens += count_new_tokens(tok, p, sampled_outs[0])

    pi.train()
    return {
        "greedy_acc": greedy_correct / max(total, 1),
        "conf_acc": conf_correct / max(total, 1),
        "conf_coverage": conf_covered / max(total, 1),
        "avg_k": total_k / max(total, 1),
        "k_usage": k_usage_counts,
        "avg_new_tokens": total_new_tokens / max(total, 1),
    }



# -------------------------
# Main training loop
# -------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    set_conformal_seed(cfg.seed)
    phase = PhaseTimer()

    # GPU placement: choose least-busy devices automatically
    n_gpus = torch.cuda.device_count()
    print(f"🔧 CUDA devices available: {n_gpus}")
    for i in range(n_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    if n_gpus >= 2:
        chosen = pick_cuda_devices(2)
        pi_device = torch.device(f"cuda:{chosen[0]}")
        ref_device = torch.device(f"cuda:{chosen[1]}")
    elif n_gpus == 1:
        pi_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:0")
        print("⚠️  Only 1 GPU detected, both models on cuda:0")
    else:
        raise RuntimeError("No CUDA GPUs available!")

    print(f"Selected devices: pi={pi_device}, ref={ref_device}")

    # Mixed precision via torch.cuda.amp
    amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler('cuda', enabled=use_scaler)  # bf16 doesn't need scaler

    # Run naming + log dirs
    _delta_tag = "auto" if getattr(cfg, "auto_delta", False) else cfg.deltas[0]
    run_name = f"{cfg.model_key}_delta{_delta_tag}_seed{cfg.seed}_{_now_str()}"
    run_dir = os.path.join(cfg.log_dir, run_name)
    ensure_dir(run_dir)

    # loggers
    jsonl, wb, tb = make_loggers(
        cfg,
        log_dir=run_dir,
        use_wandb=getattr(cfg, "use_wandb", True),
        wandb_project=getattr(cfg, "wandb_project", "conformal-grpo-rlvr"),
        run_name=run_name,
        use_tensorboard=getattr(cfg, "use_tensorboard", True),
    )

    state_path = os.path.join(run_dir, "state.json")

    # dataset
    train_ds, cal_ds, eval_ds = load_gsm8k_splits(
        cfg.dataset_name, cfg.dataset_config, cfg.n_train, cfg.n_cal, cfg.n_eval, cfg.seed
    )

    # models — placed on separate GPUs (or shared in large-model mode)
    tok, pi, ref, shared_ref = build_models(cfg, pi_device, ref_device)

    # In shared-ref mode the reference IS the policy model with adapters disabled;
    # derive ref_device from where the model actually lives.
    if shared_ref:
        ref_device = next(pi.parameters()).device

    opt = AdamW(pi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)

    # calibrate (ref model on ref_device); in shared-ref mode pass pi + flag
    delta = cfg.deltas[0]
    _cal_model  = pi if shared_ref else ref
    _cal_device = ref_device
    with phase('calibration_bootstrap'):
        qhats, delta = calibrate_conformal(cfg, tok, _cal_model, cal_ds, delta,
                                           device=_cal_device, shared_ref=shared_ref,
                                           auto_delta=getattr(cfg, "auto_delta", False))

    # Write calibration record
    jsonl.log(0, {"event": "calibration_done", "delta": delta, "qhats": qhats})
    qhat_scalars = {f"qhats/k={k}": float(v) for k, v in qhats.items()}
    if wb: wb.log(0, qhat_scalars)
    if tb: tb.log(0, qhat_scalars)
    atomic_write_json(state_path, {
        "event": "calibration_done",
        "run_name": run_name,
        "qhats": {str(k): float(v) for k, v in qhats.items()},
    })

    # Training loop params
    k_values = tuple(sorted(cfg.k_values))
    bs = cfg.batch_size
    k_max = max(k_values)
    
    # Warmup schedule for learning rate (longer warmup for stability)
    warmup_steps = min(100, cfg.steps // 5)
    base_lr = cfg.lr

    # prompts cache for speed; convert dataset examples to (text,answer) pairs
    train_pairs = [extract_example(ex, cfg.dataset_name) for ex in train_ds]
    train_prompts = [format_prompt(text, cfg.dataset_name) for text, _ in train_pairs]
    # answers are taken from the extracted pairs so we don't rely on a particular
    # column name (some datasets like aquamuse_abstractive don't have 'answer').
    train_answers = [ans for _, ans in train_pairs]

    # cadence config (safe defaults if not in cfg)
    log_every = int(getattr(cfg, "log_every", 1))
    eval_every = int(getattr(cfg, "eval_every", 50))
    sample_every = int(getattr(cfg, "sample_every", 50))
    ckpt_every = int(getattr(cfg, "ckpt_every", 100))

    step = 0
    pbar = tqdm(total=cfg.steps, desc="Train")

    # rolling stats (nice for logs)
    ema_loss = None
    ema_alpha = 0.05

    while step < cfg.steps:
        t0 = time.time()

        # minibatch indices
        idx = torch.randint(0, len(train_prompts), (bs,), device="cpu").tolist()
        prompts = [train_prompts[i] for i in idx]
        gold = [train_answers[i] for i in idx]

        # -------------------------
        # Stage 1: k=2 for all
        # -------------------------
        with phase('train_generation'):
            groups2 = generate_n(
              pi, tok, prompts, 2,
              cfg.max_new_tokens, cfg.temperature, cfg.top_p, pi_device
          )
        # Extract completions (remove echoed prompt) for answer extraction
        from verifier import canonicalize_answer
        answers2 = []
        for gi, outs in enumerate(groups2):
            completions = [o[len(prompts[gi]):] if o.startswith(prompts[gi]) else o for o in outs]
            answers2.append([canonicalize_answer(c, cfg.dataset_name) for c in completions])

        hard_mask = []
        for a2 in answers2:
            freqs = answer_freqs(a2)
            C2 = conformal_set_from_freqs(freqs, 2, qhats[2])
            hard_mask.append(int(len(C2) != 1))

        hard_rate = float(sum(hard_mask) / max(len(hard_mask), 1))

        # -------------------------
        # Expand for hard prompts
        # -------------------------
        hard_indices = [i for i, h in enumerate(hard_mask) if h == 1]
        extra_groups = {i: [] for i in hard_indices}

        if hard_indices:
            hard_prompts = [prompts[i] for i in hard_indices]
            with phase('train_generation'):
              groups_extra = generate_n(
                  pi, tok, hard_prompts, k_max - 2,
                  cfg.max_new_tokens, cfg.temperature, cfg.top_p, pi_device
              )
            for bi, outs in zip(hard_indices, groups_extra):
                extra_groups[bi] = outs

        # -------------------------
        # Build variable groups + RLVR rewards
        # -------------------------
        all_full_texts = []
        all_prompts_rep = []
        rewards_list = []
        group_ids_list = []
        token_cost = 0
        k_used_list = []

        _is_code_train = any(kw in cfg.dataset_name.lower() for kw in ("humaneval", "mbpp"))
        for i in range(bs):
            outs = groups2[i] + extra_groups.get(i, [])
            # Extract completions (remove echoed prompt) for answer extraction
            from verifier import canonicalize_answer, verify_answer as _vfy
            completions_for_k = [o[len(prompts[i]):] if o.startswith(prompts[i]) else o for o in outs]

            if _is_code_train:
                # ── Execution-based conformal k-selection for code ─────────────
                # APS frequency scoring fails for code because every generated
                # string is unique even when functionally equivalent.  qhats
                # are calibrated using score = 1 - pass_rate (execution), so
                # we must evaluate in that same score space at training time.
                # Pre-computing pass_flags also avoids redundant code execution
                # in the per-completion reward loop below.
                pass_flags = [bool(_vfy(c, gold[i], cfg.dataset_name))
                              for c in completions_for_k]
                k_used = max(k_values)
                for _k in sorted(k_values):
                    _qhat_k = qhats.get(_k, 1.0)
                    if _qhat_k >= 1.0:
                        # qhat=1.0 means calibration set mostly fails at this k
                        # (quantile hit the ceiling) — not a reliable threshold,
                        # skip to the next larger k.
                        continue
                    _score_k = 1.0 - sum(pass_flags[:_k]) / _k
                    if _score_k <= _qhat_k:
                        k_used = _k
                        break
            else:
                pass_flags = None
                ans = [canonicalize_answer(c, cfg.dataset_name) for c in completions_for_k]
                k_used, Ck = choose_k_and_sets(ans, qhats, k_values)

            k_used_list.append(int(k_used))
            outs = outs[:k_used]
            completions_for_k = completions_for_k[:k_used]
            if pass_flags is not None:
                pass_flags = pass_flags[:k_used]

            for j, o in enumerate(outs):
                # Extract completion-only text (remove the echoed prompt)
                completion = completions_for_k[j]

                # Check if output is degenerate (repeated characters, gibberish).
                # Classification datasets produce legitimately short answers
                # (e.g. "A" for commonsense_qa, "World" for ag_news) so we
                # skip this check for them to avoid wrongly penalising correct answers.
                _is_classification = any(k in cfg.dataset_name.lower()
                                         for k in ["ag_news", "commonsense_qa", "sst2", "piqa"])
                is_degenerate = False
                if not _is_classification:
                    if len(completion.strip()) < 5:
                        is_degenerate = True
                    elif len(completion) > 20:
                        # check character diversity (catches "aaaaaaa..." style outputs)
                        unique_chars = len(set(completion.replace(' ', '')))
                        if unique_chars < 8:
                            is_degenerate = True

                if is_degenerate:
                    r = -1.0  # Strong penalty for degenerate output
                else:
                    if "aquamuse" in cfg.dataset_name.lower():
                        # ---- Continuous precision reward for AQuaMUSE ----
                        # Binary verify_answer at 0.4 gives R≈0.03 with near-
                        # zero within-group variance → GRPO advantage≈0 → no
                        # learning signal.  Using raw token-precision as reward
                        # (after stripping Qwen3 <think> blocks) gives real
                        # gradient: R≈0.15-0.35, full variance within group.
                        # Shift by 0.10 (random stopword overlap baseline) and
                        # scale so prec=0.10 → r=0.0 and prec=0.70 → r=1.0.
                        from verifier import _token_precision, _strip_thinking
                        clean = _strip_thinking(completion)
                        if not clean.strip():
                            clean = completion  # fallback: all text was in <think>
                        prec = _token_precision(clean, gold[i])
                        r = max(0.0, min(1.0, (prec - 0.10) / 0.60))
                    elif _is_code_train and pass_flags is not None:
                        # Reuse pre-computed execution result (no double execution)
                        r = float(pass_flags[j])
                    else:
                        # Verify if answer is correct (use completion, not full text)
                        from verifier import verify_answer
                        r = float(verify_answer(completion, gold[i], cfg.dataset_name))

                        # Small format-shaping reward: encourage outputs with ####
                        # (mostly relevant for math problems)
                        if r == 0.0 and '####' in completion:
                            r = 0.1  # Partial credit for correct format
                
                rewards_list.append(r)
                all_full_texts.append(o)
                all_prompts_rep.append(prompts[i])
                group_ids_list.append(i)
                token_cost += count_new_tokens(tok, prompts[i], o)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=pi_device)
        group_ids = torch.tensor(group_ids_list, dtype=torch.long, device=pi_device)

        avg_k_used = float(sum(k_used_list) / max(len(k_used_list), 1))
        avg_reward = float(rewards.mean().item()) if rewards.numel() else 0.0

        # -------------------------
        # GRPO objective
        # -------------------------
        max_len = getattr(cfg, "max_seq_len", None)  # e.g., 1024; None = no cap
        mb = getattr(cfg, "logprob_microbatch", 2)   # 1 is safest, 2-4 faster if fits

        with autocast('cuda', dtype=amp_dtype):
            logp_pi, per_tok_pi = completion_logprob(
                pi, tok, all_prompts_rep, all_full_texts, pi_device,
                microbatch=mb,
                max_length=max_len,
                compute_grads=True,
            )
        with torch.no_grad():
            if shared_ref:
                if hasattr(pi, 'disable_adapter_layers'):
                    pi.disable_adapter_layers()
            try:
                logp_ref, per_tok_ref = completion_logprob(
                    pi if shared_ref else ref,
                    tok, all_prompts_rep, all_full_texts,
                    pi_device if shared_ref else ref_device,
                    microbatch=mb,
                    max_length=max_len,
                    compute_grads=False,
                )
            finally:
                if shared_ref:
                    if hasattr(pi, 'enable_adapter_layers'):
                        pi.enable_adapter_layers()
            # Move ref outputs to pi_device for loss computation
            if not shared_ref:
                logp_ref = logp_ref.to(pi_device)
                per_tok_ref = [t.to(pi_device) for t in per_tok_ref]
        
        # Apply learning rate warmup
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for param_group in opt.param_groups:
                param_group['lr'] = base_lr * lr_scale

        loss, avg_kl = grpo_objective(
            logp_pi, logp_ref, rewards, group_ids, cfg.beta_kl,
            per_tok_pi=per_tok_pi, per_tok_ref=per_tok_ref,
        )

        if use_scaler:
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            # bf16: no scaler needed
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)
        
        # Extract loss value before cleanup
        cur_loss = float(loss.item())
        
        # Cleanup large tensors
        del logp_pi, logp_ref, per_tok_pi, per_tok_ref, rewards, group_ids, loss, all_full_texts, all_prompts_rep
        
        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------
        # Step end: logging
        # -------------------------
        step += 1
        dt = time.time() - t0
        steps_per_s = 1.0 / max(dt, 1e-9)
        tokens_per_s = float(token_cost / max(dt, 1e-9))

        # EMA loss for smoother dashboard
        if ema_loss is None:
            ema_loss = cur_loss
        else:
            ema_loss = (1 - ema_alpha) * ema_loss + ema_alpha * cur_loss

        # Update tqdm display each step (light)
        pbar.set_postfix({
            "loss": f"{cur_loss:.6f}",
            "ema": f"{ema_loss:.6f}",
            "R": f"{avg_reward:.2f}",
            "KL": f"{avg_kl:.6f}",
            "k": f"{avg_k_used:.2f}",
            "hard": f"{hard_rate:.2f}",
            "tok/s": f"{tokens_per_s:.0f}",
        })
        pbar.update(1)

        # Scalar logs
        if step % log_every == 0:
            payload = {
                "train/loss": cur_loss,
                "train/loss_ema": float(ema_loss),
                "train/avg_reward": avg_reward,
                "train/avg_kl": avg_kl,
                "train/hard_rate": hard_rate,
                "train/avg_k_used": avg_k_used,
                "train/token_cost": int(token_cost),
                "perf/steps_per_s": steps_per_s,
                "perf/tokens_per_s": tokens_per_s,
            }
            jsonl.log(step, payload)
            if wb: wb.log(step, payload)
            if tb: tb.log(step, payload)

            # write state snapshot (for "where are we?" at a glance)
            atomic_write_json(state_path, {
                "run_name": run_name,
                "step": step,
                "last_loss": cur_loss,
                "loss_ema": float(ema_loss),
                "avg_reward": avg_reward,
                "avg_kl": avg_kl,
                "avg_k_used": avg_k_used,
                "hard_rate": hard_rate,
                "token_cost": int(token_cost),
                "tokens_per_s": tokens_per_s,
                "qhats": {str(k): float(v) for k, v in qhats.items()},
                "time": time.time(),
            })

        # Periodic sample table
        if step % sample_every == 0:
            sample_rows = []
            sample_n = min(3, bs)
            for j in range(sample_n):
                outs = groups2[j] + extra_groups.get(j, [])
                # Extract completion-only text for answer extraction
                from verifier import canonicalize_answer
                completions = [o[len(prompts[j]):] if o.startswith(prompts[j]) else o for o in outs]
                answers = [canonicalize_answer(c, cfg.dataset_name) for c in completions]
                k_used, Ck = choose_k_and_sets(answers, qhats, k_values)

                used_answers = answers[:k_used]
                # Show completion only (not echoed prompt)
                comp_preview = completions[0][:300] if completions[0] else "<empty>"
                sample_rows.append([
                    prompts[j][:100],
                    comp_preview,
                    used_answers,
                    Ck,
                    int(k_used),
                    gold[j][:80],
                ])
                
                # Early warning if we're generating garbage
                if step <= 10 and len(completions[0]) > 20:
                    unique_chars = len(set(completions[0].replace(' ', '')))
                    if unique_chars < 10:
                        print(f"\n⚠️  WARNING step {step}: Degenerate output detected (unique chars: {unique_chars})")
                        print(f"  Completion: {completions[0][:150]}...")

            jsonl.log(step, {"event": "samples", "rows": sample_rows})
            if wb:
                wb.log_table(
                    name="samples",
                    columns=["prompt", "output0", "answers_used", "conformal_set", "k_used", "gold"],
                    rows=sample_rows,
                    step=step,
                )

        # Periodic evaluation
        if step % eval_every == 0:
            stats = evaluate(cfg, tok, pi, eval_ds, qhats, device=pi_device)
            # Print eval results to stdout
            k_usage_str = " ".join(
                f"k{k_val}={cnt}" for k_val, cnt in sorted(stats.get("k_usage", {}).items())
            )
            print(f"\n📊 Eval@{step}: greedy_acc={stats['greedy_acc']:.3f}  "
                  f"conf_acc={stats['conf_acc']:.3f}  "
                  f"coverage={stats['conf_coverage']:.3f}  "
                  f"avg_k={stats['avg_k']:.1f}  "
                  f"[{k_usage_str}]")
            eval_payload = {
                "eval/greedy_acc": float(stats["greedy_acc"]),
                "eval/conf_acc": float(stats["conf_acc"]),
                "eval/conf_coverage": float(stats["conf_coverage"]),
                "eval/avg_k": float(stats["avg_k"]),
                "eval/avg_new_tokens": float(stats["avg_new_tokens"]),
            }
            # Add per-k usage breakdown
            for k_val, cnt in stats.get("k_usage", {}).items():
                eval_payload[f"eval/k_{k_val}_pct"] = cnt / max(sum(stats["k_usage"].values()), 1)
            jsonl.log(step, eval_payload)
            if wb: wb.log(step, eval_payload)
            if tb: tb.log(step, eval_payload)

        # Periodic checkpointing
        if step % ckpt_every == 0:
            save_checkpoint(pi, opt, step, run_dir, tokenizer=tok, qhats=qhats)

        # Periodic re-calibration of conformal qhats using current pi
        recal_every = int(getattr(cfg, "recalibrate_every", 0))
        if recal_every > 0 and step % recal_every == 0 and step > 0:
            print(f"\n🔄 Re-calibrating conformal qhats at step {step}...")
            old_qhats = dict(qhats)
            old_delta = delta
            # Re-run auto_delta with the *current trained policy* (adapters ON)
            # so that as the model improves (higher pass@k_max), δ decreases and
            # qhats tighten.  shared_ref=False here is critical: passing False
            # prevents calibrate_conformal from disabling LoRA adapters, so we
            # actually measure the policy's pass@k, not the frozen base model's.
            # Without this fix, every recalibration showed the base model's
            # pass@32 ≈ 15% → δ ≈ 0.90 → qhat=1.0 forever.
            with phase('recalibration'):
                qhats, delta = calibrate_conformal(
                    cfg, tok, pi, cal_ds, delta,
                    device=pi_device, shared_ref=False,
                    auto_delta=getattr(cfg, "auto_delta", False))
            if delta != old_delta:
                print(f"   δ updated: {old_delta:.3f} → {delta:.3f}")
            recal_payload = {
                "event": "recalibration",
                "old_qhats": {str(k): float(v) for k, v in old_qhats.items()},
                "new_qhats": {str(k): float(v) for k, v in qhats.items()},
            }
            jsonl.log(step, recal_payload)
            for k, v in qhats.items():
                qhat_scalars = {f"qhats/k={k}": float(v)}
                if wb: wb.log(step, qhat_scalars)
                if tb: tb.log(step, qhat_scalars)
            print(f"   New qhats: {qhats}")

    # Final save
    final_dir = os.path.join(run_dir, "final")
    ensure_dir(final_dir)
    pi.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    jsonl.log(step, {"event": "train_done", "final_dir": final_dir, **phase.as_dict()})
    print("\n  Phase wall-clock (h):")
    for _k, _v in sorted(phase.as_dict().items()):
        print(f"    {_k:34} {_v:7.3f}")
    atomic_write_json(state_path, {
        "event": "train_done",
        "run_name": run_name,
        "step": step,
        "final_dir": final_dir,
        "qhats": {str(k): float(v) for k, v in qhats.items()},
        "time": time.time(),
    })

    # Cleanup loggers
    try:
        jsonl.close()
    except Exception:
        pass
    try:
        if tb: tb.flush()
    except Exception:
        pass
    try:
        if wb: wb.finish()
    except Exception:
        pass

    return qhats


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
