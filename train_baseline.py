"""
Unified baseline trainer for GRPO variants.

Supported methods (--method flag):
  grpo           – standard GRPO with fixed n rollouts (no conformal)
  aero           – AERO: three-stage rolling budget (arxiv 2602.14338)
  gdro           – GDRO: Prompt-GDRO + Rollout-GDRO (arxiv 2601.19280)
  gdro_prompt    – Prompt-GDRO only (advantage scaling, fixed n)
  reinforce_ada  – Reinforce-Ada-Seq-Balance (arxiv 2510.04996)
  reinforce_est  – Reinforce-Ada-Est / EMA allocation variant
  greso          – GRESO selective rollouts (NeurIPS 2025)

Each method calls its curate_rollouts() function, which returns:
  prompts_out, texts_out, rewards_out, group_ids_out [, inv_p_weights_out]

The standard GRPO loss (grpo_objective) is applied on the curated batch.
For methods that return per-rollout gradient weights (GDRO, Reinforce-Ada),
the weights are folded into the advantage computation via the `adv_weights`
argument of grpo_objective.
"""
from __future__ import annotations
import os
import time
import json
import random
from typing import List, Optional

import torch
import numpy as np
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import TrainConfig
from data import load_gsm8k_splits, format_prompt, extract_example
from generation import generate_n, count_new_tokens
from grpo_loss import completion_logprob, grpo_objective
from logger import make_loggers
from model_registry import get_model_path, get_model_dtype_str
from utils import set_seed, pick_cuda_devices, batch_to_examples, total_free_gpu_mib
from verifier import verify_answer, canonicalize_answer

# Re-use model-building helpers from train.py
from train import (
    build_models,
    ensure_dir,
    atomic_write_json,
    _now_str,
    save_checkpoint,
    evaluate,                 # greedy + sampling eval
    calibrate_conformal,      # used only for eval qhats (not training)
)


# ──────────────────────────────────────────────────────────────────────────────
# Method dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _dispatch_curate(
    method: str,
    policy,
    tokenizer,
    prompts: List[str],
    gold: List[str],
    dataset_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    n_rollouts: int,
    state,                   # method-specific mutable state object (can be None)
):
    """Call the right curate_rollouts and return a normalised 5-tuple.

    Returns
    -------
    prompts_out    : List[str]   – per-rollout prompt
    texts_out      : List[str]   – per-rollout full text (prompt + completion)
    rewards_out    : List[float] – per-rollout 0/1 reward
    group_ids_out  : List[int]   – per-rollout group id
    adv_weights    : Optional[List[float]]  – per-rollout weight or None
    """
    if method == "grpo":
        # Vanilla GRPO: fixed n rollouts, no filtering
        groups = generate_n(
            policy, tokenizer, prompts, n_rollouts,
            max_new_tokens, temperature, top_p, device,
        )
        p_out, t_out, r_out, g_out = [], [], [], []
        for i, outs in enumerate(groups):
            for out in outs:
                comp = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
                r = float(verify_answer(comp, gold[i], dataset_name))
                p_out.append(prompts[i])
                t_out.append(out)
                r_out.append(r)
                g_out.append(i)
        return p_out, t_out, r_out, g_out, None

    elif method == "aero":
        from baselines.aero import curate_rollouts as _aero
        p, t, r, g = _aero(
            policy, tokenizer, prompts, gold, dataset_name,
            max_new_tokens, temperature, top_p, device,
        )
        return p, t, r, g, None

    elif method in ("gdro", "gdro_prompt"):
        from baselines.gdro import curate_rollouts as _gdro
        use_rollout = (method == "gdro")
        p, t, r, g, w = _gdro(
            policy, tokenizer, prompts, gold, dataset_name,
            max_new_tokens, temperature, top_p, device,
            state=state,
            use_rollout_gdro=use_rollout,
        )
        return p, t, r, g, w

    elif method == "reinforce_ada":
        from baselines.reinforce_ada import curate_rollouts_seq
        p, t, r, g, w = curate_rollouts_seq(
            policy, tokenizer, prompts, gold, dataset_name,
            max_new_tokens, temperature, top_p, device,
        )
        return p, t, r, g, w

    elif method == "reinforce_est":
        from baselines.reinforce_ada import curate_rollouts_est
        p, t, r, g, w = curate_rollouts_est(
            policy, tokenizer, prompts, gold, dataset_name,
            max_new_tokens, temperature, top_p, device,
            state=state,
            n_budget=n_rollouts,
        )
        return p, t, r, g, w

    elif method == "greso":
        from baselines.greso import curate_rollouts as _greso
        p, t, r, g = _greso(
            policy, tokenizer, prompts, gold, dataset_name,
            max_new_tokens, temperature, top_p, device,
            state=state,
            n_rollouts=n_rollouts,
        )
        return p, t, r, g, None

    else:
        raise ValueError(f"Unknown baseline method: {method!r}")


def _make_state(method: str, n_rollouts: int):
    """Create method-specific mutable state."""
    if method in ("gdro", "gdro_prompt"):
        from baselines.gdro import GDROState
        return GDROState()
    elif method == "reinforce_est":
        from baselines.reinforce_ada import ReinfAdaEMAState
        return ReinfAdaEMAState()
    elif method == "greso":
        from baselines.greso import GRESOState
        return GRESOState(n=n_rollouts)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public training entry-point
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline(cfg: TrainConfig, method: str = "grpo", n_rollouts: int = 4):
    """Train with a specified baseline rollout-curation method.

    Parameters
    ----------
    cfg        : standard TrainConfig (shared with conformal GRPO)
    method     : one of grpo | aero | gdro | gdro_prompt | reinforce_ada |
                 reinforce_est | greso
    n_rollouts : default rollouts per prompt for vanilla GRPO / GRESO
    """
    set_seed(cfg.seed)

    # ── GPU placement ────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    print(f"CUDA devices available: {n_gpus}")
    if n_gpus >= 2:
        chosen = pick_cuda_devices(2)
        pi_device = torch.device(f"cuda:{chosen[0]}")
        ref_device = torch.device(f"cuda:{chosen[1]}")
    elif n_gpus == 1:
        pi_device = ref_device = torch.device("cuda:0")
    else:
        raise RuntimeError("No CUDA GPUs available.")

    # ── AMP ──────────────────────────────────────────────────────────────────
    amp_dtype  = torch.bfloat16 if cfg.use_bf16 else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler     = GradScaler("cuda", enabled=use_scaler)

    # ── Logging dirs ─────────────────────────────────────────────────────────
    run_name = f"{method}_{cfg.model_key}_seed{cfg.seed}_{_now_str()}"
    run_dir  = os.path.join(cfg.log_dir, run_name)
    ensure_dir(run_dir)

    jsonl, wb, tb = make_loggers(
        cfg,
        log_dir=run_dir,
        use_wandb=getattr(cfg, "use_wandb", False),
        wandb_project=getattr(cfg, "wandb_project", "baseline-grpo"),
        run_name=run_name,
        use_tensorboard=getattr(cfg, "use_tensorboard", True),
    )

    state_path = os.path.join(run_dir, "state.json")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, cal_ds, eval_ds = load_gsm8k_splits(
        cfg.dataset_name, cfg.dataset_config, cfg.n_train, cfg.n_cal, cfg.n_eval, cfg.seed
    )

    # ── Models ───────────────────────────────────────────────────────────────
    tok, pi, ref, shared_ref = build_models(cfg, pi_device, ref_device)
    if shared_ref:
        ref_device = next(pi.parameters()).device

    opt = AdamW(pi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=(0.9, 0.95), eps=1e-8)

    # ── Build prompt/answer lists ─────────────────────────────────────────────
    train_pairs   = [extract_example(ex, cfg.dataset_name) for ex in train_ds]
    train_prompts = [format_prompt(text, cfg.dataset_name) for text, _ in train_pairs]
    train_answers = [ans for _, ans in train_pairs]

    # ── Minimal conformal calibration (for eval only) ────────────────────────
    # We use conformal only for the EVALUATION path (same as train.py) so we
    # report comparable greedy + conf_acc numbers. Training does NOT use qhats.
    delta = cfg.deltas[0]
    _cal_model  = pi if shared_ref else ref
    _cal_device = ref_device
    print("Running conformal calibration for eval qhats...")
    qhats, delta = calibrate_conformal(cfg, tok, _cal_model, cal_ds, delta,
                                       device=_cal_device, shared_ref=shared_ref)
    print(f"  qhats: {qhats}")
    jsonl.log(0, {"event": "calibration_done", "delta": delta, "qhats": qhats})
    atomic_write_json(state_path, {"event": "calibration_done", "run_name": run_name,
                                   "qhats": {str(k): float(v) for k, v in qhats.items()}})

    # ── Per-method state object ───────────────────────────────────────────────
    method_state = _make_state(method, n_rollouts)

    # ── Training config ───────────────────────────────────────────────────────
    bs           = cfg.batch_size
    log_every    = int(getattr(cfg, "log_every",   1))
    eval_every   = int(getattr(cfg, "eval_every",  50))
    ckpt_every   = int(getattr(cfg, "ckpt_every",  100))
    warmup_steps = min(100, cfg.steps // 5)
    base_lr      = cfg.lr
    mb           = getattr(cfg, "logprob_microbatch", 2)
    max_len      = getattr(cfg, "max_seq_len", None)
    ema_loss     = None
    ema_alpha    = 0.05

    step = 0
    pbar = tqdm(total=cfg.steps, desc=f"[{method}] Train")

    while step < cfg.steps:
        t0 = time.time()

        # ── Sample batch ──────────────────────────────────────────────────────
        idx     = torch.randint(0, len(train_prompts), (bs,)).tolist()
        prompts = [train_prompts[i] for i in idx]
        gold    = [train_answers[i]  for i in idx]

        # ── Curate rollouts ───────────────────────────────────────────────────
        p_out, t_out, r_out, g_out, adv_w = _dispatch_curate(
            method, pi, tok, prompts, gold, cfg.dataset_name,
            cfg.max_new_tokens, cfg.temperature, cfg.top_p, pi_device,
            n_rollouts, method_state,
        )

        if len(p_out) == 0:
            # All prompts skipped (GRESO) or other edge case
            step += 1
            pbar.update(1)
            continue

        rewards   = torch.tensor(r_out, dtype=torch.float32, device=pi_device)
        group_ids = torch.tensor(g_out, dtype=torch.long,    device=pi_device)
        avg_reward = float(rewards.mean().item())

        token_cost = sum(
            count_new_tokens(tok, p, t)
            for p, t in zip(p_out, t_out)
        )

        # ── GRPO loss ─────────────────────────────────────────────────────────
        with autocast("cuda", dtype=amp_dtype):
            logp_pi, per_tok_pi = completion_logprob(
                pi, tok, p_out, t_out, pi_device,
                microbatch=mb, max_length=max_len, compute_grads=True,
            )

        with torch.no_grad():
            if shared_ref:
                pi.disable_adapter_layers()
            try:
                logp_ref, per_tok_ref = completion_logprob(
                    pi if shared_ref else ref,
                    tok, p_out, t_out,
                    pi_device if shared_ref else ref_device,
                    microbatch=mb, max_length=max_len, compute_grads=False,
                )
            finally:
                if shared_ref:
                    pi.enable_adapter_layers()
            if not shared_ref:
                logp_ref     = logp_ref.to(pi_device)
                per_tok_ref  = [t.to(pi_device) for t in per_tok_ref]

        # LR warmup
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for pg in opt.param_groups:
                pg["lr"] = base_lr * lr_scale

        # Fold per-rollout advantage weights (GDRO / Reinforce-Ada)
        adv_weights_t: Optional[torch.Tensor] = None
        if adv_w is not None:
            adv_weights_t = torch.tensor(adv_w, dtype=torch.float32, device=pi_device)

        loss, avg_kl = grpo_objective(
            logp_pi, logp_ref, rewards, group_ids, cfg.beta_kl,
            per_tok_pi=per_tok_pi, per_tok_ref=per_tok_ref,
            adv_weights=adv_weights_t,
        )

        if use_scaler:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)

        cur_loss = float(loss.item())

        del logp_pi, logp_ref, per_tok_pi, per_tok_ref, rewards, group_ids, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step += 1
        dt           = time.time() - t0
        tokens_per_s = float(token_cost / max(dt, 1e-9))

        ema_loss = cur_loss if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * cur_loss

        pbar.set_postfix({
            "loss": f"{cur_loss:.5f}",
            "R":    f"{avg_reward:.2f}",
            "KL":   f"{avg_kl:.5f}",
            "tok/s":f"{tokens_per_s:.0f}",
        })
        pbar.update(1)

        # ── Logging ────────────────────────────────────────────────────────────
        if step % log_every == 0:
            payload = {
                "train/loss":        cur_loss,
                "train/loss_ema":    float(ema_loss),
                "train/avg_reward":  avg_reward,
                "train/avg_kl":      avg_kl,
                "train/token_cost":  int(token_cost),
                "train/n_rollouts":  len(t_out),
                "perf/tokens_per_s": tokens_per_s,
            }
            # GRESO: log skip fraction
            if method == "greso" and method_state is not None:
                stats = method_state.stats()
                payload["greso/frac_skipped"] = stats.get("frac_skippable", 0.0)
                payload["greso/mean_p_hat"]   = stats.get("mean_p_hat", 0.5)
            # GDRO: log mean bin weight
            if method in ("gdro", "gdro_prompt") and method_state is not None and adv_w:
                payload["gdro/mean_adv_weight"] = float(np.mean(adv_w))
            jsonl.log(step, payload)
            if wb: wb.log(step, payload)
            if tb: tb.log(step, payload)

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % eval_every == 0:
            print(f"\n[step {step}] Evaluating {method}...")
            results = evaluate(cfg, tok, pi, eval_ds, qhats, device=pi_device)
            eval_payload = {**{f"eval/{k}": v for k, v in results.items()}, "step": step}
            print(f"  greedy_acc={results['greedy_acc']:.3f}  conf_acc={results['conf_acc']:.3f}")
            jsonl.log(step, eval_payload)
            if wb: wb.log(step, eval_payload)
            if tb: tb.log(step, eval_payload)
            atomic_write_json(state_path, {"step": step, "eval": results, "method": method})

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % ckpt_every == 0:
            save_checkpoint(pi, opt, step, run_dir, tokenizer=tok, qhats=qhats)
            print(f"  [ckpt saved @ step {step}]")

    pbar.close()

    # ── Final checkpoint ─────────────────────────────────────────────────────
    save_checkpoint(pi, opt, step, run_dir, tokenizer=tok, qhats=qhats)

    # ── Final eval ───────────────────────────────────────────────────────────
    print(f"\n[FINAL] Evaluating {method}...")
    results = evaluate(cfg, tok, pi, eval_ds, qhats, device=pi_device)
    print(f"  greedy_acc={results['greedy_acc']:.3f}  conf_acc={results['conf_acc']:.3f}")
    jsonl.log(step, {"final_eval": results, "method": method})
    atomic_write_json(state_path, {"step": step, "final_eval": results, "method": method})

    return results
