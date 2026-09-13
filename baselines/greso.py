"""
GRESO: Act Only When It Pays — Efficient RL for LLM Reasoning via Selective Rollouts
Paper: NeurIPS 2025 poster 115272  (Infini-AI Lab)

Key Idea
────────
GRPO wastes compute on *uninformative* prompts:
  • all-correct  (reward=1 for all rollouts → zero advantage → zero gradient)
  • all-wrong    (reward=0 for all rollouts → zero advantage → zero gradient)

GRESO maintains a lightweight per-prompt EMA pass-rate p̂ and predicts,
*before* generating rollouts, whether a prompt will be uninformative.
If the probability of all responses being correct (p̂^n) or all wrong
((1−p̂)^n) exceeds a threshold θ, the prompt is skipped entirely.

This yields ~2.4× rollout speedup, ~2.0× total training speedup, while
preserving or improving accuracy.

Algorithm
─────────
1. For each prompt x, look up current p̂_x (EMA from previous steps,
   initialised to 0.5 so no prompts are skipped at the start).
2. Skip-predict:
   skip(x, n) ←  p̂_x^n > θ   (likely all-correct)
             OR  (1−p̂_x)^n > θ  (likely all-wrong)
3. Generate n rollouts ONLY for non-skipped prompts.
4. After generation, update EMA p̂_x for all non-skipped prompts.
5. Return rollouts (empty lists for skipped prompts).

Hyperparameters (matching paper, Table 1)
─────────────────────────────────────────
  θ   = 0.50   (skip threshold, tuned on GSM8K/MATH validation)
  λ   = 0.90   (EMA decay per training step)
  n   = 4      (base generation count — same as standard GRPO batch)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
THETA_SKIP   = 0.50   # skip threshold
LAMBDA_EMA   = 0.90   # EMA decay for pass-rate tracking
N_ROLLOUTS   = 4      # default rollouts per non-skipped prompt
ALPHA_PRIOR  = 1.0    # Beta(1,1) uniform prior: avoid cold-start skips
BETA_PRIOR   = 1.0
EMA_WARMUP   = 3      # steps before skipping is allowed (let p̂ stabilise)


class GRESOState:
    """Per-prompt EMA pass-rate state for GRESO filtering."""

    def __init__(
        self,
        theta:      float = THETA_SKIP,
        lambda_ema: float = LAMBDA_EMA,
        n:          int   = N_ROLLOUTS,
        warmup:     int   = EMA_WARMUP,
    ):
        self.theta      = theta
        self.lambda_ema = lambda_ema
        self.n          = n
        self.warmup     = warmup

        # uid → (N_total_ema, N_pos_ema, step_count)
        self._data: Dict[str, Tuple[float, float, int]] = {}
        self.step: int = 0          # global step counter

    def _uid(self, prompt: str) -> str:
        return str(hash(prompt) % (2**32))

    def get_p_hat(self, prompt: str) -> float:
        uid = self._uid(prompt)
        if uid not in self._data:
            return float(ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR))  # = 0.5
        n_total, n_pos, _ = self._data[uid]
        return float((n_pos + ALPHA_PRIOR) / (n_total + ALPHA_PRIOR + BETA_PRIOR))

    def should_skip(self, prompt: str) -> bool:
        uid  = self._uid(prompt)
        _, _, step_count = self._data.get(uid, (0.0, 0.0, 0))

        if step_count < self.warmup:
            return False  # not enough history yet

        p_hat = self.get_p_hat(prompt)
        n = self.n

        prob_all_correct = p_hat ** n
        prob_all_wrong   = (1.0 - p_hat) ** n

        return bool(prob_all_correct > self.theta or prob_all_wrong > self.theta)

    def update(self, prompt: str, n_tried: int, n_correct: int) -> None:
        uid = self._uid(prompt)
        n_total_ema, n_pos_ema, cnt = self._data.get(uid, (0.0, 0.0, 0))
        self._data[uid] = (
            self.lambda_ema * n_total_ema + n_tried,
            self.lambda_ema * n_pos_ema  + n_correct,
            cnt + 1,
        )

    def increment_step(self) -> None:
        self.step += 1

    def stats(self) -> Dict[str, float]:
        """Return diagnostic statistics."""
        n_prompts = len(self._data)
        if n_prompts == 0:
            return {"tracked": 0, "frac_skippable": 0.0, "mean_p_hat": 0.5}
        p_hats = [self.get_p_hat(p) for p in self._data]
        frac = sum(1 for p in self._data if self.should_skip(p)) / n_prompts
        return {
            "tracked":         n_prompts,
            "frac_skippable":  frac,
            "mean_p_hat":      float(np.mean(p_hats)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main curation function
# ──────────────────────────────────────────────────────────────────────────────

def curate_rollouts(
    policy,
    tokenizer,
    prompts: List[str],
    gold_answers: List[str],
    dataset_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    state: Optional[GRESOState] = None,
    n_rollouts: int = N_ROLLOUTS,
) -> Tuple[List[str], List[str], List[float], List[int]]:
    """GRESO selective-rollout curation.

    Parameters
    ----------
    state : GRESOState, optional
        Persistent EMA state across training steps.  Caller is responsible
        for creating it once and passing it every step.

    Returns
    -------
    curated_prompts   : only prompts that were NOT skipped
    curated_texts     : full strings (prompt + completion)
    curated_rewards   : 0/1 rewards
    curated_group_ids : per-rollout group id (index into original batch)
    """
    from generation import generate_n
    from verifier import verify_answer

    if state is None:
        state = GRESOState(n=n_rollouts)

    batch_size = len(prompts)

    # ── 1. Pre-rollout filter ──────────────────────────────────────────────
    keep_indices = [i for i in range(batch_size) if not state.should_skip(prompts[i])]
    skip_indices = [i for i in range(batch_size) if state.should_skip(prompts[i])]

    n_skipped = len(skip_indices)
    n_kept    = len(keep_indices)

    # Log fraction skipped (accessible via state.stats())
    # Do NOT generate rollouts for skipped prompts.
    if n_kept == 0:
        state.increment_step()
        return [], [], [], []

    kept_prompts = [prompts[i] for i in keep_indices]
    kept_golds   = [gold_answers[i] for i in keep_indices]

    # ── 2. Generate rollouts for kept prompts ─────────────────────────────
    groups = generate_n(
        policy, tokenizer, kept_prompts, n_rollouts,
        max_new_tokens, temperature, top_p, device,
    )

    # ── 3. Verify and collect ─────────────────────────────────────────────
    all_prompts_out:   List[str]   = []
    all_texts_out:     List[str]   = []
    all_rewards_out:   List[float] = []
    all_group_ids_out: List[int]   = []

    for rel_idx, orig_idx in enumerate(keep_indices):
        n_correct = 0
        for out in groups[rel_idx]:
            comp = out[len(prompts[orig_idx]):] if out.startswith(prompts[orig_idx]) else out
            r = float(verify_answer(comp, gold_answers[orig_idx], dataset_name))
            all_prompts_out.append(prompts[orig_idx])
            all_texts_out.append(out)
            all_rewards_out.append(r)
            all_group_ids_out.append(orig_idx)
            if r > 0:
                n_correct += 1

        # ── 4. Update EMA pass-rate for this prompt ────────────────────
        state.update(prompts[orig_idx], n_rollouts, n_correct)

    state.increment_step()

    return all_prompts_out, all_texts_out, all_rewards_out, all_group_ids_out
