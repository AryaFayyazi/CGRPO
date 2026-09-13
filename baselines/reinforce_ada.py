"""
Reinforce-Ada: Adaptive Sampling Framework under Non-linear RL Objectives
Paper: arxiv.org/abs/2510.04996  (RLHFlow / NUS / Microsoft)

Two variants implemented:

── Reinforce-Ada-Seq-Balance ────────────────────────────────────────────────
  Motivation: optimize the log-likelihood objective J_f(θ) = E_x[log p_θ(x)]
  whose gradient is 1/p * ∇p, naturally weighting hard prompts more.

  Algorithm (Algorithm 2 in paper, "Balance" exit condition):
    1. All prompts start active.
    2. Each round: sample M responses for each active prompt.
    3. Deactivate a prompt when BOTH K_pos correct AND K_neg incorrect are found.
    4. After max rounds or all deactivated: compute p̂ = N_pos / N_total per prompt.
    5. Downsample pool to n responses: balanced n/2 correct + n/2 incorrect.
    6. Re-weight gradient by 1/p̂ (explicit log-objective weight).

  Final gradient estimator (Eq. in Section 3.2):
    ĝ(x_i) = (1/p̂_i) · (1/n · Σ_j ∇log π(a_j) · (r_j - p̂_i))

── Reinforce-Ada-Est (EMA variant) ─────────────────────────────────────────
  1. Maintain EMA estimate p̂_x across steps for each prompt.
  2. Allocate n_i ∝ 1/p̂_i (clipped to [N_min, N_max]).
  3. Re-weight gradient by 1/p̂_i.

  EMA update: N_total ← λ*N_total + n,  N_pos ← λ*N_pos + k_correct
  Bayesian: p̂ = (N_pos + α) / (N_total + α + β)   with α=β=0.5 (Jeffreys prior)
"""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
# Ada-Seq-Balance
M_PER_ROUND  = 4    # responses sampled per round per active prompt
K_POS        = 4    # stop when this many correct found
K_NEG        = 4    # stop when this many incorrect found
N_MAX_SEQ    = 64   # hard cap on total responses generated per prompt
N_UPDATE     = 4    # responses used for gradient update (2 pos + 2 neg)

# Ada-EMA
LAMBDA_EMA   = 0.85 # EMA decay factor
ALPHA_PRIOR  = 0.5  # Jeffreys Beta prior α
BETA_PRIOR   = 0.5  # Jeffreys Beta prior β
N_MIN_EST    = 2    # minimum allocation
N_MAX_EST    = 32   # maximum allocation


class ReinfAdaEMAState:
    """Per-prompt EMA state for Reinforce-Ada-Est."""

    def __init__(self):
        self._data: Dict[str, Tuple[float, float]] = {}  # uid → (N_total, N_pos)

    def _uid(self, prompt: str) -> str:
        return str(hash(prompt) % (2**32))

    def get_p_hat(self, prompt: str) -> float:
        uid = self._uid(prompt)
        if uid not in self._data:
            return 0.5  # default prior
        n_total, n_pos = self._data[uid]
        return float((n_pos + ALPHA_PRIOR) / (n_total + ALPHA_PRIOR + BETA_PRIOR))

    def update(self, prompt: str, n_tried: int, n_correct: int) -> None:
        uid = self._uid(prompt)
        n_total, n_pos = self._data.get(uid, (0.0, 0.0))
        self._data[uid] = (
            LAMBDA_EMA * n_total + n_tried,
            LAMBDA_EMA * n_pos  + n_correct,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Ada-Seq-Balance
# ──────────────────────────────────────────────────────────────────────────────

def curate_rollouts_seq(
    policy,
    tokenizer,
    prompts: List[str],
    gold_answers: List[str],
    dataset_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[List[str], List[str], List[float], List[int], List[float]]:
    """Reinforce-Ada-Seq-Balance rollout curation.

    Returns
    -------
    curated_prompts   : prompts for each retained rollout
    curated_texts     : full (prompt + completion) strings
    curated_rewards   : 0/1 rewards relative to p̂ baseline
    curated_group_ids : group id per rollout
    inv_p_weights     : 1/p̂ weight per group (broadcast to group members)
    """
    from generation import generate_n
    from verifier import verify_answer

    batch_size = len(prompts)

    # Per-prompt response pools
    pools_correct: List[List[str]] = [[] for _ in range(batch_size)]
    pools_wrong:   List[List[str]] = [[] for _ in range(batch_size)]
    n_total_per:   List[int]       = [0] * batch_size

    active = list(range(batch_size))

    n_rounds = max(1, N_MAX_SEQ // M_PER_ROUND)
    for _round in range(n_rounds):
        if not active:
            break

        active_prompts = [prompts[i] for i in active]
        groups = generate_n(
            policy, tokenizer, active_prompts, M_PER_ROUND,
            max_new_tokens, temperature, top_p, device,
        )

        still_active = []
        for rel_idx, i in enumerate(active):
            for out in groups[rel_idx]:
                comp = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
                r = verify_answer(comp, gold_answers[i], dataset_name)
                n_total_per[i] += 1
                if r > 0:
                    pools_correct[i].append(out)
                else:
                    pools_wrong[i].append(out)

            # Balance exit: deactivate when K_pos correct AND K_neg incorrect found
            if len(pools_correct[i]) >= K_POS and len(pools_wrong[i]) >= K_NEG:
                pass  # deactivated (don't append to still_active)
            elif n_total_per[i] >= N_MAX_SEQ:
                pass  # hard budget
            else:
                still_active.append(i)

        active = still_active

    # ── Build static training batch ────────────────────────────────────────
    half = N_UPDATE // 2
    all_prompts_out:   List[str]   = []
    all_texts_out:     List[str]   = []
    all_rewards_out:   List[float] = []
    all_group_ids_out: List[int]   = []
    all_weights_out:   List[float] = []

    for i in range(batch_size):
        n_pos = len(pools_correct[i])
        n_neg = len(pools_wrong[i])
        n_total = n_total_per[i]

        # High-fidelity p̂
        if n_total > 0:
            p_hat = float(np.clip(n_pos / n_total, 1e-4, 1.0 - 1e-4))
        else:
            p_hat = 0.5

        inv_p = 1.0 / p_hat

        # Balanced down-sample: half correct, half incorrect
        chosen_correct = random.sample(pools_correct[i], min(half, n_pos))
        chosen_wrong   = random.sample(pools_wrong[i],   min(half, n_neg))

        # Pad with opposite if one pool is too small
        shortage = N_UPDATE - len(chosen_correct) - len(chosen_wrong)
        if shortage > 0:
            extra = (pools_correct[i] + pools_wrong[i])
            extra = [x for x in extra if x not in chosen_correct and x not in chosen_wrong]
            random.shuffle(extra)
            chosen_correct.extend([e for e in extra[:shortage] if e in pools_correct[i]])
            chosen_wrong.extend([e for e in extra[:shortage] if e in pools_wrong[i]])

        kept = [(out, 1.0) for out in chosen_correct] + [(out, 0.0) for out in chosen_wrong]
        if not kept:
            continue

        for out, r in kept:
            all_prompts_out.append(prompts[i])
            all_texts_out.append(out)
            all_rewards_out.append(r)
            all_group_ids_out.append(i)
            all_weights_out.append(inv_p)   # 1/p̂ re-weighting factor

    return all_prompts_out, all_texts_out, all_rewards_out, all_group_ids_out, all_weights_out


# ──────────────────────────────────────────────────────────────────────────────
# Ada-Est (EMA-based allocation)
# ──────────────────────────────────────────────────────────────────────────────

def curate_rollouts_est(
    policy,
    tokenizer,
    prompts: List[str],
    gold_answers: List[str],
    dataset_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    state: Optional[ReinfAdaEMAState] = None,
    n_budget: int = 8,          # roughly matches Ada-Seq inference cost
) -> Tuple[List[str], List[str], List[float], List[int], List[float]]:
    """Reinforce-Ada-Est rollout curation (EMA pass-rate estimation).

    Hybrid strategy (Section 3.1): allocate n_i ∝ 1/p̂_i and apply
    explicit gradient weight 1/p̂_i (both halves of the log-objective).
    """
    from generation import generate_n
    from verifier import verify_answer

    if state is None:
        state = ReinfAdaEMAState()

    batch_size = len(prompts)
    p_hats = [state.get_p_hat(p) for p in prompts]

    # Allocate n_i ∝ 1/p̂_i subject to mean = n_budget
    inv_ps     = [1.0 / max(p, 1e-3) for p in p_hats]
    total_inv  = sum(inv_ps) + 1e-8
    allocs     = [int(round(np.clip(
        (ip / total_inv) * n_budget * batch_size, N_MIN_EST, N_MAX_EST
    ))) for ip in inv_ps]

    all_prompts_out:   List[str]   = []
    all_texts_out:     List[str]   = []
    all_rewards_out:   List[float] = []
    all_group_ids_out: List[int]   = []
    all_weights_out:   List[float] = []

    for i in range(batch_size):
        n_i = allocs[i]
        inv_p = 1.0 / max(p_hats[i], 1e-4)
        outs_i = generate_n(
            policy, tokenizer, [prompts[i]], n_i,
            max_new_tokens, temperature, top_p, device,
        )[0]

        n_correct = 0
        for out in outs_i:
            comp = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
            r = float(verify_answer(comp, gold_answers[i], dataset_name))
            all_prompts_out.append(prompts[i])
            all_texts_out.append(out)
            all_rewards_out.append(r)
            all_group_ids_out.append(i)
            all_weights_out.append(inv_p)
            if r > 0:
                n_correct += 1

        state.update(prompts[i], n_i, n_correct)

    return all_prompts_out, all_texts_out, all_rewards_out, all_group_ids_out, all_weights_out
