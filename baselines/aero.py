"""
AERO: Adaptive Efficient Rollout Optimization for Group-Based RL
Paper: arxiv.org/abs/2602.14338
"Train Less, Learn More" (UCLA + AWS)

Algorithm summary (Section 3 of paper):

Stage I – Exploration (all prompts):
    Generate n_explore=8 rollouts, compute empirical success rate u = c/n_explore.

Stage II – Differentiated Exploitation:
    Strategy 1 – Iterative Rescue (u == 0, zero-accuracy):
        Iteratively generate n_extra=2 more rollouts until first success or
        total budget n_total=16 is exhausted.  This "rescues" hard prompts.

    Strategy 2 – Bayesian Posterior (all-fail after rescue OR all-correct):
        Replace empirical u with Bayesian posterior:
            ũ = (c + α₀) / (n + α₀ + β₀),  α₀=β₀=1  (Beta(1,1) flat prior)
        Keep 4 rollouts (2 from the non-empty majority outcome, 2 exploratory).
        All rollouts assigned constant advantage: ũ - (1 - ũ) = 2ũ - 1.
        This prevents zero-gradient while remaining calibrated.

    Strategy 3 – Rejection Sampling (0 < u < threshold=0.5, partial-low):
        Keep all c correct rollouts.
        Downsample incorrect to m = k * c  where k=1 (optimal by Lemma 4.2).
        (Gradient norm maximised when c == m.)

    High-success (u >= threshold): keep all n_explore rollouts unchanged.

After curation: standard GRPO advantage normalisation + policy gradient update.
"""
from __future__ import annotations
import random
from typing import List, Tuple

import torch

# NOTE: import from parent package at call time to avoid circular imports

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters (matching paper Tables 1 / Appendix)
# ──────────────────────────────────────────────────────────────────────────────
N_TOTAL    = 16   # maximum rollouts per prompt
N_EXPLORE  = 8    # initial exploration budget
N_EXTRA    = 2    # rollouts added per rescue iteration
K_RATIO    = 1    # rejection sampling ratio: m = k * c incorrect kept
ALPHA0     = 1.0  # Beta(1,1) prior: α₀
BETA0      = 1.0  # Beta(1,1) prior: β₀
THRESHOLD  = 0.5  # boundary between low-partial and high-success
N_BAYESIAN = 4    # rollouts retained after Bayesian stabilisation


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
) -> Tuple[List[str], List[str], List[float], List[int]]:
    """Run AERO rollout curation for one training step.

    Returns
    -------
    curated_prompts  : prompts repeated per retained rollout
    curated_texts    : full (prompt + completion) strings
    curated_rewards  : float rewards in {0.0, 1.0}  (or Bayesian value for all-same)
    curated_group_ids: integer group id (index into prompts list)
    """
    from generation import generate_n
    from verifier import canonicalize_answer, verify_answer

    batch_size = len(prompts)

    # ── Stage I: Exploration ─────────────────────────────────────────────────
    groups_explore = generate_n(
        policy, tokenizer, prompts, N_EXPLORE,
        max_new_tokens, temperature, top_p, device,
    )

    # For each prompt: check correctness
    outcomes_explore: List[List[float]] = []   # 0.0 / 1.0 per rollout
    for i, outs in enumerate(groups_explore):
        row = []
        for out in outs:
            completion = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
            pred = canonicalize_answer(completion, dataset_name)
            r = float(verify_answer(completion, gold_answers[i], dataset_name))
            row.append(r)
        outcomes_explore.append(row)

    # ── Stage II: Differentiated Exploitation ────────────────────────────────
    all_prompts_out:   List[str]   = []
    all_texts_out:     List[str]   = []
    all_rewards_out:   List[float] = []
    all_group_ids_out: List[int]   = []

    for i in range(batch_size):
        outs_i   = groups_explore[i][:]
        rews_i   = outcomes_explore[i][:]

        c = sum(1 for r in rews_i if r > 0)
        u = c / max(len(rews_i), 1)

        # ── Strategy 1: Rescue for zero-accuracy prompts ──────────────────
        if u == 0.0:
            budget_used = len(outs_i)
            while budget_used < N_TOTAL:
                n_new = min(N_EXTRA, N_TOTAL - budget_used)
                extra = generate_n(
                    policy, tokenizer, [prompts[i]], n_new,
                    max_new_tokens, temperature, top_p, device,
                )[0]
                for out in extra:
                    comp = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
                    r = float(verify_answer(comp, gold_answers[i], dataset_name))
                    outs_i.append(out)
                    rews_i.append(r)
                    budget_used += 1
                    if r > 0:
                        break  # first success found → stop rescue
                else:
                    continue
                break
            c = sum(1 for r in rews_i if r > 0)
            u = c / len(rews_i)

        n_total_used = len(rews_i)

        if u == 0.0 or u == 1.0:
            # ── Strategy 2: Bayesian posterior ────────────────────────────
            u_tilde = (c + ALPHA0) / (n_total_used + ALPHA0 + BETA0)
            # Constant Bayesian advantage for all retained rollouts
            bayesian_adv = 2.0 * u_tilde - 1.0
            # Keep N_BAYESIAN rollouts: prefer mixed outcomes if any
            correct_idxs = [j for j, r in enumerate(rews_i) if r > 0]
            wrong_idxs   = [j for j, r in enumerate(rews_i) if r == 0]
            keep_idxs: List[int] = []
            # Fill balanced (up to N_BAYESIAN)
            half = N_BAYESIAN // 2
            keep_idxs += correct_idxs[:half]
            keep_idxs += wrong_idxs[:half]
            # Top up with remaining
            remaining = [j for j in range(len(outs_i)) if j not in keep_idxs]
            keep_idxs += remaining[:max(0, N_BAYESIAN - len(keep_idxs))]
            keep_idxs = keep_idxs[:N_BAYESIAN]

            for j in keep_idxs:
                all_prompts_out.append(prompts[i])
                all_texts_out.append(outs_i[j])
                # Use Bayesian advantage signal instead of raw 0/1
                # We encode it as a continuous reward; GRPO normalises later.
                all_rewards_out.append(u_tilde if rews_i[j] > 0 else (1.0 - u_tilde))
                all_group_ids_out.append(i)

        elif 0.0 < u < THRESHOLD:
            # ── Strategy 3: Rejection sampling (partial-low) ──────────────
            correct_idxs = [j for j, r in enumerate(rews_i) if r > 0]
            wrong_idxs   = [j for j, r in enumerate(rews_i) if r == 0]
            # Keep all correct; downsample incorrect to m = k * c
            m = max(1, K_RATIO * len(correct_idxs))
            random.shuffle(wrong_idxs)
            keep_wrong = wrong_idxs[:m]
            keep_idxs  = correct_idxs + keep_wrong

            for j in keep_idxs:
                all_prompts_out.append(prompts[i])
                all_texts_out.append(outs_i[j])
                all_rewards_out.append(rews_i[j])
                all_group_ids_out.append(i)

        else:
            # u >= THRESHOLD: keep all n_explore rollouts unchanged
            for j in range(len(outs_i)):
                all_prompts_out.append(prompts[i])
                all_texts_out.append(outs_i[j])
                all_rewards_out.append(rews_i[j])
                all_group_ids_out.append(i)

    return all_prompts_out, all_texts_out, all_rewards_out, all_group_ids_out
