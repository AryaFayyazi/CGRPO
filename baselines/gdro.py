"""
GDRO: Group Distributionally Robust Optimization for LLM Reasoning
Paper: arxiv.org/abs/2601.19280  (Tencent AI Lab)

Implements TWO independent adversarial mechanisms on top of standard GRPO:

── Prompt-GDRO (Data Adversary) ────────────────────────────────────────────
  • Partitions prompts into B=10 difficulty bins based on online pass@k (EMA).
  • Maintains an EMA-debiased score S(b) per bin tracking mean GRPO loss.
  • Scales GRPO advantages by adversarial bin weight ω(b) = exp(η_q * S(b)).
  • Concentrates gradient updates on persistently hard bins.
  Hyperparams: B=10 bins, η_q=0.65, γ=0.01, β_ema=0.12, ω_max=15.0

── Rollout-GDRO (Compute Adversary) ────────────────────────────────────────
  • Allocates variable rollout counts n_b to bins to minimise gradient variance.
  • Uses shadow-price (dual) controller: n_b selected from {n_min,...,n_max}
    with soft-min allocation proportional to sqrt(variance_b) under fixed mean.
  • Shadow price μ updated: μ ← μ + α_μ * (n̄_realized - n̄)
  Hyperparams: n_min=2, n_max=12, n̄=4, α_μ=0.05

Both mechanisms are compute-neutral: same average rollouts as baseline GRPO.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
N_BINS       = 10      # difficulty bins (edges at 0.1, 0.2, …, 0.9)
ETA_Q        = 0.65    # adversary learning rate for Prompt-GDRO
GAMMA_EXPL   = 0.01    # exploration rate (uniform mixing)
BETA_EMA     = 0.12    # EMA decay for difficulty scores
OMEGA_MAX    = 15.0    # cap on bin weight
N_MIN        = 2       # minimum rollouts per prompt (Rollout-GDRO)
N_MAX        = 12      # maximum rollouts per prompt (Rollout-GDRO)
N_BAR        = 4       # target mean rollouts (compute-neutral budget)
ALPHA_MU     = 0.05    # dual learning rate for shadow price
SCORE_CLIP   = 3.0     # score clipping bound C


class GDROState:
    """Persistent state for the GDRO adversaries (lives across training steps)."""

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        # Prompt-GDRO: EMA difficulty scores per bin
        self.scores: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        # Rollout-GDRO: shadow price (dual variable)
        self.mu: float = 0.0
        # Per-prompt online pass@k EMA: {prompt_uid: (ema_total, ema_pos)}
        self.pass_rate_ema: Dict[str, Tuple[float, float]] = {}
        self._lambda = 0.85  # EMA decay for pass-rate tracking

    def prompt_uid(self, prompt: str) -> str:
        """Stable 64-bit hash of prompt string as dictionary key."""
        return str(hash(prompt) % (2**32))

    def get_pass_rate(self, prompt: str) -> float:
        """Return smoothed pass-rate estimate for `prompt` (default 0.5)."""
        uid = self.prompt_uid(prompt)
        if uid not in self.pass_rate_ema:
            return 0.5
        total, pos = self.pass_rate_ema[uid]
        if total < 0.1:
            return 0.5
        return float(np.clip(pos / total, 0.0, 1.0))

    def update_pass_rate(self, prompt: str, n_tried: int, n_correct: int) -> None:
        """Update EMA pass-rate for a prompt after a training step."""
        uid = self.prompt_uid(prompt)
        lam = self._lambda
        t, p = self.pass_rate_ema.get(uid, (0.0, 0.0))
        self.pass_rate_ema[uid] = (
            lam * t + n_tried,
            lam * p + n_correct,
        )

    def prompt_bin(self, pass_rate: float) -> int:
        """Map pass-rate ∈ [0,1] to a bin index 0..n_bins-1."""
        # Bin edges at 0.1, 0.2, …, 0.9 → 10 bins
        b = int(pass_rate * self.n_bins)
        return min(max(b, 0), self.n_bins - 1)

    def bin_weights(self) -> np.ndarray:
        """Return adversarial bin weights ω(b) = exp(η_q * clip(S(b), -C, C))."""
        clipped = np.clip(self.scores, -SCORE_CLIP, SCORE_CLIP)
        weights = np.exp(ETA_Q * clipped)
        return weights  # shape (n_bins,)

    def sampling_probs(self) -> np.ndarray:
        """q_t(b) = (1-γ) * ω(b)/Σω + γ/B  (with exploration)."""
        w = self.bin_weights()
        w_norm = w / (w.sum() + 1e-8)
        return (1.0 - GAMMA_EXPL) * w_norm + GAMMA_EXPL / self.n_bins

    def update_scores(self, bin_losses: Dict[int, float]) -> None:
        """EMA update for difficulty scores.

        bin_losses: dict   { bin_idx: mean_grpo_loss_for_that_bin }
        """
        for b in range(self.n_bins):
            if b in bin_losses:
                l_bar = bin_losses[b]
                self.scores[b] = (1 - BETA_EMA) * self.scores[b] + BETA_EMA * l_bar

    def rollout_allocation(self, bins: List[int], reward_variances: Dict[int, float]) -> List[int]:
        """Rollout-GDRO: allocate discrete rollout counts under fixed mean budget.

        Parameters
        ----------
        bins            : list of bin ids for each prompt in the batch
        reward_variances: dict {bin_id: empirical reward variance}

        Returns list of rollout counts n_i for each prompt (same length as bins).
        """
        unique_bins = list(set(bins))
        # Compute n_b^* ∝ sqrt(v_b) (variance-optimal allocation)
        sqrt_var = {b: math.sqrt(max(reward_variances.get(b, 0.25), 1e-6))
                    for b in unique_bins}
        sum_sqrt = sum(sqrt_var.values()) + 1e-8

        # Use shadow price to select discrete arm from {n_min,...,n_max}
        # Optimal continuous: n_b = n_bar * sqrt(v_b) / sum_j{q_j * sqrt(v_j)}
        n_alloc: Dict[int, int] = {}
        bin_counts = defaultdict(int)
        for b in bins:
            bin_counts[b] += 1

        total_realized = 0
        for b in unique_bins:
            frac = sqrt_var[b] / sum_sqrt
            n_cont = N_BAR * frac / (sum(
                bin_counts[bb] / len(bins) for bb in unique_bins) + 1e-8)
            # Clamp and round
            n_b = int(round(np.clip(n_cont, N_MIN, N_MAX)))
            n_alloc[b] = n_b
            total_realized += n_b * bin_counts[b]

        # Update shadow price
        n_realized_avg = total_realized / max(len(bins), 1)
        self.mu = max(0.0, self.mu + ALPHA_MU * (n_realized_avg - N_BAR))

        return [n_alloc.get(b, N_BAR) for b in bins]


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
    state: Optional[GDROState] = None,
    use_rollout_gdro: bool = True,
) -> Tuple[List[str], List[str], List[float], List[int], List[float]]:
    """Run GDRO rollout curation + advantage scaling for one training step.

    Returns
    -------
    curated_prompts   : prompts repeated for each retained rollout
    curated_texts     : full (prompt + completion) strings
    curated_rewards   : float rewards
    curated_group_ids : integer group id per rollout
    advantage_weights : per-rollout multiplicative advantage weight (Prompt-GDRO ω)
    state             : updated GDROState (same object, mutated in place)
    """
    from generation import generate_n
    from verifier import verify_answer

    if state is None:
        state = GDROState()

    batch_size = len(prompts)

    # ── Determine per-prompt bins using current pass-rate estimate ─────────
    pass_rates = [state.get_pass_rate(p) for p in prompts]
    bins       = [state.prompt_bin(pr) for pr in pass_rates]

    # ── Rollout-GDRO: allocate variable rollout counts per bin ─────────────
    bin_variances: Dict[int, float] = {}
    for b in set(bins):
        # Use pass-rate variance proxy: p*(1-p) per bin
        pr_list = [pass_rates[i] for i, bb in enumerate(bins) if bb == b]
        bin_variances[b] = float(np.mean([p * (1 - p) for p in pr_list]))

    if use_rollout_gdro:
        rollout_counts = state.rollout_allocation(bins, bin_variances)
    else:
        rollout_counts = [N_BAR] * batch_size  # Prompt-GDRO only

    # ── Generate rollouts (variable count per prompt) ──────────────────────
    all_prompts_out:    List[str]   = []
    all_texts_out:      List[str]   = []
    all_rewards_out:    List[float] = []
    all_group_ids_out:  List[int]   = []
    all_weights_out:    List[float] = []

    bin_loss_accum: Dict[int, List[float]] = defaultdict(list)

    # Compute Prompt-GDRO weights
    omega = state.bin_weights()
    omega_for_prompt = [min(float(omega[b]), OMEGA_MAX) for b in bins]

    for i in range(batch_size):
        n_i = rollout_counts[i]
        outs_i = generate_n(
            policy, tokenizer, [prompts[i]], n_i,
            max_new_tokens, temperature, top_p, device,
        )[0]

        correct_count = 0
        for out in outs_i:
            comp = out[len(prompts[i]):] if out.startswith(prompts[i]) else out
            r = float(verify_answer(comp, gold_answers[i], dataset_name))
            all_prompts_out.append(prompts[i])
            all_texts_out.append(out)
            all_rewards_out.append(r)
            all_group_ids_out.append(i)
            all_weights_out.append(omega_for_prompt[i])   # Prompt-GDRO weight
            if r > 0:
                correct_count += 1

            # Accumulate a proxy loss for bin-score update:
            # loss ≈ 1 - reward (higher loss = harder prompt)
            bin_loss_accum[bins[i]].append(1.0 - r)

        # Update pass-rate EMA for this prompt
        state.update_pass_rate(prompts[i], n_i, correct_count)

    # ── Update Prompt-GDRO scores with mean loss per bin ──────────────────
    bin_mean_losses = {b: float(np.mean(v)) for b, v in bin_loss_accum.items() if v}
    state.update_scores(bin_mean_losses)

    return all_prompts_out, all_texts_out, all_rewards_out, all_group_ids_out, all_weights_out
