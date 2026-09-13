"""
Conformal prediction for GRPO with Adaptive Prediction Sets (APS).

The key insight: use CUMULATIVE PROBABILITY as the nonconformity score.
This produces a continuous score in [0, 1] instead of the coarse
discrete score (1 - freq/k) which collapses to {0, 0.5, 1} for k=2.

APS method (Romano, Sesia, Candès 2020):
  1. Compute empirical probabilities: p(a) = freq(a) / k
  2. Sort candidates by decreasing probability
  3. Score = cumulative probability needed to reach the true answer
     (with uniform random tie-breaking for exact coverage)
  4. Prediction set = include answers in decreasing probability order
     until cumulative probability exceeds qhat

This gives smooth, informative qhat values even when the model is weak.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import random

# Dedicated, seedable RNG for the APS randomisation and tie-breaking.
# Using the global `random` module made scores depend on unrelated calls
# elsewhere in the process, so runs were not reproducible.
_RNG = random.Random(0)


def set_conformal_seed(seed: int) -> None:
    """Reseed the APS randomisation (call once per run for reproducibility)."""
    global _RNG
    _RNG = random.Random(seed)


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """
    Split conformal quantile with finite-sample correction:
    q = ceil((n+1)*(1-delta))-th order statistic out of n scores.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def select_delta_auto(
    scores_at_k_max: np.ndarray,
    safety_margin: float = 0.05,
    delta_min: float = 0.05,
    delta_max: float = 0.95,
) -> Tuple[float, float]:
    """
    Automatically choose conformal error rate δ from calibration scores at k_max.

    Principle: coverage target = pass@k_max − safety_margin.

    pass@k_max is estimated as the fraction of calibration examples where the
    model produced at least one correct answer within k_max rollouts (i.e.,
    nonconformity score < 1.0).

        δ = (1 − pass@k_max) + safety_margin

    This makes the coverage guarantee tight to what the model can actually
    achieve on this particular dataset, with no manual tuning.  The
    safety_margin ensures qhat sits meaningfully below 1.0 and is not right at
    the boundary where finite-sample variance could push it back to 1.0.

    Args:
        scores_at_k_max: array of nonconformity scores at k=k_max, one per cal
            example.  score=1.0 means the model never produced a correct answer
            in k_max rollouts (unsolvable given this budget).
        safety_margin: buffer added on top of the unsolvable fraction.
            Default 0.05 (5 pp) keeps qhat comfortably below 1.0.
        delta_min: minimum allowed δ.  0.05 → max claimed coverage is 95%.
            Guards against over-claiming when the model is very strong.
        delta_max: maximum allowed δ.  0.95 → minimum claimed coverage is 5%.
            Previous cap of 0.50 forced qhat=1.0 for any base model with
            pass@k_max < 45% (e.g. all code models at init), completely
            disabling adaptive k.  0.95 keeps the guarantee meaningful while
            correctly handling weak base models such as pre-training on code.

    Returns:
        (delta_selected, frac_solvable) for use and for logging/diagnostics.
    """
    scores = np.asarray(scores_at_k_max, dtype=float)
    n_total = max(len(scores), 1)
    n_solvable = int((scores < 1.0 - 1e-9).sum())
    frac_solvable = n_solvable / n_total
    # δ = (fraction of unsolvable problems) + safety margin
    delta = (1.0 - frac_solvable) + safety_margin
    selected = float(np.clip(delta, delta_min, delta_max))
    return selected, frac_solvable


def answer_freqs(extracted_answers: List[str]) -> Counter:
    """Count non-empty answers."""
    return Counter([a for a in extracted_answers if a != ""])


def _empirical_probs(freqs: Counter, k: int) -> List[Tuple[str, float]]:
    """
    Convert frequency counts to empirical probabilities, sorted descending.
    Assigns residual probability uniformly to a catch-all "unseen" category.
    Returns list of (answer, probability) sorted by probability descending.
    """
    if k <= 0:
        return []
    total_counted = sum(freqs.values())
    # Probability for answers that appeared
    probs = [(ans, count / k) for ans, count in freqs.items()]
    # Ties are broken UNIFORMLY AT RANDOM, not alphabetically. Deterministic
    # alphabetical ordering systematically favours lexicographically smaller
    # answers, which breaks the exchangeability that exact marginal coverage
    # relies on (equal-frequency answers must be interchangeable).
    probs.sort(key=lambda x: (-x[1], _RNG.random()))
    return probs


def score_true_answer(true_answer: str, freqs: Counter, k: int,
                      match_fn=None) -> float:
    """
    APS nonconformity score: cumulative probability until we include
    the true answer, when candidates are sorted by decreasing frequency.

    Score ∈ [0, 1]:
      - 0 means true answer is the single most probable (covers everything)
      - 1 means true answer never appeared in k samples
      - Intermediate values are CONTINUOUS: they equal the cumulative
        probability mass of all answers more popular than the true answer,
        plus a randomized fraction of the true answer's own mass
        (for exact marginal coverage).

    This is much more informative than the binary 1-freq/k score.

    match_fn: optional callable(generated: str, gold: str) -> bool
      If provided, used instead of exact equality to identify the best
      matching generated answer (e.g. token-F1 for abstractive QA).
    """
    if true_answer == "" or k <= 0:
        return 1.0

    probs = _empirical_probs(freqs, k)

    # Identify the effective "true" answer key in freqs.
    # For exact matching (default): the true_answer string itself.
    # For soft matching (match_fn provided): the generated answer that
    # best matches the gold, reusing its probability mass.
    effective_answer = true_answer
    if match_fn is not None and true_answer not in freqs:
        # Find any generated answer accepted by match_fn
        for ans, _ in probs:
            if match_fn(ans, true_answer):
                effective_answer = ans
                break

    # If effective answer never appeared, score = 1.0
    true_prob = 0.0
    for ans, p in probs:
        if ans == effective_answer:
            true_prob = p
            break

    if true_prob == 0.0:
        return 1.0

    # Cumulative probability of answers strictly more popular than true
    cum_prob = 0.0
    for ans, p in probs:
        if ans == effective_answer:
            break
        cum_prob += p

    # Randomized tie-breaking: add U * p(true) for exact coverage
    # U ~ Uniform(0,1) makes the score continuous
    u = _RNG.random()
    score = cum_prob + u * true_prob

    return min(score, 1.0)


def conformal_set_from_freqs(freqs: Counter, k: int, qhat: float) -> List[str]:
    """
    APS prediction set: include answers in decreasing probability order
    until the cumulative probability exceeds qhat.

    This naturally produces small sets when one answer dominates (high
    confidence) and large sets when answers are spread out (low confidence).
    """
    if k <= 0:
        return []

    probs = _empirical_probs(freqs, k)
    if not probs:
        return []

    out = []
    cum_prob = 0.0
    for ans, p in probs:
        out.append(ans)
        cum_prob += p
        if cum_prob >= qhat:
            break

    # If qhat is very large (close to 1) and we exhausted all answers
    # but haven't reached qhat, include everything we have
    return sorted(out)


def calibrate_qhats(
    cal_true_answers: List[str],
    cal_group_answers: List[List[str]],
    k_values: Tuple[int, ...],
    delta: float,
    match_fn=None,
    score_fn=None,
) -> Dict[int, float]:
    """
    Calibrate conformal quantiles using APS scores.

    For each k, compute the APS nonconformity score for each calibration
    example, then take the (1-delta) quantile with finite-sample correction.

    match_fn: optional callable(generated: str, gold: str) -> bool
      Forwarded to score_true_answer for soft (non-exact) answer matching.

    score_fn: optional callable(gold: str, gens: List[str]) -> float
      If provided, replaces the APS score entirely.  Intended for abstractive
      QA where APS (which requires the gold to appear in the generated set)
      cannot work.  Signature: score_fn(gold, group_k) -> float in [0, 1].
      Example: ``lambda gold, gens: 1 - max_precision(gens, gold)``.

    Prints diagnostics showing the score distribution.
    """
    qhats = {}
    for k in k_values:
        scores = []
        for t, group in zip(cal_true_answers, cal_group_answers):
            group_k = group[:k]
            if score_fn is not None:
                scores.append(score_fn(t, group_k))
            else:
                freqs = answer_freqs(group_k)
                scores.append(score_true_answer(t, freqs, k, match_fn=match_fn))

        scores_arr = np.array(scores)
        qhat = conformal_quantile(scores_arr, delta)
        qhats[k] = qhat

        # Diagnostics
        n_total = len(scores_arr)
        n_missing = int((scores_arr >= 1.0 - 1e-9).sum())
        print(f"  [Conformal k={k}] qhat={qhat:.4f} | "
              f"true_missing={n_missing/max(n_total,1):.1%} ({n_missing}/{n_total}) | "
              f"scores: min={scores_arr.min():.3f} p25={np.percentile(scores_arr,25):.3f} "
              f"p50={np.median(scores_arr):.3f} p75={np.percentile(scores_arr,75):.3f} "
              f"p90={np.percentile(scores_arr,90):.3f} max={scores_arr.max():.3f}")

    return qhats
