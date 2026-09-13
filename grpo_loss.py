import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

@torch.no_grad()
def _prompt_lens(tokenizer, prompts: List[str], device) -> torch.Tensor:
    lens = [tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts]
    return torch.tensor(lens, device=device, dtype=torch.long)

def _find_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """
    Return token length of prompt as a prefix of full_ids if it matches.
    If it doesn't match, try to find the longest common prefix.
    """
    L = len(prompt_ids)
    if L == 0:
        return 0
    
    # Check if full_ids starts with prompt_ids
    if len(full_ids) >= L and full_ids[:L] == prompt_ids:
        return L
    
    # Try to find longest common prefix
    for i in range(min(L, len(full_ids)), 0, -1):
        if full_ids[:i] == prompt_ids[:i]:
            return i
    
    # Last resort: return expected length
    return L


def completion_logprob(
    model,
    tokenizer,
    prompts: List[str],
    full_texts: List[str],
    device,
    microbatch: int = 1,
    max_length: Optional[int] = None,
    compute_grads: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Memory-safe completion logprob (robust to prompt/full_text token mismatch):
    - Processes sequences in microbatches.
    - Uses cross_entropy (no log_softmax tensor retained).
    - Sums logprobs ONLY over completion tokens, where completion-start is found
      by token-prefix matching prompt_ids within full_ids.
    Returns:
        sum_logprobs: (N,) sum of logprob per sequence over completion tokens.
        per_token_logprobs: list of N 1-D tensors, each containing the per-token
            logprob for that sequence's completion tokens. Used for per-token KL.
    """
    assert len(prompts) == len(full_texts), "prompts/full_texts must align"
    N = len(full_texts)
    if N == 0:
        return torch.empty((0,), device=device), []

    # Tokenize prompt/full_text consistently
    prompt_ids_list = [
        tokenizer(p, add_special_tokens=True).input_ids for p in prompts
    ]
    full_ids_list = [
        tokenizer(t, add_special_tokens=True).input_ids for t in full_texts
    ]

    out_logps = []
    out_per_token = []

    for start in range(0, N, microbatch):
        end = min(start + microbatch, N)

        # Compute prompt prefix lengths in token space for this microbatch
        prefix_lens = []
        for p_ids, f_ids in zip(prompt_ids_list[start:end], full_ids_list[start:end]):
            prefix_lens.append(_find_prefix_len(p_ids, f_ids))
        prefix_lens_t = torch.tensor(prefix_lens, device=device, dtype=torch.long)

        texts_mb = full_texts[start:end]
        enc = tokenizer(
            texts_mb,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(device)

        input_ids = enc.input_ids          # (B, T)
        attn = enc.attention_mask          # (B, T)

        if compute_grads:
            outputs = model(input_ids=input_ids, attention_mask=attn)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits            # (B, T, V)
        del outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()    # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
        shift_attn   = attn[:, 1:].contiguous()          # (B, T-1)
        
        del logits, input_ids, attn

        B, Tm1 = shift_labels.shape

        # Token-wise NLL (no reduction): shape (B, T-1)
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(B, Tm1)
        
        del shift_logits

        token_logp = (-nll) * shift_attn  # mask out padding tokens
        del nll, shift_attn

        # Build completion mask
        comp_mask = torch.zeros_like(token_logp)
        for i in range(B):
            s = int(prefix_lens_t[i].item()) - 1
            if s < 0:
                s = 0
            if s >= Tm1:
                continue
            comp_mask[i, s:] = 1.0

        token_logp_masked = token_logp * comp_mask

        # SUM of logprobs over completion tokens (standard for GRPO)
        out_logps.append(token_logp_masked.sum(dim=1))  # (B,)

        # Collect per-token logprobs for per-token KL computation
        for i in range(B):
            s = int(prefix_lens_t[i].item()) - 1
            if s < 0:
                s = 0
            if s < Tm1:
                # Count actual completion tokens (non-padding)
                n_comp = int(comp_mask[i, s:].sum().item())
                if n_comp > 0:
                    out_per_token.append(token_logp[i, s:s + n_comp])
                else:
                    out_per_token.append(torch.zeros(1, device=device))
            else:
                out_per_token.append(torch.zeros(1, device=device))

        del shift_labels, comp_mask, token_logp, token_logp_masked, prefix_lens_t

    sum_logps = torch.cat(out_logps, dim=0)
    return sum_logps, out_per_token


def per_token_kl(
    per_tok_pi: List[torch.Tensor],
    per_tok_ref: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute average per-token KL divergence KL(pi || ref) using Schulman's 
    non-negative approximation:
        kl_token = exp(logp_ref - logp_pi) - (logp_ref - logp_pi) - 1

    This is computed per-token and averaged across all completion tokens,
    avoiding the numerical overflow that occurs when exponentiating summed 
    log-ratios.

    Returns: scalar mean KL across all completion tokens.
    """
    all_kl = []
    for lp_pi, lp_ref in zip(per_tok_pi, per_tok_ref):
        min_len = min(lp_pi.shape[0], lp_ref.shape[0])
        if min_len == 0:
            continue
        pi_t = lp_pi[:min_len]
        ref_t = lp_ref[:min_len]
        # per-token: logp_ref - logp_pi
        diff = (ref_t - pi_t).clamp(-10.0, 10.0)
        kl_tokens = diff.exp() - diff - 1.0  # always >= 0
        all_kl.append(kl_tokens)
    
    if not all_kl:
        device = per_tok_pi[0].device if per_tok_pi else 'cpu'
        return torch.tensor(0.0, device=device)
    
    all_kl_cat = torch.cat(all_kl, dim=0)
    return all_kl_cat.mean()


def grpo_objective(
    logp_pi: torch.Tensor,
    logp_ref: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    beta_kl: float,
    per_tok_pi: Optional[List[torch.Tensor]] = None,
    per_tok_ref: Optional[List[torch.Tensor]] = None,
    adv_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    GRPO-style objective with group-wise baselines (no critic).
    
    Uses the GRPO formulation from DeepSeek-R1:
      loss = -E[ adv * logp_pi ]  +  beta * mean_per_token_KL(pi || ref)
    
    Key design choices:
      - Policy gradient uses REINFORCE with sum-logprobs (on-policy, no ratio needed)
      - KL is computed PER-TOKEN then averaged (avoids exp overflow from sum logratios)
      - KL flows gradients through logp_pi so beta_kl actually regularizes

    adv_weights : optional per-rollout multiplicative weight applied to advantages
        after normalization.  Used by GDRO (ω-scaling) and Reinforce-Ada (1/p̂).
    
    Returns: (scalar loss to MINIMIZE, float kl_value for logging)
    """
    assert logp_pi.shape == logp_ref.shape == rewards.shape
    assert group_ids.shape == rewards.shape

    # Compute per-group baseline b_g = mean reward for that prompt group
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    if num_groups == 0:
        return torch.tensor(0.0, device=logp_pi.device), 0.0

    sums = torch.zeros((num_groups,), device=rewards.device, dtype=torch.float32)
    cnts = torch.zeros((num_groups,), device=rewards.device, dtype=torch.float32)
    sums.scatter_add_(0, group_ids, rewards.float())
    cnts.scatter_add_(0, group_ids, torch.ones_like(rewards, dtype=torch.float32))
    means = sums / (cnts + eps)

    adv = rewards - means[group_ids]
    
    # Normalize advantages for stability (per-batch)
    if adv.numel() > 1 and adv.std() > eps:
        adv = (adv - adv.mean()) / (adv.std() + eps)

    # Optional per-rollout weight (GDRO advantage scaling, Reinforce-Ada 1/p̂)
    if adv_weights is not None:
        w = adv_weights.to(adv.device)
        # Normalise weights so mean=1 (preserves loss magnitude)
        w = w / (w.mean() + eps)
        adv = adv * w

    # ---- Policy gradient (REINFORCE-style with sum logprobs) ----
    # On-policy GRPO: loss_pg = -E[ adv * logp_pi ]
    # No importance ratio needed since we generate from pi each step
    policy_loss = -(adv.detach() * logp_pi).mean()

    # ---- KL penalty: per-token to avoid overflow ----
    if per_tok_pi is not None and per_tok_ref is not None and len(per_tok_pi) > 0:
        kl_est = per_token_kl(per_tok_pi, per_tok_ref)
    else:
        # Fallback: sequence-level approximation (less stable for large completions)
        log_ratio = (logp_pi - logp_ref).clamp(-20.0, 20.0)
        kl_est = ((-log_ratio).exp() - (-log_ratio) - 1.0).mean()

    kl_value = float(kl_est.detach().item())
    
    loss = policy_loss + beta_kl * kl_est
    return loss, kl_value
