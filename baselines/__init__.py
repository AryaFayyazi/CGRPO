# Baseline GRPO-variant methods for comparison with C-GRPO.
#
# Each module exposes a single function:
#   curate_rollouts(policy, tokenizer, prompts, gold_answers,
#                   dataset_name, cfg, device) -> (texts, rewards, group_ids, weights)
#
# All baselines use the same grpo_objective from grpo_loss.py but with
# different rollout selection / weighting strategies.
#
# Baseline inventory:
#   - aero          : AERO (2602.14338), adaptive explore+rescue+rejection-sampling
#   - gdro          : GDRO (2601.19280), EMA-debiased difficulty-bin advantage scaling
#   - reinforce_ada : Reinforce-Ada (2510.04996), sequential sampling until K_pos+K_neg found
#   - greso         : GRESO (NeurIPS 2025), pre-rollout skip of uninformative prompts
