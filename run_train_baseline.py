"""
Entry-point for baseline GRPO-variant training.

Usage:
  python run_train_baseline.py \
      --method        greso \
      --model-key     QWEN2.5-7b \
      --dataset-name  openai/gsm8k \
      --dataset-config main \
      --n-train       400 \
      --n-cal         200 \
      --n-eval        200 \
      --seed          0 \
      --log-dir       runs/baseline/greso \
      --max-new-tokens 512 \
      --steps         300 \
      --n-rollouts    4

Supported methods:
  grpo           – vanilla GRPO (fixed n rollouts)
  aero           – AERO three-stage rollout curation (arxiv 2602.14338)
  gdro           – Prompt-GDRO + Rollout-GDRO (arxiv 2601.19280)
  gdro_prompt    – Prompt-GDRO only (advantage scaling, fixed n)
  reinforce_ada  – Reinforce-Ada-Seq-Balance (arxiv 2510.04996)
  reinforce_est  – Reinforce-Ada-Est EMA variant
  greso          – GRESO selective rollouts (NeurIPS 2025)
"""
import argparse
from config import TrainConfig
from env_config import configure_hf_caches
from train_baseline import train_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline GRPO variant trainer")

    # Method selection
    parser.add_argument(
        "--method", default="grpo",
        choices=["grpo", "aero", "gdro", "gdro_prompt",
                 "reinforce_ada", "reinforce_est", "greso"],
        help="Rollout curation method (default: grpo)",
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=4,
        help="Base rollouts per prompt for grpo / greso / reinforce_est (default: 4)",
    )

    # Standard TrainConfig overrides (mirrors run_train.py)
    parser.add_argument("--model-key",       help="Registry key of base model")
    parser.add_argument("--dataset-name",    help="HuggingFace dataset name")
    parser.add_argument("--dataset-config",  help="Dataset config/subset name")
    parser.add_argument("--n-train",   type=int, help="Training examples")
    parser.add_argument("--n-cal",     type=int, help="Calibration examples")
    parser.add_argument("--n-eval",    type=int, help="Evaluation examples")
    parser.add_argument("--seed",      type=int, help="Random seed")
    parser.add_argument("--log-dir",           help="Root directory for logs/checkpoints")
    parser.add_argument("--max-new-tokens", type=int, help="Max tokens per generation")
    parser.add_argument("--steps",     type=int, help="Training steps")
    parser.add_argument("--lr",        type=float, help="Learning rate")
    parser.add_argument("--beta-kl",   type=float, help="KL penalty coefficient")
    parser.add_argument("--batch-size",type=int,   help="Prompts per step")
    parser.add_argument("--lora-r",    type=int,   help="LoRA rank")
    parser.add_argument("--eval-every",type=int,   help="Evaluation frequency (steps)")
    parser.add_argument("--ckpt-every",type=int,   help="Checkpoint frequency (steps)")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.use_wandb        = False   # baselines: no wandb by default
    cfg.use_tensorboard  = True

    # Apply overrides
    if args.model_key:       cfg.model_key        = args.model_key
    if args.dataset_name:    cfg.dataset_name     = args.dataset_name
    if args.dataset_config:  cfg.dataset_config   = args.dataset_config
    if args.n_train:         cfg.n_train          = args.n_train
    if args.n_cal:           cfg.n_cal            = args.n_cal
    if args.n_eval:          cfg.n_eval           = args.n_eval
    if args.seed is not None:cfg.seed             = args.seed
    if args.log_dir:         cfg.log_dir          = args.log_dir
    if args.max_new_tokens:  cfg.max_new_tokens   = args.max_new_tokens
    if args.steps:           cfg.steps            = args.steps
    if args.lr:              cfg.lr               = args.lr
    if args.beta_kl:         cfg.beta_kl          = args.beta_kl
    if args.batch_size:      cfg.batch_size       = args.batch_size
    if args.lora_r:          cfg.lora_r           = args.lora_r
    if args.eval_every:      cfg.eval_every       = args.eval_every
    if args.ckpt_every:      cfg.ckpt_every       = args.ckpt_every

    configure_hf_caches(cfg.hf_cache_root, cfg.datasets_root)
    train_baseline(cfg, method=args.method, n_rollouts=args.n_rollouts)
