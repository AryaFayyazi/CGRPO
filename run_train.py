import argparse

# Gemma3 (transformers 5.x) requires torch>=2.6 for the vmap-based bidirectional
# image-block masking path (or_mask_function / TransformGetItemToIndex).
# For text-only training there are no image tokens, so that path is a no-op.
# We patch create_causal_mask_mapping to:
#   1) zero-out token_type_ids (passed as positional arg[6] — kwargs check misses it)
#   2) force is_training=False to suppress the "token_type_ids required" guard
# This makes all text-only Gemma3 forward passes skip the torch>=2.6 code entirely.
try:
    import transformers.models.gemma3.modeling_gemma3 as _g3mod
    _orig_create_causal_mask_mapping = _g3mod.create_causal_mask_mapping

    def _patched_create_causal_mask_mapping(*args, **kwargs):
        # token_type_ids is the 7th positional arg (index 6): always null it out
        # for text-only training (no image tokens → or_mask_function never needed)
        args = list(args)
        if len(args) > 6:
            args[6] = None  # token_type_ids positional
        kwargs.pop("token_type_ids", None)
        kwargs["is_training"] = False  # suppress the "token_type_ids required" guard
        return _orig_create_causal_mask_mapping(*args, **kwargs)

    _g3mod.create_causal_mask_mapping = _patched_create_causal_mask_mapping
except Exception:
    pass

from config import TrainConfig
from env_config import configure_hf_caches
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conformal GRPO with configurable settings")
    parser.add_argument("--model-key", help="registry key of base model to use")
    parser.add_argument("--dataset-name", help="huggingface dataset name")
    parser.add_argument("--dataset-config", help="dataset config string")
    parser.add_argument("--n-train", type=int, help="number of train examples")
    parser.add_argument("--n-cal", type=int, help="number of calibration examples")
    parser.add_argument("--n-eval", type=int, help="number of evaluation examples")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--log-dir", help="root directory for logging and checkpoints")
    parser.add_argument("--max-new-tokens", type=int,
                        help="max tokens generated per sample (default 256; use ~20 for classification)")
    parser.add_argument("--steps", type=int, help="number of training steps")
    parser.add_argument("--k-values", help="comma-separated conformal k values, e.g. 2,4,8,16,32")
    parser.add_argument("--delta", type=str, default=None,
                        help="conformal error rate: float (e.g. 0.1) or 'auto' to select from cal scores")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="training batch size (number of prompts per gradient step)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="in-loop evaluation cadence in steps (default from config.py)")
    parser.add_argument("--recalibrate-every", type=int, default=None,
                        help="conformal re-calibration cadence in steps (default from config.py)")
    parser.add_argument("--code-score", choices=["first_success", "pass_rate"],
                        default=None,
                        help="execution score for code datasets (default: first_success, paper Eq. 5)")
    parser.add_argument("--full-finetune", action="store_true",
                        help="train all weights instead of LoRA (needs >=2 GPUs at 7B)")
    parser.add_argument("--split-delta", action="store_true",
                        help="partition D_cal so delta_auto and q_hat use disjoint halves "
                             "(restores split-conformal exactness)")
    args = parser.parse_args()

    cfg = TrainConfig()
    # override config from CLI if provided
    if args.model_key:
        cfg.model_key = args.model_key
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.dataset_config:
        cfg.dataset_config = args.dataset_config
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_cal is not None:
        cfg.n_cal = args.n_cal
    if args.n_eval is not None:
        cfg.n_eval = args.n_eval
    if args.seed is not None:
        cfg.seed = args.seed
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.steps is not None:
        cfg.steps = args.steps
    if args.k_values is not None:
        cfg.k_values = tuple(int(x) for x in args.k_values.split(","))
    if args.delta is not None:
        if args.delta.strip().lower() == "auto":
            cfg.auto_delta = True
        else:
            cfg.auto_delta = False
            cfg.deltas = (float(args.delta),)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.recalibrate_every is not None:
        cfg.recalibrate_every = args.recalibrate_every
    if args.code_score is not None:
        cfg.code_score = args.code_score
    if args.full_finetune:
        cfg.full_finetune = True
    if args.split_delta:
        cfg.split_delta_calibration = True

    configure_hf_caches(cfg.hf_cache_root, cfg.datasets_root)
    train(cfg)
