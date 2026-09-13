import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainConfig:
    model_key: str = "qwen2.5-3b"  # Use lowercase key that matches registry
    hf_cache_root: str = "~/.cache/hf_ccr"
    datasets_root: str = os.environ.get(
        "CGRPO_DATASETS_ROOT",
        os.path.expanduser("~/.cache/huggingface/datasets"))

    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"

    n_train: int = 4000
    n_cal: int = 400
    n_eval: int = 400

    seed: int = 0

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    deltas: Tuple[float, ...] = (0.1,)
    auto_delta: bool = True           # auto-select δ from cal scores; overrides deltas[0]
    # When True, D_cal is split in half: one half selects δ_auto, the other
    # calibrates q̂. Restores split-conformal exactness (δ is then independent
    # of the scores that produce q̂) at the cost of n_cal/2 effective size.
    # Default False preserves the original behaviour of all existing runs.
    split_delta_calibration: bool = False
    # Execution-based nonconformity score for code datasets:
    #   "first_success" = j*/k  (paper Eq. 5, default)
    #   "pass_rate"     = 1 - n_pass/k
    code_score: str = "first_success"
    # Full-parameter fine-tuning (no LoRA). Needs a separate frozen
    # reference model, i.e. >=2 GPUs for a 7-8B policy.
    full_finetune: bool = False
    k_values: Tuple[int, ...] = (2, 4, 8, 16)

    lr: float = 2e-6
    weight_decay: float = 0.01
    beta_kl: float = 0.05
    grad_clip: float = 0.5
    batch_size: int = 4
    steps: int = 300

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    use_bf16: bool = True

    # ✅ Logging / eval / ckpt cadence
    log_dir: str = os.environ.get("CGRPO_LOG_DIR", "runs/conformal_grpo")
    log_every: int = 1
    eval_every: int = 50
    sample_every: int = 50
    ckpt_every: int = 100
    recalibrate_every: int = 50    # re-run conformal calibration with current policy (every 50 steps)

    # W&B / TB toggles
    use_wandb: bool = True
    wandb_project: str = "conformal-grpo-rlvr"
    use_tensorboard: bool = True


    max_seq_len: int = 512           # cap the total prompt+completion tokens for logprob compute
    logprob_microbatch: int = 1      # lower if still OOM (1 is safest)

