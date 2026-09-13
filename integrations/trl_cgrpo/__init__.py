"""C-GRPO: conformal adaptive rollout budgets for TRL's GRPOTrainer."""

from .cgrpo_config import CGRPOConfig
from .cgrpo_trainer import CGRPOTrainer
from .conformal import (
    aps_score,
    conformal_quantile,
    first_success_score,
    pass_rate_score,
    prediction_set,
    select_delta_auto,
)


__version__ = "0.2.0"
__all__ = [
    "CGRPOConfig",
    "CGRPOTrainer",
    "aps_score",
    "conformal_quantile",
    "first_success_score",
    "pass_rate_score",
    "prediction_set",
    "select_delta_auto",
]
