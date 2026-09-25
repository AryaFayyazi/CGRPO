# C-GRPO for TRL

`CGRPOTrainer` is a drop-in subclass of TRL's `GRPOTrainer` that replaces the fixed group size with a per-prompt
sampling budget chosen by split-conformal prediction. Each prompt is sampled in increments along `budget_grid`
(default `2, 4, 8, 16, 32`) and stops at the first budget whose conformal prediction set over the answers drawn so far
is a singleton. Thresholds are calibrated on a held-out set with the current policy, before the first step and every
`recalibrate_every` steps.

This package is a standalone copy of `trl.experimental.cgrpo`, which is being proposed upstream, so it works with
released TRL (tested with 1.12 and TRL main).

## Install

```bash
pip install "git+https://github.com/AryaFayyazi/CGRPO.git#subdirectory=integrations"
```

or `pip install -e integrations/` from the root of a clone. Requires TRL >= 1.12.

## Usage

```python
import re

from datasets import load_dataset
from trl_cgrpo import CGRPOConfig, CGRPOTrainer


def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,\.]+)", text)
    return match.group(1).replace(",", "") if match else None


def correctness_reward(completions, answer, **kwargs):
    return [float(extract_answer(c) == a) for c, a in zip(completions, answer)]


def to_prompt(example):
    return {
        "prompt": example["question"] + "\nEnd with '#### <answer>'.",
        "answer": example["answer"].split("####")[-1].strip().replace(",", ""),
    }


data = load_dataset("openai/gsm8k", "main", split="train").map(to_prompt)
split = data.train_test_split(test_size=200, seed=0)  # calibration examples must not be trained on

trainer = CGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=correctness_reward,
    args=CGRPOConfig(budget_grid=[2, 4, 8, 16, 32], recalibrate_every=50),
    train_dataset=split["train"],
    calibration_dataset=split["test"],
    answer_extractor=extract_answer,
)
trainer.train()
```

For execution-verified tasks (code), set `score="first_success"` and pass
`verifier(completions=list_of_texts, **example) -> list[bool]` instead of `answer_extractor`.

## How it plugs into GRPO

`num_generations` is the group width and equals `max(budget_grid)`. A prompt that stops at a smaller budget is padded
up to that width with one-token completions. Padded rows get a NaN reward, so TRL's nan-aware group baseline ignores
them and their advantage is zero; they are also removed from the loss mask and the loss normalizer. Compute is saved in
generation: only the kept completions are ever generated.

Logged metrics: `cgrpo/mean_k`, `cgrpo/rollout_savings`, `cgrpo/delta`, `cgrpo/qhat_{k}`,
`cgrpo/calibration_solve_rate`, `cgrpo/calibration_rollouts`.

## Options

| Option | Effect |
|---|---|
| `delta=None` (default) | Sets δ at each calibration from the policy's solve rate at the largest budget. A fixed stringent δ on a weak policy pins every threshold at 1.0 and disables early stopping. |
| `split_delta=True` | Chooses δ and the thresholds on disjoint halves of the calibration set, restoring split-conformal exactness at half the effective calibration size. |
| `score="first_success"` | Execution score `j*/k` for code. |
| `score="pass_rate"` | `1 − n_pass/k`; discrete, so coverage is conservative rather than exact. |
| `recalibrate_every` | Steps between calibrations; `0` calibrates once. |

## Caveats

- **Calibration is not free.** Each calibration generates `num_calibration_samples × max(budget_grid)` completions,
  logged as `cgrpo/calibration_rollouts` and not included in `cgrpo/rollout_savings`.
- **vLLM.** `use_vllm=True` works in server and colocate mode; each budget increment is one generation request
  (at most `len(budget_grid)` per batch), and padded rows get no importance-sampling correction.
- **Loss types.** The token-normalized losses (`dapo`, the default, `bnpo`, `cispo`, `vespo`) are supported;
  padded rows are masked out of them exactly. `grpo`, `sapo`, `luspo` and `dr_grpo` raise `NotImplementedError`.
- **Scope.** Single process. Multi-process training, tools, environments, `rollout_func` and vision-language
  models raise `NotImplementedError`.
- Padded rows appear in TRL's completion-length statistics and completion tables as one-token completions.

## Tests

```bash
pytest integrations/trl_cgrpo/tests -q     # CPU, downloads tiny test models from the Hub
```

The conformal utilities are checked against the paper implementation (`conformal.py` in the repository root):
quantile, APS score and prediction set agree exactly on 5,000 random cases each.
