# C-GRPO: Conformal Group Relative Policy Optimization

Reference implementation of **C-GRPO**, which replaces the fixed rollout group size `G` in
GRPO with a per-prompt adaptive budget driven by split-conformal prediction.

Standard GRPO samples the same `G` completions for every prompt: wasteful on easy prompts
(zero-variance groups produce no gradient) and too small on hard ones (no positive reward to
learn from). C-GRPO calibrates a threshold `q̂_k` for each budget level `k` on a held-out
calibration split, then stops sampling at the smallest `k` whose conformal prediction set is a
singleton.

## Method components

| Component | Where | What it does |
|---|---|---|
| APS nonconformity score | [`conformal.py`](conformal.py) | closed-form math/science answers; randomized APS score over the empirical answer histogram |
| Execution score for code | [`train.py`](train.py) `calibrate_conformal` | first-success `j*/k` (paper Eq. 5, default); `--code-score pass_rate` gives `1 - n_pass/k`. APS is vacuous when every completion is a unique string |
| Split-conformal quantile | [`conformal.py`](conformal.py) `calibrate_qhats` | finite-sample `⌈(n+1)(1-δ)⌉` order statistic |
| Auto-δ selection | [`conformal.py`](conformal.py) `select_delta_auto` | sets δ from the policy's own solve rate at `K_max` |
| Adaptive group construction | [`train.py`](train.py) `choose_k_and_sets` | smallest `k ∈ K` with a singleton prediction set |
| Periodic re-calibration | [`train.py`](train.py) | re-runs calibration with the current policy every `Δ_cal` steps |
| Split-δ calibration | [`train.py`](train.py), `--split-delta` | partitions `D_cal` so δ and `q̂` use disjoint halves (see *Calibration caveat*) |

## Install

```bash
conda create -n cgrpo python=3.10 && conda activate cgrpo
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Model paths are resolved through [`model_registry.py`](model_registry.py). Point
`MODEL_REGISTRY` at your local snapshots, or set `CGRPO_MODEL_ROOT`.

## Quickstart

Train C-GRPO on GSM8K with an adaptive budget grid:

```bash
python run_train.py \
  --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main \
  --n-train 400 --n-cal 200 --n-eval 400 \
  --max-new-tokens 512 --steps 300 \
  --k-values "2,4,8,16,32" --delta auto \
  --batch-size 4 --eval-every 50 --recalibrate-every 50 \
  --log-dir runs/cgrpo_gsm8k
```

Train a fixed-`G` GRPO baseline (same pipeline, only the allocation rule differs):

```bash
python run_train_baseline.py \
  --method grpo --n-rollouts 16 \
  --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main \
  --n-train 400 --n-eval 200 --max-new-tokens 512 --steps 300 \
  --batch-size 4 --log-dir runs/grpo_G16
```

`--method` also accepts `aero`, `gdro`, `gdro_prompt`, `reinforce_ada`, `reinforce_est`,
`greso` (implementations in [`baselines/`](baselines/)).

Evaluate a checkpoint across fixed budgets and the conformal stopping rule:

```bash
python eval_pareto.py \
  --ckpt-dir runs/cgrpo_gsm8k/<run>/final \
  --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main \
  --n-cal 200 --n-eval 400 --k-max 32 --batch-size 4 --max-new-tokens 512
```

This writes `pareto_results_<dataset>_lora.json` (or `_base` with `--no-lora`) containing
accuracy at each fixed `k`, `ave@k` (mean single-rollout accuracy, the paper's `p@1`
definition), `greedy` at `k=1`, the conformal dynamic-`k` row, and `theorem2_coverage_per_k`.

## Repository layout

```
train.py                 C-GRPO training loop, calibration, adaptive group construction
train_baseline.py        fixed-G GRPO and the dynamic-budget baselines
conformal.py             nonconformity scores, split-conformal quantiles, auto-δ
grpo_loss.py             advantages, per-token KL, REINFORCE objective
generation.py            batched sampling
verifier.py              exact-match and execution-based verification
data.py                  dataset loading and prompt formatting
eval_pareto.py           fixed-k vs conformal evaluation + coverage measurement
baselines/               AERO, GDRO, GRESO, Reinforce-Ada/Est
analysis/                post-hoc analysis of completed runs (see below)
scripts/slurm/           cluster launchers used for the reported experiments
integrations/trl_cgrpo/  CGRPOTrainer: a GRPOTrainer subclass for TRL
```

## Analysis scripts

These operate on completed runs under `runs/` and emit CSV/JSON to `analysis/results/`:

- **`analysis/analyze_calibration_overhead.py`** — decomposes each run's wall clock into
  training / calibration / evaluation using event timestamps, and compares C-GRPO's true cost
  (training **plus all calibration**) against fixed-`G=K_max` costed at the same run's measured
  rollout rate. Also checks the crossover condition
  `n_cal/Δ_cal < B · s · (1 − k̄/K_max)`, where `s` is the measured throughput ratio of
  calibration generation to training rollouts.
- **`analysis/verify_theory.py`** — CPU-only checks of Theorem 2, Definition 1, Proposition 4,
  Corollary 5 and Proposition 16, by Monte Carlo plus the qhat vectors in completed runs.
- **`analysis/analyze_coverage_over_training.py`** — per-checkpoint empirical coverage against
  the nominal `1 − δ` in force at that checkpoint, with a binomial-noise test.

## Two measurement caveats worth knowing

**Calibration is not free.** Re-calibration draws `n_cal × K_max` completions at every
checkpoint. At `n_cal=200, Δ_cal=50, K_max=32` that is 128 rollouts/step against roughly 50
for training itself. Any efficiency claim should state whether calibration is inside or
outside the accounting; `analysis/analyze_calibration_overhead.py` reports both.

**Two different coverage quantities.** Theorem 2 bounds coverage at a *fixed* `k`. The
quantity that is easy to log — coverage at the adaptively selected `k_used` — is a
*post-selection* quantity: the stopping rule picks the smallest `k` with a singleton set,
which biases it downward, and the theorem does not govern it. `eval_pareto.py` reports the
fixed-`k` quantity as `theorem2_coverage_per_k`; the training loop's `eval/conf_coverage` is
the post-selection one. They are not interchangeable.

**Calibration caveat (`--split-delta`).** By default `δ_auto` is selected from the same
calibration scores that produce `q̂`, which uses the calibration data twice and voids
split-conformal exactness. `--split-delta` partitions `D_cal` into disjoint halves — one
selects δ, the other calibrates `q̂` — restoring exactness at an effective calibration size of
`n_cal/2`. Default is off, preserving the original behaviour.

## Reproducing results

Numbers should be regenerated from this code rather than copied from any write-up: run the
training commands above, then `eval_pareto.py`, then the analysis scripts. Each run directory
contains `events.jsonl` (per-step metrics), `state.json` (final evaluation), and
`pareto_results_*.json`, so every reported quantity is traceable to a run.

`scripts/slurm/` contains the exact cluster launchers, including the batched multi-arm runner.
Note that co-scheduling several arms on one GPU slows each of them substantially — generation
is memory-bandwidth bound — so wall-clock numbers should only be taken from runs that had the
device to themselves.

## Publishing an adapter to the Hugging Face Hub

```bash
huggingface-cli login
python scripts/push_to_hub.py \
  --run-dir runs/cgrpo_gsm8k/<run>/final \
  --repo-id <user>/c-grpo-qwen2.5-7b-gsm8k \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dry-run          # drop to actually upload
```

The model card's configuration and evaluation tables are generated **from the run's own
`events.jsonl` and `pareto_results_*.json`**, so the card cannot drift from the checkpoint it
describes. If a run has no evaluation artifact, the card says so instead of filling in numbers.

## Using C-GRPO with TRL

`integrations/trl_cgrpo/` provides `CGRPOTrainer`, a subclass of TRL's `GRPOTrainer`, so the method can be used
without this research codebase. It is a standalone copy of the `trl.experimental.cgrpo` implementation proposed
upstream, and works with released TRL (tested with TRL 1.12 and TRL main):

```bash
pip install -e integrations/
pytest integrations/trl_cgrpo/tests -q   # CPU; downloads tiny test models
```

```python
from trl_cgrpo import CGRPOConfig, CGRPOTrainer

trainer = CGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=correctness_reward,
    args=CGRPOConfig(budget_grid=[2, 4, 8, 16, 32], recalibrate_every=50),
    train_dataset=train_ds,
    calibration_dataset=calibration_ds,   # held-out, with an "answer" column
    answer_extractor=extract_answer,
)
```

Early-stopped prompts are padded to `max(budget_grid)` with one-token completions that get a NaN reward (zero
advantage) and are removed from the loss mask and normalizer. The conformal utilities agree exactly with
[`conformal.py`](conformal.py) on quantile, APS score and prediction set. See
[`integrations/trl_cgrpo/README.md`](integrations/trl_cgrpo/README.md) for options and current limits
(single process; vLLM supported in server and colocate mode).

## Citation

```bibtex
@misc{cgrpo,
  title  = {C-GRPO: Conformal Group Relative Policy Optimization},
  year   = {2026},
  note   = {Code: https://github.com/AryaFayyazi/CGRPO}
}
```

## License

MIT — see [LICENSE](LICENSE).
