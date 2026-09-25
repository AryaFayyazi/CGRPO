# Reproducing the runs

Every command below was run on one H100 (95 GB) with the `cgrpo` environment from
[`requirements.txt`](requirements.txt), seed 0. Model paths resolve through
[`model_registry.py`](model_registry.py) (set `CGRPO_MODEL_ROOT`, and point each entry at a snapshot
directory containing `config.json` and the tokenizer files, not at a Hugging Face cache root).

Each training run writes `events.jsonl` (per-step metrics), `state.json`, checkpoints, and — after
the evaluation step — `pareto_results_<dataset>_lora.json` into its `--log-dir`.

## Settings used

| | GSM8K (paper-era length) | MATH-500 / MBPP / GPQA | Notes |
|---|---|---|---|
| `--max-new-tokens` | 256 | 512 | The generation length changes the regime completely: at 512 tokens Qwen2.5-7B-Instruct answers GSM8K near its ceiling (89.5% greedy for the *untrained* model, against 52.2% at 256), and adaptive sampling then stops at the smallest budget for nearly every prompt. |
| `--steps` | 100 | 100 | The paper's runs are 300 steps. k̄ falls only after the k=2 threshold drops below 1.0, which had not happened by step 100 in our runs. |
| `--n-train` / `--n-eval` | 400 / 200 | 400 / 200 | |
| `--n-cal` | 200 | 100 | Calibration cost is linear in `n_cal`. |
| `--k-values` | `2,4,8,16,32` | `2,4,8,16,32` | `--k-values 2,8,32` for the coarse-grid ablation. |
| `--delta` | `auto` | `auto` | |
| `--batch-size` | 4 | 4 | Prompts per optimizer step. |

## C-GRPO

```bash
python -u run_train.py --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main --seed 0 \
  --n-train 400 --n-cal 200 --n-eval 200 \
  --max-new-tokens 256 --steps 100 --k-values "2,4,8,16,32" \
  --delta auto --batch-size 4 --eval-every 999 --recalibrate-every 50 \
  --log-dir runs/cgrpo_gsm8k

python -u eval_pareto.py --ckpt-dir runs/cgrpo_gsm8k/<run>/final --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main \
  --n-cal 200 --n-eval 200 --k-max 32 --k-values "2,4,8,16,32" \
  --batch-size 4 --max-new-tokens 256
```

`--eval-every 999` turns off the in-training evaluation (`eval_pareto.py` evaluates the checkpoint
afterwards). **Pass `--k-values` to the evaluation**: its default grid stops at 16, which would cap
the conformal stopping rule below the grid the checkpoint was trained on.

Other benchmarks use the same pair of commands with `--max-new-tokens 512`, `--n-cal 100` and:

- MATH-500: `--dataset-name HuggingFaceH4/MATH-500 --dataset-config default`
- MBPP: `--dataset-name google-research-datasets/mbpp --dataset-config full`
- GPQA-Diamond: `--dataset-name hendrydong/gpqa_diamond_mc`

Cross-model transfer replaces `--model-key` with `llama3.1-8b`, `gemma3_4b`,
`qwen2.5-math-7b-instruct` or `qwen2.5-3b`; nothing else changes.

## Baselines

```bash
python -u run_train_baseline.py --method grpo --n-rollouts 16 \
  --model-key QWEN2.5-7b --dataset-name openai/gsm8k --dataset-config main --seed 0 \
  --n-train 400 --n-cal 40 --n-eval 200 \
  --max-new-tokens 256 --steps 100 --batch-size 4 \
  --eval-every 999 --ckpt-every 50 --log-dir runs/grpo_G16
```

`--method` also accepts `gdro`, `aero`, `gdro_prompt`, `reinforce_ada`, `reinforce_est` and `greso`
(`--n-rollouts 16` for the dynamic baselines). Evaluate each checkpoint with the same
`eval_pareto.py` command as above.

## Ablations

| Ablation | Change to the C-GRPO command |
|---|---|
| Fixed δ | `--delta 0.05` or `--delta 0.2` instead of `--delta auto` |
| Re-calibration period | `--recalibrate-every 25`, or `0` for static thresholds |
| Budget grid | `--k-values 2,8,32` (coarse) or `1,2,4,8,16,32` (fine) |
| Full-parameter fine-tuning | `--full-finetune --steps 50` (3B fits on one GPU; a 7B model needs two) |

Two notes on the grids. With `k=1` in the grid a single completion always forms a singleton
prediction set, so the stopping rule keeps one completion per prompt and the group carries no
advantage — that configuration is degenerate as implemented. The coarse-grid ablation in the paper
is on MATH-500, so run it with the MATH-500 settings.

## Base-model reference

```bash
python -u eval_pareto.py --ckpt-dir runs/base_placeholder --no-lora --model-key QWEN2.5-7b \
  --dataset-name openai/gsm8k --dataset-config main \
  --n-cal 40 --n-eval 400 --k-max 8 --k-values "2,4,8" --batch-size 4 --max-new-tokens 256
```

Run it at 256 and at 512 tokens to see the effect of the generation length before reading anything
into a trained model's accuracy.

## Reading the numbers

- **Rollouts actually generated** are `2 + (K_max - 2) * train/hard_rate` per prompt, not
  `train/avg_k_used`. Training samples 2 completions for every prompt and expands to `K_max` for any
  prompt whose k=2 prediction set is not a singleton; `avg_k_used` counts the completions *kept* for
  the gradient, which can be fewer. Compute savings from the generated count.
- **Coverage.** `theorem2_coverage_per_k` in the results json is the per-fixed-k quantity Theorem 2
  bounds. `eval/conf_coverage` from training is coverage at the adaptively chosen budget, a
  post-selection quantity the theorem does not govern. The evaluation calibrates at `--delta auto`
  by default, as the paper's coverage table does; a fixed δ on a benchmark the policy mostly fails
  pins every threshold at 1.0 and makes coverage trivially 100%.
- **Calibration is not free.** Each calibration draws `n_cal * K_max` completions. At `n_cal=200`,
  `Δ_cal=50`, `K_max=32` and 4 prompts per step that is roughly twice the training rollouts, and it
  dominated wall-clock in our runs. `analysis/analyze_calibration_overhead.py` reports both
  accountings.
- A threshold of exactly 1.0 does **not** disable early stopping: any prompt whose sampled answers
  all agree still forms a singleton set and stops.

## Analysis

```bash
python analysis/verify_theory.py                     # CPU checks of the theoretical claims
python analysis/analyze_calibration_overhead.py      # wall-clock split and the crossover condition
python analysis/analyze_coverage_over_training.py    # per-checkpoint coverage vs nominal
python analysis/collect_results.py                   # tables from completed runs
```
