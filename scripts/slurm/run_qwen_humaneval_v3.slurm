#!/bin/bash
#SBATCH --job-name=cgrpo_humaneval_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_humaneval_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_humaneval_v3_%j.log

# C-GRPO training on HumanEval — v3: three root-cause fixes for dynamic-k
#
# ROOT CAUSES of qhats=1.0 throughout v2 training (all fixed here):
#
#   BUG 1 — delta_max=0.50 cap in select_delta_auto (conformal.py)
#     Auto-delta computes δ = (1 - pass@k_max) + 0.05.  With the base model
#     getting only 15% pass@32 on the cal set, this gives δ=0.90 — but the
#     old cap clipped it to δ=0.50.  With δ=0.50, qhat = 50th-percentile of
#     nonconformity scores = 1.0 (since 85% of cal problems are unsolvable
#     at step 0).  cap raised to 0.95 so δ=0.90 actually applies, and the
#     10th-percentile qhat is < 1.0 for any solvable problem.
#
#   BUG 2 — APS frequency scoring used for code k-selection in training loop
#     (train.py lines 677-679 before this fix).  qhats are calibrated with
#     execution-based scores (score = 1 - pass_rate); but the training loop
#     called choose_k_and_sets() which uses APS cumulative-frequency scores
#     — a completely different score space.  For code, every completion is a
#     unique string, so all APS scores ≈ 1.0 regardless of qhats → k_used
#     always = k_max.  Fixed: training loop now pre-executes completions,
#     computes score_k = 1 - n_pass/k, and selects min k where score_k ≤
#     qhat_k.  Same fix applied to evaluate() for accurate conf_acc / avg_k.
#
#   BUG 3 — qhats from step-0 base model never recalibrated during training
#     recalibrate_every=100 was already in config.py but the recalibration
#     block didn't re-run auto_delta — it always used the fixed δ from step 0.
#     Fixed: recalibration now passes auto_delta=True, so δ tightens as the
#     model's pass@k_max improves (e.g. pass@32=99% at step 300 → δ≈0.15).

set -euo pipefail

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true

# ---- environment --------------------------------------------------------
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate cgrpo
cd ${CGRPO_ROOT:-$PWD}
export HF_DATASETS_CACHE="hf_cache/datasets"
export HF_HOME="hf_cache/hub"
export HF_HUB_CACHE="hf_cache/hub"
export TRANSFORMERS_CACHE="hf_cache/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="QWEN2.5-7b"
DS_NAME="openai/openai_humaneval"
DS_CONFIG=""
N_TRAIN=80
N_CAL=40
MAX_TOK=512
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto   # δ = (1 - pass@k_max) + 0.05, now uncapped up to 0.95

DS_SLUG="${DS_NAME##*/}"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  max_new_tokens=$MAX_TOK"
echo " Fixes: delta_max cap removed, exec-based k-selection, recal+auto_delta"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --max-new-tokens      "$MAX_TOK" \
  --k-values            "$K_VALUES" \
  --delta               "$DELTA" \
  --log-dir             "$LOGDIR"

# ---- eval ---------------------------------------------------------------
CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

# n_eval=54: matches training eval set size (min(400, 164//3))
# n_cal=40:  same held-out cal set used throughout training

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  k-max=64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 40 --n-eval 54 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)  k-max=64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 40 --n-eval 54 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

# ── Baseline pareto evals (pass@k for all fixed k up to 64) ──────────────────
# Finds the latest ckpt_step_* in each baseline's run dir and runs eval_pareto.
# This generates pareto_results_*.json next to the ckpt, which compare_methods.py
# reads to build the pass@k comparison table (TABLE 5).

_baseline_pareto() {
  local method=$1
  local base_dir="runs/baseline/${method}/QWEN2.5-7b_humaneval_seed${SEED}"
  local ckpt_dir
  ckpt_dir=$(ls -dt "${base_dir}"/*/ckpt_step_* 2>/dev/null | head -1)
  if [[ -z "$ckpt_dir" ]]; then
    echo "WARNING: no checkpoint found for baseline ${method}, skipping."
    return 0
  fi
  echo ""
  echo "============================================================"
  echo " Baseline pareto eval: ${method}  ckpt=${ckpt_dir##*/}  k-max=64"
  echo "============================================================"
  python eval_pareto.py \
    --ckpt-dir          "$ckpt_dir" \
    --model-key         "$MODEL" \
    --dataset-name      "$DS_NAME" \
    --dataset-config    "$DS_CONFIG" \
    --n-cal 40 --n-eval 54 \
    --k-max 64 \
    --batch-size 1 \
    --max-new-tokens    "$MAX_TOK"
}

_baseline_pareto grpo
_baseline_pareto aero
_baseline_pareto gdro

# ── Paper figures ─────────────────────────────────────────────────────────────
# Locate baseline ckpt dirs (same logic as _baseline_pareto above)
GRPO_CKPT=$(ls -dt "runs/baseline/grpo/QWEN2.5-7b_humaneval_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)
AERO_CKPT=$(ls -dt "runs/baseline/aero/QWEN2.5-7b_humaneval_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)
GDRO_CKPT=$(ls -dt "runs/baseline/gdro/QWEN2.5-7b_humaneval_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)

PARETO_SLUG="openai_openai_humaneval"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "humaneval" \
  --pareto-json  "$CGRPO_PARETO" \
  --baseline-grpo "runs/baseline/grpo/QWEN2.5-7b_humaneval_seed${SEED}" \
  --baseline-aero "runs/baseline/aero/QWEN2.5-7b_humaneval_seed${SEED}" \
  --baseline-gdro "runs/baseline/gdro/QWEN2.5-7b_humaneval_seed${SEED}" \
  --baseline-pareto-grpo "${GRPO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --baseline-pareto-aero "${AERO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --baseline-pareto-gdro "${GDRO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 HumanEval + all baseline pareto evals"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset humaneval"
