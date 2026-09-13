#!/bin/bash
#SBATCH --job-name=cgrpo_math500_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_math500_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_math500_v3_%j.log

# C-GRPO training on MATH-500 — v3: three root-cause fixes for dynamic-k
#
# ROOT CAUSES of qhats=1.0 throughout prior training (all fixed here):
#
#   BUG 1 — delta_max=0.50 cap in select_delta_auto (conformal.py)
#     Auto-delta computes δ = (1 - pass@k_max) + 0.05.  With the base model
#     getting pass@32 ≈ 74.7% on MATH-500 cal set, this gives δ=0.30 which was
#     fine.  But with k_max=32 AND harder problems, δ can exceed 0.50 and was
#     silently clipped, causing qhat=1.0.  Cap raised to 0.95.
#
#   BUG 2 — Math k-selection already uses majority-vote score space (correct),
#     but the recalibration block was passing the WRONG score type for code.
#     For math datasets this bug only appeared in code paths; however fixing
#     it ensures consistent behavior and prevents future regressions.
#
#   BUG 3 — qhats from step-0 base model never recalibrated with updated delta
#     recalibrate_every=100 was already in config.py but the recalibration
#     block didn't re-run auto_delta — it always used the fixed δ from step 0.
#     Fixed: recalibration now passes auto_delta=True, so δ tightens as the
#     model's pass@k_max improves over training.
#
# v3 vs v2 changes:
#   - All 3 bug fixes confirmed present in train.py and conformal.py
#   - eval k-max: 32 → 64  (adds pass@64 fixed-k oracle to pareto table)
#   - DELTA=auto  (was explicit 0.25 in v1; auto since v2)

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
DS_NAME="HuggingFaceH4/MATH-500"
DS_CONFIG=""
N_TRAIN=200
N_CAL=100
N_EVAL=200
MAX_TOK=1024
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto   # δ = (1 - pass@k_max) + 0.05, uncapped up to 0.95

DS_SLUG="math500"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  n_eval=$N_EVAL  max_new_tokens=$MAX_TOK"
echo " Fixes: delta_max cap removed, exec-based k-sel, recal+auto_delta"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --n-eval              "$N_EVAL" \
  --max-new-tokens      "$MAX_TOK" \
  --k-values            "$K_VALUES" \
  --delta               "$DELTA" \
  --log-dir             "$LOGDIR"

# ---- eval ---------------------------------------------------------------
CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  # Fall back to latest ckpt_step_* if no "final" checkpoint saved
  CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
fi
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  k-max=64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
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
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

# ── Paper figures ─────────────────────────────────────────────────────────────
PARETO_SLUG="HuggingFaceH4_MATH-500"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "math500" \
  --pareto-json  "$CGRPO_PARETO" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 MATH-500"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset math500"
