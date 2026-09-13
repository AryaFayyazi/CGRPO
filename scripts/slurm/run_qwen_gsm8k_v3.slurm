#!/bin/bash
#SBATCH --job-name=cgrpo_gsm8k_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_gsm8k_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_gsm8k_v3_%j.log

# C-GRPO training on GSM8K — v3: all root-cause fixes for dynamic-k
#
# Fixes applied (same as HumanEval/MATH-500 v3):
#   BUG 1 — delta_max=0.50 cap raised to 0.95 (conformal.py)
#   BUG 2 — exec-based k-selection in training & evaluate() loops (train.py)
#   BUG 3 — recalibration now passes auto_delta=True (train.py)
#   BUG 4 — skip qhat>=1.0 levels in k-selection loop (train.py)
#   BUG 5 — recalibration uses shared_ref=False → trained policy (train.py)
#   CONFIG — recalibrate_every=50 (config.py)
#   GENERATION — chunked num_return_sequences (generation.py, OOM fix)
#
# Baseline checkpoints already exist at:
#   runs/baseline/grpo/QWEN2.5-7b_gsm8k_seed0/  (job 609)
#   (other baselines: aero, gdro are in same baseline dir tree)

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
DS_NAME="openai/gsm8k"
DS_CONFIG="main"
N_TRAIN=400
N_CAL=200
N_EVAL=400
MAX_TOK=512
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto

LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_gsm8k_main_v3_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME ($DS_CONFIG)  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  n_eval=$N_EVAL  max_new_tokens=$MAX_TOK"
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

# ── Baseline pareto evals (checkpoints exist from job 609) ────────────────────
_baseline_pareto() {
  local method=$1
  local base_dir="runs/baseline/${method}/${MODEL}_gsm8k_seed${SEED}"
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
    --n-cal "$N_CAL" --n-eval "$N_EVAL" \
    --k-max 64 \
    --batch-size 1 \
    --max-new-tokens    "$MAX_TOK"
}

_baseline_pareto grpo
_baseline_pareto aero
_baseline_pareto gdro

# ── Paper figures ─────────────────────────────────────────────────────────────
PARETO_SLUG="openai_gsm8k"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"
GRPO_CKPT=$(ls -dt "runs/baseline/grpo/${MODEL}_gsm8k_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)
AERO_CKPT=$(ls -dt "runs/baseline/aero/${MODEL}_gsm8k_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)
GDRO_CKPT=$(ls -dt "runs/baseline/gdro/${MODEL}_gsm8k_seed${SEED}"/*/ckpt_step_* 2>/dev/null | head -1)

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "gsm8k" \
  --pareto-json  "$CGRPO_PARETO" \
  --baseline-grpo "runs/baseline/grpo/${MODEL}_gsm8k_seed${SEED}" \
  --baseline-aero "runs/baseline/aero/${MODEL}_gsm8k_seed${SEED}" \
  --baseline-gdro "runs/baseline/gdro/${MODEL}_gsm8k_seed${SEED}" \
  --baseline-pareto-grpo "${GRPO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --baseline-pareto-aero "${AERO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --baseline-pareto-gdro "${GDRO_CKPT}/pareto_results_${PARETO_SLUG}.json" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 GSM8K"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset gsm8k"
