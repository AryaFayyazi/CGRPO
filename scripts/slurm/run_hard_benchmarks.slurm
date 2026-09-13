#!/bin/bash
#SBATCH --job-name=cgrpo_hard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/hard_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/hard_%j.log

set -euo pipefail

# ---- environment --------------------------------------------------------
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate cgrpo
cd ${CGRPO_ROOT:-$PWD}
export HF_DATASETS_CACHE="hf_cache/datasets"
export HF_HOME="hf_cache/hub"
export HF_HUB_CACHE="hf_cache/hub"
export TRANSFORMERS_CACHE="hf_cache/hub"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

MODEL="QWEN2.5-7b"
SEED=0

HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

# ---- dataset configs -------------------------------------------------------
# Format: "dataset_name  config  n_train  n_cal  max_new_tokens"
#
# Rationale for each:
#  medmcqa        – 4-choice medical MCQ; Qwen2.5-7B expected ~62-67%;
#                   182k train examples → rich reward signal
#  allenai/sciq   – 4-choice science MCQ; ~75-80%; 11.7k train;
#                   harder than ARC-Easy, easier than ARC-Challenge
#  hendrycks_math – open-ended symbolic/numeric math (levels 3-5 only);
#                   ~40-55%; algebra subset used for training size & diversity
#
# n_train / n_cal chosen to give ≥300 GRPO steps at batch=4 per step.
# max_new_tokens: MCQ needs ~8-16 tokens; math needs more headroom.
DATASETS=(
  "openlifescienceai/medmcqa  default         2000  400  16"
  "allenai/sciq               default         2000  400  16"
  "EleutherAI/hendrycks_math  algebra         1500  400  256"
)

run_experiment() {
  local DS_NAME="$1"
  local DS_CONFIG="$2"
  local N_TRAIN="$3"
  local N_CAL="$4"
  local MAX_TOK="$5"

  local DS_SLUG="${DS_NAME##*/}"
  local LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_${DS_CONFIG}_seed${SEED}"
  mkdir -p "${LOGDIR}"

  echo ""
  echo "============================================================"
  echo "Training: $DS_NAME / $DS_CONFIG  seed=$SEED"
  echo "  logdir=$LOGDIR"
  echo "  n_train=$N_TRAIN  n_cal=$N_CAL  max_new_tokens=$MAX_TOK"
  echo "============================================================"

  # ---- training ----------------------------------------------------------
  python run_train.py \
    --model-key           "$MODEL" \
    --dataset-name        "$DS_NAME" \
    --dataset-config      "$DS_CONFIG" \
    --seed                "$SEED" \
    --n-train             "$N_TRAIN" \
    --n-cal               "$N_CAL" \
    --max-new-tokens      "$MAX_TOK" \
    --log-dir             "$LOGDIR"

  # Locate checkpoint produced by this run
  local CKPT
  CKPT=$(ls -d "$LOGDIR"/*/final 2>/dev/null | sort | tail -1)
  if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint found under $LOGDIR" >&2
    return 1
  fi
  echo "Checkpoint: $CKPT"

  echo ""
  echo "---- Pareto eval (LoRA checkpoint) ----"
  python eval_pareto.py \
    --ckpt-dir            "$CKPT" \
    --model-key           "$MODEL" \
    --dataset-name        "$DS_NAME" \
    --dataset-config      "$DS_CONFIG" \
    --n-cal 400 --n-eval 400 \
    --k-max 64 \
    --max-new-tokens      "$MAX_TOK" \
    --dynamic-baselines   "$HEURISTICS" \
    --baseline-thresholds "$THRESHOLDS"

  echo ""
  echo "---- Pareto eval (base model, no LoRA) ----"
  python eval_pareto.py \
    --ckpt-dir            "$CKPT" \
    --no-lora \
    --model-key           "$MODEL" \
    --dataset-name        "$DS_NAME" \
    --dataset-config      "$DS_CONFIG" \
    --n-cal 400 --n-eval 400 \
    --k-max 64 \
    --max-new-tokens      "$MAX_TOK" \
    --dynamic-baselines   "$HEURISTICS" \
    --baseline-thresholds "$THRESHOLDS"
}

# ---- run each dataset ---------------------------------------------------
for entry in "${DATASETS[@]}"; do
  read -r DS_NAME DS_CONFIG N_TRAIN N_CAL MAX_TOK <<< "$entry"
  run_experiment "$DS_NAME" "$DS_CONFIG" "$N_TRAIN" "$N_CAL" "$MAX_TOK"
done

echo ""
echo "ALL DONE"
