#!/bin/bash
#SBATCH --job-name=rb_q4long
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/q4long_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/q4long_%j.log

# ============================================================================
# Long-horizon study: longer-horizon training curve beyond 300 steps.
#
# Pure extension of the headline GSM8K C-GRPO configuration to 600 steps.
# Every hyperparameter is identical to
#   runs/conformal_grpo/QWEN2.5-7b_gsm8k_main_v3_seed0/
#     QWEN2.5-7b_deltaauto_seed0_20260419_141159
# except `steps` (300 -> 600) and the in-loop eval cadence (50 -> 100,
# n_eval 400 -> 200) which only affects measurement cost, not training.
#
# This answers: does the conformal curriculum stay stable, or does k_used
# keep collapsing until exploration dies? The k_used / reward / qhat
# trajectories in events.jsonl are the deliverable.
# ============================================================================

set -euo pipefail

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true

# ---- environment (identical to run_qwen_gsm8k_v3.slurm) -------------------
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
N_CAL=200            # identical to the 300-step v3 run
N_EVAL=200           # halved: in-loop eval is measurement, not method
N_EVAL_PARETO=400
MAX_TOK=512
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto
STEPS=600            # <-- the only substantive change vs. v3
RECAL_EVERY=50       # identical to v3
EVAL_EVERY=100       # measurement cadence only

LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/cgrpo_long600/${MODEL}_gsm8k_main_seed${SEED}"
mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs

echo ""
echo "============================================================"
echo " Q4 long-horizon C-GRPO   steps=${STEPS}  (v3 config otherwise)"
echo " k_values=${K_VALUES}  delta=${DELTA}  recal_every=${RECAL_EVERY}"
echo " n_train=${N_TRAIN}  n_cal=${N_CAL}  max_new_tokens=${MAX_TOK}"
echo " logdir=${LOGDIR}"
echo "============================================================"

START_TS=$(date +%s)

python -u run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --n-eval              "$N_EVAL" \
  --max-new-tokens      "$MAX_TOK" \
  --steps               "$STEPS" \
  --k-values            "$K_VALUES" \
  --delta               "$DELTA" \
  --batch-size          4 \
  --eval-every          "$EVAL_EVERY" \
  --recalibrate-every   "$RECAL_EVERY" \
  --log-dir             "$LOGDIR"

TRAIN_END_TS=$(date +%s)
echo ""
echo ">>> TRAIN_WALLCLOCK_SECONDS q4_long600: $((TRAIN_END_TS - START_TS))"

CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
fi
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval  (C-GRPO @600 steps)  k-max=32"
echo "============================================================"

python -u eval_pareto.py \
  --ckpt-dir         "$CKPT" \
  --model-key        "$MODEL" \
  --dataset-name     "$DS_NAME" \
  --dataset-config   "$DS_CONFIG" \
  --n-cal            "$N_CAL" \
  --n-eval           "$N_EVAL_PARETO" \
  --k-max            32 \
  --batch-size       4 \
  --max-new-tokens   "$MAX_TOK"

END_TS=$(date +%s)
echo ""
echo ">>> TOTAL_WALLCLOCK_SECONDS q4_long600: $((END_TS - START_TS))"
echo ">>> DONE Q4 long-horizon"
