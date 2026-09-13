#!/bin/bash
#SBATCH --job-name=r2q2_llama
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q2_llama_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q2_llama_%j.log

# ============================================================================
# Stronger-base study: does the advantage survive a stronger base model?
#
# R2 asks whether savings persist "starting from a model already scoring ~85%
# on GSM8K", and whether the benefit shrinks once zero-variance pruning
# dominates. Llama-3.1-8B-Instruct is the strongest locally available base:
# the existing C-GRPO run reaches p@16 = 88.0 / p@32 = 88.75 on GSM8K versus
# Qwen2.5-7B's 68.8 / 70.0, so it probes exactly that regime.
#
# The gap in the current evidence is that no fixed-G comparison exists at that
# base strength, so "savings shrink or persist?" cannot be answered. This runs
# the same three arms as the Qwen sweep, at matched configuration:
#     ARM=cgrpo  adaptive C-GRPO
#     ARM=8      fixed G=8   GRPO
#     ARM=16     fixed G=16  GRPO
#
# Full-parameter fine-tuning of a 7-8B model is NOT attempted: AdamW states put
# it near 98 GB, above this node's 95 GB per-GPU budget. That limitation is
# reported rather than worked around.
#
# Invoke with:  sbatch --export=ALL,ARM=8 scripts/r2q2_stronger_base.slurm
# ============================================================================

set -euo pipefail

if [[ -z "${ARM:-}" ]]; then
  echo "ERROR: ARM not set. Use: sbatch --export=ALL,ARM=cgrpo|8|16 ..." >&2
  exit 1
fi

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

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

MODEL="llama3.1-8b"
DS_NAME="openai/gsm8k"; DS_CONFIG="main"
N_TRAIN=400; N_EVAL=400; MAX_TOK=512; SEED=0; STEPS=300
K_VALUES="2,4,8,16,32"; N_CAL_CGRPO=200; N_CAL_GRPO=40

START_TS=$(date +%s)

if [[ "$ARM" == "cgrpo" ]]; then
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/llama_cgrpo/${MODEL}_gsm8k_seed${SEED}"
  mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs
  echo "=== study Llama-3.1-8B C-GRPO (stronger base) ==="
  python -u run_train.py \
    --model-key "$MODEL" --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
    --seed "$SEED" --n-train "$N_TRAIN" --n-cal "$N_CAL_CGRPO" --n-eval "$N_EVAL" \
    --max-new-tokens "$MAX_TOK" --steps "$STEPS" --k-values "$K_VALUES" \
    --delta auto --batch-size 4 --eval-every 50 --recalibrate-every 50 \
    --log-dir "$LOGDIR"
  N_CAL_EVAL="$N_CAL_CGRPO"
  TAG="llama_cgrpo"
else
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/llama_grpo_G${ARM}/${MODEL}_gsm8k_seed${SEED}"
  mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs
  echo "=== study Llama-3.1-8B fixed GRPO G=${ARM} ==="
  python -u run_train_baseline.py \
    --method grpo --model-key "$MODEL" \
    --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
    --seed "$SEED" --n-train "$N_TRAIN" --n-cal "$N_CAL_GRPO" --n-eval 200 \
    --max-new-tokens "$MAX_TOK" --steps "$STEPS" --n-rollouts "$ARM" \
    --batch-size 4 --eval-every 999 --ckpt-every 100 \
    --log-dir "$LOGDIR"
  N_CAL_EVAL=200
  TAG="llama_G${ARM}"
fi

echo ">>> TRAIN_WALLCLOCK_SECONDS ${TAG}: $(( $(date +%s) - START_TS ))"

CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint under $LOGDIR" >&2; exit 1; }
echo "Checkpoint: $CKPT"

python -u eval_pareto.py \
  --ckpt-dir "$CKPT" --model-key "$MODEL" \
  --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
  --n-cal "$N_CAL_EVAL" --n-eval "$N_EVAL" \
  --k-max 32 --batch-size 4 --max-new-tokens "$MAX_TOK"

echo ">>> TOTAL_WALLCLOCK_SECONDS ${TAG}: $(( $(date +%s) - START_TS ))"
echo ">>> DONE study ${TAG}"
