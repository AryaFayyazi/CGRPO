#!/bin/bash
#SBATCH --job-name=cgrpo_qwen_mbpp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_mbpp_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_mbpp_%j.log

# GRPO + conformal prediction training on MBPP (Python coding dataset)
# Model:   QWEN2.5-7b  (Qwen2.5-7B-Instruct, ~16 GB bf16)
# Dataset: google-research-datasets/mbpp  (374 train / 500 test examples)
#
# Verification: execution-based — generated code is run in a subprocess
# against the provided unit-test assertions.
# Canonical answer: normalized Python code text for conformal clustering.
#
# Note on eval speed: each completion requires a subprocess call to check
# assertions, so eval takes longer per prompt than math datasets.
# --batch-size 1 prevents additional OOM from k_max=64 concurrent KV caches.

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
DS_NAME="google-research-datasets/mbpp"
DS_CONFIG="full"
# MBPP train split has 374 examples; use 270 for training, 80 for calibration.
N_TRAIN=270
N_CAL=80
# Code is longer than math — 512 tokens allows most MBPP solutions.
MAX_TOK=512
SEED=0
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

DS_SLUG="${DS_NAME##*/}"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_${DS_CONFIG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " Training: $MODEL  |  $DS_NAME ($DS_CONFIG)  |  seed=$SEED"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  max_new_tokens=$MAX_TOK"
echo " Verification: code execution (subprocess, 10s timeout)"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --max-new-tokens      "$MAX_TOK" \
  --log-dir             "$LOGDIR"

CKPT=$(ls -d "$LOGDIR"/*/final 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  --batch-size 1  --k-max 64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 80 --n-eval 80 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)  --batch-size 1  --k-max 64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 80 --n-eval 80 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "============================================================"
echo " Cross-eval: MBPP-trained model on HumanEval (no LoRA)"
echo " (out-of-distribution — shows domain transfer)"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "openai/openai_humaneval" \
  --dataset-config      "" \
  --n-cal 40 --n-eval 44 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS" || true  # don't abort if cross-eval fails

echo ""
echo "Qwen2.5-7B + MBPP complete."
