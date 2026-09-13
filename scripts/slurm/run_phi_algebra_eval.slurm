#!/bin/bash
#SBATCH --job-name=cgrpo_phi_alg_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/phi_algebra_eval_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/phi_algebra_eval_%j.log

set -euo pipefail

# Training for phi+algebra already completed. This script only runs eval.
# The OOM in the previous run was caused by:
#   1. PYTORCH_ALLOC_CONF (wrong name) -> fixed to PYTORCH_CUDA_ALLOC_CONF
#   2. BATCH_SIZE=4 x K_MAX=64 = 256 concurrent sequences -> fixed with --batch-size 1

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

MODEL="phi-3.5-mini-instruct"
DS_NAME="EleutherAI/hendrycks_math"
DS_CONFIG="algebra"
MAX_TOK=256
SEED=0
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

# Use the most recent completed checkpoint
CKPT=$(ls -d ${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_hendrycks_math_${DS_CONFIG}_seed${SEED}/*/final 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no final checkpoint found" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  --batch-size 1  --k-max 64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 400 --n-eval 400 \
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
  --n-cal 400 --n-eval 400 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "Phi + algebra eval complete."
