#!/bin/bash
#SBATCH --job-name=eval_math500
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/eval_math500_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/eval_math500_%j.log

set -euo pipefail

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true

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
MAX_TOK=1024
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

LOGDIR="runs/conformal_grpo/${MODEL}_math500_seed0"
CKPT=$(ls -d "${LOGDIR}"/*/final 2>/dev/null | sort | tail -1)

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  --k-max 32"
echo " ckpt: $CKPT"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 100 --n-eval 200 \
  --k-max 32 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 100 --n-eval 200 \
  --k-max 32 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "MATH-500 eval complete."
