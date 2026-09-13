#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/eval_humaneval_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/eval_humaneval_%j.log

set -euo pipefail

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

CKPT="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/QWEN2.5-7b_openai_humaneval_seed0/QWEN2.5-7b_delta0.1_seed0_20260402_132124/final"

echo "============================================================"
echo " Pareto eval (LoRA checkpoint)"
echo " CKPT=$CKPT"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "QWEN2.5-7b" \
  --dataset-name        "openai/openai_humaneval" \
  --dataset-config      "" \
  --n-cal 40 --n-eval 44 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      512 \
  --dynamic-baselines   "softmax,entropy" \
  --baseline-thresholds "0.5,0.7"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "QWEN2.5-7b" \
  --dataset-name        "openai/openai_humaneval" \
  --dataset-config      "" \
  --n-cal 40 --n-eval 44 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      512 \
  --dynamic-baselines   "softmax,entropy" \
  --baseline-thresholds "0.5,0.7"

echo "HumanEval eval-only complete."
