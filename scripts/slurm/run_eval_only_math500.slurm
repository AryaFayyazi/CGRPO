#!/bin/bash
#SBATCH --job-name=eval_math500
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/eval_math500_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/eval_math500_%j.log

# Pareto / comparison-table evaluation for MATH-500 (HuggingFaceH4/MATH-500)
# Evaluates: C-GRPO fine-tuned LoRA checkpoint AND base model (no LoRA),
# producing the same cost-vs-accuracy comparison table as other datasets.

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

CKPT="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/QWEN2.5-7b_math500_seed0/QWEN2.5-7b_delta0.1_seed0_20260408_155201/final"

echo "============================================================"
echo " Pareto eval (LoRA checkpoint — C-GRPO MATH-500)"
echo " CKPT=$CKPT"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "QWEN2.5-7b" \
  --dataset-name        "HuggingFaceH4/MATH-500" \
  --dataset-config      "" \
  --n-cal               100 \
  --n-eval              200 \
  --k-max               16 \
  --batch-size          1 \
  --max-new-tokens      1024 \
  --dynamic-baselines   "softmax,entropy" \
  --baseline-thresholds "0.5,0.7"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA — MATH-500)"
echo "============================================================"
python eval_pareto.py \
  --no-lora \
  --model-key           "QWEN2.5-7b" \
  --dataset-name        "HuggingFaceH4/MATH-500" \
  --dataset-config      "" \
  --n-cal               100 \
  --n-eval              200 \
  --k-max               16 \
  --batch-size          1 \
  --max-new-tokens      1024 \
  --dynamic-baselines   "softmax,entropy" \
  --baseline-thresholds "0.5,0.7"

echo ""
echo "MATH-500 eval complete."
