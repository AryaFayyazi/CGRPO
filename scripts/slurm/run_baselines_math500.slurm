#!/bin/bash
#SBATCH --job-name=baselines_math500
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/baselines_math500_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/baselines_math500_%j.log

# Baseline comparison on MATH-500 (HuggingFaceH4/MATH-500)
# 500 test-only examples; partitioned: 200 train / 100 cal / 200 eval.

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
N_TRAIN=200
N_CAL=100
N_EVAL=200
MAX_TOK=1024
STEPS=300
N_ROLLOUTS=4
SEED=0

METHODS=(grpo aero gdro gdro_prompt reinforce_ada reinforce_est greso)

echo ""
echo "============================================================"
echo " Baselines: $DS_NAME  |  model=$MODEL  |  seed=$SEED"
echo " methods: ${METHODS[*]}"
echo "============================================================"

for METHOD in "${METHODS[@]}"; do
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/baseline/${METHOD}/${MODEL}_math500_seed${SEED}"
  mkdir -p "${LOGDIR}"

  echo ""
  echo "---- Method: $METHOD ----"
  python run_train_baseline.py \
    --method          "$METHOD" \
    --model-key       "$MODEL" \
    --dataset-name    "$DS_NAME" \
    --dataset-config  "$DS_CONFIG" \
    --seed            "$SEED" \
    --n-train         "$N_TRAIN" \
    --n-cal           "$N_CAL" \
    --n-eval          "$N_EVAL" \
    --max-new-tokens  "$MAX_TOK" \
    --steps           "$STEPS" \
    --n-rollouts      "$N_ROLLOUTS" \
    --log-dir         "$LOGDIR"

  echo "Done: $METHOD on math500"
done

echo ""
echo "All baseline methods completed for MATH-500."
