#!/bin/bash
#SBATCH --job-name=baselines_gsm8k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/baselines_gsm8k_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/baselines_gsm8k_%j.log

# Baseline comparison on GSM8K
# Runs all 7 methods sequentially: grpo, aero, gdro, gdro_prompt,
#   reinforce_ada, reinforce_est, greso
# Model: QWEN2.5-7b

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
N_EVAL=400   # match C-GRPO eval set size for fair variance
MAX_TOK=256  # match C-GRPO token budget for fair comparison
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
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/baseline/${METHOD}/${MODEL}_gsm8k_seed${SEED}"
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

  echo "Done: $METHOD on gsm8k"
done

echo ""
echo "All baseline methods completed for GSM8K."
