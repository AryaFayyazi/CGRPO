#!/bin/bash
#SBATCH --job-name=baselines_gpqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/baselines_gpqa_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/baselines_gpqa_%j.log

# Baseline comparison on GPQA-Diamond (hendrydong/gpqa_diamond_mc)
# 198 test-only examples; partitioned: 80 train / 40 cal / 78 eval.

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
DS_NAME="hendrydong/gpqa_diamond_mc"
DS_CONFIG=""
N_TRAIN=80
N_CAL=40
N_EVAL=78
MAX_TOK=1024
STEPS=200
N_ROLLOUTS=4
SEED=0

METHODS=(grpo aero gdro gdro_prompt reinforce_ada reinforce_est greso)

echo ""
echo "============================================================"
echo " Baselines: $DS_NAME  |  model=$MODEL  |  seed=$SEED"
echo " methods: ${METHODS[*]}"
echo "============================================================"

for METHOD in "${METHODS[@]}"; do
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/baseline/${METHOD}/${MODEL}_gpqa_seed${SEED}"
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

  echo "Done: $METHOD on gpqa"
done

echo ""
echo "All baseline methods completed for GPQA."
