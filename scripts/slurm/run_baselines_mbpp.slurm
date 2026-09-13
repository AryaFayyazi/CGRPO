#!/bin/bash
#SBATCH --job-name=baselines_mbpp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/baselines_mbpp_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/baselines_mbpp_%j.log

# Baseline comparison on MBPP (mbpp, sanitized)
# ~374 test examples; partitioned: 150 train / 80 cal / 144 eval.

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
DS_NAME="mbpp"
DS_CONFIG="sanitized"
N_TRAIN=150
N_CAL=80
N_EVAL=144
MAX_TOK=512
STEPS=200
N_ROLLOUTS=4
SEED=0

METHODS=(grpo aero gdro gdro_prompt reinforce_ada reinforce_est greso)

echo ""
echo "============================================================"
echo " Baselines: $DS_NAME/$DS_CONFIG  |  model=$MODEL  |  seed=$SEED"
echo " methods: ${METHODS[*]}"
echo "============================================================"

for METHOD in "${METHODS[@]}"; do
  LOGDIR="${CGRPO_ROOT:-$PWD}/runs/baseline/${METHOD}/${MODEL}_mbpp_seed${SEED}"
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

  echo "Done: $METHOD on mbpp"
done

echo ""
echo "All baseline methods completed for MBPP."
