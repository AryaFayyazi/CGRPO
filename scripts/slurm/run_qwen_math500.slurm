#!/bin/bash
#SBATCH --job-name=cgrpo_qwen_math500
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_math500_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_math500_%j.log

# C-GRPO training on MATH-500 (HuggingFaceH4/MATH-500)
# Model:   QWEN2.5-7b  (Qwen2.5-7B-Instruct)
# Dataset: HuggingFaceH4/MATH-500 — 500 competition math problems (test-only)
#          Partitioned reproducibly: 200 train / 100 cal / 200 eval
#          Gold answers are pre-extracted (no \boxed{} wrapper in gold).

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
DS_NAME="HuggingFaceH4/MATH-500"
DS_CONFIG=""           # no sub-configs
N_TRAIN=200
N_CAL=100
N_EVAL=200
MAX_TOK=1024           # competition math needs CoT space
SEED=0

DS_SLUG="math500"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO Training: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  n_eval=$N_EVAL  max_new_tokens=$MAX_TOK"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --n-eval              "$N_EVAL" \
  --max-new-tokens      "$MAX_TOK" \
  --log-dir             "$LOGDIR"

echo ""
echo "Done: C-GRPO MATH-500"
