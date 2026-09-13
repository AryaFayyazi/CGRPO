#!/bin/bash
#SBATCH --job-name=cgrpo_gsm8k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/cgrpo_gsm8k_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/cgrpo_gsm8k_%j.log

# Conformal GRPO on GSM8K — matched parameters for fair baseline comparison
# MAX_TOK=256, N_EVAL=400 to match corrected baselines (job 609)

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
N_EVAL=400   # match baseline re-run (job 609)
MAX_TOK=256  # match baseline re-run (job 609)
STEPS=300
SEED=0

# compare_methods.py looks for runs/conformal_grpo/QWEN2.5-7b_gsm8k_main_seed0/
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_gsm8k_main_seed${SEED}"

echo ""
echo "============================================================"
echo " Conformal GRPO: $DS_NAME (config=$DS_CONFIG)"
echo " model=$MODEL | n_train=$N_TRAIN | n_cal=$N_CAL | n_eval=$N_EVAL"
echo " max_new_tokens=$MAX_TOK | steps=$STEPS | seed=$SEED"
echo " logdir=$LOGDIR"
echo "============================================================"
echo ""

python -u run_train.py \
  --model-key        "$MODEL" \
  --dataset-name     "$DS_NAME" \
  --dataset-config   "$DS_CONFIG" \
  --n-train          "$N_TRAIN" \
  --n-cal            "$N_CAL" \
  --n-eval           "$N_EVAL" \
  --max-new-tokens   "$MAX_TOK" \
  --steps            "$STEPS" \
  --seed             "$SEED" \
  --log-dir          "$LOGDIR"

echo ""
echo "Done: Conformal GRPO on gsm8k"
echo "============================================================"
