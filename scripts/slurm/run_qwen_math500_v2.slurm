#!/bin/bash
#SBATCH --job-name=cgrpo_math500_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_math500_v2_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_math500_v2_%j.log

# C-GRPO training on MATH-500 — v2 with activated conformal signal
#
# Key changes vs v1 (job 616):
#   k_values: (2,4,8,16) → (2,4,8,16,32)   [more budget per problem]
#   delta:    0.1        → 0.25             [75% coverage target]
#
# Rationale: at delta=0.1 the qhat requires 91% of 100 cal problems to be
# solvable at k=16. With Qwen2.5-7B-Instruct, MATH-500 pass@16 ≈ 74.7%
# so ~25% of cal problems are genuinely unsolvable — qhat stays at 1.0 and
# the efficiency signal never fires.  delta=0.25 + k=32 requires only 75%
# solvable (pass@32 is comfortably above that), so qhat[32] drops below 1.0
# from step 0 and C-GRPO's adaptive-k mechanism is actually active.
# A 75% coverage guarantee is also mathematically honest for this model/task.

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
DS_CONFIG=""
N_TRAIN=200
N_CAL=100
N_EVAL=200
MAX_TOK=1024
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto   # auto-selected from cal scores: δ = (1 - pass@k_max) + 0.05

DS_SLUG="math500"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v2: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (75% coverage target)"
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
  --k-values            "$K_VALUES" \
  --delta               "$DELTA" \
  --log-dir             "$LOGDIR"

# ---- eval ---------------------------------------------------------------
CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  k-max=32"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal 100 --n-eval 200 \
  --k-max 32 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)  k-max=32"
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
  --max-new-tokens      "$MAX_TOK"

echo ""
echo "Done: C-GRPO v2 MATH-500"
