#!/bin/bash
#SBATCH --job-name=cgrpo_humaneval_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_humaneval_v2_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_humaneval_v2_%j.log

# C-GRPO training on HumanEval — v2 with activated conformal signal
#
# Key changes vs v1 (job 614):
#   k_values: (2,4,8,16) → (2,4,8,16,32)   [more budget per problem]
#   delta:    0.1        → 0.25             [75% coverage target]
#
# Rationale: HumanEval has n_cal=40 small cal set. At delta=0.1 the
# conformal quantile is the 37th/40 score. Even 4 unsolvable problems push
# qhat to 1.0. During training the model starts at ~40% greedy pass@1,
# so a meaningful fraction of cal problems have 0 passing runs at k≤16.
# delta=0.25 + k=32 requires only 75% solvable (tolerates up to 10/40 hard
# problems), which is achievable and activates the efficiency signal.

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
DS_NAME="openai/openai_humaneval"
DS_CONFIG=""
N_TRAIN=80
N_CAL=40
MAX_TOK=512
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto   # auto-selected from cal scores: δ = (1 - pass@k_max) + 0.05
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

DS_SLUG="${DS_NAME##*/}"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v2: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (75% coverage target)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  max_new_tokens=$MAX_TOK"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
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
  --n-cal 40 --n-eval 44 \
  --k-max 32 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

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
  --n-cal 40 --n-eval 44 \
  --k-max 32 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "Done: C-GRPO v2 HumanEval"
