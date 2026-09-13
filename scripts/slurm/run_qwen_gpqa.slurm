#!/bin/bash
#SBATCH --job-name=cgrpo_qwen_gpqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_gpqa_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_gpqa_%j.log

# GRPO + conformal prediction training on GPQA-Diamond (graduate-level MCQ)
# Model:   QWEN2.5-7b  (Qwen2.5-7B-Instruct, ~16 GB bf16)
# Dataset: hendrydong/gpqa_diamond_mc  (198 examples, test-only split)
#          4-choice science questions (Physics / Chemistry / Biology)
#          verified by letter matching on (A-D) gold labels
#
# Since GPQA is test-only we randomly partition 198 examples into
# train / cal / eval (reproducible via seed), same as HumanEval.

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
DS_NAME="hendrydong/gpqa_diamond_mc"
DS_CONFIG=""          # no sub-configs
# 198 total: ~80 train + 40 cal + 78 eval (loader partitions test split)
N_TRAIN=80
N_CAL=40
N_EVAL=78
MAX_TOK=1024         # graduate-level questions need room for chain-of-thought
SEED=0
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

DS_SLUG="gpqa_diamond"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " Training: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  n_eval=$N_EVAL  max_new_tokens=$MAX_TOK"
echo " Note: test-only dataset, loader auto-partitions into train/cal/eval"
echo "============================================================"

python run_train.py \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --seed                "$SEED" \
  --n-train             "$N_TRAIN" \
  --n-cal               "$N_CAL" \
  --max-new-tokens      "$MAX_TOK" \
  --log-dir             "$LOGDIR"

CKPT=$(ls -d "$LOGDIR"/*/final 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval (LoRA checkpoint)  --batch-size 1  --k-max 64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal               "$N_CAL" \
  --n-eval              "$N_EVAL" \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)  --batch-size 1  --k-max 64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal               "$N_CAL" \
  --n-eval              "$N_EVAL" \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK" \
  --dynamic-baselines   "$HEURISTICS" \
  --baseline-thresholds "$THRESHOLDS"

echo ""
echo "Qwen2.5-7B + GPQA-Diamond complete."
