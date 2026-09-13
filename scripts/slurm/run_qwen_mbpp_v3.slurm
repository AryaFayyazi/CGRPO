#!/bin/bash
#SBATCH --job-name=cgrpo_mbpp_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_mbpp_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_mbpp_v3_%j.log

# C-GRPO training on MBPP — v3: all root-cause fixes for dynamic-k
#
# Fixes applied (same as HumanEval v3):
#   BUG 1 — delta_max=0.50 cap raised to 0.95 (conformal.py)
#   BUG 2 — exec-based k-selection in training & evaluate() loops (train.py)
#   BUG 3 — recalibration now passes auto_delta=True (train.py)
#   BUG 4 — skip qhat>=1.0 levels in k-selection loop (train.py)
#   BUG 5 — recalibration uses shared_ref=False → trained policy (train.py)
#   CONFIG — recalibrate_every=50 (config.py)

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
DS_NAME="google-research-datasets/mbpp"
DS_CONFIG="full"
N_TRAIN=270
N_CAL=40
MAX_TOK=512
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto   # δ = (1 - pass@k_max) + 0.05, uncapped up to 0.95

DS_SLUG="${DS_NAME##*/}"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_${DS_CONFIG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME ($DS_CONFIG)  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  max_new_tokens=$MAX_TOK"
echo " Verification: code execution (subprocess, 10s timeout)"
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
echo " Pareto eval (LoRA checkpoint)  k-max=64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal "$N_CAL" --n-eval 80 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

echo ""
echo "============================================================"
echo " Pareto eval (base model, no LoRA)  k-max=64"
echo "============================================================"
python eval_pareto.py \
  --ckpt-dir            "$CKPT" \
  --no-lora \
  --model-key           "$MODEL" \
  --dataset-name        "$DS_NAME" \
  --dataset-config      "$DS_CONFIG" \
  --n-cal "$N_CAL" --n-eval 80 \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

# ── Paper figures ─────────────────────────────────────────────────────────────
PARETO_SLUG="google-research-datasets_mbpp"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "mbpp" \
  --pareto-json  "$CGRPO_PARETO" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 MBPP"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset mbpp"
