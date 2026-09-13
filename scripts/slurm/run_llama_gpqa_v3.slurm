#!/bin/bash
#SBATCH --job-name=cgrpo_llama_gpqa_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/llama_gpqa_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/llama_gpqa_v3_%j.log

# C-GRPO training on GPQA-Diamond — v3: all root-cause fixes
# Model: Llama-3.1-8B-Instruct

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

MODEL="llama3.1-8b"
DS_NAME="hendrydong/gpqa_diamond_mc"
DS_CONFIG=""
N_TRAIN=80
N_CAL=40
N_EVAL=78
MAX_TOK=512         # reduced from 1024 to avoid OOM on Llama 3.1-8B (GPQA is MCQ)
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto

DS_SLUG="gpqa_diamond"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
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
  --max-new-tokens      "$MAX_TOK" \
  --k-values            "$K_VALUES" \
  --delta               "$DELTA" \
  --batch-size          2 \
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
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
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
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
  --k-max 64 \
  --batch-size 1 \
  --max-new-tokens      "$MAX_TOK"

# ── Paper figures ─────────────────────────────────────────────────────────────
PARETO_SLUG="hendrydong_gpqa_diamond_mc"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "gpqa" \
  --pareto-json  "$CGRPO_PARETO" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 GPQA-Diamond (Llama-3.1-8B)"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset gpqa"
