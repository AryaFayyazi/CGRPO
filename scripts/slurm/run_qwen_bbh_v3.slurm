#!/bin/bash
#SBATCH --job-name=cgrpo_bbh_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/qwen_bbh_v3_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/qwen_bbh_v3_%j.log

# C-GRPO training on BIG-Bench Hard (6-task mix) — v3
#
# Dataset: lukaemon/bbh  config=mix6  (6 subtasks concatenated)
#   - logical_deduction_five_objects  (250 ex, MCQ A-E)
#   - date_understanding              (250 ex, MCQ A-E)
#   - boolean_expressions             (250 ex, True/False)
#   - causal_judgement                (187 ex, Yes/No)
#   - multistep_arithmetic_two        (250 ex, integer)
#   - tracking_shuffled_objects_3     (250 ex, MCQ A-C)
#   Total: 1437 examples  →  train=400  cal=200  eval=300
#
# Covers three reasoning types: logical deduction, symbolic reasoning, temporal.
# Complements math (GSM8K/MATH-500) and code (HumanEval/MBPP) for domain breadth.
#
# All v3 fixes applied (shared_ref=False recal, qhat>=1.0 skip, delta_max=0.95,
# auto-delta, recalibrate_every=50, chunked generation).

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
DS_NAME="lukaemon/bbh"
DS_CONFIG="mix6"
N_TRAIN=400
N_CAL=200
N_EVAL=300
MAX_TOK=512         # BBH needs room for chain-of-thought reasoning
SEED=0
K_VALUES="2,4,8,16,32"
DELTA=auto

DS_SLUG="bbh_mix6"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${MODEL}_${DS_SLUG}_seed${SEED}"
mkdir -p "${LOGDIR}"

echo ""
echo "============================================================"
echo " C-GRPO v3: $MODEL  |  $DS_NAME ($DS_CONFIG)  |  seed=$SEED"
echo " k_values=$K_VALUES  delta=$DELTA  (auto from base model)"
echo " logdir=$LOGDIR"
echo " n_train=$N_TRAIN  n_cal=$N_CAL  n_eval=$N_EVAL  max_new_tokens=$MAX_TOK"
echo " Tasks: logical_deduction_5obj | date_understanding | boolean_expressions"
echo "        causal_judgement | multistep_arithmetic | tracking_shuffled_3obj"
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
PARETO_SLUG="lukaemon_bbh"
CGRPO_PARETO="${CKPT}/pareto_results_${PARETO_SLUG}.json"

echo ""
echo "============================================================"
echo " Generating paper figures"
echo "============================================================"
python plot_paper.py \
  --run-dir      "$LOGDIR" \
  --dataset      "bbh" \
  --pareto-json  "$CGRPO_PARETO" \
  --out-dir      "${LOGDIR}/figures" \
  --fmt          pdf || echo "WARNING: plot_paper.py failed, continuing"

echo ""
echo "Done: C-GRPO v3 BBH-mix6"
echo "  Figures: ${LOGDIR}/figures/"
echo "  → Run: python compare_methods.py --dataset bbh"
