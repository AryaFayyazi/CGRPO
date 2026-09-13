#!/bin/bash
#SBATCH --job-name=rb_fixedG
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/fixedG_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/fixedG_%j.log

# ============================================================================
# Fixed-G baselines: fixed-rollout GRPO baselines at group size G.
#
# Trains vanilla GRPO with a FIXED group size G, then runs the same
# post-hoc Pareto evaluation used for the C-GRPO checkpoint, so the
# accuracy-vs-training-compute frontier can be read off directly.
#
# Every hyperparameter below is matched to the C-GRPO GSM8K v3 run
#   runs/conformal_grpo/QWEN2.5-7b_gsm8k_main_v3_seed0/
#     QWEN2.5-7b_deltaauto_seed0_20260419_141159
# so that the ONLY difference between arms is the rollout-allocation rule:
#   model, dataset, n_train, n_eval, seed, max_new_tokens, steps,
#   batch size, lr, beta_kl, LoRA rank/alpha/dropout, verifier.
#
# Invoke with:  sbatch --export=ALL,G=8 scripts/q1q2_fixed_g.slurm
# ============================================================================

set -euo pipefail

if [[ -z "${G:-}" ]]; then
  echo "ERROR: G (group size) not set. Use: sbatch --export=ALL,G=8 ..." >&2
  exit 1
fi

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true

# ---- environment (identical to run_qwen_gsm8k_v3.slurm) -------------------
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

# ---- config: matched to the C-GRPO v3 arm --------------------------------
MODEL="QWEN2.5-7b"
DS_NAME="openai/gsm8k"
DS_CONFIG="main"
N_TRAIN=400          # matched to C-GRPO v3
N_EVAL_TRAIN=200     # cheap in-loop final eval only; real eval is eval_pareto
N_EVAL_PARETO=400    # matched to existing pareto_results_*.json
MAX_TOK=512          # matched to C-GRPO v3
STEPS=300            # matched to C-GRPO v3
SEED=0
# GRPO does not use conformal calibration during training (train_baseline.py:234).
# n_cal here only feeds the diagnostic conf_acc field in state.json, so keep it
# small to avoid charging GRPO for compute its training never uses.
N_CAL_TRAIN=40
N_CAL_PARETO=200     # matched to existing pareto evals so 'conf' rows compare

LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/grpo_G${G}/${MODEL}_gsm8k_seed${SEED}"
mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs

echo ""
echo "============================================================"
echo " Q1/Q2 fixed-G GRPO   G=${G}   steps=${STEPS}   seed=${SEED}"
echo " model=${MODEL}  dataset=${DS_NAME}/${DS_CONFIG}"
echo " n_train=${N_TRAIN}  max_new_tokens=${MAX_TOK}  batch=4"
echo " logdir=${LOGDIR}"
echo " expected training rollouts = ${STEPS} x 4 x ${G} = $((STEPS * 4 * G))"
echo "============================================================"

START_TS=$(date +%s)

python -u run_train_baseline.py \
  --method          grpo \
  --model-key       "$MODEL" \
  --dataset-name    "$DS_NAME" \
  --dataset-config  "$DS_CONFIG" \
  --seed            "$SEED" \
  --n-train         "$N_TRAIN" \
  --n-cal           "$N_CAL_TRAIN" \
  --n-eval          "$N_EVAL_TRAIN" \
  --max-new-tokens  "$MAX_TOK" \
  --steps           "$STEPS" \
  --n-rollouts      "$G" \
  --batch-size      4 \
  --eval-every      999 \
  --ckpt-every      100 \
  --log-dir         "$LOGDIR"

TRAIN_END_TS=$(date +%s)
echo ""
echo ">>> TRAIN_WALLCLOCK_SECONDS G=${G}: $((TRAIN_END_TS - START_TS))"

# ---- Pareto eval on the final checkpoint ---------------------------------
CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $LOGDIR" >&2
  exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "============================================================"
echo " Pareto eval  G=${G}  k-max=32  n_eval=${N_EVAL_PARETO}"
echo "============================================================"

python -u eval_pareto.py \
  --ckpt-dir         "$CKPT" \
  --model-key        "$MODEL" \
  --dataset-name     "$DS_NAME" \
  --dataset-config   "$DS_CONFIG" \
  --n-cal            "$N_CAL_PARETO" \
  --n-eval           "$N_EVAL_PARETO" \
  --k-max            32 \
  --batch-size       4 \
  --max-new-tokens   "$MAX_TOK"

END_TS=$(date +%s)
echo ""
echo ">>> TOTAL_WALLCLOCK_SECONDS G=${G}: $((END_TS - START_TS))"
echo ">>> DONE fixed-G arm G=${G}"
