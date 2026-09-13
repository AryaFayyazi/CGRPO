#!/bin/bash
#SBATCH --job-name=r2q1_split
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q1_split_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q1_split_%j.log

# ============================================================================
# Coverage audit: restore split-conformal exactness across re-calibration.
#
# DIAGNOSIS (analysis/results/r2q1_coverage_over_training.csv): in the shipped
# configuration, delta_auto is selected from the SAME calibration scores that
# then produce q_hat. The calibration data is used twice, which voids
# split-conformal exactness. Empirically 42/72 audited checkpoints under-cover
# and 21/72 do so by more than 2 binomial SE.
#
# FIX under test: --split-delta partitions D_cal into two disjoint halves --
# half A selects delta_auto, half B computes q_hat. delta is then independent
# of the scores producing q_hat, so Theorem 2 applies exactly at each
# checkpoint (with effective n = n_cal/2, i.e. slack 2/(n_cal+2) = 1.0%).
#
# Everything else is identical to the headline GSM8K v3 run, so the coverage
# trajectories are directly comparable checkpoint-by-checkpoint.
# ============================================================================

set -euo pipefail

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

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
DS_NAME="openai/gsm8k"; DS_CONFIG="main"
N_TRAIN=400; N_CAL=200; N_EVAL=400; MAX_TOK=512; SEED=0
K_VALUES="2,4,8,16,32"; STEPS=300; RECAL=50

LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/cgrpo_splitdelta/${MODEL}_gsm8k_seed${SEED}"
mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs

echo "============================================================"
echo " study split-delta C-GRPO   n_cal=${N_CAL} (100 select delta / 100 calibrate qhat)"
echo " steps=${STEPS}  recal_every=${RECAL}  K=${K_VALUES}"
echo "============================================================"

START_TS=$(date +%s)

python -u run_train.py \
  --model-key         "$MODEL" \
  --dataset-name      "$DS_NAME" \
  --dataset-config    "$DS_CONFIG" \
  --seed              "$SEED" \
  --n-train           "$N_TRAIN" \
  --n-cal             "$N_CAL" \
  --n-eval            "$N_EVAL" \
  --max-new-tokens    "$MAX_TOK" \
  --steps             "$STEPS" \
  --k-values          "$K_VALUES" \
  --delta             auto \
  --batch-size        4 \
  --eval-every        50 \
  --recalibrate-every "$RECAL" \
  --split-delta \
  --log-dir           "$LOGDIR"

echo ">>> TRAIN_WALLCLOCK_SECONDS r2q1_splitdelta: $(( $(date +%s) - START_TS ))"

CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint" >&2; exit 1; }

python -u eval_pareto.py \
  --ckpt-dir "$CKPT" --model-key "$MODEL" \
  --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
  --k-max 32 --batch-size 4 --max-new-tokens "$MAX_TOK"

echo ">>> TOTAL_WALLCLOCK_SECONDS r2q1_splitdelta: $(( $(date +%s) - START_TS ))"
echo ">>> DONE study split-delta"
