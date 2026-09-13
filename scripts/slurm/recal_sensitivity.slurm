#!/bin/bash
#SBATCH --job-name=r2q5_dcal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q5_dcal_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/r2q5_dcal_%j.log

# ============================================================================
# Sensitivity: sensitivity to the re-calibration period Delta_cal.
#
# The Appendix B ablation reports Delta in {25, 50, 100, inf} but there is no
# script that produces those numbers from run data. This is a controlled sweep:
# everything is identical to the headline GSM8K v3 run except Delta_cal, so the
# accuracy / k_bar / calibration-overhead trade-off is isolated.
#
# Delta_cal = 50 is the existing v3 run and is NOT re-run here.
# DCAL=0 disables re-calibration entirely after the step-0 bootstrap
# (train.py: `if recal_every > 0`), i.e. the Delta_cal = infinity arm.
#
# Invoke with:  sbatch --export=ALL,DCAL=100 scripts/r2q5_recal_sensitivity.slurm
# ============================================================================

set -euo pipefail

if [[ -z "${DCAL:-}" ]]; then
  echo "ERROR: DCAL not set. Use: sbatch --export=ALL,DCAL=100 ..." >&2
  exit 1
fi

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
K_VALUES="2,4,8,16,32"; STEPS=300

TAG="dcal${DCAL}"
[[ "$DCAL" == "0" ]] && TAG="dcalINF"
LOGDIR="${CGRPO_ROOT:-$PWD}/runs/scripts/cgrpo_${TAG}/${MODEL}_gsm8k_seed${SEED}"
mkdir -p "$LOGDIR" ${CGRPO_ROOT:-$PWD}/scripts/logs

echo "============================================================"
echo " study recalibration-period sensitivity   Delta_cal=${DCAL} (${TAG})"
echo " everything else identical to the v3 GSM8K run"
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
  --recalibrate-every "$DCAL" \
  --log-dir           "$LOGDIR"

echo ">>> TRAIN_WALLCLOCK_SECONDS r2q5_${TAG}: $(( $(date +%s) - START_TS ))"

CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint" >&2; exit 1; }

python -u eval_pareto.py \
  --ckpt-dir "$CKPT" --model-key "$MODEL" \
  --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
  --n-cal "$N_CAL" --n-eval "$N_EVAL" \
  --k-max 32 --batch-size 4 --max-new-tokens "$MAX_TOK"

echo ">>> TOTAL_WALLCLOCK_SECONDS r2q5_${TAG}: $(( $(date +%s) - START_TS ))"
echo ">>> DONE study ${TAG}"
