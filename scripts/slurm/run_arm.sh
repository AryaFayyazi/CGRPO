#!/bin/bash
# ============================================================================
# Run one experiment arm (train + pareto eval) end to end.
#
# Called by parallel_batch.slurm, which launches several of these concurrently
# inside a SINGLE slurm job so they share one GPU (the partition has
# OverSubscribe=NO, so slurm itself will not co-schedule jobs on a GPU).
#
# Usage:  bash scripts/run_arm.sh <arm>
#
# Arms:
#   qwen_G<N>    fixed-G GRPO, Qwen2.5-7B          (N = 2,4,8,16,32,64)
#   llama_G<N>   fixed-G GRPO, Llama-3.1-8B
#   llama_cgrpo  C-GRPO, Llama-3.1-8B              (study stronger base)
#   splitdelta   C-GRPO + --split-delta, Qwen      (study coverage fix)
#   dcal<N>      C-GRPO with recalibrate_every=N   (study; N=0 means never)
#   q4long       C-GRPO 600 steps, Qwen            (study long horizon)
#
# Every arm is matched to the headline GSM8K v3 configuration except for the
# single factor under study.
# ============================================================================

set -euo pipefail
ARM="${1:?usage: run_arm.sh <arm>}"

cd ${CGRPO_ROOT:-$PWD}
export HF_DATASETS_CACHE="hf_cache/datasets"
export HF_HOME="hf_cache/hub"
export HF_HUB_CACHE="hf_cache/hub"
export TRANSFORMERS_CACHE="hf_cache/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DS_NAME="openai/gsm8k"; DS_CONFIG="main"
N_TRAIN=400; N_EVAL=400; MAX_TOK=512; SEED=0; STEPS=300
K_VALUES="2,4,8,16,32"
N_CAL_CGRPO=200      # matched to v3
N_CAL_GRPO=40        # GRPO never uses qhats in training (train_baseline.py)
R="${CGRPO_ROOT:-$PWD}/runs/ablations"

MODEL="QWEN2.5-7b"; MODE=""; G=""; DCAL=50; SPLIT=""; LOGDIR=""; NCAL=""

case "$ARM" in
  qwen_G*)   MODE=grpo;  MODEL="QWEN2.5-7b";  G="${ARM#qwen_G}";  LOGDIR="$R/grpo_G${G}" ;;
  llama_G*)  MODE=grpo;  MODEL="llama3.1-8b"; G="${ARM#llama_G}"; LOGDIR="$R/llama_grpo_G${G}" ;;
  llama_cgrpo) MODE=cgrpo; MODEL="llama3.1-8b"; LOGDIR="$R/llama_cgrpo" ;;
  splitdelta)  MODE=cgrpo; SPLIT="--split-delta"; LOGDIR="$R/cgrpo_splitdelta" ;;
  dcal0)       MODE=cgrpo; DCAL=0;   LOGDIR="$R/cgrpo_dcalINF" ;;
  dcal*)       MODE=cgrpo; DCAL="${ARM#dcal}"; LOGDIR="$R/cgrpo_dcal${DCAL}" ;;
  q4long)      MODE=cgrpo; STEPS=500; LOGDIR="$R/cgrpo_long500" ;;
  *) echo "ERROR: unknown arm '$ARM'" >&2; exit 2 ;;
esac

LOGDIR="${LOGDIR}/${MODEL}_gsm8k_seed${SEED}"
mkdir -p "$LOGDIR"
START_TS=$(date +%s)

echo "=== ARM ${ARM}  model=${MODEL}  mode=${MODE}  steps=${STEPS} ==="

if [[ "$MODE" == "grpo" ]]; then
  NCAL="$N_CAL_GRPO"
  echo "    fixed group size G=${G};  expected rollouts = ${STEPS} x 4 x ${G}"
  python -u run_train_baseline.py \
    --method grpo --model-key "$MODEL" \
    --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
    --seed "$SEED" --n-train "$N_TRAIN" --n-cal "$NCAL" --n-eval 200 \
    --max-new-tokens "$MAX_TOK" --steps "$STEPS" --n-rollouts "$G" \
    --batch-size 4 --eval-every 999 --ckpt-every 100 \
    --log-dir "$LOGDIR"
  NCAL_EVAL=200
else
  NCAL="$N_CAL_CGRPO"
  EVAL_EVERY=50
  [[ "$ARM" == "q4long" ]] && EVAL_EVERY=100
  echo "    C-GRPO  K=${K_VALUES}  n_cal=${NCAL}  recal_every=${DCAL} ${SPLIT}"
  python -u run_train.py \
    --model-key "$MODEL" \
    --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
    --seed "$SEED" --n-train "$N_TRAIN" --n-cal "$NCAL" \
    --n-eval $([[ "$ARM" == "q4long" ]] && echo 200 || echo "$N_EVAL") \
    --max-new-tokens "$MAX_TOK" --steps "$STEPS" --k-values "$K_VALUES" \
    --delta auto --batch-size 4 \
    --eval-every "$EVAL_EVERY" --recalibrate-every "$DCAL" \
    $SPLIT \
    --log-dir "$LOGDIR"
  NCAL_EVAL="$NCAL"
fi

echo ">>> TRAIN_WALLCLOCK_SECONDS ${ARM}: $(( $(date +%s) - START_TS ))"

CKPT=$(ls -dt "$LOGDIR"/*/final 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && CKPT=$(ls -dt "$LOGDIR"/*/ckpt_step_* 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint under $LOGDIR" >&2; exit 1; }
echo "    checkpoint: $CKPT"

python -u eval_pareto.py \
  --ckpt-dir "$CKPT" --model-key "$MODEL" \
  --dataset-name "$DS_NAME" --dataset-config "$DS_CONFIG" \
  --n-cal "$NCAL_EVAL" --n-eval "$N_EVAL" \
  --k-max 32 --batch-size 4 --max-new-tokens "$MAX_TOK"

echo ">>> TOTAL_WALLCLOCK_SECONDS ${ARM}: $(( $(date +%s) - START_TS ))"
echo ">>> DONE ${ARM}"
