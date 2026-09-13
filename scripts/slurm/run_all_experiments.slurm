#!/bin/bash
#SBATCH --job-name=all_exps
#SBATCH --output=all_exps5.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00          # 14 days
#SBATCH --partition=gpu

# ============================================================================
# Conformal GRPO — multi-dataset experiment runner
#
# Datasets:
#   ChilleD/SVAMP        – math word problems (700 train / 300 test)
#   ag_news              – 4-class news classification (120k train / 7.6k test)
#   commonsense_qa       – 5-way MC commonsense QA (9.7k train / 1.2k VAL)
#                          NOTE: commonsense_qa test set is unlabeled; data.py
#                          automatically uses the validation split instead.
#
# Model: QWEN2.5-7b (Instruct, 7B) – a capable instruct model that follows
#        output-format instructions (output only the letter, etc.).
#
# Per-dataset tuning:
#   SVAMP          : max_new_tokens=256 (needs brief reasoning + ####)
#   ag_news        : max_new_tokens=32  (just the category name)
#   commonsense_qa : max_new_tokens=20  (just the letter A-E)
# ============================================================================

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate cgrpo

# cache directories – user-local since /data subdirs are not writable by the cluster user
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ---- experiment config -------------------------------------------------------
DATASETS=(
  "ChilleD/SVAMP:default"         # math word problems
  "ag_news:default"               # news topic classification (4 classes)
  "commonsense_qa:default"        # commonsense MC QA (5 choices A-E)
)

MODELS=(
  "QWEN2.5-7b"   # Qwen2.5-7B-Instruct
)

SEEDS=(0)

HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

# ---- pre-download datasets ---------------------------------------------------
echo "Pre-downloading datasets..."
for dsconf in "${DATASETS[@]}"; do
  ds="${dsconf%%:*}"
  cfg="${dsconf##*:}"
  echo "  Loading $ds (config=$cfg)..."
  python - <<PYEOF
import os
os.environ['HF_HOME'] = '$HOME/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '$HOME/.cache/huggingface/datasets'
os.environ['HF_HUB_CACHE'] = os.environ['HF_HOME'] + '/hub'
from datasets import load_dataset
_ds, _cfg = "${ds}", "${cfg}"
try:
    load_dataset(_ds, _cfg, cache_dir=os.environ['HF_DATASETS_CACHE'])
    print(f"    {_ds} downloaded OK")
except Exception as e:
    print(f"    WARNING: {_ds}:{_cfg} -> {e}")
PYEOF
done

# ---- main training + eval loop -----------------------------------------------
for model in "${MODELS[@]}"; do
  for dsconf in "${DATASETS[@]}"; do
    ds="${dsconf%%:*}"
    cfg="${dsconf##*:}"
    for seed in "${SEEDS[@]}"; do

      echo ""
      echo "================================================================"
      echo " MODEL=$model  DATASET=$ds ($cfg)  SEED=$seed"
      echo "================================================================"

      LOGDIR="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/${model}_${ds##*/}_${cfg}_seed${seed}"
      mkdir -p "${LOGDIR}"

      # ---- per-dataset hyperparameters ----------------------------------------
      # max_new_tokens: classification needs only a few tokens; math needs more.
      if   [[ "$ds" == *"svamp"* ]];          then MAX_TOK=256; N_TRAIN=400; N_CAL=200
      elif [[ "$ds" == *"ag_news"* ]];        then MAX_TOK=32;  N_TRAIN=4000; N_CAL=400
      elif [[ "$ds" == *"commonsense_qa"* ]]; then MAX_TOK=20;  N_TRAIN=4000; N_CAL=400
      else                                         MAX_TOK=256; N_TRAIN=4000; N_CAL=400
      fi

      # ---- 1. Train LoRA adapter -----------------------------------------------
      echo "-- Training LoRA adapter (max_new_tokens=$MAX_TOK) --"
      python run_train.py \
        --model-key       "$model" \
        --dataset-name    "$ds" \
        --dataset-config  "$cfg" \
        --seed            $seed \
        --log-dir         "$LOGDIR" \
        --max-new-tokens  $MAX_TOK \
        --n-train         $N_TRAIN \
        --n-cal           $N_CAL

      # ---- 2. Pareto eval with LoRA --------------------------------------------
      CKPT=$(ls -d "$LOGDIR"/*/final 2>/dev/null | sort | tail -1)
      if [ -z "$CKPT" ]; then
          echo "WARNING: no checkpoint found under $LOGDIR — skipping eval"
      else
          echo "-- LoRA evaluation (ckpt=$CKPT) --"
          python eval_pareto.py \
            --ckpt-dir          "$CKPT" \
            --model-key         "$model" \
            --dataset-name      "$ds" \
            --dataset-config    "$cfg" \
            --n-cal 400 --n-eval 400 \
            --k-max 64 \
            --max-new-tokens    $MAX_TOK \
            --dynamic-baselines "$HEURISTICS" \
            --baseline-thresholds "$THRESHOLDS"

          # ---- 3. Pareto eval on base model (no LoRA) -------------------------
          echo "-- Base-model evaluation (no LoRA) --"
          python eval_pareto.py \
            --ckpt-dir          "$CKPT" \
            --no-lora \
            --model-key         "$model" \
            --dataset-name      "$ds" \
            --dataset-config    "$cfg" \
            --n-cal 400 --n-eval 400 \
            --k-max 64 \
            --max-new-tokens    $MAX_TOK \
            --dynamic-baselines "$HEURISTICS" \
            --baseline-thresholds "$THRESHOLDS"
      fi

    done   # seeds
  done     # datasets
done       # models

echo ""
echo "All experiments complete."
