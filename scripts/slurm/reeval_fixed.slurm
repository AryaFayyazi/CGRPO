#!/bin/bash
#SBATCH --job-name=cgrpo_reeval
#SBATCH --output=reeval_fixed.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=14-00:00:00          # 14 days
#SBATCH --partition=gpu

set -e
cd ${CGRPO_ROOT:-$PWD}
conda run -n cgrpo --no-capture-output bash << 'INNER'
set -e
cd ${CGRPO_ROOT:-$PWD}

MODEL="QWEN2.5-7b"
HEURISTICS="softmax,entropy"
THRESHOLDS="0.5,0.7"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific settings: ckpt_dir, dataset_name, config, max_new_tokens
# ─────────────────────────────────────────────────────────────────────────────
declare -A CKPT_MAP
CKPT_MAP["ag_news"]="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/QWEN2.5-7b_ag_news_default_seed0/QWEN2.5-7b_delta0.1_seed0_20260320_184210/final"
CKPT_MAP["commonsense_qa"]="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/QWEN2.5-7b_commonsense_qa_default_seed0/QWEN2.5-7b_delta0.1_seed0_20260320_220549/final"
CKPT_MAP["ChilleD/SVAMP"]="${CGRPO_ROOT:-$PWD}/runs/conformal_grpo/QWEN2.5-7b_SVAMP_default_seed0/QWEN2.5-7b_delta0.1_seed0_20260320_125606/final"

declare -A DSCFG_MAP
DSCFG_MAP["ag_news"]="default"
DSCFG_MAP["commonsense_qa"]="default"
DSCFG_MAP["ChilleD/SVAMP"]="default"

declare -A MAXTOK_MAP
MAXTOK_MAP["ag_news"]="32"
MAXTOK_MAP["commonsense_qa"]="20"
MAXTOK_MAP["ChilleD/SVAMP"]="256"

for ds in "ag_news" "commonsense_qa" "ChilleD/SVAMP"; do
  cfg="${DSCFG_MAP[$ds]}"
  CKPT="${CKPT_MAP[$ds]}"
  MAX_TOK="${MAXTOK_MAP[$ds]}"

  echo ""
  echo "========================================================================"
  echo "Re-evaluating: $ds  (max_new_tokens=$MAX_TOK)"
  echo "========================================================================"

  echo "-- LoRA evaluation --"
  python eval_pareto.py \
    --ckpt-dir          "$CKPT" \
    --model-key         "$MODEL" \
    --dataset-name      "$ds" \
    --dataset-config    "$cfg" \
    --n-cal 400 --n-eval 400 \
    --k-max 64 \
    --max-new-tokens    "$MAX_TOK" \
    --dynamic-baselines "$HEURISTICS" \
    --baseline-thresholds "$THRESHOLDS"

  echo "-- Base-model evaluation (no LoRA) --"
  python eval_pareto.py \
    --ckpt-dir          "$CKPT" \
    --no-lora \
    --model-key         "$MODEL" \
    --dataset-name      "$ds" \
    --dataset-config    "$cfg" \
    --n-cal 400 --n-eval 400 \
    --k-max 64 \
    --max-new-tokens    "$MAX_TOK" \
    --dynamic-baselines "$HEURISTICS" \
    --baseline-thresholds "$THRESHOLDS"

done

echo ""
echo "All re-evaluations complete."
INNER
