#!/bin/bash
# ============================================================================
# Revised single-GPU plan: FOUR batches of TWO arms each.
#
# WHY TWO AND NOT THREE. Batch 1 ran three arms and was measured, not guessed:
#     qwen_G8     64.2 s/step observed vs 30.3 solo  -> 2.12x slowdown
#     qwen_G16    85.4 s/step observed vs 44.8 solo  -> 1.90x slowdown
#     splitdelta  0.88 gen/s observed vs 4.17 solo   -> 4.72x slowdown
#     aggregate throughput = 1.21x a single arm
# i.e. three-way packing bought 21%, not 200%: one arm already saturates the
# SMs even though it uses only ~17 GB of the 93 GB. Three arms also exhausted
# VRAM -- one process ballooned to 49.9 GB under device_map="auto" (it sizes
# max_memory from free VRAM at load time, so whichever arm loads first claims
# the largest cap) and qwen_G8 died with CUDA OOM at step 6.
#
# Two arms fit the memory (~45 + ~33 GB, 15 GB headroom) and should contend
# less. The scaling report at the end of each batch keeps measuring this.
#
# SCOPE. Capacity is ~44 h wall x 1.2-1.3x = ~53-57 GPU-hours of solo work.
# The arms below total ~46 h solo. Dropped for budget, with measured costs:
#     qwen_G4 1.9 h   qwen_G64 12.3 h   qwen_G128 20.7 h
#     dcal25 10.2 h   dcal0 3.7 h       llama_G8 5.0 h
# q4long was trimmed 600 -> 500 steps (still 67% beyond the paper's horizon).
#
# Usage:  bash scripts/submit_batches2.sh [--dry-run]
# ============================================================================

set -euo pipefail
cd ${CGRPO_ROOT:-$PWD}
mkdir -p scripts/logs

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

submit() {
  local desc="$1"; shift
  if [[ $DRY -eq 1 ]]; then
    echo "[dry-run] $desc :: sbatch $*" >&2
    echo "$((RANDOM + 400000))"
    return
  fi
  sbatch --parsable "$@"
}

PREV=""; CHAIN=()
batch() {
  local desc="$1" arms="$2" name="$3"
  local dep=()
  [[ -n "$PREV" ]] && dep=(--dependency=afterany:"$PREV")
  local jid
  jid=$(submit "$desc" "${dep[@]}" --export=ALL,ARMS="$arms" \
        --job-name="$name" scripts/parallel_batch.slurm)
  echo "  ${desc}"
  echo "      arms: ${arms}"
  echo "      job:  ${jid}"
  PREV="$jid"; CHAIN+=("$jid  ${desc}")
}

echo "Submitting revised 2-arm batched plan..."
echo ""

# G8 died of OOM and splitdelta was cut short in batch 1; both restart here.
# They are the two highest-value arms (G=8 anchors the frontier; split-delta is
# R2's central theoretical concern), so they go first.
batch "Batch A  (study coverage fix + G=8)" \
      "splitdelta qwen_G8" rb_a

# G16 is re-run from scratch rather than resumed: batch 1's copy trained under
# 3-way contention for ~20 min, and a clean arm is worth more than 20 minutes.
batch "Batch B  (G=16 + G=32, the paper's claimed baseline)" \
      "qwen_G16 qwen_G32" rb_b

batch "Batch C  (study long horizon + study sensitivity)" \
      "q4long dcal100" rb_c

batch "Batch D  (study stronger base)" \
      "llama_cgrpo llama_G16" rb_d

echo ""
echo "Chain:"
for c in "${CHAIN[@]}"; do echo "   $c"; done
echo ""
echo "Monitor:   squeue -u \$USER ; tail -f scripts/logs/batch_*.log"
echo "Per arm:   tail -f scripts/logs/arm_<name>.log"
echo "Collect:   python scripts/collect_results.py ; python scripts/collect_r2_results.py"
