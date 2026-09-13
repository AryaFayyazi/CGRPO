#!/bin/bash
#SBATCH --job-name=rb_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${CGRPO_ROOT:-$PWD}/scripts/logs/batch_%j.log
#SBATCH --error=${CGRPO_ROOT:-$PWD}/scripts/logs/batch_%j.log

# ============================================================================
# Run several arms CONCURRENTLY on a single GPU.
#
# Rationale: one arm occupies ~17 GB of the 95 GB H100 NVL and leaves the GPU
# at roughly 60% utilisation, so a single arm neither fills the memory nor
# saturates the SMs. The partition sets OverSubscribe=NO, so slurm will not
# place two JOBS on one GPU -- instead we launch several training PROCESSES
# inside one job and let them share the device.
#
# Each arm writes its own stdout to scripts/logs/arm_<name>.log so the
# interleaved output stays readable, and its own run directory under
# runs/scripts/, so concurrent arms never collide.
#
# Scaling is measured, not assumed: SCALING REPORT at the end of this script
# prints each arm's mean seconds/step against the single-arm cost model
# t_step = 15.77 + 0.4541 * rollouts_per_step (R^2 = 0.80, n = 300), so the
# next batch's width can be chosen from evidence.
#
# Invoke with:  sbatch --export=ALL,ARMS="qwen_G8 qwen_G16 splitdelta" \
#                      scripts/parallel_batch.slurm
# ============================================================================

set -uo pipefail   # NOT -e: one failing arm must not kill its siblings

if [[ -z "${ARMS:-}" ]]; then
  echo "ERROR: ARMS not set. Use: sbatch --export=ALL,ARMS='qwen_G8 qwen_G16' ..." >&2
  exit 1
fi

echo "Waiting 15s for GPU to be fully ready..."
sleep 15
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate cgrpo
cd ${CGRPO_ROOT:-$PWD}
mkdir -p scripts/logs

BATCH_START=$(date +%s)
echo ""
echo "============================================================"
echo " PARALLEL BATCH on 1 GPU: ${ARMS}"
echo " started $(date)"
echo "============================================================"

# ---- background a VRAM sampler so OOM risk is quantified, not guessed -----
VRAMLOG="scripts/logs/vram_${SLURM_JOB_ID:-local}.csv"
( echo "ts,used_mib,util_pct"
  while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ',' | sed 's/,$//')"
    sleep 20
  done ) > "$VRAMLOG" 2>/dev/null &
SAMPLER_PID=$!

PIDS=(); NAMES=()
for ARM in $ARMS; do
  LOG="scripts/logs/arm_${ARM}.log"
  echo "  launching ${ARM}  ->  ${LOG}"
  bash scripts/run_arm.sh "$ARM" > "$LOG" 2>&1 &
  PIDS+=($!); NAMES+=("$ARM")
  sleep 90   # stagger model loading so N copies don't allocate simultaneously
done

echo ""
echo "  ${#PIDS[@]} arms running concurrently; waiting..."

FAILED=()
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  [ok]   ${NAMES[$i]}"
  else
    echo "  [FAIL] ${NAMES[$i]} (see scripts/logs/arm_${NAMES[$i]}.log)"
    FAILED+=("${NAMES[$i]}")
  fi
done

kill "$SAMPLER_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo " BATCH WALLCLOCK_SECONDS: $(( $(date +%s) - BATCH_START ))"
[[ ${#FAILED[@]} -gt 0 ]] && echo " FAILED ARMS: ${FAILED[*]}"

echo ""
echo " PEAK VRAM / MEAN UTIL during batch:"
awk -F, 'NR>1 && $2!="" {if($2>m)m=$2; s+=$3; n++} END{
  if(n) printf("   peak %.1f GiB   mean util %.0f%%   (%d samples)\n", m/1024, s/n, n)}' \
  "$VRAMLOG" 2>/dev/null || echo "   (no samples)"

echo ""
echo " SCALING REPORT (measured s/step vs single-arm cost model):"
python - "$ARMS" <<'PYEOF'
import glob, json, os, statistics, sys
# single-arm model fitted on 300 measured steps of the v3 GSM8K run
A, B = 15.77, 0.4541
R = "${CGRPO_ROOT:-$PWD}/runs/ablations"
DIRS = {"qwen_G": "grpo_G", "llama_G": "llama_grpo_G"}
for arm in sys.argv[1].split():
    if arm.startswith("qwen_G"):     root, g = f"{R}/grpo_G{arm[6:]}", int(arm[6:])
    elif arm.startswith("llama_G"):  root, g = f"{R}/llama_grpo_G{arm[7:]}", int(arm[7:])
    elif arm == "llama_cgrpo":       root, g = f"{R}/llama_cgrpo", None
    elif arm == "splitdelta":        root, g = f"{R}/cgrpo_splitdelta", None
    elif arm == "dcal0":             root, g = f"{R}/cgrpo_dcalINF", None
    elif arm.startswith("dcal"):     root, g = f"{R}/cgrpo_dcal{arm[4:]}", None
    elif arm == "q4long":            root, g = f"{R}/cgrpo_long600", None
    else:                            continue
    ev = sorted(glob.glob(f"{root}/**/events.jsonl", recursive=True),
                key=os.path.getmtime)
    if not ev:
        print(f"   {arm:14} no events.jsonl"); continue
    rows = [json.loads(l) for l in open(ev[-1])]
    tr = [r for r in rows if r.get("perf/steps_per_s")]
    if not tr:
        print(f"   {arm:14} no timing rows"); continue
    obs = statistics.median(1.0 / r["perf/steps_per_s"] for r in tr)
    roll = (4 * g if g is not None
            else 4 * statistics.mean(r["train/avg_k_used"] for r in tr
                                     if "train/avg_k_used" in r))
    pred = A + B * roll
    print(f"   {arm:14} {obs:7.1f} s/step observed   {pred:7.1f} predicted alone"
          f"   slowdown {obs/pred:4.2f}x")
print("\n   slowdown ~1.0x => the GPU absorbed the arm for free.")
print("   slowdown ~Nx    => arms are fully serialising; batching gains nothing.")
PYEOF

echo "============================================================"
[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
