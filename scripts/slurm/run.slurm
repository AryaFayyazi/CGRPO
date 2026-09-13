#!/usr/bin/env bash

# Slurm configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:3
#SBATCH --output=output.log
#SBATCH --time=05-00:00:00

# PyTorch memory optimization - prevents fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Do NOT set CUDA_LAUNCH_BLOCKING=1 in production — it serializes GPU ops and kills performance
# export CUDA_LAUNCH_BLOCKING=1

# Activate conda environment with PyTorch
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate cgrpo

# Debugging: Check GPU visibility and allocation
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Set up HF cache directories
mkdir -p ./hf_cache/datasets ./hf_cache/hub
export HF_HOME=./hf_cache
export HF_DATASETS_CACHE=./hf_cache/datasets
export TRANSFORMERS_CACHE=./hf_cache/hub

# For the paper‑scale sweep over datasets/models you can run the
# companion script `run_all_experiments.slurm` which will train and
# evaluate every combination (see that file for details).
# Otherwise run the training script below for a single config.
python -u run_train.py
# python main.py --ablation gsm8k_category --outdir extra_plots
