#!/usr/bin/env bash

# Slurm configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=pareto_output.log
#SBATCH --time=01-00:00:00
#SBATCH --job-name=pareto_eval

# PyTorch memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate conda environment
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

# Run the Pareto evaluation script
python -u eval_pareto.py
