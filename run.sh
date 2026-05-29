#!/bin/bash
#SBATCH --job-name=dfew_probe
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/dfew_probe_%j.out
#SBATCH --error=logs/dfew_probe_%j.err

mkdir -p logs

source activate vlm_xai

echo "======================================"
echo "DFEW linear probe training"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "======================================"

python 01_extract_hidden_states_and_linear_probe.py \
  --run_mode extract_and_probe \
  --shard_index 0 \
  --shard_count 1

echo "======================================"
echo "End time: $(date)"
echo "======================================"
