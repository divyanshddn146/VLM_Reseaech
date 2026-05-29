#!/bin/bash
#SBATCH --job-name=dfew_extract
#SBATCH --partition=a100
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/dfew_extract_%A_%a.out
#SBATCH --error=logs/dfew_extract_%A_%a.err

mkdir -p logs

source activate vlm_xai

echo "======================================"
echo "DFEW hidden-state extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "======================================"

python 01_extract_hidden_states_and_linear_probe.py \
  --run_mode extract_only \
  --shard_index ${SLURM_ARRAY_TASK_ID} \
  --shard_count 3 \
  --force_reextract

echo "======================================"
echo "End time: $(date)"
echo "======================================"
