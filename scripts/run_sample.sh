#!/bin/bash
#SBATCH --job-name=ddpm_sample
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=sample_%j.out
#SBATCH --error=sample_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Generate samples from trained model
python3 demo_sampling.py --checkpoint checkpoints/final_model.pt --num-samples 64 --device cuda
