#!/bin/bash
#SBATCH --job-name=ddim_sample
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=ddim_%j.out
#SBATCH --error=ddim_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Generate samples with DDIM (50 steps instead of 1000)
echo "Running DDIM with 50 steps..."
python3 demo_ddim.py --checkpoint checkpoints/final_model.pt --num-samples 64 --ddim-steps 50 --eta 0.0 --device cuda

# Also try with 100 steps for comparison
echo "Running DDIM with 100 steps..."
python3 demo_ddim.py --checkpoint checkpoints/final_model.pt --num-samples 64 --ddim-steps 100 --eta 0.0 --device cuda --output-dir samples_ddim_100
