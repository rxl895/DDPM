#!/bin/bash
#SBATCH --job-name=ddpm_eval
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Install scipy if needed
pip install --user scipy

# Run comprehensive evaluation
python3 evaluate.py --checkpoint checkpoints/final_model.pt --num-samples 1000 --device cuda --output evaluation_results.txt

echo "Evaluation complete! Results saved to evaluation_results.txt"
