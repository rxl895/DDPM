#!/bin/bash
#SBATCH --job-name=latent_diff
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=latent_train_%j.out
#SBATCH --error=latent_train_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Train latent diffusion model
# Stage 1: Train autoencoder (20 epochs, ~20 min)
# Stage 2: Train diffusion in latent space (100 epochs, ~3-4 hours)
python3 train_latent_diffusion.py \
    --ae-epochs 20 \
    --diffusion-epochs 100 \
    --batch-size 390 \
    --lr-ae 1e-3 \
    --lr-diffusion 1e-4 \
    --device cuda

echo "Latent diffusion training complete!"
