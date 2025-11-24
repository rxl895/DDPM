#!/bin/bash
#SBATCH --job-name=latent_sample
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=latent_sample_%j.out
#SBATCH --error=latent_sample_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Generate samples from latent diffusion
python3 demo_latent_diffusion.py \
  --ae-checkpoint checkpoints/autoencoder.pt \
  --diffusion-checkpoint checkpoints/final_model.pt \
  --num-samples 64 \
  --ddim-steps 50 \
  --device cuda \
  --output-dir samples_latent

echo "Latent diffusion sampling complete!"
