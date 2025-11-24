#!/bin/bash
#SBATCH --job-name=ddpm
#SBATCH --partition=gpu
#SBATCH --account=rxf131
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Load PyTorch module
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Install compatible dependencies
pip install --user torchvision==0.12.0 tqdm matplotlib

# Run training
python3 train_cifar10.py --epochs 100 --batch-size 390 --lr 1e-4 --num-workers 8 --device cuda
