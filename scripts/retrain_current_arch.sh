#!/bin/bash
#SBATCH --job-name=retrain_ddpm
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=retrain_ddpm_%j.out
#SBATCH --error=retrain_ddpm_%j.err

# Quick retraining with current UNet architecture for ablation studies
# 50 epochs should be sufficient for ablation comparisons

module purge
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

echo "Starting DDPM Retraining"
echo "========================================"
date
echo ""
echo "Working directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Backup old checkpoints
echo "Backing up old checkpoints..."
mkdir -p checkpoints_old
mv checkpoints/*.pt checkpoints_old/ 2>/dev/null || true

# Train with current architecture
python examples/train_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --timesteps 1000 \
    --base-ch 64 \
    --num-workers 4 \
    --checkpoint-dir checkpoints \
    --save-interval 10 \
    --device cuda

echo ""
echo "========================================"
echo "Training complete!"
echo "New checkpoint: checkpoints/final_model.pt"
date
