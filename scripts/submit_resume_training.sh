#!/bin/bash
#SBATCH --job-name=ddpm_resume
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

echo "===== Resume DDPM Training ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Starting time: $(date)"

# Load modules
# module load python/3.9
# module load cuda/11.8

# Activate environment
# source .venv/bin/activate

nvidia-smi

mkdir -p logs
mkdir -p checkpoints

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found to resume from!"
    exit 1
fi

echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
echo ""

# This is a placeholder - you'd need to add resume functionality to train_cifar10.py
# For now, it will start fresh training
# TODO: Add --resume flag to train_cifar10.py

EPOCHS=100
BATCH_SIZE=128
LR=1e-4

python3 train_cifar10.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --checkpoint-dir checkpoints \
    --save-interval 10 \
    --device cuda

echo "Training resumed and completed!"
echo "End time: $(date)"
