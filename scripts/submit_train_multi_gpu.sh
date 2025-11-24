#!/bin/bash
#SBATCH --job-name=ddpm_train_multigpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00                # 48 hours for longer training
#SBATCH --partition=gpu                # Partition name
#SBATCH --gres=gpu:4                   # Request 4 GPUs
#SBATCH --cpus-per-task=32             # More CPU cores for multi-GPU
#SBATCH --mem=128G                     # More memory
#SBATCH --ntasks=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6
# module load nccl/2.12

# Activate environment
# source .venv/bin/activate

# Print GPU info
echo "GPU information:"
nvidia-smi

mkdir -p logs
mkdir -p checkpoints

# Training parameters for multi-GPU
EPOCHS=200
BATCH_SIZE=512                 # Larger batch for multi-GPU
LR=2e-4
TIMESTEPS=1000
BASE_CH=128                    # Larger model
SAVE_INTERVAL=20
NUM_WORKERS=16

echo ""
echo "Multi-GPU Training configuration:"
echo "  Number of GPUs: 4"
echo "  Epochs: $EPOCHS"
echo "  Total batch size: $BATCH_SIZE (per GPU: $((BATCH_SIZE / 4)))"
echo "  Learning rate: $LR"
echo "  Base channels: $BASE_CH"
echo ""

# For multi-GPU training, you'll need to use torch.distributed
# This is a placeholder - you'd need to modify train_cifar10.py for DDP
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

# Run with torchrun for distributed training
# Note: Requires modifications to train_cifar10.py for distributed training
torchrun --nproc_per_node=4 train_cifar10.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --timesteps $TIMESTEPS \
    --base-ch $BASE_CH \
    --num-workers $NUM_WORKERS \
    --checkpoint-dir checkpoints \
    --save-interval $SAVE_INTERVAL \
    --device cuda

echo "Multi-GPU training completed!"
echo "End time: $(date)"
