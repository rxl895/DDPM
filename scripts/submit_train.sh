#!/bin/bash
#SBATCH --job-name=ddpm_train          # Job name
#SBATCH --output=logs/train_%j.out     # Standard output log (%j = job ID)
#SBATCH --error=logs/train_%j.err      # Standard error log
#SBATCH --time=24:00:00                # Time limit (24 hours)
#SBATCH --partition=gpu                # Partition name (adjust for your HPC)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=32G                      # Memory per node
#SBATCH --ntasks=1                     # Number of tasks

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "Current directory: $(pwd)"

# Load required modules (adjust for your HPC environment)
# Uncomment and modify these based on your cluster:
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment (if using one)
# source .venv/bin/activate

# Print GPU information
echo "GPU information:"
nvidia-smi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Training parameters
EPOCHS=100
BATCH_SIZE=128
LR=1e-4
TIMESTEPS=1000
BASE_CH=64
SAVE_INTERVAL=10
NUM_WORKERS=8

echo ""
echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Timesteps: $TIMESTEPS"
echo "  Base channels: $BASE_CH"
echo "  Save interval: $SAVE_INTERVAL"
echo "  Number of workers: $NUM_WORKERS"
echo ""

# Run training
python3 train_cifar10.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --timesteps $TIMESTEPS \
    --base-ch $BASE_CH \
    --num-workers $NUM_WORKERS \
    --checkpoint-dir checkpoints \
    --save-interval $SAVE_INTERVAL \
    --device cuda

# Print completion info
echo ""
echo "Training completed!"
echo "End time: $(date)"

# Generate samples from final model
echo ""
echo "Generating samples from trained model..."
python3 demo_sampling.py \
    --checkpoint checkpoints/final_model.pt \
    --num-samples 64 \
    --timesteps $TIMESTEPS \
    --output-dir samples_final \
    --device cuda

echo "Job finished!"
