#!/bin/bash
#SBATCH --job-name=ddpm_train_long
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00                # 3 days
#SBATCH --partition=gpu-long           # Long partition (adjust name)
#SBATCH --gres=gpu:v100:1              # Request specific GPU type
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Email notifications (optional - uncomment and add your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@university.edu

echo "===== DDPM Long Training Job ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo ""

# Load modules
# module load python/3.9
# module load cuda/11.8

# Activate environment
# source .venv/bin/activate

nvidia-smi

mkdir -p logs
mkdir -p checkpoints

# Long training configuration
EPOCHS=500                     # Train for many epochs
BATCH_SIZE=128
LR=1e-4
TIMESTEPS=1000
BASE_CH=128                    # Larger model for better quality
SAVE_INTERVAL=25               # Save less frequently
NUM_WORKERS=8

echo "Long training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Base channels: $BASE_CH (larger model)"
echo "  Expected time: ~60-70 hours"
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

echo ""
echo "Long training completed!"
echo "End time: $(date)"

# Generate high-quality samples
echo "Generating final samples..."
python3 demo_sampling.py \
    --checkpoint checkpoints/final_model.pt \
    --num-samples 100 \
    --timesteps $TIMESTEPS \
    --output-dir samples_500epochs \
    --device cuda

echo "All done!"
