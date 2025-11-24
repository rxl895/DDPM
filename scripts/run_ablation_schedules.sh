#!/bin/bash
#SBATCH --job-name=ablation_schedules
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ablation_schedules_%j.out
#SBATCH --error=ablation_schedules_%j.err

# Ablation Study: Noise Schedule Comparison
# Compares linear, cosine, quadratic, sigmoid, and exponential schedules

module purge
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Install compatible versions of dependencies
pip install --user "numpy>=1.21,<1.23" scipy "matplotlib>=3.5,<3.8"

echo "Starting Noise Schedule Ablation Study"
echo "========================================"
date
echo ""

# Change to repository root to ensure imports work
cd /scratch/rxl895/DDPM || cd ~/DDPM || cd $SLURM_SUBMIT_DIR

# Run ablation study with DDIM sampling (faster)
python examples/ablation_noise_schedules.py \
    --checkpoint checkpoints/final_model.pt \
    --num_samples 5000 \
    --batch_size 64 \
    --method ddim \
    --ddim_steps 100 \
    --save_path ablation_results \
    --device cuda

echo ""
echo "========================================"
echo "Ablation study complete!"
date
