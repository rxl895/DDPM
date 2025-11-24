# HPC Job Submission Guide for DDPM Training

This guide explains how to submit your DDPM training job to an HPC cluster.

## 📋 Prerequisites

1. Access to an HPC cluster with:
   - SLURM job scheduler
   - GPU nodes (CUDA-capable)
   - Python 3.9+
   - PyTorch with CUDA support

2. Your code uploaded to the cluster

## 🚀 Quick Start

### 1. Prepare Your Environment

```bash
# SSH into your HPC cluster
ssh your_username@hpc.cluster.edu

# Navigate to your project directory
cd /path/to/DDPM

# Create logs directory
mkdir -p logs

# Make submission scripts executable
chmod +x submit_*.sh
```

### 2. Install Dependencies (if not already done)

```bash
# Option A: Using pip in user space
python3 -m pip install --user -r requirements.txt

# Option B: Using virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option C: Using conda
conda create -n ddpm python=3.9
conda activate ddpm
pip install -r requirements.txt
```

### 3. Submit Training Job

```bash
# Basic training (100 epochs, 1 GPU)
sbatch submit_train.sh

# Long training (500 epochs, 1 GPU)
sbatch submit_train_long.sh

# Multi-GPU training (4 GPUs) - requires code modifications
sbatch submit_train_multi_gpu.sh
```

### 4. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# View output logs (while running)
tail -f logs/train_<JOB_ID>.out

# View error logs
tail -f logs/train_<JOB_ID>.err

# Cancel a job
scancel <JOB_ID>
```

## 📝 Available Job Scripts

### `submit_train.sh` - Standard Training
- **Time**: 24 hours
- **GPUs**: 1
- **Memory**: 32GB
- **Epochs**: 100
- **Batch size**: 128
- **Best for**: Quick experiments, testing

### `submit_train_long.sh` - Long Training
- **Time**: 72 hours (3 days)
- **GPUs**: 1 (V100 or better)
- **Memory**: 64GB
- **Epochs**: 500
- **Batch size**: 128
- **Model**: Larger (128 base channels)
- **Best for**: High-quality results, final models

### `submit_train_multi_gpu.sh` - Multi-GPU Training
- **Time**: 48 hours
- **GPUs**: 4
- **Memory**: 128GB
- **Epochs**: 200
- **Batch size**: 512 (128 per GPU)
- **Best for**: Faster training with large batches
- **Note**: Requires distributed training implementation

## ⚙️ Customizing Job Parameters

Edit the job scripts to customize training:

```bash
# In submit_train.sh, modify these variables:
EPOCHS=100              # Number of training epochs
BATCH_SIZE=128          # Batch size (adjust based on GPU memory)
LR=1e-4                 # Learning rate
TIMESTEPS=1000          # Number of diffusion timesteps
BASE_CH=64              # UNet base channels (64, 128, 256)
SAVE_INTERVAL=10        # Save checkpoint every N epochs
NUM_WORKERS=8           # Data loading workers
```

### SLURM Parameters

Modify the `#SBATCH` directives at the top of the script:

```bash
#SBATCH --time=24:00:00           # Max time (HH:MM:SS)
#SBATCH --partition=gpu           # Partition name (check with: sinfo)
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --gres=gpu:v100:1         # Request specific GPU type
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=32G                 # Memory
#SBATCH --mail-user=your@email    # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
```

## 🔍 Cluster-Specific Adjustments

### Check Available Resources

```bash
# View available partitions
sinfo

# View available GPU types
sinfo -o "%20N %10c %10m %25f %10G"

# Check your account limits
sacctmgr show user $USER

# View available modules
module avail
```

### Load Required Modules

Uncomment and adjust these lines in the job script:

```bash
# Example for most clusters:
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Check available modules:
# module spider python
# module spider cuda
```

### Common Partition Names
- `gpu` - General GPU partition
- `gpu-long` - Long GPU jobs
- `gpu-v100` - V100 GPUs
- `gpu-a100` - A100 GPUs
- `batch` - CPU-only jobs

## 📊 Expected Training Times

On different GPUs (for 100 epochs on CIFAR-10):

| GPU Model | Batch Size 128 | Batch Size 256 |
|-----------|----------------|----------------|
| V100      | ~8-10 hours    | ~6-8 hours     |
| A100      | ~4-6 hours     | ~3-4 hours     |
| RTX 3090  | ~10-12 hours   | ~7-9 hours     |
| CPU Only  | ~100+ hours    | Not recommended|

## 🐛 Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/train_<JOB_ID>.err

# Common issues:
# 1. Module not loaded - add correct module load commands
# 2. Environment not activated - check virtualenv activation
# 3. Dependencies missing - install requirements.txt
# 4. Wrong partition - check with sinfo
```

### Out of Memory
```bash
# Reduce batch size in submit_train.sh:
BATCH_SIZE=64  # or 32

# Or request more memory:
#SBATCH --mem=64G
```

### Job Time Limit Exceeded
```bash
# Option 1: Request more time
#SBATCH --time=48:00:00

# Option 2: Reduce epochs
EPOCHS=50

# Option 3: Use resume functionality (see below)
```

### GPU Not Available
```bash
# Check GPU queue
squeue --partition=gpu

# Try different partition
#SBATCH --partition=gpu-long

# Request specific GPU
#SBATCH --gres=gpu:v100:1
```

## 🔄 Advanced: Resume Training

To add resume functionality, you'll need to modify `train_cifar10.py`:

```python
# Add to argument parser:
parser.add_argument("--resume", type=str, default=None, 
                    help="Path to checkpoint to resume from")

# In main():
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

# Modify training loop:
for epoch in range(start_epoch, num_epochs):
    ...
```

## 📧 Email Notifications

Add these lines to get email updates:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu
```

## 💾 Managing Checkpoints

```bash
# List checkpoints by size
ls -lh checkpoints/

# Keep only the last 5 checkpoints (save space)
ls -t checkpoints/checkpoint_epoch_*.pt | tail -n +6 | xargs rm

# Copy final model to safe location
cp checkpoints/final_model.pt /path/to/backup/
```

## 📈 Monitoring Training Progress

```bash
# Watch loss in real-time
tail -f logs/train_<JOB_ID>.out | grep "Average Loss"

# Check GPU usage
ssh <node_name>
nvidia-smi -l 1

# Use TensorBoard (if implemented)
tensorboard --logdir=runs --host=0.0.0.0
```

## 🎯 Recommended Workflow

1. **Test locally** (1 epoch):
   ```bash
   python3 train_cifar10.py --epochs 1 --batch-size 32
   ```

2. **Quick HPC test** (10 epochs):
   ```bash
   # Edit submit_train.sh: EPOCHS=10
   sbatch submit_train.sh
   ```

3. **Full training** (100-500 epochs):
   ```bash
   sbatch submit_train_long.sh
   ```

4. **Generate samples**:
   ```bash
   python3 demo_sampling.py --checkpoint checkpoints/final_model.pt
   ```

## 📚 Additional Resources

- [SLURM Documentation](https://slurm.schedmd.com/)
- Your cluster's user guide (check cluster website)
- `man sbatch` - SLURM submission manual
- `man squeue` - Job queue manual

## 🆘 Getting Help

If you encounter issues:

1. Check your cluster's documentation
2. Contact your HPC support team
3. Check SLURM logs in `logs/` directory
4. Verify module versions: `module list`
5. Test with minimal example first
