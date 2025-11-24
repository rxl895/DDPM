# DDPM - Denoising Diffusion Probabilistic Models

A simple, clean PyTorch implementation of DDPM based on the paper ["Denoising Diffusion Probabilistic Models" by Ho et al.](https://arxiv.org/abs/2006.11239)

From Denoising Diffusion Models to Latent Diffusion: A Comparative Case Study with Hands-On Implementation.

## Features

- ✅ Forward noising process with configurable beta schedules
- ✅ Epsilon-prediction UNet with time conditioning
- ✅ Reverse sampling (ancestral sampling)
- ✅ Training loop with checkpointing
- ✅ CIFAR-10 and MNIST data loaders
- ✅ Sampling visualization scripts
- ✅ Clean, readable code with comprehensive tests

## Installation

```bash
# Clone the repository
git clone https://github.com/rxl895/DDPM.git
cd DDPM

# Install dependencies
pip install -r requirements.txt
```

Or install PyTorch separately if you need specific CUDA versions:
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (check https://pytorch.org for your version)
pip install torch torchvision
```

## Quick Start

### 1. Test the Implementation

Run the test suite to verify everything works:

```bash
# Test forward noising
python3 -m ddpm.test_forward

# Test UNet
python3 -m ddpm.test_unet

# Test sampling
python3 -m ddpm.test_sampling
```

### 2. Train on CIFAR-10

```bash
# Basic training (100 epochs, batch size 128)
python train_cifar10.py --epochs 100 --batch-size 128 --lr 1e-4

# Training with custom settings
python train_cifar10.py \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-4 \
    --timesteps 1000 \
    --base-ch 128 \
    --checkpoint-dir my_checkpoints \
    --save-interval 5
```

### 3. Generate Samples

```bash
# Generate samples from a trained model
python demo_sampling.py --checkpoint checkpoints/final_model.pt --num-samples 64

# Generate from untrained model (for testing)
python demo_sampling.py --num-samples 16 --output-dir test_samples

# Custom sampling settings
python demo_sampling.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --num-samples 100 \
    --timesteps 1000 \
    --image-size 32 \
    --output-dir my_samples
```

## Project Structure

```
DDPM/
├── ddpm/
│   ├── __init__.py           # Package initialization
│   ├── forward.py            # Forward noising process (q_sample)
│   ├── unet.py               # SmallUNet epsilon-prediction model
│   ├── sample.py             # Reverse sampling (p_sample, p_sample_loop)
│   ├── train.py              # Training loss and loop
│   ├── data.py               # Data loaders (CIFAR-10, MNIST)
│   ├── test_forward.py       # Tests for forward process
│   ├── test_unet.py          # Tests for UNet
│   └── test_sampling.py      # Tests for sampling
├── train_cifar10.py          # Training script for CIFAR-10
├── demo_sampling.py          # Sampling visualization demo
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Architecture Details

### Forward Process (Noising)

The forward process gradually adds Gaussian noise to images:

```python
from ddpm.forward import get_named_beta_schedule, q_sample

# Create beta schedule
betas = get_named_beta_schedule("linear", timesteps=1000)

# Add noise to image x_0 at timestep t
x_t = q_sample(x_0, t, betas)
```

### UNet Model

A small time-conditioned UNet that predicts noise (epsilon):

```python
from ddpm.unet import SmallUNet

model = SmallUNet(
    in_channels=3,      # RGB images
    base_ch=64,         # Base channel count
    time_emb_dim=128    # Timestep embedding dimension
)

# Predict noise
epsilon = model(x_t, t)
```

### Reverse Process (Sampling)

Generate images by iteratively denoising from random noise:

```python
from ddpm.sample import p_sample_loop

# Generate samples
samples = p_sample_loop(
    model,
    shape=(batch_size, 3, 32, 32),
    betas=betas,
    device=device
)
```

### Training

The training loss is MSE between predicted and actual noise:

```python
from ddpm.train import ddpm_loss, train

# Compute loss for a batch
loss = ddpm_loss(model, x_0, betas, device)

# Or use the full training loop
history = train(
    model=model,
    dataloader=dataloader,
    num_epochs=100,
    lr=1e-4,
    betas=betas
)
```

## Usage Examples

### Custom Training Loop

```python
import torch
from ddpm.unet import SmallUNet
from ddpm.data import get_cifar10_dataloader
from ddpm.train import train_epoch
from ddpm.forward import get_named_beta_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=3, base_ch=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataloader = get_cifar10_dataloader(batch_size=128)
betas = get_named_beta_schedule("linear", 1000)

for epoch in range(100):
    avg_loss = train_epoch(model, dataloader, optimizer, betas, device, epoch)
    print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
```

### Generate and Save Samples

```python
import torch
from ddpm.unet import SmallUNet
from ddpm.sample import p_sample_loop
from ddpm.forward import get_named_beta_schedule
from ddpm.train import load_checkpoint

# Load trained model
model = SmallUNet(in_channels=3, base_ch=64)
load_checkpoint(model, None, "checkpoints/final_model.pt")
model.eval()

# Generate samples
betas = get_named_beta_schedule("linear", 1000)
with torch.no_grad():
    samples = p_sample_loop(
        model,
        (16, 3, 32, 32),
        betas,
        device=torch.device("cpu")
    )

# samples is a tensor of shape (16, 3, 32, 32) in range [-1, 1]
```

## Hyperparameters

### Recommended Settings for CIFAR-10

- **Timesteps**: 1000
- **Beta schedule**: linear (1e-4 to 0.02)
- **Learning rate**: 1e-4 to 2e-4
- **Batch size**: 128
- **Base channels**: 64-128
- **Epochs**: 100-500 (more is better)

### Recommended Settings for MNIST

- **Timesteps**: 1000
- **Beta schedule**: linear
- **Learning rate**: 1e-4
- **Batch size**: 128
- **Base channels**: 32-64
- **Epochs**: 50-100

## Performance Notes

- Training time depends heavily on GPU. On a modern GPU (e.g., RTX 3090), one epoch on CIFAR-10 takes ~1-2 minutes.
- The small UNet architecture is designed for quick experimentation. For better quality, increase `base_ch` to 128 or 256.
- Sampling is slow (1000 forward passes through the network). Consider DDIM for faster sampling.

## Checkpoints

Checkpoints are saved in the format:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'loss': float
}
```

Load checkpoints with:

```python
from ddpm.train import load_checkpoint

info = load_checkpoint(model, optimizer, "checkpoints/checkpoint_epoch_50.pt")
print(f"Loaded epoch {info['epoch']} with loss {info['loss']}")
```

## Testing

All core functionality is tested:

```bash
# Run all tests
python3 -m ddpm.test_forward
python3 -m ddpm.test_unet
python3 -m ddpm.test_sampling
```

Expected output:
- Forward: shape validation and noise addition at t=0
- UNet: correct output shapes
- Sampling: full reverse loop completes successfully

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) - Ho et al., NeurIPS 2020
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, ICML 2021

## License

MIT License - feel free to use for research and educational purposes.

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Install PyTorch: `pip install torch torchvision`

**Issue**: Out of memory during training
- **Solution**: Reduce batch size: `--batch-size 64` or `--batch-size 32`

**Issue**: Training is very slow
- **Solution**: Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use `--device cuda` to force GPU usage

**Issue**: Poor sample quality
- **Solution**: Train for more epochs (200+), increase model size (`--base-ch 128`), or check that your model is actually trained (not randomly initialized)

## Citation

If you use this code in your research, please cite the original DDPM paper:

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
