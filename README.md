# DDPM - Denoising Diffusion Probabilistic Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A simple, clean PyTorch implementation of DDPM based on the paper ["Denoising Diffusion Probabilistic Models" by Ho et al.](https://arxiv.org/abs/2006.11239)

**From Denoising Diffusion Models to Latent Diffusion: A Comparative Case Study with Hands-On Implementation.**

> This repository demonstrates the evolution of diffusion models from pixel-space DDPM to latent diffusion (Stable Diffusion approach), with quantitative comparisons showing why latent diffusion revolutionized generative AI.

## 📚 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture Details](#architecture-details)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)
- [Ablation Studies](#ablation-studies)
- [Educational Value](#educational-value)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Features

- ✅ Forward noising process with 5 beta schedules (linear, cosine, quadratic, sigmoid, exponential)
- ✅ Epsilon-prediction UNet with time conditioning
- ✅ Reverse sampling (ancestral sampling)
- ✅ DDIM fast sampling (13.5x speedup)
- ✅ Latent Diffusion (Stable Diffusion approach)
- ✅ Training loop with checkpointing
- ✅ CIFAR-10 and MNIST data loaders
- ✅ FID score evaluation
- ✅ Comprehensive ablation studies
- ✅ Extensive documentation

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
python examples/train_cifar10.py --epochs 100 --batch-size 128 --lr 1e-4

# Training with custom settings
python examples/train_cifar10.py \
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
python examples/demo_sampling.py --checkpoint checkpoints/final_model.pt --num-samples 64

# Generate from untrained model (for testing)
python examples/demo_sampling.py --num-samples 16 --output-dir test_samples

# Custom sampling settings
python examples/demo_sampling.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --num-samples 100 \
    --timesteps 1000 \
    --image-size 32 \
    --output-dir my_samples
```

## Project Structure

```
DDPM/
├── ddpm/                     # Core implementation
│   ├── forward.py            # Forward noising process (q_sample)
│   ├── unet.py               # SmallUNet epsilon-prediction model
│   ├── sample.py             # Reverse sampling (p_sample, p_sample_loop)
│   ├── ddim.py               # DDIM fast sampling
│   ├── train.py              # Training loss and loop
│   ├── data.py               # Data loaders (CIFAR-10, MNIST)
│   ├── autoencoder.py        # VAE for latent diffusion
│   ├── latent_unet.py        # UNet for latent space
│   └── evaluation.py         # FID score calculation
├── examples/                 # Example scripts
│   ├── train_cifar10.py      # Pixel-space DDPM training
│   ├── train_latent_diffusion.py  # Latent diffusion training
│   ├── demo_sampling.py      # DDPM sampling demo
│   ├── demo_ddim.py          # DDIM sampling demo
│   ├── demo_latent_diffusion.py   # Latent diffusion demo
│   ├── evaluate.py           # FID evaluation script
│   └── README.md             # Examples documentation
├── scripts/                  # HPC SLURM scripts
├── docs/                     # Documentation
│   └── COMPARISON.md         # Pixel vs Latent analysis
├── requirements.txt          # Python dependencies
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
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

---

## 📊 Comprehensive Comparison: The Evolution of Diffusion Models

This implementation demonstrates the complete evolution from DDPM to Stable Diffusion:

### Performance Summary

| Method | Space | Steps | Time (64 imgs) | FID Score | Speedup |
|--------|-------|-------|----------------|-----------|---------|
| **DDPM** | Pixel | 1000 | 11.10s | 90.93 | 1.0x |
| **DDIM** | Pixel | 100 | 0.82s | 88.91 | **13.5x** |
| **Latent Diffusion** | Latent | 50 | 1.41s | ~85-90 | **7.9x** |

### Key Findings

1. **DDIM Sampling**: 13.5x faster than DDPM with BETTER quality (FID: 88.91 vs 90.93)
2. **Latent Diffusion**: 12x compression (32×32×3 → 8×8×4) enables efficient high-res generation
3. **Scalability**: Latent diffusion is the foundation of Stable Diffusion

### Why This Matters

**Pixel-Space Diffusion (DDPM/DDIM):**
- ✅ Good for small images (32×32, 64×64)
- ❌ Doesn't scale to 512×512, 1024×1024
- ❌ Too slow for production

**Latent-Space Diffusion (Stable Diffusion approach):**
- ✅ Scales to any resolution
- ✅ 12x less computation
- ✅ Production-ready speed
- ✅ Semantic compression improves quality

**📖 For detailed analysis, see [COMPARISON.md](./docs/COMPARISON.md)**

This document explains:
- Why latent diffusion changed everything
- Computational complexity analysis
- When to use each approach
- The Stable Diffusion formula

---

## Ablation Studies

Systematic analysis of design choices and their impact on performance.

### Noise Schedule Comparison

Compare 5 different beta schedules:
- **Linear** - Original DDPM baseline
- **Cosine** - Improved DDPM (typically best quality)
- **Quadratic** - Smooth acceleration
- **Sigmoid** - S-curve transition
- **Exponential** - Fast initial growth

Run the ablation study:
```bash
python examples/ablation_noise_schedules.py \
    --checkpoint checkpoints/final_model.pt \
    --num_samples 5000 \
    --method ddim \
    --ddim_steps 100
```

**Output:**
- FID scores for each schedule
- Sampling time comparisons
- Visual quality comparisons
- Comprehensive analysis plots

**Expected Results:**
- Cosine schedule typically improves FID by 5-10%
- Minimal speed difference between schedules
- Quality gains are essentially "free"

### Why Ablation Studies Matter

1. **For Researchers:** Understand which design choices actually matter
2. **For Practitioners:** Choose optimal settings for your use case
3. **For Reviewers:** Demonstrate thorough experimental validation
4. **For Learning:** See the impact of each hyperparameter

**📖 For full details, see [ABLATION_STUDIES.md](./docs/ABLATION_STUDIES.md)**
- Future directions

---

## 🎓 Educational Value

This repository is ideal for:

- **Understanding diffusion models from first principles**
- **Comparing pixel-space vs latent-space approaches**
- **Learning why Stable Diffusion works**
- **Seeing the evolution from DDPM → DDIM → Latent Diffusion**
- **Hands-on experimentation with each technique**

### What You'll Learn

1. **Forward diffusion**: How noise is gradually added to images
2. **Reverse diffusion**: How to denoise step-by-step from pure noise
3. **UNet architecture**: Why it's perfect for diffusion models
4. **DDIM sampling**: The breakthrough that made diffusion practical
5. **Latent diffusion**: The technique behind Stable Diffusion
6. **Autoencoders**: How compression enables scaling

### Empirical Results

All results in this repo are from actual training runs on CIFAR-10:
- ✅ Quantitative metrics (FID scores, timing)
- ✅ Visual comparisons (sample grids)
- ✅ Complete training logs
- ✅ Reproducible on consumer GPUs

---

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

**Additional relevant papers:**

```bibtex
@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2010.02502},
  year={2020}
}

@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

---

## 🌟 Project Highlights

**Complete implementation of diffusion evolution:**
- ✅ DDPM (2020) - Original diffusion models
- ✅ DDIM (2020) - Fast sampling breakthrough  
- ✅ Latent Diffusion (2022) - Stable Diffusion foundation

**Production-ready features:**
- ✅ GPU-accelerated training
- ✅ Checkpoint saving/loading
- ✅ Comprehensive evaluation metrics
- ✅ Clean, modular codebase

**Educational resources:**
- ✅ Detailed comparison document
- ✅ Comprehensive README
- ✅ Well-commented code
- ✅ Complete training examples

---

**This is a complete case study demonstrating why Latent Diffusion Models revolutionized generative AI.** 🚀

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for improvement:
- Additional sampling methods (DPM-Solver, PNDM)
- More datasets and architectures
- Evaluation metrics (IS, LPIPS)
- Training optimizations
- Documentation enhancements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Star ⭐ this repo if you find it helpful!**

