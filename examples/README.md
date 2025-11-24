# Examples

This directory contains example scripts demonstrating how to use the DDPM implementation.

## Training

### Pixel-Space DDPM
```bash
python train_cifar10.py --data_dir ./data --save_dir ./checkpoints --epochs 100 --batch_size 128
```

### Latent Diffusion
```bash
python train_latent_diffusion.py --data_dir ./data --save_dir ./checkpoints --autoencoder_epochs 20 --diffusion_epochs 100
```

## Sampling

### DDPM (1000 steps)
```bash
python demo_sampling.py --checkpoint ./checkpoints/final_model.pt --num_samples 64 --output_dir ./samples
```

### DDIM (50 steps, fast)
```bash
python demo_ddim.py --checkpoint ./checkpoints/final_model.pt --ddim_steps 50 --num_samples 64 --output_dir ./samples_ddim
```

### Latent Diffusion (50 steps)
```bash
python demo_latent_diffusion.py --checkpoint ./checkpoints/latent_diffusion.pt --autoencoder_checkpoint ./checkpoints/autoencoder.pt --num_samples 64 --output_dir ./samples_latent
```

## Evaluation

### FID Score Calculation
```bash
python evaluate.py --checkpoint ./checkpoints/final_model.pt --data_dir ./data --num_samples 10000
```

This will compare:
- DDPM (1000 steps)
- DDIM-50 (50 steps)
- DDIM-100 (100 steps)
- DDIM-250 (250 steps)

And report FID scores and sampling times for each method.

## Ablation Studies

### Noise Schedule Comparison
```bash
python ablation_noise_schedules.py \
    --checkpoint ./checkpoints/final_model.pt \
    --num_samples 5000 \
    --batch_size 64 \
    --method ddim \
    --ddim_steps 100
```

Compares 5 beta schedules (linear, cosine, quadratic, sigmoid, exponential) with:
- FID score evaluation
- Sampling time measurements
- Visual quality comparison
- Comprehensive analysis plots

See `../docs/ABLATION_STUDIES.md` for detailed documentation.

## HPC Scripts

See `../scripts/` for SLURM job submission scripts if running on an HPC cluster.
