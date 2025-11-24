# Ablation Studies

This directory contains ablation studies that systematically analyze the impact of different design choices on DDPM performance.

> **Note:** The ablation studies require a model checkpoint trained with the current UNet architecture. If you have an older checkpoint, you'll need to retrain with the current code. See the Training section in the main README.

## 1. Noise Schedule Ablation

**Purpose:** Compare different beta schedules and their impact on sample quality and generation speed.

### Schedules Compared

1. **Linear** (baseline)
   - Simple linear interpolation from β_start to β_end
   - Standard choice in original DDPM paper
   - Pros: Simple, well-understood
   - Cons: Uniform noise addition may not be optimal

2. **Cosine**
   - Proposed in "Improved Denoising Diffusion Probabilistic Models"
   - Slower noise addition at start/end, faster in middle
   - Pros: Better signal retention, improved sample quality
   - Cons: Slightly more complex

3. **Quadratic**
   - Quadratic growth from β_start to β_end
   - Retains more signal in early timesteps
   - Pros: Smooth acceleration
   - Cons: May rush noise addition at end

4. **Sigmoid**
   - S-curve transition using sigmoid function
   - Gradual at extremes, rapid in middle
   - Pros: Smooth, natural transition
   - Cons: Less interpretable

5. **Exponential**
   - Exponential growth in log space
   - Fast initial growth, slower later
   - Pros: Good for preserving coarse structure
   - Cons: May be too aggressive early on

### Running the Ablation Study

#### Local (CPU/GPU)
```bash
python examples/ablation_noise_schedules.py \
    --checkpoint checkpoints/final_model.pt \
    --num_samples 5000 \
    --batch_size 64 \
    --method ddim \
    --ddim_steps 100
```

#### HPC (SLURM)
```bash
sbatch scripts/run_ablation_schedules.sh
```

### Evaluation Metrics

1. **FID Score** (Fréchet Inception Distance)
   - Measures sample quality vs real data distribution
   - Lower is better
   - Industry standard for generative models

2. **Sampling Time**
   - Total time to generate N samples
   - Measures computational efficiency
   - Important for deployment

3. **Samples per Second**
   - Throughput metric
   - Higher is better for production use

### Output

The script generates:

1. **`schedule_comparison.png`**
   - 4-panel visualization showing:
     - Beta values over timesteps
     - Alpha values (signal retention)
     - Cumulative alpha (α̅_t)
     - Signal-to-Noise Ratio (SNR)

2. **`ablation_results.png`**
   - 3-panel comparison:
     - FID scores (quality)
     - Sampling times (speed)
     - Quality vs Speed scatter plot

3. **`sample_comparison.png`**
   - Visual grid of samples from each schedule
   - Side-by-side quality comparison

4. **`results.json`**
   - Detailed numerical results
   - Machine-readable format for further analysis

### Expected Results

Based on literature and empirical observations:

| Schedule    | Expected FID | Relative Speed | Notes |
|-------------|--------------|----------------|-------|
| Linear      | ~90-95       | Baseline (1x)  | Original DDPM |
| Cosine      | ~85-90       | ~1x            | Often best quality |
| Quadratic   | ~88-93       | ~1x            | Middle ground |
| Sigmoid     | ~87-92       | ~1x            | Smooth transition |
| Exponential | ~90-95       | ~1x            | Similar to linear |

**Key Insight:** The cosine schedule typically provides 5-10% improvement in FID score with minimal computational overhead, explaining its widespread adoption in modern diffusion models.

### Analysis Questions

The ablation study helps answer:

1. **Does schedule choice significantly impact quality?**
   - Yes, FID can vary by 5-15 points
   - Cosine typically outperforms linear

2. **Is there a speed/quality trade-off?**
   - Minimal speed difference between schedules
   - Quality gains are "free" improvements

3. **Why does schedule matter?**
   - Controls signal-to-noise ratio trajectory
   - Affects how model learns different frequencies
   - Impacts both training stability and sample quality

4. **Which schedule should I use?**
   - For best quality: **Cosine**
   - For simplicity: **Linear**
   - For experimentation: Try all, measure on your data

### Integration with Training

To train with different schedules:

```python
from ddpm.forward import get_named_beta_schedule

# In your training script
betas = get_named_beta_schedule('cosine', timesteps=1000)
```

### References

1. **Linear Schedule:**
   - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

2. **Cosine Schedule:**
   - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021

3. **Theoretical Analysis:**
   - Kingma et al., "Variational Diffusion Models", NeurIPS 2021

---

## Future Ablation Studies

### 2. Architecture Ablation (Planned)
- UNet depth (2, 3, 4 levels)
- Channel multipliers (1x, 2x, 4x)
- Attention mechanisms (self-attention, cross-attention)

### 3. Training Hyperparameters (Planned)
- Learning rates (1e-3, 1e-4, 1e-5)
- Batch sizes (32, 64, 128, 256)
- Timesteps (100, 500, 1000, 2000)

### 4. Sampling Methods (Planned)
- DDPM vs DDIM vs DPM-Solver
- Number of sampling steps
- Eta parameter in DDIM

### 5. Loss Functions (Planned)
- MSE on epsilon (standard)
- MSE on x_0
- Weighted combinations
- Perceptual losses

---

## Contributing

To add new ablation studies:

1. Create script in `examples/ablation_*.py`
2. Add SLURM script in `scripts/run_ablation_*.sh`
3. Document in this file
4. Include visualization and analysis

See `examples/ablation_noise_schedules.py` as a template.
