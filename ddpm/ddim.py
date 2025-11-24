"""DDIM (Denoising Diffusion Implicit Models) sampling.

DDIM provides:
1. Deterministic sampling (no random noise injection)
2. Faster sampling with fewer steps (e.g., 50 instead of 1000)
3. Often sharper/clearer images than DDPM

Reference: Song et al. "Denoising Diffusion Implicit Models" (2020)
https://arxiv.org/abs/2010.02502
"""
from typing import Optional, Tuple
import torch
from ddpm.forward import compute_alphas
from ddpm.sample import _extract


def ddim_sample_step(
    model,
    x_t: torch.Tensor,
    t: int,
    t_prev: int,
    betas: torch.Tensor,
    eta: float = 0.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Single DDIM sampling step from timestep t to t_prev.
    
    Args:
        model: noise prediction model
        x_t: current sample at timestep t, shape (B, C, H, W)
        t: current timestep (scalar)
        t_prev: previous timestep (scalar), typically < t
        betas: noise schedule, shape (T,)
        eta: stochasticity parameter (0 = deterministic, 1 = DDPM-like)
        device: torch device
        
    Returns:
        x_{t_prev}: sample at previous timestep
    """
    device = device or x_t.device
    betas = betas.to(device)
    
    # Compute alphas
    comps = compute_alphas(betas)
    alphas_cumprod = comps["alphas_cumprod"].to(device)
    
    # Get alpha values for current and previous timesteps
    batch_size = x_t.shape[0]
    t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
    
    alpha_t = _extract(alphas_cumprod, t_tensor, x_t.shape)
    
    if t_prev >= 0:
        t_prev_tensor = torch.full((batch_size,), t_prev, dtype=torch.long, device=device)
        alpha_t_prev = _extract(alphas_cumprod, t_prev_tensor, x_t.shape)
    else:
        alpha_t_prev = torch.ones_like(alpha_t)
    
    # Predict noise
    eps = model(x_t, t_tensor)
    
    # Predict x0
    x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
    
    # Compute direction pointing to x_t
    dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * eps
    
    # Compute x_{t_prev}
    x_t_prev = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt
    
    # Add stochasticity if eta > 0
    if eta > 0 and t_prev >= 0:
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        noise = torch.randn_like(x_t)
        x_t_prev = x_t_prev + sigma_t * noise
    
    return x_t_prev


@torch.no_grad()
def ddim_sample_loop(
    model,
    shape: Tuple[int, ...],
    betas: torch.Tensor,
    num_steps: int = 50,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
    noise: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> torch.Tensor:
    """DDIM sampling loop with reduced timesteps.
    
    Args:
        model: noise prediction model
        shape: output shape (B, C, H, W)
        betas: full noise schedule, shape (T,)
        num_steps: number of sampling steps (much less than T)
        eta: stochasticity (0=deterministic, 1=DDPM)
        device: torch device
        noise: optional starting noise
        verbose: show progress
        
    Returns:
        Generated samples, shape (B, C, H, W)
    """
    device = device or torch.device("cpu")
    T = betas.shape[0]
    
    # Create sub-sequence of timesteps
    # Evenly spaced from T-1 down to 0
    timesteps = list(range(0, T, T // num_steps))[:num_steps]
    timesteps = list(reversed(timesteps))
    
    # Start from noise
    if noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = noise.to(device)
    
    if verbose:
        try:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="DDIM sampling")
        except ImportError:
            pass
    
    # Iterate through timesteps
    for i, t in enumerate(timesteps):
        # Get previous timestep
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        
        # Perform DDIM step
        x = ddim_sample_step(model, x, t, t_prev, betas, eta=eta, device=device)
    
    return x


__all__ = ["ddim_sample_step", "ddim_sample_loop"]
