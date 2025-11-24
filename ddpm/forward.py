"""Forward noising process for DDPM (PyTorch).

This module implements a basic beta schedule, alpha calculations, and
the q_sample function which produces x_t from x_0 by adding Gaussian noise
according to the DDPM forward process.

References:
- Ho et al., Denoising Diffusion Probabilistic Models (DDPM)
  https://arxiv.org/abs/2006.11239
"""
from typing import Optional

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Create a linear beta schedule from beta_start to beta_end over timesteps.

    Returns a 1-D tensor of shape (timesteps,).
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def get_named_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(timesteps)
    raise ValueError(f"Unknown beta schedule: {schedule}")


def compute_alphas(betas: torch.Tensor):
    """Compute alpha, alpha_cumprod and useful sqrt terms from betas.

    Returns a dict containing:
      - betas
      - alphas
      - alphas_cumprod
      - sqrt_alphas_cumprod
      - sqrt_one_minus_alphas_cumprod
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """Extract values from a 1-D tensor `a` at indices `t` and reshape to `x_shape` batch dims.

    a: (T,), t: (B,) or scalar tensor, x_shape: shape of x (B, C, ...)
    Returns a tensor of shape (B, 1, 1, ...) suitable for broadcasting with x.
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    batch_size = t.shape[0]
    out = a.gather(0, t.to(a.device))
    # reshape to [B, 1, 1, ...] to broadcast with x
    return out.view(batch_size, *([1] * (len(x_shape) - 1)))


def q_sample(x_start: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Produce a noisy sample x_t given original x_start and timestep(s) t.

    x_start: tensor of shape (B, C, H, W) (or (B, C, L) etc.)
    t: LongTensor of shape (B,) with values in [0, T-1] or scalar tensor
    betas: 1-D tensor of length T
    noise: optional noise tensor of same shape as x_start; if None, sampled from N(0,1)

    Returns: x_t of same shape as x_start.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    comps = compute_alphas(betas)
    sqrt_alphas_cumprod = comps["sqrt_alphas_cumprod"].to(x_start.device)
    sqrt_one_minus_alphas_cumprod = comps["sqrt_one_minus_alphas_cumprod"].to(x_start.device)

    # ensure t is LongTensor
    if not torch.is_tensor(t):
        t = torch.tensor([t], dtype=torch.long, device=x_start.device)
    t = t.long()

    return _extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    ) * noise


__all__ = ["linear_beta_schedule", "get_named_beta_schedule", "compute_alphas", "q_sample"]
