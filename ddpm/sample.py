"""Sampling (reverse) utilities for DDPM.

Implements the ancestral sampling step from Ho et al. using the UNet to predict
epsilon (noise). The implementation is intentionally straightforward and easy
to read rather than highly optimized.
"""
from typing import Optional, Tuple

import torch

from ddpm.forward import get_named_beta_schedule, compute_alphas


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients for a batch of timesteps and reshape for broadcasting.

    a: (T,), t: (B,) long tensor, x_shape: full tensor shape (B, C, H, W)
    returns: (B, 1, 1, 1, ...) shaped tensor to broadcast with x
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    batch_size = t.shape[0]
    out = a.gather(0, t.to(a.device))
    return out.view(batch_size, *([1] * (len(x_shape) - 1)))


def p_mean_variance(model, x_t: torch.Tensor, t: torch.Tensor, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute predicted mean, variance and predicted x0 for p(x_{t-1} | x_t).

    Returns (model_mean, posterior_variance, x0_pred)
    All tensors have same shape as x_t.
    """
    T = betas.shape[0]
    betas = betas.to(x_t.device)
    comps = compute_alphas(betas)
    alphas = comps["alphas"].to(x_t.device)
    alphas_cumprod = comps["alphas_cumprod"].to(x_t.device)
    sqrt_alphas_cumprod = comps["sqrt_alphas_cumprod"].to(x_t.device)
    sqrt_one_minus_alphas_cumprod = comps["sqrt_one_minus_alphas_cumprod"].to(x_t.device)

    # predict epsilon
    eps = model(x_t, t)

    # predict x0 from x_t and eps
    x0_pred = (x_t - _extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps) / _extract(sqrt_alphas_cumprod, t, x_t.shape)

    # compute posterior variance
    betas_t = _extract(betas, t, x_t.shape)
    alphas_t = _extract(alphas, t, x_t.shape)
    alphas_cumprod_t = _extract(alphas_cumprod, t, x_t.shape)

    # alphas_cumprod_{t-1}
    alphas_cumprod_prev = _extract(torch.cat([alphas_cumprod.new_tensor([1.0]), alphas_cumprod[:-1]]), t, x_t.shape)

    posterior_variance = betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)

    # posterior mean coefficients
    coef1 = betas_t * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)
    coef2 = torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)

    model_mean = coef1 * x0_pred + coef2 * x_t
    return model_mean, posterior_variance, x0_pred


@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample x_{t-1} given x_t using the model to predict epsilon.

    model: callable(x, t) -> eps
    x_t: (B, C, H, W)
    t: (B,) long tensor
    betas: (T,) tensor
    returns: x_{t-1} tensor of same shape as x_t
    """
    device = device or x_t.device
    model_mean, posterior_variance, x0_pred = p_mean_variance(model, x_t, t, betas)

    # when t == 0, return the mean (no noise)
    zero_mask = (t == 0).view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device)

    noise = torch.randn_like(x_t)
    sample = model_mean + torch.sqrt(posterior_variance) * noise
    sample = torch.where(zero_mask, model_mean, sample)
    return sample


@torch.no_grad()
def p_sample_loop(model, shape: Tuple[int, ...], betas: torch.Tensor, device: Optional[torch.device] = None, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run the reverse diffusion chain starting from noise.

    shape: full shape of image tensor, e.g., (B, C, H, W)
    betas: (T,) tensor
    """
    device = device or torch.device("cpu")
    T = betas.shape[0]
    if noise is None:
        img = torch.randn(shape, device=device)
    else:
        img = noise.to(device)

    for t in range(T - 1, -1, -1):
        tt = torch.full((shape[0],), t, dtype=torch.long, device=device)
        img = p_sample(model, img, tt, betas, device=device)
    return img


__all__ = ["p_sample", "p_sample_loop", "p_mean_variance"]
