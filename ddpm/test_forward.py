"""Simple test for forward noising (q_sample).

Run with: python3 -m ddpm.test_forward
"""
import torch

from ddpm.forward import get_named_beta_schedule, q_sample


def test_q_sample():
    T = 1000
    betas = get_named_beta_schedule("linear", T)
    B, C, H, W = 4, 3, 16, 16
    x = torch.randn(B, C, H, W)
    # random timesteps in [0, T-1]
    t = torch.randint(0, T, (B,), dtype=torch.long)
    x_noisy = q_sample(x, t, betas)
    assert x_noisy.shape == x.shape
    print("q_sample output shape:", x_noisy.shape)

    # Test with t = 0 and zero noise: x_t should be approximately sqrt(alpha_0) * x
    t0 = torch.zeros(B, dtype=torch.long)
    x_no_noise = q_sample(x, t0, betas, noise=torch.zeros_like(x))
    # since sqrt_alpha_cumprod[0] ~ sqrt(1 - beta_0) close to 1, diff should be small
    diff = (x_no_noise - x).abs().max()
    print("max diff at t=0 with zero noise:", diff.item())


if __name__ == "__main__":
    test_q_sample()
