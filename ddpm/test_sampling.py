"""Quick smoke test for the reverse sampling loop.

Runs a small sampling chain with the random-initialized SmallUNet to make sure the
loop and shapes work.
"""
import torch

from ddpm.unet import SmallUNet
from ddpm.sample import p_sample_loop
from ddpm.forward import get_named_beta_schedule


def test_sampling():
    T = 10
    betas = get_named_beta_schedule("linear", T)
    B, C, H, W = 1, 3, 16, 16
    device = torch.device("cpu")
    model = SmallUNet(in_channels=C, base_ch=16, time_emb_dim=64).to(device)
    out = p_sample_loop(model, (B, C, H, W), betas, device=device)
    assert out.shape == (B, C, H, W)
    print("sampled image shape:", out.shape)


if __name__ == "__main__":
    test_sampling()
