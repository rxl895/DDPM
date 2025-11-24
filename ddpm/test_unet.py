"""Quick shape test for the SmallUNet."""
import torch

from ddpm.unet import SmallUNet


def test_unet_shapes():
    B, C, H, W = 2, 3, 32, 32
    model = SmallUNet(in_channels=C, base_ch=32, time_emb_dim=64)
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    out = model(x, t)
    assert out.shape == x.shape
    print("UNet forward output shape:", out.shape)


if __name__ == "__main__":
    test_unet_shapes()
