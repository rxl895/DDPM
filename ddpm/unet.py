"""A small UNet that predicts epsilon (noise) given x_t and timestep t.

This is a tiny, intentionally simple UNet for DDPM experiments. It uses
small residual blocks, down/upsampling by stride-2 conv/convtranspose,
and a sinusoidal timestep embedding passed through an MLP to condition the blocks.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for timesteps.

    timesteps: (B,) long tensor
    returns: (B, dim) float tensor
    """
    assert timesteps.dim() == 1
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(-torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1)))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = None
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_ch)
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x, t_emb: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        if self.time_proj is not None and t_emb is not None:
            # add time embedding (B, C) -> (B, C, 1, 1)
            h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)


class SmallUNet(nn.Module):
    """A small UNet for epsilon prediction with aligned skip connections.

    Architecture (channels shown relative to base_ch):
      init -> down1 (base) -> downsample -> down2 (base*2) -> downsample -> down3 (base*4)
      mid (base*4)
      upsample -> concat with down2 -> up block -> upsample -> concat with down1 -> up block -> out
    """

    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # down: blocks and downsampling convs that increase channels while halving spatial dims
        self.down1 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.downsample1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)

        self.down2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.downsample2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)

        self.down3 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # mid
        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # up: upsample (convtranspose) then ResBlock
        self.upconv1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim)

        self.upconv2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock(base_ch * 2, base_ch, time_emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict epsilon given x_t and timestep tensor t.

        x: (B, C, H, W)
        t: (B,) long tensor
        returns: (B, C, H, W) predicted noise
        """
        t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        x0 = self.init_conv(x)

        d1 = self.down1(x0, t_emb)  # (B, base, H, W)
        p1 = self.downsample1(d1)  # (B, base*2, H/2, W/2)

        d2 = self.down2(p1, t_emb)  # (B, base*2, H/2, W/2)
        p2 = self.downsample2(d2)  # (B, base*4, H/4, W/4)

        d3 = self.down3(p2, t_emb)  # (B, base*4, H/4, W/4)

        m = self.mid(d3, t_emb)

        u1 = self.upconv1(m)  # (B, base*2, H/2, W/2)
        # concat with d2 (channels base*2)
        u1 = torch.cat([u1, d2], dim=1)  # (B, base*4, H/2, W/2)
        u1 = self.up1(u1, t_emb)  # -> (B, base*2, H/2, W/2)

        u2 = self.upconv2(u1)  # (B, base, H, W)
        u2 = torch.cat([u2, d1], dim=1)  # (B, base*2, H, W)
        u2 = self.up2(u2, t_emb)  # -> (B, base, H, W)

        out = self.out_norm(u2)
        out = F.silu(out)
        out = self.out_conv(out)
        return out


__all__ = ["SmallUNet", "timestep_embedding"]
