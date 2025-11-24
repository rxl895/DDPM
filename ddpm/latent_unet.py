"""UNet for latent space diffusion.

Works on 8x8x4 latent representations instead of 32x32x3 images.
This is much more efficient (16x fewer pixels).
"""
import torch
import torch.nn as nn
import math


class LatentUNet(nn.Module):
    """Smaller UNet for latent space (8x8x4).
    
    Similar to SmallUNet but adapted for latent dimensions.
    """
    
    def __init__(self, latent_channels=4, base_ch=64, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (8x8 -> 4x4)
        self.enc1 = self._make_layer(latent_channels, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        
        # Bottleneck (4x4)
        self.mid = self._make_layer(base_ch, base_ch * 2, time_emb_dim)
        
        # Decoder (4x4 -> 8x8)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = self._make_layer(base_ch * 2, base_ch, time_emb_dim)  # concat with skip
        
        # Output
        self.out = nn.Conv2d(base_ch, latent_channels, 3, padding=1)
        
    def _make_layer(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            ),
            nn.Linear(time_emb_dim, out_ch),
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            )
        ])
    
    def _apply_layer(self, layer, x, t_emb):
        """Apply a layer with time embedding."""
        conv1, time_proj, conv2 = layer
        h = conv1(x)
        h = h + time_proj(t_emb)[:, :, None, None]
        h = conv2(h)
        return h
    
    def _timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, x, t):
        """
        x: (B, latent_channels, 8, 8) latent tensor
        t: (B,) timesteps
        """
        # Time embedding
        t_emb = self._timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        h1 = self._apply_layer(self.enc1, x, t_emb)
        h2 = self.down1(h1)
        
        # Bottleneck
        h = self._apply_layer(self.mid, h2, t_emb)
        
        # Decoder
        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)  # skip connection
        h = self._apply_layer(self.dec1, h, t_emb)
        
        # Output
        out = self.out(h)
        return out


__all__ = ["LatentUNet"]
