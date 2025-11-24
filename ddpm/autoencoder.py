"""Simple Autoencoder for Latent Diffusion.

A lightweight VAE-style autoencoder to compress images into a lower-dimensional
latent space before applying diffusion. This demonstrates the core idea behind
Stable Diffusion without the full complexity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encode images to latent representation."""
    
    def __init__(self, in_channels=3, latent_dim=4, base_ch=64):
        super().__init__()
        
        # Downsampling path
        # 32x32 -> 16x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),  # downsample
        )
        
        # 16x16 -> 8x8
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1),  # downsample
        )
        
        # 8x8 latent space
        self.to_latent = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, latent_dim, 3, padding=1)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        latent = self.to_latent(x)
        return latent


class Decoder(nn.Module):
    """Decode latent representation to images."""
    
    def __init__(self, latent_dim=4, out_channels=3, base_ch=64):
        super().__init__()
        
        # 8x8 -> 16x16
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, base_ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1),  # upsample
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        
        # 16x16 -> 32x32
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),  # upsample
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
        )
        
        self.to_rgb = nn.Conv2d(base_ch, out_channels, 3, padding=1)
        
    def forward(self, latent):
        x = self.from_latent(latent)
        x = self.upconv1(x)
        x = self.upconv2(x)
        rgb = self.to_rgb(x)
        return rgb


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for latent space compression.
    
    Compresses 32x32x3 images to 8x8x4 latents (16x compression).
    """
    
    def __init__(self, latent_dim=4, base_ch=64):
        super().__init__()
        self.encoder = Encoder(in_channels=3, latent_dim=latent_dim, base_ch=base_ch)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=3, base_ch=base_ch)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """Encode images to latent space."""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latents to images."""
        return self.decoder(latent)
    
    def forward(self, x):
        """Full autoencoder pass."""
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def train_autoencoder(
    autoencoder: nn.Module,
    dataloader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = None
):
    """Train the autoencoder on reconstruction.
    
    Args:
        autoencoder: SimpleAutoencoder model
        dataloader: training data
        epochs: number of epochs
        lr: learning rate
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    autoencoder = autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    print(f"Training autoencoder for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for batch, _ in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            recon, latent = autoencoder(batch)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(recon, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{epochs} - Reconstruction Loss: {avg_loss:.6f}")
    
    print("Autoencoder training complete!")
    return autoencoder


__all__ = ["SimpleAutoencoder", "train_autoencoder", "Encoder", "Decoder"]
