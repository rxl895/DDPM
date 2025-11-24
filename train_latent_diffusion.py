"""Train Latent Diffusion Model.

Two-stage training:
1. Train autoencoder for image compression (32x32x3 -> 8x8x4)
2. Train diffusion model in latent space (much faster)
"""
import torch
import argparse
from pathlib import Path

from ddpm.autoencoder import SimpleAutoencoder, train_autoencoder
from ddpm.latent_unet import LatentUNet
from ddpm.forward import get_named_beta_schedule
from ddpm.train import train, save_checkpoint
from ddpm.data import get_cifar10_dataloader


def train_latent_diffusion(
    ae_epochs=20,
    diffusion_epochs=100,
    batch_size=128,
    lr_ae=1e-3,
    lr_diffusion=1e-4,
    device=None,
    ae_checkpoint=None,
    diffusion_checkpoint=None
):
    """Train latent diffusion model.
    
    Args:
        ae_epochs: epochs for autoencoder training
        diffusion_epochs: epochs for diffusion training
        batch_size: training batch size
        lr_ae: learning rate for autoencoder
        lr_diffusion: learning rate for diffusion model
        device: torch device
        ae_checkpoint: path to pretrained autoencoder (skip stage 1 if provided)
        diffusion_checkpoint: path to resume diffusion training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10...")
    dataloader = get_cifar10_dataloader(batch_size=batch_size, num_workers=8)
    
    # Stage 1: Train autoencoder
    print("\n" + "="*80)
    print("STAGE 1: Training Autoencoder")
    print("="*80)
    
    autoencoder = SimpleAutoencoder(latent_dim=4, base_ch=64)
    
    if ae_checkpoint:
        print(f"Loading pretrained autoencoder: {ae_checkpoint}")
        autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    else:
        autoencoder = train_autoencoder(
            autoencoder, dataloader, epochs=ae_epochs, lr=lr_ae, device=device
        )
        
        # Save autoencoder
        Path("checkpoints").mkdir(exist_ok=True)
        ae_path = "checkpoints/autoencoder.pt"
        torch.save(autoencoder.state_dict(), ae_path)
        print(f"Saved autoencoder: {ae_path}")
    
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Test autoencoder
    print("\nTesting autoencoder reconstruction...")
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch[:8].to(device)
            recon, latent = autoencoder(batch)
            mse = torch.mean((recon - batch) ** 2).item()
            print(f"Reconstruction MSE: {mse:.6f}")
            print(f"Latent shape: {latent.shape} (compression: {batch.numel() / latent.numel():.1f}x)")
            break
    
    # Stage 2: Train diffusion in latent space
    print("\n" + "="*80)
    print("STAGE 2: Training Latent Diffusion Model")
    print("="*80)
    
    # Create latent UNet
    latent_unet = LatentUNet(latent_channels=4, base_ch=64, time_emb_dim=128)
    latent_unet = latent_unet.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in latent_unet.parameters())
    print(f"Latent UNet parameters: {params:,}")
    
    # Create latent dataloader (encode images on-the-fly)
    class LatentDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, encoder, device):
            self.base_dataset = base_dataset
            self.encoder = encoder
            self.device = device
            
        def __len__(self):
            return len(self.base_dataset.dataset)
        
        def __getitem__(self, idx):
            img, label = self.base_dataset.dataset[idx]
            with torch.no_grad():
                img = img.unsqueeze(0).to(self.device)
                latent = self.encoder.encode(img).squeeze(0).cpu()
            return latent, label
    
    # Pre-encode dataset (faster training)
    print("Pre-encoding dataset to latent space...")
    latent_dataset = LatentDataset(dataloader, autoencoder, device)
    latent_loader = torch.utils.data.DataLoader(
        latent_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Train diffusion model
    optimizer = torch.optim.Adam(latent_unet.parameters(), lr=lr_diffusion)
    betas = get_named_beta_schedule("linear", 1000)
    
    train(
        latent_unet,
        latent_loader,
        optimizer,
        betas,
        epochs=diffusion_epochs,
        device=device,
        checkpoint_dir="checkpoints",
        checkpoint_prefix="latent_diffusion"
    )
    
    print("\n" + "="*80)
    print("Latent Diffusion Training Complete!")
    print("="*80)
    print(f"Autoencoder: checkpoints/autoencoder.pt")
    print(f"Diffusion: checkpoints/latent_diffusion_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--ae-epochs", type=int, default=20, help="Autoencoder epochs")
    parser.add_argument("--diffusion-epochs", type=int, default=100, help="Diffusion epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr-ae", type=float, default=1e-3, help="Autoencoder learning rate")
    parser.add_argument("--lr-diffusion", type=float, default=1e-4, help="Diffusion learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--ae-checkpoint", type=str, default=None, help="Pretrained autoencoder")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    train_latent_diffusion(
        ae_epochs=args.ae_epochs,
        diffusion_epochs=args.diffusion_epochs,
        batch_size=args.batch_size,
        lr_ae=args.lr_ae,
        lr_diffusion=args.lr_diffusion,
        device=device,
        ae_checkpoint=args.ae_checkpoint
    )


if __name__ == "__main__":
    main()
