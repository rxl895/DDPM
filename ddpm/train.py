"""Training utilities for DDPM.

Implements the training loss (MSE on predicted epsilon) and a training loop.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from tqdm import tqdm

from ddpm.forward import get_named_beta_schedule, q_sample


def ddpm_loss(model, x_0: torch.Tensor, betas: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute the DDPM training loss (MSE between predicted and actual noise).
    
    Args:
        model: epsilon prediction model
        x_0: clean images (B, C, H, W)
        betas: beta schedule (T,)
        device: device to run on
    
    Returns:
        scalar loss tensor
    """
    B = x_0.shape[0]
    T = betas.shape[0]
    
    # Sample random timesteps for each image in batch
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Get noisy images
    x_t = q_sample(x_0, t, betas, noise=noise)
    
    # Predict noise
    noise_pred = model(x_t, t)
    
    # MSE loss
    loss = nn.functional.mse_loss(noise_pred, noise)
    return loss


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    betas: torch.Tensor,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
) -> float:
    """Train for one epoch.
    
    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(device)
        
        optimizer.zero_grad()
        loss = ddpm_loss(model, x, betas, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    model,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float = 1e-4,
    betas: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    save_interval: int = 10
) -> Dict[str, Any]:
    """Full training loop.
    
    Args:
        model: epsilon prediction model
        dataloader: training data loader
        num_epochs: number of epochs to train
        lr: learning rate
        betas: beta schedule (if None, uses default linear schedule)
        device: device to train on
        checkpoint_dir: directory to save checkpoints
        save_interval: save checkpoint every N epochs
    
    Returns:
        dict with training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if betas is None:
        betas = get_named_beta_schedule("linear", 1000)
    
    betas = betas.to(device)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {"epoch": [], "loss": []}
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, betas, device, epoch)
        
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, num_epochs, history["loss"][-1], final_path)
    print(f"Saved final model: {final_path}")
    
    return history


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path: str, device: Optional[torch.device] = None):
    """Load model checkpoint.
    
    Returns:
        dict with checkpoint info (epoch, loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


__all__ = ["ddpm_loss", "train_epoch", "train", "save_checkpoint", "load_checkpoint"]
