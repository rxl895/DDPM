"""Training script for DDPM on CIFAR-10.

Example usage:
    python train_cifar10.py --epochs 100 --batch-size 128 --lr 1e-4
"""
import torch
import argparse

from ddpm.unet import SmallUNet
from ddpm.data import get_cifar10_dataloader
from ddpm.train import train
from ddpm.forward import get_named_beta_schedule


def main():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--base-ch", type=int, default=64, help="Base channels for UNet")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create model
    model = SmallUNet(in_channels=3, base_ch=args.base_ch, time_emb_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Create beta schedule
    betas = get_named_beta_schedule("linear", args.timesteps)
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = train(
        model=model,
        dataloader=dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        betas=betas,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval
    )
    
    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
