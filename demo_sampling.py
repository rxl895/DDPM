"""Demo script for DDPM sampling with visualization.

Generates samples from a trained (or untrained) model and saves them as images.
"""
import torch
import os
from PIL import Image
import numpy as np
import argparse
from typing import Optional

from ddpm.unet import SmallUNet
from ddpm.sample import p_sample_loop
from ddpm.forward import get_named_beta_schedule
from ddpm.train import load_checkpoint


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor image to PIL Image.
    
    tensor: (C, H, W) in range [-1, 1]
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of images.
    
    images: (B, C, H, W) tensor
    """
    B, C, H, W = images.shape
    nrow = min(nrow, B)
    ncol = (B + nrow - 1) // nrow
    
    grid = Image.new('RGB', (W * nrow, H * ncol))
    
    for idx in range(B):
        img = tensor_to_pil(images[idx])
        row = idx // nrow
        col = idx % nrow
        grid.paste(img, (col * W, row * H))
    
    grid.save(path)
    print(f"Saved image grid: {path}")


def sample_and_visualize(
    model_path: Optional[str] = None,
    num_samples: int = 64,
    timesteps: int = 1000,
    image_size: int = 32,
    channels: int = 3,
    base_ch: int = 64,
    output_dir: str = "samples",
    device: Optional[torch.device] = None
):
    """Generate samples and save visualizations.
    
    Args:
        model_path: path to model checkpoint (None for random initialization)
        num_samples: number of samples to generate
        timesteps: number of diffusion timesteps
        image_size: size of generated images
        channels: number of image channels
        base_ch: base channels for UNet
        output_dir: directory to save outputs
        device: device to run on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create model
    model = SmallUNet(in_channels=channels, base_ch=base_ch, time_emb_dim=128)
    
    # Load checkpoint if provided
    if model_path is not None:
        print(f"Loading checkpoint: {model_path}")
        info = load_checkpoint(model, None, model_path, device)
        print(f"Loaded model from epoch {info['epoch']} with loss {info['loss']:.6f}")
    else:
        print("Using randomly initialized model (untrained)")
    
    model = model.to(device)
    model.eval()
    
    # Create beta schedule
    betas = get_named_beta_schedule("linear", timesteps)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        samples = p_sample_loop(
            model,
            (num_samples, channels, image_size, image_size),
            betas,
            device=device
        )
    
    # Save grid
    grid_path = os.path.join(output_dir, "sample_grid.png")
    save_image_grid(samples, grid_path, nrow=8)
    
    # Save individual samples
    for idx in range(min(num_samples, 16)):  # save first 16 individual samples
        img = tensor_to_pil(samples[idx])
        img_path = os.path.join(output_dir, f"sample_{idx:03d}.png")
        img.save(img_path)
    
    print(f"Done! Samples saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="DDPM Sampling Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--image-size", type=int, default=32, help="Image size")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--base-ch", type=int, default=64, help="Base channels for UNet")
    parser.add_argument("--output-dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = None
    if args.device is not None:
        device = torch.device(args.device)
    
    sample_and_visualize(
        model_path=args.checkpoint,
        num_samples=args.num_samples,
        timesteps=args.timesteps,
        image_size=args.image_size,
        channels=args.channels,
        base_ch=args.base_ch,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == "__main__":
    main()
