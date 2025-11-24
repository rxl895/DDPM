"""Sample from Latent Diffusion Model.

Generates images by:
1. Running DDIM in latent space (8x8x4)
2. Decoding latents to images with autoencoder
"""
import torch
import os
import argparse
from PIL import Image
import numpy as np

from ddpm.autoencoder import SimpleAutoencoder
from ddpm.latent_unet import LatentUNet
from ddpm.ddim import ddim_sample_loop
from ddpm.forward import get_named_beta_schedule


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of images."""
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
    print(f"Saved: {path}")


def sample_latent_diffusion(
    ae_checkpoint: str,
    diffusion_checkpoint: str,
    num_samples: int = 64,
    ddim_steps: int = 50,
    device=None,
    output_dir: str = "samples_latent"
):
    """Generate samples from latent diffusion model.
    
    Args:
        ae_checkpoint: path to autoencoder checkpoint
        diffusion_checkpoint: path to latent diffusion model
        num_samples: number of samples to generate
        ddim_steps: number of DDIM steps
        device: torch device
        output_dir: output directory
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load autoencoder
    print(f"Loading autoencoder: {ae_checkpoint}")
    autoencoder = SimpleAutoencoder(latent_dim=4, base_ch=64)
    autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Load latent diffusion model
    print(f"Loading diffusion model: {diffusion_checkpoint}")
    latent_unet = LatentUNet(latent_channels=4, base_ch=64, time_emb_dim=128)
    checkpoint = torch.load(diffusion_checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        latent_unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f}")
    else:
        latent_unet.load_state_dict(checkpoint)
    
    latent_unet = latent_unet.to(device)
    latent_unet.eval()
    
    # Generate latents with DDIM
    print(f"Generating {num_samples} latent codes with DDIM ({ddim_steps} steps)...")
    betas = get_named_beta_schedule("linear", 1000)
    
    import time
    start = time.time()
    
    with torch.no_grad():
        latents = ddim_sample_loop(
            latent_unet,
            (num_samples, 4, 8, 8),  # latent shape
            betas,
            num_steps=ddim_steps,
            eta=0.0,
            device=device,
            verbose=True
        )
    
    elapsed = time.time() - start
    print(f"Latent generation took {elapsed:.2f}s ({elapsed/num_samples:.3f}s per sample)")
    
    # Decode latents to images
    print("Decoding latents to images...")
    with torch.no_grad():
        images = autoencoder.decode(latents)
    
    total_time = time.time() - start
    print(f"Total time: {total_time:.2f}s ({total_time/num_samples:.3f}s per image)")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    grid_path = os.path.join(output_dir, f"latent_grid_steps{ddim_steps}.png")
    save_image_grid(images, grid_path, nrow=8)
    
    # Save individual samples
    for idx in range(min(num_samples, 16)):
        img = tensor_to_pil(images[idx])
        img.save(os.path.join(output_dir, f"latent_sample_{idx:03d}.png"))
    
    print(f"Done! Samples saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Sample from Latent Diffusion")
    parser.add_argument("--ae-checkpoint", type=str, required=True, help="Autoencoder checkpoint")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Diffusion checkpoint")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output-dir", type=str, default="samples_latent", help="Output directory")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    sample_latent_diffusion(
        ae_checkpoint=args.ae_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        device=device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
