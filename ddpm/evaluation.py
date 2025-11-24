"""Evaluation metrics for DDPM/DDIM.

Computes FID (Fréchet Inception Distance) to quantitatively measure
the quality of generated samples compared to real data.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from scipy import linalg
from tqdm import tqdm


def get_inception_features(images: torch.Tensor, model, device: torch.device) -> np.ndarray:
    """Extract features from InceptionV3 for a batch of images.
    
    Args:
        images: (B, C, H, W) tensor in range [-1, 1]
        model: InceptionV3 feature extractor
        device: torch device
        
    Returns:
        features: (B, 2048) numpy array
    """
    model.eval()
    with torch.no_grad():
        # Normalize from [-1, 1] to [0, 1] then to ImageNet stats
        images = (images + 1) / 2  # to [0, 1]
        
        # Resize to 299x299 for InceptionV3
        if images.shape[-1] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images = (images - mean) / std
        
        features = model(images)
        
    return features.cpu().numpy()


def calculate_activation_statistics(
    images: torch.Tensor,
    model,
    device: torch.device,
    batch_size: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of InceptionV3 features.
    
    Args:
        images: (N, C, H, W) tensor
        model: InceptionV3 model
        device: torch device
        batch_size: batch size for processing
        
    Returns:
        mu: mean vector
        sigma: covariance matrix
    """
    features_list = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(images))
        batch = images[start:end].to(device)
        
        feats = get_inception_features(batch, model, device)
        features_list.append(feats)
    
    features = np.concatenate(features_list, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 50
) -> float:
    """Calculate FID score between real and generated images.
    
    Args:
        real_images: (N, C, H, W) tensor of real images
        generated_images: (M, C, H, W) tensor of generated images
        device: torch device
        batch_size: batch size for processing
        
    Returns:
        fid: FID score (lower is better)
    """
    # Load InceptionV3
    try:
        from torchvision.models import inception_v3
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove final layer
        inception = inception.to(device)
        inception.eval()
    except Exception as e:
        print(f"Error loading InceptionV3: {e}")
        print("Installing torchvision with inception support...")
        raise
    
    # Calculate statistics for real images
    print("Calculating statistics for real images...")
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, inception, device, batch_size
    )
    
    # Calculate statistics for generated images
    print("Calculating statistics for generated images...")
    mu_gen, sigma_gen = calculate_activation_statistics(
        generated_images, inception, device, batch_size
    )
    
    # Calculate FID
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return float(fid)


def evaluate_fid_from_loader(
    model,
    dataloader: DataLoader,
    num_samples: int,
    sample_fn,
    device: torch.device,
    **sample_kwargs
) -> float:
    """Evaluate FID by generating samples and comparing to real data.
    
    Args:
        model: trained diffusion model
        dataloader: DataLoader with real images
        num_samples: number of samples to generate
        sample_fn: sampling function (p_sample_loop or ddim_sample_loop)
        device: torch device
        **sample_kwargs: additional arguments for sampling function
        
    Returns:
        fid: FID score
    """
    # Collect real images
    print(f"Collecting {num_samples} real images...")
    real_images = []
    for batch, _ in dataloader:
        real_images.append(batch)
        if sum(b.shape[0] for b in real_images) >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    B, C, H, W = real_images.shape
    
    with torch.no_grad():
        generated_images = sample_fn(
            model,
            (num_samples, C, H, W),
            device=device,
            **sample_kwargs
        )
    
    # Calculate FID
    fid = calculate_fid(real_images, generated_images, device)
    
    return fid


__all__ = ["calculate_fid", "evaluate_fid_from_loader"]
