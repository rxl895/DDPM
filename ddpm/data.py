"""Data loading utilities for DDPM training.

Provides loaders for CIFAR-10 and other common datasets.
"""
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_dataloader(
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    data_dir: str = "./data"
) -> DataLoader:
    """Get CIFAR-10 training dataloader.
    
    Images are normalized to [-1, 1] range.
    
    Args:
        batch_size: batch size
        num_workers: number of workers for data loading
        download: whether to download dataset if not present
        data_dir: directory to store data
    
    Returns:
        DataLoader for CIFAR-10
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_mnist_dataloader(
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    data_dir: str = "./data"
) -> DataLoader:
    """Get MNIST training dataloader.
    
    Images are normalized to [-1, 1] range and converted to 3-channel.
    
    Args:
        batch_size: batch size
        num_workers: number of workers for data loading
        download: whether to download dataset if not present
        data_dir: directory to store data
    
    Returns:
        DataLoader for MNIST
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # normalize to [-1, 1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # convert to 3 channels
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


__all__ = ["get_cifar10_dataloader", "get_mnist_dataloader"]
