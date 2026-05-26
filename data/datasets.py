"""
Dataset module for BeyondMap project.
Provides lightweight datasets optimized for CPU training.
"""

import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """Simple dataset for testing and prototyping."""

    def __init__(self, num_samples=1000, input_dim=512, output_dim=128):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate random data for demonstration
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, input_dim)
        self.targets = torch.randn(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0):
    """
    Create DataLoader optimized for CPU.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size (default: 1 for low memory)
        shuffle: Whether to shuffle data
        num_workers: Number of workers (default: 0 for CPU)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataloader