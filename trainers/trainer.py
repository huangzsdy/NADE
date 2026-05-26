"""
Trainer module for BeyondMap project.
Optimized for CPU execution with 8GB RAM and no GPU.
"""

import torch
import torch.nn as nn
import yaml
import os
from pathlib import Path


class Trainer:
    """Lightweight trainer optimized for CPU."""

    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        
        # Device configuration - CPU only
        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)
        
        # Training settings
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 10)
        self.log_interval = config.get('log_interval', 10)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {'loss': []}

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.log_interval == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}] "
                      f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.history['loss'].append(avg_loss)
        return avg_loss

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Device: {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        print("-" * 50)
        
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss: {loss:.4f}")
            print("-" * 50)
        
        print("Training completed!")
        return self.history


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default configuration
        return {
            'device': 'cpu',
            'batch_size': 1,
            'num_workers': 0,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'log_interval': 10,
            'use_amp': False,
        }