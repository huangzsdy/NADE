"""
BeyondMap Model
A lightweight neural network model optimized for CPU execution.
"""

import torch
import torch.nn as nn


class BeyondMapModel(nn.Module):
    """Lightweight model optimized for CPU training."""

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output


def create_model(config):
    """Create model instance with configuration."""
    model = BeyondMapModel(
        input_dim=config.get('input_dim', 512),
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 128)
    )
    return model