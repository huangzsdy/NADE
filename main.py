"""
BeyondMap Main Entry
Optimized for CPU execution on laptop with 8GB RAM and no GPU.
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.beyondmap_model import BeyondMapModel, create_model
from data.datasets import SimpleDataset, create_dataloader
from trainers.trainer import Trainer, load_config


def main():
    """Main entry point for training."""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config('configs/config.yaml')
    
    # Print configuration
    print("=" * 50)
    print("Configuration:")
    print(f"  Device: {config.get('device', 'cpu')}")
    print(f"  Batch Size: {config.get('batch_size', 1)}")
    print(f"  Num Workers: {config.get('num_workers', 0)}")
    print(f"  Learning Rate: {config.get('learning_rate', 0.001)}")
    print(f"  Num Epochs: {config.get('num_epochs', 10)}")
    print(f"  Use AMP: {config.get('use_amp', False)}")
    print("=" * 50)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = SimpleDataset(
        num_samples=config.get('num_samples', 100),
        input_dim=config.get('input_dim', 512),
        output_dim=config.get('output_dim', 128)
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    print("\nCreating dataloader...")
    train_loader = create_dataloader(
        dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    print(f"Batch size: {config.get('batch_size', 1)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, config)
    
    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    history = trainer.train()
    
    # Print final results
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final Loss: {history['loss'][-1]:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()