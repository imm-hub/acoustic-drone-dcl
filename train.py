#!/usr/bin/env python3
"""
Training Script for Acoustic Drone Detection

This script provides a complete training pipeline from data loading to model evaluation.

Usage:
    python train.py --config configs/config.yaml
    python train.py --datasets alemadi thesis --epochs 50 --batch-size 32
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import yaml
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import create_dataloaders, CombinedDroneDataset
from src.features.extractor import AudioFeatureExtractor
from src.features.augmentation import SpecAugment
from src.detection.models import create_model, count_parameters
from src.detection.trainer import Trainer, compute_metrics


class SpectrogramTransform:
    """
    Transform that converts raw audio to spectrogram for CNN input.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        augment: bool = False,
    ):
        self.extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            device='cpu'  # Do on CPU for dataloader compatibility
        )
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for librosa-based extraction
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Extract mel spectrogram
        mel_spec = self.extractor.mel_spectrogram(audio, normalize=True)
        
        # Ensure 3D (C, H, W)
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
            
        # Apply augmentation
        if self.augment and self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
            
        return mel_spec


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train acoustic drone detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='data/external',
                        help='Root directory for datasets')
    parser.add_argument('--datasets', nargs='+', default=['alemadi'],
                        choices=['dads', 'alemadi', 'thesis', 'droneaudioset','acolab'],
                        help='Datasets to use for training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['custom_cnn', 'crnn', 'attention_crnn', 
                                'efficientnet_b0', 'efficientnet_b2',
                                'resnet18', 'resnet34', 'vgg11'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Audio arguments
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Audio clip duration in seconds')
    parser.add_argument('--n-mels', type=int, default=128,
                        help='Number of mel filterbanks')
    
    # Other arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides other arguments)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def main():
    args = parse_args()
    
    # Load config if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    print("=" * 60)
    print("ACOUSTIC DRONE DETECTION TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Create experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.model}_{timestamp}"
        
    exp_dir = Path(args.output_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment directory: {exp_dir}")
    
    # Save config
    config_save = vars(args).copy()
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_save, f)
        
    # Create transforms
    train_transform = SpectrogramTransform(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        augment=args.augment
    )
    
    val_transform = SpectrogramTransform(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        duration=args.duration,
        transform=train_transform,
        seed=args.seed
    )
    
    # Create model
    print("\nCreating model...")
    # model = create_model(
    #     args.model,
    #     num_classes=args.num_classes,
    #     pretrained=args.pretrained
    # )
    if args.model in ['efficientnet_b0', 'efficientnet_b2', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg16']:
        model = create_model(
            args.model,
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
    else:
        model = create_model(
            args.model,
            num_classes=args.num_classes
    )    
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {args.model}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    model = model.to(device)
    
    # Create trainer config
    trainer_config = {
        'training': {
            'epochs': args.epochs,
            'optimizer': {
                'name': 'adamw',
                'lr': args.lr,
                'weight_decay': args.weight_decay
            },
            'scheduler': {
                'name': 'cosine',
                'T_max': args.epochs
            },
            'mixed_precision': args.mixed_precision and device == 'cuda',
            'gradient_clip': 1.0,
            'early_stopping': {
                'enabled': True,
                'patience': 10
            }
        },
        'logging': {
            'tensorboard': True
        },
        'paths': {
            'logs': str(exp_dir / 'logs')
        }
    }
    
    # Train
    print("\nStarting training...")
    trainer = Trainer(model, trainer_config, device=device)
    history = trainer.fit(
        train_loader,
        val_loader,
        save_dir=exp_dir / 'checkpoints'
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load(exp_dir / 'checkpoints' / 'best_model.pt', weights_only=False)['model_state_dict']
    )
    
    test_metrics = trainer.validate(test_loader)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print("=" * 60)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'training_history': history,
        'config': config_save
    }
    
    import json
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
        
    print(f"\nResults saved to: {exp_dir}")
    print("Training complete!")
    
    return test_metrics


if __name__ == '__main__':
    main()
