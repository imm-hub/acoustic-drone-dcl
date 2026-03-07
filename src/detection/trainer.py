"""
Training Module for Acoustic Drone Detection

Provides training loops, metrics, and utilities for model training.

Author: ITU Telecommunication Engineering
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Tuple, Optional, Callable
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class MetricTracker:
    """Track and compute running metrics during training."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.values = defaultdict(list)
        self.counts = defaultdict(int)
        
    def update(self, metrics: Dict[str, float], n: int = 1):
        for key, value in metrics.items():
            self.values[key].append(value * n)
            self.counts[key] += n
            
    def compute(self) -> Dict[str, float]:
        return {
            key: sum(values) / self.counts[key]
            for key, values in self.values.items()
        }


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        outputs: Model outputs (logits or probabilities)
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1
    """
    # Convert logits to predictions
    if outputs.dim() > 1 and outputs.size(1) > 1:
        # Multi-class
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
    else:
        # Binary
        probs = torch.sigmoid(outputs.squeeze())
        preds = (probs > threshold).long()
        
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute metrics
    correct = (preds == targets).sum()
    total = len(targets)
    accuracy = correct / total
    
    # For binary classification
    if len(np.unique(targets)) <= 2:
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        # Macro average for multi-class
        precision, recall, f1 = 0, 0, 0
        for cls in np.unique(targets):
            tp = ((preds == cls) & (targets == cls)).sum()
            fp = ((preds == cls) & (targets != cls)).sum()
            fn = ((preds != cls) & (targets == cls)).sum()
            
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            precision += p
            recall += r
            f1 += 2 * p * r / (p + r + 1e-8)
            
        n_classes = len(np.unique(targets))
        precision /= n_classes
        recall /= n_classes
        f1 /= n_classes
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class Trainer:
    """
    Trainer class for acoustic drone detection models.
    
    Supports:
    - Mixed precision training (AMP)
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - TensorBoard/W&B logging
    - Checkpoint saving
    
    Example:
        >>> model = create_model('efficientnet_b0', num_classes=2)
        >>> trainer = Trainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Training settings
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.mixed_precision = train_config.get('mixed_precision', True)
        
        # Initialize components
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Early stopping
        es_config = train_config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=es_config.get('patience', 15),
            min_delta=es_config.get('min_delta', 0.001)
        ) if es_config.get('enabled', True) else None
        
        # Tracking
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
    def _setup_criterion(self):
        """Setup loss function."""
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_optimizer(self):
        """Setup optimizer."""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        opt_name = opt_config.get('name', 'adamw').lower()
        lr = opt_config.get('lr', 0.001)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        if opt_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
            
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        sched_config = self.config.get('training', {}).get('scheduler', {})
        sched_name = sched_config.get('name', 'cosine').lower()
        
        if sched_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', self.epochs),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            self.scheduler = None
            
    def _setup_logging(self):
        """Setup logging (TensorBoard/W&B)."""
        log_config = self.config.get('logging', {})
        
        # TensorBoard
        self.writer = None
        if log_config.get('tensorboard', True) and TENSORBOARD_AVAILABLE:
            log_dir = Path(self.config.get('paths', {}).get('logs', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
        # W&B
        if log_config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE:
            wandb.init(
                project=log_config['wandb'].get('project', 'drone-detection'),
                config=self.config
            )
            
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricTracker()
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle attention models
                    loss = self.criterion(outputs, targets)
                    
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    
                self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs, targets)
                metrics['loss'] = loss.item()
                metric_tracker.update(metrics, n=inputs.size(0))
                
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })
            
        return metric_tracker.compute()
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = self.criterion(outputs, targets)
            
            metrics = compute_metrics(outputs, targets)
            metrics['loss'] = loss.item()
            metric_tracker.update(metrics, n=inputs.size(0))
            
        return metric_tracker.compute()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir or self.config.get('checkpoints', {}).get('save_dir', 'models/checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining on {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Epochs: {self.epochs}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
                
            # Print progress
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(save_dir / 'best_model.pt', is_best=True)
                print(f"  --> New best model saved!")
                
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['loss']):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
                    
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pt')
        
        # Cleanup
        if self.writer is not None:
            self.writer.close()
            
        return dict(self.history)
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ):
        """Log metrics to TensorBoard/W&B."""
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
            
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': dict(self.history)
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = defaultdict(list, checkpoint.get('history', {}))
        
        
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Quick test
    print("Testing training module...")
    
    # Create dummy model and data
    from src.detection.models import create_model
    
    model = create_model('custom_cnn', num_classes=2)
    
    # Dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            x = torch.randn(1, 128, 128)
            y = torch.randint(0, 2, (1,)).item()
            return x, y
            
    dataset = DummyDataset(100)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(DummyDataset(20), batch_size=16)
    
    # Config
    config = {
        'training': {
            'epochs': 3,
            'optimizer': {'name': 'adamw', 'lr': 0.001},
            'scheduler': {'name': 'cosine'},
            'mixed_precision': torch.cuda.is_available(),
            'early_stopping': {'enabled': False}
        },
        'logging': {'tensorboard': False}
    }
    
    # Train
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader, save_dir='models/test')
    
    print("\nTraining completed!")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
