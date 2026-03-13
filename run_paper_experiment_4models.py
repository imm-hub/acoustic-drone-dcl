#!/usr/bin/env python3
"""
Domain Gap Experiments - Final Version
======================================

Tüm datasetleri destekler:
- Al-Emadi: WAV dosyaları (Binary_Drone_Audio/yes_drone, unknown)
- DADS: HuggingFace arrow format
- DroneThesis: WAV dosyaları (Data/Audio/yes_drone, no_drone)  
- AcoLab: WAV dosyaları (drone, no_drone)

8 deney çalıştırır (4 model × 2 dataset konfigürasyonu):
- Public only: Al-Emadi + DADS + DroneThesis
- Public + AcoLab: Yukarıdakiler + AcoLab

İki test seti:
- Public hold-out (%15)
- AcoLab hold-out (acolab_test/)

Kullanım:
    python run_experiments_final.py --data-root data/external --output-dir experiments/domain_gap

Yazar: İhsan Mert Muhacıroğlu
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import librosa
from tqdm import tqdm

# HuggingFace datasets
try:
    from datasets import load_from_disk, Audio
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets not installed. Install with: pip install datasets")

# Pretrained modeller için
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
HYPERPARAMS = {
    'sample_rate': 16000,
    'duration': 1.0,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'early_stopping_patience': 10,
    'seed': 42,
    'public_holdout_ratio': 0.15,
    'dads_max_samples': 10000,  # DADS'tan max sample (hız için)
}

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class CustomCNN(nn.Module):
    """4-layer CNN baseline"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CRNN(nn.Module):
    """CNN + Bidirectional LSTM"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(128 * 2, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def create_efficientnet(num_classes=2):
    """EfficientNet-B0"""
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    original_conv = model.conv_stem
    model.conv_stem = nn.Conv2d(1, original_conv.out_channels,
                                 kernel_size=original_conv.kernel_size,
                                 stride=original_conv.stride,
                                 padding=original_conv.padding, bias=False)
    with torch.no_grad():
        model.conv_stem.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
    return model


def create_resnet18(num_classes=2):
    """ResNet-18"""
    model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(1, original_conv.out_channels,
                            kernel_size=original_conv.kernel_size,
                            stride=original_conv.stride,
                            padding=original_conv.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
    return model


def create_model(model_name, num_classes=2):
    if model_name == 'cnn':
        return CustomCNN(num_classes)
    elif model_name == 'efficientnet':
        return create_efficientnet(num_classes)
    elif model_name == 'resnet18':
        return create_resnet18(num_classes)
    elif model_name == 'crnn':
        return CRNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DATASET
# ============================================================================

class AudioDataset(Dataset):
    """Audio dataset - supports both file paths and raw audio arrays"""
    
    def __init__(self, audio_data, labels, sample_rate=16000, duration=1.0,
                 n_mels=128, n_fft=2048, hop_length=512, is_raw_audio=False):
        """
        Args:
            audio_data: List of file paths OR list of audio arrays
            labels: List of labels (0=background, 1=drone)
            is_raw_audio: If True, audio_data contains raw numpy arrays
        """
        self.audio_data = audio_data
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_raw_audio = is_raw_audio
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.is_raw_audio:
            audio = self.audio_data[idx]
            if isinstance(audio, dict):
                audio = audio['array']
            audio = np.array(audio, dtype=np.float32)
        else:
            audio_path = self.audio_data[idx]
            try:
                audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            except Exception as e:
                audio = np.zeros(self.num_samples, dtype=np.float32)
        
        # Pad or trim
        if len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))
        else:
            audio = audio[:self.num_samples]
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0)
        
        return mel_spec, label


class CombinedDataset(Dataset):
    """Combines multiple datasets"""
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = np.cumsum([0] + self.lengths)
    
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative[:-1], self.cumulative[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_alemadi(data_root, sample_rate, duration, n_mels, n_fft, hop_length):
    """Load Al-Emadi dataset"""
    alemadi_path = Path(data_root) / 'alemadi'
    
    audio_paths = []
    labels = []
    
    # Drone
    drone_dirs = [
        alemadi_path / 'Binary_Drone_Audio' / 'yes_drone',
        alemadi_path / 'yes_drone',
    ]
    for d in drone_dirs:
        if d.exists():
            for f in d.glob('**/*.wav'):
                audio_paths.append(str(f))
                labels.append(1)
    
    # Background
    bg_dirs = [
        alemadi_path / 'Binary_Drone_Audio' / 'unknown',
        alemadi_path / 'unknown',
    ]
    for d in bg_dirs:
        if d.exists():
            for f in d.glob('**/*.wav'):
                audio_paths.append(str(f))
                labels.append(0)
    
    drone_count = sum(labels)
    bg_count = len(labels) - drone_count
    print(f"  Al-Emadi: {drone_count} drone, {bg_count} background")
    
    if len(audio_paths) == 0:
        return None
    
    return AudioDataset(audio_paths, labels, sample_rate, duration, n_mels, n_fft, hop_length)


def load_dads(data_root, sample_rate, duration, n_mels, n_fft, hop_length, max_samples=10000):
    """Load DADS dataset from HuggingFace format"""
    dads_path = Path(data_root) / 'dads'
    
    if not dads_path.exists() or not HF_AVAILABLE:
        print(f"  DADS: Skipped (path not found or datasets not installed)")
        return None
    
    try:
        ds = load_from_disk(str(dads_path))
        
        # Get train split
        if 'train' in ds:
            ds = ds['train']
        
        # Sample if too large
        total = len(ds)
        if total > max_samples:
            indices = np.random.choice(total, max_samples, replace=False)
            ds = ds.select(indices)
        
        # Extract audio and labels
        audio_arrays = []
        labels = []
        
        # Resample to target sample rate
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        
        for item in tqdm(ds, desc="  Loading DADS", leave=False):
            audio_arrays.append(item['audio']['array'])
            labels.append(item['label'])
        
        drone_count = sum(labels)
        bg_count = len(labels) - drone_count
        print(f"  DADS: {drone_count} drone, {bg_count} background (sampled from {total})")
        
        return AudioDataset(audio_arrays, labels, sample_rate, duration, n_mels, n_fft, hop_length, is_raw_audio=True)
    
    except Exception as e:
        print(f"  DADS: Error loading - {e}")
        return None


def load_thesis(data_root, sample_rate, duration, n_mels, n_fft, hop_length):
    """Load DroneThesis dataset"""
    thesis_path = Path(data_root) / 'drone_detection_thesis'
    
    audio_paths = []
    labels = []
    
    # Check different possible structures
    audio_dirs = [
        thesis_path / 'Data' / 'Audio',
        thesis_path / 'Audio',
        thesis_path,
    ]
    
    for audio_dir in audio_dirs:
        if not audio_dir.exists():
            continue
            
        # Drone
        drone_dirs = [audio_dir / 'yes_drone', audio_dir / 'drone']
        for d in drone_dirs:
            if d.exists():
                for f in d.glob('**/*.wav'):
                    audio_paths.append(str(f))
                    labels.append(1)
        
        # Background
        bg_dirs = [audio_dir / 'no_drone', audio_dir / 'background', audio_dir / 'noise']
        for d in bg_dirs:
            if d.exists():
                for f in d.glob('**/*.wav'):
                    audio_paths.append(str(f))
                    labels.append(0)
    
    drone_count = sum(labels)
    bg_count = len(labels) - drone_count
    print(f"  DroneThesis: {drone_count} drone, {bg_count} background")
    
    if len(audio_paths) == 0:
        return None
    
    return AudioDataset(audio_paths, labels, sample_rate, duration, n_mels, n_fft, hop_length)


def load_acolab_train(data_root, sample_rate, duration, n_mels, n_fft, hop_length):
    """Load AcoLab training data"""
    acolab_path = Path(data_root) / 'acolab'
    
    audio_paths = []
    labels = []
    
    drone_path = acolab_path / 'drone'
    if drone_path.exists():
        for f in drone_path.glob('**/*.wav'):
            audio_paths.append(str(f))
            labels.append(1)
    
    bg_path = acolab_path / 'no_drone'
    if bg_path.exists():
        for f in bg_path.glob('**/*.wav'):
            audio_paths.append(str(f))
            labels.append(0)
    
    drone_count = sum(labels)
    bg_count = len(labels) - drone_count
    print(f"  AcoLab Train: {drone_count} drone, {bg_count} background")
    
    if len(audio_paths) == 0:
        return None
    
    return AudioDataset(audio_paths, labels, sample_rate, duration, n_mels, n_fft, hop_length)


def load_acolab_test(data_root, sample_rate, duration, n_mels, n_fft, hop_length):
    """Load AcoLab hold-out test data"""
    acolab_path = Path(data_root) / 'acolab_test'
    
    audio_paths = []
    labels = []
    
    drone_path = acolab_path / 'drone'
    if drone_path.exists():
        for f in drone_path.glob('**/*.wav'):
            audio_paths.append(str(f))
            labels.append(1)
    
    bg_path = acolab_path / 'no_drone'
    if bg_path.exists():
        for f in bg_path.glob('**/*.wav'):
            audio_paths.append(str(f))
            labels.append(0)
    
    drone_count = sum(labels)
    bg_count = len(labels) - drone_count
    print(f"  AcoLab Test: {drone_count} drone, {bg_count} background")
    
    if len(audio_paths) == 0:
        return None
    
    return AudioDataset(audio_paths, labels, sample_rate, duration, n_mels, n_fft, hop_length)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="    Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"    {desc}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Metrics
    total = len(all_targets)
    correct = (all_preds == all_targets).sum()
    
    drone_mask = all_targets == 1
    bg_mask = all_targets == 0
    
    drone_acc = (all_preds[drone_mask] == 1).mean() if drone_mask.sum() > 0 else 0
    bg_acc = (all_preds[bg_mask] == 0).mean() if bg_mask.sum() > 0 else 0
    
    tp = ((all_preds == 1) & (all_targets == 1)).sum()
    fp = ((all_preds == 1) & (all_targets == 0)).sum()
    fn = ((all_preds == 0) & (all_targets == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'loss': running_loss / total,
        'accuracy': correct / total,
        'drone_accuracy': float(drone_acc),
        'background_accuracy': float(bg_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'drone_total': int(drone_mask.sum()),
        'bg_total': int(bg_mask.sum()),
    }


def train_model(model, train_loader, val_loader, config, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config['epochs']):
        print(f"  Epoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, "Validating")
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"    Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"    Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
            }, save_path)
            print(f"    ✓ Saved best model (acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    return history, best_val_acc


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(model_name, experiment_name, train_dataset, val_dataset,
                   public_test_dataset, acolab_test_dataset,
                   output_dir, device, config):
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"{'='*70}")
    
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = create_model(model_name, num_classes=2)
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    model = model.to(device)
    
    # Train
    model_path = exp_dir / 'best_model.pt'
    start_time = time.time()
    history, best_val_acc = train_model(model, train_loader, val_loader, config, device, model_path)
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Load best model
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    
    # Test on Public hold-out
    print("\n--- Public Hold-out Test ---")
    public_test_loader = DataLoader(public_test_dataset, batch_size=config['batch_size'],
                                    shuffle=False, num_workers=4)
    public_metrics = evaluate(model, public_test_loader, criterion, device, "Public Test")
    print(f"  Accuracy: {public_metrics['accuracy']*100:.2f}%")
    print(f"  Drone:    {public_metrics['drone_accuracy']*100:.2f}% ({public_metrics['drone_total']})")
    print(f"  BG:       {public_metrics['background_accuracy']*100:.2f}% ({public_metrics['bg_total']})")
    print(f"  F1:       {public_metrics['f1']:.4f}")
    
    # Test on AcoLab hold-out
    print("\n--- AcoLab Hold-out Test ---")
    acolab_test_loader = DataLoader(acolab_test_dataset, batch_size=config['batch_size'],
                                    shuffle=False, num_workers=4)
    acolab_metrics = evaluate(model, acolab_test_loader, criterion, device, "AcoLab Test")
    print(f"  Accuracy: {acolab_metrics['accuracy']*100:.2f}%")
    print(f"  Drone:    {acolab_metrics['drone_accuracy']*100:.2f}% ({acolab_metrics['drone_total']})")
    print(f"  BG:       {acolab_metrics['background_accuracy']*100:.2f}% ({acolab_metrics['bg_total']})")
    print(f"  F1:       {acolab_metrics['f1']:.4f}")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'model_name': model_name,
        'total_parameters': total_params,
        'training_samples': len(train_dataset),
        'training_time_minutes': train_time / 60,
        'best_val_accuracy': best_val_acc,
        'public_test': public_metrics,
        'acolab_test': acolab_metrics,
        'history': history,
        'config': config,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/external')
    parser.add_argument('--output-dir', type=str, default='experiments/domain_gap')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['cnn', 'efficientnet', 'resnet18', 'crnn'])
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    
    print("="*70)
    print("DOMAIN GAP EXPERIMENTS")
    print("="*70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data: {args.data_root}")
    print(f"Output: {args.output_dir}")
    print(f"Models: {args.models}")
    print("="*70)
    
    torch.manual_seed(HYPERPARAMS['seed'])
    np.random.seed(HYPERPARAMS['seed'])
    
    config = HYPERPARAMS
    sr = config['sample_rate']
    dur = config['duration']
    n_mels = config['n_mels']
    n_fft = config['n_fft']
    hop = config['hop_length']
    
    # ========== LOAD ALL DATASETS ==========
    print("\n--- Loading Datasets ---")
    
    alemadi_ds = load_alemadi(args.data_root, sr, dur, n_mels, n_fft, hop)
    dads_ds = load_dads(args.data_root, sr, dur, n_mels, n_fft, hop, config['dads_max_samples'])
    thesis_ds = load_thesis(args.data_root, sr, dur, n_mels, n_fft, hop)
    acolab_train_ds = load_acolab_train(args.data_root, sr, dur, n_mels, n_fft, hop)
    acolab_test_ds = load_acolab_test(args.data_root, sr, dur, n_mels, n_fft, hop)
    
    # Combine public datasets
    public_datasets = [d for d in [alemadi_ds, dads_ds, thesis_ds] if d is not None]
    if not public_datasets:
        print("ERROR: No public datasets loaded!")
        return
    
    public_combined = CombinedDataset(public_datasets)
    print(f"\nPublic Combined: {len(public_combined)} samples")
    
    # Split public into train and test
    public_test_size = int(len(public_combined) * config['public_holdout_ratio'])
    public_train_size = len(public_combined) - public_test_size
    
    torch.manual_seed(config['seed'])
    public_train_ds, public_test_ds = random_split(public_combined, [public_train_size, public_test_size])
    print(f"Public Train: {len(public_train_ds)}, Public Test: {len(public_test_ds)}")
    
    # Check AcoLab
    if acolab_train_ds is None:
        print("WARNING: No AcoLab training data!")
    if acolab_test_ds is None:
        print("ERROR: No AcoLab test data!")
        return
    
    # Further split public_train into train/val
    val_size = int(len(public_train_ds) * 0.1)
    train_size = len(public_train_ds) - val_size
    public_train_only, public_val = random_split(public_train_ds, [train_size, val_size])
    
    print(f"Final splits - Train: {len(public_train_only)}, Val: {len(public_val)}")
    
    # ========== RUN EXPERIMENTS ==========
    all_results = []
    
    for model_name in args.models:
        # Experiment 1: Public only
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*70}")
        
        results = run_experiment(
            model_name=model_name,
            experiment_name=f"{model_name}_public_only",
            train_dataset=public_train_only,
            val_dataset=public_val,
            public_test_dataset=public_test_ds,
            acolab_test_dataset=acolab_test_ds,
            output_dir=args.output_dir,
            device=device,
            config=config
        )
        all_results.append(results)
        
        # Experiment 2: Public + AcoLab
        if acolab_train_ds is not None:
            combined_train = CombinedDataset([public_train_only.dataset, acolab_train_ds])
            # Re-select indices from public_train_only
            combined_indices = list(public_train_only.indices) + list(range(len(public_train_only.dataset), len(combined_train)))
            
            # Simpler approach: just combine the datasets directly
            combined_for_train = CombinedDataset([
                torch.utils.data.Subset(public_combined, public_train_only.indices),
                acolab_train_ds
            ])
            
            # Split combined into train/val
            combined_val_size = int(len(combined_for_train) * 0.1)
            combined_train_size = len(combined_for_train) - combined_val_size
            combined_train_only, combined_val = random_split(combined_for_train, [combined_train_size, combined_val_size])
            
            results = run_experiment(
                model_name=model_name,
                experiment_name=f"{model_name}_public_plus_acolab",
                train_dataset=combined_train_only,
                val_dataset=combined_val,
                public_test_dataset=public_test_ds,
                acolab_test_dataset=acolab_test_ds,
                output_dir=args.output_dir,
                device=device,
                config=config
            )
            all_results.append(results)
    
    # ========== SUMMARY ==========
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"{'Model':<15} {'Training':<18} {'Public Test':<15} {'AcoLab Test':<15} {'Domain Gap':<12}")
    print("-"*100)
    
    for r in all_results:
        model = r['model_name']
        training = 'Public+AcoLab' if 'plus_acolab' in r['experiment_name'] else 'Public'
        public_acc = r['public_test']['accuracy'] * 100
        acolab_acc = r['acolab_test']['accuracy'] * 100
        gap = acolab_acc - public_acc
        
        print(f"{model:<15} {training:<18} {public_acc:<15.2f} {acolab_acc:<15.2f} {gap:+.2f}")
    
    print("="*100)
    
    # Save summary
    with open(Path(args.output_dir) / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
