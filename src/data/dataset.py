"""
Dataset Loaders for Acoustic Drone Detection

This module provides unified data loading for all available drone audio datasets.
Each dataset has different structure, so we provide specific loaders that output
a consistent format.

Author: ITU Telecommunication Engineering

Supported Datasets:
1. DADS (Drone Audio Detection Samples) - Hugging Face
2. Al-Emadi Drone Audio Dataset - GitHub
3. DroneDetectionThesis - GitHub (multi-sensor)
4. DroneAudioset - Hugging Face (search & rescue)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
import librosa
import json
from tqdm import tqdm
import warnings

# Try to import huggingface datasets
try:
    from datasets import load_dataset, Audio
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Hugging Face 'datasets' not installed. Install with: pip install datasets")


class BaseAudioDataset(Dataset):
    """
    Base class for audio datasets.
    
    All dataset loaders inherit from this and implement their own loading logic.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.transform = transform
        self.target_transform = target_transform
        
        self.audio_paths: List[str] = []
        self.labels: List[int] = []
        self.label_names: Dict[int, str] = {}
        
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio = self._load_audio(audio_path)
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Apply transforms
        if self.transform is not None:
            audio = self.transform(audio)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return audio, label
    
    def _load_audio(self, path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as e:
            warnings.warn(f"Error loading {path}: {e}. Returning zeros.")
            return np.zeros(self.num_samples, dtype=np.float32)
        
        # Pad or truncate to fixed length
        if len(audio) < self.num_samples:
            # Pad with zeros
            audio = np.pad(audio, (0, self.num_samples - len(audio)))
        elif len(audio) > self.num_samples:
            # Random crop during training, center crop otherwise
            start = np.random.randint(0, len(audio) - self.num_samples)
            audio = audio[start:start + self.num_samples]
            
        return audio.astype(np.float32)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset."""
        from collections import Counter
        counts = Counter(self.labels)
        return {self.label_names.get(k, str(k)): v for k, v in counts.items()}


class DADSDataset(BaseAudioDataset):
    """
    DADS - Drone Audio Detection Samples from Hugging Face.
    
    Source: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
    
    This is currently the largest publicly available drone audio database.
    Structure: Binary classification (0 = no drone, 1 = drone)
    
    Example:
        >>> dataset = DADSDataset(root='data/external/dads')
        >>> audio, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = 'data/external/dads',
        split: str = 'train',  # 'train' or 'test'
        download: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.root = Path(root)
        self.split = split
        self.label_names = {0: 'no_drone', 1: 'drone'}
        
        if download:
            self._download()
            
        self._load_data()
        
    def _download(self):
        """Download dataset from Hugging Face."""
        if not HF_AVAILABLE:
            raise RuntimeError("Please install datasets: pip install datasets")
            
        print("Downloading DADS dataset from Hugging Face...")
        dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
        
        self.root.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(self.root))
        print(f"Dataset saved to {self.root}")
        
    def _load_data(self):
        """Load dataset from disk."""
        if not HF_AVAILABLE:
            raise RuntimeError("Please install datasets: pip install datasets")
            
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(self.root))
            
            # Get the appropriate split
            if self.split in dataset:
                data = dataset[self.split]
            else:
                # If no split, use the whole dataset
                data = dataset
                if hasattr(data, 'train'):
                    data = data['train']
                    
            # Extract audio paths and labels
            # DADS structure: 'audio' column with {'path': ..., 'array': ...}
            for item in tqdm(data, desc=f"Loading DADS {self.split}"):
                if 'audio' in item:
                    audio_info = item['audio']
                    if isinstance(audio_info, dict) and 'path' in audio_info:
                        self.audio_paths.append(audio_info['path'])
                    elif isinstance(audio_info, str):
                        self.audio_paths.append(audio_info)
                        
                if 'label' in item:
                    self.labels.append(item['label'])
                elif 'target' in item:
                    self.labels.append(item['target'])
                    
        except Exception as e:
            warnings.warn(f"Error loading DADS: {e}")
            self._load_from_directory()
            
    def _load_from_directory(self):
        """Fallback: load from directory structure."""
        for label_dir in ['0', '1', 'no_drone', 'drone']:
            label_path = self.root / label_dir
            if label_path.exists():
                label = 0 if label_dir in ['0', 'no_drone'] else 1
                for audio_file in label_path.glob('*.wav'):
                    self.audio_paths.append(str(audio_file))
                    self.labels.append(label)


class AlEmadiDataset(BaseAudioDataset):
    """
    Sara Al-Emadi Drone Audio Dataset from GitHub.
    
    Source: https://github.com/saraalemadi/DroneAudioDataset
    
    Structure:
    - Binary: drone vs unknown (noise)
    - Multiclass: Bebop, AR, Phantom, Unknown
    
    Example:
        >>> dataset = AlEmadiDataset(root='data/external/alemadi', task='binary')
        >>> audio, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = 'data/external/alemadi',
        task: str = 'Binary_Drone_Audio',  # 'binary' or 'multiclass'
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.root = Path(root)
        self.task = task
        
        if task == 'Binary_Drone_Audio':
            self.label_names = {0: 'unknown', 1: 'yes'}
        else:
            self.label_names = {0: 'Unknown', 1: 'Bebop', 2: 'AR', 3: 'Phantom'}
            
        self._load_data()
        
    def _load_data(self):
        """Load dataset from directory structure."""
        # Try different possible directory structures
        possible_paths = [
            self.root / 'DroneAudioDataset' / self.task,
            self.root / self.task,
            self.root / ('Binary' if self.task == 'binary' else 'Multiclass'),
            self.root / ('Binary_Drone_Audio' if self.task == 'binary' else 'Multiclass_Drone_Audio'),
            self.root,
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                # Check if this path has the expected subfolders
                subfolders = [f.name.lower() for f in path.iterdir() if f.is_dir()]
                if any('drone' in s or 'unknown' in s or 'bebop' in s for s in subfolders):
                    data_path = path
                    break
                    
        if data_path is None:
            warnings.warn(f"Al-Emadi dataset not found at {self.root}")
            return
        
        print(f"Found Al-Emadi data at: {data_path}")
            
        # Load based on task
        if self.task == 'binary':
            # Binary structure - handle various naming conventions
            for subfolder in data_path.iterdir():
                if not subfolder.is_dir():
                    continue
                    
                folder_name = subfolder.name.lower()
                
                # Determine label based on folder name
                if 'drone' in folder_name and 'unknown' not in folder_name and 'no' not in folder_name:
                    label = 1  # Drone
                elif 'unknown' in folder_name or 'noise' in folder_name or 'no_drone' in folder_name:
                    label = 0  # Not drone
                else:
                    continue  # Skip unrecognized folders
                
                # Load all wav files from this folder
                for audio_file in subfolder.glob('**/*.wav'):
                    self.audio_paths.append(str(audio_file))
                    self.labels.append(label)
                    
        else:
            # Multiclass structure
            label_mapping = {
                'unknown': 0,
                'bebop': 1,
                'bebop_1': 1,
                'ar': 2,
                'mambo': 2,
                'membo_1': 2,
                'mambo_1': 2,
                'phantom': 3,
                'phantom_1': 3,
            }
            
            for subfolder in data_path.iterdir():
                if not subfolder.is_dir():
                    continue
                    
                folder_name = subfolder.name.lower()
                
                # Find matching label
                label = None
                for key, value in label_mapping.items():
                    if key in folder_name:
                        label = value
                        break
                
                if label is None:
                    print(f"  Skipping unrecognized folder: {subfolder.name}")
                    continue
                
                # Load all wav files from this folder
                for audio_file in subfolder.glob('**/*.wav'):
                    self.audio_paths.append(str(audio_file))
                    self.labels.append(label)
                        
        print(f"Loaded {len(self.audio_paths)} samples from Al-Emadi dataset")
        
        # Show distribution
        if self.audio_paths:
            from collections import Counter
            dist = Counter(self.labels)
            for label_id, count in sorted(dist.items()):
                label_name = self.label_names.get(label_id, f"Label {label_id}")
                print(f"  - {label_name}: {count} samples")


class DroneThesisDataset(BaseAudioDataset):
    """
    DroneDetectionThesis Multi-Sensor Dataset from GitHub.
    
    Source: https://github.com/DroneDetectionThesis/Drone-detection-dataset
    
    Contains 90 audio clips: drones, helicopters, background noise
    
    Example:
        >>> dataset = DroneThesisDataset(root='data/external/drone_detection_thesis')
        >>> audio, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = 'data/external/drone_detection_thesis',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.root = Path(root)
        self.label_names = {0: 'background', 1: 'drone', 2: 'helicopter'}
        
        self._load_data()
        
    def _load_data(self):
        """Load dataset from directory structure."""
        # Try to find audio directory
        possible_paths = [
            self.root / 'Drone-detection-dataset' / 'Audio',
            self.root / 'Audio',
            self.root / 'audio',
            self.root,
        ]
        
        audio_path = None
        for path in possible_paths:
            if path.exists():
                audio_path = path
                break
                
        if audio_path is None:
            warnings.warn(f"DroneThesis dataset not found at {self.root}")
            return
            
        # Load all audio files
        # Try to infer labels from filename or directory
        for audio_file in audio_path.glob('**/*.wav'):
            filename = audio_file.name.lower()
            
            # Infer label from filename
            if 'drone' in filename:
                label = 1
            elif 'heli' in filename or 'helicopter' in filename:
                label = 2
            else:
                label = 0  # background/noise
                
            self.audio_paths.append(str(audio_file))
            self.labels.append(label)
            
        print(f"Loaded {len(self.audio_paths)} samples from DroneThesis dataset")

class AcoLabDataset(BaseAudioDataset):
    """
    AcoLab Dataset - Custom recordings from ITU Acoustic Lab.
    """
    
    def __init__(
        self,
        root: str = 'data/external/acolab',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.label_names = {0: 'no_drone', 1: 'drone'}
        self._load_data()
        
    def _load_data(self):
        for label_name, label_id in [('no_drone', 0), ('drone', 1)]:
            label_path = self.root / label_name
            if label_path.exists():
                for audio_file in label_path.glob('*.wav'):
                    self.audio_paths.append(str(audio_file))
                    self.labels.append(label_id)
        
        print(f"Loaded {len(self.audio_paths)} samples from AcoLab dataset")
        if self.audio_paths:
            from collections import Counter
            dist = Counter(self.labels)
            for label_id, count in sorted(dist.items()):
                print(f"  - {self.label_names[label_id]}: {count} samples")

class DroneAudiosetDataset(BaseAudioDataset):
    """
    DroneAudioset - Search and Rescue Dataset from Hugging Face.
    
    Source: https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/
    
    23.5 hours of annotated recordings with various SNR conditions.
    
    Example:
        >>> dataset = DroneAudiosetDataset(root='data/external/droneaudioset')
        >>> audio, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = 'data/external/droneaudioset',
        download: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.root = Path(root)
        
        if download:
            self._download()
            
        self._load_data()
        
    def _download(self):
        """Download from Hugging Face."""
        if not HF_AVAILABLE:
            raise RuntimeError("Please install datasets: pip install datasets")
            
        print("Downloading DroneAudioset from Hugging Face...")
        dataset = load_dataset("ahlab-drone-project/DroneAudioSet")
        
        self.root.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(self.root))
        print(f"Dataset saved to {self.root}")
        
    def _load_data(self):
        """Load dataset."""
        # Similar to DADS loading
        if not self.root.exists():
            warnings.warn(f"DroneAudioset not found at {self.root}")
            return
            
        # Try loading as HF dataset or from directory
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(self.root))
            # Process similar to DADS
        except:
            # Fallback to directory loading
            for audio_file in self.root.glob('**/*.wav'):
                self.audio_paths.append(str(audio_file))
                self.labels.append(1)  # Assume all are drone audio


class CombinedDroneDataset(Dataset):
    """
    Combines multiple drone datasets into a unified dataset.
    
    This allows training on all available data with consistent labels.
    
    Example:
        >>> combined = CombinedDroneDataset(
        ...     data_root='data/external',
        ...     datasets=['dads', 'alemadi', 'thesis'],
        ...     task='binary'
        ... )
        >>> audio, label = combined[0]
    """
    
    def __init__(
        self,
        data_root: str = 'data/external',
        datasets: List[str] = ['dads', 'alemadi', 'thesis'],
        task: str = 'binary',  # 'binary' for drone/no-drone
        sample_rate: int = 16000,
        duration: float = 3.0,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.task = task
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        
        # Binary labels: 0 = no drone, 1 = drone
        self.label_names = {0: 'no_drone', 1: 'drone'}
        
        self.audio_paths = []
        self.labels = []
        
        # Load each dataset
        for ds_name in datasets:
            self._load_dataset(ds_name)
            
        print(f"\nCombined dataset: {len(self.audio_paths)} total samples")
        print(f"Label distribution: {self.get_label_distribution()}")
        
    def _load_dataset(self, name: str):
        """Load a specific dataset and add to combined data."""
        name = name.lower()
        
        try:
            if name == 'dads':
                ds = DADSDataset(
                    root=self.data_root / 'dads',
                    sample_rate=self.sample_rate,
                    duration=self.duration
                )
            elif name == 'alemadi':
                ds = AlEmadiDataset(
                    root=self.data_root / 'alemadi',
                    task='binary',
                    sample_rate=self.sample_rate,
                    duration=self.duration
                )
            elif name == 'thesis':
                ds = DroneThesisDataset(
                    root=self.data_root / 'drone_detection_thesis',
                    sample_rate=self.sample_rate,
                    duration=self.duration
                )
            elif name == 'acolab':
                ds = AcoLabDataset(
                root=self.data_root / 'acolab',
                sample_rate=self.sample_rate,
                duration=self.duration
                )
            else:
                warnings.warn(f"Unknown dataset: {name}")
                return
                
            # Convert labels to binary if needed
            for path, label in zip(ds.audio_paths, ds.labels):
                self.audio_paths.append(path)
                # Map to binary: anything > 0 is drone
                if self.task == 'binary':
                    self.labels.append(1 if label > 0 else 0)
                else:
                    self.labels.append(label)
                    
            print(f"  Loaded {len(ds)} samples from {name}")
            
        except Exception as e:
            warnings.warn(f"Error loading {name}: {e}")
            
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except:
            audio = np.zeros(int(self.sample_rate * self.duration))
            
        # Pad/truncate
        target_len = int(self.sample_rate * self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            start = np.random.randint(0, max(1, len(audio) - target_len))
            audio = audio[start:start + target_len]
            
        audio = torch.from_numpy(audio).float()
        
        if self.transform:
            audio = self.transform(audio)
            
        return audio, label
    
    def get_label_distribution(self) -> Dict[str, int]:
        from collections import Counter
        counts = Counter(self.labels)
        return {self.label_names.get(k, str(k)): v for k, v in counts.items()}


def create_dataloaders(
    data_root: str = 'data/external',
    datasets: List[str] = ['alemadi'],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 3.0,
    transform: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from available datasets.
    
    Args:
        data_root: Root directory containing datasets
        datasets: List of dataset names to use
        batch_size: Batch size
        val_split: Fraction for validation
        test_split: Fraction for test
        num_workers: Number of data loading workers
        sample_rate: Target sample rate
        duration: Audio clip duration in seconds
        transform: Optional transform to apply
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
        
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     data_root='data/external',
        ...     datasets=['alemadi', 'thesis'],
        ...     batch_size=32
        ... )
        >>> for audio, labels in train_loader:
        ...     print(audio.shape, labels.shape)
    """
    # Create combined dataset
    dataset = CombinedDroneDataset(
        data_root=data_root,
        datasets=datasets,
        sample_rate=sample_rate,
        duration=duration,
        transform=transform
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Quick Test
# =============================================================================
if __name__ == "__main__":
    print("Testing dataset loaders...")
    print("=" * 60)
    
    # Test Al-Emadi dataset (most likely to be available)
    print("\n1. Testing Al-Emadi Dataset Loader")
    try:
        dataset = AlEmadiDataset(
            root='data/external/alemadi',
            task='binary',
            sample_rate=16000,
            duration=3.0
        )
        print(f"   Samples: {len(dataset)}")
        print(f"   Distribution: {dataset.get_label_distribution()}")
        
        if len(dataset) > 0:
            audio, label = dataset[0]
            print(f"   Audio shape: {audio.shape}")
            print(f"   Label: {label}")
    except Exception as e:
        print(f"   Error: {e}")
        
    # Test combined dataset
    print("\n2. Testing Combined Dataset")
    try:
        combined = CombinedDroneDataset(
            data_root='data/external',
            datasets=['alemadi', 'thesis'],
        )
        print(f"   Total samples: {len(combined)}")
    except Exception as e:
        print(f"   Error: {e}")
        
    # Test dataloader creation
    print("\n3. Testing DataLoader Creation")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root='data/external',
            datasets=['alemadi'],
            batch_size=16
        )
        
        # Get one batch
        for audio, labels in train_loader:
            print(f"   Batch audio shape: {audio.shape}")
            print(f"   Batch labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"   Error: {e}")
        
    print("\n" + "=" * 60)
    print("Dataset loader tests complete!")
