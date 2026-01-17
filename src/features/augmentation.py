"""
Data Augmentation Module for Acoustic Drone Detection

Provides audio-specific augmentation techniques to improve model robustness.

Author: ITU Telecommunication Engineering
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Union, Tuple, Optional, Callable
import random


class AudioAugmentation:
    """
    Audio augmentation pipeline for drone acoustic data.
    
    Supports both time-domain and spectrogram augmentations.
    
    Example:
        >>> aug = AudioAugmentation(sample_rate=16000)
        >>> augmented_audio = aug(audio)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        p: float = 0.5  # Probability of applying each augmentation
    ):
        self.sample_rate = sample_rate
        self.p = p
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to audio."""
        if random.random() < self.p:
            audio = self.time_stretch(audio)
        if random.random() < self.p:
            audio = self.pitch_shift(audio)
        if random.random() < self.p:
            audio = self.add_noise(audio)
        if random.random() < self.p:
            audio = self.random_gain(audio)
        return audio
    
    def time_stretch(
        self,
        audio: np.ndarray,
        rate_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Time stretching without changing pitch.
        
        Args:
            audio: Input audio
            rate_range: Range of stretch rates
            
        Returns:
            Stretched audio
        """
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        steps_range: Tuple[int, int] = (-2, 2)
    ) -> np.ndarray:
        """
        Pitch shifting.
        
        Args:
            audio: Input audio
            steps_range: Range of semitone shifts
            
        Returns:
            Pitch-shifted audio
        """
        n_steps = np.random.randint(*steps_range)
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )
    
    def add_noise(
        self,
        audio: np.ndarray,
        snr_range: Tuple[float, float] = (10, 30)
    ) -> np.ndarray:
        """
        Add Gaussian noise at specified SNR.
        
        Args:
            audio: Input audio
            snr_range: Range of SNR values in dB
            
        Returns:
            Noisy audio
        """
        snr_db = np.random.uniform(*snr_range)
        
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
        
        return audio + noise
    
    def add_background_noise(
        self,
        audio: np.ndarray,
        noise_audio: np.ndarray,
        snr_range: Tuple[float, float] = (5, 20)
    ) -> np.ndarray:
        """
        Add real background noise at specified SNR.
        
        Args:
            audio: Input audio (drone sound)
            noise_audio: Background noise to add
            snr_range: Range of SNR values in dB
            
        Returns:
            Mixed audio
        """
        snr_db = np.random.uniform(*snr_range)
        
        # Match lengths
        if len(noise_audio) > len(audio):
            start = np.random.randint(0, len(noise_audio) - len(audio))
            noise_audio = noise_audio[start:start + len(audio)]
        else:
            # Repeat noise
            noise_audio = np.tile(noise_audio, int(np.ceil(len(audio) / len(noise_audio))))
            noise_audio = noise_audio[:len(audio)]
            
        # Calculate powers
        signal_power = np.mean(audio ** 2) + 1e-8
        noise_power = np.mean(noise_audio ** 2) + 1e-8
        
        # Scale noise to achieve desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        return audio + scale * noise_audio
    
    def random_gain(
        self,
        audio: np.ndarray,
        gain_range: Tuple[float, float] = (0.5, 1.5)
    ) -> np.ndarray:
        """
        Apply random gain.
        
        Args:
            audio: Input audio
            gain_range: Range of gain values
            
        Returns:
            Gained audio
        """
        gain = np.random.uniform(*gain_range)
        return audio * gain
    
    def random_crop(
        self,
        audio: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Randomly crop audio to target length.
        
        Args:
            audio: Input audio
            target_length: Target length in samples
            
        Returns:
            Cropped audio
        """
        if len(audio) <= target_length:
            # Pad if too short
            padding = target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
            
        start = np.random.randint(0, len(audio) - target_length)
        return audio[start:start + target_length]


class SpecAugment(nn.Module):
    """
    SpecAugment: Augmentation for spectrogram inputs.
    
    Implements frequency masking and time masking as described in:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    
    Args:
        freq_mask_param: Maximum width of frequency mask
        time_mask_param: Maximum width of time mask
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        mask_value: Value to use for masking (default: mean of spectrogram)
    """
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: Optional[float] = None
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (C, F, T) or (F, T)
            
        Returns:
            Augmented spectrogram
        """
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        spectrogram = spectrogram.clone()
        _, n_freq, n_time = spectrogram.shape
        
        mask_value = self.mask_value if self.mask_value is not None else spectrogram.mean()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = min(np.random.randint(0, self.freq_mask_param + 1), n_freq)
            f0 = np.random.randint(0, max(1, n_freq - f))
            spectrogram[:, f0:f0 + f, :] = mask_value
            
        # Time masking
        for _ in range(self.num_time_masks):
            t = min(np.random.randint(0, self.time_mask_param + 1), n_time)
            t0 = np.random.randint(0, max(1, n_time - t))
            spectrogram[:, :, t0:t0 + t] = mask_value
            
        if squeeze:
            spectrogram = spectrogram.squeeze(0)
            
        return spectrogram


class MixUp:
    """
    MixUp augmentation for spectrogram classification.
    
    Creates virtual training examples by linearly combining pairs of samples.
    
    Args:
        alpha: Beta distribution parameter for mixing ratio
    """
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        
    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp.
        
        Args:
            x1, y1: First sample and label
            x2, y2: Second sample and label
            
        Returns:
            Mixed sample, label1, label2, mixing ratio
        """
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed, y1, y2, lam


class CutMix:
    """
    CutMix augmentation for spectrogram classification.
    
    Replaces part of the spectrogram with a patch from another sample.
    
    Args:
        alpha: Beta distribution parameter
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix.
        
        Args:
            x1, y1: First sample and label
            x2, y2: Second sample and label
            
        Returns:
            Mixed sample, label1, label2, mixing ratio
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get dimensions
        if x1.dim() == 3:
            _, h, w = x1.shape
        else:
            h, w = x1.shape
            
        # Calculate cut region
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random position
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        
        # Boundary
        y1_cut = np.clip(cy - cut_h // 2, 0, h)
        y2_cut = np.clip(cy + cut_h // 2, 0, h)
        x1_cut = np.clip(cx - cut_w // 2, 0, w)
        x2_cut = np.clip(cx + cut_w // 2, 0, w)
        
        # Apply cut
        x_mixed = x1.clone()
        if x1.dim() == 3:
            x_mixed[:, y1_cut:y2_cut, x1_cut:x2_cut] = x2[:, y1_cut:y2_cut, x1_cut:x2_cut]
        else:
            x_mixed[y1_cut:y2_cut, x1_cut:x2_cut] = x2[y1_cut:y2_cut, x1_cut:x2_cut]
            
        # Adjust lambda based on actual cut region
        lam = 1.0 - (y2_cut - y1_cut) * (x2_cut - x1_cut) / (h * w)
        
        return x_mixed, y1, y2, lam


class AugmentationPipeline:
    """
    Combined augmentation pipeline for training.
    
    Example:
        >>> pipeline = AugmentationPipeline(
        ...     audio_augment=True,
        ...     spec_augment=True,
        ...     mixup=True
        ... )
        >>> augmented_spec, label = pipeline(audio, label)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        audio_augment: bool = True,
        spec_augment: bool = True,
        audio_p: float = 0.5,
        spec_p: float = 0.5
    ):
        self.audio_augment = AudioAugmentation(sample_rate, p=audio_p) if audio_augment else None
        self.spec_augment = SpecAugment() if spec_augment else None
        self.spec_p = spec_p
        
    def __call__(
        self,
        audio: np.ndarray,
        spectrogram_transform: Callable = None
    ) -> torch.Tensor:
        """
        Apply augmentation pipeline.
        
        Args:
            audio: Raw audio
            spectrogram_transform: Function to convert audio to spectrogram
            
        Returns:
            Augmented spectrogram
        """
        # Audio augmentation
        if self.audio_augment is not None:
            audio = self.audio_augment(audio)
            
        # Convert to spectrogram
        if spectrogram_transform is not None:
            spectrogram = spectrogram_transform(audio)
        else:
            spectrogram = torch.from_numpy(audio).float()
            
        # Spectrogram augmentation
        if self.spec_augment is not None and random.random() < self.spec_p:
            spectrogram = self.spec_augment(spectrogram)
            
        return spectrogram


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentation module...")
    
    # Create dummy audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    # Test audio augmentations
    aug = AudioAugmentation(sample_rate=sr)
    
    print("\nAudio Augmentation:")
    print(f"  Original length: {len(audio)}")
    
    stretched = aug.time_stretch(audio)
    print(f"  After time stretch: {len(stretched)}")
    
    shifted = aug.pitch_shift(audio)
    print(f"  After pitch shift: {len(shifted)}")
    
    noisy = aug.add_noise(audio)
    print(f"  After adding noise: {len(noisy)}")
    
    # Test SpecAugment
    spec = torch.randn(1, 128, 128)
    spec_aug = SpecAugment()
    
    print("\nSpecAugment:")
    print(f"  Input shape: {spec.shape}")
    augmented_spec = spec_aug(spec)
    print(f"  Output shape: {augmented_spec.shape}")
    
    # Test MixUp
    mixup = MixUp(alpha=0.4)
    x1, y1 = torch.randn(1, 128, 128), torch.tensor(0)
    x2, y2 = torch.randn(1, 128, 128), torch.tensor(1)
    
    x_mixed, _, _, lam = mixup(x1, y1, x2, y2)
    print(f"\nMixUp:")
    print(f"  Lambda: {lam:.3f}")
    print(f"  Mixed shape: {x_mixed.shape}")
    
    # Test CutMix
    cutmix = CutMix(alpha=1.0)
    x_cut, _, _, lam = cutmix(x1, y1, x2, y2)
    print(f"\nCutMix:")
    print(f"  Lambda: {lam:.3f}")
    print(f"  Cut shape: {x_cut.shape}")
    
    print("\nAll augmentation tests passed!")
