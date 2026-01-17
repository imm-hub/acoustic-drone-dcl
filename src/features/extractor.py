"""
Feature Extraction Module for Acoustic Drone Detection

This module provides various audio feature extraction methods commonly used
in acoustic drone detection research.

Author: ITU Telecommunication Engineering
"""

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Dict, Tuple, Optional, Union
from pathlib import Path


class AudioFeatureExtractor:
    """
    Comprehensive audio feature extractor for drone acoustic analysis.
    
    Supports:
    - Mel Spectrogram
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - STFT (Short-Time Fourier Transform)
    - Spectral features (contrast, centroid, bandwidth, rolloff)
    - Chroma features
    
    Example:
        >>> extractor = AudioFeatureExtractor(sample_rate=16000)
        >>> audio, sr = librosa.load('drone.wav', sr=16000)
        >>> features = extractor.extract_all(audio)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCCs to compute
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank (None = sr/2)
            device: Computation device ('cuda' or 'cpu')
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.device = device
        
        # Initialize torchaudio transforms (GPU accelerated)
        self._init_transforms()
        
    def _init_transforms(self):
        """Initialize torchaudio transforms for GPU acceleration."""
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
        ).to(self.device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'f_min': self.fmin,
                'f_max': self.fmax,
            }
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
    def load_audio(
        self, 
        path: Union[str, Path], 
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            path: Path to audio file
            duration: Duration in seconds to load (None = full file)
            offset: Start time in seconds
            
        Returns:
            audio: Audio waveform as numpy array
            sr: Sample rate
        """
        audio, sr = librosa.load(
            path,
            sr=self.sample_rate,
            duration=duration,
            offset=offset,
            mono=True
        )
        return audio, sr
    
    def mel_spectrogram(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        to_db: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute mel spectrogram.
        
        Args:
            audio: Audio waveform (numpy array or torch tensor)
            to_db: Convert to decibel scale
            normalize: Normalize to [0, 1] range
            
        Returns:
            Mel spectrogram tensor of shape (n_mels, time_frames)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        audio = audio.to(self.device)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        mel_spec = self.mel_transform(audio)
        
        if to_db:
            mel_spec = self.amplitude_to_db(mel_spec)
            
        if normalize:
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
            
        return mel_spec.squeeze(0)
    
    def mfcc(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        delta: bool = True,
        delta_delta: bool = True
    ) -> torch.Tensor:
        """
        Compute MFCCs with optional delta and delta-delta features.
        
        Args:
            audio: Audio waveform
            delta: Include first derivative
            delta_delta: Include second derivative
            
        Returns:
            MFCC tensor, optionally stacked with deltas
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        audio = audio.to(self.device)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        mfcc = self.mfcc_transform(audio).squeeze(0)
        
        features = [mfcc]
        
        if delta:
            delta_mfcc = torchaudio.functional.compute_deltas(mfcc)
            features.append(delta_mfcc)
            
        if delta_delta:
            delta2_mfcc = torchaudio.functional.compute_deltas(
                torchaudio.functional.compute_deltas(mfcc)
            )
            features.append(delta2_mfcc)
            
        return torch.cat(features, dim=0)
    
    def stft(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        return_magnitude: bool = True
    ) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio waveform
            return_magnitude: Return magnitude (True) or complex (False)
            
        Returns:
            STFT tensor
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        audio = audio.to(self.device)
        
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        if return_magnitude:
            return torch.abs(stft)
        return stft
    
    def spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute various spectral features using librosa.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dictionary containing spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )[0]
        
        return features
    
    def chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chroma features.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Chroma feature matrix
        """
        return librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
    
    def extract_all(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        include_spectral: bool = True
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Extract all features from audio.
        
        Args:
            audio: Audio waveform
            include_spectral: Include librosa spectral features
            
        Returns:
            Dictionary containing all extracted features
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio
            
        features = {
            'mel_spectrogram': self.mel_spectrogram(audio),
            'mfcc': self.mfcc(audio),
            'stft_magnitude': self.stft(audio),
        }
        
        if include_spectral:
            features['spectral'] = self.spectral_features(audio_np)
            features['chroma'] = self.chroma(audio_np)
            
        return features
    
    def extract_for_cnn(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        feature_type: str = 'mel_spectrogram',
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract features formatted for CNN input (C, H, W).
        
        Args:
            audio: Audio waveform
            feature_type: Type of feature ('mel_spectrogram', 'mfcc', 'stft')
            target_length: Pad/truncate to this length (time dimension)
            
        Returns:
            Feature tensor of shape (1, freq_bins, time_frames)
        """
        if feature_type == 'mel_spectrogram':
            features = self.mel_spectrogram(audio)
        elif feature_type == 'mfcc':
            features = self.mfcc(audio, delta=True, delta_delta=True)
        elif feature_type == 'stft':
            features = self.stft(audio)
            features = self.amplitude_to_db(features)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        # Ensure 3D tensor (C, H, W)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        # Pad or truncate time dimension
        if target_length is not None:
            current_length = features.shape[-1]
            if current_length < target_length:
                # Pad
                padding = target_length - current_length
                features = torch.nn.functional.pad(features, (0, padding))
            elif current_length > target_length:
                # Truncate
                features = features[..., :target_length]
                
        return features


class FeatureDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading and extracting features from audio files.
    
    Example:
        >>> dataset = FeatureDataset(
        ...     audio_paths=['drone1.wav', 'drone2.wav', 'noise1.wav'],
        ...     labels=[1, 1, 0],
        ...     feature_type='mel_spectrogram'
        ... )
        >>> features, label = dataset[0]
    """
    
    def __init__(
        self,
        audio_paths: list,
        labels: list,
        feature_type: str = 'mel_spectrogram',
        sample_rate: int = 16000,
        duration: float = 3.0,
        augment: bool = False,
        target_length: Optional[int] = None,
        **extractor_kwargs
    ):
        """
        Initialize dataset.
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels corresponding to audio files
            feature_type: Feature extraction method
            sample_rate: Target sample rate
            duration: Duration to load from each file
            augment: Apply data augmentation
            target_length: Target time dimension length
            **extractor_kwargs: Additional arguments for AudioFeatureExtractor
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_type = feature_type
        self.duration = duration
        self.augment = augment
        self.target_length = target_length
        
        self.extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            **extractor_kwargs
        )
        
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio, _ = self.extractor.load_audio(audio_path, duration=self.duration)
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment(audio)
            
        # Extract features
        features = self.extractor.extract_for_cnn(
            audio, 
            feature_type=self.feature_type,
            target_length=self.target_length
        )
        
        return features, label
    
    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to audio."""
        # Time stretch
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            
        # Pitch shift
        if np.random.random() < 0.5:
            steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(
                audio, sr=self.extractor.sample_rate, n_steps=steps
            )
            
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
            
        return audio


def compute_blade_passing_frequency(
    rpm: float, 
    num_blades: int
) -> float:
    """
    Compute the blade passing frequency (BPF) for a drone propeller.
    
    BPF is a key acoustic signature for drone identification.
    
    Args:
        rpm: Rotations per minute
        num_blades: Number of blades on propeller
        
    Returns:
        Blade passing frequency in Hz
        
    Example:
        >>> # DJI Phantom 4 at hover (~4100 RPM, 2-blade props)
        >>> bpf = compute_blade_passing_frequency(4100, 2)
        >>> print(f"BPF: {bpf:.1f} Hz")  # ~136.7 Hz
    """
    rotation_frequency = rpm / 60  # Convert to Hz
    bpf = rotation_frequency * num_blades
    return bpf


def estimate_rpm_from_spectrum(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_blades: int = 2,
    rpm_range: Tuple[int, int] = (2000, 8000)
) -> Tuple[float, float]:
    """
    Estimate RPM from audio spectrum by finding blade passing frequency.
    
    This is an experimental feature for determining propeller characteristics.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        num_blades: Assumed number of blades
        rpm_range: Expected RPM range (min, max)
        
    Returns:
        Estimated RPM and confidence score
    """
    # Compute power spectrum
    fft = np.fft.rfft(audio)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Convert RPM range to frequency range
    min_bpf = rpm_range[0] / 60 * num_blades
    max_bpf = rpm_range[1] / 60 * num_blades
    
    # Find frequency range indices
    freq_mask = (freqs >= min_bpf) & (freqs <= max_bpf)
    masked_freqs = freqs[freq_mask]
    masked_power = power[freq_mask]
    
    if len(masked_power) == 0:
        return 0.0, 0.0
        
    # Find dominant frequency
    peak_idx = np.argmax(masked_power)
    peak_freq = masked_freqs[peak_idx]
    
    # Convert back to RPM
    estimated_rpm = (peak_freq / num_blades) * 60
    
    # Confidence based on peak prominence
    confidence = masked_power[peak_idx] / (np.mean(masked_power) + 1e-8)
    confidence = min(confidence / 10, 1.0)  # Normalize
    
    return estimated_rpm, confidence


if __name__ == "__main__":
    # Quick test
    print("Testing AudioFeatureExtractor...")
    
    # Create dummy audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate drone sound (fundamental + harmonics)
    bpf = 136.7  # Blade passing frequency
    audio = np.sin(2 * np.pi * bpf * t)  # Fundamental
    audio += 0.5 * np.sin(2 * np.pi * 2 * bpf * t)  # 2nd harmonic
    audio += 0.25 * np.sin(2 * np.pi * 3 * bpf * t)  # 3rd harmonic
    audio += 0.1 * np.random.randn(len(t))  # Noise
    audio = audio.astype(np.float32)
    
    # Extract features
    extractor = AudioFeatureExtractor(sample_rate=sr)
    
    mel_spec = extractor.mel_spectrogram(audio)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    mfcc = extractor.mfcc(audio)
    print(f"MFCC (with deltas) shape: {mfcc.shape}")
    
    cnn_features = extractor.extract_for_cnn(audio, target_length=128)
    print(f"CNN-ready features shape: {cnn_features.shape}")
    
    # Test RPM estimation
    estimated_rpm, confidence = estimate_rpm_from_spectrum(audio, sr, num_blades=2)
    print(f"Estimated RPM: {estimated_rpm:.0f} (confidence: {confidence:.2f})")
    
    print("\nAll tests passed!")
