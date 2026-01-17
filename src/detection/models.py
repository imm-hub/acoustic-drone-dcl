"""
Detection Models for Acoustic Drone Detection

This module provides neural network architectures for binary drone detection
(drone vs. no-drone classification).

Author: ITU Telecommunication Engineering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import timm


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for drone audio classification.
    
    Designed for mel-spectrogram input with shape (1, n_mels, time_frames).
    
    Args:
        in_channels: Number of input channels (1 for single spectrogram)
        num_classes: Number of output classes
        channels: List of channel sizes for conv layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: list = [32, 64, 128, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Build convolutional layers
        conv_layers = []
        prev_channels = in_channels
        
        for ch in channels:
            conv_layers.append(ConvBlock(prev_channels, ch, dropout=dropout))
            conv_layers.append(nn.MaxPool2d(2))
            prev_channels = ch
            
        self.features = nn.Sequential(*conv_layers)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before final classification."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return x


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for temporal audio analysis.
    
    Combines CNN for local feature extraction with LSTM for temporal modeling.
    Particularly effective for capturing temporal dynamics in drone audio.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        cnn_channels: List of CNN channel sizes
        rnn_hidden: Hidden size for LSTM
        rnn_layers: Number of LSTM layers
        bidirectional: Use bidirectional LSTM
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        cnn_channels: list = [32, 64, 128],
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # CNN feature extractor
        conv_layers = []
        prev_channels = in_channels
        
        for ch in cnn_channels:
            conv_layers.append(ConvBlock(prev_channels, ch, dropout=dropout))
            conv_layers.append(nn.MaxPool2d((2, 1)))  # Pool only in frequency
            prev_channels = ch
            
        self.cnn = nn.Sequential(*conv_layers)
        
        # Calculate CNN output size (after frequency pooling)
        # Assuming input is (batch, 1, n_mels, time)
        self.freq_pool_factor = 2 ** len(cnn_channels)
        
        # LSTM
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Classifier
        rnn_output_size = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, freq, time)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_channels[-1], freq', time)
        
        # Reshape for RNN: combine freq bins, keep time as sequence
        x = x.mean(dim=2)  # Global average over frequency -> (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        
        # LSTM
        x, _ = self.rnn(x)  # (batch, time, rnn_hidden*2)
        
        # Use last timestep
        x = x[:, -1, :]
        
        # Classify
        x = self.classifier(x)
        return x


class AttentionCRNN(nn.Module):
    """
    CRNN with attention mechanism for improved temporal focus.
    
    The attention mechanism learns to weight important time frames
    in the audio, which can be useful for detecting intermittent drone sounds.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        cnn_channels: list = [32, 64, 128],
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # CNN
        conv_layers = []
        prev_channels = in_channels
        for ch in cnn_channels:
            conv_layers.append(ConvBlock(prev_channels, ch, dropout=dropout))
            conv_layers.append(nn.MaxPool2d((2, 1)))
            prev_channels = ch
        self.cnn = nn.Sequential(*conv_layers)
        
        # LSTM
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CNN
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        
        # LSTM
        rnn_out, _ = self.rnn(x)  # (batch, time, hidden*2)
        
        # Attention weights
        attn_weights = self.attention(rnn_out)  # (batch, time, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(rnn_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Classify
        output = self.classifier(context)
        
        return output, attn_weights.squeeze(-1)


class PretrainedCNN(nn.Module):
    """
    Wrapper for pretrained CNN models from timm.
    
    Supports EfficientNet, ResNet, VGG, etc. with transfer learning.
    Adapts single-channel spectrogram input to 3-channel pretrained models.
    
    Args:
        model_name: Name of pretrained model (e.g., 'efficientnet_b0', 'resnet18')
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate for classifier
        freeze_backbone: Freeze pretrained layers initially
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=1  # Single channel input
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        return self.backbone(x)


def create_model(
    architecture: str,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create detection models.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
        
    Supported architectures:
        - 'custom_cnn': Custom CNN
        - 'crnn': Convolutional Recurrent Neural Network
        - 'attention_crnn': CRNN with attention
        - 'efficientnet_b0', 'efficientnet_b2': EfficientNet variants
        - 'resnet18', 'resnet34', 'resnet50': ResNet variants
        - 'vgg11', 'vgg16': VGG variants
    """
    architecture = architecture.lower()
    
    if architecture == 'custom_cnn':
        return CustomCNN(num_classes=num_classes, **kwargs)
    elif architecture == 'crnn':
        return CRNN(num_classes=num_classes, **kwargs)
    elif architecture == 'attention_crnn':
        return AttentionCRNN(num_classes=num_classes, **kwargs)
    elif architecture in ['efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4',
                          'resnet18', 'resnet34', 'resnet50',
                          'vgg11', 'vgg16', 'vgg19']:
        return PretrainedCNN(model_name=architecture, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test models
    print("Testing detection models...")
    
    # Dummy input (batch, channels, freq_bins, time_frames)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    
    print(f"\nInput shape: {x.shape}")
    print("-" * 50)
    
    # Test each architecture
    architectures = ['custom_cnn', 'crnn', 'attention_crnn', 'efficientnet_b0']
    
    for arch in architectures:
        print(f"\n{arch.upper()}")
        model = create_model(arch, num_classes=2, dropout=0.3)
        
        total, trainable = count_parameters(model)
        print(f"  Parameters: {total:,} total, {trainable:,} trainable")
        
        # Forward pass
        if arch == 'attention_crnn':
            output, attention = model(x)
            print(f"  Output shape: {output.shape}")
            print(f"  Attention shape: {attention.shape}")
        else:
            output = model(x)
            print(f"  Output shape: {output.shape}")
            
    print("\n" + "=" * 50)
    print("All model tests passed!")
