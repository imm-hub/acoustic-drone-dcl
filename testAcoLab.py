import torch
import librosa
import numpy as np
from pathlib import Path
from src.detection.models import create_model
from src.features.extractor import AudioFeatureExtractor

# Settings
DATA_PATH = 'data/raw/drone_dataset/drone'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature extractor
extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=128, device='cpu')

def predict(model, audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
    if len(y) < 48000:
        y = np.pad(y, (0, 48000 - len(y)))
    
    mel = extractor.mel_spectrogram(y, normalize=True)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    mel = mel.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(mel)
        pred = output.argmax(dim=1).item()
    
    return pred

# Define experiments and their model types
experiments = {
    'baseline_v1': 'custom_cnn',
    'combined_v1': 'efficientnet_b0',
    'alemadi_only': 'efficientnet_b0',
    'efficientnet_v1': 'efficientnet_b0',
    'combined_acolab_v1': 'efficientnet_b0',
}

# Get all drone files
drone_files = sorted(Path(DATA_PATH).glob('*.wav'))
total_files = len(drone_files)

print("=" * 60)
print(f"TESTING ALL MODELS ON {total_files} DRONE FILES")
print("=" * 60)

results = {}

for exp_name, model_type in experiments.items():
    checkpoint_path = Path(f'experiments/{exp_name}/checkpoints/best_model.pt')
    
    if not checkpoint_path.exists():
        print(f"\n{exp_name}: checkpoint not found, skipping...")
        continue
    
    # Load model - handle pretrained argument
    if model_type in ['efficientnet_b0', 'efficientnet_b2', 'resnet18', 'resnet34', 'vgg11']:
        model = create_model(model_type, num_classes=2, pretrained=False)
    else:
        model = create_model(model_type, num_classes=2)
    
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Test all files
    correct = 0
    for audio_file in drone_files:
        pred = predict(model, audio_file)
        if pred == 1:
            correct += 1
    
    accuracy = 100 * correct / total_files
    results[exp_name] = accuracy
    
    print(f"\n{exp_name} ({model_type})")
    print(f"  Accuracy: {correct}/{total_files} ({accuracy:.1f}%)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY - REAL DRONE DATA ACCURACY")
print("=" * 60)
for exp_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {exp_name}: {acc:.1f}%")