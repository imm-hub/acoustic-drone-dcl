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

# Models to test
experiments = {
    
    'combined_acolab_v1 (with acolab)': ('experiments/combined_acolab_v2/checkpoints/best_model.pt', 'efficientnet_b0'),
}

# Get drone files
drone_files = sorted(Path(DATA_PATH).glob('*.wav'))
total_files = len(drone_files)

print("=" * 60)
print(f"TESTING ON {total_files} REAL DRONE RECORDINGS")
print("=" * 60)

for exp_name, (model_path, model_type) in experiments.items():
    if not Path(model_path).exists():
        print(f"\n{exp_name}: not found")
        continue
    
    model = create_model(model_type, num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    correct = 0
    for audio_file in drone_files:
        pred = predict(model, audio_file)
        if pred == 1:
            correct += 1
    
    accuracy = 100 * correct / total_files
    print(f"\n{exp_name}")
    print(f"  Accuracy: {correct}/{total_files} ({accuracy:.1f}%)")

print("\n" + "=" * 60)