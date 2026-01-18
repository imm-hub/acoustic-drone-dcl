"""Quick evaluation script"""
import torch
import json
from pathlib import Path
from src.detection.models import create_model
from src.data.dataset import create_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Settings
EXPERIMENT = 'baseline_v1'
MODEL_TYPE = 'custom_cnn'

# Load model
model = create_model(MODEL_TYPE, num_classes=2)
checkpoint = torch.load(f'experiments/{EXPERIMENT}/checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.cuda()

# Load test data
_, _, test_loader = create_dataloaders(
    data_root='data/external',
    datasets=['alemadi'],
    batch_size=32,
    num_workers=0
)

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for audio, labels in test_loader:
        # Simple approach: use raw audio reshaped as "image"
        # This matches training transform
        audio = audio.cuda()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim
        
        outputs = model(audio)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"\nExperiment: {EXPERIMENT}")
print(f"Model: {MODEL_TYPE}")
print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['No Drone', 'Drone']))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Save results
results = {
    'experiment': EXPERIMENT,
    'model': MODEL_TYPE,
    'best_epoch': checkpoint.get('epoch'),
    'val_loss': float(checkpoint.get('best_val_loss', 0)),
    'test_samples': len(all_labels),
    'accuracy': float(np.mean(np.array(all_preds) == np.array(all_labels))),
    'confusion_matrix': cm.tolist()
}

with open(f'experiments/{EXPERIMENT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to experiments/{EXPERIMENT}/results.json")