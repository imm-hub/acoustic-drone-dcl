import torch
from src.detection.models import create_model

# Load model
model = create_model('custom_cnn', num_classes=2)
checkpoint = torch.load('experiments/baseline_v1/checkpoints/best_model.pt', map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print("Model loaded successfully!")