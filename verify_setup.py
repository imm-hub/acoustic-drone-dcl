#!/usr/bin/env python3
"""
Quick Verification Script

Run this after setup to verify everything is working correctly.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"  ✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name or module_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("ACOUSTIC DRONE DETECTION - SETUP VERIFICATION")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Check Python version
    print("\n1. Python Version")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"  ✗ Python {py_version.major}.{py_version.minor} (need 3.8+)")
        all_ok = False
        
    # 2. Check core dependencies
    print("\n2. Core Dependencies")
    deps = [
        ('numpy', 'numpy'),
        ('torch', 'PyTorch'),
        ('torchaudio', 'torchaudio'),
        ('librosa', 'librosa'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    for module, name in deps:
        if not check_import(module, name):
            all_ok = False
            
    # 3. Check optional dependencies
    print("\n3. Optional Dependencies")
    optional_deps = [
        ('datasets', 'Hugging Face datasets'),
        ('timm', 'timm (pretrained models)'),
        ('tensorboard', 'TensorBoard'),
        ('wandb', 'Weights & Biases'),
    ]
    for module, name in optional_deps:
        check_import(module, name)  # Don't fail on optional
        
    # 4. Check GPU
    print("\n4. GPU Status")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠ CUDA not available (will use CPU)")
    except:
        print("  ✗ Could not check GPU")
        
    # 5. Check project modules
    print("\n5. Project Modules")
    sys.path.insert(0, str(Path(__file__).parent))
    
    project_modules = [
        ('src.features.extractor', 'Feature Extractor'),
        ('src.features.augmentation', 'Augmentation'),
        ('src.detection.models', 'Detection Models'),
        ('src.detection.trainer', 'Trainer'),
        ('src.data.dataset', 'Dataset Loaders'),
    ]
    for module, name in project_modules:
        if not check_import(module, name):
            all_ok = False
            
    # 6. Check data directories
    print("\n6. Data Directories")
    data_dirs = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models',
        'experiments',
    ]
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ⚠ {dir_path}/ (will be created)")
            path.mkdir(parents=True, exist_ok=True)
            
    # 7. Check for datasets
    print("\n7. Available Datasets")
    dataset_paths = {
        'DADS': 'data/external/dads',
        'Al-Emadi': 'data/external/alemadi',
        'DroneThesis': 'data/external/drone_detection_thesis',
    }
    datasets_found = 0
    for name, path in dataset_paths.items():
        if Path(path).exists() and any(Path(path).iterdir()):
            print(f"  ✓ {name}")
            datasets_found += 1
        else:
            print(f"  ✗ {name} (not downloaded)")
            
    if datasets_found == 0:
        print("\n  → Run: python scripts/download_datasets.py --dataset all")
        
    # 8. Quick model test
    print("\n8. Quick Model Test")
    try:
        from src.detection.models import create_model
        import torch
        
        model = create_model('custom_cnn', num_classes=2)
        x = torch.randn(1, 1, 128, 128)
        y = model(x)
        
        if y.shape == (1, 2):
            print(f"  ✓ Model forward pass OK")
        else:
            print(f"  ✗ Unexpected output shape: {y.shape}")
            all_ok = False
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        all_ok = False
        
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("  1. Download datasets:")
        print("     python scripts/download_datasets.py --dataset alemadi")
        print("")
        print("  2. Start training:")
        print("     python train.py --datasets alemadi --epochs 50")
        print("")
        print("  3. Or explore with notebook:")
        print("     jupyter notebook notebooks/01_getting_started.ipynb")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return all_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
