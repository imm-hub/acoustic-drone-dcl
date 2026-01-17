#!/usr/bin/env python3
"""
Dataset Download Script for Acoustic Drone Detection Research

This script helps download publicly available drone audio datasets.
Run from the repository root directory.

Usage:
    python scripts/download_datasets.py --dataset [dads|alemadi|all]
"""

import os
import argparse
import subprocess
from pathlib import Path


def get_data_dir():
    """Get the data/external directory path."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "external"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_dads_dataset(data_dir):
    """
    Download DADS (Drone Audio Detection Samples) from Hugging Face.
    
    Source: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
    """
    print("=" * 60)
    print("Downloading DADS Dataset from Hugging Face...")
    print("=" * 60)
    
    dads_dir = data_dir / "dads"
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset via Hugging Face datasets library...")
        dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
        
        # Save dataset info
        print(f"Dataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Save to disk
        dads_dir.mkdir(exist_ok=True)
        dataset.save_to_disk(str(dads_dir))
        print(f"Dataset saved to: {dads_dir}")
        
    except ImportError:
        print("Hugging Face 'datasets' library not installed.")
        print("Install with: pip install datasets")
        print("\nAlternatively, visit:")
        print("https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples")
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True


def download_alemadi_dataset(data_dir):
    """
    Download Sara Al-Emadi Drone Audio Dataset from GitHub.
    
    Source: https://github.com/saraalemadi/DroneAudioDataset
    """
    print("=" * 60)
    print("Downloading Al-Emadi Drone Audio Dataset from GitHub...")
    print("=" * 60)
    
    alemadi_dir = data_dir / "alemadi"
    
    if alemadi_dir.exists():
        print(f"Directory already exists: {alemadi_dir}")
        response = input("Re-download? (y/n): ").lower()
        if response != 'y':
            return True
    
    try:
        # Clone the repository
        cmd = [
            "git", "clone", 
            "https://github.com/saraalemadi/DroneAudioDataset.git",
            str(alemadi_dir)
        ]
        subprocess.run(cmd, check=True)
        print(f"Dataset cloned to: {alemadi_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False
    except FileNotFoundError:
        print("Git not found. Please install git or download manually from:")
        print("https://github.com/saraalemadi/DroneAudioDataset")
        return False


def download_drone_detection_thesis(data_dir):
    """
    Download DroneDetectionThesis multi-sensor dataset from GitHub.
    
    Source: https://github.com/DroneDetectionThesis/Drone-detection-dataset
    Note: This includes audio, IR, and visible data.
    """
    print("=" * 60)
    print("Downloading DroneDetectionThesis Dataset from GitHub...")
    print("=" * 60)
    
    thesis_dir = data_dir / "drone_detection_thesis"
    
    if thesis_dir.exists():
        print(f"Directory already exists: {thesis_dir}")
        response = input("Re-download? (y/n): ").lower()
        if response != 'y':
            return True
    
    try:
        cmd = [
            "git", "clone",
            "https://github.com/DroneDetectionThesis/Drone-detection-dataset.git",
            str(thesis_dir)
        ]
        subprocess.run(cmd, check=True)
        print(f"Dataset cloned to: {thesis_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False


def print_manual_download_instructions():
    """Print instructions for datasets that require manual download."""
    print("\n" + "=" * 60)
    print("DATASETS REQUIRING MANUAL DOWNLOAD")
    print("=" * 60)
    
    print("""
1. DroneAudioset (Search and Rescue)
   URL: https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/
   License: MIT
   
2. Zenodo - Drone Fault Classification
   URL: https://doi.org/10.5281/zenodo.7779574
   
3. ESC-50 (Environmental Sound Classification) - for augmentation
   URL: https://github.com/karolpiczak/ESC-50
   
4. UrbanSound8K - for augmentation
   URL: https://urbansounddataset.weebly.com
   Note: Requires registration
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download drone audio datasets for research"
    )
    parser.add_argument(
        "--dataset",
        choices=["dads", "alemadi", "thesis", "all", "info"],
        default="info",
        help="Which dataset to download (default: info)"
    )
    
    args = parser.parse_args()
    data_dir = get_data_dir()
    
    print(f"Data directory: {data_dir}\n")
    
    if args.dataset == "info":
        print("Available datasets to download:")
        print("  --dataset dads     : DADS from Hugging Face")
        print("  --dataset alemadi  : Al-Emadi dataset from GitHub")
        print("  --dataset thesis   : DroneDetectionThesis from GitHub")
        print("  --dataset all      : Download all above")
        print_manual_download_instructions()
        return
    
    results = {}
    
    if args.dataset in ["dads", "all"]:
        results["DADS"] = download_dads_dataset(data_dir)
    
    if args.dataset in ["alemadi", "all"]:
        results["Al-Emadi"] = download_alemadi_dataset(data_dir)
    
    if args.dataset in ["thesis", "all"]:
        results["Thesis"] = download_drone_detection_thesis(data_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")
    
    print_manual_download_instructions()


if __name__ == "__main__":
    main()
