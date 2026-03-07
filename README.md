# Acoustic Drone Detection, Localization, and Classification

## Master's Thesis Project

This repository contains the research code and documentation for acoustic-based drone detection, localization, and classification.

## Research Objectives

1. **Detection**: Detect the presence of drones using acoustic signals
2. **Localization**: Determine the position/direction of detected drones
3. **Extended Features** (if feasible):
   - Number of drones
   - Number of rotors/wings
   - Drone type/model identification

## Repository Structure

```
acoustic-drone-detection/
├── data/
│   ├── raw/              # Original unprocessed audio files
│   ├── processed/        # Preprocessed data (spectrograms, features)
│   └── external/         # Downloaded datasets
├── src/
│   ├── detection/        # Drone presence detection modules
│   ├── localization/     # Sound source localization algorithms
│   ├── classification/   # Drone type/model classification
│   ├── features/         # Feature extraction (MFCC, Mel-spectrogram, etc.)
│   └── utils/            # Helper functions and utilities
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── docs/                 # Documentation and literature review
├── experiments/          # Experiment logs and results
├── models/               # Trained model weights
├── configs/              # Configuration files
└── tests/                # Unit tests
```

## Available Datasets

See `docs/DATASETS.md` for a comprehensive list of available acoustic drone datasets.

## Literature Review

See `docs/LITERATURE_REVIEW.md` for summarized research papers and methods.

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation
```bash
git clone <repository-url>
cd acoustic-drone-detection
pip install -r requirements.txt
```

## Methodology

### Feature Extraction
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Mel Spectrograms
- Short-Time Fourier Transform (STFT)
- Spectral Contrast
- Chroma Features

### Detection Approaches
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Convolutional Recurrent Neural Networks (CRNN)

### Localization Techniques
- Time Difference of Arrival (TDOA)
- Direction of Arrival (DOA)
- GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- Beamforming

## Author

İhsan Mert Muhacıroğlu
muhaciroglu19@itu.edu.tr
Please Give mention if you are using this code, or implementations.

