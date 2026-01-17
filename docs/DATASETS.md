# Available Acoustic Drone Datasets

This document contains verified acoustic drone datasets that can be used for this research.

## 1. Drone Audio Detection Samples (DADS) - Hugging Face

**Source**: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples

**Description**: Currently the largest publicly available drone audio database, specifically designed for developing drone detection systems using deep learning techniques.

**Specifications**:
- Sample rate: 16,000 Hz
- Bit depth: 16-bit
- Channels: Mono
- Duration: 500ms to several minutes per file
- Structure: Binary classification (0 = no drone, 1 = drone)

**Notes**: Most drone audio files were manually trimmed. Some datasets contain drone audio captured from distant microphones.

---

## 2. Sara Al-Emadi Drone Audio Dataset

**Source**: https://github.com/saraalemadi/DroneAudioDataset

**Paper**: Al-Emadi, S.A., Al-Ali, A.K., Al-Ali, A., & Mohamed, A. (2019). "Audio Based Drone Detection and Identification using Deep Learning". IWCMC 2019.

**Description**: Drone propeller noise recorded in an indoor environment, artificially augmented with random noise clips.

**Contents**:
- Binary classification folder
- Multiclass classification folder
- Noise clips from ESC-50 dataset
- White noise from Speech Commands dataset

**Citation**:
```bibtex
@INPROCEEDINGS{AlEm1906:Audio,
  AUTHOR="Sara A Al-Emadi and Abdulla K Al-Ali and Abdulaziz Al-Ali and Amr Mohamed",
  TITLE="Audio Based Drone Detection and Identification using Deep Learning",
  BOOKTITLE="IWCMC 2019",
  ADDRESS="Tangier, Morocco",
  MONTH=jun,
  YEAR=2019
}
```

---

## 3. DroneDetectionThesis Multi-Sensor Dataset

**Source**: https://github.com/DroneDetectionThesis/Drone-detection-dataset

**Description**: Multi-sensor dataset containing IR, visible video, and audio data for drone detection.

**Audio Contents**:
- 90 audio clips
- Classes: Drones, helicopters, background noise

**Additional Data**:
- 650 videos (365 IR, 285 visible)
- 203,328 annotated frames total

**Distance Categories**: Close, Medium, Distant (max 200m for drones)

---

## 4. DroneAudioset - Search and Rescue Dataset

**Source**: https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/

**Paper**: arXiv:2510.15383 (October 2025)

**Description**: Comprehensive drone audition dataset for search and rescue applications, focusing on human presence detection under drone ego-noise.

**Specifications**:
- Total duration: 23.5 hours of annotated recordings
- SNR range: -57.2 dB to -2.5 dB
- Coverage: Various drone types, throttle levels, microphone configurations, and environments

**License**: MIT

---

## 5. UAV Multiclass Acoustic Dataset (32 Classes)

**Source**: https://arxiv.org/html/2509.04715v1

**Description**: Novel dataset with audio recordings, spectrograms, and MFCC plots for 32 different drone categories.

**Recording Equipment**:
- 2021-2023: MacBook Air (Intel Core i5)
- 2024+: MacBook Air (Apple M3)
- Internal microphone, no external processing

**Feature Extraction Settings** (using Librosa):
- n_mfcc: 20
- n_fft: 2048
- hop_length: 512
- n_mels: 128

---

## 6. Zenodo - Drone Fault Classification Dataset

**Source**: https://doi.org/10.5281/zenodo.7779574

**Paper**: Yi, W., Choi, J.-W., & Lee, J.-W. (2023). "Sound-Based Drone Fault Classification Using Multi-Task Learning"

**Description**: Dataset for drone fault classification using acoustic signals.

---

## 7. European Acoustics Association Database

**Source**: Forum Acusticum 2023

**Paper**: Kümmritz, S., & Paul, L. (2023). "Comprehensive Database of Drone Sounds for Machine Learning"

**Description**: Extensive open-access database of drone sounds including:
- Existing drone recordings
- Own recorded drone audio
- Covers EU drone classes C0 (<250g) to C4 (<25kg)

---

## Supplementary Datasets (for noise augmentation)

### ESC-50 (Environmental Sound Classification)
**Source**: https://github.com/karolpiczak/ESC-50
**Use**: Environmental noise for data augmentation

### UrbanSound8K
**Source**: https://urbansounddataset.weebly.com
**Use**: Urban background noise augmentation

### TUT Acoustic Scenes 2017
**Source**: https://doi.org/10.5281/zenodo.1040168
**Use**: Acoustic scene classification, background noise

---

## Recommended Download Priority

1. **DADS (Hugging Face)** - Largest, well-structured for binary detection
2. **Sara Al-Emadi Dataset** - Well-documented with paper, good for initial experiments
3. **DroneAudioset** - High quality, diverse conditions
4. **UAV 32-class Dataset** - Best for classification tasks
5. **Multi-sensor Dataset** - Useful if expanding to multimodal approaches

---

## Data Combination Strategy

For creating a larger combined dataset:

1. Standardize all audio to:
   - Sample rate: 16,000 Hz (or 44,100 Hz if higher quality needed)
   - Bit depth: 16-bit
   - Channels: Mono
   - Format: WAV

2. Create unified labeling scheme:
   - Level 1: Binary (drone/no-drone)
   - Level 2: Drone type (quadcopter, hexacopter, fixed-wing)
   - Level 3: Specific model (DJI Phantom 4, Mavic, etc.)

3. Document provenance of each sample

4. Apply consistent preprocessing pipeline
