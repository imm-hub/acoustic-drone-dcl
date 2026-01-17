# Complete Guide to Acoustic Drone Datasets

## Overview

This document provides a comprehensive overview of ALL publicly available acoustic drone datasets, what can be done with each, and how to access them.

---

## Quick Summary Table

| Dataset | Size | Access | Detection | Classification | Localization | Multi-channel |
|---------|------|--------|-----------|----------------|--------------|---------------|
| **DADS** | ~500MB | ✅ Free (HF) | ✅ | ❌ | ❌ | ❌ |
| **Al-Emadi** | ~50MB | ✅ Free (GitHub) | ✅ | ✅ (4 classes) | ❌ | ❌ |
| **DroneThesis** | ~100MB | ✅ Free (GitHub) | ✅ | ✅ (3 classes) | ❌ | ❌ |
| **DroneAudioset** | ~2GB | ✅ Free (HF) | ✅ | ❌ | ❌ | ✅ (17 mics) |
| **32-Class UAV** | ~200MB | ✅ Free | ✅ | ✅ (32 classes) | ❌ | ❌ |
| **DREGON** | ~1GB | ✅ Free | ✅ | ❌ | ✅ | ✅ |
| **UaVirBASE** | ~500MB | ✅ Free | ✅ | ❌ | ✅ | ✅ (8 ch) |
| **DroneNoise DB** | ~300MB | ✅ Free | ✅ | ✅ | ❌ | ❌ |
| **Zenodo Fault** | ~100MB | ✅ Free | ✅ | ✅ (fault types) | ❌ | ❌ |
| **Kaggle Malicious** | ~50MB | ✅ Free | ✅ | ❌ | ❌ | ❌ |

---

## Detailed Dataset Descriptions

### 1. DADS (Drone Audio Detection Samples) ⭐ RECOMMENDED FOR STARTING

**Source:** https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples

**Description:** Currently the largest publicly available drone audio database, aggregated from multiple sources.

**Specifications:**
- Sample rate: 16,000 Hz
- Format: 16-bit mono WAV
- Duration: 500ms to several minutes per file
- Labels: Binary (drone / no-drone)

**What you can do:**
- ✅ Binary drone detection
- ✅ Train robust detection models
- ✅ Benchmark different architectures

**Download:**
```python
from datasets import load_dataset
dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
```

---

### 2. Al-Emadi Drone Audio Dataset ⭐ RECOMMENDED FOR CLASSIFICATION

**Source:** https://github.com/saraalemadi/DroneAudioDataset

**Paper:** Al-Emadi et al. (2019) "Audio Based Drone Detection and Identification using Deep Learning"

**Description:** Indoor recordings of drone propeller noise with noise augmentation.

**Contents:**
- **Binary folder:** Drone vs Unknown (noise)
- **Multiclass folder:** 
  - Bebop drone
  - AR drone  
  - Phantom drone
  - Unknown (noise)

**What you can do:**
- ✅ Binary drone detection
- ✅ Drone model classification (3 drone types)
- ✅ GAN-based data augmentation experiments

**Download:**
```bash
git clone https://github.com/saraalemadi/DroneAudioDataset
```

---

### 3. DroneDetectionThesis Multi-Sensor Dataset

**Source:** https://github.com/DroneDetectionThesis/Drone-detection-dataset

**Paper:** Svanström et al. (2021) "A Dataset for Multi-Sensor Drone Detection"

**Description:** Multi-modal dataset with audio, IR video, and visible video.

**Audio Contents:**
- 90 audio clips
- Classes: Drone, Helicopter, Background noise
- Distance categories: Close, Medium, Distant (up to 200m)

**What you can do:**
- ✅ Drone vs helicopter vs background classification
- ✅ Distance-based analysis
- ✅ Multi-modal fusion (if using video data too)

**Download:**
```bash
git clone https://github.com/DroneDetectionThesis/Drone-detection-dataset
```

---

### 4. DroneAudioset (NeurIPS 2025) ⭐ BEST FOR NOISE SUPPRESSION

**Source:** https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/

**Paper:** Gupta et al. (2025) "DroneAudioset: An Audio Dataset for Drone-based Search and Rescue"

**Description:** Comprehensive dataset for drone-based search and rescue with extreme noise conditions.

**Specifications:**
- **Duration:** 23.5 hours of annotated recordings
- **SNR Range:** -57.2 dB to -2.5 dB (very challenging!)
- **Drone Types:** DJI F450 (large), DJI F330 (small)
- **Microphones:** 17 channels (two 8-channel circular arrays + 1 central)
- **Environments:** Conference room, multi-purpose halls

**Recording Categories:**
1. Drone Noise + Sound Source (~15 hours)
2. Drone Noise Only (~7 hours)
3. Sound Source Only (~1.5 hours)

**What you can do:**
- ✅ Ego-noise suppression research
- ✅ Speech enhancement under drone noise
- ✅ Human presence detection
- ✅ Microphone array processing
- ✅ Hardware-software co-design research

**Download:**
```python
from datasets import load_dataset
dataset = load_dataset("ahlab-drone-project/DroneAudioSet")
```

---

### 5. 32-Class UAV Acoustic Dataset ⭐ BEST FOR MODEL CLASSIFICATION

**Source:** https://arxiv.org/html/2509.04715v1

**Web Tool:** https://mackenzie-jane.github.io/drone-visualization/

**Description:** Dataset with 32 different drone categories differentiated by brand and model.

**Features Provided:**
- Raw audio recordings
- Pre-computed spectrograms
- Pre-computed MFCC plots
- Interactive web visualization

**Recording Equipment:**
- MacBook Air internal microphone
- No external processing (raw characteristics preserved)

**What you can do:**
- ✅ Fine-grained drone model classification
- ✅ Brand identification
- ✅ Acoustic signature analysis
- ✅ Feature comparison studies

---

### 6. DREGON Dataset ⭐ BEST FOR LOCALIZATION

**Source:** IEEE (https://ieeexplore.ieee.org/document/8593581)

**Paper:** "DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization"

**Description:** Dataset for sound source localization using drone-mounted microphone array.

**Contents:**
- Clean and noisy in-flight recordings
- Continuous 3D position annotations (motion capture)
- Rotor speed data
- Inertial measurements

**What you can do:**
- ✅ Sound source localization (3D)
- ✅ Beamforming experiments
- ✅ DOA estimation
- ✅ Noise-robust localization

---

### 7. UaVirBASE Dataset (2025) ⭐ BEST FOR LOCALIZATION RESEARCH

**Source:** https://www.mdpi.com/2076-3417/15/10/5378

**Description:** Dedicated dataset for UAV sound source localization.

**Specifications:**
- **Channels:** 8-channel microphone array
- **Sample Rate:** 96 kHz
- **Bit Depth:** 32-bit
- **Annotations:** Azimuth, distance, height, orientation

**Variations Included:**
- Different distances
- Different altitudes
- Different azimuths
- Different orientations (front, back, left, right)

**What you can do:**
- ✅ Sound source localization
- ✅ Direction of Arrival (DOA) estimation
- ✅ Distance estimation
- ✅ Orientation classification

---

### 8. DroneNoise Database (University of Salford)

**Source:** https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411

**Description:** Drone noise recordings for environmental noise research.

**What you can do:**
- ✅ Drone acoustic characterization
- ✅ Environmental noise studies
- ✅ Psychoacoustic analysis

---

### 9. Zenodo - Drone Fault Classification

**Source:** https://doi.org/10.5281/zenodo.7779574

**Paper:** Yi et al. (2023) "Sound-Based Drone Fault Classification Using Multi-Task Learning"

**Description:** Dataset for classifying drone faults based on acoustic signatures.

**What you can do:**
- ✅ Fault detection
- ✅ Anomaly detection
- ✅ Predictive maintenance research

---

### 10. AUDROK Drone Sound Data

**Source:** https://mobilithek.info/offers/605778370199691264

**Description:** German dataset for drone acoustic research.

---

### 11. Kaggle - Malicious UAVs Detection

**Source:** https://www.kaggle.com/datasets/sonain/malicious-uavs-detection

**Description:** Dataset focused on detecting potentially malicious drones.

---

### 12. AcousticPrint Dataset

**Source:** https://github.com/AcousticPrint/AcousticPrint

**Description:** Acoustic fingerprinting for drone identification.

**What you can do:**
- ✅ Drone fingerprinting
- ✅ Individual drone identification

---

## Supplementary Datasets (For Augmentation)

### ESC-50 (Environmental Sound Classification)
**Source:** https://github.com/karolpiczak/ESC-50
**Use:** Background noise augmentation

### UrbanSound8K
**Source:** https://urbansounddataset.weebly.com
**Use:** Urban noise augmentation
**Note:** Requires registration

### TUT Acoustic Scenes 2017
**Source:** https://doi.org/10.5281/zenodo.1040168
**Use:** Acoustic scene classification, background noise

---

## What Can Be Done With These Datasets?

### Task 1: Detection (Drone vs No-Drone)
**Best Datasets:** DADS, Al-Emadi, DroneThesis
**Typical Accuracy:** 94-98%
**Methods:** CNN on Mel spectrograms, MFCC + SVM

### Task 2: Drone Type Classification
**Best Datasets:** 32-Class UAV, Al-Emadi
**Classes:** Quadcopter, Hexacopter, Fixed-wing, or specific models
**Methods:** CNN, Transfer learning (EfficientNet)

### Task 3: Model/Brand Identification
**Best Datasets:** 32-Class UAV, Al-Emadi
**Challenge:** Fine-grained classification
**Methods:** Deep CNN, Attention mechanisms

### Task 4: Localization
**Best Datasets:** UaVirBASE, DREGON, DroneAudioset
**Requirements:** Multi-channel recordings
**Methods:** GCC-PHAT, TDOA, Beamforming

### Task 5: Distance Estimation
**Best Datasets:** DroneThesis, UaVirBASE
**Challenge:** Signal attenuation modeling
**Methods:** Regression on spectral features

### Task 6: Noise Suppression
**Best Datasets:** DroneAudioset
**Challenge:** Extreme SNR conditions (-57 dB)
**Methods:** Beamforming, Neural enhancement (MPSENet)

### Task 7: Rotor/Blade Counting
**Best Datasets:** None directly (research gap!)
**Approach:** Analyze Blade Passing Frequency harmonics
**Potential:** Your thesis contribution!

### Task 8: Fault Detection
**Best Datasets:** Zenodo Fault Classification
**Use Case:** Predictive maintenance

---

## Recommended Dataset Combinations for Your Thesis

### For Detection + Classification:
```
DADS + Al-Emadi + 32-Class UAV
↓
Combined: ~1000+ samples
Tasks: Detection, Type classification, Model identification
```

### For Full Pipeline (Detection + Classification + Localization):
```
Al-Emadi (detection/classification) + UaVirBASE (localization)
↓
Requires: Separate models for each task
Challenge: Different recording conditions
```

### For Research Novelty:
```
DroneAudioset (noise suppression) + Your own recordings
↓
Novel contribution: Robust detection under extreme noise
```

---

## Download Commands Summary

```bash
# 1. Al-Emadi (GitHub) - Start here
git clone https://github.com/saraalemadi/DroneAudioDataset data/external/alemadi

# 2. DroneThesis (GitHub)
git clone https://github.com/DroneDetectionThesis/Drone-detection-dataset data/external/drone_detection_thesis

# 3. DADS (Hugging Face)
python -c "from datasets import load_dataset; d=load_dataset('geronimobasso/drone-audio-detection-samples'); d.save_to_disk('data/external/dads')"

# 4. DroneAudioset (Hugging Face)
python -c "from datasets import load_dataset; d=load_dataset('ahlab-drone-project/DroneAudioSet'); d.save_to_disk('data/external/droneaudioset')"

# 5. ESC-50 for augmentation
git clone https://github.com/karolpiczak/ESC-50 data/external/esc50
```

---

## Research Gaps (Opportunities for Your Thesis!)

1. **Rotor count estimation from audio** - No dataset explicitly labels this
2. **Blade count detection** - Could be derived from BPF analysis
3. **Real outdoor recordings with ground truth** - Most datasets are indoor
4. **Multi-drone scenarios** - Very limited data available
5. **Turkish/regional drone models** - Not covered in existing datasets
6. **Combined detection + localization + classification pipeline** - Few integrated solutions

---

## Citation Requirements

When using these datasets, remember to cite the original papers. See `docs/DATASETS.md` in your repository for full citations.
