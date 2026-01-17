# Literature Review: Acoustic Drone Detection, Localization, and Classification

## Overview

This document summarizes key research papers and methodologies for acoustic-based drone detection, localization, and classification.

---

## 1. DETECTION

### 1.1 Deep Learning Approaches

#### Al-Emadi et al. (2021) - GAN-Enhanced Detection
**Paper**: "Audio-Based Drone Detection and Identification Using Deep Learning Techniques with Dataset Enhancement through Generative Adversarial Networks"
**Source**: MDPI Sensors (https://www.mdpi.com/1424-8220/21/15/4953)

**Key Contributions**:
- Compared CNN, RNN, and CRNN for drone detection
- Used GAN to generate artificial drone audio to enhance dataset
- Addresses the challenge of limited acoustic drone datasets

**Methods**:
- Feature extraction: Raw audio waveforms
- Models: CNN, RNN, CRNN with similar complexity (~70k-78k trainable parameters)
- Data augmentation: GAN-generated drone audio

**Results**:
- Deep learning techniques effective for drone detection
- GAN-generated data improved detection of unfamiliar drone types

---

#### Casabianca & Zhang (2021) - Ensemble Learning
**Focus**: Ensemble deep learning models for acoustic UAV identification

**Results**:
- 94.7% accuracy on test datasets
- 91.0% on augmented and real test datasets
- CNN and CRNN outperformed RNN

**Feature Extraction**:
- Mel spectrograms as primary input
- CNNs shown best suited for spectrogram image processing

---

#### Jeon et al. (2017) - Distance-Based Detection
**Paper**: "Empirical Study of Drone Sound Detection in Real-Life Environment with Deep Neural Networks"

**Key Findings**:
- UAV detection feasible within 150-meter range
- Acoustic signal fidelity degrades significantly beyond 150m
- Compared GMM, CNN, and RNN approaches
- MFCC and Mel spectrogram features evaluated

**Data Augmentation**: Environmental sounds added to drone recordings

---

#### Seo et al. - High Accuracy Detection
**Method**: STFT features with CNNs
**Dataset**: Hovering DJI Phantom 3 and Phantom 4 recordings
**Results**:
- Detection rate: >98%
- False alarm rate: 1.28%
- Normalized time-frequency features effective for clean recordings

---

#### Kim et al. (2021) - Self-Supervised Learning
**Method**: SimCLR contrastive learning on MFCC-based image representations
**Results**:
- Top-1 accuracy: 87.91%
- No labeled data required
- Generalized well to unseen drone types

---

### 1.2 Feature Extraction Techniques

#### Wang et al. - Feature Comparison Study
**Source**: Librosa Python library evaluation
**Drones Tested**: DJI Phantom 4, EVO 2 Pro

**Features Compared**:
1. MFCCs (Mel-Frequency Cepstral Coefficients)
2. Chroma features
3. Mel spectrograms
4. Spectral contrast
5. Tonnetz

**Finding**: Combining multiple acoustic features significantly enhanced discriminative capacity

---

### 1.3 Feature Extraction Best Practices

| Feature | Typical Parameters | Use Case |
|---------|-------------------|----------|
| MFCC | n_mfcc=20, n_fft=2048, hop_length=512 | Classification |
| Mel Spectrogram | n_mels=128, n_fft=2048 | CNN input |
| STFT | n_fft=2048 | Time-frequency analysis |
| Spectral Contrast | 6 frequency bands | Audio discrimination |

---

## 2. LOCALIZATION

### 2.1 Microphone Array Techniques

#### GCC-PHAT Method
**Paper**: "Performance Enhancement of Drone Acoustic Source Localization Through Distributed Microphone Arrays" (MDPI Sensors, 2025)

**Key Concepts**:
- Generalized Cross-Correlation with Phase Transform
- Uses TDOA (Time Difference of Arrival) measurements
- Distributed microphone arrays improve accuracy

**Results**:
- Up to 2.13m improvement in localization at SNR > 0 dB
- Reduced mean and variance of localization errors

**Challenge Addressed**: 
- AOA vs azimuth discrepancy significant for high-altitude drones
- Single microphone array limited for elevation estimation

---

#### Pourmohammad et al. - 4-Microphone Array
**System**: Real-time 3D localization
**Limitation**: Difficulty capturing elevation information for airborne targets

---

#### Lee & Park (2021) - Drone-Mounted Array
**Paper**: "An Acoustic Source Localization Method Using a Drone-Mounted Phased Microphone Array" (MDPI Drones)

**System**:
- 32-channel time-synchronized MEMS microphone array
- Mounted on drone for detecting ground sources

**Methods**:
- Spectral subtraction for noise reduction
- Beamforming for DOA estimation
- Data fusion with drone flight navigation

**Results**:
- ~10 degrees error in azimuth and elevation
- Effective at 150m ground distance

---

### 2.2 DOA Estimation Techniques

#### IEEE/ACM TASLP (2022) - Low SdNR Conditions
**Paper**: "Drone Audition: Sound Source Localization Using On-Board Microphones"

**Challenge**: Signal-to-drone noise ratio (SdNR) extremely low due to motor/propeller noise

**Method**:
- Cross-correlation based DOA estimation
- TDOA at different microphone pairs
- Noise angular spectrum subtraction
- Current-specific drone noise spectrum measurement

**Results**:
- Effective at SdNR as low as -30 dB
- Can localize multiple simultaneous sound sources

---

### 2.3 Classification + Localization Combined

#### HMM with Circular Microphone Array
**Paper**: "Classification, positioning, and tracking of drones by HMM using acoustic circular microphone array beamforming" (EURASIP, 2020)

**System**:
- 32 microphone elements in circular array
- Switched Beam Forming (SBF) for scanning
- Hidden Markov Model (HMM) for classification
- RLS adaptive beamforming for tracking

**Process**:
1. Scan sky with switched beamforming
2. Classify sound source using HMM
3. Track identified drone with adaptive beamforming

---

### 2.4 AIM: Acoustic Inertial Measurement

**Paper**: "AIM: Acoustic Inertial Measurement for Indoor Drone Localization and Tracking" (2023)

**Innovation**:
- Exploits dual acoustic channel from rotating propellers:
  - DOA indicates drone orientation
  - Frequency properties correspond to propeller rotation (motion)

**Features Used**:
- DOA (Direction of Arrival)
- Frequencies
- MFCC (for drone structure identification)

**System**:
- Single 4-microphone array
- Kalman filter for error reduction
- IQR (Interquartile Range rule) for outlier elimination

**Results**:
- Mean localization error 46% lower than commercial UWB systems
- Works in NLoS (Non-Line-of-Sight) conditions

---

## 3. CLASSIFICATION (Type/Model Identification)

### 3.1 Acoustic Signatures

**Key Insight**: Drone motors and rotors emit characteristic sounds that vary across models, creating unique "acoustic fingerprints"

**Frequency Characteristics**:
- Blade Passing Frequency (BPF) = Rotational frequency × Number of blades
- Example: 4100 RPM with 2 blades → BPF = 136.6 Hz
- BPF and its harmonics dominate drone noise
- Most energy concentrated within 1000 Hz

---

### 3.2 Propeller-Based Features

**Blade Count Detection**:
| Blade Count | Characteristics |
|-------------|-----------------|
| 2-blade | More efficient, faster, distinct BPF |
| 3-blade | Balanced thrust/efficiency, common in FPV |
| 4+ blade | More stable, different acoustic signature |

**Rotor Configuration**:
- Quadcopter: 4 rotors (most common)
- Hexacopter: 6 rotors
- Octocopter: 8 rotors
- Each configuration produces distinct frequency patterns

---

### 3.3 Multi-Drone Detection

**Wright State University Thesis**: "Multiple Drone Detection and Acoustic Scene Classification with Deep Learning"

**Objective**: Detect if zero, one, or two drones present in scene

**Method**:
- CNN with spectrograms
- DNN with hand-engineered features
- Stereo microphone setup

**Application**: Augment information from sensors less capable of counting multiple UAVs

---

## 4. RECOMMENDED METHODOLOGY

### 4.1 Preprocessing Pipeline

```
Raw Audio → Resampling (16kHz) → Normalization → Segmentation (1-3s windows)
         → Feature Extraction → Data Augmentation → Model Input
```

### 4.2 Feature Extraction Recommendations

**Primary Features**:
1. Mel Spectrogram (for CNN input)
2. MFCCs (for traditional ML and some DL)
3. STFT (for frequency analysis)

**Librosa Parameters**:
```python
# Mel Spectrogram
librosa.feature.melspectrogram(y, sr=16000, n_mels=128, n_fft=2048, hop_length=512)

# MFCC
librosa.feature.mfcc(y, sr=16000, n_mfcc=20, n_fft=2048, hop_length=512)
```

### 4.3 Model Architecture Suggestions

**For Detection**:
- Start with VGG11/VGG16 (proven effective for spectrogram classification)
- Consider CRNN for temporal patterns

**For Classification**:
- CNN for spectrogram-based type identification
- Ensemble methods for improved robustness

**For Localization**:
- GCC-PHAT with microphone array (if available)
- DOA estimation using TDOA

---

## 5. RESEARCH GAPS & OPPORTUNITIES

1. **Limited datasets** covering all drone types and models
2. **Real-world noise conditions** not well represented
3. **Multi-drone detection** still challenging
4. **Integration of detection + localization + classification** in single system
5. **Number of rotors/blades estimation** from audio not well studied
6. **Distance estimation** from acoustic signals
7. **Real-time implementation** on embedded systems

---

## 6. KEY REFERENCES

1. Al-Emadi, S.A. et al. (2021). MDPI Sensors. DOI: 10.3390/s21154953
2. Lee, Y.J. & Park, J. (2021). MDPI Drones. DOI: 10.3390/drones5030075
3. Kim et al. (2021). Self-supervised learning framework
4. Forum Acusticum 2023. European Acoustics Association drone database
5. EURASIP J. Wireless Comm. (2020). HMM + Circular array
6. IEEE/ACM TASLP (2022). Drone audition localization
