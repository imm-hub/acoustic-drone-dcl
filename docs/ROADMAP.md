# Research Roadmap

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Environment Setup
- [ ] Set up Python environment
- [ ] Install required packages
- [ ] Configure GPU support (if available)
- [ ] Set up experiment tracking (TensorBoard/Wandb)

### 1.2 Data Collection
- [ ] Download DADS dataset from Hugging Face
- [ ] Download Sara Al-Emadi dataset from GitHub
- [ ] Download DroneAudioset if accessible
- [ ] Create data inventory spreadsheet

### 1.3 Data Exploration
- [ ] Analyze audio properties (duration, sample rate, etc.)
- [ ] Visualize sample spectrograms
- [ ] Listen to samples to understand acoustic characteristics
- [ ] Document dataset statistics

---

## Phase 2: Detection Module (Weeks 3-5)

### 2.1 Preprocessing Pipeline
- [ ] Implement audio loading and resampling
- [ ] Implement segmentation (windowing)
- [ ] Implement normalization
- [ ] Create data augmentation functions:
  - Time stretching
  - Pitch shifting
  - Background noise addition
  - Time masking / Frequency masking (SpecAugment)

### 2.2 Feature Extraction
- [ ] Implement Mel Spectrogram extraction
- [ ] Implement MFCC extraction
- [ ] Implement STFT
- [ ] Implement spectral contrast (optional)
- [ ] Compare feature effectiveness

### 2.3 Model Development
- [ ] Implement baseline CNN model
- [ ] Implement RNN/LSTM model
- [ ] Implement CRNN model
- [ ] Train and evaluate models
- [ ] Hyperparameter tuning

### 2.4 Evaluation
- [ ] Define metrics (Accuracy, Precision, Recall, F1)
- [ ] Cross-validation
- [ ] Test on unseen data
- [ ] Document results

---

## Phase 3: Classification Module (Weeks 6-8)

### 3.1 Multi-class Setup
- [ ] Organize data by drone type/model
- [ ] Create label mapping
- [ ] Handle class imbalance

### 3.2 Type Classification
- [ ] Quadcopter vs Hexacopter vs Fixed-wing
- [ ] Train classifier
- [ ] Evaluate performance

### 3.3 Model Classification (if feasible)
- [ ] DJI Phantom vs Mavic vs other models
- [ ] Fine-tune models
- [ ] Document limitations

### 3.4 Extended Features (Research)
- [ ] Investigate rotor count estimation from BPF
- [ ] Study blade count signatures
- [ ] Explore distance estimation

---

## Phase 4: Localization Module (Weeks 9-11)

### 4.1 Theory Study
- [ ] Study TDOA theory
- [ ] Study DOA estimation methods
- [ ] Study GCC-PHAT algorithm
- [ ] Review beamforming techniques

### 4.2 Simulation
- [ ] Create simulated multi-microphone scenarios
- [ ] Implement GCC-PHAT
- [ ] Implement DOA estimation
- [ ] Test with synthetic data

### 4.3 Implementation (if multi-channel data available)
- [ ] Process multi-channel recordings
- [ ] Estimate direction of arrival
- [ ] Evaluate localization accuracy

**Note**: Localization requires microphone array data. If single-channel data only available, this phase will be theoretical/simulated.

---

## Phase 5: Integration & Refinement (Weeks 12-14)

### 5.1 System Integration
- [ ] Create unified pipeline
- [ ] Detection → Classification → Localization
- [ ] End-to-end testing

### 5.2 Performance Optimization
- [ ] Model compression (if needed)
- [ ] Inference speed optimization
- [ ] Memory optimization

### 5.3 Real-world Testing
- [ ] Test with diverse noise conditions
- [ ] Document failure cases
- [ ] Suggest improvements

---

## Phase 6: Documentation & Thesis Writing (Weeks 15-18)

### 6.1 Code Documentation
- [ ] Complete docstrings
- [ ] Create API documentation
- [ ] Write usage examples

### 6.2 Thesis Writing
- [ ] Introduction & Background
- [ ] Literature Review
- [ ] Methodology
- [ ] Implementation
- [ ] Results & Discussion
- [ ] Conclusion & Future Work

### 6.3 Final Deliverables
- [ ] Clean repository
- [ ] Trained model weights
- [ ] Thesis document
- [ ] Presentation

---

## Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| M1 | Week 2 | Data collected and explored |
| M2 | Week 5 | Detection model working |
| M3 | Week 8 | Classification model working |
| M4 | Week 11 | Localization implemented |
| M5 | Week 14 | Integrated system |
| M6 | Week 18 | Thesis complete |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Limited dataset diversity | Medium | High | Combine multiple datasets, use augmentation |
| No multi-channel data for localization | High | Medium | Focus on simulation, theoretical analysis |
| Classification accuracy low | Medium | Medium | Focus on coarse categories (quad/hex/fixed) |
| Computational resources limited | Medium | Medium | Use efficient architectures, cloud computing |

---

## Questions to Clarify

1. Is there access to GPU resources for training?
2. Is there possibility to record own drone audio data?
3. What is the primary application context (security, research, etc.)?
4. Is real-time performance a requirement?
5. What programming language preference (Python recommended)?
