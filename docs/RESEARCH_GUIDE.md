# Master's Thesis Research Guide
## Acoustic Drone Detection, Localization, and Classification
### Istanbul Technical University - Telecommunication Engineering

---

## Your Research Goals (Reminder)

1. **Detection** - Detect presence of drones from acoustic signals
2. **Localization** - Determine drone position/direction
3. **Extended Features** (if feasible):
   - Number of drones
   - Number of rotors/wings
   - Drone type/model identification

---

## Research Process Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PHASE 1          PHASE 2          PHASE 3          PHASE 4          PHASE 5
│   ────────         ────────         ────────         ────────         ────────
│   Foundation  →    Detection   →    Features    →    Writing    →    Defense
│   (2-3 weeks)      (4-5 weeks)      (4-5 weeks)      (3-4 weeks)      (1 week)
│                                                                             │
│   ▪ Literature     ▪ Data prep      ▪ Classif.       ▪ Thesis         ▪ Present
│   ▪ Setup env      ▪ Train model    ▪ Localiz.       ▪ Paper          ▪ Q&A
│   ▪ Get data       ▪ Evaluate       ▪ Analysis       ▪ Review               
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# PHASE 1: FOUNDATION (Weeks 1-3)

## Week 1: Environment & Literature

### Day 1-2: Setup Development Environment

```bash
# 1. Extract repository
unzip acoustic-drone-detection.zip
cd acoustic-drone-detection

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python verify_setup.py
```

### Day 3-4: Read Core Literature

**Must Read (in order):**

1. **Al-Emadi et al. (2021)** - "Audio-Based Drone Detection and Identification Using Deep Learning"
   - URL: https://www.mdpi.com/1424-8220/21/15/4953
   - Key: CNN, RNN, CRNN comparison + GAN augmentation

2. **Jeon et al. (2017)** - "Empirical Study of Drone Sound Detection in Real-Life Environment"
   - Key: Detection range limits (~150m), feature comparison

3. **Casabianca & Zhang (2021)** - Ensemble deep learning for UAV identification
   - Key: 94.7% accuracy with CNN on mel spectrograms

4. **UaVirBASE (2025)** - For localization concepts
   - URL: https://www.mdpi.com/2076-3417/15/10/5378

**Take Notes On:**
- [ ] What features work best? (MFCC, Mel spectrogram, etc.)
- [ ] What architectures are used? (CNN, RNN, CRNN)
- [ ] What accuracy is achievable?
- [ ] What are the limitations?
- [ ] What research gaps exist?

### Day 5-7: Download and Explore Datasets

```bash
# Download primary datasets
python scripts/download_datasets.py --dataset alemadi
python scripts/download_datasets.py --dataset thesis

# For larger dataset (optional, ~500MB)
python scripts/download_datasets.py --dataset dads
```

**Exploration Tasks:**
- [ ] Listen to 10+ drone audio samples
- [ ] Listen to 10+ non-drone audio samples
- [ ] Visualize spectrograms of both
- [ ] Note differences you can hear/see

---

## Week 2: Data Understanding & Baseline

### Day 1-3: Deep Data Exploration

Open and complete the notebook:
```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

**Create New Notebook: `02_data_exploration.ipynb`**

Tasks:
- [ ] Count samples per class
- [ ] Check audio durations distribution
- [ ] Visualize spectrograms for each drone type
- [ ] Analyze frequency content (where is energy concentrated?)
- [ ] Identify potential issues (noise, clipping, etc.)

### Day 4-5: Establish Baseline

```bash
# Train your first model
python train.py \
    --datasets alemadi \
    --model custom_cnn \
    --epochs 30 \
    --batch-size 32 \
    --experiment-name baseline_v1
```

**Record Results:**
- [ ] Training accuracy
- [ ] Validation accuracy
- [ ] Test accuracy
- [ ] Training time
- [ ] Confusion matrix

### Day 6-7: Document Findings

Create `experiments/baseline_v1/NOTES.md`:
```markdown
# Baseline Experiment v1

## Setup
- Dataset: Al-Emadi
- Model: Custom CNN
- Epochs: 30
- Batch size: 32

## Results
- Train acc: XX%
- Val acc: XX%
- Test acc: XX%

## Observations
- ...

## Next Steps
- ...
```

---

## Week 3: Literature Deep Dive & Research Questions

### Refine Your Research Questions

After exploring data and reading papers, answer:

1. **What specific problem will you solve?**
   - Example: "Robust drone detection under urban noise conditions"
   - Example: "Multi-class drone identification from acoustic signatures"

2. **What is your novel contribution?**
   - New architecture?
   - New feature combination?
   - New dataset (your recordings)?
   - Better performance on existing benchmarks?
   - Rotor counting (unexplored area)?

3. **What are your evaluation metrics?**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - ROC curve, AUC
   - Real-time performance (if applicable)

### Write Research Proposal (1-2 pages)

```
Title: [Your Thesis Title]

1. Problem Statement
   - Why is drone detection important?
   - What are current limitations?

2. Research Objectives
   - Primary: Detection with X% accuracy
   - Secondary: Classification of drone types
   - Tertiary: Localization (if time permits)

3. Methodology
   - Datasets to use
   - Features to extract
   - Models to try
   - Evaluation approach

4. Timeline
   - Week-by-week plan

5. Expected Contributions
   - What will be novel?
```

---

# PHASE 2: DETECTION MODULE (Weeks 4-8)

## Week 4-5: Feature Engineering & Model Selection

### Systematic Feature Comparison

Create `notebooks/03_feature_comparison.ipynb`:

```python
# Compare these features:
features_to_test = [
    'mel_spectrogram',
    'mfcc',
    'mfcc_delta',      # MFCC + first derivative
    'mfcc_delta_delta', # MFCC + first + second derivative
    'stft',
    'combined'          # Multiple features concatenated
]

# For each feature, train same model and compare
```

### Model Architecture Comparison

```bash
# Test different architectures
for model in custom_cnn crnn attention_crnn efficientnet_b0 resnet18; do
    python train.py \
        --datasets alemadi thesis \
        --model $model \
        --epochs 50 \
        --experiment-name detection_${model}
done
```

**Create Comparison Table:**

| Model | Params | Train Acc | Val Acc | Test Acc | F1 | Time |
|-------|--------|-----------|---------|----------|-----|------|
| Custom CNN | | | | | | |
| CRNN | | | | | | |
| EfficientNet-B0 | | | | | | |
| ... | | | | | | |

## Week 6-7: Optimization & Robustness

### Hyperparameter Tuning

Key parameters to tune:
- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [16, 32, 64]
- Dropout: [0.2, 0.3, 0.5]
- Number of mel bins: [64, 128, 256]
- Audio duration: [2.0, 3.0, 5.0]

### Data Augmentation Experiments

```python
# Test augmentation impact
augmentations = [
    'none',
    'noise_only',
    'time_stretch_only',
    'spec_augment_only',
    'all_augmentations'
]
```

### Robustness Testing

- Test with added noise at different SNR levels
- Test with different environmental sounds
- Cross-dataset evaluation (train on A, test on B)

## Week 8: Detection Module Completion

### Finalize Best Model

- [ ] Select best architecture
- [ ] Select best features
- [ ] Select best hyperparameters
- [ ] Train final model on full data
- [ ] Save model weights

### Document Results

Create `docs/DETECTION_RESULTS.md`:
- Final architecture diagram
- Performance metrics
- Comparison with literature
- Failure case analysis

---

# PHASE 3: EXTENDED FEATURES (Weeks 9-13)

## Week 9-10: Classification Module

### Multi-class Classification

```bash
# Train classifier for drone types
python train.py \
    --datasets alemadi \
    --model efficientnet_b0 \
    --num-classes 4 \  # Unknown, Bebop, AR, Phantom
    --experiment-name classification_v1
```

### Analyze What Makes Drones Different

Create `notebooks/04_drone_signatures.ipynb`:
- Compare spectrograms of different drone types
- Analyze Blade Passing Frequencies
- Identify distinguishing features

## Week 11-12: Localization (If Multi-channel Data Available)

### Option A: With Real Multi-channel Data

If you can access UaVirBASE or DREGON:
- Implement GCC-PHAT
- Implement DOA estimation
- Evaluate localization accuracy

### Option B: Simulation-based

If only single-channel data:
- Use `pyroomacoustics` to simulate microphone array
- Generate synthetic multi-channel data
- Implement and test localization algorithms
- Document as "proof of concept"

## Week 13: Novel Contribution - Rotor Analysis

### Blade Passing Frequency Analysis

```python
# This is a research opportunity!
# No one has systematically studied rotor counting from audio

def analyze_rotor_signature(audio, sr):
    """
    Attempt to estimate number of rotors from BPF harmonics.
    
    Theory:
    - BPF = (RPM / 60) × num_blades
    - Quadcopter: 4 rotors → 4 fundamental frequencies (slightly different)
    - Hexacopter: 6 rotors → 6 fundamental frequencies
    
    The frequency spectrum should show this pattern.
    """
    # Your implementation here
    pass
```

---

# PHASE 4: WRITING (Weeks 14-17)

## Thesis Structure

```
1. Introduction (5-10 pages)
   1.1 Background and Motivation
   1.2 Problem Statement
   1.3 Research Objectives
   1.4 Contributions
   1.5 Thesis Organization

2. Literature Review (15-20 pages)
   2.1 Drone Detection Methods
       2.1.1 Acoustic-based
       2.1.2 RF-based
       2.1.3 Vision-based
       2.1.4 Radar-based
   2.2 Audio Feature Extraction
   2.3 Deep Learning for Audio Classification
   2.4 Sound Source Localization
   2.5 Summary and Research Gaps

3. Methodology (15-20 pages)
   3.1 System Overview
   3.2 Dataset Description
   3.3 Preprocessing Pipeline
   3.4 Feature Extraction
   3.5 Model Architectures
   3.6 Training Procedure
   3.7 Evaluation Metrics

4. Experiments and Results (20-25 pages)
   4.1 Experimental Setup
   4.2 Detection Results
   4.3 Classification Results
   4.4 Localization Results (if applicable)
   4.5 Ablation Studies
   4.6 Comparison with State-of-the-Art
   4.7 Discussion

5. Conclusion and Future Work (5 pages)
   5.1 Summary of Contributions
   5.2 Limitations
   5.3 Future Research Directions

References (3-5 pages)

Appendices
   A. Additional Results
   B. Code Documentation
   C. Dataset Details
```

## Writing Schedule

- **Week 14:** Chapters 1-2 (Introduction, Literature Review)
- **Week 15:** Chapter 3 (Methodology)
- **Week 16:** Chapter 4 (Results)
- **Week 17:** Chapter 5 + Revision

---

# PHASE 5: DEFENSE PREPARATION (Week 18)

## Prepare Presentation (15-20 slides)

1. Title & Introduction (2 slides)
2. Problem & Motivation (2 slides)
3. Literature Summary (2 slides)
4. Methodology (4 slides)
5. Results (5 slides)
6. Demo (if applicable) (1 slide)
7. Conclusion & Future Work (2 slides)
8. Q&A

## Anticipate Questions

Common thesis defense questions:
- Why did you choose this approach?
- What are the limitations?
- How does this compare to existing work?
- What would you do differently?
- What are the practical applications?
- How would this work in real-time?

---

# WEEKLY CHECKLIST TEMPLATE

Use this for each week:

```markdown
# Week X Checklist

## Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Experiments Run
- Experiment 1: Result
- Experiment 2: Result

## Problems Encountered
- Problem 1: Solution
- Problem 2: Still working on it

## Next Week Plan
- ...

## Notes for Thesis
- ...
```

---

# IMMEDIATE NEXT STEPS (This Week)

## Today
1. [ ] Extract the repository zip file
2. [ ] Set up Python environment
3. [ ] Run `verify_setup.py`
4. [ ] Download Al-Emadi dataset

## Tomorrow
5. [ ] Open `01_getting_started.ipynb`
6. [ ] Listen to drone audio samples
7. [ ] Visualize spectrograms

## This Week
8. [ ] Read Al-Emadi paper (primary reference)
9. [ ] Train baseline model
10. [ ] Document initial results

---

# QUESTIONS TO DISCUSS WITH YOUR ADVISOR

Before your next advisor meeting, prepare answers for:

1. What is your specific thesis title?
2. What datasets will you use?
3. What is your novel contribution?
4. What is your timeline?
5. Do you have access to recording equipment?
6. Are there any computational resource concerns?

---

Good luck with your research! 🎓

Remember: Start simple, iterate often, document everything.
