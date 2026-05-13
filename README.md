# CMI - Detect Behavior with Sensor Data ~ `Actual 71st Place Solution`
> Detecting hand gestures and behavioral patterns from multimodal wearable sensor data using deep learning and ensemble strategies.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep--Learning-Gesture%20Recognition-purple?style=for-the-badge)
![Best Score](https://img.shields.io/badge/Best%20Private%20Score-0.83713-2ECC71?style=for-the-badge)
![Rank](https://img.shields.io/badge/Rank-137%20of%202657-brightgreen?style=for-the-badge)
![Solo](https://img.shields.io/badge/Submission-Solo-orange?style=for-the-badge)

### Project Duration: Aug 10, 2025 - Sept 1, 2025

---

## File Structure

```bash
├── INFO.md
├── inference
│   ├── BASE.md
│   ├── ensemble-inference.py
│   └── inference-notebook.ipynb
├── masked-batchnorm.py
├── metric.py
├── model.py
├── requirements.txt
├── test.py
├── train.py
├── utils.py
└── extras
   ├── analysis-AI.md
   ├── class-gestures.png
   └── optim-trials.png
```

### Installation

```bash
git clone https://github.com/krishnaura45/cmi-behavior-detection.git
cd cmi-behavior-detection

pip install -r requirements.txt
```

### Usage
> After donloading competition data, run in CLI
```bash
python train.py
python test.py
```

---

## Problem Statement

The objective of the **CMI - Detect Behavior with Sensor Data** Kaggle competition was to classify human behavioral gestures using multimodal wearable sensor signals collected from IMU, TOF, and THM sensors. 

The challenge focused on recognizing gesture patterns and orientations from sequential sensor readings while handling missing modalities, subject-specific inconsistencies, and noisy temporal dynamics.

Hosted on Kaggle, submissions were evaluated using the official competition metric, where higher scores indicated better classification performance on hidden behavioral sequences.

---

## Approach 1

### Data Handling & Missingness

- Trained separate model variants depending on sensor availability:
  - IMU rotation present / absent
  - THM and TOF present / absent
- Created specialized models to improve robustness against missing modalities.

### Feature Engineering

#### IMU Features
- Accelerometer signals `(x/y/z)`
- Angular velocity `(x/y/z)`
- Linear acceleration `(x/y/z)`
- Quaternion 6D representation for stable rotational encoding

#### THM / TOF Processing
- Replaced `NaN` and `-1` values with zeros
- Applied transformations to TOF grids for consistent spatial representation

#### Left-Handed Subject Corrections
- Reflected selected IMU channels
- Swapped TOF indices and horizontally flipped TOF grids

#### Subject-Specific Correction
- Corrected orientation inconsistencies for specific subjects rotated by 180° around the z-axis
- Ignored unreliable TOF information for corrected samples

---

### Architecture

The architecture combines:

- Residual SE-CNN Blocks
- Attention-based temporal modeling
- Multi-branch modality-specific CNN stems

#### Modality-Specific Learning
Independent 1D CNN branches were used for:
- Accelerometer
- Quaternion embeddings
- Angular velocity
- Linear acceleration
- Individual TOF sensors

#### TOF Processing
- Applied 2D CNNs on TOF grids per frame
- Mean pooled temporal representations before sequence modeling

---

### Phase-Aware Attention

Instead of relying on plain temporal attention:

- Introduced auxiliary phase prediction:
  - Relaxes / Moves hand
  - Hand at target location
  - Performs gesture

Three separate attentions were constructed and weighted using phase probabilities, enabling better focus on gesture-specific temporal regions.

---

### Training Strategy

- Optimizer: `Adam`
- Scheduler: `Cosine Annealing`
- Epochs: `50`
- Batch Size: `32`
- Folds: `10-fold CV`

Additional techniques:
- Phase-aligned Mixup augmentation
- Online pseudo-labeling during inference
- Ensemble of multiple architecture variants

---

### Post-Processing Strategy

A major leaderboard improvement came from exploiting dataset structure:

- Each subject contained:
  - `51` unique `(orientation, gesture)` pairs
  - Each repeated with two initial behavior states
  - Total: `102` behavioral classes

Instead of independent argmax predictions:

- Applied **joint probability maximization**
- Enforced a **no-repeat constraint** across labels for each subject
- Solved efficiently using assignment optimization techniques similar to the Hungarian algorithm

This significantly improved consistency and leaderboard performance.

---

## Approach 2 (Ensemble)

### Models Utilized

- Multiple fold-based prediction pipelines
- Architecture depth variants
- Pseudo-labeled inference checkpoints

### Ensemble Strategy

- Combined prediction probabilities from multiple inference runs
- Applied weighted averaging across folds and model variants
- Leveraged post-processing constraints during final prediction generation

### Other Details

- Inference pipeline implemented inside: `inference/best-inference.ipynb`

---

## Results 

- **Public Leaderboard Scores**:
  - `0.39238`
  - `0.77163`
  - `0.83702`
  - `0.84054`
  - `0.84936`
  - `0.85507`
  - `0.85690`

- **Private Leaderboard Scores**:
  - `0.39413`
  - `0.76978`
  - `0.82031`
  - `0.83221`
  - `0.83615`
  - `0.83688`
  - `0.83713`

- **Best Private Score**:
  - `0.837125`

- **Rank Achieved**:
  - Final official rank: **137 / 3178 participants** and **2657 teams** (solo)
  - The best private score would have corresponded to approximately **71st place**, but an older public-focused submission (`0.849004`) was mistakenly selected before the competition deadline.

---

## Important Links

- Kaggle Competition: [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data)
- Competition Info: `INFO.md`
- Baseline Notes: `inference/BASE.md`
- Ensemble Inference: `inference/ensemble-inference.py`

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - `torch`, `torchvision`
  - `numpy`, `pandas`
  - `scikit-learn`
  - `opencv-python`
- **Techniques**:
  - Residual CNNs
  - Attention Mechanisms
  - Pseudo Labeling
  - Phase-aware Temporal Modeling
  - Ensemble Learning
- **Tools**:
  - Jupyter Notebook
  - Kaggle Kernels
  - Google Colab

---

📌 *This project demonstrates how domain-aware preprocessing, temporal attention mechanisms, pseudo-labeling, and constrained post-processing can significantly improve behavioral gesture recognition performance from multimodal sensor data.*

<!--## Overview

This project implements a deep learning model that processes sensor data to detect and classify various behaviors. The model combines IMU data (accelerometer and gyroscope readings) with TOF sensor data to make predictions.

## Features

- **Multi-modal Sensor Fusion**: Combines IMU and TOF sensor data
- **Advanced Feature Engineering**: 
  - 6D rotation representation from quaternions
  - Angular velocity calculation
  - Linear acceleration computation (gravity removed)
- **Multiple Model Architectures**: 
  - IMU-only models
  - IMU+TOF combined models
  - Various configurations (simple, deep, 2.5D)
- **Data Augmentation**:
  - Mixup augmentation (phase-aware)
  - TOF masking augmentation
  - Subject-specific corrections
- **Training Framework**:
  - PyTorch Lightning for organized training
  - StratifiedGroupKFold cross-validation
  - Mixed precision training (TF32)
  - Cosine annealing scheduler with warmup
  - Gradient clipping and early stopping
- **Evaluation Metrics**:
  - 72-class accuracy (orientation × gesture × phase1)
  - Hierarchical F1 scores (18-class and 9-class)
  - Binary F1 (Non-Target vs Target gestures)

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- torch
- lightning
- pandas
- polars
- scikit-learn
- transformers
- wandb
- tyro

## Usage

### Training

Full training (10-fold cross-validation):
```bash
python train.py
```

Single fold training:
```bash
python train.py --fold 0
```

Debug mode (1 batch for quick testing):
```bash
python train.py --debug
```

### Model Configuration

- Disable TOF features (IMU-only): `python train.py --model_type imu`
- Enable TOF features: `python train.py --model_type all`
- Use mixup augmentation: `python train.py --use_mixup --mixup_alpha 0.5`
- TOF masking augmentation: `python train.py --use_tof_mask_augmentation_prob 0.1`
- Zero out rotation features: `python train.py --rot_zero`

Available model types:
- `imu`: IMU-only model
- `all`: IMU + TOF model
- `imu_simple`: Simple IMU model
- `all_simple`: Simple IMU+TOF model
- `imu_deep`: Deep IMU model
- `all_25d`: 2.5D IMU+TOF model

## Data Pipeline

1. **Data Loading**: Loads and merges train.csv with train_demographics.csv
2. **Feature Engineering**: 
   - Processes raw IMU data (acc_x/y/z, rot_x/y/z/w, handedness)
   - Creates engineered features: 6D rotation representation, angular velocity, linear acceleration
   - Processes TOF sensor data (5 sensors × 64 vertices each)
3. **Label Creation**: Combines orientation, gesture, and phase1_behavior into multi-class labels
4. **Dataset Processing**: Handles sequences, applies transformations, and manages subject-specific corrections
5. **Augmentation**: Applies mixup, TOF masking, and subject-specific augmentations

## Model Architecture

The model consists of several key components:
- **Residual SECNN blocks** with Masked Batch Normalization
- **Squeeze-and-Excitation (SE) blocks** for channel attention
- **Phase prediction head** (3 phases: 0, 1, 2)
- **Phase-aware attention pooling** for each phase
- **Dense layers** for final classification

## Special Handling

- **Subject Corrections**: Specific subjects (SUBJ_019262, SUBJ_045235) require axis inversion
- **TOF Processing**: 5 sensors treated as 8×8 grids with spatial convolutions
- **Sequence Truncation**: Sequences longer than 200 timesteps are truncated from the beginning
- **Padding**: Variable-length sequences handled via masking in collate function and model components

## Output

- Model checkpoints saved to `../../output/<exp_name>/<run_idx>/checkpoints/`
- Validation logits saved for ensembling when `save_logits=True`
- Weights & Biases integration for metrics tracking (project: "CMI2025")
-->

## References

For more details about the competition, visit the [Kaggle CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) page.
