# CMI - Detect Behavior with Sensor Data
> ## Actual 71st Place Solution

## Overview

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

## References

For more details about the competition, visit the [Kaggle CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) page.
