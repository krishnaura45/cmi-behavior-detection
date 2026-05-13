# CMI - Detect Behavior with Sensor Data Competition Analysis & Improvement Tips

## Competition Overview

### What is the Competition About?
The **CMI - Detect Behavior with Sensor Data** is a Kaggle competition focused on **Body-Focused Repetitive Behaviors (BFRBs) detection** using wrist-worn sensor data. The goal is to classify 18 different gestures from time-series sensor data collected from IMU (Inertial Measurement Unit) sensors.

### Target Classes (8 BFRBs - Body-Focused Repetitive Behaviors):
1. Above ear - pull hair
2. Cheek - pinch skin  
3. Eyebrow - pull hair
4. Eyelash - pull hair
5. Forehead - pull hairline
6. Forehead - scratch
7. Neck - pinch skin
8. Neck - scratch

### Control Classes (10 Non-target behaviors):
1. Write name on leg
2. Wave hello
3. Glasses on/off
4. Text on phone
5. Write name in air
6. Feel around in tray and pull out an object
7. Scratch knee/leg skin
8. Pull air toward your face
9. Drink from bottle/cup
10. Pinch knee/leg skin

### Dataset Structure
- **Sensor Data**: IMU sensors (accelerometer + gyroscope) + ToF (Time-of-Flight) sensors + thermal sensors
- **Features**: ~341 columns including acc_x, acc_y, acc_z, rot_x, rot_y, rot_z, rot_w, thm_*, tof_*
- **Demographics**: age, sex, handedness, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm
- **Sequence-based**: Variable length time series (average ~69 timesteps, up to ~700)

### Evaluation Metric
The competition uses a **Hierarchical F1-score** that considers the hierarchical relationship between target (BFRB) and non-target (control) behaviors.

## Approach 2 Analysis (LB 0.86)
This approach uses an **ensemble of 3 models**:
1. **Model 1 (LB 0.820)**: TensorFlow BlendingModel with IMU+THM/TOF data (40 models total)
2. **Model 2 (LB 0.829)**: PyTorch BERT-based model (5-fold) 
3. **Model 3 (LB 0.835)**: Gated GRU + Hybrid Ensemble (10 models total)

## Improvement Thoughts

### 1. Advanced Feature Engineering

#### World Coordinate Transformation
```python
def transform_to_world_coordinates(acc_data, rot_data):
    # Transform sensor data from device coordinate to world coordinate
    # This makes the data more interpretable and potentially better for ML
```

**Benefits**:
- Makes accelerometer data independent of device orientation
- Enables extraction of gravity-independent linear acceleration
- Provides more physically meaningful features

#### Enhanced IMU Processing
- **Linear acceleration** (gravity removed using quaternion rotation)
- **Angular velocity** calculated from quaternion derivatives  
- **World coordinate acceleration** using quaternion transformations
- **Advanced ToF statistics** (IQR, range, etc.)

### 2. Physically Plausible Data Augmentation

Traditional augmentation methods often create unrealistic sensor data. The new approach implements:

#### Realistic Wrist Rotation
```python
def apply_rotation(self, imu_data):
    # Apply realistic wrist rotation (±30 degrees around wrist axis)
    angle = np.random.uniform(-np.pi/6, np.pi/6)
    axis = np.array([1, 0, 0])  # Rotation around wrist axis
```

#### Sensor-Specific Noise
- Different noise levels for accelerometer vs gyroscope
- Preserves gravitational component in acceleration
- Physically motivated scaling factors

#### Time Warping
- Smooth time distortions that preserve signal characteristics
- Models natural variations in gesture execution speed

### 3. TSLANet-Inspired Architecture

#### Adaptive Spectral Block
```python
class AdaptiveSpectralBlock(nn.Module):
    # Uses FFT for frequency domain processing
    # Adaptive thresholding for noise removal
    # Combines local and global filters
```

**Expected Benefits**:
- Frequency domain noise filtering
- Adaptive thresholding based on signal characteristics
- Both local (CNN-like) and global (Transformer-like) processing

#### Interactive Convolution Block
- Multi-scale feature extraction
- Residual connections for better gradient flow
- Spectral processing integration

### 4. Advanced Ensemble Strategy

#### Stacking Ensemble
```python
class StackingEnsemble:
    # Cross-validation based meta-learning
    # Combines predictions using Ridge regression
    # Reduces overfitting compared to simple averaging
```

**Benefits**:
- Meta-learner learns optimal combination weights
- Cross-validation prevents overfitting
- More sophisticated than weighted averaging

#### Hierarchical Loss Function
```python
class HierarchicalF1Loss(nn.Module):
    # Considers target vs non-target hierarchy
    # Balances fine-grained and coarse-grained classification
```

### 5. Enhanced Cross-Validation Strategy

- **Stratified K-Fold** on both subject and gesture class
- **Group-aware splitting** to prevent data leakage
- **Out-of-fold prediction generation** for stacking

### 6. Post-Processing

#### Confidence Boosting
```python
# Boost confidence for target classes (BFRBs)
target_gesture_indices = [0, 1, 2, 3, 4, 5, 6, 7]
ensemble_pred[target_gesture_indices] *= 1.05
```

#### Bias Correction
- Maintains existing bias corrections that worked well
- Adds adaptive corrections based on prediction confidence

## Implementation Tips

### Phase 1: Drop-in Replacement
The improved notebook is designed as a **drop-in replacement** for your current ensemble:
- Maintains compatibility with existing model loading
- Adds enhanced features gradually
- Falls back to existing approach if enhanced features fail

### Phase 2: Training Enhanced Models
1. Train new models with enhanced features and architecture
2. Use physically plausible augmentation
3. Apply hierarchical F1 loss during training

### Phase 3: Advanced Ensemble
1. Implement stacking ensemble with cross-validation
2. Combine existing models + new enhanced models
3. Use meta-learner for optimal weight learning

## Technical Considerations

### Computational Overhead
- Enhanced feature engineering: +10-20% preprocessing time
- TSLANet architecture: Similar to existing PyTorch models
- Stacking ensemble: +5-10% inference time
- Overall: Minimal impact on submission time limits

### Memory Usage
- Enhanced features: +20-30% memory for feature matrix
- Model ensemble: Comparable to current approach
- Batch processing: Optimized for Kaggle GPU memory limits

### Robustness
- Fallback mechanisms for missing features
- Compatible with existing data preprocessing
- Handles variable sequence lengths gracefully

This approach should provide a meaningful boost to your current LB score of 0.86, with potential to reach 0.865 - 0.885 range based on the sophistication of the improvements and their synergistic effects.
