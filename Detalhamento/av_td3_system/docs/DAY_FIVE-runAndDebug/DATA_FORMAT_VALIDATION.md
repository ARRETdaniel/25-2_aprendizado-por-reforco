# Data Format Validation Guide
**Date**: 2025-11-05  
**Purpose**: Step-by-step validation of data formats at each pipeline stage  
**Documentation**: CARLA 0.9.16, OpenAI Spinning Up TD3, TD3 Paper (Fujimoto et al. 2018)

---

## 📋 **Overview**

This document provides the expected data format at each transformation stage of the TD3+CNN autonomous driving system. Each section includes:
1. **Official documentation reference** (CARLA 0.9.16 or TD3 specs)
2. **Expected input/output formats** with dimensions
3. **Validation criteria** (shape, dtype, range, statistics)
4. **Debug log patterns** to check

---

## 🎯 **Complete Data Flow**

```
CARLA RGB Camera → Preprocessing → Frame Stacking → CNN Feature Extractor → State Vector → TD3 Actor → Action → CARLA Control
     BGRA 32-bit      RGB→Gray→84x84    Stack 4 frames   Extract 512D      Concat features   [steer,throttle]  Vehicle control
     (raw_data)       (1,84,84)         (4,84,84)        features          (535D vector)     [-1,1]²          [steer,throttle,brake]
```

---

## 1️⃣ **CARLA RGB Camera Output**

### Official Documentation
**Source**: [CARLA Sensors Reference - RGB Camera](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera)

> **Blueprint**: `sensor.camera.rgb`  
> **Output**: `carla.Image` per step  
> **raw_data**: Array of **BGRA 32-bit pixels**

**Output Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `width` | int | Image width in pixels (e.g., 800) |
| `height` | int | Image height in pixels (e.g., 600) |
| `fov` | float | Horizontal field of view in degrees (e.g., 90.0) |
| `raw_data` | bytes | **Array of BGRA 32-bit pixels** |

### Expected Format

**Raw Output**:
```python
# CARLA camera callback
def _on_camera_image(image):
    # Raw data format (OFFICIAL CARLA 0.9.16 SPEC):
    # - Format: BGRA (Blue, Green, Red, Alpha)
    # - Depth: 32-bit per pixel (8-bit per channel)
    # - Shape: (height, width, 4) when reshaped
    # - Data type: uint8
    # - Value range: [0, 255]
    
    assert image.width == 800  # Default width
    assert image.height == 600  # Default height
    assert len(image.raw_data) == 800 * 600 * 4  # BGRA = 4 channels
    
    # Convert to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # (600, 800, 4)
    # array[:,:,0] = Blue channel
    # array[:,:,1] = Green channel
    # array[:,:,2] = Red channel
    # array[:,:,3] = Alpha channel (usually 255)
```

### Validation Criteria
✅ **Shape**: `(height, width, 4)` - exactly 4 channels (BGRA)  
✅ **Dtype**: `uint8` - 8-bit unsigned integer  
✅ **Range**: `[0, 255]` - no negative values  
✅ **Alpha channel**: All values should be 255 (fully opaque)

### Debug Log Pattern
```log
[CARLA][Camera] Received image: 800x600x4 (BGRA), dtype=uint8, range=[0, 255]
[CARLA][Camera] Alpha channel check: min=255, max=255 ✓
```

---

## 2️⃣ **Image Preprocessing**

### Implementation Reference
**File**: `src/environment/carla_env.py` or preprocessing module

**Official Rationale**: CARLA outputs BGRA, but deep learning pipelines expect RGB or grayscale.

### Transformation Steps

#### Step 2.1: BGRA → RGB Conversion
```python
# Input: (600, 800, 4) BGRA uint8
bgra_image = camera_raw_data  # From CARLA

# Convert BGRA to RGB (drop alpha, reorder channels)
rgb_image = bgra_image[:, :, [2, 1, 0]]  # Output: (600, 800, 3) RGB uint8
# [:,:,0] = Red (was [:,:,2] in BGRA)
# [:,:,1] = Green (unchanged)
# [:,:,2] = Blue (was [:,:,0] in BGRA)
```

**Validation**:
- ✅ Shape: `(600, 800, 3)` (lost alpha channel)
- ✅ Dtype: `uint8` (unchanged)
- ✅ Range: `[0, 255]` (unchanged)

#### Step 2.2: RGB → Grayscale Conversion
```python
# Input: (600, 800, 3) RGB uint8
rgb_image = preprocessed_rgb

# Convert to grayscale using luminosity method (ITU-R BT.601 standard)
# Grayscale = 0.299*R + 0.587*G + 0.114*B
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # Output: (600, 800) uint8

# Alternative numpy implementation:
# gray_image = (0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]).astype(np.uint8)
```

**Validation**:
- ✅ Shape: `(600, 800)` (2D, single channel)
- ✅ Dtype: `uint8`
- ✅ Range: `[0, 255]`

#### Step 2.3: Resize to 84x84
```python
# Input: (600, 800) grayscale uint8
gray_image = preprocessed_gray

# Resize to 84x84 (standard for DQN/TD3 vision tasks)
# Rationale: Smaller size reduces computation, maintains spatial structure
resized_image = cv2.resize(gray_image, (84, 84), interpolation=cv2.INTER_AREA)
# Output: (84, 84) uint8
```

**Validation**:
- ✅ Shape: `(84, 84)` (target size)
- ✅ Dtype: `uint8` (unchanged)
- ✅ Range: `[0, 255]` (may have slight interpolation artifacts)

#### Step 2.4: Normalization to [0, 1]
```python
# Input: (84, 84) uint8
resized_image = preprocessed_resized

# Normalize to [0, 1] for neural network input
normalized_image = resized_image.astype(np.float32) / 255.0
# Output: (84, 84) float32, range [0, 1]
```

**Validation**:
- ✅ Shape: `(84, 84)` (unchanged)
- ✅ Dtype: `float32` (changed for NN compatibility)
- ✅ Range: `[0.0, 1.0]` (normalized)

### Debug Log Pattern
```log
[Preprocessing] BGRA→RGB: (600,800,4)→(600,800,3) ✓
[Preprocessing] RGB→Gray: (600,800,3)→(600,800) ✓
[Preprocessing] Resize: (600,800)→(84,84) ✓
[Preprocessing] Normalize: dtype=uint8→float32, range=[0,255]→[0.0,1.0] ✓
[Preprocessing] Output: shape=(84,84), dtype=float32, min=0.003, max=0.997, mean=0.451
```

---

## 3️⃣ **Frame Stacking**

### TD3/DRL Rationale
**Source**: TD3 paper (Fujimoto et al. 2018), DQN paper (Mnih et al. 2015)

> "For Atari environments, we stack 4 consecutive frames to provide temporal information (velocity, acceleration) that single frames cannot capture."

**Purpose**:
- Capture **temporal dynamics** (velocity, direction of movement)
- Help agent infer **motion** from static images
- Standard practice in DRL for vision-based control

### Implementation

```python
# Frame buffer (FIFO queue)
frame_buffer = collections.deque(maxlen=4)  # Holds last 4 frames

# At each timestep:
current_frame = preprocess_image(camera_data)  # (84, 84) float32
frame_buffer.append(current_frame)

# Stack frames (oldest to newest)
if len(frame_buffer) < 4:
    # Pad with zeros if not enough frames yet (episode start)
    while len(frame_buffer) < 4:
        frame_buffer.appendleft(np.zeros((84, 84), dtype=np.float32))

stacked_frames = np.stack(frame_buffer, axis=0)  # Output: (4, 84, 84) float32
# stacked_frames[0] = oldest frame (t-3)
# stacked_frames[1] = frame at t-2
# stacked_frames[2] = frame at t-1
# stacked_frames[3] = current frame (t)
```

### Validation Criteria
✅ **Shape**: `(4, 84, 84)` - exactly 4 frames  
✅ **Dtype**: `float32` (matches preprocessed frames)  
✅ **Range**: `[0.0, 1.0]` (inherited from normalization)  
✅ **Temporal ordering**: `[t-3, t-2, t-1, t]` (oldest to newest)  
✅ **Episode start**: Zero-padding if fewer than 4 frames available

### Debug Log Pattern
```log
[Frame Stacking] Buffer size: 4/4 ✓
[Frame Stacking] Stacked shape: (4, 84, 84), dtype=float32
[Frame Stacking] Temporal order: [t-3, t-2, t-1, t(current)] ✓
[Frame Stacking] Statistics: min=0.0, max=1.0, mean=0.423
```

---

## 4️⃣ **CNN Feature Extraction**

### Architecture Reference
**Source**: OpenAI Spinning Up, Stable-Baselines3 NatureCNN

**TD3 + CNN Architecture**:
```
Input: (batch, 4, 84, 84) float32
  ↓
Conv2D(4→32, kernel=8, stride=4) + ReLU
  ↓ (batch, 32, 20, 20)
Conv2D(32→64, kernel=4, stride=2) + ReLU
  ↓ (batch, 64, 9, 9)
Conv2D(64→64, kernel=3, stride=1) + ReLU
  ↓ (batch, 64, 7, 7)
Flatten
  ↓ (batch, 3136)
Linear(3136→512) + ReLU
  ↓ (batch, 512)
Output: 512D feature vector
```

### Input Format

```python
# Expected input to CNN
stacked_frames = ... # (4, 84, 84) float32 from frame stacking

# Add batch dimension for PyTorch
batch_input = torch.FloatTensor(stacked_frames).unsqueeze(0)  # (1, 4, 84, 84)

# Forward pass
with torch.no_grad():
    cnn_features = actor_cnn(batch_input)  # Output: (1, 512) float32
    cnn_features = cnn_features.cpu().numpy().squeeze()  # (512,) numpy array
```

### Validation Criteria
✅ **Input shape**: `(1, 4, 84, 84)` or `(batch, 4, 84, 84)`  
✅ **Input dtype**: `torch.FloatTensor` (float32)  
✅ **Input range**: `[0.0, 1.0]` (normalized images)  
✅ **Output shape**: `(1, 512)` or `(batch, 512)`  
✅ **Output dtype**: `torch.FloatTensor` (float32)  
✅ **Activation health**:
   - No NaN or Inf values
   - 30-60% of neurons active (ReLU > 0.01)
   - L2 norm typically 2-5 (feature magnitude)

### Debug Log Pattern
```log
[CNN FORWARD PASS - INPUT]
  Shape: (1, 4, 84, 84), dtype=float32
  Range: [0.0, 1.0], Mean: 0.423, Std: 0.285

[CNN FORWARD PASS - OUTPUT]
  Shape: (1, 512), dtype=float32
  Feature Stats:
    L2 Norm: 2.872
    Mean: 0.127, Std: 0.341
    Range: [-0.891, 1.423]
    Active neurons (>0.01): 234/512 (45.7%)
  ✓ No NaN/Inf detected
  ✓ Healthy activation pattern
```

---

## 5️⃣ **State Vector Construction**

### TD3 State Representation
**Source**: TD3 paper, project configuration

**State Vector Composition**:
```
Final State = [CNN Features, Kinematic Features, Waypoint Features]
            = [512D,          3D,               20D]
            = 535D total
```

### Components

#### 5.1: CNN Features (512D)
```python
# From CNN forward pass
cnn_features = actor_cnn(stacked_frames)  # (512,) float32
```

#### 5.2: Kinematic Features (3D)
```python
# Vehicle state from CARLA
velocity = vehicle.get_velocity()  # carla.Vector3D
v_magnitude = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s

# Lane deviation (from waypoint calculations)
lateral_deviation = distance_from_lane_center  # meters

# Heading error (angle difference between vehicle heading and lane direction)
heading_error = angle_to_lane_direction  # radians

# Normalize kinematic features
kinematic_vector = np.array([
    v_magnitude / 30.0,  # Normalize by max speed (30 m/s = 108 km/h)
    lateral_deviation / 5.0,  # Normalize by lane width (~5m)
    heading_error / np.pi  # Normalize by π radians (180°)
], dtype=np.float32)  # (3,)
```

#### 5.3: Waypoint Features (20D)
```python
# 10 waypoints × 2 coordinates (x, y relative to vehicle)
num_waypoints = 10
waypoint_coords = []  # List of (x, y) tuples in vehicle-local frame

for wp in waypoints[:num_waypoints]:
    # Transform to vehicle-local coordinates
    local_x = ...  # Forward distance
    local_y = ...  # Lateral offset
    waypoint_coords.append([local_x / 50.0, local_y / 5.0])  # Normalize

waypoint_vector = np.array(waypoint_coords, dtype=np.float32).flatten()  # (20,)
```

#### 5.4: Concatenation
```python
# Concatenate all features
state_vector = np.concatenate([
    cnn_features,      # (512,)
    kinematic_vector,  # (3,)
    waypoint_vector    # (20,)
], axis=0)  # Output: (535,) float32
```

### Validation Criteria
✅ **Total shape**: `(535,)` = 512 + 3 + 20  
✅ **Dtype**: `float32` (consistent across all components)  
✅ **Range**: Typically `[-5.0, 5.0]` (most normalized features)  
✅ **No NaN/Inf**: Critical for TD3 stability  
✅ **Component boundaries**:
   - `state[0:512]` = CNN features
   - `state[512:515]` = Kinematic features
   - `state[515:535]` = Waypoint features

### Debug Log Pattern
```log
[State Construction]
  CNN features: (512,), mean=0.127, std=0.341
  Kinematic: velocity=0.523, lateral_dev=0.102, heading_err=0.034
  Waypoints: 10 waypoints, mean_dist=25.3m
  Final state: (535,), dtype=float32, no NaN/Inf ✓
```

---

## 6️⃣ **TD3 Actor Forward Pass**

### TD3 Actor Architecture
**Source**: TD3 paper (Fujimoto et al. 2018), OpenAI Spinning Up

**Network Structure**:
```
Input: (batch, 535) float32
  ↓
Linear(535→256) + ReLU
  ↓ (batch, 256)
Linear(256→256) + ReLU
  ↓ (batch, 256)
Linear(256→2) + Tanh
  ↓ (batch, 2)
Output: [steering, throttle/brake] in [-1, 1]²
```

### Forward Pass

```python
# Input state (constructed above)
state_vector = ...  # (535,) numpy array

# Convert to PyTorch tensor with batch dimension
state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)  # (1, 535)

# Forward pass through actor network
with torch.no_grad():
    action = actor(state_tensor)  # (1, 2) torch.FloatTensor
    action = action.cpu().numpy().flatten()  # (2,) numpy array

# Action components
steering = action[0]  # [-1, 1]
throttle_brake = action[1]  # [-1, 1]
```

### Validation Criteria
✅ **Input shape**: `(1, 535)` or `(batch, 535)`  
✅ **Input dtype**: `torch.FloatTensor` (float32)  
✅ **Output shape**: `(1, 2)` or `(batch, 2)`  
✅ **Output dtype**: `torch.FloatTensor` (float32)  
✅ **Output range**: Exactly `[-1.0, 1.0]` due to Tanh activation  
✅ **Action validity**:
   - `steering ∈ [-1, 1]` (full left to full right)
   - `throttle_brake ∈ [-1, 1]` (full brake to full throttle)

### Debug Log Pattern
```log
[TD3 Actor - Forward Pass]
  Input state: (1, 535), dtype=float32
  Hidden layer 1: (1, 256), mean=0.234, active=187/256 (73%)
  Hidden layer 2: (1, 256), mean=0.189, active=172/256 (67%)
  Output action: [steering=0.123, throttle=0.456]
  ✓ Actions in valid range [-1, 1]
```

---

## 7️⃣ **Action Mapping to CARLA Control**

### CARLA VehicleControl
**Source**: [CARLA Python API - VehicleControl](https://carla.readthedocs.io/en/latest/python_api/#carlavehiclecontrol)

**Control Attributes**:
```python
control = carla.VehicleControl()
control.steer = ...      # float in [-1.0, 1.0]
control.throttle = ...   # float in [0.0, 1.0]
control.brake = ...      # float in [0.0, 1.0]
control.hand_brake = ... # bool
control.reverse = ...    # bool
```

### Action Mapping

```python
# TD3 output
steering = action[0]  # [-1, 1]
throttle_brake = action[1]  # [-1, 1]

# CARLA control (separate throttle and brake)
control = carla.VehicleControl()
control.steer = float(np.clip(steering, -1.0, 1.0))  # Direct mapping

if throttle_brake >= 0:
    # Positive = throttle
    control.throttle = float(np.clip(throttle_brake, 0.0, 1.0))
    control.brake = 0.0
else:
    # Negative = brake
    control.throttle = 0.0
    control.brake = float(np.clip(-throttle_brake, 0.0, 1.0))

control.hand_brake = False
control.reverse = False
```

### Validation Criteria
✅ **Steering range**: `[-1.0, 1.0]` (CARLA requirement)  
✅ **Throttle range**: `[0.0, 1.0]` (CARLA requirement)  
✅ **Brake range**: `[0.0, 1.0]` (CARLA requirement)  
✅ **Mutual exclusivity**: throttle=0 when braking, brake=0 when accelerating

### Debug Log Pattern
```log
[Action Mapping]
  TD3 output: steering=0.123, throttle_brake=0.456
  CARLA control: steer=0.123, throttle=0.456, brake=0.000
  ✓ All values in valid CARLA ranges
```

---

## 🎯 **Complete Validation Checklist**

Use this checklist to validate the complete data pipeline:

### Stage 1: CARLA Camera ✅
- [ ] Shape: (height, width, 4) BGRA
- [ ] Dtype: uint8
- [ ] Range: [0, 255]
- [ ] Alpha channel: all 255

### Stage 2: Preprocessing ✅
- [ ] BGRA→RGB: (H,W,4)→(H,W,3)
- [ ] RGB→Gray: (H,W,3)→(H,W)
- [ ] Resize: (H,W)→(84,84)
- [ ] Normalize: uint8→float32, [0,255]→[0,1]

### Stage 3: Frame Stacking ✅
- [ ] Shape: (4, 84, 84)
- [ ] Dtype: float32
- [ ] Range: [0, 1]
- [ ] Temporal order: [t-3, t-2, t-1, t]

### Stage 4: CNN Features ✅
- [ ] Input: (1, 4, 84, 84) float32
- [ ] Output: (1, 512) float32
- [ ] No NaN/Inf
- [ ] Healthy activations (30-60% active)
- [ ] L2 norm: 2-5

### Stage 5: State Vector ✅
- [ ] Shape: (535,) = 512 + 3 + 20
- [ ] Dtype: float32
- [ ] No NaN/Inf
- [ ] Component concatenation correct

### Stage 6: TD3 Actor ✅
- [ ] Input: (1, 535) float32
- [ ] Output: (1, 2) float32
- [ ] Range: exactly [-1, 1] (Tanh)
- [ ] No NaN/Inf

### Stage 7: CARLA Control ✅
- [ ] Steering: [-1, 1]
- [ ] Throttle: [0, 1]
- [ ] Brake: [0, 1]
- [ ] Mutual exclusivity: throttle XOR brake

---

## 📚 **References**

1. **CARLA 0.9.16 Documentation**
   - Sensor Reference: https://carla.readthedocs.io/en/latest/ref_sensors/
   - RGB Camera: BGRA 32-bit pixel format specification
   - Python API: https://carla.readthedocs.io/en/latest/python_api/

2. **TD3 Algorithm (Fujimoto et al. 2018)**
   - Paper: "Addressing Function Approximation Error in Actor-Critic Methods"
   - OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/td3.html
   - Actor-Critic architecture with 2×256 hidden layers

3. **Deep Q-Network (Mnih et al. 2015)**
   - NatureCNN architecture for visual RL
   - Frame stacking rationale (temporal information)

4. **Stable-Baselines3**
   - TD3 implementation reference
   - https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

---

## 🔧 **Troubleshooting**

### Issue: Shape Mismatch
**Symptom**: `RuntimeError: size mismatch, expected (1,4,84,84), got (4,84,84)`  
**Solution**: Add batch dimension: `tensor.unsqueeze(0)`

### Issue: NaN in CNN Output
**Symptom**: `nan` values in CNN features  
**Possible Causes**:
1. Input not normalized (values > 1.0 can cause exploding activations)
2. Learning rate too high (gradient explosion)
3. Batch normalization issues (mean/std divergence)

**Solution**: Check input normalization, reduce LR, inspect gradients

### Issue: Actions Outside [-1,1]
**Symptom**: `ValueError: CARLA control requires values in [0,1]`  
**Cause**: Actor output should be Tanh-bounded, but might have gradient issues  
**Solution**: Verify Tanh activation, clip actions: `np.clip(action, -1, 1)`

---

**End of Data Format Validation Guide**
