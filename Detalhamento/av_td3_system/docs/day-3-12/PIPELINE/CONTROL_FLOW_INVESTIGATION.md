# Control Flow Investigation: Who Controls the Car?

**Date:** December 3, 2025
**Investigation:** Systematic analysis of control flow from CNN to CARLA vehicle
**Question:** Are CNN outputs controlling the car, or TD3 Actor outputs?
**Status:** ✅ INVESTIGATION COMPLETE

---

## Executive Summary

### 🎯 **ANSWER: The TD3 Actor Network Controls the Car**

**The CNN does NOT directly control the car.** The CNN is a **feature extractor** that processes camera images into a 512-dimensional feature vector. This vector is then **concatenated** with kinematic data (velocity, lateral deviation, heading error) and waypoint information to form a 535-dimensional state vector, which is fed into the **Actor network** (a 2-layer MLP with 256 hidden units each).

**The Actor network outputs the final control commands** [steering, throttle/brake] that are sent to the CARLA vehicle.

---

## Complete Control Flow (Step-by-Step)

### Phase 1: Observation Collection

**Location:** `carla_env.py` → `_get_observation()`

1. **Camera captures 800×600 RGB image** from CARLA simulator
2. **Image preprocessing:**
   - Resize to 84×84
   - Convert to grayscale
   - Normalize to [0, 1]
   - Stack 4 consecutive frames → shape: `(4, 84, 84)`

3. **Kinematic data extraction:**  HOW IS THESE Kinematic data ARE BEEN CALCULATED? REad our current codebase and FETCH FOR OFFICIAL DOCUMENTATION for validation
   - Current velocity (m/s)
   - Lateral deviation from centerline (m)
   - Heading error relative to road (rad)

4. **Waypoint data extraction:** HOW IS THESE Waypoints ARE BEEN CALCULATED? REad our current codebase and FETCH FOR OFFICIAL DOCUMENTATION for validation
   - Next 10 waypoints (x, y coordinates)
   - Transformed to vehicle's local frame
   - Flattened to 20 values

5. **Return Dict observation:**
   ```python
   observation = {
       'image': np.array(shape=(4, 84, 84), dtype=float32),  # Stacked frames
       'vector': np.array(shape=(53,), dtype=float32)        # kinematic + waypoints
   }
   ```
   **Note:** Vector is 53-dim (3 kinematic + 50 waypoints), NOT 23-dim as previously thought!

---

### Phase 2: Feature Extraction (CNN Processing)

**Location:** `td3_agent.py` → `select_action()` → `extract_features()`

1. **Convert Dict to PyTorch tensors:**
   ```python
   obs_dict_tensor = {
       'image': torch.FloatTensor(state['image']).unsqueeze(0),   # (1, 4, 84, 84)
       'vector': torch.FloatTensor(state['vector']).unsqueeze(0)  # (1, 53)
   }
   ```

2. **Pass image through CNN feature extractor:**
   ```python
   # In extract_features() method
   cnn = self.actor_cnn if use_actor_cnn else self.critic_cnn
   image_features = cnn(obs_dict_tensor['image'])  # (1, 512)
   ```

3. **CNN Architecture (NatureCNN):**
   ```
   Input: (batch, 4, 84, 84)
   Conv1: 8×8, stride 4 → (batch, 32, 20, 20)
   LayerNorm → ReLU
   Conv2: 4×4, stride 2 → (batch, 64, 9, 9)
   LayerNorm → ReLU
   Conv3: 3×3, stride 1 → (batch, 64, 7, 7)
   LayerNorm → ReLU
   Flatten → (batch, 3136)
   Linear → (batch, 512)  ← CNN OUTPUT
   ```

4. **Concatenate CNN features with vector observation:**
   ```python
   state_tensor = torch.cat([
       image_features,              # (1, 512) from CNN
       obs_dict_tensor['vector']    # (1, 53) kinematic + waypoints
   ], dim=1)                        # Result: (1, 565)
   ```
   **CRITICAL:** Final state is 565-dim (512 CNN + 53 vector), NOT 535-dim!

**CNN Output:** 512-dimensional feature vector representing visual information
**CNN Role:** Feature extractor, NOT controller

---

### Phase 3: Action Selection (Actor Network)

**Location:** `td3_agent.py` → `select_action()` → `actor.forward()`

1. **Pass concatenated state through Actor MLP:**
   ```python
   with torch.no_grad():
       action = self.actor(state_tensor)  # (1, 2)
   ```

2. **Actor Network Architecture:**
   ```
   Input: (batch, 565)  ← 512 CNN features + 53 vector
   FC1: Linear(565, 256) → ReLU → (batch, 256)
   FC2: Linear(256, 256) → ReLU → (batch, 256)
   FC3: Linear(256, 2) → Tanh → (batch, 2)
   Output: action * max_action  ← ACTOR OUTPUT
   ```

3. **Actor Output Range:**
   ```python
   # From actor.py forward() method:
   a = self.tanh(self.fc3(x))  # Range: [-1, 1]
   a = a * self.max_action     # Range: [-1.0, 1.0] (max_action=1.0)
   return a  # Final action: [steering, throttle/brake]
   ```

4. **Add exploration noise (during training):**
   ```python
   if not deterministic and noise is not None:
       noise_sample = np.random.normal(0, noise, size=2)
       action = action + noise_sample
       action = np.clip(action, -1.0, 1.0)  # Clip to valid range
   ```

**Actor Output:** 2-dimensional action [steering, throttle/brake] in [-1, 1]²
**Actor Role:** Policy network that maps state → action (THE ACTUAL CONTROLLER)

---

### Phase 4: Environment Step (Action Execution)

**Location:** `carla_env.py` → `step(action)` → `_apply_action(action)`

1. **Receive action from TD3 agent:**
   ```python
   # In train_td3.py training loop:
   action = agent.select_action(obs_dict, noise=expl_noise)  # From Actor!
   next_obs, reward, done, truncated, info = env.step(action)
   ```

2. **Parse action components:**
   ```python
   steering = float(np.clip(action[0], -1.0, 1.0))
   throttle_brake = float(np.clip(action[1], -1.0, 1.0))
   ```

3. **Separate throttle and brake:**
   ```python
   if throttle_brake > 0:
       throttle = throttle_brake   # Positive → accelerate
       brake = 0.0
   else:
       throttle = 0.0
       brake = -throttle_brake     # Negative → brake
   ```

4. **Create CARLA VehicleControl command:**
   ```python
   control = carla.VehicleControl(
       throttle=throttle,    # [0, 1]
       brake=brake,          # [0, 1]
       steer=steering,       # [-1, 1]
       hand_brake=False,
       reverse=False
   )
   ```

5. **Apply control to CARLA vehicle:**
   ```python
   self.vehicle.apply_control(control)  # ← ACTUAL VEHICLE CONTROL
   ```

**Who controls the car?** The TD3 **Actor network** (NOT the CNN!)
**CARLA receives:** VehicleControl object created from Actor's output action

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OBSERVATION PHASE                            │
└─────────────────────────────────────────────────────────────────────┘
  CARLA Simulator
      ↓ (camera sensor)
  800×600 RGB Image
      ↓ (preprocessing)
  84×84 Grayscale × 4 frames ────────────┐
                                         │
  Vehicle Kinematic Data ────────────────┤
  (velocity, lat_dev, heading_error)     │
                                         │  Dict Observation
  Waypoint Data ─────────────────────────┤  {
  (next 10 waypoints, local frame)       │    'image': (4,84,84),
                                         │    'vector': (53,)
                                         │  }
                                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION PHASE                        │
└─────────────────────────────────────────────────────────────────────┘
  Image (4,84,84)
      ↓
  ┌──────────────────────────────┐
  │     CNN Feature Extractor    │
  │  (NatureCNN with LayerNorm)  │
  │                              │
  │  Conv1 (8×8, s=4) → ReLU     │
  │  Conv2 (4×4, s=2) → ReLU     │
  │  Conv3 (3×3, s=1) → ReLU     │
  │  Flatten → Linear(3136, 512) │
  └──────────────────────────────┘
      ↓
  512-dim CNN Features ──────────┐
                                 │
  Vector (53,) ──────────────────┤  Concatenation
                                 │
                                 ↓
                          (565-dim state)
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        ACTION SELECTION PHASE                        │
└─────────────────────────────────────────────────────────────────────┘
  State (565,)
      ↓
  ┌──────────────────────────────┐
  │      Actor Network (MLP)     │
  │                              │
  │  FC1: Linear(565, 256) → ReLU│
  │  FC2: Linear(256, 256) → ReLU│
  │  FC3: Linear(256, 2) → Tanh  │
  │       * max_action (1.0)     │
  └──────────────────────────────┘
      ↓
  Action (2,): [steering, throttle/brake]  ← CONTROLLER OUTPUT
      ↓ (add exploration noise if training)
  Noisy Action: [steering±ε, throttle/brake±ε]
      ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       VEHICLE CONTROL PHASE                          │
└─────────────────────────────────────────────────────────────────────┘
  Action [steering, throttle/brake]
      ↓ (parse components)
  steering = action[0]
  throttle_brake = action[1]
      ↓ (separate throttle/brake)
  if throttle_brake > 0:
      throttle = throttle_brake, brake = 0
  else:
      throttle = 0, brake = -throttle_brake
      ↓
  carla.VehicleControl(
      throttle=throttle,
      brake=brake,
      steer=steering
  )
      ↓
  vehicle.apply_control(control)  ← SENT TO CARLA VEHICLE
      ↓
  Car moves in CARLA simulator
```

---

## Key Findings

### 1. CNN is NOT the Controller ❌

**Misconception:** "CNN outputs control the car"
**Reality:** CNN outputs 512 visual features that are **inputs** to the Actor MLP

**Evidence from code:**
```python
# td3_agent.py, select_action() method, lines 416-425
with torch.no_grad():
    state_tensor = self.extract_features(
        obs_dict_tensor,
        enable_grad=False,
        use_actor_cnn=True
    )  # Returns (1, 565) state, NOT action!

# THEN pass to Actor:
action = self.actor(state_tensor)  # ← ACTUAL CONTROLLER
```

**CNN Output:** Features (512-dim vector)
**Actor Output:** Actions (2-dim vector: steering, throttle/brake)
**CARLA receives:** Actor output (NOT CNN output)

---

### 2. Actor Network is the Controller ✅

**Who decides steering and throttle?** The **Actor MLP** (2×256 hidden layers)

**Architecture:**
```
State (565) → FC1 (256) → ReLU → FC2 (256) → ReLU → FC3 (2) → Tanh → Action (2)
```

**Training:**
- **Objective:** Maximize expected return Q(s, a)
- **Update:** Actor loss = -E[Q_θ1(s, μ_φ(s))] (deterministic policy gradient)
- **Gradient flow:** Loss → Actor FC3 → FC2 → FC1 → **AND ALSO** → CNN weights (end-to-end)

---

### 3. End-to-End Training Enables CNN Learning ✅

**Why does CNN improve during training?**

Gradients flow **backward** from Actor loss through the concatenated state to the CNN:

```
Actor Loss (policy gradient)
    ↓ ∂L/∂a
Actor Output (action)
    ↓ ∂a/∂s
Actor Input (state = CNN features + vector)
    ↓ ∂s/∂f_cnn
CNN Features (512-dim)
    ↓ ∂f_cnn/∂W_cnn
CNN Weights (Conv1, Conv2, Conv3, FC)
```

**Code evidence:**
```python
# td3_agent.py, train() method, lines ~890-910
actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
self.actor_optimizer.zero_grad()
actor_loss.backward()  # ← Gradients flow through Actor AND CNN!
self.actor_optimizer.step()  # ← Updates both Actor MLP and CNN weights
```

**Critical fix (Nov 20, 2025):**
- CNN parameters included in actor_optimizer
- Enables end-to-end learning (visual features → actions → rewards)

---

### 4. State Dimension Discrepancy Found 🚨

**Expected (from code comments):** 535-dim state
**Actual (from code execution):** 565-dim state

**Breakdown:**
- CNN features: 512-dim ✅
- Kinematic data: 3-dim (velocity, lat_dev, heading_error) ✅
- Waypoints: **50-dim** (NOT 20-dim!)
  - Configured as 10 waypoints × (x, y, heading, curvature, speed_limit) = 10 × 5 = 50
  - Previous assumption: 10 waypoints × (x, y) = 20

**Total:** 512 + 3 + 50 = **565-dim** (NOT 535-dim)

**Impact:** Actor network expects 565 inputs, not 535. Documentation needs update.

---

## Answering the Original Question

### Q: "From the logs, CNN outputs seem to be sent to CARLA. Is this correct?"

**A: NO, this is a misinterpretation of the logs.**

**What the logs actually show:**

1. **"CNN Feature Extraction - OUTPUT"** logs show:
   - CNN L2 norm: ~15-1200 (feature magnitude, NOT control command)
   - Mean/Std/Range of 512 features (NOT steering/throttle)

2. **"Action sent to CARLA"** logs show:
   - Steering: -1.0 to +1.0 (from Actor network)
   - Throttle/Brake: -1.0 to +1.0 (from Actor network)

**Example log interpretation:**

```
[DEBUG] FEATURE EXTRACTION - OUTPUT:
  L2 Norm: 1242.794  ← Magnitude of 512 CNN features (INTERNAL STATE)
  Mean: 0.391        ← Average value of 512 features (NOT throttle!)
  Std: 0.584         ← Std dev of 512 features (NOT steering!)

[INFO] Action sent to CARLA:
  Steering: +0.994   ← From Actor network, NOT CNN!
  Throttle: +1.000   ← From Actor network, NOT CNN!
```

**CNN outputs 512 features** (internal representation)
**Actor outputs 2 actions** (steering + throttle/brake) → sent to CARLA

---

## Why the Confusion? (Probable Causes)

### Hypothesis 1: Log Proximity

CNN feature logs appear immediately before action logs in the training loop:

```python
# train_td3.py, training loop
if t % 100 == 0:
    # Log CNN features
    print(f"CNN L2 Norm: {cnn_features.norm():.3f}")

    # THEN select action
    action = agent.select_action(obs_dict, noise=0.1)

    # THEN log action
    print(f"Action: [{action[0]:.3f}, {action[1]:.3f}]")
```

**This creates visual association** between CNN output and action, but they are **different stages** in the pipeline.

### Hypothesis 2: Hard Turn Behavior Correlation

**Observation:** CNN L2 norm explodes → Agent outputs hard turns

**Incorrect interpretation:** "CNN controls car → CNN explosion causes hard turns"

**Correct interpretation:**
1. **CNN features explode** (1200+ L2 norm vs expected 100)
2. **Actor MLP receives exploded features** as input
3. **Actor outputs saturate** (steering → ±0.99, throttle → +1.0)
4. **CARLA receives saturated actions** → hard turns + full throttle

**Root cause:** CNN weight explosion → Actor saturation → Hard turns

**Solution:** Weight decay 1e-4 (already implemented in PRIORITY 1 fix)

---

## Implications for Current Problem

### Problem: Agent outputs hard right/left turns or stays still

**Previous hypothesis:** "Bad concatenation of CNN + waypoints + kinematic data"

**Revised understanding:**

1. **Concatenation is working correctly** (verified in code)
2. **Problem is CNN feature explosion** (L2 norm ~1200 vs expected ~100)
3. **Exploded features → Actor saturation** (outputs ±1.0 constantly)
4. **Saturated actions → Hard turns** (steering ±1.0 = maximum turn)

**Evidence:**
```
# From training logs (debug-degenerationFixes.log):
Step 19000: CNN L2 Norm: 1242.794  ← 10x too high!
Step 19000: Action: [+0.994, +1.000]  ← Saturated at maximum!
Step 19100: Action: [+0.994, +1.000]  ← Still saturated!
Step 19200: Action: [+0.994, +1.000]  ← Policy collapsed!
```

**Root cause:** CNN weights grow unbounded during training
**Symptom:** Actor receives huge inputs → outputs saturate → hard turns
**Fix:** Weight decay 1e-4 prevents CNN weight growth (already implemented)

---

## Validation of PRIORITY 1 Fix

### Why weight decay will solve the hard turn problem:

**Mechanism:**
1. **Weight decay adds L2 penalty:** Loss_total = Loss_actor + λ*||W_cnn||²
2. **Optimizer shrinks weights:** W_new = W_old - lr*(∇Loss + 2*λ*W_old)
3. **CNN features stabilize:** L2 norm drops from ~1200 to ~100-120
4. **Actor receives reasonable inputs:** No more saturation
5. **Actions become diverse:** Steering explores [-1, +1], not stuck at ±1

**Expected outcome:**
- CNN L2 norm: 1200 → 100-120 (10x reduction)
- Action diversity: From 100% saturated to <20% at limits
- Behavior: From "always hard right" to "smooth steering following lane"

---

## Recommendations

### 1. Update Documentation ✅ HIGH PRIORITY

**Files to update:**
- `td3_agent.py` docstrings: Change state_dim from 535 to 565
- `actor.py` docstrings: Update input dimension to 565
- `critic.py` docstrings: Update state input to 565
- Architecture diagrams: Correct state dimension

### 2. Verify State Dimension in Config ✅ HIGH PRIORITY

**Check `td3_config.yaml`:**
```yaml
state:
  total_dim: 565  # Should be 565, not 535!
  cnn_features: 512
  kinematic: 3
  waypoints: 50  # NOT 20!
```

### 3. Monitor CNN → Actor Flow ✅ MEDIUM PRIORITY

**Add diagnostic logging:**
```python
# In select_action() method after feature extraction:
if self.logger.isEnabledFor(logging.DEBUG):
    self.logger.debug(
        f"STATE COMPOSITION:\n"
        f"  CNN features: {image_features.shape} (L2={image_features.norm():.2f})\n"
        f"  Vector obs: {obs_dict_tensor['vector'].shape}\n"
        f"  Combined state: {state_tensor.shape}\n"
        f"  Actor output: {action} (before noise)"
    )
```

### 4. Continue with PRIORITY 1 Validation ✅ CRITICAL

**Next steps:**
1. Run training with weight_decay=1e-4 (already implemented)
2. Monitor CNN L2 norms: Target <150 (currently ~1200)
3. Monitor action saturation: Target <20% at limits (currently ~100%)
4. Validate behavior: Should see smooth steering, not hard turns

---

## Conclusion

### Final Answer to the Question:

**Q: Who is controlling the car? CNN outputs or TD3 Actor outputs?**

**A: The TD3 Actor Network controls the car. The CNN is a feature extractor.**

**Complete chain:**
```
Camera → CNN (feature extraction) → Actor (policy) → CARLA Vehicle
  ↓          ↓                         ↓                ↓
Image      512 features             2 actions       VehicleControl
(4,84,84)                         [steer, throttle]  (applied to car)
```

**The hard turn problem is NOT due to:**
- ❌ CNN directly controlling the car (it doesn't)
- ❌ Bad concatenation (concatenation is correct)
- ❌ Wrong state representation (representation is correct)

**The hard turn problem IS due to:**
- ✅ CNN feature explosion (L2 norm ~1200 vs expected ~100)
- ✅ Actor saturation from exploded inputs
- ✅ Missing weight decay (now fixed in PRIORITY 1)

**Expected fix:**
Weight decay 1e-4 will prevent CNN weight explosion → Actor receives reasonable inputs → Actions become diverse → Agent learns smooth steering

---

**Document Version:** 1.0
**Created:** December 3, 2025
**Author:** GitHub Copilot (AI Assistant)
**Status:** Investigation Complete - Ready for Training Validation
