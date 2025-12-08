# CNN Implementation Systematic Diagnosis
## Investigation of Hard-Right-Turn Behavior During TD3 Learning Phase

**Date:** December 1, 2025  
**Context:** TD3 agent producing extreme actions (steer=0.6-0.8, throttle=0.9-1.0) when learning phase starts  
**SimpleTD3 Validation:** ✅ CONVERGED on Pendulum-v1 (-1224 → -120 reward) - Core TD3 algorithm is CORRECT  

---

## 🎯 Key Finding from SimpleTD3 Validation

**The core TD3 algorithm works correctly** - SimpleTD3 converged successfully on Pendulum-v1:
- Initial reward: -1224.40 (random policy)
- Final reward: -119.89 (near-optimal)
- Convergence time: ~10K steps
- **Conclusion:** Issues are NOT in TD3 mechanics (twin critics, delayed updates, target smoothing)

Therefore, the hard-right-turn problem must be in **CARLA-specific components**:
1. **CNN Feature Extraction** ← PRIMARY SUSPECT
2. State preprocessing/concatenation
3. Reward function design
4. Action mapping to CARLA controls

---

## 📚 Official Documentation Review

### Stable-Baselines3 CNN Best Practices

From https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html:

#### ✅ What SB3 Does (Industry Standard):

1. **Feature Extractor Architecture:**
   ```
   For image observations: "Nature CNN" is used for feature extraction
   - Conv1: 32 filters, 8×8 kernel, stride 4
   - Conv2: 64 filters, 4×4 kernel, stride 2
   - Conv3: 64 filters, 3×3 kernel, stride 1
   - Flatten → Linear(3136, features_dim)
   ```

2. **Separate CNNs for Actor and Critic (OFF-POLICY algorithms):**
   > "Off-policy algorithms (TD3, DDPG, SAC, …) have separate feature extractors: 
   > one for the actor and one for the critic, since the best performance is obtained 
   > with this configuration."
   
   **CRITICAL:** TD3/DDPG should use **SEPARATE** CNN extractors for actor and critic!

3. **Image Normalization:**
   > "All observations are first pre-processed (e.g. images are normalized, discrete 
   > obs are converted to one-hot vectors, …) before being fed to the features extractor."

4. **Default TD3 Network Architecture:**
   > "For 1D observation space: [400, 300] units for TD3/DDPG (values are taken from 
   > the original TD3 paper)"

---

## 🔍 Current Implementation Analysis

### Our CNN Architecture (cnn_extractor.py)

```python
class NatureCNN(nn.Module):
    Architecture:
        Input:   (batch, 4, 84, 84) - normalized to [-1, 1]
        Conv1:   32 filters, 8×8 kernel, stride 4 → (batch, 32, 20, 20)
        LN1:     LayerNorm([32, 20, 20])
        LeakyReLU(0.01)
        
        Conv2:   64 filters, 4×4 kernel, stride 2 → (batch, 64, 9, 9)
        LN2:     LayerNorm([64, 9, 9])
        LeakyReLU(0.01)
        
        Conv3:   64 filters, 3×3 kernel, stride 1 → (batch, 64, 7, 7)
        LN3:     LayerNorm([64, 7, 7])
        LeakyReLU(0.01)
        
        Flatten: (batch, 3136)
        FC:      Linear(3136, 512)
        LN4:     LayerNorm(512)
        LeakyReLU(0.01)
        
        Output:  (batch, 512) features
```

#### ✅ What We Got RIGHT:
1. **Architecture matches Nature DQN:** Conv layers follow exact same spec as SB3
2. **Layer Normalization:** Prevents feature explosion (documented: 7.36×10¹² → 10-100)
3. **Leaky ReLU:** Preserves negative values from zero-centered normalization
4. **Weight initialization:** Kaiming for Leaky ReLU activations
5. **Input normalization:** Zero-centered [-1, 1] (modern best practice)
6. **Debug instrumentation:** Comprehensive logging for diagnostics

#### ⚠️ POTENTIAL ISSUES:

1. **Separate CNN for Actor vs Critic?**
   - **SB3 Recommendation:** OFF-policy algorithms should use separate CNNs
   - **Our Implementation:** UNKNOWN - need to check td3_agent.py
   - **Impact:** Gradient interference between actor and critic updates

2. **LayerNorm vs BatchNorm:**
   - **Our Choice:** LayerNorm (advantages for RL: batch-size independent, deterministic)
   - **Standard Practice:** Most DQN implementations use NO normalization or BatchNorm
   - **Potential Issue:** LayerNorm BEFORE activation might suppress features differently
   
3. **Feature Dimension:**
   - **Our Output:** 512-dim
   - **Standard NatureCNN:** Usually 256-dim or directly use flattened 3136-dim
   - **Impact:** Extra FC layer adds parameters, potential bottleneck

4. **Activation Pattern:**
   - **Our Pattern:** Conv → LayerNorm → LeakyReLU
   - **Standard Pattern:** Conv → ReLU (no normalization)
   - **Potential Issue:** Normalization before activation changes learned representations

---

## 🔬 Diagnostic Questions

### Q1: Are We Using Separate CNNs for Actor and Critic?
**Status:** ❓ NEED TO CHECK `td3_agent.py`

**From td3_agent.py initialization:**
```python
self.actor_cnn = actor_cnn if actor_cnn else NatureCNN(...)
self.critic_cnn = critic_cnn if critic_cnn else NatureCNN(...)
```

**Expected:** Two independent CNN instances (actor_cnn ≠ critic_cnn)  
**If shared:** Gradient conflict → unstable learning → extreme actions

---

### Q2: Is Image Preprocessing Consistent?
**Status:** ❓ NEED TO CHECK `sensors.py`

**Expected preprocessing pipeline:**
1. CARLA RGB (800×600×3) → Grayscale (800×600×1)
2. Resize → (84×84×1)
3. Normalize [0, 255] → [-1, 1]: `(pixel / 255.0) * 2 - 1`
4. Stack 4 frames → (4, 84, 84)
5. Dtype: `float32`, Device: `cuda`

**Critical checks:**
- [ ] Normalization range matches CNN expectation [-1, 1]
- [ ] Frame stacking maintains temporal order (oldest → newest)
- [ ] No NaN/Inf values in preprocessed images
- [ ] Data type consistency (float32 throughout)

---

### Q3: Is State Concatenation Correct?
**Status:** ❓ NEED TO CHECK `td3_agent.py`

**Expected state vector (535-dim):**
```
CNN features (512) + Kinematic (3) + Waypoints (20) = 535
```

**Critical checks:**
- [ ] CNN features normalized/scaled appropriately
- [ ] Kinematic features normalized (velocity, deviation, heading)
- [ ] Waypoint features in vehicle-local frame
- [ ] No dimension mismatch errors
- [ ] Gradient flows through concatenation properly

---

### Q4: Is Action Mapping Correct?
**Status:** ❓ NEED TO CHECK CARLA API docs + our mapping code

**Actor output:** `[-1, 1]` for both steering and throttle/brake  
**CARLA expects:**
```python
VehicleControl(
    steer=-1 to 1,      # OK - matches actor output
    throttle=0 to 1,    # Need mapping: actor [-1,1] → throttle [0,1]
    brake=0 to 1,       # Need mapping: actor [-1,1] → brake [0,1]
)
```

**Expected mapping logic:**
```python
if throttle_brake_output > 0:
    throttle = throttle_brake_output  # [0, 1]
    brake = 0.0
else:
    throttle = 0.0
    brake = -throttle_brake_output  # [0, 1]
```

**Potential bug:** If mapping is wrong, actor learns to max out throttle to move forward

---

## 🧪 Systematic Investigation Plan

### Phase 1: Verify CNN Gradient Flow (IN PROGRESS)

**Already instrumented:** cnn_extractor.py has debug logging  
**TODO:**
1. ✅ Read current CNN implementation
2. ✅ Verify architecture matches SB3/Nature DQN
3. ⏳ Check if separate CNNs used for actor/critic
4. ⏳ Add gradient norm tracking
5. ⏳ Add activation distribution histograms

### Phase 2: Verify State Pipeline

**TODO:**
1. ⏳ Read sensors.py image preprocessing
2. ⏳ Verify normalization [-1, 1]
3. ⏳ Check frame stacking order
4. ⏳ Read td3_agent.py state concatenation
5. ⏳ Verify kinematic feature normalization

### Phase 3: Verify Action Mapping

**TODO:**
1. ⏳ Fetch CARLA VehicleControl API docs
2. ⏳ Read our action mapping code
3. ⏳ Verify throttle/brake conversion logic
4. ⏳ Check steering mapping

### Phase 4: Debugging Session

**TODO:**
1. ⏳ Enable all debug prints in CNN
2. ⏳ Run training for 1K steps
3. ⏳ Collect logs:
   - CNN input statistics (per batch)
   - CNN output statistics (per batch)
   - Actor output actions (per step)
   - Mapped CARLA controls (per step)
4. ⏳ Identify anomalies:
   - Feature collapse (all outputs similar)
   - Gradient explosion/vanishing
   - Action saturation (always max)

---

## 🚩 RED FLAGS to Look For

### CNN Issues:
- ❌ Features collapse to near-zero or constant values
- ❌ L2 norm explosion (>1000)
- ❌ Activation sparsity (>90% zeros after ReLU)
- ❌ NaN or Inf in any layer
- ❌ Shared CNN between actor and critic (gradient conflict)

### State Issues:
- ❌ Images not normalized to [-1, 1]
- ❌ Kinematic features not scaled (raw values like 30 km/h vs 0.5 rad)
- ❌ Dimension mismatch in concatenation
- ❌ NaN or Inf in state vector

### Action Issues:
- ❌ Actor always outputs extreme values (tanh saturation)
- ❌ Wrong mapping: throttle/brake inversion
- ❌ Action clipping before CARLA mapping

### Reward Issues:
- ❌ Reward only incentivizes speed (not steering)
- ❌ No penalty for extreme steering
- ❌ Collision penalty too small

---

## 📊 Expected vs Actual Behavior

### Exploration Phase (Steps 0-1000):
**Expected:** Random actions, uniform distribution [-1, 1]  
**Actual:** ✅ CORRECT - actions distributed across range

### Learning Phase (Steps 1000+):
**Expected:** Gradual policy improvement, balanced actions  
**Actual:** ❌ BROKEN - immediate switch to (steer=0.8, throttle=1.0)

**This suggests:**
- Actor learned a STRONG policy quickly (too quickly = suspicious)
- Policy is DEGENERATE (one fixed action for all states)
- Possible causes:
  1. CNN features are NOT discriminative (all states look the same)
  2. Reward function ONLY rewards speed (ignores steering)
  3. Q-values are WRONG (overestimation despite twin critics)

---

## 🎯 Next Steps (Priority Order)

1. **HIGH PRIORITY:** Check if separate CNNs for actor/critic
   - File: td3_agent.py
   - Look for: `self.actor_cnn` vs `self.critic_cnn`
   - Expected: Two independent instances

2. **HIGH PRIORITY:** Verify image preprocessing
   - File: sensors.py
   - Check: Normalization range, dtype, NaN/Inf handling

3. **MEDIUM PRIORITY:** Verify state concatenation
   - File: td3_agent.py
   - Check: Dimension matching, feature scaling

4. **MEDIUM PRIORITY:** Verify action mapping
   - Fetch CARLA docs
   - Check: throttle/brake conversion logic

5. **LOW PRIORITY:** Run debug training session
   - Enable all debug logs
   - Collect 1K steps of data
   - Analyze for anomalies

---

## 📖 References

1. **Stable-Baselines3 Custom Policies:**  
   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

2. **Nature DQN (Mnih et al., 2015):**  
   "Human-level control through deep reinforcement learning"

3. **TD3 Paper (Fujimoto et al., 2018):**  
   "Addressing Function Approximation Error in Actor-Critic Methods"

4. **SimpleTD3 Validation Results:**  
   `/L4/ps4-dev.ipynb` - Pendulum-v1 convergence confirms core TD3 is correct

---

## ✅ Validation Checkpoints

- [x] SimpleTD3 converges on Pendulum-v1 → Core TD3 is CORRECT
- [x] CNN architecture matches Nature DQN spec
- [x] Layer Normalization prevents feature explosion
- [ ] Separate CNNs for actor and critic (SB3 recommendation)
- [ ] Image preprocessing matches CNN expectations
- [ ] State concatenation preserves gradient flow
- [ ] Action mapping matches CARLA API

---

**CONCLUSION SO FAR:**

Based on SimpleTD3 validation and documentation review, the hard-right-turn behavior is **NOT caused by the core TD3 algorithm**. The issue must be in:

1. **CNN feature extraction** (most likely - needs gradient flow analysis)
2. **State preprocessing** (possible - normalization mismatch)
3. **Action mapping** (possible - throttle/brake conversion)
4. **Reward function** (less likely - but worth checking)

**Next action:** Continue systematic investigation of each component with focus on CNN gradient flow and actor/critic separation.
