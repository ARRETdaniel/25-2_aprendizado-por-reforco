# Run 4 Debug Analysis - Critical Issues Todo List

**Date**: 2025-11-21  
**Debug Log**: `run-1validation_5k_post_all_fixes_20251120_170736.log` (433,508 lines)  
**Issues**: Two critical runtime failures preventing training  

---

## 📋 Executive Summary

### Issue #1: Actor Network CNN Gradient Explosion
- **Symptom**: Agent outputs constant full-right steering (+1.0) regardless of state
- **Timeline**: Starts around line 61,821 (before timestep 3,001)
- **Evidence**: Actor CNN features explode from [-6, 6] to [-31360, 30689] while Critic remains normal
- **Impact**: Agent learns saturated policy despite positive rewards

### Issue #2: CARLA Vehicle State Corruption Post-EVAL
- **Symptom**: After EVAL completes, vehicle freezes (velocity=0, reward=+0.767, image frozen)
- **Timeline**: Immediately after line 328696 (`[EVAL] Closing evaluation environment...`)
- **Evidence**: Applied steering = `-973852377088`, hand_brake=True, reverse=True, gear=5649815
- **Impact**: Training loop broken, connection with CARLA lost

---

## 🔍 Issue #1: Actor Network CNN Gradient Explosion

### Root Cause Analysis

**Evidence from Log**:
```
Line 61821 (FIRST EXPLOSION):
   Mode: ACTOR
   Gradient: ENABLED
   Image features: Range [-31360.885, 30689.830]
   Mean: 1836.773, Std: 20397.355
   L2 norm: 454370.875

Line 61800 (BEFORE EXPLOSION):
   Mode: CRITIC
   Gradient: ENABLED  
   Image features: Range [-5.481, 5.613]
   L2 norm: 33.842
   Status: NORMAL

Pattern: Actor CNN explodes while Critic CNN remains stable
```

**Paradoxical Observation**:
- Line 61850: Reward = +18.9692 (HIGH POSITIVE)
- Line 61850: Action = steering=+1.0000 (SATURATED)
- **Contradiction**: Good reward but extreme action suggests learned bad policy

### Documentation-Backed Hypotheses

#### Hypothesis 1.1: Actor Loss Computation Issue ⭐ **PRIMARY**
**Theory**: Actor loss `-Q(s, μ(s))` may be exploding gradients backward through CNN

**Official TD3 Paper Reference**:
- Paper: "Addressing Function Approximation Error in Actor-Critic Methods" (TD3/TD3.py)
- Actor Loss: `actor_loss = -self.critic_1(state, self.actor(state)).mean()`
- **Key Insight**: Actor gradients flow backward through **both** actor and critic networks

**Files to Investigate**:
```
src/agents/td3_agent.py:
  - Line ~900-950: Actor loss computation
  - Line ~920: self.actor_optimizer.step()
  - Check gradient clipping BEFORE optimizer.step()

TD3/TD3.py:
  - Line 120-124: Reference actor update implementation
  - Verify delayed policy updates (self.it % policy_freq == 0)
```

**Action Items**:
- [ ] **Task 1.1.1**: Read `TD3/TD3.py` lines 115-130 for reference actor update
- [ ] **Task 1.1.2**: Compare with `src/agents/td3_agent.py` lines 900-950
- [ ] **Task 1.1.3**: Verify gradient clipping is applied BEFORE `actor_optimizer.step()`
- [ ] **Task 1.1.4**: Check if critic gradients are detached when computing actor loss
- [ ] **Task 1.1.5**: Add gradient norm logging BEFORE and AFTER clipping for actor

**Validation Criteria**:
- Actor CNN grad norm should stay < 10.0 (based on Critic's 33.8 norm)
- No explosion in actor features after 10K training steps

---

#### Hypothesis 1.2: Actor Learning Rate Too High
**Theory**: LR=3e-4 may be too high for visual features

**Stable-Baselines3 Reference**:
- File: `e2e/stable-baselines3/stable_baselines3/td3/td3.py`
- Default LR: 3e-4 for MLP policies, but **1e-4** for CNN policies

**Action Items**:
- [ ] **Task 1.2.1**: Search Stable-Baselines3 TD3 implementation for learning rate configurations
- [ ] **Task 1.2.2**: Compare with `src/agents/td3_agent.py` line ~786 (actor_optimizer)
- [ ] **Task 1.2.3**: Test with reduced actor LR (1e-4 or 1e-5)

**Validation Criteria**:
- Actor CNN features should stabilize within [-10, 10] range
- No saturation in actions after 5K steps

---

#### Hypothesis 1.3: Missing Batch Normalization
**Theory**: CNN lacks normalization layers, causing feature distribution drift

**PyTorch Best Practice** (from contextual research):
- ResNet/MobileNet use BatchNorm after each Conv layer
- Without normalization, features can explode exponentially

**Action Items**:
- [ ] **Task 1.3.1**: Read `src/agents/td3_agent.py` lines ~200-300 (CNN architecture)
- [ ] **Task 1.3.2**: Verify presence of `nn.BatchNorm2d` layers
- [ ] **Task 1.3.3**: If missing, add BatchNorm after each Conv layer
- [ ] **Task 1.3.4**: Ensure `.eval()` mode during inference to freeze BN stats

**Validation Criteria**:
- Feature mean/std should remain stable (std < 5.0) during training
- No feature explosion after 10K steps

---

#### Hypothesis 1.4: Reward Signal Not Backpropagating
**Theory**: Negative rewards (-46 from lane invasion) not updating actor

**Evidence from Log**:
- Line 52929-54769: 30+ instances of full-right steering
- All episodes end with lane invasion (-46 reward)
- Actor continues saturating despite negative feedback

**Action Items**:
- [ ] **Task 1.4.1**: Read `scripts/train_td3.py` lines ~800-850 (replay buffer storage)
- [ ] **Task 1.4.2**: Verify negative rewards are stored correctly in buffer
- [ ] **Task 1.4.3**: Log sampled batch statistics during training (reward distribution)
- [ ] **Task 1.4.4**: Check if lane invasion episodes reach `done=True` correctly

**Validation Criteria**:
- Replay buffer should contain mix of positive and negative rewards
- Actor loss should increase when sampling negative reward transitions

---

### Proposed Fix Implementation Order

**Phase 1: Gradient Diagnostics** (1-2 hours)
1. Add detailed gradient logging before/after clipping
2. Verify gradient flow through critic when computing actor loss
3. Check if gradient clipping is applied to actor CNN

**Phase 2: Architecture Validation** (2-3 hours)
1. Compare actor network architecture with Stable-Baselines3 TD3
2. Add BatchNorm layers if missing
3. Reduce actor learning rate to 1e-4

**Phase 3: Reward Signal Verification** (1 hour)
1. Log replay buffer reward distribution
2. Verify negative rewards are sampled during training
3. Ensure lane invasion episodes terminate correctly

---

## 🔍 Issue #2: CARLA Vehicle State Corruption Post-EVAL

### Root Cause Analysis

**Evidence from Log**:
```
Line 328696: [EVAL] Closing evaluation environment...
Line 328716: [EVAL] Mean Reward: 64.19 | Success Rate: 0.0%
Line 328742: Applied Control: throttle=0.0000, brake=0.0000, steer=-973852377088.0000
Line 328742: Hand Brake: True, Reverse: True, Gear: 5649815
```

**CARLA API Documentation**: `carla.VehicleControl`
- **Valid Ranges**:
  - `throttle`: [0.0, 1.0]
  - `brake`: [0.0, 1.0]
  - `steer`: [-1.0, 1.0]
  - `hand_brake`: bool (True/False)
  - `gear`: int (1-6 for forward, -1 for reverse)

**Critical Finding**: Vehicle's `apply_control()` is receiving/returning corrupted values

### Documentation-Backed Hypotheses

#### Hypothesis 2.1: EVAL Environment Not Properly Isolated ⭐ **PRIMARY**
**Theory**: EVAL creates separate environment but shares client/vehicle reference

**CARLA 0.9.16 Documentation Reference**:
```
Client.load_world(map_name, reset_settings=True):
  - "All actors present in the current world will be destroyed"
  - "Traffic manager instances will stay alive"
  
World.get_actor(actor_id):
  - "Returns None if actor_id does not exist"
```

**Evidence from Log**:
```
Line 316115: [EVAL] Creating temporary evaluation environment (TM port 8050)...
Line 328696: [EVAL] Closing evaluation environment...
```

**Hypothesis**: After EVAL env is destroyed, main env's vehicle reference becomes **stale** (points to destroyed actor)

**Action Items**:
- [ ] **Task 2.1.1**: Fetch CARLA docs on `carla.Actor.is_alive` and `carla.Actor.is_active`
- [ ] **Task 2.1.2**: Read `scripts/train_td3.py` EVAL loop implementation
- [ ] **Task 2.1.3**: Verify EVAL env creates **separate** CARLA client (not shared)
- [ ] **Task 2.1.4**: Check if main env vehicle reference is validated after EVAL
- [ ] **Task 2.1.5**: Add `vehicle.is_alive` check before every `apply_control()`

**Validation Criteria**:
- After EVAL, `vehicle.is_alive` should return True
- `vehicle.get_control()` should return valid VehicleControl object
- No corrupted steering values in log

---

#### Hypothesis 2.2: Traffic Manager Port Conflict
**Theory**: EVAL TM (port 8050) not cleaned up, conflicts with main TM (port 8000)

**CARLA Traffic Manager Documentation**:
```
Client.get_trafficmanager(client_connection=8000):
  - "Returns an instance of the traffic manager related to the specified port"
  - "If it does not exist, this will be created"

TrafficManager.shut_down():
  - "Shuts down the traffic manager"
```

**Evidence from Log**:
- Main TM: Default port 8000 (assumed)
- EVAL TM: Port 8050 (explicit)

**Action Items**:
- [ ] **Task 2.2.1**: Fetch CARLA docs on `TrafficManager` lifecycle
- [ ] **Task 2.2.2**: Read `src/environment/carla_env.py` TM initialization
- [ ] **Task 2.2.3**: Verify EVAL env calls `tm.shut_down()` before closing
- [ ] **Task 2.2.4**: Add explicit TM cleanup in EVAL finally block

**Validation Criteria**:
- After EVAL, only one TM instance should exist (port 8000)
- No TM errors in CARLA server logs
- Vehicle control remains responsive

---

#### Hypothesis 2.3: Sensor Data Pipeline Not Re-initialized
**Theory**: Camera sensor queue not flushed after EVAL, frozen frame persists

**CARLA Sensor Documentation**:
```
Sensor.stop():
  - "Commands the sensor to stop listening for data"

Sensor.listen(callback):
  - "The function the sensor will be calling to every time a new measurement is received"
```

**Evidence from Log**:
- User report: "image passed to the Windows...is frezzed after the EVAL finishes"

**Action Items**:
- [ ] **Task 2.3.1**: Fetch CARLA docs on camera sensor lifecycle
- [ ] **Task 2.3.2**: Read `src/environment/sensors.py` camera initialization
- [ ] **Task 2.3.3**: Verify main env sensors are re-initialized after EVAL
- [ ] **Task 2.3.4**: Add sensor `stop()` + `listen()` cycle after EVAL

**Validation Criteria**:
- Camera image timestamp should increment after EVAL
- No frozen frames in debug log
- Image features should change frame-to-frame

---

#### Hypothesis 2.4: World Tick Synchronization Lost
**Theory**: EVAL runs in async mode, main env loses sync after EVAL

**CARLA Synchronous Mode Documentation**:
```
WorldSettings.synchronous_mode = True:
  - "Server will wait for a client tick in order to move forward"

World.tick():
  - "Makes the client wait for a server tick"
```

**Action Items**:
- [ ] **Task 2.4.1**: Fetch CARLA docs on synchronous mode best practices
- [ ] **Task 2.4.2**: Read `scripts/train_td3.py` EVAL environment settings
- [ ] **Task 2.4.3**: Verify main env world settings are restored after EVAL
- [ ] **Task 2.4.4**: Add explicit `world.apply_settings(original_settings)` after EVAL

**Validation Criteria**:
- World tick delta should remain constant (0.05s) after EVAL
- No desync warnings in CARLA server logs

---

### Proposed Fix Implementation Order

**Phase 1: Environment Isolation** (2-3 hours)
1. Verify EVAL creates separate CARLA client
2. Add vehicle `is_alive` validation after EVAL
3. Test with simple post-EVAL control command

**Phase 2: Resource Cleanup** (1-2 hours)
1. Add explicit Traffic Manager shutdown in EVAL
2. Verify sensor pipeline re-initialization
3. Test sensor data flow after EVAL

**Phase 3: Synchronization Validation** (1 hour)
1. Check world settings before/after EVAL
2. Restore original settings explicitly
3. Validate tick synchronization

---

## 🔄 Comparison with Reference Implementations

### Stable-Baselines3 TD3 Evaluation Pattern

**Expected Pattern** (from `e2e/stable-baselines3/stable_baselines3/td3/td3.py`):
```python
def evaluate_policy(model, env, n_eval_episodes=10):
    """
    Evaluation should use SAME env or create separate env with:
    1. Same settings as training env
    2. No sharing of actors/sensors
    3. Deterministic actions (no exploration noise)
    """
    eval_env = make_env()  # Separate environment
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
    eval_env.close()  # CRITICAL: Clean up
    # Main env continues training without interruption
```

**Action Items**:
- [ ] **Task 3.1**: Read full Stable-Baselines3 evaluation implementation
- [ ] **Task 3.2**: Compare with current `scripts/train_td3.py` EVAL loop
- [ ] **Task 3.3**: Identify missing cleanup steps
- [ ] **Task 3.4**: Implement proper environment isolation

---

## 📝 Immediate Next Steps (Priority Order)

### Step 1: Actor Gradient Diagnostics (HIGH PRIORITY)
**Duration**: 1-2 hours  
**Owner**: AI/Developer  

1. ✅ Read `TD3/TD3.py` actor update (lines 115-130)
2. ✅ Compare with `src/agents/td3_agent.py` actor training (lines 900-950)
3. ✅ Add gradient logging BEFORE `actor_optimizer.step()`
4. ✅ Run 1K step debug training
5. ✅ Analyze gradient magnitudes and identify explosion point

**Success Criteria**:
- Identify exact line where actor gradients explode
- Understand gradient flow path through networks

---

### Step 2: EVAL Environment Isolation Fix (HIGH PRIORITY)
**Duration**: 2-3 hours  
**Owner**: AI/Developer  

1. ✅ Read CARLA docs on `carla.Client` and `carla.World`
2. ✅ Read `scripts/train_td3.py` EVAL environment creation
3. ✅ Implement separate client for EVAL (not shared)
4. ✅ Add vehicle `is_alive` validation before every control
5. ✅ Test post-EVAL training continuation

**Success Criteria**:
- Training continues smoothly after EVAL
- No corrupted control values in log
- Vehicle velocity > 0 after EVAL

---

### Step 3: Comprehensive Logging Enhancement (MEDIUM PRIORITY)
**Duration**: 1 hour  
**Owner**: AI/Developer  

1. ✅ Add actor/critic gradient norms to TensorBoard (BEFORE and AFTER clipping)
2. ✅ Log replay buffer reward distribution every 1000 steps
3. ✅ Log EVAL environment lifecycle events (create, close, restore)
4. ✅ Add vehicle state validation checks (is_alive, control validity)

**Success Criteria**:
- All critical metrics visible in TensorBoard
- Easy to diagnose future issues from logs

---

## 🧪 Validation Plan

### Test 1: Actor Gradient Fix Validation
**Objective**: Verify actor CNN features remain stable

**Procedure**:
1. Apply gradient clipping fix
2. Run 5K step training with --debug
3. Monitor actor CNN feature range every 100 steps

**Expected Results**:
- Actor CNN features stay within [-10, 10]
- No steering saturation after 2K steps
- Positive and negative rewards both present in episodes

**Acceptance Criteria**:
- ✅ Actor features stable for 5K steps
- ✅ No full-right steering saturation
- ✅ Agent learns from lane invasion penalties

---

### Test 2: Post-EVAL Training Continuation
**Objective**: Verify training loop survives EVAL

**Procedure**:
1. Apply EVAL environment isolation fix
2. Run training through first EVAL (timestep 3,001)
3. Monitor vehicle state and control validity

**Expected Results**:
- EVAL completes successfully
- Vehicle velocity > 0 after EVAL
- Training continues for 1000 more steps

**Acceptance Criteria**:
- ✅ No corrupted control values
- ✅ Vehicle remains responsive
- ✅ Reward updates normally post-EVAL

---

## 📚 Official Documentation References

### CARLA 0.9.16 Core References
1. **Vehicle Control**: https://carla.readthedocs.io/en/latest/python_api/#carla.VehicleControl
2. **Client/World**: https://carla.readthedocs.io/en/latest/python_api/#carla.Client
3. **Traffic Manager**: https://carla.readthedocs.io/en/latest/python_api/#carla.TrafficManager
4. **Sensor Lifecycle**: https://carla.readthedocs.io/en/latest/python_api/#carla.Sensor
5. **Synchronous Mode**: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/

### TD3 Algorithm References
1. **Original Paper**: `contextual/Addressing Function Approximation Error in Actor-Critic Methods.tex`
2. **Reference Implementation**: `TD3/TD3.py`
3. **Stable-Baselines3**: `e2e/stable-baselines3/stable_baselines3/td3/td3.py`

### Gymnasium Environment API
1. **Environment Lifecycle**: https://gymnasium.farama.org/api/env/
2. **Reset/Close Spec**: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset

---

## ✅ Success Criteria Summary

**Issue #1 RESOLVED When**:
- [ ] Actor CNN features remain stable (< 10.0 L2 norm)
- [ ] No steering saturation for 10K steps
- [ ] Agent responds to negative rewards (reduces lane invasions)
- [ ] Reward distribution shows mix of positive/negative

**Issue #2 RESOLVED When**:
- [ ] Training continues smoothly after EVAL
- [ ] Vehicle control values remain valid (steer ∈ [-1, 1])
- [ ] Image feed updates normally post-EVAL
- [ ] No CARLA connection errors after EVAL

**Overall SUCCESS**:
- [ ] Complete 5K step training run with EVAL at 3K
- [ ] All metrics logged correctly to TensorBoard
- [ ] No critical errors in final 433K-line debug log
- [ ] Agent shows learning (reward trend upward)

---

## 🔄 Iteration Protocol

After each fix attempt:
1. ✅ Rebuild Docker image with changes
2. ✅ Run 5K step debug training (--debug flag)
3. ✅ Analyze new log file (search for explosion/corruption patterns)
4. ✅ Update this document with findings
5. ✅ Proceed to next hypothesis if issue persists

---

**Last Updated**: 2025-11-21  
**Status**: ANALYSIS COMPLETE - READY FOR FIX IMPLEMENTATION  
**Next Action**: Implement Step 1 (Actor Gradient Diagnostics)
