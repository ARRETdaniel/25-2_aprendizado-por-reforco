# EVAL Phase Solution - Executive Summary

**Date**: 2025-11-20  
**Issue**: CARLA vehicle state corruption after evaluation phase  
**Status**: ✅ **SOLUTION IDENTIFIED** - Ready for implementation

---

## 🎯 The Problem

When the evaluation phase (`evaluate()`) creates a **separate CARLA environment**, it causes catastrophic failure:

```
[EVAL] Closing evaluation environment...          ← EVAL env destroyed
Applied Control: steer=-973852377088.0000          ← CORRUPTED vehicle state
Hand Brake: True, Reverse: True, Gear: 5649815     ← Invalid values
```

**Root Cause**: CARLA uses a **singleton world architecture**. Creating a "new" environment doesn't create a new world—it connects to the SAME CARLA server with the SAME world instance. When `eval_env.close()` destroys actors, the training environment's vehicle references become **stale** (point to destroyed actors).

---

## ✅ The Solution: PHASE-BASED Architecture

**Stop creating separate evaluation environments.** Instead, follow the pattern already used for EXPLORATION and LEARNING phases:

### Current Architecture (3 phases, 1 environment):

```
EXPLORATION PHASE (t < start_timesteps):
  ├─ Random actions
  ├─ No training
  └─ Uses self.env

LEARNING PHASE (t ≥ start_timesteps):
  ├─ Policy actions + exploration noise  
  ├─ Training enabled
  └─ Uses self.env
```

### Proposed Architecture (Add 3rd phase):

```
EXPLORATION PHASE (t < start_timesteps):
  ├─ Random actions
  ├─ No training
  └─ Uses self.env

LEARNING PHASE (t ≥ start_timesteps && not eval):
  ├─ Policy actions + exploration noise
  ├─ Training enabled
  └─ Uses self.env

EVAL PHASE (t % eval_freq == 0):  ← NEW
  ├─ Policy actions (NO noise - deterministic)
  ├─ NO training
  ├─ Uses self.env  ← SAME environment!
  └─ Multiple episodes for statistics
```

**Key Change**: EVAL uses `self.env` (the training environment) with **deterministic policy** (no exploration noise), instead of creating a separate environment.

---

## 📋 Implementation Changes

### Change #1: Remove Separate Eval Environment

**Before (BROKEN)**:
```python
def evaluate(self):
    eval_env = CARLANavigationEnv(..., tm_port=8050)  # ❌ Creates new env
    # ... run eval ...
    eval_env.close()  # ❌ Destroys actors, corrupts training env
```

**After (FIXED)**:
```python
def evaluate(self):
    """Evaluate using SAME environment (self.env)"""
    # ✅ No separate environment creation!
    # ✅ No env.close() that destroys actors!
```

### Change #2: Add EVAL Phase Flag

```python
class TD3TrainingPipeline:
    def __init__(self, ...):
        self.in_eval_phase = False  # NEW: Track when in EVAL mode
```

### Change #3: Modify Training Loop

```python
for t in range(int(self.max_timesteps)):
    # ✅ EVAL phase (when t % eval_freq == 0)
    if t > 0 and t % self.eval_freq == 0:
        self.in_eval_phase = True
        eval_metrics, obs_dict = self.evaluate()  # Uses self.env
        self.in_eval_phase = False
    
    # EXPLORATION phase
    if t < self.start_timesteps:
        action = self.env.action_space.sample()
    
    # LEARNING phase
    elif not self.in_eval_phase:
        action = self.agent.select_action(obs_dict, deterministic=False)
        # Add exploration noise...
```

### Change #4: Rewrite evaluate() Method

```python
def evaluate(self):
    """Evaluate agent using TRAINING environment (no separate env)."""
    eval_rewards = []
    # ... metrics lists ...
    
    for episode in range(self.num_eval_episodes):
        obs_dict, _ = self.env.reset()  # ✅ Reset TRAINING env
        
        while not done:
            action = self.agent.select_action(obs_dict, deterministic=True)  # ✅ No noise
            next_obs_dict, reward, done, truncated, info = self.env.step(action)
            # ... collect metrics ...
        
        eval_rewards.append(episode_reward)
    
    # ✅ Reset after EVAL to start fresh training episode
    obs_dict, _ = self.env.reset()
    
    return eval_metrics, obs_dict  # Return fresh observation
```

---

## 🔍 Why This Works

### CARLA Documentation Evidence

From **CARLA 0.9.16 Python API**:

```python
Client.get_world(self)
    Returns the world object currently active in the simulation.
    Note: This world will be later used for example to load maps.
```

**Implication**: There is **ONE world per CARLA server**. Multiple `CARLANavigationEnv()` instances share the SAME world, leading to actor lifecycle conflicts.

### TD3 Paper Comparison

**TD3 Reference Code** (`TD3/main.py`):
```python
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)  # Creates INDEPENDENT environment
    # ... evaluates with deterministic policy (no noise) ...
```

**Why it works for Gym/MuJoCo**: Each `gym.make()` creates a **fully independent** simulation instance. No shared state.

**Why it FAILS for CARLA**: CARLA client connects to a **shared server** with a **singleton world**. Destroying eval actors affects training actors.

**Our Solution**: Keep TD3's deterministic evaluation but use the SAME environment (like EXPLORATION/LEARNING phases).

---

## ✅ Benefits of This Approach

1. **Eliminates Actor Corruption**:
   - No actor destruction during eval
   - Vehicle references stay valid
   - No corrupted VehicleControl values

2. **Simplifies Code**:
   - Remove `eval_tm_port` configuration
   - Remove separate environment creation logic
   - Follows existing EXPLORATION/LEARNING pattern

3. **Aligns with CARLA Architecture**:
   - Respects singleton world design
   - Single TrafficManager instance
   - No port conflicts

4. **Maintains Deterministic Evaluation**:
   - Still uses `deterministic=True` (no exploration noise)
   - Multiple episodes for statistical significance
   - Same evaluation quality as separate env approach

5. **Proven Pattern**:
   - Already working for EXPLORATION phase (no issues)
   - Already working for LEARNING phase (no issues)
   - EVAL is just another phase in the same environment

---

## 🧪 Validation Plan

### Test #1: 100-Step Micro-Test
```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 100 \
  --eval-freq 50 \
  --num-eval-episodes 2 \
  --seed 42 \
  --debug
```

**Expected**:
- Step 0-49: EXPLORATION
- Step 50: EVAL (2 episodes, deterministic)
- Step 51-99: LEARNING
- Step 100: EVAL again
- **NO vehicle corruption**
- **NO CARLA timeout**

### Test #2: Vehicle State Validation

Add logging:
```python
def evaluate(self):
    vehicle_id_before = self.env.vehicle.id
    # ... run eval ...
    vehicle_id_after = self.env.vehicle.id
    assert vehicle_id_before == vehicle_id_after  # Should pass!
```

### Test #3: 5K Validation

Full test with multiple EVAL phases:
```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 5000 \
  --eval-freq 1000 \
  --num-eval-episodes 10 \
  --seed 42
```

**Expected**: 5 EVAL phases (steps 1000, 2000, 3000, 4000, 5000), all complete successfully.

---

## 📝 Implementation Checklist

- [ ] Remove `self.eval_tm_port = 8050` from `__init__`
- [ ] Add `self.in_eval_phase = False` flag
- [ ] Rewrite `evaluate()` to use `self.env` (no separate env)
- [ ] Update training loop to handle EVAL phase
- [ ] Ensure `evaluate()` returns fresh observation
- [ ] Add vehicle state validation logging
- [ ] Run 100-step micro-test
- [ ] Run 5K validation test
- [ ] Verify TensorBoard metrics continuity

---

## 🎓 Key Takeaway

**CARLA is not Gym.** The TD3 paper's approach of creating separate evaluation environments works for Gym/MuJoCo (independent simulations) but **fails catastrophically for CARLA** (shared server, singleton world).

**Solution**: Treat EVAL as a **third phase** (like EXPLORATION and LEARNING) that uses the **same environment** with **deterministic policy** (no exploration noise).

This is simpler, safer, and respects CARLA's architecture.

---

**Status**: ✅ **READY FOR IMPLEMENTATION**  
**Next Step**: Modify `scripts/train_td3.py` to implement phase-based EVAL  
**Reference**: See `EVAL_PHASE_SOLUTION_ANALYSIS.md` for detailed implementation guide
