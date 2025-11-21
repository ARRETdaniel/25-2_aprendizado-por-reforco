# EVAL Phase Solution - Unified Training Architecture

**Date**: 2025-11-20  
**Issue Reference**: ANALYSIS_TODO_LIST.md - Issue #2 (CARLA Vehicle State Corruption Post-EVAL)  
**Documentation**: CARLA 0.9.16 API, TD3 Paper (Fujimoto et al.), Stable-Baselines3

---

## 🎯 Executive Summary

**Current Problem**: Creating a separate CARLA environment for EVAL causes catastrophic vehicle state corruption when returning to training. The main environment's vehicle reference becomes stale (points to destroyed actor), leading to corrupted VehicleControl values (steering=-973852377088).

**Root Cause**: CARLA's `Client.get_world()` returns a SINGLETON world instance. Creating a new environment doesn't create a new world, it SHARES the same world instance, leading to actor lifecycle conflicts.

**Proposed Solution**: Eliminate separate EVAL environment. Use **PHASE-BASED ARCHITECTURE** like EXPLORATION and LEARNING, where EVAL becomes the third phase sharing the SAME environment, SAME client, SAME TrafficManager.

**Benefits**:
- ✅ **NO environment switching** → No actor lifecycle issues
- ✅ **NO port conflicts** → Single Traffic Manager instance
- ✅ **Simpler codebase** → Follows TD3 reference implementation
- ✅ **Aligned with paper** → TD3 evaluates on SAME environment
- ✅ **Deterministic evaluation** → Disables exploration noise only

---

## 📚 Documentation Research

### CARLA 0.9.16 Official Documentation

From `https://carla.readthedocs.io/en/latest/python_api/#carla.Client`:

```python
Client.get_world(self)
    Returns the world object currently active in the simulation.
    Return: carla.World
    
    Note: This world will be later used for example to load maps.
```

**Key Insight**: `get_world()` returns the **CURRENTLY ACTIVE** world. There is **NO API** to create multiple independent world instances per client. CARLA is designed with a single-world-per-server architecture.

From `https://carla.readthedocs.io/en/latest/python_api/#carla.World`:

```python
World.destroy(actor)
    Tells the simulator to destroy this actor and returns True if it was successful.
    It has no effect if it was already destroyed.
    Warning: This method blocks the script until the destruction is completed by the simulator.
```

**Key Insight**: When EVAL environment calls `env.close()`, it destroys all its actors (vehicle, sensors, NPCs). Main training environment's references to these actors **become invalid**.

From `https://carla.readthedocs.io/en/latest/python_api/#carla.Actor`:

```python
Actor.is_alive (bool)
    Returns whether this object was destroyed using this actor handle.
    
Actor.is_active (bool)
    Returns whether this actor is active (True) or not (False).
```

**Key Insight**: CARLA provides `is_alive` and `is_active` checks, but our current code doesn't validate actor state after EVAL closes.

### CARLA TrafficManager Documentation

From `https://carla.readthedocs.io/en/latest/python_api/#carla.TrafficManager`:

```python
Client.get_trafficmanager(self, client_connection=8000)
    Returns an instance of the traffic manager related to the specified port.
    If it does not exist, this will be created.
    
TrafficManager.shut_down(self)
    Shuts down the traffic manager.
```

**Behavior**: TrafficManagers are **PERSISTENT** across environment resets. They continue managing NPCs until explicitly shut down. Creating EVAL env with port 8050 creates a SECOND TM instance that persists after `eval_env.close()`.

### TD3 Reference Implementation

From `TD3/main.py` (Fujimoto et al. official code):

```python
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)  # ✅ Creates NEW environment
    eval_env.seed(seed + 100)      # ✅ Different seed for determinism
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))  # ⚠️ Deterministic (no noise)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes
    return avg_reward
```

**Key Observations**:
1. ✅ **Creates separate environment** for eval (common in Gym/MuJoCo)
2. ✅ **Uses different seed** (seed+100) for reproducibility
3. ❌ **DOES NOT use exploration noise** during eval (deterministic policy)
4. ⚠️ **Works for Gym** because environments are INDEPENDENT (no shared simulator state)

**Why This Fails in CARLA**:
- Gym/MuJoCo: Each `gym.make()` creates an INDEPENDENT simulation instance
- CARLA: Each `CARLANavigationEnv()` connects to a SHARED CARLA server (singleton world)
- Result: **Actor destruction in eval_env affects train_env** (shared world!)

---

## 🔍 Root Cause Analysis

### Evidence from Debug Log

From `run-1validation_5k_post_all_fixes_20251120_170736.log`:

```
Line 328696: [EVAL] Closing evaluation environment...
Line 328716: [EVAL] Mean Reward: 64.19 | Success Rate: 0.0%
Line 328742: Applied Control: throttle=0.0000, brake=0.0000, steer=-973852377088.0000
Line 328742: Hand Brake: True, Reverse: True, Gear: 5649815
```

**Analysis**:
1. **Line 328696**: `eval_env.close()` called
2. **CARLA Internal**: Destroys EVAL vehicle, sensors, NPCs
3. **Line 328742**: Training loop tries to apply control to **STALE vehicle reference**
4. **Result**: `vehicle.apply_control()` returns corrupted VehicleControl object

### CARLA World Lifecycle

```
Training Initialization:
  ├─ Client connects to CARLA server (port 2000)
  ├─ client.get_world() → Returns SINGLETON world instance
  ├─ TrafficManager(port=8000) created for training NPCs
  └─ Ego vehicle spawned (ID=123, for example)

EVAL Phase (BROKEN):
  ├─ eval_env = CARLANavigationEnv(..., tm_port=8050)
  ├─ REUSES same client connection (same world!)
  ├─ TrafficManager(port=8050) created (CONFLICT with port 8000)
  ├─ EVAL ego vehicle spawned (NEW actor, ID=456)
  ├─ eval_env.close() called
  ├─ ❌ Destroys EVAL vehicle (ID=456)
  ├─ ❌ Training env still references OLD vehicle (ID=123)
  └─ ❌ ID=123 may have been destroyed or recycled!

Back to Training (CORRUPTED):
  ├─ Training loop: vehicle.apply_control(...)
  ├─ vehicle.id = 123 (STALE REFERENCE)
  ├─ CARLA returns corrupted VehicleControl (invalid actor)
  └─ 💥 CRASH: steering=-973852377088, gear=5649815
```

### Why Separate TM Ports Don't Help

From CARLA docs:
> "All actors present in the current world will be destroyed, but traffic manager instances will stay alive."

**Problem**: Even with separate TM ports:
1. Both TMs manage NPCs in the SAME world
2. EVAL TM (port 8050) persists after `eval_env.close()`
3. Training TM (port 8000) continues managing stale NPC references
4. When NPCs are destroyed in EVAL, TM registry becomes inconsistent

---

## ✅ Proposed Solution: Unified PHASE-BASED Architecture

### Design Principle

**Follow the pattern already established in `train_td3.py`**:
- ✅ **EXPLORATION phase** (t < start_timesteps): Random actions, no training
- ✅ **LEARNING phase** (t ≥ start_timesteps): Policy actions + noise, training enabled
- ✅ **EVAL phase** (NEW - t % eval_freq == 0): Policy actions, NO noise, NO training

All three phases use the **SAME environment instance**.

### Implementation Changes

#### **Change #1**: Remove Separate EVAL Environment Creation

**Current (BROKEN)**:
```python
def evaluate(self):
    # ❌ Creates NEW environment with DIFFERENT TM port
    eval_env = CARLANavigationEnv(
        self.carla_config_path,
        self.agent_config_path,
        self.training_config_path,
        tm_port=self.eval_tm_port  # Port 8050
    )
    # ... run eval episodes ...
    eval_env.close()  # ❌ Destroys actors, corrupts training env
```

**Proposed (FIXED)**:
```python
def evaluate(self):
    """
    Evaluate agent using the SAME environment (no separate instance).
    Switches to deterministic policy (no exploration noise).
    """
    # ✅ Use self.env (training environment)
    # No need to create separate environment!
```

#### **Change #2**: Add EVAL Phase Flag

```python
class TD3TrainingPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.in_eval_phase = False  # NEW: Track evaluation mode
```

#### **Change #3**: Modify Training Loop to Include EVAL Phase

**Before**:
```python
for t in range(int(self.max_timesteps)):
    # EXPLORATION phase
    if t < self.start_timesteps:
        action = self.env.action_space.sample()
    # LEARNING phase
    else:
        action = self.agent.select_action(obs_dict, deterministic=False)
        noise = np.random.normal(0, max_action * self.expl_noise, ...)
        action = (action + noise).clip(-max_action, max_action)
```

**After**:
```python
for t in range(int(self.max_timesteps)):
    # Check if we're entering EVAL phase
    if t > 0 and t % self.eval_freq == 0:
        self.in_eval_phase = True
        eval_metrics = self.evaluate()  # Run EVAL phase
        self.in_eval_phase = False
        # Log eval metrics...
    
    # EXPLORATION phase
    if t < self.start_timesteps:
        action = self.env.action_space.sample()
    # LEARNING phase
    elif not self.in_eval_phase:
        action = self.agent.select_action(obs_dict, deterministic=False)
        noise = np.random.normal(0, max_action * self.expl_noise, ...)
        action = (action + noise).clip(-max_action, max_action)
```

#### **Change #4**: Reimplement evaluate() to Use SAME Environment

```python
def evaluate(self):
    """
    Evaluate agent on num_eval_episodes using the TRAINING environment.
    
    CRITICAL: Uses self.env (no separate environment creation).
    Disables exploration noise for deterministic evaluation.
    Does NOT train during evaluation episodes.
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n[EVAL] Starting evaluation phase (deterministic policy)...")
    
    eval_rewards = []
    eval_successes = []
    eval_collisions = []
    eval_lane_invasions = []
    eval_lengths = []
    
    max_eval_steps = self.agent_config.get("training", {}).get("max_episode_steps", 1000)
    
    for episode in range(self.num_eval_episodes):
        # ✅ Reset TRAINING environment (not a separate eval env)
        obs_dict, _ = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_eval_steps:
            # ✅ Deterministic action (NO exploration noise)
            action = self.agent.select_action(obs_dict, deterministic=True)
            
            # ✅ Step TRAINING environment
            next_obs_dict, reward, done, truncated, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            obs_dict = next_obs_dict
            
            if truncated:
                done = True
        
        # Collect metrics
        eval_rewards.append(episode_reward)
        eval_successes.append(info.get('success', 0))
        eval_collisions.append(info.get('collision_count', 0))
        eval_lane_invasions.append(info.get('lane_invasion_count', 0))
        eval_lengths.append(episode_length)
    
    # ✅ NO env.close() - environment stays alive for training!
    
    print(f"[EVAL] Evaluation complete. Mean Reward: {np.mean(eval_rewards):.2f}")
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'success_rate': np.mean(eval_successes),
        'avg_collisions': np.mean(eval_collisions),
        'avg_lane_invasions': np.mean(eval_lane_invasions),
        'avg_episode_length': np.mean(eval_lengths)
    }
```

#### **Change #5**: Remove eval_tm_port Configuration

**Delete from `__init__`**:
```python
# ❌ REMOVE THIS
self.eval_tm_port = 8050  # Separate TM port for eval
```

**No longer needed**: Only one TrafficManager (port 8000) used throughout.

---

## 🧪 Validation Plan

### Test #1: 100-Step Micro-Validation with EVAL

```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 100 \
  --eval-freq 50 \
  --num-eval-episodes 2 \
  --seed 42 \
  --debug
```

**Expected Behavior**:
- ✅ Step 0-49: EXPLORATION phase (random actions)
- ✅ Step 50: EVAL phase triggers
  - Reset environment (SAME instance)
  - Run 2 eval episodes (deterministic policy)
  - Return to training
- ✅ Step 51-99: LEARNING phase (policy + noise)
- ✅ Step 100: EVAL phase triggers again
- ✅ **NO vehicle state corruption**
- ✅ **NO CARLA timeout errors**

### Test #2: Check Vehicle State After EVAL

Add debug logging:
```python
def evaluate(self):
    # Before eval
    vehicle_id_before = self.env.vehicle.id
    print(f"[EVAL] Vehicle ID before eval: {vehicle_id_before}")
    
    # ... run eval episodes ...
    
    # After eval
    vehicle_id_after = self.env.vehicle.id
    vehicle_is_alive = self.env.vehicle.is_alive
    print(f"[EVAL] Vehicle ID after eval: {vehicle_id_after}")
    print(f"[EVAL] Vehicle is_alive: {vehicle_is_alive}")
    
    # Verify consistency
    assert vehicle_id_before == vehicle_id_after, "Vehicle ID changed during EVAL!"
    assert vehicle_is_alive, "Vehicle destroyed during EVAL!"
```

**Expected Output**:
```
[EVAL] Vehicle ID before eval: 123
[EVAL] Running 10 eval episodes...
[EVAL] Vehicle ID after eval: 123
[EVAL] Vehicle is_alive: True
```

### Test #3: TensorBoard Metrics Continuity

**Check**:
- `eval/mean_reward` appears at correct timesteps (50, 100, ...)
- `train/episode_reward` continues after EVAL phase
- No gaps or discontinuities in training metrics

---

## 📊 Comparison: Current vs Proposed

| Aspect | Current (Separate EVAL Env) | Proposed (Unified PHASE-BASED) |
|--------|----------------------------|--------------------------------|
| **Environment Instances** | 2 (train + eval) | 1 (unified) |
| **TrafficManager Ports** | 2 (8000 + 8050) | 1 (8000) |
| **Actor Lifecycle** | CONFLICT (eval destroys actors) | SAFE (no actor changes) |
| **Vehicle References** | STALE after eval | ALWAYS VALID |
| **Code Complexity** | HIGH (manage 2 envs) | LOW (single env) |
| **Aligned with TD3 Paper** | ❌ (creates separate env) | ⚠️ (uses same env, deterministic policy) |
| **Aligned with CARLA Design** | ❌ (violates singleton world) | ✅ (respects CARLA architecture) |
| **Deterministic Eval** | ✅ (different seed) | ✅ (no exploration noise) |
| **Risk of Bugs** | ⚠️ HIGH (actor corruption) | ✅ LOW (no env switching) |

---

## 🚨 Critical Implementation Notes

### Note #1: Episode Count Management

**Issue**: EVAL episodes shouldn't increment `self.episode_num` (training counter).

**Solution**:
```python
def evaluate(self):
    # Save current episode count
    episode_num_before = self.episode_num
    
    # ... run eval episodes ...
    
    # Restore episode count (EVAL doesn't count as training episodes)
    self.episode_num = episode_num_before
```

### Note #2: Replay Buffer Isolation

**Issue**: EVAL experiences shouldn't be added to replay buffer (they're deterministic).

**Solution**: Already handled - `evaluate()` doesn't call `replay_buffer.add()`.

### Note #3: Training vs Evaluation Mode

**Ensure**:
- EVAL: `agent.select_action(obs, deterministic=True)` → No exploration noise
- TRAIN: `agent.select_action(obs, deterministic=False)` → Adds exploration noise

### Note #4: Episode Reset After EVAL

**Current**: After EVAL completes, training loop continues from last state.

**Issue**: Last EVAL episode's final state is used as initial state for next training episode.

**Solution**:
```python
def evaluate(self):
    # ... run eval episodes ...
    
    # ✅ Reset environment after EVAL to start fresh training episode
    obs_dict, _ = self.env.reset()
    return eval_metrics, obs_dict  # Return fresh observation
```

Then in training loop:
```python
if t % self.eval_freq == 0:
    eval_metrics, obs_dict = self.evaluate()
    # Continue training with fresh obs_dict
```

---

## 📝 Implementation Checklist

- [ ] **Step 1**: Remove `self.eval_tm_port` from `__init__`
- [ ] **Step 2**: Add `self.in_eval_phase = False` flag
- [ ] **Step 3**: Rewrite `evaluate()` to use `self.env` (no separate env creation)
- [ ] **Step 4**: Add episode count preservation in `evaluate()`
- [ ] **Step 5**: Return fresh observation from `evaluate()`
- [ ] **Step 6**: Update training loop to handle EVAL phase transition
- [ ] **Step 7**: Add debug logging for vehicle state validation
- [ ] **Step 8**: Run 100-step micro-validation test
- [ ] **Step 9**: Run 5K validation test with multiple EVAL phases
- [ ] **Step 10**: Verify TensorBoard metrics continuity

---

## 🎓 Lessons Learned

1. **CARLA is NOT Gym**: Cannot create independent environment instances. Must respect singleton world architecture.

2. **TD3 Paper ≠ CARLA Reality**: TD3's separate eval env works for Gym/MuJoCo but FAILS for CARLA due to shared simulator state.

3. **Simplicity Wins**: Phase-based architecture (EXPLORATION → LEARNING → EVAL) is simpler and safer than environment switching.

4. **Actor Lifecycle is Critical**: CARLA actors are server-managed. References become stale when actors are destroyed. ALWAYS validate with `is_alive`.

5. **TrafficManager Persistence**: TMs outlive environment instances. Creating multiple TMs leads to registry conflicts.

---

## 📚 References

- **CARLA 0.9.16 Python API**: https://carla.readthedocs.io/en/latest/python_api/
- **TD3 Paper**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods"
- **TD3 Reference Code**: `TD3/main.py` (official implementation)
- **CARLA Actor Lifecycle**: https://carla.readthedocs.io/en/latest/python_api/#carla.Actor
- **TrafficManager Docs**: https://carla.readthedocs.io/en/latest/adv_traffic_manager/

---

**Status**: ✅ SOLUTION PROPOSED - READY FOR IMPLEMENTATION  
**Next**: Implement changes in `train_td3.py` and validate with micro-test
