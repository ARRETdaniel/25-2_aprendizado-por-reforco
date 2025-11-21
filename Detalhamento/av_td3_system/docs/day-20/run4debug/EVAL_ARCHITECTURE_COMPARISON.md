# EVAL Architecture Comparison: Current vs Proposed

**Date**: 2025-11-20  
**Purpose**: Visual comparison of evaluation architectures

---

## 🏗️ Current Architecture (BROKEN)

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA SERVER (Port 2000)                  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SINGLETON WORLD (Town01)                  │ │
│  │                                                        │ │
│  │  TRAINING PHASE:                                       │ │
│  │  ┌────────────────┐  ┌─────────────────┐              │ │
│  │  │ Ego Vehicle    │  │ TrafficManager  │              │ │
│  │  │ ID: 123        │  │ Port: 8000      │              │ │
│  │  │ Status: ACTIVE │  │ 20 NPCs         │              │ │
│  │  └────────────────┘  └─────────────────┘              │ │
│  │                                                        │ │
│  │  EVAL PHASE (t = 5000):                                │ │
│  │  ┌────────────────┐  ┌─────────────────┐              │ │
│  │  │ EVAL Vehicle   │  │ TrafficManager  │              │ │
│  │  │ ID: 456        │  │ Port: 8050      │ ← CONFLICT!  │ │
│  │  │ Status: ACTIVE │  │ 20 NPCs         │              │ │
│  │  └────────────────┘  └─────────────────┘              │ │
│  │                                                        │ │
│  │  eval_env.close() called:                              │ │
│  │  ┌────────────────┐  ┌─────────────────┐              │ │
│  │  │ EVAL Vehicle   │  │ TrafficManager  │              │ │
│  │  │ ID: 456        │  │ Port: 8050      │ ← PERSISTS!  │ │
│  │  │ Status: ❌DEAD │  │ (orphaned)      │              │ │
│  │  └────────────────┘  └─────────────────┘              │ │
│  │                                                        │ │
│  │  BACK TO TRAINING:                                     │ │
│  │  ┌────────────────┐                                   │ │
│  │  │ Ego Vehicle    │ ← Reference may be STALE!         │ │
│  │  │ ID: 123 (?)    │   (Could point to recycled actor) │ │
│  │  │ Status: ???    │                                   │ │
│  │  └────────────────┘                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

Result: 💥 vehicle.apply_control() returns CORRUPTED values
        (steering=-973852377088, gear=5649815)
```

### Code Flow

```python
# Training initialization
env = CARLANavigationEnv(tm_port=8000)          # Creates client, spawns vehicle ID=123
vehicle_ref = env.vehicle                        # Reference to ID=123

# Training loop (t=0 to 4999)
for t in range(5000):
    action = agent.select_action(obs)
    vehicle_ref.apply_control(action)            # ✅ Works (ID=123 alive)

# EVAL phase (t=5000)
eval_env = CARLANavigationEnv(tm_port=8050)     # ❌ Spawns EVAL vehicle ID=456
                                                 # ❌ Creates TM on port 8050
eval_env.reset()                                 # ❌ May destroy NPCs from training TM
# ... run 10 eval episodes ...
eval_env.close()                                 # ❌ Destroys EVAL vehicle ID=456
                                                 # ❌ TM on port 8050 persists (orphaned)

# Back to training (t=5001)
action = agent.select_action(obs)
vehicle_ref.apply_control(action)                # 💥 CRASH! ID=123 may be invalid
                                                 # Returns corrupted VehicleControl
```

### Why It Fails

1. **Shared World**: `eval_env` connects to the SAME CARLA world as `env`
2. **Actor Destruction**: `eval_env.close()` destroys actors in the shared world
3. **Stale References**: `env.vehicle` (ID=123) may become invalid
4. **TM Conflict**: Two TMs (port 8000 and 8050) manage NPCs in the same world
5. **Orphaned TM**: TM on port 8050 persists after eval, keeps managing dead NPCs

---

## ✅ Proposed Architecture (FIXED)

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA SERVER (Port 2000)                  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SINGLETON WORLD (Town01)                  │ │
│  │                                                        │ │
│  │  ┌────────────────┐  ┌─────────────────┐              │ │
│  │  │ Ego Vehicle    │  │ TrafficManager  │              │ │
│  │  │ ID: 123        │  │ Port: 8000      │              │ │
│  │  │ Status: ACTIVE │  │ 20 NPCs         │              │ │
│  │  └────────────────┘  └─────────────────┘              │ │
│  │        ↑                     ↑                         │ │
│  │        │                     │                         │ │
│  │        └─────────────────────┘                         │ │
│  │               SHARED BY:                               │ │
│  │        - EXPLORATION phase                             │ │
│  │        - LEARNING phase                                │ │
│  │        - EVAL phase ← NEW!                             │ │
│  │                                                        │ │
│  │  NO separate EVAL environment!                         │ │
│  │  ALL phases use the SAME actor (ID=123)                │ │
│  │  ALL phases use the SAME TrafficManager (port 8000)    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

Result: ✅ No actor destruction
        ✅ No stale references
        ✅ No TM conflicts
        ✅ Simple, safe architecture
```

### Code Flow

```python
# Training initialization
env = CARLANavigationEnv(tm_port=8000)          # Creates client, spawns vehicle ID=123
vehicle_ref = env.vehicle                        # Reference to ID=123

# Training loop with EVAL phase
for t in range(max_timesteps):
    
    # ──────────────────────────────────────────
    # EVAL PHASE (t % eval_freq == 0)
    # ──────────────────────────────────────────
    if t > 0 and t % eval_freq == 0:
        in_eval_phase = True
        
        # Run evaluation using SAME environment
        eval_metrics = evaluate(env, agent)     # ✅ Uses env (not eval_env!)
        
        # evaluate() implementation:
        for eval_ep in range(num_eval_episodes):
            obs, _ = env.reset()                # ✅ Reset SAME environment
            while not done:
                action = agent.select_action(obs, deterministic=True)  # ✅ No noise
                obs, reward, done, _, info = env.step(action)
                # No training, just collect metrics
        
        # Reset after eval for fresh training episode
        obs, _ = env.reset()                    # ✅ Fresh start
        in_eval_phase = False
    
    # ──────────────────────────────────────────
    # EXPLORATION PHASE (t < start_timesteps)
    # ──────────────────────────────────────────
    if t < start_timesteps:
        action = env.action_space.sample()      # ✅ Random action
    
    # ──────────────────────────────────────────
    # LEARNING PHASE (t >= start_timesteps)
    # ──────────────────────────────────────────
    elif not in_eval_phase:
        action = agent.select_action(obs, deterministic=False)
        noise = np.random.normal(...)           # ✅ Exploration noise
        action = (action + noise).clip(...)
    
    # Execute action (ALL phases use same vehicle_ref)
    next_obs, reward, done, _, info = env.step(action)  # ✅ Always valid
    
    # Store transition (LEARNING phase only)
    if not in_eval_phase and t >= start_timesteps:
        replay_buffer.add(obs, action, next_obs, reward, done)
    
    # Train agent (LEARNING phase only)
    if not in_eval_phase and t >= start_timesteps:
        agent.train(replay_buffer, batch_size)

# Clean shutdown
env.close()  # ✅ Only closed at the very end
```

### Why It Works

1. **Single Environment**: `env` instance persists throughout entire training
2. **No Actor Destruction**: Vehicle ID=123 stays alive (only reset, not destroyed)
3. **Valid References**: `env.vehicle` always points to live actor
4. **Single TM**: Port 8000 manages all NPCs consistently
5. **Phase Transitions**: Seamless switching between EXPLORATION/LEARNING/EVAL

---

## 📊 Side-by-Side Comparison

| Aspect | Current (Separate EVAL Env) | Proposed (Phase-Based) |
|--------|----------------------------|------------------------|
| **Environment Instances** | 2 (train + eval) | 1 (unified) |
| **Vehicle Instances** | 2 (train vehicle + eval vehicle) | 1 (same vehicle reset) |
| **TrafficManager Ports** | 2 (8000 + 8050) | 1 (8000) |
| **Actor Lifecycle** | ❌ CONFLICT (eval destroys actors) | ✅ SAFE (only resets) |
| **Vehicle Reference** | ❌ STALE after eval | ✅ ALWAYS VALID |
| **Code Complexity** | 🟡 HIGH (manage 2 envs) | 🟢 LOW (single env) |
| **CARLA Compliance** | ❌ Violates singleton world | ✅ Respects architecture |
| **TD3 Paper Compliance** | 🟡 Similar (separate env) | 🟢 Same (deterministic policy) |
| **Deterministic Eval** | ✅ Yes (different seed) | ✅ Yes (no noise) |
| **Risk Level** | 🔴 HIGH (vehicle corruption) | 🟢 LOW (proven pattern) |

---

## 🔄 Phase Transition Diagram

### Proposed Architecture Flow

```
Timestep:  0     1000   1001   2000   2001   ...   5000   5001
           ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
Phase:    EXPL   EVAL   LEARN  EVAL   LEARN  ... EVAL   LEARN
           │      │      │      │      │      │      │      │
Action:   RND    DET    POL+N  DET    POL+N  ... DET    POL+N
           │      │      │      │      │      │      │      │
Train:     ✗     ✗      ✓      ✗      ✓      ... ✗      ✓
           │      │      │      │      │      │      │      │
Env:     SAME   SAME   SAME   SAME   SAME   ... SAME   SAME
           └──────┴──────┴──────┴──────┴──────...──┴──────┘
                    ALL USE: env (ID=123)

Legend:
  EXPL = EXPLORATION (random actions)
  EVAL = EVALUATION (deterministic policy, no training)
  LEARN = LEARNING (policy + noise, training enabled)
  RND = Random action
  DET = Deterministic policy (no noise)
  POL+N = Policy + exploration noise
```

### Current Architecture Flow (BROKEN)

```
Timestep:  0     1000            1001   2000            2001
           ↓      ↓               ↓      ↓               ↓
Phase:    EXPL   EVAL            LEARN  EVAL            LEARN
           │      │               │      │               │
Env:      env    eval_env        env    eval_env        env
           │      │               │      │               │
Vehicle:  123    456 ← NEW!      123    456 ← NEW!      123
           │      │               │      │               │
Status:  ALIVE   ALIVE           ???    ALIVE           ???
                  ↓ close()               ↓ close()
                 DEAD                    DEAD
                  ↓                       ↓
                env.vehicle (123)       env.vehicle (123)
                MAY BE INVALID!         MAY BE INVALID!
                      ↓                       ↓
                    💥 CRASH               💥 CRASH
```

---

## 🎯 Implementation Diff Preview

### Current Code (BROKEN)

```python
def evaluate(self):
    # ❌ Creates NEW environment with DIFFERENT TM port
    eval_env = CARLANavigationEnv(
        self.carla_config_path,
        self.agent_config_path,
        self.training_config_path,
        tm_port=self.eval_tm_port  # Port 8050
    )
    
    for episode in range(self.num_eval_episodes):
        obs_dict, _ = eval_env.reset()  # Uses eval_env
        # ... run episode ...
        next_obs_dict, reward, done, _, info = eval_env.step(action)
    
    # ❌ Destroys EVAL actors (corrupts training env)
    eval_env.close()
    
    return eval_metrics
```

### Proposed Code (FIXED)

```python
def evaluate(self):
    """Evaluate using TRAINING environment (no separate instance)."""
    print(f"[EVAL] Entering evaluation phase...")
    
    # ✅ Save current episode count (EVAL doesn't count as training episodes)
    episode_num_before = self.episode_num
    
    for episode in range(self.num_eval_episodes):
        obs_dict, _ = self.env.reset()  # ✅ Uses TRAINING env (self.env)
        
        while not done:
            # ✅ Deterministic action (no exploration noise)
            action = self.agent.select_action(obs_dict, deterministic=True)
            next_obs_dict, reward, done, _, info = self.env.step(action)
            # ... collect metrics ...
    
    # ✅ Restore episode count
    self.episode_num = episode_num_before
    
    # ✅ Reset after EVAL for fresh training episode
    obs_dict, _ = self.env.reset()
    
    print(f"[EVAL] Exiting evaluation phase")
    
    # ✅ Return metrics AND fresh observation
    return eval_metrics, obs_dict
```

---

## 📝 Migration Checklist

### Files to Modify

- [ ] `scripts/train_td3.py`:
  - [ ] Remove `self.eval_tm_port = 8050` from `__init__`
  - [ ] Add `self.in_eval_phase = False` flag
  - [ ] Rewrite `evaluate()` method (use `self.env`)
  - [ ] Update training loop to handle EVAL phase

### Variables to Remove

```python
# DELETE THESE:
self.eval_tm_port = 8050
```

### New Variables to Add

```python
# ADD THESE:
self.in_eval_phase = False  # Track evaluation mode
```

### Methods to Modify

```python
# BEFORE: def evaluate(self):
#   Creates eval_env, closes it after
#
# AFTER: def evaluate(self):
#   Uses self.env, returns (metrics, obs_dict)
```

---

## ✅ Expected Outcomes After Implementation

1. **NO vehicle state corruption**:
   - `vehicle.apply_control()` always receives valid actor
   - No corrupted steering/brake/gear values

2. **NO CARLA timeout errors**:
   - No actor lifecycle conflicts
   - No TM registry inconsistencies

3. **Simplified codebase**:
   - Single environment instance
   - Single TrafficManager port
   - Cleaner training loop logic

4. **Same evaluation quality**:
   - Still runs `num_eval_episodes` deterministic episodes
   - Still collects mean/std metrics
   - Still logs to TensorBoard

5. **Proven reliability**:
   - Uses same pattern as EXPLORATION/LEARNING phases
   - No environment switching complexity
   - Respects CARLA's singleton world design

---

**Status**: ✅ **ANALYSIS COMPLETE - READY TO IMPLEMENT**  
**Next**: Modify `scripts/train_td3.py` according to proposed changes  
**Validation**: Run 100-step micro-test, then 5K validation
