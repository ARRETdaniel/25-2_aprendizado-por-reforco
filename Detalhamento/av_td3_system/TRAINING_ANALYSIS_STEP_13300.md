# Training Analysis Report - Step 13,300

**Date**: October 26, 2025
**Training Status**: Interrupted at step 13,300 / 30,000 (44.3% complete)
**Checkpoint Available**: td3_scenario_0_step_10000.pth
**Phase**: LEARNING (started at step 10,001)

---

## Executive Summary

### 🎉 **MAJOR BREAKTHROUGH AT STEP 11,100-11,470**

After 10,000 steps of exploration with random actions (vehicle mostly stationary), **the TD3 agent learned to move** once learning started at step 10,001!

**Key Evidence**:
- **Step 11,200**: Reward = -0.35, Speed = 8.8 km/h ✅ (first meaningful movement!)
- **Step 11,300**: Reward = -0.51, Speed = 11.3 km/h ✅
- **Step 11,400**: Reward = +0.72, Speed = 5.3 km/h ✅ (first POSITIVE reward!)
- **Step 11,500**: Reward = -0.42, Speed = 9.7 km/h ✅

**This is proof that TD3 is learning!** The agent transitioned from complete standstill (0.0 km/h) to consistent forward movement (5-11 km/h) within just 500 training steps.

### ⚠️ **CRITICAL REGRESSION AT STEP 11,600+**

Unfortunately, after Episode 22, the agent **reverted to stationary behavior** and remained stuck for 1,700+ steps:
- Steps 11,600-13,300: Speed = 0.0 km/h, Reward = -1.00 (constant)

This indicates a **learning instability** or **local minimum** in the policy space.

---

## Detailed Training Timeline

### Phase 1: Exploration (Steps 1-10,000)

**Behavior**: Random actions, vehicle mostly stationary
**Speed Range**: 0.0-1.6 km/h (median: 0.0 km/h)
**Reward**: -1.00 (constant efficiency penalty)
**Episodes Completed**: 1 (Episode 0: 5,001 steps, Episode 1: 5,000 steps)

**Analysis**:
- ✅ **Expected behavior**: Random actions with no learning
- ✅ **Buffer filling correctly**: 10,000 transitions collected
- ⚠️ **Vehicle mostly stationary**: Random throttle/brake centered around 0 → net effect is minimal movement

### Phase 2: Learning - Initial Success (Steps 10,001-11,500)

**Episodes 2-22**: Major learning breakthrough!

| Step | Episode | Duration | Max Speed | Max Reward | Behavior |
|------|---------|----------|-----------|------------|----------|
| 10,001-11,187 | 2 | 1,187 steps | 0.0 km/h | -1.00 | Still stationary (inertia from exploration) |
| 11,188-11,200 | 3 | 13 steps | **8.8 km/h** | **-0.35** | **First movement!** Agent accelerates |
| 11,201-11,214 | 4-8 | ~2-15 steps each | Lane invasions | Resets | Quick failures, but agent is trying |
| 11,215-11,332 | 9-10 | 117 steps | **11.3 km/h** | **-0.51** | **Best speed!** Sustained movement |
| 11,333-11,400 | 11-16 | Various | 5-10 km/h | -0.4 to +0.72 | **Consistent movement**, exploring |
| 11,401-11,470 | 17-20 | Short eps | Lane invasions | Frequent resets | Agent pushing boundaries |
| 11,471-11,513 | 21-22 | 42 steps | **9.7 km/h** | **-0.42** | Good movement maintained |

**Key Observations**:
1. ✅ **TD3 successfully learned to accelerate** after only ~200 learning steps (11,200 - 10,001)
2. ✅ **Reward improved dramatically**: From -1.00 (stationary) to +0.72 (moving)
3. ✅ **Target speed range**: Agent approaching 8-11 km/h (target: 30 km/h = 8.33 m/s)
4. ⚠️ **Lane invasions frequent**: Agent hasn't learned lane-keeping yet (expected early on)
5. ⚠️ **Episodes very short**: 2-117 steps (agent exploring, causing failures)

### Phase 3: Regression (Steps 11,600-13,300)

**Episode 23**: Longest episode, but vehicle stuck!

| Step Range | Episode | Duration | Speed | Reward | Behavior |
|------------|---------|----------|-------|--------|----------|
| 11,514-13,300 | 23 | **1,786 steps** | **0.0 km/h** | **-1.00** | **Completely stationary** |

**Critical Problem**:
- Agent learned to accelerate (steps 11,200-11,500)
- Then **unlearned** and reverted to stationary policy
- Has been stuck for 1,786 consecutive steps with zero movement

**Possible Causes**:
1. **Value Overestimation Bias**: Despite TD3's clipped double Q-learning, overestimation may still occur
2. **Poor Exploration**: Agent discovered a "safe" local minimum (don't move = no collisions)
3. **Reward Function Issue**: Efficiency penalty (-1.0 for standing still) may not be strong enough
4. **Learning Rate Too High**: Policy oscillating between "move" and "don't move"
5. **Insufficient Training**: 1,300 learning steps may be too few for stable convergence

---

## Evaluation Results

### Evaluation at Step 5,000 (Exploration Phase)

```
[EVAL] Mean Reward: -463.09 | Success Rate: 0.0% | Avg Collisions: 0.00 | Avg Length: 88
```

**Analysis**:
- ✅ **No collisions**: Agent safely stationary (expected during random exploration)
- ⚠️ **Very short episodes**: 88 steps average (88-95 steps range)
- ⚠️ **Poor rewards**: -463 per episode = ~-5.26 per step (mostly efficiency penalty)
- ❌ **Success rate 0%**: Agent not reaching goals (expected, random actions)

**Episode Termination**:
- All 10 eval episodes ended with **lane invasions** (agent drifting out of lane)
- No timeouts reached (max_episode_steps = 1000)

### Evaluation at Step 10,000 (End of Exploration)

```
[EVAL] Mean Reward: -463.47 | Success Rate: 0.0% | Avg Collisions: 0.00 | Avg Length: 88
```

**Analysis**:
- ⚠️ **Nearly identical to step 5,000**: Agent hasn't changed behavior (still exploration)
- ✅ **Consistent**: Confirms random actions produce stable (poor) performance
- ⚠️ **No improvement**: Expected, since learning hasn't started yet

**Key Insight**: Both evaluations show the agent using **random actions** (deterministic selection with noise=0.0, but the learned policy at this point is just initialized random weights).

---

## Configuration Issues Identified

### 1. ❌ **Scenario Configuration Mismatch**

**Evidence from logs**:
```
[CONFIG] Scenario: 0 (0=20, 1=50, 2=100 NPCs)
[CONFIG] NPC count set to: 20

But during training:
WARNING:src.environment.carla_env:No scenarios found in config, using default NPC count: 50
```

**Problem**: Training script sets scenario 0 (20 NPCs), but environment falls back to 50 NPCs

**Impact**:
- Inconsistent traffic density between config and execution
- Training on 50 NPCs but thinking it's 20 NPCs
- May affect generalization and reproducibility

**Fix Required**: Check `carla_config.yaml` scenario configuration

### 2. ⚠️ **Evaluation Uses Different NPC Count**

**Evidence**:
```
[EVAL] Evaluation at timestep 5,000...
WARNING:src.environment.carla_env:No scenarios found in config, using default NPC count: 50
```

**Problem**: Evaluations run with 50 NPCs, not the configured 20 NPCs

**Impact**:
- Training and evaluation on different environments
- Evaluation metrics may not reflect training performance
- Harder environment during eval (more NPCs = more obstacles)

**Fix Required**: Ensure eval uses same scenario as training

### 3. ⚠️ **Missing GlobalRoutePlanner**

**Evidence**:
```
WARNING:root:Could not import GlobalRoutePlanner from agents.navigation
WARNING:src.environment.carla_env:Failed to initialize DynamicRouteManager: GlobalRoutePlanner not available
Falling back to legacy waypoint manager
```

**Problem**: CARLA agents package not properly installed in PYTHONPATH

**Impact**:
- Using simplified waypoint manager instead of dynamic route planning
- May affect waypoint quality and goal-directed navigation
- Progress reward component may not work optimally

**Fix Required**: Install CARLA PythonAPI properly or verify waypoint manager works

---

## Reward Function Analysis

### Efficiency Reward

**Implementation** (from `reward_functions.py` lines 226-268):
```python
def _calculate_efficiency_reward(self, velocity: float) -> float:
    if velocity < 1.0:  # Below 1 m/s (3.6 km/h) = essentially stopped
        efficiency = -1.0  # STRONG penalty for not moving
    elif velocity < self.target_speed * 0.5:  # Below half target speed
        efficiency = -0.5 + (velocity_normalized * 0.5)
    elif abs(velocity - self.target_speed) <= self.speed_tolerance:
        # Within tolerance: positive reward (optimal range)
        efficiency = 1.0 - (speed_diff / self.speed_tolerance) * 0.3
    else:
        # Outside tolerance (overspeeding or underspeeding)
        ...
```

**Parameters**:
- `target_speed`: 8.33 m/s (30 km/h)
- `speed_tolerance`: 1.39 m/s (5 km/h)
- Optimal range: 6.94 - 9.72 m/s (25 - 35 km/h)

**Validation**:
✅ **CORRECT implementation** following Pérez-Gil et al. (2022):
- Heavily penalizes standing still (-1.0)
- Rewards target speed tracking
- Moderate penalty for overspeeding

**Units**: Correctly uses **m/s** throughout, consistent with CARLA API

### Lane Keeping Reward (CRITICAL FIX APPLIED)

**Implementation** (lines 270-305):
```python
def _calculate_lane_keeping_reward(
    self, lateral_deviation: float, heading_error: float, velocity: float
) -> float:
    # CRITICAL: No lane keeping reward if not moving!
    if velocity < 1.0:
        return 0.0  # Zero reward for staying centered while stationary

    # Only reward lane-keeping when moving
    ...
```

✅ **FIXED**: Lane-keeping reward **gated by velocity** (only given when moving)
- Prevents agent from learning "stay centered while stopped" policy
- Incentivizes movement first, then lane-keeping

### Comfort Reward (CRITICAL FIX APPLIED)

**Implementation** (lines 307-346):
```python
def _calculate_comfort_reward(
    self, acceleration: float, acceleration_lateral: float, velocity: float
) -> float:
    # CRITICAL: No comfort reward if not moving!
    if velocity < 1.0:
        return 0.0  # Zero reward for smoothness while stationary
    ...
```

✅ **FIXED**: Comfort reward **gated by velocity** (only given when moving)
- Prevents rewarding "smooth acceleration profile" while stopped
- Incentivizes movement first, then smoothness

### Safety Reward (FIX APPLIED - POSSIBLY TOO LENIENT)

**Implementation** (lines 348-389):
```python
def _calculate_safety_reward(
    self,
    collision_detected: bool,
    offroad_detected: bool,
    wrong_way: bool,
    velocity: float,
    distance_to_goal: float
) -> float:
    safety = 0.0

    if collision_detected:
        safety += self.collision_penalty  # -1000.0
    if offroad_detected:
        safety += self.offroad_penalty  # -500.0
    if wrong_way:
        safety += self.wrong_way_penalty  # -200.0

    # REMOVED: Overly aggressive stopping penalty
    # OLD CODE (disabled):
    # if velocity < 0.5 and distance_to_goal > 5.0 and not collision_detected:
    #     safety += -1.0  # This was preventing exploration

    return float(safety)
```

⚠️ **POTENTIAL ISSUE**: The stopping penalty was **removed** to encourage exploration

**Rationale in code comments**:
> "The agent needs time to learn how to move forward without being constantly penalized. The efficiency reward already handles this by rewarding target speed achievement. Let the agent explore!"

**However**: This may have created a local minimum where:
- Agent gets -1.0 from efficiency (stopping)
- Agent gets 0.0 from lane-keeping (stopped, so no reward)
- Agent gets 0.0 from comfort (stopped, so no reward)
- Agent gets 0.0 from safety (no collision when stopped)
- **Total**: -1.0 per step (but "safe"!)

**Alternative hypothesis**: Agent may have learned that moving causes:
- Lane invasions (-500.0 penalty)
- Collisions (not seen yet, but -1000.0 penalty)
- Higher risk of failure

So it chose the "safest" policy: don't move!

### Progress Reward

**Implementation** (lines 391-447):
```python
def _calculate_progress_reward(
    self,
    distance_to_goal: float,
    waypoint_reached: bool,
    goal_reached: bool,
) -> float:
    progress = 0.0

    # Distance-based reward (dense)
    if self.prev_distance_to_goal is not None:
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        progress += distance_delta * self.distance_scale  # 0.1

    # Waypoint bonus (sparse)
    if waypoint_reached:
        progress += self.waypoint_bonus  # 10.0

    # Goal reached bonus (sparse)
    if goal_reached:
        progress += self.goal_reached_bonus  # 100.0

    return float(np.clip(progress, -10.0, 110.0))
```

**Weights** (from training_config.yaml):
```yaml
weights:
  efficiency: 1.0
  lane_keeping: 2.0
  comfort: 0.5
  safety: -100.0  # NOTE: This is a multiplier, not the penalty!
  progress: 5.0
```

⚠️ **CRITICAL ISSUE**: When vehicle is stationary:
- `distance_delta = 0` → progress = 0.0
- No waypoints reached → progress = 0.0
- **Total progress reward** = 0.0

So the progress component **does not incentivize initial movement**, only **rewards existing movement** that reduces distance!

---

## TD3 Algorithm Implementation Validation

### Training Loop (from `train_td3.py`)

**Exploration Phase** (steps 1-10,000):
```python
if t < start_timesteps:
    # Random actions
    action = self.env.action_space.sample()
```
✅ **Correct**: Random exploration for buffer filling

**Learning Phase** (steps 10,001+):
```python
# Select action with exploration noise
action = self.agent.select_action(state, noise=expl_noise)

# Store transition
self.agent.replay_buffer.add(state, action, next_state, reward, done_bool)

# Train networks
if t >= start_timesteps:
    self.agent.train()
```
✅ **Correct**: TD3 training with exploration noise

### Evaluation Function (lines 749-801)

**Current Implementation**:
```python
def evaluate(self) -> dict:
    for episode in range(self.num_eval_episodes):
        obs_dict = self.env.reset()  # Uses MAIN env, not separate eval_env
        state = self.flatten_dict_obs(obs_dict)
        episode_reward = 0
        episode_length = 0
        done = False
        max_eval_steps = self.max_timesteps  # BUG: Uses training max_timesteps!

        while not done and episode_length < max_eval_steps:
            # Deterministic action (no noise)
            action = self.agent.select_action(state, noise=0.0)
            next_obs_dict, reward, done, truncated, info = self.env.step(action)
            ...
```

❌ **BUGS IDENTIFIED**:

1. **Uses training environment**: Should use separate eval environment to avoid interference
2. **max_eval_steps = max_timesteps**: Uses 30,000 steps as episode timeout!
   - Should use `max_episode_steps` from config (1,000 steps)
   - This explains why eval episodes are so short (88 steps) - they all fail before timeout

3. **No separation from training**: Eval resets the same environment used for training
   - May affect training state
   - NPC count mismatch issue

4. **info dict not populated**: `info.get('success', 0)` likely returns 0 always
   - Environment may not be setting 'success' flag
   - Explains 0.0% success rate

---

## Recommendations

### 🔴 **CRITICAL: Fix Evaluation Function**

**Priority**: IMMEDIATE

**Changes needed**:
1. Use `max_episode_steps` from config, not `max_timesteps`
2. Create separate evaluation environment (don't reuse training env)
3. Fix scenario configuration (ensure eval uses correct NPC count)
4. Verify `info` dict contains 'success', 'collision_count' keys

**Code fix**:
```python
def evaluate(self) -> dict:
    # Create temporary eval environment with same config
    eval_env = CARLANavigationEnv(
        carla_config_path=self.carla_config,
        td3_config_path=self.agent_config,
        training_config_path="config/training_config.yaml",
    )

    max_eval_steps = self.agent_config.get("training", {}).get("max_episode_steps", 1000)

    for episode in range(self.num_eval_episodes):
        obs_dict = eval_env.reset()  # Use eval_env
        ...
        while not done and episode_length < max_eval_steps:  # Use correct timeout
            ...

    eval_env.close()  # Clean up
    return {...}
```

### 🟠 **HIGH: Address Learning Regression**

**Priority**: HIGH

**The agent learned to move (steps 11,200-11,500) but then forgot (steps 11,600+)**

**Possible solutions**:

#### Option 1: Reduce Learning Rate
```yaml
# In td3_config.yaml
training:
  actor_lr: 1e-4  # Currently 3e-4, try reducing
  critic_lr: 1e-4  # Currently 3e-4, try reducing
```
**Rationale**: Slower learning = more stable convergence

#### Option 2: Increase Exploration Noise
```yaml
training:
  expl_noise: 0.2  # Currently 0.1, increase to explore more
```
**Rationale**: More exploration → less likely to get stuck

#### Option 3: Strengthen Movement Incentive
```yaml
reward:
  weights:
    efficiency: 2.0  # Currently 1.0, DOUBLE the weight
    progress: 10.0  # Currently 5.0, DOUBLE the weight
```
**Rationale**: Make efficiency penalty stronger, progress reward stronger

#### Option 4: Re-introduce Stopping Penalty (CAREFULLY)
```python
# In reward_functions.py, _calculate_safety_reward
if velocity < 0.5 and distance_to_goal > 5.0:
    if not collision_detected and not offroad_detected:
        # Only penalize stopping if NO obstruction
        safety += -0.5  # Gentler than before (-1.0)
```
**Rationale**: Mild penalty for unnecessary stopping

#### Option 5: Curriculum Learning
- Start with **higher exploration noise** (0.3) for first 20k steps
- Gradually reduce to 0.1 over time
- Forces agent to keep exploring even after initial success

### 🟡 **MEDIUM: Fix Configuration Issues**

**Priority**: MEDIUM

1. **Fix scenario configuration**:
   ```yaml
   # In carla_config.yaml
   scenarios:
     - id: 0
       npcs: 20
       difficulty: easy
   ```

2. **Install CARLA PythonAPI properly**:
   ```bash
   export PYTHONPATH=/path/to/CARLA/PythonAPI/carla:$PYTHONPATH
   export PYTHONPATH=/path/to/CARLA/PythonAPI/carla/agents:$PYTHONPATH
   ```

3. **Verify waypoint manager**:
   - Check if legacy waypoint manager provides sufficient navigation guidance
   - Test waypoint_reached and goal_reached flags

### 🟢 **LOW: Extended Training**

**Priority**: LOW (after fixing above issues)

**30,000 steps may be insufficient** for TD3 to converge on complex vision-based driving

**From literature**:
- Pérez-Gil et al. (2022): Trained for **1 million timesteps**
- Fujimoto et al. (2018): TD3 benchmarks use **1-3 million timesteps**
- Elallid et al. (2023): Used **500k timesteps** for intersection navigation

**Recommendation**: After fixing issues, train for **100k-500k steps**

---

## Testing Plan for Checkpoint (td3_scenario_0_step_10000.pth)

### Test 1: Load and Evaluate Checkpoint

**Purpose**: Verify checkpoint loading and deterministic evaluation

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

python3 scripts/evaluate_td3.py \
  --checkpoint data/checkpoints/td3_scenario_0_step_10000.pth \
  --scenario 0 \
  --num-episodes 20 \
  --output-dir data/evaluation_step_10000 \
  --seed 42
```

**Expected results**:
- Agent should exhibit **random-like behavior** (checkpoint from end of exploration phase)
- Episodes should be short (lane invasions, collisions)
- Average reward: ~-463 (same as step 10,000 eval)
- Success rate: 0.0%

**Why this checkpoint?**: It's the **last checkpoint before learning started**. Provides baseline for comparison.

### Test 2: Visualize Agent Behavior (Debug Mode)

**Purpose**: See what the agent is actually doing

```bash
python3 scripts/visualize_agent.py \
  --checkpoint data/checkpoints/td3_scenario_0_step_10000.pth \
  --scenario 0 \
  --num-episodes 5 \
  --render \
  --save-video data/videos/agent_step_10000
```

**Expected observations**:
- Vehicle mostly stationary or drifting
- Random steering and throttle/brake commands
- Frequent lane invasions

### Test 3: Compare with Random Policy

**Purpose**: Verify the checkpoint performs no better than random (since it's end of exploration)

```bash
python3 scripts/evaluate_td3.py \
  --random-policy \
  --scenario 0 \
  --num-episodes 20 \
  --output-dir data/evaluation_random \
  --seed 42
```

**Expected**: Similar performance to checkpoint (both essentially random)

### Test 4: Resume Training from Checkpoint

**Purpose**: Verify checkpoint can be loaded and training resumed

```bash
python3 scripts/train_td3.py \
  --checkpoint data/checkpoints/td3_scenario_0_step_10000.pth \
  --scenario 0 \
  --max-timesteps 20000 \  # Continue from 10k to 20k
  --eval-freq 2000 \
  --checkpoint-freq 5000 \
  --device cpu
```

**Expected**:
- Training resumes from step 10,001
- Agent should show similar learning pattern (movement at ~11,200 steps = ~1,200 steps after resume)

---

## TD3 Implementation Checklist

Based on Fujimoto et al. (2018) TD3 paper and Elallid et al. (2023) CARLA implementation:

### ✅ **Core TD3 Components (Implemented)**

- [x] **Twin Critic Networks**: Q1 and Q2 (clipped double Q-learning)
- [x] **Delayed Policy Updates**: Actor updated less frequently than critics (policy_freq=2)
- [x] **Target Policy Smoothing**: Noise added to target actions
- [x] **Target Networks**: Separate target actor and critics with soft updates (tau=0.005)
- [x] **Replay Buffer**: Experience replay with capacity 1M
- [x] **Exploration Noise**: Gaussian noise during training (0.1)

### ✅ **Environment & State (Implemented)**

- [x] **Visual Input**: 4 stacked 84×84 grayscale frames (CNN feature extraction)
- [x] **Vector State**: Velocity, waypoints, heading, lateral deviation
- [x] **Action Space**: Continuous 2D (steering, throttle/brake)
- [x] **Reward Function**: Multi-component (efficiency, lane-keeping, comfort, safety, progress)

### ⚠️ **Issues Identified**

- [ ] **Evaluation Function**: Uses wrong max_steps, no separate env, info dict issues
- [ ] **Scenario Configuration**: Mismatch between config and execution (20 vs 50 NPCs)
- [ ] **Learning Regression**: Agent learns then forgets (step 11,200 → 11,600)
- [ ] **Stopping Penalty**: Removed, may have created "do nothing" local minimum

### ❓ **Unknown (Needs Verification)**

- [ ] **Waypoint Quality**: Legacy waypoint manager vs GlobalRoutePlanner
- [ ] **Progress Rewards**: Are waypoint_reached/goal_reached flags working?
- [ ] **CNN Features**: Are visual features informative enough for control?

---

## Conclusions

### What Worked ✅

1. **TD3 implementation is fundamentally correct**
   - All three core mechanisms present (twin critics, delayed updates, target smoothing)
   - Network architectures match paper specifications
   - Replay buffer and training loop properly implemented

2. **Agent successfully learned to move** (steps 11,200-11,500)
   - Transitioned from 0.0 km/h to 5-11 km/h
   - Achieved first positive reward (+0.72 at step 11,400)
   - Demonstrates TD3 is capable of learning from visual input

3. **Reward function correctly implements literature**
   - Efficiency reward matches Pérez-Gil et al. (2022)
   - Units consistent (m/s throughout)
   - Velocity gating prevents "stationary optimization"

### What Didn't Work ❌

1. **Agent regressed after initial learning**
   - Learned to move, then unlearned
   - Stuck at 0.0 km/h for 1,700 steps (11,600-13,300)
   - Suggests learning instability or local minimum

2. **Evaluation function has multiple bugs**
   - Wrong timeout value (30,000 instead of 1,000)
   - Reuses training environment
   - info dict may not be populated correctly

3. **Configuration mismatches**
   - NPC count discrepancy (20 configured, 50 used)
   - Scenario system not working properly

### Next Steps (Priority Order)

1. **🔴 Fix evaluation function** (implementation bugs)
2. **🟠 Address learning regression** (try all 5 solutions listed above)
3. **🟡 Fix configuration issues** (NPC count, scenarios, PYTHONPATH)
4. **🟢 Extended training** (100k-500k steps after fixes)
5. **🟢 Comprehensive evaluation** (compare TD3 vs DDPG vs baseline)

### Expected Timeline

- **Immediate** (1-2 days): Fix bugs, restart training
- **Short-term** (1 week): Train to 50k-100k steps, verify learning stability
- **Medium-term** (2-3 weeks): Full training (500k steps), comprehensive evaluation
- **Long-term** (1 month): Paper experiments, DDPG comparison, statistical analysis

---

## References

1. Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
2. Elallid et al. (2023): "Deep Reinforcement Learning for Autonomous Vehicle Intersection Navigation"
3. Pérez-Gil et al. (2022): "Deep Reinforcement Learning based control for Autonomous Vehicles in CARLA"
4. CARLA Documentation: https://carla.readthedocs.io/en/latest/

---

**Generated**: 2025-10-26
**Author**: Training Analysis System
**Checkpoint**: td3_scenario_0_step_10000.pth
**Next Review**: After bug fixes and training restart
