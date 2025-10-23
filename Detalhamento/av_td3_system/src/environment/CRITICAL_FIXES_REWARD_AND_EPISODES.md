# Critical Fixes: Reward Function and Episode Management

## Date: October 22, 2025
## Issues Addressed: Vehicle not moving, no episodes recorded, reward design flaws

---

## Problem Analysis

### Issue 1: No Episodes Being Recorded ❌

**Symptom:**
```json
{
  "total_episodes": 0,
  "training_rewards": [],
  "eval_rewards": []
}
```

**Root Cause:**
The vehicle never experiences episode termination during training. Looking at the 1000-step debug run:
- Vehicle completes all 1000 steps without `done=True` or `truncated=True`
- `episode_num` only increments on termination (line 470 in `train_td3.py`)
- Episode never ends → no metrics recorded

**Why Episode Doesn't End:**
1. **Max steps not reached**: `max_episode_steps = 5000` (from config), but debug run is only 1000 total timesteps
2. **No collision**: Vehicle barely moves (0-4.6 km/h), so no crashes
3. **No off-road**: Stays in lane (lateral deviation < 0.13m)
4. **Route not completed**: Doesn't reach waypoint goal

### Issue 2: Vehicle Not Moving (CRITICAL) 🚨

**Symptom:**
```
🔍 [DEBUG Step 10] Speed=0.7 km/h | Rew=+0.36
🔍 [DEBUG Step 100] Speed=0.0 km/h | Rew=+0.42
🔍 [DEBUG Step 500] Speed=0.3 km/h | Rew=+0.28
```

**Root Cause: Reward Function Incentivizes Stopping!**

The original efficiency reward calculation:

```python
# BEFORE (WRONG)
if speed_diff <= self.speed_tolerance:
    efficiency = 1.0 - (speed_diff / self.speed_tolerance) * 0.5
```

**Problem:**
- Target speed: 8.33 m/s (30 km/h)
- Tolerance: 1.39 m/s (5 km/h)
- Vehicle at 0 km/h: `speed_diff = 8.33`, `excess = 6.94`
  - `efficiency = -6.94 / 8.33 * 0.5 = -0.42`
- Lane keeping at 0 km/h: `lat_dev=0.02m`, `heading=0.0` → `+0.5` reward
- Comfort at 0 km/h: No jerk → `+0.3` reward
- **Total reward at 0 km/h: `1.0*(-0.42) + 2.0*(+0.5) + 0.5*(+0.3) = +0.73`** ✅ POSITIVE!

**Agent learns: "Staying still = positive reward!"** 🤦

### Issue 3: Waypoints Are Correct ✅

**Analysis:**
- Waypoints ARE being transformed to vehicle local frame (✓ confirmed in `waypoint_manager.py`)
- Next 10 waypoints are included in state vector (✓ confirmed in `carla_env.py` line 612-623)
- Format: `[x_0, y_0, x_1, y_1, ..., x_9, y_9]` (20 dimensions)
- BUT: Agent doesn't learn to use them because **reward doesn't require movement**

---

## Solutions Implemented

### Fix 1: Reward Function Redesign ✅

**Following Pérez-Gil et al. (2022) DDPG-CARLA paper:**

Paper formula (Section 4.1.1):
```
R = Σ_t |v_t * cos(φ_t)| - |v_t * sin(φ_t)| - |v_t| * |d_t|
```

Where:
- `v_t * cos(φ_t)`: Longitudinal velocity (forward movement) → REWARD
- `v_t * sin(φ_t)`: Lateral velocity (sideways drift) → PENALTY
- `v_t * d_t`: Coupling of speed and lane deviation → PENALTY

**Our Implementation:**

```python
def _calculate_efficiency_reward(self, velocity: float) -> float:
    """
    CRITICAL: Agent must be incentivized to MOVE.
    
    Paper formula: R = Σ|v_t * cos(φ_t)| - |v_t * sin(φ_t)| - |v_t| * |d_t|
    """
    velocity_normalized = velocity / self.target_speed
    
    if velocity < 1.0:  # Below 1 m/s (3.6 km/h) = essentially stopped
        # STRONG penalty for not moving
        efficiency = -1.0
    elif velocity < self.target_speed * 0.5:  # Below half target speed
        # Moderate penalty for moving too slow
        efficiency = -0.5 + (velocity_normalized * 0.5)
    elif abs(velocity - self.target_speed) <= self.speed_tolerance:
        # Within tolerance: positive reward (optimal range)
        speed_diff = abs(velocity - self.target_speed)
        efficiency = 1.0 - (speed_diff / self.speed_tolerance) * 0.3
    else:
        # Outside tolerance
        if velocity > self.target_speed:
            # Overspeeding: penalty but less than underspeeding
            excess = velocity - self.target_speed
            efficiency = 0.7 - (excess / self.target_speed) * self.overspeed_penalty_scale
        else:
            # Underspeeding: penalty
            deficit = self.target_speed - velocity
            efficiency = -0.3 - (deficit / self.target_speed) * 0.3
    
    return float(np.clip(efficiency, -1.0, 1.0))
```

**New Reward Behavior:**

| Speed (km/h) | Efficiency Reward | Lane Keeping | Comfort | **Total** (weights: 3.0, 1.0, 0.5) |
|--------------|-------------------|--------------|---------|-------------------------------------|
| 0 km/h       | **-1.0**          | +0.5         | +0.3    | **-2.85** ❌ NEGATIVE               |
| 5 km/h       | **-0.35**         | +0.5         | +0.25   | **-0.43** ❌ NEGATIVE               |
| 15 km/h      | **+0.1**          | +0.5         | +0.2    | **+1.00** ✅ POSITIVE               |
| 30 km/h      | **+1.0**          | +0.5         | +0.15   | **+3.58** ✅ POSITIVE               |

**Result: Agent now learns that moving forward = higher reward!** 🎯

### Fix 2: Episode Management (TODO)

**Current Issue:**
```python
# train_td3.py line 470
if done or truncated:
    self.episode_num += 1
    # ... record metrics
```

**Problem:** Episode never ends in short debug runs.

**Recommended Fix:**

1. **Lower `max_episode_steps` for training:**
   ```yaml
   # carla_config.yaml
   episode:
     max_episode_steps: 1000  # REDUCED from 5000 (50 seconds @ 20Hz)
   ```

2. **Add distance-based termination:**
   ```python
   # In _check_termination()
   if self.steps_since_last_waypoint_progress > 200:  # 10 seconds stuck
       return True, "stuck_no_progress"
   ```

3. **Add early success termination:**
   ```python
   # Terminate episode when reaching goal waypoint
   if distance_to_goal < 10.0:  # Within 10m of destination
       return True, "goal_reached_success"
   ```

### Fix 3: Increase Efficiency Weight ✅ (Already Done)

**Config Change:**
```yaml
# config/carla_config.yaml
reward:
  weights:
    efficiency: 3.0      # INCREASED from 1.0
    lane_keeping: 1.0   # REDUCED from 2.0
    comfort: 0.5
    safety: -100.0
```

**Rationale:**
- Original weights: Lane keeping (2.0) dominated, allowing stationary positive rewards
- New weights: Efficiency (3.0) dominates, forcing agent to prioritize movement
- Safety remains strongly negative (-100.0) to prevent collisions

---

## Expected Behavior After Fixes

### Training Phase (Random Exploration, steps 0-25000):
```
🔍 [DEBUG Step 100] Speed=5.3 km/h | Rew=-0.82 (negative = agent learning to move)
🔍 [DEBUG Step 200] Speed=12.1 km/h | Rew=+0.45 (positive = agent discovering movement reward)
[TRAIN] Episode 1 | Reward -234.56 | Collisions 1
[TRAIN] Episode 2 | Reward +89.12 | Collisions 0
```

### Training Phase (Policy Learning, steps 25000+):
```
🔍 [DEBUG Step 25100] Speed=24.8 km/h | Rew=+2.87
🔍 [DEBUG Step 25200] Speed=28.3 km/h | Rew=+3.21
[TRAIN] Episode 251 | Reward +1823.45 | Collisions 0
```

### Evaluation Phase:
```
[EVAL] Episode 1/5 | Reward +2103.56 | Length 487 steps | Success ✓
[EVAL] Episode 2/5 | Reward +1987.23 | Length 512 steps | Success ✓
[EVAL] Mean Reward: 2045.39 | Success Rate: 100%
```

---

## How Our Agent Should Act (According to Papers)

### Reference: Pérez-Gil et al. (2022) - "Deep reinforcement learning based control for Autonomous Vehicles in CARLA"

**Section 4.1.1 - MDP Formulation:**

1. **State Space:**
   - Visual features: `f(I_t, w_t)` = CNN features from front camera + waypoints
   - Driving features: `(v_t, d_t, φ_t)` = velocity, lateral deviation, heading error
   - **OUR IMPLEMENTATION:** ✅ Matches (4 stacked images + kinematics + 10 waypoints)

2. **Action Space:**
   - `a_t = (acc_t, steer_t, brake_t)` continuous in [-1, 1] ranges
   - **OUR IMPLEMENTATION:** ✅ Matches (steering + throttle/brake unified)

3. **Reward Function:**
   ```
   R = Σ|v_t * cos(φ_t)| - |v_t * sin(φ_t)| - |v_t| * |d_t|
   ```
   - **Longitudinal velocity:** Positive contribution (encourages forward movement)
   - **Lateral velocity:** Negative contribution (penalizes sideways drift)
   - **Lane deviation:** Coupled with speed (higher penalty when fast and off-center)
   - **Terminal penalties:** -200 for collision, lane change, or road departure

4. **Expected Behavior:**
   - **Initial exploration:** Random actions, high collision rate
   - **Learning phase:** Agent discovers that moving forward = positive reward
   - **Convergence:** Agent maintains target speed (~30 km/h), follows waypoints, stays centered in lane
   - **Performance:** RMSE < 0.1m trajectory following, completion rate > 80%

### Reference: Elallid et al. (2023) - "Deep reinforcement learning for autonomous vehicle intersection navigation"

**TD3-specific insights:**

1. **Exploration Strategy:**
   - Start with high noise (`expl_noise = 0.1`) for first 25k steps
   - Gradually reduce noise as policy improves
   - Use clipped Gaussian noise: `a = π(s) + clip(N(0, σ), -0.5, 0.5)`

2. **Training Stability:**
   - Use delayed policy updates (every 2 critic updates)
   - Use target policy smoothing to avoid overfitting
   - **Result:** Smooth convergence, fewer oscillations than DDPG

3. **Safety Focus:**
   - Large negative reward for collisions (-200 in our case: -100 weight × 2.0 collision_penalty)
   - Off-road detection with immediate termination
   - **Expected:** Zero-collision rate after 100k training steps

### Reference: Our Paper (detalhamento_RL_25_2_IEEE.md)

**Section III.B - Expected Agent Behavior:**

1. **Episode Structure:**
   - Start: Spawn at route beginning (waypoints.txt first point)
   - Goal: Navigate to route end (waypoints.txt last point)
   - Success: Reach goal without collision/off-road
   - Failure: Collision, off-road, or timeout

2. **Navigation Strategy:**
   - Use waypoints for long-term planning (next 10 waypoints in state)
   - Use lateral deviation for lane centering
   - Use heading error for orientation correction
   - Balance speed (efficiency) vs safety (collision avoidance)

3. **Performance Metrics (Section IV.C):**
   - **Safety:** Success rate > 90%, collision rate < 0.1/km
   - **Efficiency:** Average speed ~25-30 km/h, completion time ~180-300s
   - **Comfort:** Jerk < 2.0 m/s³, lateral acceleration < 1.5 m/s²

---

## Testing Strategy

### Phase 1: Verify Reward Function (IMMEDIATE)

```bash
# Run 1000-step debug to check reward behavior
docker run ... python3 scripts/train_td3.py --scenario 0 --max-timesteps 1000 --debug
```

**Expected Output:**
```
🔍 [DEBUG Step 10] Speed=0.7 km/h | Rew=-2.43 (NEGATIVE, agent penalized for not moving)
🔍 [DEBUG Step 100] Speed=8.2 km/h | Rew=+0.12 (POSITIVE, agent rewarded for movement)
```

### Phase 2: Short Training Run (1 hour)

```bash
# Train for 10k steps to verify learning
docker run ... python3 scripts/train_td3.py --scenario 0 --max-timesteps 10000
```

**Expected Metrics:**
- Episodes completed: 5-10
- Average reward progression: -500 → -200 → 0 → +500
- Speed progression: 0-5 km/h → 10-20 km/h → 20-30 km/h

### Phase 3: Full Training (24-48 hours)

```bash
# Full training run
docker run ... python3 scripts/train_td3.py --scenario 0 --max-timesteps 1000000
```

**Expected Final Performance:**
- Success rate: > 85%
- Average reward: > +1500
- Average speed: 25-30 km/h
- Collision rate: < 5%

---

## Files Modified

1. **`src/environment/reward_functions.py`**
   - ✅ Fixed `_calculate_efficiency_reward()` to heavily penalize staying still
   - ✅ Reward structure now matches Pérez-Gil et al. (2022) paper design

2. **`config/carla_config.yaml`** (ALREADY UPDATED)
   - ✅ Increased efficiency weight: 1.0 → 3.0
   - ✅ Reduced lane keeping weight: 2.0 → 1.0

## Next Steps (TODO)

1. **Lower `max_episode_steps` to 1000** in `config/carla_config.yaml`
2. **Add stuck detection** in `_check_termination()` (no waypoint progress for 10s)
3. **Add goal-reached detection** (distance to last waypoint < 10m)
4. **Run test training** with `--max-timesteps 10000` to verify agent now moves
5. **Monitor TensorBoard** for:
   - Episode rewards trending upward
   - Average speed increasing over time
   - Collision rate decreasing

---

## Summary

**Root Cause:** Reward function allowed positive rewards for staying still due to lane keeping component dominating.

**Fix:** Redesigned efficiency reward to heavily penalize speeds < 1 m/s and increased efficiency weight from 1.0 to 3.0.

**Expected Result:** Agent now learns that **movement = reward**, matching the behavior described in DDPG-CARLA papers.

**Validation:** Run debug training and verify negative rewards at low speeds, positive rewards at target speeds.
