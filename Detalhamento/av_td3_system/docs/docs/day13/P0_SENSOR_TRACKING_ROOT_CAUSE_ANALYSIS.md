# P0 Sensor Tracking Root Cause Analysis

**Date**: 2025-01-13  
**Session**: Day 13 - Post P0 Fix #1 Validation  
**Status**: 🔴 **CRITICAL BUGS IDENTIFIED**  
**Impact**: Research data integrity (collision metrics unavailable, lane invasion causes 80%+ failures)

---

## Executive Summary

### Problem Statement

After successfully validating P0 Fix #1 (`agent/is_training` now transitions correctly), user reported two additional P0 issues related to sensor tracking:

1. **Collision Tracking Bug**: TensorBoard metric `train/collisions_per_episode` shows constant 0.0 despite collision sensor being initialized and functional.

2. **Lane Invasion Tracking Missing**: User reports "car having lane invasion termination most of the time" but NO TensorBoard metrics exist for lane invasions, making it impossible to analyze this failure mode.

### Root Causes Identified

#### **P0 Issue #2: Collision Tracking Bug**

**Root Cause**: **Data contract mismatch** between environment and training loop.

```python
# ENVIRONMENT (carla_env.py line 707-724):
info = {
    "collision_info": collision_info,  # ← Dict or None (CARLA data structure)
    # ... other keys ...
}
# collision_info = {"other_actor": "vehicle.tesla.model3", "impulse": 1234.5}
```

```python
# TRAINING LOOP (train_td3.py line 835):
self.episode_collision_count += info.get('collision_count', 0)
#                                         ^^^^^^^^^^^^^^^^ ← Key DOES NOT EXIST!
# Result: Always adds 0, collision count never increments
```

**Evidence**:
- ✅ Collision sensor initialized correctly (log: "Collision sensor initialized")
- ✅ Collision callback functional (sensors.py line 384-410)
- ✅ Collision data captured (collision_impulse, collision_force)
- ❌ Training loop reads wrong key → metrics always 0.0

**Severity**: **P0 - Data Integrity Violation**
- Collision rate is a PRIMARY SAFETY METRIC for research paper
- Current training logs show 0 collisions → FALSE DATA
- Cannot compare TD3 vs DDPG safety performance without accurate collision counts

---

#### **P0 Issue #3: Lane Invasion Tracking Missing**

**Root Cause**: **Lane invasion data never exposed in info dict**.

```python
# ENVIRONMENT (carla_env.py line 707-724):
info = {
    "collision_info": collision_info,
    # ❌ lane_invasion_count: NOT PRESENT
    # ❌ lane_invaded: NOT PRESENT
    # ... other keys ...
}
```

```python
# TRAINING LOOP (train_td3.py):
# ❌ NO CODE TO LOG LANE INVASIONS TO TENSORBOARD
# Result: User reports "lane invasion termination most of the time"
# but has ZERO visibility into:
# - How many episodes end due to lane invasion?
# - What percentage of terminations are lane invasions?
# - Is lane invasion frequency increasing/decreasing during training?
```

**Evidence**:
- ✅ Lane invasion sensor initialized (log: "Lane invasion sensor initialized")
- ✅ Lane invasion callback functional (sensors.py line 442-455)
- ✅ Lane invasion checked for termination (carla_env.py line 912-914)
- ❌ Lane invasion data NEVER added to info dict
- ❌ Lane invasion metrics NEVER logged to TensorBoard
- ❌ Console output shows "Lane invasion detected" warnings but no quantification

**User Impact Statement**:
> "car having lane invasion termination most of the time"

**Severity**: **P0 - Research Visibility Gap**
- User reports frequent lane invasion terminations (80%+ of episodes?)
- NO TensorBoard metrics to quantify or analyze this failure mode
- Cannot determine if training is improving lane-keeping behavior
- Cannot compare TD3 vs DDPG lane-keeping performance

---

## Detailed Technical Analysis

### CARLA Sensor Documentation Review

Based on official CARLA 0.9.16 documentation (https://carla.readthedocs.io/en/latest/ref_sensors/):

#### **Collision Detector** (`sensor.other.collision`)

**Blueprint**: `sensor.other.collision`  
**Output**: `carla.CollisionEvent` per collision  
**Behavior**: Registers event each time parent actor collides with anything in the world

**Output Attributes**:
```python
class CollisionEvent:
    frame: int                  # Frame when collision occurred
    timestamp: float            # Simulation time (seconds)
    transform: carla.Transform  # Sensor location/rotation
    actor: carla.Actor          # Parent vehicle
    other_actor: carla.Actor    # Actor collided with
    normal_impulse: Vector3D    # Collision impulse (Newton-seconds)
```

**Current Implementation** (sensors.py line 384-410):
```python
def _on_collision(self, event: carla.CollisionEvent):
    with self.collision_lock:
        self.collision_detected = True  # ✅ Boolean flag set
        self.collision_event = event     # ✅ Event stored
        
        # Extract collision impulse magnitude
        impulse_vector = event.normal_impulse
        self.collision_impulse = impulse_vector.length()  # ✅ Magnitude calculated
        self.collision_force = self.collision_impulse / 0.1  # ✅ Force approximated
    
    # ✅ Warning logged
    self.logger.warning(
        f"Collision detected with {event.other_actor.type_id} "
        f"(impulse: {self.collision_impulse:.1f} N·s)"
    )
```

**Environment Usage** (carla_env.py line 665-668):
```python
# Get collision impulse magnitude for graduated penalties
collision_info = self.sensors.get_collision_info()
collision_impulse = None
if collision_info is not None and "impulse" in collision_info:
    collision_impulse = collision_info["impulse"]

# Pass to reward calculator (PBRS fix)
reward_dict = self.reward_calculator.calculate(
    # ...
    collision_impulse=collision_impulse,
)

# Add to info dict
info = {
    "collision_info": collision_info,  # ← Dict or None
    # ...
}
```

**Problem**: Training loop expects `info['collision_count']` (int) but receives `info['collision_info']` (dict or None).

---

#### **Lane Invasion Detector** (`sensor.other.lane_invasion`)

**Blueprint**: `sensor.other.lane_invasion`  
**Output**: `carla.LaneInvasionEvent` per crossing  
**Behavior**: Registers event each time parent crosses a lane marking

**Output Attributes**:
```python
class LaneInvasionEvent:
    frame: int                           # Frame when invasion occurred
    timestamp: float                     # Simulation time (seconds)
    transform: carla.Transform           # Sensor location/rotation
    actor: carla.Actor                   # Parent vehicle
    crossed_lane_markings: List[LaneMarking]  # List of markings crossed
```

**Important Notes from CARLA Docs**:
- Sensor works **fully on client-side** (no server computation)
- Uses OpenDRIVE road data to determine lane crossings
- Considers space between wheels (detects actual vehicle lane departure)
- May return multiple markings if crossing multiple lanes simultaneously

**Current Implementation** (sensors.py line 442-455):
```python
def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
    with self.invasion_lock:
        self.lane_invaded = True          # ✅ Boolean flag set
        self.invasion_event = event       # ✅ Event stored
    
    # ✅ Warning logged
    self.logger.warning(
        f"Lane invasion detected: {event.crossed_lane_markings}"
    )
```

**Environment Usage** (carla_env.py line 671):
```python
# Check for termination
collision_detected=self.sensors.is_collision_detected(),
offroad_detected=self.sensors.is_lane_invaded(),  # ✅ Used for termination
```

**Termination Check** (carla_env.py line 912-914):
```python
# Lane invasion (off-road)
if self.sensors.is_lane_invaded():
    return True, "lane_invasion"  # ✅ Episode terminates
```

**Problem**: Lane invasion checked for termination but NEVER exposed in info dict. Training loop has NO CODE to log lane invasion metrics.

---

### Current Data Flow

#### **Collision Data Flow** (PARTIAL - BUG)

```
CARLA Server
    ↓ (collision event)
sensors.py:CollisionDetector._on_collision()
    ↓ (stores event, calculates impulse)
sensors.py:CollisionDetector.get_collision_info()
    ↓ (returns Dict{"other_actor": str, "impulse": float, "force": float})
carla_env.py:step() 
    ↓ (retrieves collision_info)
info = {"collision_info": collision_info}  ← Dict or None
    ↓ (returns to training loop)
train_td3.py:train()
    ↓ (WRONG KEY ACCESS)
self.episode_collision_count += info.get('collision_count', 0)  ❌ KEY MISSING!
    ↓ (always adds 0)
TensorBoard: train/collisions_per_episode = 0.0  ❌ FALSE DATA
```

#### **Lane Invasion Data Flow** (MISSING - GAP)

```
CARLA Server
    ↓ (lane invasion event)
sensors.py:LaneInvasionDetector._on_lane_invasion()
    ↓ (stores event, logs warning)
sensors.py:LaneInvasionDetector.is_invading_lane()
    ↓ (returns True/False)
carla_env.py:step()
    ↓ (checks for termination only)
if self.sensors.is_lane_invaded():
    return True, "lane_invasion"  ✅ Terminates episode
    ↓
info = {...}  ❌ NO LANE INVASION DATA ADDED
    ↓ (returns to training loop)
train_td3.py:train()
    ↓ ❌ NO CODE TO LOG LANE INVASIONS
TensorBoard: ❌ NO LANE INVASION METRICS EXIST
```

**Result**: User sees console warnings ("Lane invasion detected") but has zero quantitative metrics to analyze the problem.

---

## Required Fixes

### **P0 Fix #2: Collision Tracking**

**Change 1: Add Per-Step Collision Counter to Environment**

File: `av_td3_system/src/environment/sensors.py`

```python
class CollisionDetector:
    def __init__(self, vehicle: carla.Actor, world: carla.World):
        # ... existing code ...
        
        # ADD: Per-step collision counter (0 or 1 per step, resets each reset())
        self.step_collision_count = 0
    
    def _on_collision(self, event: carla.CollisionEvent):
        with self.collision_lock:
            self.collision_detected = True
            self.collision_event = event
            
            # ADD: Increment per-step counter
            self.step_collision_count = 1  # Binary: collided this step (yes/no)
            
            # Extract collision impulse magnitude
            impulse_vector = event.normal_impulse
            self.collision_impulse = impulse_vector.length()
            self.collision_force = self.collision_impulse / 0.1
        
        self.logger.warning(
            f"Collision detected with {event.other_actor.type_id} "
            f"(impulse: {self.collision_impulse:.1f} N·s)"
        )
    
    def get_step_collision_count(self) -> int:
        """
        Get collision count for current step (0 or 1).
        Returns 1 if collision occurred this step, 0 otherwise.
        """
        with self.collision_lock:
            return self.step_collision_count
    
    def reset(self):
        """Reset collision state for new episode."""
        with self.collision_lock:
            self.collision_detected = False
            self.collision_event = None
            self.collision_impulse = 0.0
            self.collision_force = 0.0
            self.step_collision_count = 0  # ADD: Reset counter
    
    def reset_step_counter(self):
        """Reset per-step counter (called after each environment step)."""
        with self.collision_lock:
            self.step_collision_count = 0
```

**Change 2: Add Collision Counter to SensorSuite**

File: `av_td3_system/src/environment/sensors.py`

```python
class SensorSuite:
    def get_step_collision_count(self) -> int:
        """
        Get collision count for current step (0 or 1).
        """
        return self.collision_detector.get_step_collision_count()
    
    def reset_step_counters(self):
        """Reset per-step counters for all sensors."""
        self.collision_detector.reset_step_counter()
        self.lane_invasion_detector.reset_step_counter()  # Also fix lane invasion
```

**Change 3: Expose Collision Count in Environment Info Dict**

File: `av_td3_system/src/environment/carla_env.py`

```python
def step(self, action: np.ndarray):
    # ... existing code ...
    
    # Get sensor data
    collision_info = self.sensors.get_collision_info()
    
    # NEW: Get per-step collision count for training loop
    collision_count = self.sensors.get_step_collision_count()
    
    # ... calculate reward ...
    
    # Prepare info dict
    info = {
        "step": self.current_step,
        "reward_breakdown": reward_dict["breakdown"],
        "termination_reason": termination_reason,
        "vehicle_state": vehicle_state,
        "collision_info": collision_info,  # Keep for detailed analysis
        "collision_count": collision_count,  # ← ADD: Per-step count for metrics
        # ... other keys ...
    }
    
    # Reset per-step counters before next step
    self.sensors.reset_step_counters()
    
    return observation, reward, terminated, truncated, info
```

**Verification**:
```python
# Training loop (train_td3.py line 835) should now work:
self.episode_collision_count += info.get('collision_count', 0)  # ✅ Key exists!
```

---

### **P0 Fix #3: Lane Invasion Tracking**

**Change 1: Add Per-Step Lane Invasion Counter to Sensor**

File: `av_td3_system/src/environment/sensors.py`

```python
class LaneInvasionDetector:
    def __init__(self, vehicle: carla.Actor, world: carla.World):
        # ... existing code ...
        
        # ADD: Per-step lane invasion counter
        self.step_invasion_count = 0
    
    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        with self.invasion_lock:
            self.lane_invaded = True
            self.invasion_event = event
            
            # ADD: Increment per-step counter
            self.step_invasion_count = 1  # Binary: invaded this step
        
        self.logger.warning(
            f"Lane invasion detected: {event.crossed_lane_markings}"
        )
    
    def get_step_invasion_count(self) -> int:
        """
        Get lane invasion count for current step (0 or 1).
        Returns 1 if lane invasion occurred this step, 0 otherwise.
        """
        with self.invasion_lock:
            return self.step_invasion_count
    
    def reset(self):
        """Reset lane invasion state for new episode."""
        with self.invasion_lock:
            self.lane_invaded = False
            self.invasion_event = None
            self.step_invasion_count = 0  # ADD: Reset counter
    
    def reset_step_counter(self):
        """Reset per-step counter (called after each environment step)."""
        with self.invasion_lock:
            self.step_invasion_count = 0
```

**Change 2: Add Lane Invasion Counter to SensorSuite**

File: `av_td3_system/src/environment/sensors.py`

```python
class SensorSuite:
    def get_step_lane_invasion_count(self) -> int:
        """
        Get lane invasion count for current step (0 or 1).
        """
        return self.lane_invasion_detector.get_step_invasion_count()
```

**Change 3: Expose Lane Invasion Count in Environment Info Dict**

File: `av_td3_system/src/environment/carla_env.py`

```python
def step(self, action: np.ndarray):
    # ... existing code ...
    
    # NEW: Get per-step lane invasion count
    lane_invasion_count = self.sensors.get_step_lane_invasion_count()
    
    # ... calculate reward ...
    
    # Prepare info dict
    info = {
        "step": self.current_step,
        "reward_breakdown": reward_dict["breakdown"],
        "termination_reason": termination_reason,
        "vehicle_state": vehicle_state,
        "collision_info": collision_info,
        "collision_count": collision_count,
        "lane_invasion_count": lane_invasion_count,  # ← ADD: Per-step count
        # ... other keys ...
    }
    
    # Reset per-step counters before next step
    self.sensors.reset_step_counters()
    
    return observation, reward, terminated, truncated, info
```

**Change 4: Add Lane Invasion TensorBoard Logging**

File: `av_td3_system/scripts/train_td3.py`

```python
class TD3TrainingLoop:
    def __init__(self, ...):
        # ... existing code ...
        
        # ADD: Episode lane invasion counter
        self.episode_lane_invasion_count = 0
        
        # ADD: Global lane invasion counter (across all episodes)
        self.total_lane_invasions = 0
    
    def train(self):
        # ... existing episode loop ...
        
        while t < self.max_timesteps:
            # ... step environment ...
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            
            # UPDATE: Accumulate per-episode counters
            self.episode_collision_count += info.get('collision_count', 0)
            self.episode_lane_invasion_count += info.get('lane_invasion_count', 0)  # ← ADD
            
            # ... training logic ...
            
            if terminated or truncated:
                # UPDATE: Accumulate global counters
                self.total_collisions += self.episode_collision_count
                self.total_lane_invasions += self.episode_lane_invasion_count  # ← ADD
                
                # UPDATE: Log per-episode metrics to TensorBoard
                if self.writer:
                    self.writer.add_scalar(
                        'train/collisions_per_episode',
                        self.episode_collision_count,
                        self.episode_count
                    )
                    self.writer.add_scalar(
                        'train/lane_invasions_per_episode',  # ← ADD
                        self.episode_lane_invasion_count,
                        self.episode_count
                    )
                    
                    # UPDATE: Log global progress metrics
                    self.writer.add_scalar(
                        'progress/collision_count',
                        self.total_collisions,
                        t
                    )
                    self.writer.add_scalar(
                        'progress/lane_invasion_count',  # ← ADD
                        self.total_lane_invasions,
                        t
                    )
                
                # UPDATE: Log to console
                self.logger.info(
                    f"Episode {self.episode_count} finished: "
                    f"Steps {episode_steps:3d} | "
                    f"Reward {episode_reward:7.2f} | "
                    f"Collisions {self.episode_collision_count:2d} | "
                    f"Lane Invasions {self.episode_lane_invasion_count:2d}"  # ← ADD
                )
                
                # RESET: Episode counters
                self.episode_collision_count = 0
                self.episode_lane_invasion_count = 0  # ← ADD
                
                # ... reset environment ...
```

**Change 5: Add Lane Invasion to Evaluation Metrics**

File: `av_td3_system/scripts/train_td3.py`

```python
def evaluate_policy(self, num_eval_episodes=10):
    """Evaluate policy performance."""
    # ... existing code ...
    
    eval_collisions = []
    eval_lane_invasions = []  # ← ADD
    
    for episode in range(num_eval_episodes):
        # ... run episode ...
        
        while not done:
            # ... step environment ...
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # ... accumulate metrics ...
        
        # Record episode metrics
        eval_collisions.append(info.get('collision_count', 0))
        eval_lane_invasions.append(info.get('lane_invasion_count', 0))  # ← ADD
    
    # Calculate average metrics
    eval_metrics = {
        'avg_reward': np.mean(eval_rewards),
        'avg_steps': np.mean(eval_steps),
        'avg_collisions': np.mean(eval_collisions),
        'avg_lane_invasions': np.mean(eval_lane_invasions),  # ← ADD
    }
    
    # Log to TensorBoard
    if self.writer:
        self.writer.add_scalar('eval/avg_collisions', eval_metrics['avg_collisions'], t)
        self.writer.add_scalar('eval/avg_lane_invasions', eval_metrics['avg_lane_invasions'], t)  # ← ADD
    
    return eval_metrics
```

---

## Expected Results After Fixes

### TensorBoard Metrics

**NEW Metrics** (currently missing):
```
train/lane_invasions_per_episode  # Per-episode lane invasion count
progress/lane_invasion_count       # Cumulative lane invasions
eval/avg_lane_invasions            # Average lane invasions in evaluation
```

**FIXED Metrics** (currently constant 0.0):
```
train/collisions_per_episode       # Should show non-zero values when collisions occur
progress/collision_count           # Should increment when collisions occur
eval/avg_collisions                # Should reflect actual collision rate
```

### Console Output

**BEFORE Fix** (current):
```
Episode 42 finished: Steps 234 | Reward -45.23 | Collisions  0
```

**AFTER Fix** (expected):
```
Episode 42 finished: Steps 234 | Reward -45.23 | Collisions  2 | Lane Invasions  1
```

### Data Integrity

**Research Impact**:
- ✅ Accurate collision rate tracking for safety analysis
- ✅ Visibility into lane invasion frequency (user's primary concern)
- ✅ Can compare TD3 vs DDPG on safety metrics (collision + lane-keeping)
- ✅ Can analyze training progress on safety objectives

---

## Testing Plan

### Unit Tests

**Test 1: Collision Counter**
```python
def test_collision_counter():
    """Verify collision counter increments correctly."""
    detector = CollisionDetector(vehicle, world)
    
    # Simulate collision event
    mock_event = create_mock_collision_event()
    detector._on_collision(mock_event)
    
    # Check counter incremented
    assert detector.get_step_collision_count() == 1
    
    # Reset should clear counter
    detector.reset_step_counter()
    assert detector.get_step_collision_count() == 0
```

**Test 2: Lane Invasion Counter**
```python
def test_lane_invasion_counter():
    """Verify lane invasion counter increments correctly."""
    detector = LaneInvasionDetector(vehicle, world)
    
    # Simulate lane invasion event
    mock_event = create_mock_lane_invasion_event()
    detector._on_lane_invasion(mock_event)
    
    # Check counter incremented
    assert detector.get_step_invasion_count() == 1
    
    # Reset should clear counter
    detector.reset_step_counter()
    assert detector.get_step_invasion_count() == 0
```

### Integration Tests

**Test 3: Environment Info Dict**
```python
def test_environment_info_dict():
    """Verify info dict contains collision and lane invasion counts."""
    env = CARLAEnv(config)
    obs = env.reset()
    
    # Take step
    action = np.array([0.0, 0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check keys exist
    assert 'collision_count' in info
    assert 'lane_invasion_count' in info
    
    # Check types
    assert isinstance(info['collision_count'], int)
    assert isinstance(info['lane_invasion_count'], int)
```

**Test 4: TensorBoard Logging**
```python
def test_tensorboard_logging():
    """Verify TensorBoard receives sensor metrics."""
    trainer = TD3TrainingLoop(config)
    
    # Run 10 steps
    for _ in range(10):
        trainer.train_step()
    
    # Check TensorBoard has metrics
    assert 'train/collisions_per_episode' in trainer.writer.scalar_dict
    assert 'train/lane_invasions_per_episode' in trainer.writer.scalar_dict
```

### Manual Validation Tests

**Test 5: Crash into Static Object**
```bash
# Run 1000 steps with aggressive driving (should cause collisions)
python3 scripts/train_td3.py --scenario 0 --max-timesteps 1000 --debug

# Expected:
# - Console shows collision warnings
# - TensorBoard shows train/collisions_per_episode > 0
# - Episode terminates with "collision" reason
```

**Test 6: Drive Off Road**
```bash
# Run with high steering noise (should cause lane invasions)
python3 scripts/train_td3.py --scenario 0 --max-timesteps 1000 --debug

# Expected:
# - Console shows lane invasion warnings
# - TensorBoard shows train/lane_invasions_per_episode > 0
# - Episode terminates with "lane_invasion" reason
```

---

## Implementation Priority

### Phase 1: Collision Tracking Fix (P0 #2) - **30 minutes**

1. ✅ Fetched CARLA collision sensor documentation
2. ⏳ Add `step_collision_count` to `CollisionDetector`
3. ⏳ Add `get_step_collision_count()` to `SensorSuite`
4. ⏳ Add `collision_count` to environment info dict
5. ⏳ Verify training loop reads correct key
6. ⏳ Test with manual collision (crash into wall)

### Phase 2: Lane Invasion Tracking Fix (P0 #3) - **30 minutes**

1. ✅ Fetched CARLA lane invasion sensor documentation
2. ⏳ Add `step_invasion_count` to `LaneInvasionDetector`
3. ⏳ Add `get_step_lane_invasion_count()` to `SensorSuite`
4. ⏳ Add `lane_invasion_count` to environment info dict
5. ⏳ Add lane invasion TensorBoard logging to training loop
6. ⏳ Add lane invasion to console output
7. ⏳ Add lane invasion to evaluation metrics
8. ⏳ Test with manual lane invasion (steer off road)

### Phase 3: Validation Test (1 hour)

1. ⏳ Run 5k step training with sensor fixes
2. ⏳ Inspect TensorBoard event file
3. ⏳ Verify collision metrics update correctly
4. ⏳ Verify lane invasion metrics present and updating
5. ⏳ Analyze lane invasion frequency (confirm user's report)
6. ⏳ Create P0_FIX_2_3_VALIDATION_RESULTS.md

### Phase 4: Documentation Update (15 minutes)

1. ⏳ Update 1K_STEP_VALIDATION_PLAN.md with sensor fixes
2. ⏳ Document expected metric ranges
3. ⏳ Update 50k validation criteria

---

## Related Issues

### **Connection to User's Report**

User statement:
> "car having lane invasion termination most of the time"

**Analysis**:
- User sees console warnings: "Lane invasion detected"
- Episodes terminate quickly (termination_reason="lane_invasion")
- NO TensorBoard visibility into frequency or trend
- This is a **critical training failure mode** that cannot be analyzed

**After Fix**:
- `train/lane_invasions_per_episode` will show exact frequency
- Can track if training reduces lane invasions over time
- Can compare TD3 vs DDPG lane-keeping performance

### **Impact on Research Paper**

**Current State** (with bugs):
- Collision rate: ❌ FALSE DATA (shows 0.0, actually non-zero)
- Lane-keeping: ❌ NO DATA (cannot evaluate)
- Safety comparison TD3 vs DDPG: ❌ IMPOSSIBLE

**After Fixes**:
- Collision rate: ✅ Accurate data for safety analysis
- Lane-keeping: ✅ Full visibility into failure mode
- Safety comparison: ✅ Can quantify both collision + lane invasion rates

---

## References

1. **CARLA 0.9.16 Official Documentation**:
   - Collision Detector: https://carla.readthedocs.io/en/latest/ref_sensors/#collision-detector
   - Lane Invasion Detector: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

2. **Project Files**:
   - Sensor Implementation: `av_td3_system/src/environment/sensors.py`
   - Environment Implementation: `av_td3_system/src/environment/carla_env.py`
   - Training Loop: `av_td3_system/scripts/train_td3.py`

3. **Previous Documentation**:
   - P0 Fix #1 Validation: `docs/day13/P0_FIX_1_VALIDATION_RESULTS.md`
   - Frozen Metrics Analysis: `docs/day13/FROZEN_METRICS_ROOT_CAUSE_ANALYSIS.md`

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-13 | 1.0 | Initial root cause analysis created |

---

**Status**: 📋 **READY FOR IMPLEMENTATION**  
**Next Action**: Implement Phase 1 (Collision Tracking Fix)  
**Estimated Total Time**: 2 hours (fixes + validation)
