# TASK 1.6: Lane Invasion Penalty Inconsistency

**Created**: January 6, 2025  
**Status**: Investigation (Phase 1)  
**Priority**: Critical (Inconsistent reward signal violates TD3 requirements)  
**Related Issues**: Issue 1.6 from manual reward validation testing

## 📋 Problem Statement

Lane invasion warnings appear in logs but penalty is NEVER applied to reward. Sensor detects lane crossing but reward system doesn't penalize it, creating inconsistent safety signal.

### User's Observation

> "We are not been properly penalized with WARNING:src.environment.reward_functions:[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)... sometimes it shows WARNING:src.environment.sensors:Lane invasion detected but we are never penalized for it with the -1 for lane invasion, it does not show on the total reward count"

### Behavior Details

- **Symptom 1**: Sensor warning appears: `"WARNING:src.environment.sensors:Lane invasion detected: [<carla.libcarla.LaneMarking object>]"`
- **Symptom 2**: Reward warning NEVER appears: `"WARNING:src.environment.reward_functions:[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)"`
- **Symptom 3**: Lane invasion penalty doesn't show in total reward count
- **Expected**: Every sensor warning → reward penalty warning + reward impact
- **Actual**: Sensor detects, but reward ignores
- **TD3 Implication**: Critical inconsistency violates reward function invariance, corrupts policy gradient

---

## 🔍 Phase 1: Documentation Research

### 1.1 CARLA Lane Invasion Sensor Documentation

**Source**: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

**Critical Characteristics**:

1. **Event-Based**: "Registers an event each time its parent crosses a lane marking"
   - Fires callback `_on_lane_invasion()` when crossing detected
   - No continuous state updates
   - Event contains list of crossed markings

2. **Client-Side Processing**: "This sensor works fully on the client-side"
   - All processing in Python client code
   - Asynchronous callback execution
   - **CRITICAL**: Callback runs in separate thread from step() method

3. **Timing Implications**:
   - Sensor callback: Async, fires when CARLA detects crossing
   - Environment step(): Synchronous, runs at fixed timestep
   - **Race Condition Risk**: Callback may fire AFTER reward is calculated

### 1.2 Current Implementation Analysis

**File**: `av_td3_system/src/environment/sensors.py` (Lines 569-719)

**LaneInvasionDetector - Critical Components**:

```python
class LaneInvasionDetector:
    def __init__(self, vehicle, world):
        self.lane_invaded = False  # State flag
        self.step_invasion_count = 0  # Per-step counter ← CRITICAL
        self.invasion_lock = threading.Lock()  # Thread safety
        
    def _on_lane_invasion(self, event):
        """Async callback when sensor fires"""
        with self.invasion_lock:
            self.lane_invaded = True
            self.step_invasion_count = 1  # Set to 1 when invasion occurs
            
    def get_step_invasion_count(self) -> int:
        """Get invasion count for current step (0 or 1)"""
        with self.invasion_lock:
            return self.step_invasion_count
            
    def reset_step_counter(self):
        """Reset per-step counter (MUST be called after each step)"""
        with self.invasion_lock:
            self.step_invasion_count = 0  # Reset to 0
```

**CRITICAL MECHANISM**: Per-step counter pattern
- Counter set to `1` when callback fires
- Counter read by reward function
- Counter MUST be reset to `0` after each step
- If not reset → counter stuck at 1 → penalty applied every step OR never applied

### 1.3 Reward Function Implementation

**File**: `av_td3_system/src/environment/reward_functions.py` (Lines 865-886)

**Lane Invasion Penalty Code**:

```python
if lane_invasion_detected:
    safety += self.lane_invasion_penalty  # e.g., -5.0
    self.logger.warning(
        f"[SAFETY-LANE_INVASION] penalty={self.lane_invasion_penalty:.1f} "
        f"(crossed lane markings)"
    )
```

**Parameter**: `lane_invasion_detected` is a boolean passed to `_calculate_safety_reward()`

**CRITICAL QUESTIONS**:
1. Where does `lane_invasion_detected` come from in `carla_env.py`?
2. Is it reading `get_step_invasion_count()` correctly?
3. Is `reset_step_counter()` being called after reward calculation?

### 1.4 Gymnasium step() Timing Requirements

**Source**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

**Step Sequence**:
```python
def step(self, action):
    # 1. Apply action to environment
    # 2. Wait for simulation tick (sensor callbacks fire here)
    # 3. Read sensor states
    # 4. Calculate reward (must use current sensor data)
    # 5. Reset per-step flags
    # 6. Return (obs, reward, terminated, truncated, info)
```

**Critical Timing**:
- Sensors update asynchronously during tick
- Reward must read sensor state AFTER tick completes
- Per-step flags must reset AFTER reward reads them
- If reset too early → reward misses detection
- If not reset → detection persists incorrectly

### 1.5 Thread Safety Considerations

**From CARLA Docs**:
- Sensor callbacks execute in separate thread
- Python GIL provides some safety, but not sufficient
- Locks required for shared state (sensor does this correctly ✓)

**LaneInvasionDetector Thread Safety**:
- Uses `threading.Lock` for all state access ✓
- Callback writes: `_on_lane_invasion()` under lock ✓
- Reward reads: `get_step_invasion_count()` under lock ✓
- Counter reset: `reset_step_counter()` under lock ✓

**Conclusion**: Thread safety implemented correctly. Issue is NOT race condition in sensor class.

---

## 🎯 Root Cause Hypotheses

### Hypothesis 1: reset_step_counter() Never Called ⭐ **MOST LIKELY**

**Evidence**:
- Sensor warning appears → callback IS firing
- Reward warning never appears → flag never read as True
- `step_invasion_count` defaults to 0
- If callback fires AFTER reward reads counter → reads 0
- If counter never reset → should read 1 continuously (but doesn't)

**Contradiction Analysis**:
- If counter stuck at 1: Should penalize EVERY step after first invasion
- If counter always 0: Never penalizes (matches observation ✓)

**Most Likely Scenario**:
```python
# In carla_env.py step():
lane_invasion = self.sensors.lane_detector.get_step_invasion_count()  # Reads 0
reward = self.reward_fn.calculate(..., lane_invasion_detected=bool(lane_invasion))  # False
# MISSING: self.sensors.lane_detector.reset_step_counter()
# Callback fires HERE (after reward calculated, before next step)
# Counter set to 1, but too late for this step's reward
# Next step: Counter still 1, but... wait, why doesn't it read 1?
```

**Wait, This Doesn't Fully Explain It!**

If counter set to 1 and never reset, next step should read 1. Unless...

### Hypothesis 2: Wrong Method Called or Parameter Not Passed

**Possibility A**: Reading `lane_invaded` flag instead of `step_invasion_count`
```python
# WRONG:
lane_invasion = self.sensors.lane_detector.is_invading_lane()  # Returns bool from flag
# Correct flag cleared by recovery logic → always False after recovery
# But counter would still be 1!

# CORRECT:
lane_invasion = self.sensors.lane_detector.get_step_invasion_count()  # Returns 0 or 1
```

**Possibility B**: Parameter not passed to reward function
```python
# carla_env.py might be:
reward = self.reward_fn._calculate_safety_reward(
    collision_detected=...,
    offroad_detected=...,
    wrong_way=...,
    # lane_invasion_detected=...  ← MISSING?
)
# Defaults to False if not provided
```

### Hypothesis 3: Timing Issue - Callback Fires Too Late

**CARLA Synchronous Mode Timing**:
```
Step N:
  1. Apply action → CARLA tick starts
  2. Tick completes → world state updates
  3. Read sensors (immediate state like velocity, position)
  4. Calculate reward (uses sensor data)
  5. Callbacks fire (MAY OCCUR AFTER step() returns!)
  
Step N+1:
  1. Callbacks from N might still be pending
  2. Counter from N set AFTER N's reward calculated
  3. Counter reset (if called) BEFORE callback fires
  4. Counter always 0 when reward reads it
```

**CRITICAL INSIGHT**: Event-based sensors (lane invasion, collision) may have **delayed callbacks** compared to polling-based sensors (velocity, position).

**From CARLA Docs**:
> "This sensor works fully on the client-side"

This means processing is client-side but callback timing still controlled by CARLA tick cycle.

---

## 🔍 Investigation Plan (Phase 2)

### Task 1.6.1: Read carla_env.py reward calculation (15 min)

**File**: `av_td3_system/src/environment/carla_env.py` (around line 744)

**Questions to Answer**:
1. How is `lane_invasion_detected` obtained?
   - [ ] Uses `get_step_invasion_count()`?
   - [ ] Uses `is_invading_lane()`?
   - [ ] Uses `lane_invaded` flag directly?
   - [ ] Not passed at all (defaults to False)?

2. Is `reset_step_counter()` called?
   - [ ] Where in step() sequence?
   - [ ] Before or after reward calculation?
   - [ ] Called at all?

3. What is the exact step() execution order?
   ```
   [ ] Apply action
   [ ] CARLA tick
   [ ] Read sensor states
   [ ] Calculate reward
   [ ] Reset sensor counters
   [ ] Return reward
   ```

### Task 1.6.2: Add Diagnostic Logging (20 min)

**Locations to Add Logging**:

1. **In LaneInvasionDetector._on_lane_invasion()**:
   ```python
   import time
   def _on_lane_invasion(self, event):
       timestamp = time.time()
       with self.invasion_lock:
           self.step_invasion_count = 1
       self.logger.warning(
           f"[CALLBACK] Lane invasion at t={timestamp:.6f}, "
           f"counter set to 1, thread={threading.current_thread().name}"
       )
   ```

2. **In LaneInvasionDetector.get_step_invasion_count()**:
   ```python
   def get_step_invasion_count(self):
       with self.invasion_lock:
           count = self.step_invasion_count
       self.logger.debug(
           f"[READ] get_step_invasion_count() → {count}, "
           f"thread={threading.current_thread().name}"
       )
       return count
   ```

3. **In LaneInvasionDetector.reset_step_counter()**:
   ```python
   def reset_step_counter(self):
       with self.invasion_lock:
           old_val = self.step_invasion_count
           self.step_invasion_count = 0
       self.logger.debug(
           f"[RESET] reset_step_counter() {old_val} → 0, "
           f"thread={threading.current_thread().name}"
       )
   ```

4. **In carla_env.py step() method**:
   ```python
   # Before reward calculation
   lane_invasion = self.sensors.lane_detector.get_step_invasion_count()
   self.logger.debug(f"[STEP] Lane invasion count before reward: {lane_invasion}")
   
   # After reward calculation
   self.logger.debug(f"[STEP] Resetting lane invasion counter...")
   self.sensors.lane_detector.reset_step_counter()
   ```

### Task 1.6.3: Manual Test with Logging (15 min)

**Test Procedure**:
1. Start CARLA in synchronous mode
2. Spawn vehicle with lane invasion sensor
3. Drive normally (no invasion) for 5 steps
4. Deliberately cross lane marking
5. Drive normally (no invasion) for 5 more steps
6. Analyze logs

**Expected Log Pattern (IF WORKING CORRECTLY)**:
```
Step 0: [READ] get_step_invasion_count() → 0
Step 0: [STEP] Lane invasion count before reward: 0
Step 0: [STEP] Resetting lane invasion counter...
Step 0: [RESET] reset_step_counter() 0 → 0

Step 3: [CALLBACK] Lane invasion at t=123.456, counter set to 1
Step 3: [READ] get_step_invasion_count() → 1  ← SHOULD SEE THIS
Step 3: [STEP] Lane invasion count before reward: 1
Step 3: [REWARD] [SAFETY-LANE_INVASION] penalty=-5.0  ← SHOULD SEE THIS
Step 3: [STEP] Resetting lane invasion counter...
Step 3: [RESET] reset_step_counter() 1 → 0

Step 4: [READ] get_step_invasion_count() → 0
Step 4: [STEP] Lane invasion count before reward: 0
```

**Diagnosis Based on Logs**:
- If `[CALLBACK]` appears but `[READ]` always shows 0 → Timing issue (callback after read)
- If `[RESET]` never appears → `reset_step_counter()` not called
- If `[READ]` not in logs → `get_step_invasion_count()` not called
- If callback timestamp after reward timestamp → CARLA callback delay

### Task 1.6.4: Check CARLA Synchronous Mode Configuration (10 min)

**Verify Settings**:
```python
# In CARLA initialization
settings = world.get_settings()
settings.synchronous_mode = True  # MUST be True
settings.fixed_delta_seconds = 0.05  # Fixed timestep (20 Hz)
world.apply_settings(settings)

# Verify tick behavior
world.tick()  # Blocks until simulation advances one step
```

**Callback Timing in Sync Mode**:
- Callbacks SHOULD fire during `world.tick()`
- But may process asynchronously
- Need to verify callback completes BEFORE tick returns

---

## 📊 Expected Outcomes from Investigation

### Scenario 1: reset_step_counter() Not Called
**Symptoms**:
- Counter stuck at 1 after first invasion
- All subsequent steps should show penalty
- But user says "never penalized"
- **Contradiction**: Suggests counter always 0, not stuck at 1

### Scenario 2: get_step_invasion_count() Not Called
**Symptoms**:
- `[READ]` log never appears
- `lane_invasion_detected` always False
- No penalty ever applied ✓ (matches observation)

### Scenario 3: Callback Timing Issue
**Symptoms**:
- Callback timestamp after reward timestamp
- Counter set to 1 AFTER reward reads 0
- Counter reset to 0 immediately
- Net effect: Counter always 0 when read ✓ (matches observation)

### Scenario 4: Parameter Not Passed to Reward Function
**Symptoms**:
- Sensor correctly tracks invasions
- Env correctly reads counter
- But `lane_invasion_detected=...` not in reward function call
- Defaults to False
- No penalty applied ✓ (matches observation)

---

## 🔄 Potential Fixes (Pending Investigation)

### Fix Option 1: Ensure reset_step_counter() Called Correctly

```python
# In carla_env.py step() method:
def step(self, action):
    # Apply action
    control = self._action_to_control(action)
    self.vehicle.apply_control(control)
    
    # Tick simulation (callbacks fire during this)
    self.world.tick()
    
    # ⭐ CRITICAL: Small delay to ensure callbacks complete
    time.sleep(0.001)  # 1ms grace period for async callbacks
    
    # Read sensor states (AFTER callbacks complete)
    lane_invasion_count = self.sensors.lane_detector.get_step_invasion_count()
    
    # Calculate reward
    reward = self._calculate_reward(..., lane_invasion_detected=bool(lane_invasion_count))
    
    # Reset per-step counters (AFTER reward uses them)
    self.sensors.lane_detector.reset_step_counter()
    
    return obs, reward, done, truncated, info
```

### Fix Option 2: Use Persistent Flag Instead of Counter

```python
# Alternative: Don't use step counter, use persistent flag
# Read flag at reward time, clear after step
def step(self, action):
    # ... tick ...
    
    lane_invasion = self.sensors.lane_detector.is_invading_lane(
        lateral_deviation=self.vehicle_state.lateral_deviation,
        lane_half_width=self.vehicle_state.lane_half_width
    )
    
    reward = self._calculate_reward(..., lane_invasion_detected=lane_invasion)
    
    # Clear flag if vehicle returned to center
    # (handled inside is_invading_lane() with lateral deviation check)
    
    return ...
```

### Fix Option 3: Queue-Based Event System

```python
class LaneInvasionDetector:
    def __init__(self):
        self.invasion_queue = queue.Queue()  # Thread-safe queue
        
    def _on_lane_invasion(self, event):
        self.invasion_queue.put(event)  # Add to queue
        
    def get_step_invasions(self):
        """Get all invasions that occurred this step"""
        invasions = []
        while not self.invasion_queue.empty():
            invasions.append(self.invasion_queue.get())
        return invasions
```

---

## 📚 References

### Official Documentation
- **CARLA Lane Invasion Sensor**: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
- **CARLA Synchronous Mode**: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
- **Gymnasium Env.step()**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

### Related Files
- `av_td3_system/src/environment/sensors.py`: Lines 569-719 (LaneInvasionDetector)
- `av_td3_system/src/environment/reward_functions.py`: Lines 865-886 (Lane invasion penalty)
- `av_td3_system/src/environment/carla_env.py`: Line ~744 (Reward calculation)

### Key Concepts
- **Event-Based Sensors**: Fire callbacks asynchronously
- **Per-Step Counters**: Require manual reset after each step
- **Thread Safety**: Locks required for async callback state access
- **Synchronous Mode**: Callbacks fire during tick(), but timing not guaranteed

---

## 🔄 Status Log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2025-01-06 | Phase 1 | Complete | Documentation research finished. 4 hypotheses formed. |
| 2025-01-06 | Phase 2 | Pending | Investigation tasks defined (4 tasks, ~60 min total). |

---

**Next Action**: Begin Phase 2 - Investigation Task 1.6.1 (Read carla_env.py reward calculation section)
