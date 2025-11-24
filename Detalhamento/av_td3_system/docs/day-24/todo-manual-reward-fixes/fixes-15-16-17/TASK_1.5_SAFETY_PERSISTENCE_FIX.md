# TASK 1.5: Safety Reward Persistence After Recovery

**Created**: January 6, 2025  
**Status**: Investigation (Phase 1)  
**Priority**: High (Affects TD3 training quality)  
**Related Issues**: Issue 1.5 from manual reward validation testing

## 📋 Problem Statement

After vehicle recovers from sidewalk invasion or lane invasion, safety reward stays at negative value (e.g., `-0.607`) even when vehicle is stopped at center of correct lane. When vehicle starts progressing toward goal, it continues receiving negative safety reward for correct behavior after successful recovery.

### User's Observation

> "After recovering from sidewalk invasion (offroad_detected) or lane invasion (is_lane_invaded()), safety stays at -0.607 while stopped at center of correct lane. When we start having progress towards goal, we keep receiving negative safety reward for good behavior after recovery. The safety reward seem to be keeping some negative value cached."

### Behavior Details

- **Trigger**: Vehicle invades sidewalk OR crosses lane markings
- **Expected**: Safety returns to `0.0` after vehicle recovers to safe position
- **Actual**: Safety remains at `-0.607` (or similar negative value)
- **Impact**: Penalizes correct driving behavior after successful recovery
- **TD3 Implication**: Inconsistent reward signal violates Markov property, degrades Q-value estimates

---

## 🔍 Phase 1: Documentation Research

### 1.1 CARLA Lane Invasion Sensor Documentation

**Source**: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

**Key Findings**:

1. **Event-Based Operation**: "Registers an event each time its parent crosses a lane marking"
   - Fires **per crossing**, not continuous state
   - No automatic "return to lane" event
   - Client-side processing (important for timing)

2. **Output**: `carla.LaneInvasionEvent`
   - `crossed_lane_markings`: list of `carla.LaneMarking` objects
   - Event contains marking types but no state clearing mechanism

3. **Detection Method**: "Uses road data provided by OpenDRIVE description"
   - Based on space between wheels
   - May detect multiple markings simultaneously

4. **Important Note**: "This sensor works fully on the client-side"
   - Processing happens in client Python code
   - No server-side state management
   - Recovery logic must be implemented manually

**Implications for Issue 1.5**:
- CARLA provides NO automatic recovery detection
- `lane_invaded` flag persists until manually cleared
- Current recovery logic (lateral deviation check in `is_invading_lane()`) may not be executing
- Need to verify `is_invading_lane()` is called with proper parameters

### 1.2 Current Implementation Analysis

**File**: `av_td3_system/src/environment/sensors.py` (Lines 569-719)

**LaneInvasionDetector Implementation**:

```python
def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
    """Callback when lane invasion occurs (sets flag)"""
    with self.invasion_lock:
        self.lane_invaded = True  # ← Flag SET
        self.invasion_event = event
        self.step_invasion_count = 1

def is_invading_lane(self, lateral_deviation: float = None, lane_half_width: float = None) -> bool:
    """Check invasion status AND clear flag if recovered"""
    with self.invasion_lock:
        if lateral_deviation is not None and lane_half_width is not None:
            recovery_threshold = lane_half_width * 0.8  # 80% of lane width
            
            if abs(lateral_deviation) < recovery_threshold:
                self.lane_invaded = False  # ← Flag CLEARED (recovery logic)
                self.invasion_event = None
                
        return self.lane_invaded
```

**Recovery Logic**:
- Recovery threshold: 80% of lane half-width
- Example: 1.75m lane → half=0.875m → threshold=0.7m
- Vehicle at 0.5m deviation → clears flag ✓
- Vehicle at 0.85m deviation → keeps flag (still near edge)

**CRITICAL QUESTION**: Is `is_invading_lane()` being called with `lateral_deviation` and `lane_half_width` parameters in `carla_env.py`?

### 1.3 OffroadDetector Implementation

**File**: `av_td3_system/src/environment/sensors.py` (Lines ~890-1020)

**OffroadDetector.reset() Analysis**:

```python
def reset(self):
    """Reset offroad state for new episode."""
    with self.offroad_lock:
        self.is_offroad = False
        self.offroad_duration = 0.0
```

**CRITICAL OBSERVATION**: `reset()` only resets state for **new episode**, not after recovery!

**Recovery Logic Location**: Must be in `_check_offroad_state()` method which is called during `step()`.

### 1.4 Safety Reward Calculation

**File**: `av_td3_system/src/environment/reward_functions.py` (Lines 659-920)

**PBRS Proximity Penalty** (Lines 710-745):

```python
if distance_to_nearest_obstacle is not None:
    if distance_to_nearest_obstacle < 10.0:  # Only penalize within 10m range
        proximity_penalty = -1.0 / max(distance_to_nearest_obstacle, 0.5)
        safety += proximity_penalty
```

**Formula**: `proximity_penalty = -1.0 / max(distance, 0.5)`

**Distance → Penalty Mapping**:
- 10.0m → `-0.10`
- 5.0m  → `-0.20`
- 3.0m  → `-0.33`
- **1.65m → `-0.607`** ← User's observed value!
- 1.0m  → `-1.00`
- 0.5m  → `-2.00` (maximum)

**FINDING**: `-0.607` matches PBRS penalty at **approximately 1.65m distance** from nearest obstacle.

**Hypothesis**: `distance_to_nearest_obstacle` is not being updated correctly after obstacle clears, OR obstacle detector is caching stale data.

**Stopping Penalty** (Lines 893-909):

```python
if not collision_detected and not offroad_detected:
    if velocity < 0.5:  # Essentially stopped (< 1.8 km/h)
        stopping_penalty = -0.1  # Base penalty
        
        if distance_to_goal > 10.0:
            stopping_penalty += -0.4  # Total: -0.5 when far from goal
        elif distance_to_goal > 5.0:
            stopping_penalty += -0.2  # Total: -0.3 when moderately far
            
        safety += stopping_penalty
```

**Finding**: Stopping penalty is conditional on:
- NOT in collision
- NOT offroad
- Velocity < 0.5 m/s

This is CORRECT behavior - it shouldn't penalize stopping during recovery.

### 1.5 Gymnasium step() Requirements

**Source**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

**Key Requirements**:

1. **State Management**: "When the end of an episode is reached (terminated or truncated), it is necessary to call reset()"
   - Episode-level reset, not step-level
   - Our sensors have `reset()` for episodes ✓
   - Need step-level state clearing mechanism

2. **Return Values**:
   ```python
   observation, reward, terminated, truncated, info = env.step(action)
   ```
   - Reward must reflect current timestep only
   - No state persistence across steps unless justified

3. **Info Dict**: "Contains auxiliary diagnostic information (helpful for debugging, learning, and logging)"
   - Could log sensor state for debugging

**Implication**: Reward calculation must use **current frame state only**, not cached values from previous frames (unless PBRS, which is mathematically valid).

---

## 🎯 Root Cause Hypotheses

### Hypothesis 1: Obstacle Distance Not Updating After Recovery ⭐ **MOST LIKELY**

**Evidence**:
- `-0.607` precisely matches PBRS penalty at 1.65m distance
- PBRS formula is stateless (only depends on current distance)
- If `distance_to_nearest_obstacle` stuck at 1.65m → persistent `-0.607`

**Investigation Needed**:
1. Check how `distance_to_nearest_obstacle` is calculated in `carla_env.py`
2. Verify obstacle detection updates correctly after vehicle moves
3. Check if obstacle sensor caching stale detection results

**Files to Investigate**:
- `av_td3_system/src/environment/carla_env.py` (obstacle distance calculation)
- `av_td3_system/src/environment/sensors.py` (obstacle detection sensor)

### Hypothesis 2: LaneInvasionDetector Recovery Not Executing

**Evidence**:
- User mentions "lane invasion" as trigger
- `is_invading_lane()` has recovery logic (80% threshold)
- Recovery logic only executes if called with lateral deviation parameters

**Investigation Needed**:
1. Check if `carla_env.py` calls `is_invading_lane(lateral_deviation, lane_half_width)`
2. Verify `lateral_deviation` and `lane_half_width` calculated correctly
3. Add logging to recovery logic to confirm execution

**Files to Investigate**:
- `av_td3_system/src/environment/carla_env.py` (line ~744, reward calculation point)
- `av_td3_system/src/environment/sensors.py` (LaneInvasionDetector.is_invading_lane())

### Hypothesis 3: OffroadDetector Not Clearing State

**Evidence**:
- User mentions "sidewalk invasion (offroad_detected)" as trigger
- `reset()` only clears state for new episodes, not recovery
- Must have recovery logic in `_check_offroad_state()` or similar

**Investigation Needed**:
1. Read `OffroadDetector._check_offroad_state()` implementation
2. Verify offroad recovery conditions
3. Check if waypoint.lane_type correctly updates after recovery

**Files to Investigate**:
- `av_td3_system/src/environment/sensors.py` (OffroadDetector class, ~890-1020)

---

## 📝 Next Steps (Phase 2: Investigation)

### Investigation Tasks

1. **Task 1.5.1**: Read `carla_env.py` reward calculation section
   - Find where `_calculate_safety_reward()` is called
   - Check if `lateral_deviation` and `lane_half_width` provided to `is_invading_lane()`
   - Verify `distance_to_nearest_obstacle` calculation
   - **Estimated Time**: 15 minutes

2. **Task 1.5.2**: Read `OffroadDetector` full implementation
   - Understand `_check_offroad_state()` recovery logic
   - Verify waypoint.lane_type updates correctly
   - Check state clearing conditions
   - **Estimated Time**: 10 minutes

3. **Task 1.5.3**: Add debug logging to test hypothesis
   - Log `distance_to_nearest_obstacle` every step
   - Log lateral deviation and recovery threshold
   - Log obstacle detection updates
   - Run manual test: invade → recover → check logs
   - **Estimated Time**: 20 minutes

4. **Task 1.5.4**: Trace `-0.607` value origin
   - Calculate exact distance: `-0.607 = -1.0 / d` → `d ≈ 1.647m`
   - Confirm this matches user's scenario geometry
   - Identify which obstacle causing penalty
   - **Estimated Time**: 10 minutes

### Phase 2 Deliverables

- [ ] Root cause identified (obstacle distance, lane invasion, or offroad)
- [ ] Evidence documented with code line numbers
- [ ] Test case reproducing issue
- [ ] Logging output showing state persistence

**Total Estimated Time for Phase 2**: 55 minutes

---

## 📚 References

### Official Documentation

- **CARLA Lane Invasion Sensor**: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
- **CARLA Waypoint API**: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint
- **Gymnasium Env.step()**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

### Related Files

- `SYSTEMATIC_REWARD_FIXES_TODO.md`: Master plan for all reward fixes
- `TASK_1_OFFROAD_DETECTION_FIX.md`: Previous fix example (semantic mismatch pattern)
- `av_td3_system/src/environment/sensors.py`: Lines 569-719 (LaneInvasionDetector)
- `av_td3_system/src/environment/sensors.py`: Lines ~890-1020 (OffroadDetector)
- `av_td3_system/src/environment/reward_functions.py`: Lines 659-920 (Safety calculation)

### Key Formulas

**PBRS Proximity Penalty**:
```python
proximity_penalty = -1.0 / max(distance_to_nearest_obstacle, 0.5)
```

**Lane Recovery Threshold**:
```python
recovery_threshold = lane_half_width * 0.8  # 80% of half-width
if abs(lateral_deviation) < recovery_threshold:
    lane_invaded = False  # Clear flag
```

---

## 🔄 Status Log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2025-01-06 | Phase 1 | Complete | Documentation research finished. Hypotheses formed. |
| 2025-01-06 | Phase 2 | Pending | Investigation tasks defined. Ready to start. |

---

**Next Action**: Begin Phase 2 - Investigation Task 1.5.1 (Read carla_env.py reward calculation section)
