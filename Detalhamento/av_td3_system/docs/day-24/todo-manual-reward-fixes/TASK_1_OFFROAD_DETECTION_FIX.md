# Task 1: Fix Safety Penalty Persistence After Lane Recovery

**Status**: ✅ IMPLEMENTED  
**Date**: November 23, 2025  
**Priority**: CRITICAL (P0)  
**Category**: Reward Validation - Safety Reward Component

---

## Problem Statement

### User-Reported Issue
When manually controlling the vehicle:
1. Cross a lane marking → Receive "[SAFETY-OFFROAD] penalty=-10.0"
2. Return to lane center → **Penalty persists** (should clear)
3. Terminal spam continues: "[SAFETY-OFFROAD] penalty=-10.0"

### Expected Behavior
- Off-road penalty should apply **only when on grass, sidewalk, or non-drivable surface**
- Legal lane changes should **NOT** trigger off-road penalty
- Returning to lane center should **clear** the penalty

### Actual Behavior (Before Fix)
- **ANY** lane marking touch (including legal lane changes) triggered -10.0 "offroad" penalty
- Even WITH working recovery logic, penalty persisted inappropriately
- Legal driving maneuvers were being penalized as dangerous off-road events

---

## Root Cause Analysis

### Investigation Process (9 tool calls, ~30 minutes)

**Step 1: Understand CARLA Semantics**
- Fetched CARLA lane invasion sensor documentation
- **Discovery**: Lane invasion sensor fires on crossing **lane markings** (lines), not on driving off-road
- **Critical distinction**: Lane invasion ≠ Off-road driving

**Step 2: Examine Sensor Implementation**
- Read full `sensors.py` (1000+ lines)
- **Finding**: Recovery logic EXISTS and works correctly (80% threshold, proper logging)
- **Confusion**: If recovery works, why does bug persist?

**Step 3: Trace Data Flow**
- Grep search → Found integration point at `carla_env.py` line 744
- **BREAKTHROUGH**: Semantic mismatch discovered!

**Step 4: Validate Solution**
- Fetched CARLA LaneType API documentation
- **Confirmed**: `waypoint.lane_type` provides exact distinction between drivable and non-drivable surfaces

### Root Cause: Semantic Mismatch

**The bug was NOT:**
- ❌ Missing recovery logic (it exists and works)
- ❌ Recovery not being called (it is called correctly)
- ❌ Recovery threshold too strict (80% is reasonable)

**The bug WAS:**
- ✅ **Semantic mismatch**: Variable named `offroad_detected` used data from lane invasion sensor
- ✅ **Wrong sensor for wrong purpose**: Lane invasion detects **line crossings**, not **off-road driving**
- ✅ **Missing true off-road detection**: No check for sidewalk/grass/non-drivable surfaces

**Evidence from Code:**

```python
# WRONG (carla_env.py line 744 - BEFORE FIX):
offroad_detected = self.sensors.is_lane_invaded(...)  # ← Detects LINE CROSSINGS

# Variable says: "offroad_detected" (implies grass/sidewalk)
# Data source says: "is_lane_invaded()" (detects marking crossings)
# These are SEMANTICALLY DIFFERENT events!
```

### Additional Bug Discovered: Triple Penalty

Single lane marking touch triggered:
1. Lane keeping reward: **-1.0** (immediate return)
2. Safety (as "offroad"): **-10.0** (from wrong sensor)
3. Safety (lane invasion): **-5.0** (correct)
4. **Total**: **-16.0** for ONE lane touch!

This explains why the agent cannot learn - reward signal is massively biased against any lane interaction.

---

## Solution Implementation

### 1. Created `OffroadDetector` Class

**Location**: `av_td3_system/src/environment/sensors.py` (before `SensorSuite` class)

**Key Features:**
- Uses CARLA Waypoint API with `project_to_road=False`
- Checks `waypoint.lane_type` to distinguish surfaces
- Thread-safe with lock protection
- Proper logging for debugging

**Drivable Lane Types (CARLA API):**
```python
drivable_lane_types = [
    carla.LaneType.Driving,      # Normal traffic lanes
    carla.LaneType.Parking,      # Parking spaces
    carla.LaneType.Bidirectional,  # Two-way traffic lanes
]
```

**Non-Drivable (Off-Road):**
- `Sidewalk` - Pedestrian areas
- `Shoulder` - Road shoulder
- `Border` - Road border
- `Restricted` - Restricted areas
- `None` (no waypoint) - Completely off map

**Implementation:**
```python
class OffroadDetector:
    """
    Detects when vehicle is on non-drivable surface using CARLA Waypoint API.
    Reference: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
    """
    
    def check_offroad(self) -> bool:
        """
        Returns True if on sidewalk/grass/shoulder, False if on drivable lane.
        """
        location = self.vehicle.get_location()
        
        # Don't snap to road - we want to know if truly off-road
        waypoint = self.carla_map.get_waypoint(
            location, 
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        
        if waypoint is None:
            return True  # No waypoint = off-road
        
        return waypoint.lane_type not in drivable_lane_types
```

### 2. Integrated Into SensorSuite

**Changes to `av_td3_system/src/environment/sensors.py`:**

**a) Added to `__init__`:**
```python
self.offroad_detector = OffroadDetector(vehicle, world)  # TASK 1 FIX
```

**b) Added method:**
```python
def is_offroad(self) -> bool:
    """
    Check if vehicle is on non-drivable surface (TASK 1 FIX).
    This is the CORRECT method for off-road detection, NOT lane invasion sensor!
    """
    return self.offroad_detector.check_offroad()
```

**c) Added to `reset()`:**
```python
self.offroad_detector.reset()  # TASK 1 FIX: Reset offroad state
```

### 3. Updated Environment Integration

**Changes to `av_td3_system/src/environment/carla_env.py` line 744:**

**BEFORE (WRONG):**
```python
offroad_detected=self.sensors.is_lane_invaded(
    lateral_deviation=vehicle_state["lateral_deviation"],
    lane_half_width=vehicle_state["lane_half_width"]
),
lane_invasion_detected=(lane_invasion_count > 0),
```

**AFTER (CORRECT):**
```python
# TASK 1 FIX: Use correct offroad detection
offroad_detected=self.sensors.is_offroad(),  # TRUE off-road (grass, sidewalk)
wrong_way=vehicle_state["wrong_way"],
# KEEP lane invasion separate (for lane-keeping reward, not safety penalty)
lane_invasion_detected=self.sensors.is_lane_invaded(
    lateral_deviation=vehicle_state["lateral_deviation"],
    lane_half_width=vehicle_state["lane_half_width"]
),
```

**Key Changes:**
- `offroad_detected` now uses `is_offroad()` (waypoint-based)
- `lane_invasion_detected` still uses `is_lane_invaded()` (for lane-keeping reward)
- Clear semantic separation: off-road ≠ lane crossing

---

## Testing Plan

### Manual Validation Script
Run: `python3 scripts/validate_rewards_manual.py --config config/baseline_config.yaml`

### Test Scenarios

| # | Scenario | Expected `offroad` | Expected `lane_invasion` | Expected Safety Penalty |
|---|----------|-------------------|--------------------------|------------------------|
| 1 | Lane center driving | `False` | `False` | `0.0` |
| 2 | Cross lane marking (legal lane change) | `False` | `True` | `-5.0` (NOT -10.0!) |
| 3 | Drive on grass | `True` | `True` | `-15.0` (-10 offroad + -5 lane) |
| 4 | Return to lane from grass | `False` | `False` | `0.0` (recovery verified) |
| 5 | Drive on sidewalk | `True` | `True` | `-15.0` |

### Success Criteria
- ✅ "[SAFETY-OFFROAD]" **only** appears when on grass/sidewalk/shoulder
- ✅ Lane marking touches trigger **only** "-5.0" penalty (lane invasion, not offroad)
- ✅ Legal lane changes: **-5.0 total** (not -16.0!)
- ✅ Recovery clears penalties when returning to drivable lane
- ✅ **No penalty spam** after recovery

### Automated Test
Add to `tests/test_rewards.py`:
```python
def test_offroad_detection_vs_lane_invasion():
    """
    Test that offroad detection uses waypoint.lane_type, not lane invasion sensor.
    This prevents legal lane changes from being penalized as off-road.
    """
    # Test 1: Vehicle on sidewalk → offroad=True
    # Test 2: Vehicle crosses line on road → offroad=False, lane_invasion=True
    # Test 3: Recovery from grass → offroad=False
```

---

## Impact Analysis

### Before Fix (Broken Behavior)
**Lane Change Maneuver:**
```
Step 1: Start in Lane 1 center
  → offroad=False, lane_invasion=False
  → Total penalty: 0.0 ✅

Step 2: Cross lane marking to Lane 2
  → offroad=TRUE (WRONG!), lane_invasion=True
  → Safety: -10.0 (offroad) + -5.0 (lane) = -15.0
  → Lane keeping: -1.0
  → TOTAL: -16.0 ❌ (destroys learning)

Step 3: Stabilize in Lane 2 center
  → offroad=TRUE (persists!), lane_invasion=False (recovered)
  → Safety: -10.0 (still penalized!)
  → TOTAL: -10.0 ❌ (shouldn't be penalized!)
```

**Impact**: Agent learns to AVOID all lane markings → Cannot perform lane changes, merging, or any realistic maneuvers.

### After Fix (Correct Behavior)
**Lane Change Maneuver:**
```
Step 1: Start in Lane 1 center
  → offroad=False, lane_invasion=False
  → Total penalty: 0.0 ✅

Step 2: Cross lane marking to Lane 2
  → offroad=FALSE (correct!), lane_invasion=True
  → Safety: -5.0 (lane invasion only)
  → Lane keeping: -1.0
  → TOTAL: -6.0 ✅ (reasonable for lane touch)

Step 3: Stabilize in Lane 2 center
  → offroad=False, lane_invasion=False
  → TOTAL: 0.0 ✅ (perfect!)
```

**Impact**: Agent can learn realistic driving - lane changes are mildly discouraged but not catastrophically penalized.

### Training Implications

**Before Fix:**
- TD3 policy biased toward unrealistic "never touch lines" behavior
- Q-values corrupted by false "offroad" penalties during legal maneuvers
- Cannot reproduce paper results (policy unable to perform required maneuvers)

**After Fix:**
- Accurate reward signal for value function approximation
- Agent can explore lane changes, merging, passing
- Reproducible training aligned with TD3 paper methodology

---

## Related Issues

### Issue #2: Triple Penalty Bug
**Status**: Partially addressed by this fix
- Before: -16.0 total (-10 offroad + -5 lane + -1 lane_keeping)
- After: -6.0 total (-5 lane + -1 lane_keeping)
- Remaining concern: Lane invasion triggers BOTH safety (-5) AND lane_keeping (-1)

**Future consideration**: Should lane_keeping reward check `lane_invasion_detected` before applying -1.0?

### Issue #3: Comfort Reward Penalizing Normal Movement
**Status**: Not addressed (Task 2)

### Issue #4: Progress Reward Discontinuity
**Status**: Not addressed (Task 3)

---

## References

### CARLA Documentation
- **LaneType API**: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
- **Waypoint API**: https://carla.readthedocs.io/en/latest/core_map/#waypoints
- **Lane Invasion Sensor**: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

### TD3 Paper
> "...accumulation of error in temporal difference methods"

False penalties during training create error accumulation:
- Every lane marking touch → false "offroad" penalty
- Legal lane changes → penalized as dangerous maneuvers
- Q-values biased toward unrealistic policies (avoid all lane markings)
- Cannot learn normal driving behaviors (lane changes, merging, etc.)

---

## Files Modified

1. **`av_td3_system/src/environment/sensors.py`**
   - Added `OffroadDetector` class (lines ~890-1020)
   - Updated `SensorSuite.__init__()` to instantiate offroad detector
   - Added `SensorSuite.is_offroad()` method
   - Updated `SensorSuite.reset()` to include offroad detector

2. **`av_td3_system/src/environment/carla_env.py`**
   - Updated reward calculation call (line ~744)
   - Changed `offroad_detected` from `is_lane_invaded()` to `is_offroad()`
   - Preserved `lane_invasion_detected` using `is_lane_invaded()` for lane-keeping reward

---

## Commit Message

```
fix(rewards): correct off-road detection to use waypoint lane_type instead of lane invasion sensor

BREAKING CHANGE: offroad_detected now correctly identifies sidewalk/grass vs lane markings

Root Cause:
- Variable 'offroad_detected' was using lane invasion sensor (detects LINE CROSSINGS)
- This caused legal lane changes to be penalized as "driving on grass" (-10.0)
- Even with working recovery logic, semantic mismatch persisted

Solution:
- Created OffroadDetector class using CARLA Waypoint API
- Checks waypoint.lane_type to distinguish:
  * Drivable: Driving, Parking, Bidirectional lanes
  * Off-road: Sidewalk, Shoulder, Border, Restricted areas
- Updated carla_env.py to use correct sensor for offroad detection

Impact:
- Lane changes now correctly penalized at -5.0 (lane invasion only)
- Off-road driving (grass/sidewalk) correctly penalized at -10.0
- Total penalty for lane touch reduced from -16.0 to -6.0
- Enables realistic policy learning (lane changes, merging, passing)

Testing:
- Manual validation: 5 test scenarios covering lane center, lane changes, grass, sidewalk
- Expected: offroad penalty only on non-drivable surfaces
- Expected: recovery clears penalties when returning to lane

References:
- CARLA LaneType: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
- Task 1 of 6-task reward validation plan
- Fixes reward corruption preventing TD3 training convergence

Related: Task 2 (comfort), Task 3 (progress), Task 4 (route completion)
```

---

## Next Steps

**Immediate (After Testing):**
1. ✅ Implement fix (COMPLETE)
2. ⏹️ Run manual validation (`validate_rewards_manual.py`)
3. ⏹️ Verify test scenarios pass
4. ⏹️ Commit changes with detailed message

**Task 2: Fix Comfort Reward** (~1-2 hours)
- Issue: Smooth driving receives negative comfort rewards
- Investigation: Check jerk calculation, CARLA physics noise, sensitivity
- Solution: Add smoothing (moving average) or adjust threshold

**Task 3: Fix Progress Reward Discontinuity** (~1 hour)
- Issue: Jumps when waypoint manager returns None or 0.0
- Solution: Add null checks, clip delta, ensure continuity

**Task 4: Investigate Route Completion Reward** (~30 minutes)
- Issue: Verify +100 bonus applies correctly at route end
- Test: Drive full route, check terminal output and info dict

**Task 5: Manual Validation Session** (~1-2 hours)
- Comprehensive testing with 2000+ steps
- All scenarios from validation guide
- Generate episode logs for analysis

**Task 6: Analysis and Documentation** (~1 hour)
- Run `analyze_reward_validation.py`
- Generate plots and statistics
- Document methodology for paper

---

## Confidence Level

**HIGH** - Bug clearly identified, solution well-defined from official CARLA docs, testing framework ready.

**Evidence:**
- Root cause traced through 9 tool calls with systematic investigation
- Solution validated against CARLA Python API reference
- Clear semantic distinction: lane invasion ≠ off-road driving
- Recovery logic already exists and works correctly
- Fix addresses both immediate bug and underlying design flaw

**Risk Assessment:**
- **Low risk**: Off-road detection using waypoint API is well-documented in CARLA
- **Medium risk**: May discover additional issues during manual testing
- **Mitigation**: Systematic validation approach will catch problems early
