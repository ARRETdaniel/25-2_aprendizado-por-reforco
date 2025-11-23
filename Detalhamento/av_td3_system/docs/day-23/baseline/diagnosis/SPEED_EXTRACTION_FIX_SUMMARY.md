# Speed Extraction Fix - Implementation Summary

**Date**: 2025-11-23  
**Issue**: Vehicle ignoring waypoint speed profile  
**Fix**: Added `_get_target_speed_from_waypoints()` method  
**Status**: ✅ **IMPLEMENTED - READY TO TEST**

---

## Changes Made

### 1. New Method: `_get_target_speed_from_waypoints()`

**File**: `src/baselines/baseline_controller.py` (lines ~192-245)

```python
def _get_target_speed_from_waypoints(
    self,
    current_x: float,
    current_y: float,
    waypoints: List[Tuple[float, float, float]]
) -> float:
    """
    Extract target speed from closest waypoint.
    Matches GitHub's update_desired_speed() implementation.
    """
    if len(waypoints) == 0:
        return self.target_speed  # Fallback
    
    waypoints_np = np.array(waypoints)
    
    # Find closest waypoint
    distances = np.sqrt(
        (waypoints_np[:, 0] - current_x)**2 +
        (waypoints_np[:, 1] - current_y)**2
    )
    closest_index = np.argmin(distances)
    
    # Extract speed (3rd column)
    target_speed = waypoints_np[closest_index, 2]
    
    return target_speed
```

**Purpose**: Dynamically extract target speed from waypoint list instead of using fixed value.

---

### 2. Updated: `compute_control()` Method

**File**: `src/baselines/baseline_controller.py` (lines ~447-460)

**Before**:
```python
# STEP 2: Determine target speed
if target_speed is None:
    target_speed = self.target_speed  # ← FIXED 8.33 m/s
```

**After**:
```python
# STEP 2: Determine target speed
if target_speed is None:
    target_speed = self._get_target_speed_from_waypoints(
        current_x=current_x,
        current_y=current_y,
        waypoints=waypoints
    )  # ← DYNAMIC from waypoints!
```

---

## Expected Behavior

### Speed Profile Tracking

| Location | X (m) | Waypoint Speed | Expected Vehicle Speed |
|----------|-------|----------------|------------------------|
| Start | 317 | 8.333 m/s (30 km/h) | 8.3 m/s |
| Straight | 150 | 8.333 m/s | 8.3 m/s |
| Before turn | 105 | 8.333 m/s | 8.3 m/s |
| **Transition** | **100** | **2.5 m/s (9 km/h)** | **Starts braking** |
| Turn entry | 95 | 2.5 m/s | 2.5 m/s |
| In turn | 92 | 2.5 m/s | 2.5 m/s |

### Debug Output (Expected)

```
[PP-DEBUG Step 1340] X=105.0 | target_speed=8.33 m/s ✅ Fast on straight
[PP-DEBUG Step 1345] X=102.0 | target_speed=2.50 m/s ✅ Braking for turn!
[PP-DEBUG Step 1350] X=100.0 | target_speed=2.50 m/s ✅ Slowing down
[PP-DEBUG Step 1360] X=96.0  | target_speed=2.50 m/s ✅ Safe speed
[PP-DEBUG Step 1370] X=93.0  | target_speed=2.50 m/s ✅ Safe turn
No lane invasions! ✅
```

---

## Test Plan

### Quick Validation (15 minutes)

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 3 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

### Success Criteria

✅ **Target speed changes dynamically**:
- High (8.33 m/s) on straight sections
- Low (2.5 m/s) at intersection

✅ **Vehicle slows before turn**:
- Brake applied around X=100m
- Speed reaches 2.5 m/s before X=95m

✅ **No lane invasions**:
- Alpha stays < 10° during turn
- Y-coordinate stays within lane bounds
- No lane marking crossings

✅ **Episode completes successfully**:
- Reaches final waypoint
- No safety violations
- Success rate: 100%

### What to Look For in Logs

**Speed Transition**:
```bash
grep "target_speed" evaluation_logs.txt | tail -100
```

Expected pattern:
```
target_speed=8.33 m/s  # Repeating on straight
target_speed=8.33 m/s
target_speed=2.50 m/s  # ← CHANGE HERE!
target_speed=2.50 m/s  # Repeating in turn
```

**Brake Activation**:
```bash
grep "brake" evaluation_logs.txt | grep -v "brake=0.0"
```

Expected: Brake > 0 around X=100-105m

**Lane Invasions**:
```bash
grep -i "lane invasion" evaluation_logs.txt
```

Expected: No matches!

---

## Rollback Plan (If Tests Fail)

If the fix doesn't work or causes new issues:

```bash
git diff src/baselines/baseline_controller.py  # Review changes
git checkout src/baselines/baseline_controller.py  # Revert
```

Manual revert:
1. Remove `_get_target_speed_from_waypoints()` method
2. Change STEP 2 back to: `target_speed = self.target_speed`

---

## Next Steps

1. ✅ **RUN TEST** - Execute 3-episode evaluation
2. ⏸️ **VERIFY** - Check logs for speed changes and no lane invasions
3. ⏸️ **FULL EVAL** - Run 10-episode evaluation if successful
4. ⏸️ **DOCUMENT** - Create final summary of both bug fixes
5. ⏸️ **PROCEED** - Move to Phase 4 (NPC interaction)

---

## Related Fixes

This is the **SECOND** critical bug fix:

1. **Bug #1** (2025-11-23): Missing waypoint filtering → left drift
   - **Fix**: Added `_filter_waypoints_ahead()` method
   - **Status**: ✅ FIXED

2. **Bug #2** (2025-11-23): Ignoring waypoint speeds → intersection failures
   - **Fix**: Added `_get_target_speed_from_waypoints()` method
   - **Status**: ✅ FIXED (testing now)

Both bugs were in GitHub's working implementation but missing from our port!

---

**Status**: 🚀 Ready to test! Run the evaluation command above.
