# Fix Implementation: Heading Error Calculation

**Date**: 2025-01-26  
**Issue**: wrong_way_penalty and efficiency reward waypoint-dependent failures  
**Status**: ✅ IMPLEMENTED - Awaiting validation  

---

## Summary

**Root Cause**: Both `wrong_way_penalty` and `efficiency_reward` used **vehicle-to-waypoint bearing** instead of **route tangent direction**, causing heading errors to vary based on vehicle position.

**Solution**: Changed heading calculation to use **route segment tangent** (waypoint[i] → waypoint[i+1]) or **CARLA waypoint API** (`waypoint.transform.rotation.yaw`).

**Files Modified**:
1. `/src/environment/waypoint_manager.py` - `get_target_heading()` method
2. `/src/environment/carla_env.py` - `_check_wrong_way_penalty()` method

---

## Changes Made

### 1. Fixed `waypoint_manager.get_target_heading()` (Lines 406-500)

#### Before (BUGGY)
```python
def get_target_heading(self, vehicle_location) -> float:
    """Get target heading to next waypoint."""
    
    # Get vehicle position
    if hasattr(vehicle_location, 'x'):
        vx, vy = vehicle_location.x, vehicle_location.y
    else:
        vx, vy = vehicle_location[0], vehicle_location[1]
    
    next_wp = self.waypoints[self.current_waypoint_idx]
    
    # ❌ BUG: Calculates bearing FROM vehicle TO waypoint
    dx = next_wp[0] - vx  
    dy = next_wp[1] - vy
    
    heading_carla = math.atan2(dy, dx)  # Vehicle→Waypoint bearing
    return heading_carla
```

**Problem**: This calculates `atan2(next_waypoint - vehicle)`, which is the bearing from the vehicle's current position to the next waypoint. This changes dramatically as the vehicle moves!

#### After (CORRECT)
```python
def get_target_heading(self, vehicle_location) -> float:
    """
    Get target heading (route tangent direction) at vehicle's current position.
    
    FIX #1 (Jan 26, 2025): Route Tangent vs. Vehicle-to-Waypoint Bearing
    """
    
    # Method B (Primary): Use CARLA Waypoint API
    if hasattr(self, 'carla_map') and self.carla_map is not None and carla is not None:
        # Convert to carla.Location if needed
        if hasattr(vehicle_location, 'x'):
            loc = vehicle_location
        else:
            loc = carla.Location(x=vehicle_location[0], 
                               y=vehicle_location[1],
                               z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)
        
        # Get waypoint at lane center (projects vehicle to road)
        waypoint = self.carla_map.get_waypoint(
            loc, 
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint is not None:
            # ✅ CORRECT: Use road's tangent direction from OpenDRIVE
            heading_rad = np.radians(waypoint.transform.rotation.yaw)
            return heading_rad
    
    # Method A (Fallback): Use waypoint-to-waypoint direction
    if self.current_waypoint_idx >= len(self.waypoints) - 1:
        idx = len(self.waypoints) - 2
    else:
        idx = self.current_waypoint_idx
    
    wp_current = self.waypoints[idx]
    wp_next = self.waypoints[idx + 1]
    
    # ✅ CORRECT: Calculate route segment direction (WP[i] → WP[i+1])
    dx = wp_next[0] - wp_current[0]  # Route direction, not vehicle position
    dy = wp_next[1] - wp_current[1]
    
    heading_rad = math.atan2(dy, dx)
    return heading_rad
```

**Key Changes**:
1. **Primary method** (CARLA API): Uses `waypoint.transform.rotation.yaw` from CARLA's road definition
   - Automatically handles curves, intersections, lane changes
   - Official CARLA-recommended approach
   - Most robust solution

2. **Fallback method** (Manual calculation): Uses `waypoint[i] → waypoint[i+1]` direction
   - Calculates route segment tangent, NOT vehicle bearing
   - Independent of vehicle position within segment
   - Works for straight roads (our Town01 case)

### 2. Fixed `carla_env._check_wrong_way_penalty()` (Lines 1160-1270)

#### Before (BUGGY)
```python
def _check_wrong_way_penalty(self, velocity: float) -> float:
    """Check if vehicle facing wrong direction."""
    
    # Get next waypoint
    if hasattr(self.waypoint_manager, '_current_waypoint_index'):
        current_idx = self.waypoint_manager._current_waypoint_index
        next_idx = min(current_idx + 1, len(waypoints) - 1)
    else:
        next_idx = min(1, len(waypoints) - 1)
    
    next_waypoint = waypoints[next_idx]
    next_x = next_waypoint[0]
    next_y = next_waypoint[1]
    
    # ❌ BUG: Same issue - vehicle→waypoint bearing
    dx = next_x - vehicle_location.x
    dy = next_y - vehicle_location.y
    route_direction = np.degrees(np.arctan2(dy, dx))
    
    heading_error = vehicle_yaw - route_direction
    # ... rest of penalty calculation
```

**Problem**: Same bug as `get_target_heading()` - calculates bearing instead of route tangent.

#### After (CORRECT)
```python
def _check_wrong_way_penalty(self, velocity: float) -> float:
    """
    Check if vehicle facing wrong direction.
    
    FIX #3 (Nov 24, 2025): Wrong-Way Detection Based on Heading vs. Route
    FIX #1 (Jan 26, 2025): Use Corrected Heading Calculation (Route Tangent)
    """
    
    # ✅ FIX: Use corrected heading calculation (route tangent)
    route_direction_rad = self.waypoint_manager.get_target_heading(vehicle_location)
    route_direction_deg = np.degrees(route_direction_rad)
    
    # Calculate heading error: vehicle_yaw - route_direction
    heading_error = vehicle_yaw - route_direction_deg
    
    # Normalize to [-180, 180]
    while heading_error > 180.0:
        heading_error -= 360.0
    while heading_error < -180.0:
        heading_error += 360.0
    
    # ... rest of penalty calculation (unchanged)
```

**Key Changes**:
1. Removed manual waypoint index lookup
2. Removed manual bearing calculation
3. **Delegated to `waypoint_manager.get_target_heading()`** for consistency
4. Ensures both efficiency and wrong-way use identical heading calculation

---

## Expected Impact

### Before Fix (BUGGY Behavior)

#### Spawn (Step 0)
```
Vehicle: (317.74, 129.49) - exactly at WP0
Next WP: (314.74, 129.49) - WP1 is 3m west
Vehicle yaw: 180° (facing west)

Buggy calculation:
  dx = 314.74 - 317.74 = -3.0
  dy = 129.49 - 129.49 = 0.0
  route_direction = atan2(0, -3) = ±180°
  
  heading_error = 180° - (±180°) = 0° or 360°
  
  ✓ Works by accident (vehicle at exact waypoint)
```

#### After Movement (Step 5+)
```
Vehicle: (314.80, 129.50) - moved 3m west, drifted 1cm north
Next WP: (311.63, 129.49) - WP2 is 3.17m west
Vehicle yaw: -180° (still aligned!)

Buggy calculation:
  dx = 311.63 - 314.80 = -3.17
  dy = 129.49 - 129.50 = -0.01  (1cm drift)
  route_direction = atan2(-0.01, -3.17) 
                  = -180° + atan(0.01/3.17)
                  ≈ -180° + 0.18° = -179.82°
  
  heading_error = -180° - (-179.82°) = -0.18°
  
  ✓ Still works (small error from 1cm drift)
```

#### Dense Waypoints (Step 50+)
```
Vehicle: (300.005, 129.50) - 0.5cm east, 1cm north of WP[1000]
WP[1000]: (300.00, 129.49)
WP[1001]: (299.99, 129.49) - only 1cm ahead!
Vehicle yaw: -180° (aligned)

Buggy calculation:
  dx = 299.99 - 300.005 = -0.015  (1.5cm)
  dy = 129.49 - 129.50 = -0.01    (1cm)
  route_direction = atan2(-0.01, -0.015)
                  ≈ -180° + 33.7° = -146.3°
  
  heading_error = -180° - (-146.3°) = -33.7° ❌ WRONG!
  
  Efficiency: cos(-33.7°) = 0.83 instead of 1.0
  Loss: 17% reward for perfect driving!
```

### After Fix (CORRECT Behavior)

#### Method B (CARLA API - Primary)
```python
waypoint = carla_map.get_waypoint(vehicle_location, project_to_road=True)
route_direction = waypoint.transform.rotation.yaw

# CARLA waypoint always returns road tangent direction
# Independent of vehicle position!
```

**Result**: Heading error = vehicle_yaw - road_tangent_yaw
- Always correct, regardless of position
- Handles curves automatically
- No sensitivity to dense waypoints

#### Method A (Fallback - Manual)
```python
# Get current route segment
wp_current = waypoints[1000]  # (300.00, 129.49)
wp_next = waypoints[1001]     # (299.99, 129.49)

# Route tangent direction (WP → WP)
dx = 299.99 - 300.00 = -0.01
dy = 129.49 - 129.49 = 0.0
route_direction = atan2(0, -0.01) = ±180°

# Vehicle at (300.005, 129.50) with yaw=-180°
heading_error = -180° - (±180°) = 0° ✓ CORRECT!
```

**Result**: Heading error independent of vehicle position within segment

---

## Validation Plan

### Test 1: Spawn Alignment
**Expected**: heading_error ≈ 0° at spawn (vehicle aligned with route)

**Before fix**: -150.64° (WRONG!)  
**After fix**: Should be ~0°

### Test 2: Straight Driving
**Expected**: heading_error stays ~0° for well-aligned vehicle

**Before fix**: Varies wildly (±30-40°) due to lateral drift  
**After fix**: Should stay within ±5° for good driving

### Test 3: Efficiency Reward Consistency
**Expected**: High efficiency reward (~0.9-1.0) for aligned, moving vehicle

**Before fix**: 0.71-0.83 due to false heading errors  
**After fix**: Should be 0.95+ for straight driving

### Test 4: Wrong-Way Detection
**Expected**: Penalty triggers when vehicle reverses (yaw ≈ 0° on westbound route)

**Before fix**: May not trigger correctly  
**After fix**: Should trigger -3.0 to -5.0 penalty

### Test 5: Dense Waypoint Segments
**Expected**: No heading jitter as vehicle progresses through 26,396 waypoints

**Before fix**: Large variations every few centimeters  
**After fix**: Smooth, consistent heading regardless of position

---

## Verification Commands

### 1. Run Manual Validation Script
```bash
docker compose -f docker-compose.test.yml run --rm carla-agent \
  python scripts/validate_rewards_manual.py
```

**Check in logs**:
- `Heading error: X.XX°` should stay near 0° for aligned vehicle
- `EFFICIENCY` reward should be high (~0.9+) when moving forward
- `WRONG-WAY` penalty should only trigger when actually reversing

### 2. Check First 10 Steps
```bash
grep "Heading error:" validation_logs/debug_test/progress.log | head -10
```

**Expected output**:
```
Heading error: -0.05° (-0.001 rad)  ← Near zero!
Heading error: 0.12° (0.002 rad)
Heading error: -0.08° (-0.001 rad)
...
```

**NOT** (before fix):
```
Heading error: -150.64° (-2.629 rad)  ← Wrong!
Heading error: -0.00° (-0.000 rad)
Heading error: -33.7° (-0.588 rad)   ← Wild variations
...
```

### 3. Compare Efficiency Rewards
```bash
grep "EFFICIENCY" validation_logs/debug_test/progress.log | head -20
```

**Expected** (after fix):
```
EFFICIENCY (target speed tracking):
   Raw: 0.9523  ← High for aligned vehicle!
   Weight: 1.00
   Contribution: 0.9523
```

**NOT** (before fix):
```
EFFICIENCY (target speed tracking):
   Raw: 0.7123  ← Low due to false heading error
   Weight: 1.00
   Contribution: 0.7123
```

---

## Rollback Plan

If validation fails:

1. **Revert commits**:
```bash
git checkout HEAD~1 src/environment/waypoint_manager.py
git checkout HEAD~1 src/environment/carla_env.py
```

2. **Check original logs** to verify reversion:
```bash
docker compose -f docker-compose.test.yml run --rm carla-agent \
  python scripts/validate_rewards_manual.py
```

3. **Re-analyze** using ROOT_CAUSE_ANALYSIS_HEADING_ERROR.md

---

## References

### Documentation
- **CARLA Waypoint API**: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint
- **CARLA Transform**: https://carla.readthedocs.io/en/latest/python_api/#carla.Transform
- **Gymnasium API**: https://gymnasium.farama.org/api/env/

### Related Documents
- `ROOT_CAUSE_ANALYSIS_HEADING_ERROR.md` - Detailed bug analysis
- `CORRECTED_ANALYSIS_SUMMARY.md` - Previous fixes (waypoint bonus, wrong-way v1)
- `IMPLEMENTATION_FIXES_NOV_24.md` - Nov 24 fix documentation

### Literature
- **TD3 Paper** (Fujimoto et al.): Continuous, differentiable rewards required
- **Chen et al. (2019)**: End-to-end learning with explicit traffic rules
- **Kendall et al. (2019)**: Learning to drive in a day

---

## Status

**Implementation**: ✅ COMPLETE  
**Code Review**: ⏳ PENDING  
**Testing**: 🔄 IN PROGRESS  
**Deployment**: ⏳ AWAITING VALIDATION  

**Next Step**: Run `validate_rewards_manual.py` and check logs for expected behavior.

---

**Author**: GitHub Copilot (Agent Mode)  
**Date**: 2025-01-26  
**Issue**: #1 (Heading Error Calculation Bug)
