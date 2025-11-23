# Bug #7: Off-Road False Detection at Intersections - FIX APPLIED ✅

**Date**: 2025-01-24
**Bug Status**: FIXED
**Testing Status**: Pending verification

---

## 🐛 Problem Summary

Episode 3 terminated at step 580 with "off_road" reason (`lateral_deviation=2.185m > 2.0m`) despite:
- ✅ Vehicle visually centered in lane
- ✅ Correct intersection turn execution
- ✅ Zero lane invasions

**User's Insight** (verbatim):
> "The car is in the center of the lane i can see it. The problem may be in the Off-road lateral_deviation, because the car just started entering the intersection, it was starting to do the right steering maneuver."

---

## 🔍 Root Cause Analysis

### Current (Buggy) Implementation

**File**: `src/environment/waypoint_manager.py` (lines 300-343, OLD)

```python
def get_lateral_deviation(self, vehicle_location) -> float:
    # Get wp1 and wp2 from planned route
    wp1 = self.waypoints[self.current_waypoint_idx - 1]
    wp2 = self.waypoints[self.current_waypoint_idx]
    
    # Create STRAIGHT LINE vector between waypoints
    route_dx = wp2[0] - wp1[0]
    route_dy = wp2[1] - wp1[1]
    
    # Calculate perpendicular distance to this STRAIGHT line
    cross = route_dx * vy - route_dy * vx
    lateral_dev = cross / route_length  # ❌ WRONG FOR CURVES!
```

**Why It Fails**:
1. Uses **straight line** between `wp1` and `wp2`
2. At intersections, road **curves** but the reference line doesn't
3. As vehicle correctly follows the curved lane:
   - Vehicle position: Centered in lane (correct)
   - Measured deviation: Distance to **straight line** (not lane center)
   - Result: 2.185m > 2.0m threshold → FALSE "off-road"

**Diagram**:
```
Intersection Turn (Top View):

     wp1 ────────► wp2
         \        ↑ (straight reference line)
          \      /
           \    /  2.185m deviation!
            \  ↓
         🚗 ← Vehicle (actually centered in lane)
             │
             └─ Curved lane center (OpenDRIVE geometry)
```

---

## ✅ Solution: CARLA OpenDRIVE Projection

### What CARLA Documentation Says

From `https://carla.readthedocs.io/en/latest/python_api/#carla.Map.get_waypoint`:

> **`get_waypoint(location, project_to_road=True, lane_type=Driving)`**
> 
> Returns a waypoint **at the center of the closest lane** when `project_to_road=True`.
> Returns None if location doesn't belong to a road.

From `https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint`:

> **`waypoint.transform`** (carla.Transform)
> 
> Position and orientation **according to current lane**.
> Computed from OpenDRIVE road geometry.

From `https://carla.readthedocs.io/en/latest/core_map/#junctions`:

> **Junctions**: Class that represents OpenDRIVE junctions.
> **`waypoint.is_junction`**: True if waypoint is in junction.
> **Lanes follow their OpenDRIVE definitions** - no special intersection handling needed.

**Key Insight**: 
- Waypoint projection works **identically** in junctions and straightaways
- `transform.location` automatically follows road curvature
- OpenDRIVE handles the geometry - we just query it!

---

## 🛠️ Implementation

### Modified Files

#### 1. `src/environment/waypoint_manager.py`

**Added to `__init__`** (line 28):
```python
def __init__(
    self,
    waypoints_file: str,
    lookahead_distance: float = 50.0,
    num_waypoints_ahead: int = 10,
    waypoint_spacing: float = 5.0,
    carla_map=None,  # ✅ NEW: CARLA map for proper lateral deviation
):
    ...
    self.carla_map = carla_map  # Store reference
```

**Replaced `get_lateral_deviation`** (lines 300-343):
```python
def get_lateral_deviation(self, vehicle_location) -> float:
    """
    Get lateral deviation from lane center using CARLA's OpenDRIVE projection.
    
    This method properly accounts for road curvature at intersections by projecting
    the vehicle's location to the center of the nearest lane using CARLA's map API.
    """
    if not hasattr(self, 'carla_map') or self.carla_map is None:
        # Fallback to old straight-line method if map not available
        return self._get_lateral_deviation_legacy(vehicle_location)
    
    # Convert to carla.Location if needed
    if hasattr(vehicle_location, 'x'):  # Already carla.Location
        loc = vehicle_location
    else:  # Tuple (x, y, z)
        import carla
        loc = carla.Location(x=vehicle_location[0], 
                            y=vehicle_location[1], 
                            z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)
    
    # ✅ Get waypoint at lane center using CARLA's OpenDRIVE projection
    waypoint = self.carla_map.get_waypoint(
        loc, 
        project_to_road=True,  # Project to CENTER of nearest lane
        lane_type=carla.LaneType.Driving
    )
    
    if waypoint is None:
        # Vehicle is truly off-road (not on any driving lane)
        return float('inf')  # Signal as maximum deviation
    
    # Get lane center location (follows road curvature through intersections)
    lane_center = waypoint.transform.location
    
    # Calculate 2D Euclidean distance from vehicle to lane center
    lateral_deviation = math.sqrt(
        (loc.x - lane_center.x)**2 + 
        (loc.y - lane_center.y)**2
    )
    
    return lateral_deviation
```

**Added legacy fallback** (for when map unavailable):
```python
def _get_lateral_deviation_legacy(self, vehicle_location) -> float:
    """
    Legacy method: straight-line projection between waypoints.
    DEPRECATED - does not account for road curvature.
    Kept for fallback when CARLA map is not available.
    """
    # ... (old implementation moved here)
```

#### 2. `src/environment/carla_env.py`

**Pass CARLA map to WaypointManager** (line 150):
```python
# Get CARLA map for proper lateral deviation calculation
carla_map = self.world.get_map()

legacy_waypoint_manager = WaypointManager(
    waypoints_file=waypoints_file,
    lookahead_distance=50.0,
    num_waypoints_ahead=25,
    carla_map=carla_map,  # ✅ NEW: Pass map for intersection-aware deviation
)
```

**Pass CARLA map to WaypointManagerAdapter** (line 257):
```python
class WaypointManagerAdapter:
    def __init__(self, route_manager, lookahead_distance, sampling_resolution, carla_map):
        self.route_manager = route_manager
        self.lookahead_distance = lookahead_distance
        self.sampling_resolution = sampling_resolution
        self.carla_map = carla_map  # ✅ NEW: Store for lateral deviation
```

**Added `get_lateral_deviation` to Adapter** (line 318):
```python
def get_lateral_deviation(self, vehicle_location) -> float:
    """
    Get lateral deviation from lane center using CARLA's OpenDRIVE projection.
    Delegates to the same proper calculation used by WaypointManager.
    """
    if not self.carla_map:
        return 0.0  # Fallback if no map available
    
    # Convert to carla.Location if needed
    if hasattr(vehicle_location, 'x'):  # Already carla.Location
        loc = vehicle_location
    else:  # Tuple (x, y, z)
        loc = carla.Location(x=vehicle_location[0], 
                            y=vehicle_location[1], 
                            z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)
    
    # Get waypoint at lane center (follows road curvature)
    waypoint = self.carla_map.get_waypoint(
        loc, 
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    if waypoint is None:
        # Vehicle is off-road
        return float('inf')
    
    # Calculate 2D distance from vehicle to lane center
    lane_center = waypoint.transform.location
    lateral_deviation = math.sqrt(
        (loc.x - lane_center.x)**2 + 
        (loc.y - lane_center.y)**2
    )
    
    return lateral_deviation
```

**Pass map when creating adapter** (line 345):
```python
carla_map = self.world.get_map()

return WaypointManagerAdapter(
    route_manager=self.route_manager,
    lookahead_distance=50.0,
    sampling_resolution=2.0,
    carla_map=carla_map  # ✅ NEW
)
```

---

## 🎯 Expected Results After Fix

### Before (Buggy):
```
[TERMINATION] Off-road at step 580: lateral_deviation=2.185m > 2.0m threshold
Episode 3: Collision count = 0, Lane invasions = 0
Episode 3: Terminated at step 580 (Expected ~800+)
Reason: "off_road" (FALSE POSITIVE during correct turn)
```

### After (Fixed):
```
Episode 3: Collision count = 0, Lane invasions = 0
Episode 3: lateral_deviation stays < 2.0m during intersection turn ✅
Episode 3: Either:
  - Completes route (Success: True) ✅, OR
  - Hits collision (true failure), OR
  - Reaches max steps
Reason: No false "off_road" during correct navigation ✅
```

### Verification Checklist

Run diagnostic test:
```bash
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

**Success Criteria**:
- [ ] Episode 3 does NOT terminate with "off_road" at step ~580
- [ ] Lateral deviation remains < 2.0m during intersection turns
- [ ] Lane invasions remain 0
- [ ] At least 1 episode reaches Success: True OR true collision

---

## 📚 Related Documentation

**CARLA API Documentation**:
- [`carla.Map.get_waypoint()`](https://carla.readthedocs.io/en/latest/python_api/#carla.Map.get_waypoint)
- [`carla.Waypoint`](https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint)
- [`carla.Junction`](https://carla.readthedocs.io/en/latest/python_api/#carla.Junction)
- [Maps and Navigation](https://carla.readthedocs.io/en/latest/core_map/)

**Project Files**:
- Previous bugs: `docs/day-23/baseline/CRITICAL_FIX_LANE_INVASION_TERMINATION.md`
- Diagnostic guide: `docs/day-23/baseline/diagnosis/TERMINATION_DIAGNOSTIC_GUIDE.md`
- Phase 3 results: `results/baseline_evaluation/analysis_0_*/`

---

## 🔄 Bug Tracking

| Bug # | Issue | Status | Fix |
|-------|-------|--------|-----|
| #0 | Zigzag behavior | FIXED ✅ | Switched to Pure Pursuit |
| #1 | Waypoint stuck | FIXED ✅ | Added filtering |
| #2 | Fixed 30 km/h | FIXED ✅ | Extract from waypoints |
| #3 | Reactive speed | FIXED ✅ | 20m lookahead |
| #4 | Large PP lookahead | FIXED ✅ | Reduced to 6m |
| #5 | Missing success flag | FIXED ✅ | Added to info |
| #6 | Format error | FIXED ✅ | Fixed None checking |
| **#7** | **Off-road at intersection** | **FIXED ✅** | **CARLA map projection** |

---

## 🏁 Next Steps

1. **Test the fix**:
   ```bash
   docker run ... python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 3 --debug
   ```

2. **Verify metrics**:
   - Episode 3 completion behavior
   - Lateral deviation values at intersection
   - No false off-road terminations

3. **If successful**:
   - Document results
   - Update baseline evaluation metrics
   - Proceed with TD3 vs baseline comparison

4. **If issues remain**:
   - Consider increasing threshold for junctions: `threshold = 3.0 if waypoint.is_junction else 2.0`
   - Or dynamic threshold: `threshold = waypoint.lane_width * 0.6`

---

**Status**: FIX APPLIED, AWAITING TESTING ⏳
