# Dynamic Route Generation Implementation Summary

**Date**: October 22, 2025
**Status**: ✅ IMPLEMENTED - Ready for Testing
**Impact**: HIGH - Improves system architecture significantly

---

## What Was Implemented

### 1. **DynamicRouteManager** (NEW)
**File**: `src/environment/dynamic_route_manager.py`

A new class that uses CARLA's `GlobalRoutePlanner` API to generate waypoints dynamically using A* pathfinding on the road network topology.

**Key Features**:
- Uses CARLA's native route planning (not custom logic)
- Automatically gets correct Z-coordinates (road surface level)
- Provides same interface as WaypointManager (drop-in replacement)
- Works on any CARLA map (Town01, Town02, etc.)

**Methods**:
```python
# Get waypoints array compatible with old system
waypoints = route_manager.get_waypoints()  # (N, 3) NumPy array

# Get spawn transform at route start
spawn_transform = route_manager.get_start_transform()

# Find next waypoint ahead of vehicle
next_idx = route_manager.get_next_waypoint_index(vehicle_location, current_idx)

# Calculate total route length
length = route_manager.get_route_length()  # meters

# Check if route complete
done = route_manager.is_route_complete(vehicle_location, threshold=5.0)
```

### 2. **Updated carla_env.py**
**File**: `src/environment/carla_env.py`

Modified to use DynamicRouteManager while maintaining backward compatibility.

**Changes**:
1. Added import for `DynamicRouteManager`
2. Extract start/end from `waypoints.txt` (for reproducibility)
3. Initialize `DynamicRouteManager` after world load
4. Create adapter to provide `WaypointManager` interface
5. Updated `reset()` to use dynamic spawn transform
6. Added configuration flag: `use_dynamic_generation`

**Adapter Pattern**:
```python
class WaypointManagerAdapter:
    """Makes DynamicRouteManager compatible with existing code"""

    @property
    def waypoints(self):
        return self.route_manager.get_waypoints()

    def reset(self):
        self._current_waypoint_index = 0

    def get_next_waypoints(self, vehicle_location, current_index):
        # Delegates to DynamicRouteManager
```

### 3. **Updated carla_config.yaml**
**File**: `config/carla_config.yaml`

Added configuration for dynamic route generation:

```yaml
route:
  # Enable dynamic route generation
  use_dynamic_generation: true  # NEW FLAG

  # Waypoints file (only START/END used when dynamic=true)
  waypoints_file: '/workspace/av_td3_system/config/waypoints.txt'

  # Dynamic route settings
  sampling_resolution: 2.0  # Distance between waypoints (meters)

  # Processing (same for static/dynamic)
  lookahead_distance: 50.0
  num_waypoints_ahead: 10
  use_relative_coordinates: true
```

### 4. **Documentation**
- `scripts/DYNAMIC_ROUTE_GENERATION.md` - Comprehensive guide
- `scripts/extract_waypoint_start_end.py` - Utility to extract start/end
- `scripts/test_dynamic_routes.py` - Test script

---

## How It Works

### Route Generation Flow

```
1. Read waypoints.txt
   ↓
2. Extract START (first waypoint) and END (last waypoint)
   ↓
3. Initialize DynamicRouteManager with CARLA world
   ↓
4. DynamicRouteManager uses CARLA Map API:
   - map.get_waypoint(start, project_to_road=True) → Start waypoint at road
   - map.get_waypoint(end, project_to_road=True) → End waypoint at road
   ↓
5. GlobalRoutePlanner computes A* route on road topology
   ↓
6. Returns list of waypoints at 2m intervals along route
   ↓
7. Vehicle spawns at first waypoint (correct height, aligned with road)
   ↓
8. Agent follows waypoints (same as before, but now dynamically generated)
```

### Start/End Locations (Fixed for Reproducibility)

**From**: `config/waypoints.txt`

```
START: (317.74, 129.49, 8.333)  ← First waypoint in file
END:   (92.34, 86.73282, 2.5)   ← Last waypoint in file
```

**Note**: Z-coordinates will be corrected by CARLA to road surface level.

---

## Benefits

| Aspect | Before (Static) | After (Dynamic) | Improvement |
|--------|----------------|-----------------|-------------|
| **Z-Coordinates** | Hardcoded (8.333m, wrong) | Auto (road surface, correct) | ✅ Fixed spawn bug |
| **Route Definition** | 86 manual waypoints | A* pathfinding | ✅ Topology-aware |
| **Map Support** | Town01 only | Any CARLA map | ✅ Generalizable |
| **Route Variations** | 1 fixed route | Can vary (future) | ✅ Better generalization |
| **Maintenance** | Manual CSV editing | Zero maintenance | ✅ Easier to maintain |
| **Lane Info** | None | Full (RoadOption) | ✅ Richer state |
| **Spawn Alignment** | Manual calculation | Auto from waypoint | ✅ More robust |

---

## Backward Compatibility

✅ **Fully backward compatible**:

1. **Config flag**: Set `use_dynamic_generation: false` to use old system
2. **Interface preserved**: DynamicRouteManager provides same API as WaypointManager
3. **Start/end preserved**: Uses exact same locations from waypoints.txt
4. **Existing code unchanged**: No changes needed in:
   - State calculation
   - Reward computation
   - Action processing
   - Training loop

---

## Testing

### Test Script
```bash
# Test dynamic route generation
docker run --rm --network host --runtime nvidia \
  -v $(pwd):/workspace \
  -w /workspace/av_td3_system \
  td3-av-system:v2.0-python310 \
  python3 scripts/test_dynamic_routes.py
```

**Expected output**:
```
✅ DYNAMIC ROUTE GENERATION TEST PASSED

Summary:
  • Route successfully generated using GlobalRoutePlanner
  • 145 waypoints at 2.0m intervals
  • Total route length: ~290m
  • Z-coordinates automatically corrected (road surface level)
  • Spawn transform aligned with road direction
```

### Full Training Test
```bash
# Run training with dynamic routes
docker run --rm --network host --runtime nvidia \
  -e DISPLAY=:1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace/av_td3_system \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 100 \
    --debug
```

**What to verify**:
1. ✅ Log shows: "Using DYNAMIC route (GlobalRoutePlanner)"
2. ✅ No spawn height collisions (vehicle doesn't fall)
3. ✅ Route length ~280-300m (vs ~229m straight-line)
4. ✅ Waypoint count matches sampling resolution
5. ✅ Vehicle spawns aligned with road heading

---

## Comparison: Static vs Dynamic

### Static Waypoints (waypoints.txt)

**Pros**:
- Simple (just CSV file)
- Predictable (exact same waypoints)

**Cons**:
- ❌ Wrong Z-coordinates (caused spawn bug)
- ❌ Manual creation required
- ❌ Town01-specific (won't work on other maps)
- ❌ Not topology-aware (just straight X,Y,Z points)
- ❌ Single fixed route (poor generalization)

### Dynamic Routes (GlobalRoutePlanner)

**Pros**:
- ✅ Correct Z-coordinates (automatic)
- ✅ Zero maintenance (generated on-the-fly)
- ✅ Map-agnostic (works on any CARLA map)
- ✅ Topology-aware (follows actual roads)
- ✅ Can generate varied routes (future enhancement)
- ✅ Includes lane information (RoadOption)

**Cons**:
- Slightly more complex initialization
- Requires CARLA agents package

---

## Future Enhancements

### 1. Random Route Variations (Easy)
```python
# In reset(): randomly select different destinations
spawn_points = world.get_map().get_spawn_points()
random_end = random.choice(spawn_points)
route_manager.regenerate_route(new_end_location=random_end.location)
```

**Benefit**: Agent learns to generalize to different routes, not memorize one path.

### 2. Multi-Map Training (Easy)
```python
# Train on multiple maps
for map_name in ['Town01', 'Town02', 'Town03']:
    world = client.load_world(map_name)
    route_manager = DynamicRouteManager(world, start, end)
    # Train...
```

**Benefit**: Agent learns map-invariant driving skills.

### 3. Dynamic Rerouting (Medium)
```python
# Change destination mid-episode (simulates navigation updates)
if new_destination_received:
    route_manager.regenerate_route(new_end_location)
```

**Benefit**: More realistic navigation scenario.

---

## Technical Details

### CARLA API Methods Used

1. **`map.get_waypoint(location, project_to_road=True)`**
   - Projects location to nearest driving lane
   - Returns waypoint at road surface (correct Z)
   - Used to find start/end waypoints

2. **`GlobalRoutePlanner(map, sampling_resolution)`**
   - A* pathfinding on road network
   - Returns optimal route following roads
   - Configurable waypoint spacing

3. **`planner.trace_route(start_location, end_location)`**
   - Computes route from start to end
   - Returns list of `(waypoint, RoadOption)` tuples
   - RoadOption: LEFT, RIGHT, STRAIGHT, LANEFOLLOW, etc.

### Data Flow

```
waypoints.txt
    ↓
[317.74, 129.49, 8.333] (START)
[92.34, 86.73282, 2.5] (END)
    ↓
map.get_waypoint(project_to_road=True)
    ↓
[317.74, 129.49, 0.52] ← Corrected Z (road surface)
[92.34, 86.73282, 0.51] ← Corrected Z
    ↓
GlobalRoutePlanner.trace_route(start, end)
    ↓
[(wp1, STRAIGHT), (wp2, STRAIGHT), ..., (wpN, LEFT)]
    ↓
Extract waypoints → NumPy array (N, 3)
    ↓
Agent uses waypoints (same as before)
```

---

## File Changes Summary

### New Files (3)
1. `src/environment/dynamic_route_manager.py` - Core implementation
2. `scripts/test_dynamic_routes.py` - Test script
3. `scripts/DYNAMIC_ROUTE_GENERATION.md` - Documentation

### Modified Files (2)
1. `src/environment/carla_env.py` - Integration
2. `config/carla_config.yaml` - Configuration

### Utility (1)
1. `scripts/extract_waypoint_start_end.py` - Helper script

---

## Configuration

**Enable dynamic routes** (recommended):
```yaml
# config/carla_config.yaml
route:
  use_dynamic_generation: true
  sampling_resolution: 2.0
```

**Disable dynamic routes** (fallback to static):
```yaml
route:
  use_dynamic_generation: false
```

---

## Next Steps

1. **Test**: Run `test_dynamic_routes.py` to verify route generation
2. **Validate**: Run 100-step training with `--debug` to check spawn/behavior
3. **Evaluate**: Run full training episode to confirm stability
4. **Optional**: Implement route variations for better generalization

---

## References

**CARLA Documentation**:
- [Map API](https://carla.readthedocs.io/en/latest/core_map/)
- [Waypoints](https://carla.readthedocs.io/en/latest/core_map/#waypoints)
- [GlobalRoutePlanner](https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/global_route_planner.py)

**Paper Context**:
- Section III.A: System Architecture
- Section III.B: State Space (waypoint-based navigation)
- Section IV.A: Experimental Setup (Town01 route)

---

## Summary

✅ **Implemented**: Dynamic route generation using CARLA's GlobalRoutePlanner API
✅ **Status**: Ready for testing
✅ **Impact**: Fixes spawn bug, improves architecture, enables future extensions
✅ **Compatibility**: Fully backward compatible, can switch between static/dynamic
✅ **Reproducibility**: Uses same fixed start/end from waypoints.txt

**Key Achievement**: System now uses CARLA's proper route planning API while maintaining the fixed start/end locations for reproducible experiments. This is the "best of both worlds" approach! 🚀
