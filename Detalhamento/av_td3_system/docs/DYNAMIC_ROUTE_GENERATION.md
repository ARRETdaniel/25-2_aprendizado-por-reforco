# Dynamic Route Generation with CARLA's GlobalRoutePlanner

## Overview

The system now uses **CARLA's GlobalRoutePlanner API** to generate waypoints dynamically using A* pathfinding on the road network topology, instead of relying on static `waypoints.txt` files.

This provides significant benefits:
- ✅ **Topology-aware routes**: Follows actual road network
- ✅ **Correct Z-coordinates**: Automatically at road surface level
- ✅ **Map-agnostic**: Works on any CARLA map (Town01, Town02, etc.)
- ✅ **Future extensibility**: Easy to add route variations, random destinations

**While maintaining**:
- ✅ **Fixed start/end**: Uses first/last waypoints from `waypoints.txt` for reproducibility
- ✅ **Backward compatibility**: Can switch back to static waypoints if needed

---

## How It Works

### 1. Extract Start/End from waypoints.txt

```python
# Read waypoints.txt
route_start = waypoints[0]   # First waypoint: (317.74, 129.49, 8.333)
route_end = waypoints[-1]    # Last waypoint: (92.34, 86.73282, 2.5)
```

**Purpose**: Maintain reproducibility by using the same start and end locations that were manually defined in the original waypoints file.

### 2. Initialize DynamicRouteManager

```python
from src.environment.dynamic_route_manager import DynamicRouteManager

route_manager = DynamicRouteManager(
    carla_world=world,
    start_location=(317.74, 129.49, 8.333),  # From waypoints.txt
    end_location=(92.34, 86.73282, 2.5),     # From waypoints.txt
    sampling_resolution=2.0  # Distance between waypoints (meters)
)
```

**What happens**:
1. Uses `carla.Map.get_waypoint(location, project_to_road=True)` to find road waypoints
2. Initializes CARLA's `GlobalRoutePlanner` with the map
3. Computes A* pathfinding route from start to end
4. Returns list of waypoints at 2m intervals along the route

### 3. Generate Route Using A* Pathfinding

```python
# Inside DynamicRouteManager._generate_route()
route = self.route_planner.trace_route(
    self.start_waypoint.transform.location,
    self.end_waypoint.transform.location
)

# route = [(waypoint1, RoadOption), (waypoint2, RoadOption), ...]
# RoadOption: LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, etc.
```

**Result**: Array of waypoints with:
- **Correct X, Y coordinates**: Following the road network
- **Correct Z coordinates**: At the road surface level
- **Lane information**: Which lane the waypoint is in
- **Road options**: What maneuver to perform (turn, lane change, etc.)

### 4. Spawn Vehicle at Route Start

```python
spawn_transform = route_manager.get_start_transform()

# spawn_transform contains:
# - Location: First waypoint position (at road surface + 0.5m)
# - Rotation: Aligned with road direction
```

**Benefits**:
- No more falling from height (Z is correct)
- Vehicle spawns aligned with road direction
- Consistent with CARLA's recommended spawn method

---

## File Structure

```
av_td3_system/
├── src/
│   └── environment/
│       ├── dynamic_route_manager.py  ← NEW: Dynamic route generation
│       ├── waypoint_manager.py       ← LEGACY: Static waypoints (fallback)
│       └── carla_env.py             ← UPDATED: Uses DynamicRouteManager
├── config/
│   ├── waypoints.txt                ← START/END extracted from here
│   └── carla_config.yaml           ← NEW: use_dynamic_generation flag
└── scripts/
    └── extract_waypoint_start_end.py  ← Utility to view start/end
```

---

## Configuration

**File**: `config/carla_config.yaml`

```yaml
route:
  # Enable dynamic route generation (recommended)
  use_dynamic_generation: true

  # Waypoints file (only START and END are used when dynamic=true)
  waypoints_file: '/workspace/av_td3_system/config/waypoints.txt'

  # Sampling resolution (distance between waypoints in meters)
  sampling_resolution: 2.0  # Default: 2m

  # Waypoint processing (same for both static and dynamic)
  lookahead_distance: 50.0
  num_waypoints_ahead: 10
  use_relative_coordinates: true
```

**To disable dynamic routes** (use old static waypoints):
```yaml
use_dynamic_generation: false  # Falls back to waypoint_manager.py
```

---

## Comparison: Static vs Dynamic Routes

| Feature | Static (waypoints.txt) | Dynamic (GlobalRoutePlanner) |
|---------|------------------------|------------------------------|
| **Z-Coordinates** | Hardcoded (8.333m, 2.5m) | Automatic (road surface) |
| **Route Definition** | 86 manual waypoints | A* pathfinding |
| **Map Support** | Town01 only | Any CARLA map |
| **Route Variations** | Single fixed route | Can generate varied routes |
| **Maintenance** | Manual CSV editing | Zero maintenance |
| **Topology-Aware** | No (just X,Y,Z points) | Yes (follows roads) |
| **Lane Information** | No | Yes (RoadOption, lane ID) |
| **Spawn Height Bug** | ✅ Fixed (with map API) | ✅ Never had bug |

---

## Start/End Locations (from waypoints.txt)

**From**: `/workspace/av_td3_system/config/waypoints.txt`

```python
# First waypoint (route START)
start = (317.74, 129.49, 8.333)  # Z will be corrected by CARLA

# Last waypoint (route END)
end = (92.34, 86.73282, 2.5)  # Z will be corrected by CARLA

# Straight-line distance: ~229.39 meters
# Actual route length (following roads): ~280-300 meters (estimated)
```

**Extract start/end**:
```bash
python3 scripts/extract_waypoint_start_end.py
```

---

## API Reference

### DynamicRouteManager

**Constructor**:
```python
DynamicRouteManager(
    carla_world: carla.World,
    start_location: Tuple[float, float, float],
    end_location: Tuple[float, float, float],
    sampling_resolution: float = 2.0,
    logger: Optional[logging.Logger] = None
)
```

**Key Methods**:

```python
# Get waypoints array (N, 3) - compatible with WaypointManager
waypoints = route_manager.get_waypoints()

# Get spawn transform (at route start, aligned with road)
spawn_transform = route_manager.get_start_transform()

# Get next waypoint index ahead of vehicle
next_idx = route_manager.get_next_waypoint_index(vehicle_location, current_idx)

# Calculate total route length
length = route_manager.get_route_length()  # meters

# Check if route is complete
done = route_manager.is_route_complete(vehicle_location, threshold=5.0)

# Regenerate route with new destination (future feature)
route_manager.regenerate_route(new_end_location)
```

---

## Future Extensions

### 1. Random Routes Per Episode
```python
# In reset():
spawn_points = world.get_map().get_spawn_points()
random_start = random.choice(spawn_points[:10])
random_end = random.choice(spawn_points[10:20])

route_manager.regenerate_route(
    new_end_location=(random_end.location.x, random_end.location.y, 0.0)
)
```

**Benefit**: Agent learns to generalize to different routes, not just one fixed path.

### 2. Multi-Map Training
```python
# Train on multiple maps
maps = ['Town01', 'Town02', 'Town03']
for map_name in maps:
    world = client.load_world(map_name)
    spawn_points = world.get_map().get_spawn_points()

    route_manager = DynamicRouteManager(
        world,
        spawn_points[0],
        spawn_points[-1]
    )
```

**Benefit**: Agent learns map-invariant driving skills.

### 3. Dynamic Goal Updates
```python
# Change destination mid-episode (rerouting)
if condition:
    new_goal = get_new_destination()
    route_manager.regenerate_route(new_goal)
```

**Benefit**: Simulates real-world navigation with changing destinations.

---

## Backward Compatibility

The system maintains full backward compatibility with the original static waypoint system:

1. **Fallback Mode**: Set `use_dynamic_generation: false` in `carla_config.yaml`
2. **Adapter Pattern**: `DynamicRouteManager` provides same interface as `WaypointManager`
3. **Start/End Preservation**: Uses exact same start/end from `waypoints.txt`

**No changes needed** in:
- State calculation (`_get_observation`)
- Reward computation (`reward_calculator`)
- Action processing (`step`)

---

## Testing

**Test dynamic route generation**:
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento

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

**What to check**:
1. ✅ Log shows: "Using DYNAMIC route (GlobalRoutePlanner)"
2. ✅ Route length is reported (~280-300m for Town01)
3. ✅ Waypoint count matches sampling resolution (route_length / 2.0)
4. ✅ No spawn height collisions
5. ✅ Vehicle spawns aligned with road

**Expected output**:
```
[INFO] DynamicRouteManager initialized:
  Start: (317.74, 129.49, 0.52)
  End: (92.34, 86.73, 0.51)
  Total waypoints: 145
  Sampling resolution: 2.0m

[INFO] ✅ Using DYNAMIC route (GlobalRoutePlanner):
   Start: (317.74, 129.49, 0.52)
   Heading: -90.15°
   Route length: ~290m
   Waypoints: 145
```

---

## Troubleshooting

### ImportError: Cannot import GlobalRoutePlanner

**Solution**: Ensure CARLA PythonAPI is in PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/home/carla/carla/PythonAPI/carla
```

### RuntimeError: Could not find route

**Possible causes**:
1. Start/end locations not on drivable roads
2. No path exists between start and end
3. Map not loaded correctly

**Solution**: Check waypoints.txt coordinates are valid for the map.

### Different route each episode

**This is expected** if using random destinations. For fixed routes (training), ensure:
```yaml
use_dynamic_generation: true  # Uses fixed start/end from waypoints.txt
```

---

## References

**CARLA Documentation**:
- [Map API](https://carla.readthedocs.io/en/latest/core_map/)
- [Waypoints](https://carla.readthedocs.io/en/latest/core_map/#waypoints)
- [GlobalRoutePlanner](https://carla.readthedocs.io/en/latest/python_api/#carlaglobalrouteplanner)

**Paper Reference**:
- Section III.A: System Architecture
- Section III.B: State Space (waypoint-based navigation)

---

## Summary

✅ **Implemented**: Dynamic route generation using CARLA's GlobalRoutePlanner
✅ **Maintains**: Fixed start/end from waypoints.txt for reproducibility
✅ **Benefits**: Correct Z-coordinates, topology-aware, map-agnostic
✅ **Compatible**: Drop-in replacement for WaypointManager
✅ **Future-ready**: Easy to add route variations, multi-map support

**Status**: Ready for testing and deployment 🚀
