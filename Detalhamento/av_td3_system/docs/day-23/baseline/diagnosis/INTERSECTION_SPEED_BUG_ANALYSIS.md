# **CRITICAL BUG - Missing Speed Reduction at Intersection**

**Date**: 2025-11-23
**Issue**: Vehicle performs lane invasion at intersection due to excessive speed
**Root Cause**: Controller ignores waypoint speed profile (fixed 30 km/h instead of 9 km/h at turn)
**Status**: 🔴 **BUG IDENTIFIED - FIX NEEDED**

---

## TL;DR

Our baseline controller uses a **FIXED target speed** (30 km/h = 8.33 m/s) for the entire route, **ignoring the waypoint speed profile** that specifies 9 km/h (2.5 m/s) for the intersection turn. This causes the vehicle to enter the turn at high speed, leading to lane invasion.

**The Fix**: Implement `_get_target_speed_from_waypoints()` method to extract speed from the closest waypoint, matching GitHub's `update_desired_speed()`.

---

## The Problem - Lane Invasion at Intersection

### Observed Behavior

```
[PP-DEBUG Step 1360] Pos=(111.93, 129.3009) | Alpha=+4.06° | Steer=+0.0234 ✅
[PP-DEBUG Step 1370] Pos=(106.46, 129.2211) | Alpha=+17.43° | Steer=+0.0976 ⚠️ High steering!
WARNING: Lane invasion detected: [<carla.libcarla.LaneMarking>]
WARNING: [LANE_KEEPING] Lane invasion detected - penalty (-1.0)
WARNING: [SAFETY-OFFROAD] penalty=-10.0
Episode 3 complete: Success=False, Lane Invasions: 0, Avg Speed: 29.87 km/h
```

### Timeline of Events

| Step | X (m) | Y (m) | Alpha | Speed | Waypoint Speed | Issue |
|------|-------|-------|-------|-------|----------------|-------|
| 1360 | 111.93 | 129.30 | +4.06° | ~30 km/h | **2.5 m/s** | Still fast! |
| 1370 | 106.46 | 129.22 | +17.43° | ~30 km/h | **2.5 m/s** | **LANE INVASION** |

**Key Observation**: Vehicle is traveling at ~30 km/h (8.33 m/s) when waypoints specify **9 km/h (2.5 m/s)** for the turn!

---

## Waypoint Speed Profile Analysis

### Waypoint Data (from `config/waypoints.txt`)

```csv
# Straight section (waypoints 0-68): HIGH SPEED
317.74, 129.49, 8.333   ← 30 km/h
...
104.62, 129.49, 8.333   ← 30 km/h

# Transition + Turn (waypoints 69-86): LOW SPEED
98.59, 129.22, 2.5      ← 9 km/h (SPEED CHANGE!)
95.98, 127.76, 2.5      ← 9 km/h
93.88, 125.62, 2.5      ← 9 km/h
92.51, 122.97, 2.5      ← 9 km/h
92.34, 119.99, 2.5      ← 9 km/h
...
92.34, 86.73, 2.5       ← 9 km/h (end of path)
```

**Critical Waypoint**: Index 69 (X=98.59m) - **Speed drops from 8.333 m/s → 2.5 m/s**!

### Speed Profile Visualization

```
Speed (m/s)
9 ┤ ████████████████████████████████████████████  (straight section)
  │                                          ██
  │                                          ██
  │                                          ██
  │                                          ▐▌
3 ┤                                          ▐▌  ████████  (turn)
  │                                          ▐▌  ██████████
  └──────────────────────────────────────────────────────────> X (m)
    317                                    98              92
    (start)                                (turn starts)   (end)
```

**Deceleration Zone**: From X=104m (last 8.333 m/s) to X=98m (first 2.5 m/s)
- Distance: ~6 meters
- Speed change: 8.333 → 2.5 m/s (Δv = 5.833 m/s)
- **Required deceleration**: a = Δv² / (2×d) = 5.833² / 12 ≈ **2.84 m/s²**

This is a **significant braking maneuver** that must be executed before the turn!

---

## GitHub's Working Implementation

### How GitHub Extracts Target Speed

**From `controller2d.py` (lines 55-68)**:

```python
def update_desired_speed(self):
    """
    Find closest waypoint to vehicle and extract its speed.
    This is called BEFORE update_controls() in module_7.py.
    """
    min_idx = 0
    min_dist = float("inf")

    # Find closest waypoint
    for i in range(len(self._waypoints)):
        dist = np.linalg.norm(np.array([
            self._waypoints[i][0] - self._current_x,
            self._waypoints[i][1] - self._current_y
        ]))
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    # Extract speed from closest waypoint
    if min_idx < len(self._waypoints) - 1:
        self._desired_speed = self._waypoints[min_idx][2]  # ← KEY LINE!
        return min_idx
    else:
        self._desired_speed = self._waypoints[-1][2]
        return -1
```

**From `module_7.py` (lines 655-660)**:

```python
# Update controller with latest state
controller.update_values(current_x, current_y, current_yaw,
                         current_speed, current_timestamp, frame)

# Compute controls
controller.update_controls()  # ← Calls update_desired_speed() internally!
cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
```

**Key Insight**: GitHub **ALWAYS** uses the speed from the closest waypoint, dynamically updating every frame!

---

## Our Broken Implementation

### Current Code (baseline_controller.py)

```python
def __init__(self, ..., target_speed: float = 30.0):  # km/h
    """Initialize controller with FIXED target speed."""
    ...
    self.target_speed = target_speed / 3.6  # Convert to m/s (8.33 m/s)

def compute_control(self, vehicle, waypoints, dt, target_speed=None):
    """Compute control commands."""
    ...
    # STEP 2: Determine target speed
    if target_speed is None:
        target_speed = self.target_speed  # ← FIXED 8.33 m/s! ❌

    # STEP 3: PID control with WRONG target speed
    throttle, brake = self.pid_controller.update(
        current_speed=current_speed,
        target_speed=target_speed,  # ← Always 8.33 m/s!
        dt=dt
    )
```

### The Problem

1. **Initialization**: Sets `target_speed = 30 km/h = 8.33 m/s`
2. **compute_control()**: Uses fixed value (line 385)
3. **PID Controller**: Tries to maintain 8.33 m/s **everywhere**
4. **At intersection**: Should slow to 2.5 m/s, but maintains 8.33 m/s
5. **Result**: Enters turn at 3.3× the safe speed → lane invasion!

---

## Why This Causes Lane Invasion

### Physics of High-Speed Turns

**Centripetal Acceleration Required**:

$$a_c = \frac{v^2}{R}$$

Where:
- $v$ = vehicle speed
- $R$ = turn radius
- $a_c$ = lateral acceleration required to follow path

**Turn Analysis** (from waypoints):
- Turn radius: ~10m (estimated from waypoint curvature)
- **At 2.5 m/s (safe speed)**: $a_c = 2.5^2 / 10 = 0.625$ m/s² ✅ Safe
- **At 8.33 m/s (current speed)**: $a_c = 8.33^2 / 10 = 6.94$ m/s² ❌ **UNSAFE!**

**Vehicle Limits**:
- Typical max lateral acceleration: 0.8-1.0 g ≈ 8 m/s² (race car)
- Comfortable limit: 0.3-0.4 g ≈ 3 m/s² (passenger car)
- **Our requirement**: 6.94 m/s² → Very close to limit!

**Result**: Even if vehicle CAN physically make the turn, it requires:
1. **High steering angle** → Alpha = +17.43° (vs normal 0-5°)
2. **Large lateral deviation** → Crosses lane markings
3. **Risk of instability** → Tire slip possible

---

## The Fix - Extract Speed from Waypoints

### Proposed Implementation

Add method to baseline_controller.py:

```python
def _get_target_speed_from_waypoints(
    self,
    current_x: float,
    current_y: float,
    waypoints: List[Tuple[float, float, float]]
) -> float:
    """
    Extract target speed from closest waypoint.

    Matches GitHub's update_desired_speed() logic:
    1. Find waypoint closest to vehicle position
    2. Return speed (3rd element) from that waypoint
    3. Fallback to last waypoint if at end of path

    This enables speed profile following:
    - High speed (8.333 m/s) on straight sections
    - Low speed (2.5 m/s) at intersections/curves
    - Smooth transitions between speed zones

    Args:
        current_x: Vehicle X position in meters
        current_y: Vehicle Y position in meters
        waypoints: List of (x, y, speed_m_s) tuples

    Returns:
        target_speed: Speed from closest waypoint in m/s

    Example:
        >>> waypoints = [(100, 129, 8.333), (98, 129, 2.5), (95, 127, 2.5)]
        >>> # Vehicle at X=99m (between waypoints)
        >>> speed = self._get_target_speed_from_waypoints(99, 129, waypoints)
        >>> print(speed)  # 2.5 m/s (closest to 2nd waypoint)
    """
    if len(waypoints) == 0:
        return self.target_speed  # Fallback to default

    waypoints_np = np.array(waypoints)

    # Find closest waypoint to vehicle
    distances = np.sqrt(
        (waypoints_np[:, 0] - current_x)**2 +
        (waypoints_np[:, 1] - current_y)**2
    )
    closest_index = np.argmin(distances)

    # Extract speed from closest waypoint (3rd column)
    target_speed = waypoints_np[closest_index, 2]

    return target_speed
```

### Updated compute_control()

```python
def compute_control(self, vehicle, waypoints, dt, target_speed=None):
    """Compute control commands."""
    ...
    # STEP 2: Determine target speed (NEW IMPLEMENTATION!)
    if target_speed is None:
        # Extract from waypoints instead of using fixed value
        target_speed = self._get_target_speed_from_waypoints(
            current_x=current_x,
            current_y=current_y,
            waypoints=waypoints  # Full waypoint list
        )

    # STEP 3: PID control with CORRECT target speed
    throttle, brake = self.pid_controller.update(
        current_speed=current_speed,
        target_speed=target_speed,  # ← Now varies with position!
        dt=dt
    )
```

---

## Expected Behavior After Fix

### Speed Profile Tracking

```
Step 1200: X=150m → closest WP speed=8.333 m/s → PID target=8.333 m/s ✅
Step 1300: X=110m → closest WP speed=8.333 m/s → PID target=8.333 m/s ✅
Step 1350: X=100m → closest WP speed=2.5 m/s   → PID target=2.5 m/s ✅ BRAKING!
Step 1360: X=98m  → closest WP speed=2.5 m/s   → PID target=2.5 m/s ✅ Slowed
Step 1370: X=95m  → closest WP speed=2.5 m/s   → PID target=2.5 m/s ✅ Safe turn
```

### Deceleration Timeline

| Step | X (m) | Closest WP | WP Speed | Vehicle Speed | Brake |
|------|-------|------------|----------|---------------|-------|
| 1340 | 105 | WP 68 (104m, 8.333) | 8.333 m/s | 8.2 m/s | 0.0 |
| 1345 | 102 | WP 69 (98m, 2.5) | **2.5 m/s** | 7.5 m/s | **0.7** |
| 1350 | 100 | WP 69 (98m, 2.5) | 2.5 m/s | 6.0 m/s | 0.8 |
| 1355 | 98 | WP 69 (98m, 2.5) | 2.5 m/s | 4.0 m/s | 0.5 |
| 1360 | 96 | WP 70 (95m, 2.5) | 2.5 m/s | 2.8 m/s | 0.1 |
| 1370 | 93 | WP 71 (93m, 2.5) | 2.5 m/s | **2.5 m/s** | 0.0 ✅ |

**Key Changes**:
1. **Step 1345**: PID detects speed error (+5 m/s) → applies brake
2. **Steps 1345-1360**: Vehicle decelerates from 8.3 → 2.5 m/s
3. **Step 1370**: Enters turn at SAFE speed (2.5 m/s)
4. **Result**: Alpha stays < 10°, no lane invasion! ✅

---

## Comparison with GitHub

### GitHub's Approach ✅

```python
# module_7.py (line 656)
controller.update_values(current_x, current_y, current_yaw, current_speed, ...)
controller.update_controls()  # ← Calls update_desired_speed() inside

# controller2d.py (line 93)
def update_controls(self):
    ...
    min_index = self.update_desired_speed()  # ← Extract from waypoints!
    v_desired = self._desired_speed  # ← Use extracted value

    # PID uses v_desired (varies with position)
    throttle_output = k_p * (v_desired - v) + ...
```

**Advantages**:
- ✅ Dynamic speed adaptation
- ✅ Follows waypoint speed profile
- ✅ Safe intersection navigation
- ✅ Efficient (one distance calculation per update)

### Our Current Approach ❌

```python
# baseline_controller.py (line 385)
if target_speed is None:
    target_speed = self.target_speed  # ← FIXED VALUE!

# PID always uses 8.33 m/s
throttle, brake = self.pid_controller.update(
    current_speed=current_speed,
    target_speed=8.33,  # ← WRONG!
    dt=dt
)
```

**Problems**:
- ❌ Ignores waypoint speed profile
- ❌ Dangerous at intersections
- ❌ Fails safety requirements
- ❌ Not comparable to GitHub baseline

---

## Implementation Priority

**Severity**: 🔴 **CRITICAL** - Causes episode failures, safety violations

**Impact**:
- Prevents successful episode completion
- Invalidates baseline comparison (unfair to DRL if baseline fails)
- Violates safety requirements (lane invasion)

**Effort**: 🟢 **LOW** - ~30 lines of code, 15 minutes to implement

**Dependencies**: None (self-contained fix)

---

## Next Steps

1. ✅ **Implement `_get_target_speed_from_waypoints()` method**
2. ✅ **Update `compute_control()` to use extracted speed**
3. ✅ **Test with 3-episode evaluation**
4. ✅ **Verify speed reduction at intersection** (check logs for target_speed)
5. ✅ **Confirm no lane invasions**
6. ✅ **Measure episode success rate**

**Expected Results**:
- Vehicle slows to 2.5 m/s before turn
- Alpha stays < 10° during turn
- No lane invasions
- Episode success rate: 100% (vs current ~0%)

---

## Lessons Learned

1. **Read the complete GitHub implementation** - We missed `update_desired_speed()`!
2. **Waypoint format matters** - 3rd column is speed, not just decoration
3. **Fixed parameters can break** - Speed varies by scenario
4. **Test edge cases** - Straight paths worked fine, intersection exposed bug
5. **Safety requires attention to detail** - High-speed turns are dangerous

---

**Status**: Ready to implement fix! 🚀

This is the SECOND critical bug discovered through systematic debugging:
1. **Bug #1**: Missing waypoint filtering → left drift (FIXED ✅)
2. **Bug #2**: Ignoring waypoint speeds → intersection failures (FIXING NOW 🔧)

Both bugs were in GitHub's working code but missing from our implementation. This highlights the importance of complete code porting!
