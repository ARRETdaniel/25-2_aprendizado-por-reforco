# Phase 2: Debug Visualization Implementation

**Date**: 2025-01-21
**Status**: ✅ COMPLETED
**Task**: Add visual debug window to baseline evaluation (matching train_td3.py pattern)

---

## 1. Implementation Summary

Successfully implemented OpenCV-based debug visualization for `scripts/evaluate_baseline.py`, providing real-time visual feedback during baseline controller evaluation. The implementation follows the **exact pattern** from `train_td3.py` for consistency.

### 1.1 Changes Made

**File**: `scripts/evaluate_baseline.py`

**Lines Modified**: ~200 lines added

**Key Components**:
1. ✅ OpenCV import (`import cv2`)
2. ✅ Debug window setup in `__init__()` method
3. ✅ `_display_debug_frame()` method implementation
4. ✅ Integration into evaluation loop
5. ✅ Window cleanup on exit

---

## 2. Implementation Details

### 2.1 Import Addition

```python
# Line 39
import cv2
```

### 2.2 Window Setup (__init__ method)

**Location**: Lines 181-191

```python
# Setup debug visualization if enabled
if self.debug:
    print(f"\n[DEBUG MODE ENABLED]")
    print(f"[DEBUG] Visual feedback enabled (OpenCV display)")
    print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")

    # Setup OpenCV window
    self.window_name = "Baseline Evaluation - Debug View"
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.window_name, 1200, 600)  # 800px camera + 400px info
    self.paused = False
```

**Parameters**:
- Window name: "Baseline Evaluation - Debug View"
- Window size: 1200x600 pixels
  - Camera view: 800x600 (left panel)
  - Info panel: 400x600 (right panel)
- Resizable: Yes (`cv2.WINDOW_NORMAL`)

### 2.3 Display Debug Frame Method

**Location**: Lines 226-406

**Method Signature**:
```python
def _display_debug_frame(self, obs_dict, control, info, step, episode_reward):
    """
    Display debug visualization for baseline evaluation.

    Shows:
    - Camera view (800x600) from front-facing camera
    - Info panel (400x600) with control/state information

    Args:
        obs_dict: Observation dictionary with 'image' key (4-frame stack, shape: (4, 84, 84))
        control: CARLA VehicleControl object with steer, throttle, brake
        info: Info dictionary from environment
        step: Current step number
        episode_reward: Cumulative episode reward
    """
```

**Key Features**:

#### A. Camera Frame Processing

```python
# Extract latest frame from 4-frame stack
latest_frame = obs_dict['image'][-1]  # Shape: (84, 84) grayscale

# CRITICAL: Denormalize from [-1, 1] to [0, 1]
# (Environment normalizes images to [-1, 1] range)
latest_frame_denorm = (latest_frame + 1.0) / 2.0

# Convert to uint8 [0, 255]
frame_uint8 = (latest_frame_denorm * 255).astype(np.uint8)

# Resize for display (84x84 -> 800x600)
frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

# Convert grayscale to BGR for color overlay
display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)
```

**Why Denormalization is Critical**:
- The `CARLANavigationEnv` normalizes camera images to `[-1, 1]` range for neural network input
- OpenCV expects `[0, 255]` uint8 range
- Without denormalization, the displayed image would be black (negative values clipped to 0)

#### B. Info Panel Layout

**Panel Size**: 400x600 pixels
**Background**: Black (`np.zeros((600, 400, 3), dtype=np.uint8)`)

**Content Sections** (from top to bottom):

1. **Title** (Yellow, line 266-268):
   - Text: "BASELINE EVALUATION - DEBUG"
   - Font scale: 0.6, Thickness: 2

2. **Separator Line** (line 272):
   - Horizontal divider below title

3. **Step Counter** (line 276-277):
   - Display: `Step: <step_number>`

4. **Control Commands** (Green header, lines 280-295):
   - Steering: `+/-0.XXX` (signed, 3 decimals)
   - Throttle: `0.XXX` (3 decimals)
   - Brake: `0.XXX` (3 decimals)

5. **Vehicle State** (Green header, lines 298-313):
   - Speed: `XX.XX km/h` (2 decimals)
   - Lateral Deviation: `±X.XXX m` (3 decimals)
   - Heading Error: `±XX.XX deg` (converted from radians, 2 decimals)

6. **Reward** (Green header, lines 316-322):
   - Episode Reward: `±XXX.XX` (cumulative, 2 decimals)

7. **Progress** (Green header, lines 325-337):
   - Distance to Goal: `XXX.XX m` (2 decimals)
   - Waypoint Index: Integer counter

8. **Safety Metrics** (Green header, lines 340-355):
   - Collision Count: Integer (RED if > 0)
   - Lane Invasion Count: Integer (BLUE if > 0)

9. **Controls** (Green header, lines 358-368):
   - 'q' - Quit
   - 'p' - Pause/Unpause

**Color Coding**:
- Normal text: White `(255, 255, 255)`
- Section headers: Green `(0, 255, 0)`
- Title: Yellow `(0, 255, 255)`
- Collision count (if > 0): Red `(0, 0, 255)`
- Lane invasion count (if > 0): Blue `(255, 0, 0)`

#### C. Keyboard Controls

**Implementation** (lines 371-391):

```python
key = cv2.waitKey(1) & 0xFF

if key == ord('q'):
    print(f"\n[DEBUG] User requested quit (pressed 'q')")
    cv2.destroyAllWindows()
    sys.exit(0)
elif key == ord('p'):
    self.paused = not self.paused
    if self.paused:
        print(f"\n[DEBUG] PAUSED - Press 'p' to resume")
        while self.paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('p'):
                self.paused = False
                print(f"[DEBUG] RESUMED")
            elif key == ord('q'):
                print(f"\n[DEBUG] User requested quit (pressed 'q')")
                cv2.destroyAllWindows()
                sys.exit(0)
```

**Supported Keys**:
- **'q'**: Quit evaluation immediately and close all windows
- **'p'**: Pause/unpause evaluation loop
  - When paused, enters blocking loop checking for 'p' (resume) or 'q' (quit)
  - Prints status messages to terminal

**Error Handling** (lines 393-396):
```python
except Exception as e:
    print(f"[DEBUG] Visualization error: {e}")
    import traceback
    traceback.print_exc()
```

### 2.4 Integration into Evaluation Loop

**Location**: Lines 456-460

```python
next_obs_dict, reward, done, truncated, info = self.env.step(action)

# Display debug visualization if enabled
if self.debug:
    self._display_debug_frame(next_obs_dict, control, info, episode_length, episode_reward)

# Collect metrics
episode_reward += reward
```

**Timing**: Display is updated **after** each environment step, showing the most recent observation and control command.

### 2.5 Window Cleanup

**Location**: Lines 527-530

```python
# Cleanup debug window if enabled
if self.debug:
    cv2.destroyAllWindows()
    print(f"\n[DEBUG] Closed debug window")
```

**When**: Called at the end of `evaluate()` method after all episodes are complete.

---

## 3. Usage

### 3.1 Command Line

**Enable Debug Mode**:
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

**Without Debug Mode** (default):
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml
```

### 3.2 Docker Execution (with X11 Forwarding)

**Prerequisites**:
```bash
# Allow Docker to access X11 display
xhost +local:docker
```

**Run with Debug Window**:
```bash
cd /path/to/av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

**X11 Environment Variables**:
- `DISPLAY=$DISPLAY`: Forward host display to container
- `QT_X11_NO_MITSHM=1`: Disable shared memory for X11 (required for Docker)
- `-v /tmp/.X11-unix:/tmp/.X11-unix:rw`: Mount X11 socket
- `--privileged`: Required for X11 access

**Cleanup**:
```bash
# Revoke X11 access after testing
xhost -local:docker
```

---

## 4. Verification Checklist

### 4.1 Syntax Check

✅ **Python Compilation**:
```bash
python3 -m py_compile scripts/evaluate_baseline.py
# Result: No syntax errors
```

### 4.2 Code Review

✅ **Import Statement**: cv2 imported at line 39
✅ **Window Setup**: Properly initialized in `__init__` when `debug=True`
✅ **Display Method**: Complete implementation with camera + info panel
✅ **Loop Integration**: Called after `env.step()` with correct arguments
✅ **Cleanup**: Window destroyed at end of evaluation
✅ **Error Handling**: Try-except block catches visualization errors

### 4.3 Pattern Consistency

Comparison with `train_td3.py`:

| Feature | train_td3.py | evaluate_baseline.py | Match |
|---------|--------------|---------------------|-------|
| Window size | 1200x600 | 1200x600 | ✅ |
| Camera panel | 800x600 | 800x600 | ✅ |
| Info panel | 400x600 | 400x600 | ✅ |
| Frame denormalization | `(frame + 1.0) / 2.0` | `(frame + 1.0) / 2.0` | ✅ |
| Resize interpolation | `INTER_LINEAR` | `INTER_LINEAR` | ✅ |
| Color conversion | `GRAY2BGR` | `GRAY2BGR` | ✅ |
| 'q' key handling | `sys.exit(0)` | `sys.exit(0)` | ✅ |
| 'p' key handling | Pause loop | Pause loop | ✅ |
| Window cleanup | `cv2.destroyAllWindows()` | `cv2.destroyAllWindows()` | ✅ |

---

## 5. Testing Plan

### 5.1 Phase 2 Test Execution

**From** `INTEGRATION_TESTING_PLAN.md`:

**Goal**: Verify control commands with visual debugging

**Test Command**:
```bash
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

**Expected Behavior**:
1. ✅ CARLA connects successfully
2. ✅ Vehicle spawns at initial waypoint
3. ✅ Debug window opens showing:
   - Camera feed from front-facing camera
   - Real-time control values (steering, throttle, brake)
   - Vehicle state (speed, lateral deviation, heading error)
   - Progress metrics (distance to goal, waypoint index)
   - Safety counters (collisions, lane invasions)
4. ✅ Pressing 'p' pauses/unpauses evaluation
5. ✅ Pressing 'q' exits cleanly

**Success Criteria**:
- Control values in valid ranges (steering [-1,1], throttle/brake [0,1])
- Speed converges toward target (30 km/h)
- Camera feed shows Town01 road
- No NaN or Inf in display

### 5.2 Diagnostic Use Cases

**Use Case 1: Speed Convergence Debugging**
- **Problem**: Vehicle not reaching target speed (Phase 1 showed 17 km/h vs 30 km/h target)
- **Debug Strategy**: Watch throttle/brake values in info panel
  - If throttle < 0.5 consistently → Increase PID kp gain
  - If brake activating unexpectedly → Check PID integral term

**Use Case 2: Steering Oscillation**
- **Problem**: Vehicle weaving between lane lines
- **Debug Strategy**: Watch steering value in info panel
  - If oscillating rapidly (e.g., -0.3 ↔ +0.3) → Reduce kp_heading gain
  - Check heading error display for large fluctuations

**Use Case 3: Waypoint Following**
- **Problem**: Vehicle deviating from route
- **Debug Strategy**:
  - Watch lateral deviation in info panel (should be < 1.0m typically)
  - Check waypoint index incrementing
  - Observe camera feed to see if waypoints align with road

---

## 6. Known Limitations

### 6.1 Current Issues

1. **Frame Stack Assumption**:
   - Code assumes `obs_dict['image']` is a 4-frame stack (shape: 4, 84, 84)
   - Uses `[-1]` to extract latest frame
   - **Mitigation**: If environment changes to single frame, update to `obs_dict['image']` directly

2. **Info Dictionary Dependencies**:
   - Relies on specific keys from `carla_env.py`:
     - `'speed'`, `'lateral_deviation'`, `'heading_error'`
     - `'distance_to_goal'`, `'waypoint_index'`
     - `'collision_count'`, `'lane_invasion_count'`
   - **Mitigation**: Use `.get()` with default values (already implemented)

3. **Docker X11 Permissions**:
   - Requires `xhost +local:docker` on host
   - Security consideration: Disables some X11 access control
   - **Mitigation**: Use `xhost -local:docker` after testing

### 6.2 Future Enhancements

1. **Overlay Waypoints on Camera**:
   - Project next waypoint onto camera image
   - Draw target path overlay
   - Requires camera intrinsics and transformation matrices

2. **Control Value Graphs**:
   - Real-time plots of steering/throttle/brake over time
   - Speed tracking graph (current vs. target)
   - Requires additional plotting library (e.g., matplotlib in separate window)

3. **Bird's Eye View**:
   - Top-down view of vehicle position relative to waypoints
   - Requires additional rendering logic

---

## 7. Related Files

**Modified**:
- `scripts/evaluate_baseline.py` (200+ lines added)

**Referenced**:
- `train_td3.py` (debug pattern source)
- `src/environment/carla_env.py` (observation format, info keys)
- `config/baseline_config.yaml` (controller parameters)
- `docs/day-23/baseline/INTEGRATION_TESTING_PLAN.md` (test plan)
- `docs/RUN-COMMAND.md` (Docker X11 forwarding)

**Dependencies**:
- OpenCV (cv2) - already in requirements.txt
- NumPy - for image processing
- sys - for exit handling

---

## 8. Next Steps

**Phase 2 Continuation**:

1. ✅ **Debug visualization implemented** (this document)

2. ⏳ **Run Phase 2 test** (with debug window):
   ```bash
   # Test with debug visualization
   docker run [X11 flags] ... --debug
   ```

3. ⏳ **Analyze control behavior**:
   - Observe throttle/brake response to speed error
   - Check steering response to waypoint direction
   - Verify speed convergence toward 30 km/h

4. ⏳ **Tune controller if needed**:
   - Adjust PID gains in `config/baseline_config.yaml`
   - Adjust Pure Pursuit lookahead/heading gain
   - Re-run until satisfactory performance

5. ❌ **Proceed to Phase 3** (Waypoint Following):
   - Enable trajectory saving
   - Visualize actual path vs. reference waypoints
   - Calculate crosstrack error statistics

---

## 9. Conclusion

Successfully implemented visual debug capabilities for baseline controller evaluation, matching the pattern established in `train_td3.py`. The implementation provides:

- ✅ Real-time camera feed visualization
- ✅ Comprehensive control/state information overlay
- ✅ Interactive pause/quit controls
- ✅ Docker compatibility with X11 forwarding
- ✅ Error handling and cleanup
- ✅ Consistent with TD3 training debug pattern

**Ready for Phase 2 testing** with visual debugging enabled.

---

**Author**: AI Assistant (based on user requirements)
**Date**: 2025-01-21
**Review Status**: Ready for testing
