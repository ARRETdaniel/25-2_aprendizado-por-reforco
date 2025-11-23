# Quick Start: Baseline Evaluation with Debug Visualization

**Date**: 2025-01-21  
**Purpose**: Quick reference for running baseline evaluation with visual debug window

---

## Prerequisites

1. **CARLA Server Running**:
   ```bash
   docker run --rm --network host --gpus all carlasim/carla:0.9.16 ./CarlaUE4.sh
   ```

2. **X11 Access Granted**:
   ```bash
   xhost +local:docker
   ```

---

## Quick Commands

### Test Run (1 Episode with Debug Window)

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && \
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

### Without Debug Window

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml
```

---

## Debug Window Controls

| Key | Action |
|-----|--------|
| `q` | Quit evaluation immediately |
| `p` | Pause/unpause evaluation |

---

## Debug Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌────────────────────────┐  ┌───────────────────────────────┐ │
│  │                        │  │ BASELINE EVALUATION - DEBUG   │ │
│  │                        │  ├───────────────────────────────┤ │
│  │                        │  │ Step: 123                     │ │
│  │   Camera View          │  │                               │ │
│  │   (800x600)            │  │ CONTROL COMMANDS:             │ │
│  │                        │  │   Steering:  +0.123           │ │
│  │   Front-facing         │  │   Throttle:   0.456           │ │
│  │   grayscale camera     │  │   Brake:      0.000           │ │
│  │                        │  │                               │ │
│  │                        │  │ VEHICLE STATE:                │ │
│  │                        │  │   Speed:      27.34 km/h      │ │
│  │                        │  │   Lat Dev:    -0.123 m        │ │
│  │                        │  │   Head Err:   -5.67 deg       │ │
│  │                        │  │                               │ │
│  │                        │  │ REWARD:                       │ │
│  │                        │  │   Episode:    123.45          │ │
│  │                        │  │                               │ │
│  │                        │  │ PROGRESS:                     │ │
│  └────────────────────────┘  │   Dist Goal:  234.56 m        │ │
│                              │   Waypoint:   12              │ │
│  1200px x 600px              │                               │ │
│                              │ SAFETY:                       │ │
│                              │   Collisions:     0           │ │
│                              │   Lane Inv:       2           │ │
│                              │                               │ │
│                              │ CONTROLS:                     │ │
│                              │   'q' - Quit                  │ │
│                              │   'p' - Pause/Unpause         │ │
│                              └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Debug Window Not Appearing

**Cause**: X11 forwarding not configured

**Solution**:
```bash
# 1. Grant X11 access
xhost +local:docker

# 2. Verify DISPLAY variable
echo $DISPLAY  # Should show :0 or :1

# 3. Check docker run includes:
#    -e DISPLAY=$DISPLAY
#    -e QT_X11_NO_MITSHM=1
#    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
#    --privileged
```

### "Cannot Open Display" Error

**Solution**:
```bash
# Allow local connections
xhost +local:

# Or allow all (less secure)
xhost +
```

### Window Opens but Shows Black Screen

**Cause**: Camera denormalization issue or environment not providing images

**Debug**:
1. Check `obs_dict['image']` is not None
2. Verify normalization range (should be [-1, 1])
3. Check error messages in terminal

---

## Expected Output

### Terminal Output (with Debug)

```
======================================================================
BASELINE CONTROLLER EVALUATION - PID + PURE PURSUIT
======================================================================

[CONFIG] Loading configurations...
[CONFIG] CARLA config: config/carla_config.yaml
[CONFIG] Baseline config: config/baseline_config.yaml
[CONFIG] Scenario: 0 (0=20, 1=50, 2=100 NPCs)
[CONFIG] Set CARLA_SCENARIO_INDEX=0
[CONFIG] Expected NPC count: 20

[ENV] Creating CARLA navigation environment...
[ENV] Environment created successfully

[CONTROLLER] Initializing baseline controller (PID + Pure Pursuit)
[CONTROLLER] PID gains: kp=0.5, ki=0.3, kd=0.13
[CONTROLLER] Pure Pursuit: lookahead=2.0m, kp_heading=8.0
[CONTROLLER] Target speed: 30 km/h

[DEBUG MODE ENABLED]
[DEBUG] Visual feedback enabled (OpenCV display)
[DEBUG] Press 'q' to quit, 'p' to pause/unpause

[INIT] Baseline evaluation pipeline ready!
[INIT] Number of episodes: 1
[INIT] Seed: 42
[INIT] Debug mode: True
======================================================================

[EVAL] Starting evaluation...

[EVAL] Episode 1/1
[ENV] Resetting environment...
[ENV] Episode reset complete

[EVAL] Episode 1 complete:
       Reward: 123.45
       Success: True
       Collisions: 0
       Lane Invasions: 2
       Length: 456 steps
       Avg Speed: 27.34 km/h

======================================================================
EVALUATION SUMMARY
======================================================================

Safety Metrics:
  Success Rate: 100.0%
  Avg Collisions: 0.00 ± 0.00
  Avg Lane Invasions: 2.00 ± 0.00

Efficiency Metrics:
  Mean Reward: 123.45 ± 0.00
  Avg Speed: 27.34 ± 0.00 km/h
  Avg Episode Length: 456.0 ± 0.0 steps

[DEBUG] Closed debug window
```

---

## Performance Monitoring

### What to Look For

**Good Indicators**:
- Speed converges to 30 km/h (target)
- Steering values smooth (not oscillating)
- Lateral deviation < 1.0m
- No collisions

**Bad Indicators**:
- Speed stuck at low value (e.g., 15-20 km/h) → Tune PID throttle
- Steering oscillating wildly → Reduce heading gain
- Large lateral deviation → Check waypoint alignment
- Frequent lane invasions → Adjust lateral control

---

## Cleanup

```bash
# Revoke X11 access after testing
xhost -local:docker
```

---

## Next Steps

After successful Phase 2 test:

1. **Analyze control behavior** (from debug window observations)
2. **Tune controller parameters** (if needed) in `config/baseline_config.yaml`
3. **Proceed to Phase 3** (Waypoint Following with trajectory analysis)

---

**Quick Reference**: See `PHASE2_DEBUG_VISUALIZATION_IMPLEMENTATION.md` for detailed implementation details
