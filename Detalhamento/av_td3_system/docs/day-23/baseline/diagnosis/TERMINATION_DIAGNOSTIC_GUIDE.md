# Episode Termination Diagnostic Guide
**Date:** 2025-11-23  
**Issue:** Episodes terminating early (step 350, 510, 580 instead of completing ~86 waypoint route)  
**Status:** Diagnostic logging added to investigate

---

## Problem Description

Baseline controller evaluation episodes are ending prematurely:
- **Episode 1**: 510 steps, collision with NPC ✅ (expected)
- **Episode 2**: 350 steps, collision with NPC ✅ (expected)  
- **Episode 3**: **580 steps, NO collision, NO lane invasion, but Success: False** ❌ (INVESTIGATING)

Episode 3 reached waypoint 68/86 (79% complete) and terminated without obvious cause.

---

## Diagnostic Logging Added

### 1. Environment Termination Logging (`src/environment/carla_env.py`)

Added detailed logging in `_check_termination()`:

```python
# Collision detection
if self.sensors.is_collision_detected():
    self.logger.warning(f"[TERMINATION] Collision detected at step {self.current_step}")
    return True, "collision"

# Off-road detection (>2.0m lateral deviation)
if lateral_deviation > 2.0:
    self.logger.warning(
        f"[TERMINATION] Off-road at step {self.current_step}: "
        f"lateral_deviation={lateral_deviation:.3f}m > 2.0m threshold"
    )
    return True, "off_road"

# Route completion
if self.waypoint_manager.is_route_finished():
    self.logger.info(
        f"[TERMINATION] Route completed at step {self.current_step}! "
        f"Waypoint {current_idx}/{total_waypoints-1}"
    )
    return True, "route_completed"
```

Added truncation logging in `step()`:

```python
# Approaching max steps warning
if self.current_step >= self.max_episode_steps - 10 and not done:
    self.logger.warning(
        f"[DIAGNOSTIC] Approaching max_episode_steps! "
        f"Step {self.current_step}/{self.max_episode_steps}"
    )

# Truncation logging
if truncated:
    self.logger.warning(
        f"[TERMINATION] Episode TRUNCATED at step {self.current_step} "
        f"(reached max_episode_steps={self.max_episode_steps})"
    )
```

### 2. Evaluation Script Logging (`scripts/evaluate_baseline.py`)

Added comprehensive termination summary after each episode:

```python
print(f"\n[TERMINATION] Episode ended after {episode_length} steps:")
print(f"  Reason: {termination_reason}")
print(f"  Done: {done} (natural MDP termination)")
print(f"  Truncated: {truncated} (time limit)")
print(f"  Max Steps: {max_episode_steps}")
print(f"  Success: {bool(success)}")
print(f"  Collisions: {collision_count}")
print(f"  Lane Invasions: {lane_invasion_count}")
print(f"  Final Position: (x, y)")
print(f"  Waypoint Index: {waypoint_idx}/{total_waypoints-1}")
print(f"  Progress: {progress_pct:.1f}%")
print(f"  Distance to Goal: {distance_to_goal:.2f} m")
```

### 3. Success Flag Addition

Added missing `success` flag to environment info dict:

```python
"success": 1 if (termination_reason == "route_completed" and goal_reached) else 0
```

This ensures evaluation script can properly track route completion.

---

## How to Run Diagnostic Test

```bash
cd /path/to/av_td3_system

# Run 3-episode evaluation with full logging
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
    --debug 2>&1 | tee /tmp/diagnostic_test.log
```

---

## What to Look For in Logs

### Expected Output for Each Episode

**Episode with Collision:**
```
[TERMINATION] Collision detected at step 350
[TERMINATION] Episode ended after 350 steps:
  Reason: collision
  Done: True (natural MDP termination)
  Truncated: False (time limit)
```

**Episode with Off-Road:**
```
[TERMINATION] Off-road at step 420: lateral_deviation=2.150m > 2.0m threshold
[TERMINATION] Episode ended after 420 steps:
  Reason: off_road
  Done: True (natural MDP termination)
  Truncated: False (time limit)
```

**Episode Completing Route:**
```
[TERMINATION] Route completed at step 680! Waypoint 85/85
[TERMINATION] Episode ended after 680 steps:
  Reason: route_completed
  Done: True (natural MDP termination)
  Truncated: False (time limit)
  Success: True
  Progress: 100.0%
```

**Episode Hitting Time Limit:**
```
[DIAGNOSTIC] Approaching max_episode_steps! Step 990/1000
[DIAGNOSTIC] Approaching max_episode_steps! Step 995/1000
[TERMINATION] Episode TRUNCATED at step 1000 (reached max_episode_steps=1000)
[TERMINATION] Episode ended after 1000 steps:
  Reason: running
  Done: False (natural MDP termination)
  Truncated: True (time limit)
  Success: False
  Progress: 85.5%
```

---

## Possible Root Causes

### 1. **Sensor Timeout (Most Likely)**
- CARLA camera/collision sensor hanging
- No data arriving, causing silent failure
- **Solution**: Check CARLA server logs, increase timeout limits

### 2. **Off-Road Termination (Check lateral_deviation)**
- Vehicle drifting >2.0m from lane center at intersection
- **Check logs for**: `[TERMINATION] Off-road at step...`
- **Solution**: If found, adjust Pure Pursuit parameters further

### 3. **NPC Blocking**
- Vehicle stuck behind slow/stopped NPC
- Timeout occurring due to no progress
- **Check**: Speed profile at termination step
- **Solution**: Implement NPC avoidance or timeout handling

### 4. **Waypoint Manager Bug**
- `is_route_finished()` returning True prematurely
- **Check logs for**: `[TERMINATION] Route completed...` at step 580
- **Solution**: Verify waypoint index logic

### 5. **Config Mismatch**
- `max_episode_steps` set too low in config files
- **Check**: `baseline_config.yaml` → `training.max_steps`
- **Current**: Should be 1000 (step 580 < 1000, so not this)

---

## Key Diagnostic Questions

When reviewing logs, answer:

1. **What is `termination_reason`?**
   - "collision" → NPC hit (expected)
   - "off_road" → Lateral deviation issue
   - "route_completed" → Waypoint manager bug
   - "running" → Timeout/truncation issue

2. **What are `done` and `truncated` values?**
   - `done=True, truncated=False` → Natural termination
   - `done=False, truncated=True` → Time limit
   - `done=True, truncated=True` → IMPOSSIBLE (logic error)

3. **What is lateral deviation at termination?**
   - If >2.0m → Off-road termination
   - If <2.0m → Not off-road

4. **What is waypoint progress?**
   - Episode 3 at 79% (waypoint 68/86)
   - Should continue to 100%

5. **Are there any WARNING/ERROR messages?**
   - Sensor timeouts
   - CARLA connection issues
   - OpenDRIVE parsing errors

---

## Next Steps Based on Findings

### If "off_road" termination:
- Review lateral deviation values leading up to step 580
- Check if intersection geometry causes >2.0m deviation
- Consider increasing threshold from 2.0m to 2.5m for intersections

### If "route_completed" termination:
- Bug in `waypoint_manager.is_route_finished()`
- Check if waypoint 68 is incorrectly marked as final
- Review waypoint file for duplicates/errors

### If "running" + truncated:
- Max steps reached (but 580 < 1000, so unlikely)
- Check config files for hidden limits

### If sensor timeout (no clear termination):
- CARLA server overwhelmed
- Reduce NPC count (scenario 0 = 20 NPCs)
- Increase sensor update rate tolerance

---

## Files Modified

1. `src/environment/carla_env.py`:
   - Lines ~1095-1115: Added termination logging
   - Lines ~710-730: Added truncation logging
   - Line ~732: Added `success` flag to info dict

2. `scripts/evaluate_baseline.py`:
   - Lines ~490-510: Added comprehensive termination summary

---

## Related Documentation

- `docs/day-23/COMPLETE_INTERSECTION_FIX.md` - Speed lookahead + Pure Pursuit fix
- CARLA Sensor Timeout: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
- Gymnasium Termination API: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

---

## Status

✅ **Diagnostic logging added**  
🔄 **Ready to run test and analyze logs**  
❓ **Root cause: TBD based on test results**

**Run the diagnostic test command above and review the output!**
