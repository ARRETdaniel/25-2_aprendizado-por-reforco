# Baseline Controller Tuning Plan

**Date**: November 23, 2025  
**Status**: 🔧 **IN PROGRESS - TUNING PHASE**  
**Priority**: HIGH - Required to reduce zigzag behavior

---

## Executive Summary

Phase 3 analysis revealed significant zigzag behavior and potential PID overshooting in the baseline controller. This document outlines the systematic tuning approach to resolve these issues while maintaining fair comparison with TD3.

**Key Findings from Phase 3**:
- ✅ Lane invasion fix working (0.00 lane invasions)
- ⚠️ **Zigzag behavior**: Mean lateral deviation 0.865m, max 1.980m (approaching 2.0m threshold)
- ⚠️ **Heading oscillations**: Mean 9.74°, max 33.45°
- ⚠️ **Aggressive steering**: High-frequency switching between ±1.0 saturation
- ✅ **Speed tracking excellent**: 0.7% error (PID working well for longitudinal control)

**Root Causes Identified**:
1. **Aggressive heading gain**: `kp_heading = 8.0` → overcorrection
2. **Short lookahead**: `lookahead_distance = 2.0m` @ 30 km/h = 0.24s planning horizon
3. **No speed-crosstrack coupling**: `k_speed_crosstrack = 0.0` → full speed during corrections

---

## Comparison: TCC vs Current Implementation

### Parameter Comparison

| Parameter | TCC (controller2d.py) | Current (baseline_config.yaml) | Status |
|-----------|----------------------|--------------------------------|--------|
| **PID kp** | 0.50 | 0.50 | ✅ Match |
| **PID ki** | 0.30 | 0.30 | ✅ Match |
| **PID kd** | 0.13 | 0.13 | ✅ Match |
| **Lookahead** | 2.0m | 2.0m | ✅ Match |
| **kp_heading** | 8.00 | 8.00 | ✅ Match |
| **k_speed_crosstrack** | 0.00 | 0.00 | ✅ Match |
| **cross_track_deadband** | 0.01 | 0.01 | ✅ Match |

**Observation**: Parameters are **identical** to TCC implementation that didn't exhibit zigzag!

### Why Is Our System Exhibiting Zigzag When TCC Didn't?

**Hypothesis 1: CARLA API Differences** (MOST LIKELY)
- **TCC used CARLA 0.8.x** with different physics engine
- **Current system uses CARLA 0.9.16** with updated vehicle dynamics
- **Steering response may be more sensitive** in 0.9.16
- **Vehicle inertia/friction models may have changed**

**Hypothesis 2: Control Loop Frequency**
- **TCC timestep**: Unknown (likely 0.05s from CARLA 0.8.x default)
- **Current timestep**: 0.05s (confirmed in carla_config.yaml)
- **Potential mismatch in discrete-time PID tuning**

**Hypothesis 3: Waypoint Density/Geometry**
- **Same waypoints.txt file** used
- **CARLA 0.9.16 coordinate system** might have subtle differences
- **Waypoint interpolation** might be handled differently

**Hypothesis 4: Sensor/State Update Latency**
- **TCC**: Direct synchronous API calls
- **Current**: Gymnasium environment wrapper with image processing
- **Additional latency from frame stacking** (4 frames @ 84x84)

---

## Tuning Strategy

### Phase 1: Diagnose Root Cause (15 min)

**Objective**: Determine if issue is Pure Pursuit or PID

**Test 1: Disable PID (Coast Test)**
```yaml
pid:
  kp: 0.0
  ki: 0.0
  kd: 0.0
# Vehicle will coast - observe steering behavior only
```

**Expected**:
- If zigzag persists → Pure Pursuit issue
- If zigzag disappears → PID interference issue

**Test 2: Disable Pure Pursuit (Straight Line Test)**
```yaml
pure_pursuit:
  kp_heading: 0.0
# Vehicle will go straight - observe speed behavior only
```

**Expected**:
- If speed overshoots → PID tuning issue
- If speed stable → PID is fine

---

### Phase 2: Pure Pursuit Tuning (Primary Focus)

**Based on Phase 3 analysis, zigzag is primarily a lateral control issue.**

#### Option A: Conservative Tuning (Recommended First)

**Goal**: Reduce overcorrection without making controller too sluggish

```yaml
pure_pursuit:
  lookahead_distance: 3.0  # ↑ from 2.0 (0.36s @ 30km/h → smoother)
  kp_heading: 6.0          # ↓ from 8.0 (reduce gain by 25%)
  k_speed_crosstrack: 0.0  # KEEP disabled initially
  cross_track_deadband: 0.02  # ↑ from 0.01 (larger deadband)
```

**Rationale**:
- **Lookahead +50%**: Gives controller more "preview" time, reduces reactiveness
- **kp_heading -25%**: Reduces steering command magnitude for same error
- **Deadband +100%**: Reduces micro-corrections near centerline

**Expected Improvement**:
- Lateral deviation: 0.865m → 0.6m (30% reduction)
- Max deviation: 1.980m → 1.4m (30% reduction)
- Heading oscillation: 9.74° → 7.0° (28% reduction)

#### Option B: Aggressive Tuning (If Option A Insufficient)

```yaml
pure_pursuit:
  lookahead_distance: 4.0  # ↑ from 2.0 (0.48s @ 30km/h → very smooth)
  kp_heading: 4.5          # ↓ from 8.0 (reduce gain by 44%)
  k_speed_crosstrack: 0.1  # ENABLE speed reduction when off-center
  cross_track_deadband: 0.05  # ↑ from 0.01 (large deadband)
```

**Rationale**:
- **Lookahead +100%**: Maximum smoothness without losing responsiveness
- **kp_heading -44%**: Significant gain reduction for gentle steering
- **Speed coupling**: Slow down when deviating → safer corrections
- **Large deadband**: Eliminate micro-oscillations entirely

**Expected Improvement**:
- Lateral deviation: 0.865m → 0.4m (54% reduction)
- Max deviation: 1.980m → 1.0m (50% reduction)
- Heading oscillation: 9.74° → 5.0° (49% reduction)

#### Option C: Adaptive Lookahead (Advanced)

**Implementation requires code change** - not just config tuning

```python
# Adaptive lookahead based on speed and lateral error
lookahead_base = 2.0
lookahead_speed_factor = 0.3  # 0.3 * (v in m/s)
lookahead_error_factor = 1.0  # 1.0 * lateral_error

lookahead = lookahead_base + \
            lookahead_speed_factor * current_speed + \
            lookahead_error_factor * abs(lateral_error)
```

**Skip for now** - only if Options A/B fail

---

### Phase 3: PID Tuning (If Needed)

**From Phase 3 speed_profile.png analysis**:
- **No significant overshoot observed** in speed tracking
- **Mean speed error: 0.22 km/h** (0.7% - excellent!)
- **PID appears well-tuned for longitudinal control**

**RECOMMENDATION**: **Skip PID tuning** unless Test 2 reveals issues

If PID tuning is needed:

```yaml
pid:
  kp: 0.4  # ↓ from 0.5 (reduce proportional response)
  ki: 0.2  # ↓ from 0.3 (reduce integral accumulation)
  kd: 0.15 # ↑ from 0.13 (increase damping)
```

---

## Tuning Workflow

### Step 1: Baseline Run (Record Current Performance)

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

**Record**:
- Mean lateral deviation
- Max lateral deviation
- Mean heading error
- Steering command saturation frequency

### Step 2: Apply Option A Tuning

**Edit** `config/baseline_config.yaml`:
```yaml
pure_pursuit:
  lookahead_distance: 3.0
  kp_heading: 6.0
  cross_track_deadband: 0.02
```

**Run evaluation** (same command as Step 1)

**Compare**:
- Lateral deviation reduced?
- Heading error reduced?
- Steering smoother?

### Step 3: Iterate if Needed

**If Option A insufficient** → Apply Option B
**If Option B over-corrects** → Interpolate between A and B

---

## Success Criteria

### Minimum Requirements (Must Achieve)

1. ✅ **Max lateral deviation < 1.5m** (currently 1.980m)
   - Maintain safety margin from 2.0m termination threshold

2. ✅ **Mean lateral deviation < 0.6m** (currently 0.865m)
   - Demonstrate consistent lane-center tracking

3. ✅ **Mean heading error < 7.0°** (currently 9.74°)
   - Reduce oscillatory behavior

4. ✅ **No steering saturation** (currently frequent ±1.0 saturation)
   - Steering commands should stay within [-0.8, 0.8] most of the time

5. ✅ **Maintain speed tracking < 1.0% error** (currently 0.7% - excellent)
   - Don't break what's already working

### Target Goals (Ideal)

1. 🎯 Mean lateral deviation < 0.4m
2. 🎯 Max lateral deviation < 1.0m
3. 🎯 Mean heading error < 5.0°
4. 🎯 Max heading error < 15.0°
5. 🎯 Zero lane marking touches (no SAFETY-OFFROAD warnings)

---

## Validation Protocol

### Test Suite (After Each Tuning Iteration)

**Test 1: 3-Episode Evaluation**
```bash
python3 scripts/evaluate_baseline.py \
  --scenario 0 \
  --num-episodes 3 \
  --baseline-config config/baseline_config.yaml \
  --debug
```

**Test 2: Automatic Analysis**
- Analysis runs automatically after evaluation
- Review trajectory_map.png for visual confirmation
- Check statistics in console output

**Test 3: Regression Check**
- Verify speed tracking still < 1% error
- Verify no new failures introduced
- Verify episode lengths remain > 500 steps

---

## Rollback Plan

If tuning makes performance worse:

```bash
# Restore original parameters
git checkout config/baseline_config.yaml
```

Or manually restore:
```yaml
pure_pursuit:
  lookahead_distance: 2.0
  kp_heading: 8.00
  cross_track_deadband: 0.01
```

---

## Documentation Requirements

### For Each Tuning Iteration

Create file: `docs/day-23/baseline/tuning_iteration_N.md`

**Content**:
1. Parameters changed (before/after)
2. Results comparison (table with metrics)
3. Trajectory map comparison (side-by-side screenshots)
4. Decision (keep/rollback/iterate)

### Final Tuning Report

Create file: `docs/day-23/baseline/FINAL_TUNING_RESULTS.md`

**Content**:
1. Initial vs final parameters
2. Improvement metrics (% change)
3. Best trajectory map
4. Justification for final parameter choice
5. Impact on TD3 comparison (fair baseline established)

---

## Timeline

| Phase | Duration | Task |
|-------|----------|------|
| **Diagnosis** | 15 min | Run diagnostic tests (disable PID/Pure Pursuit) |
| **Iteration 1** | 20 min | Apply Option A, evaluate, analyze |
| **Iteration 2** | 20 min | Apply Option B (if needed), evaluate, analyze |
| **Iteration 3** | 20 min | Fine-tune between A/B (if needed) |
| **Validation** | 30 min | Run full 5-episode test, verify stability |
| **Documentation** | 15 min | Create tuning report |
| **Total** | **~2 hours** | Complete controller tuning |

---

## Next Steps After Tuning

1. ✅ **Update baseline_config.yaml** with final parameters
2. ✅ **Re-run Phase 3** with tuned controller (verify improvement)
3. ✅ **Proceed to Phase 4** (NPC Interaction Test)
4. ✅ **Update paper methodology** with tuned baseline parameters
5. ✅ **Document tuning process** in supplementary materials

---

## References

- Phase 3 Analysis: `PHASE3_COMPLETION_SUMMARY.md`
- TCC Controller: `controller2d.py`
- Current Implementation: `src/baselines/pure_pursuit_controller.py`
- Configuration: `config/baseline_config.yaml`

---

**Status**: Ready to begin tuning
**Recommended Start**: Option A (Conservative Tuning)
**Estimated Time**: 2 hours to completion
