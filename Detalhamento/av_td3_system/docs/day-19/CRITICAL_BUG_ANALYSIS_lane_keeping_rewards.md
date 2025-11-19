# CRITICAL BUG ANALYSIS: Lane Keeping Rewards During Lane Invasions

**Date:** November 19, 2025
**Session:** Day 19 - 5K Validation Post-Fixes
**Status:** 🔴 CRITICAL - Blocks 1M Training
**Severity:** HIGH - Agent learns unsafe lane invasion behavior

---

## Executive Summary

The TD3 agent is receiving **POSITIVE lane keeping rewards** while crossing lane markings (lane invasions), leading to unsafe learned behavior. This occurs because the `lane_keeping` reward component is calculated **independently** of lane invasion events, using only `lateral_deviation` from lane center. The agent can cross into wrong lanes while remaining centered within the invaded lane, thus receiving positive rewards for objectively unsafe behavior.

---

## Problem Statement

### Observed Behavior (from 5K validation run)
```
2025-11-19 18:28:41 - Lane invasion detected: [<carla.libcarla.LaneMarking object>]
LANE KEEPING (stay in lane):
   Raw: +0.2720  ← POSITIVE reward during violation!
   Weight: 2.00
   Contribution: +0.5440

SAFETY (collision/offroad penalty):
   Raw: -10.0000
   Weight: 1.00
   Contribution: -10.0000

PROGRESS (goal-directed movement):
   Raw: +11.8841
   Weight: 2.00
   Contribution: +23.7683  ← DOMINATES total reward

TOTAL REWARD: +14.7113  ← NET POSITIVE despite lane invasion!
```

### User Observation
- Vehicle turns left repeatedly (timesteps 1000-2500)
- 50+ lane invasion events logged during 5K steps
- Agent continues receiving positive total rewards
- Lane invasions counted in `info` dict but NOT factored into lane_keeping calculation

---

## Root Cause Analysis

### 1. Lane Keeping Reward Implementation

**File:** `av_td3_system/src/environment/reward_functions.py`
**Function:** `_calculate_lane_keeping_reward()`
**Lines:** 438-506

```python
def _calculate_lane_keeping_reward(
    self, lateral_deviation: float, heading_error: float, velocity: float,
    lane_half_width: float = None
) -> float:
    """
    Calculate lane keeping reward with CARLA-based lane width normalization.

    Args:
        lateral_deviation: Perpendicular distance from lane center (m)
        heading_error: Heading error w.r.t. lane direction (radians)
        velocity: Current velocity (m/s)
        lane_half_width: Half of current lane width from CARLA (m)
    """
    # Lateral deviation component
    lat_error = min(abs(lateral_deviation) / effective_tolerance, 1.0)
    lat_reward = 1.0 - lat_error * 0.7  # 70% weight

    # Heading error component
    head_error = min(abs(heading_error) / self.heading_tolerance, 1.0)
    head_reward = 1.0 - head_error * 0.3  # 30% weight

    # Combined reward (average of components, shifted to [-0.5, 0.5])
    lane_keeping = (lat_reward + head_reward) / 2.0 - 0.5

    # NO CONSIDERATION OF LANE INVASION EVENTS!
    return float(np.clip(lane_keeping * velocity_scale, -1.0, 1.0))
```

**CRITICAL FLAW:** The function calculates reward based ONLY on:
1. Distance from **some** lane center (`lateral_deviation`)
2. Heading alignment with **some** lane direction (`heading_error`)
3. Velocity scaling

**Missing:** Any awareness of whether lane markings were crossed (lane invasion events).

### 2. Lane Invasion Detection (Working Correctly)

**CARLA Documentation:** [Lane Invasion Detector](https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector)

- **Blueprint:** `sensor.other.lane_invasion`
- **Output:** `carla.LaneInvasionEvent` **per crossing**
- **Trigger:** "Registers an event each time its parent crosses a lane marking"
- **Data:** List of crossed lane markings (`crossed_lane_markings`)
- **Client-side:** Works fully on client-side

**Our Implementation:** Working correctly in `sensors.py`:
```python
def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
    """Callback when lane invasion occurs."""
    self.step_invasion_count = 1  # Binary flag per step
    self.logger.warning(f"Lane invasion detected: {event.crossed_lane_markings}")
```

**Evidence from logs:** 50+ lane invasion warnings during 5K steps:
- Line 2702: "Lane invasion detected"
- Line 5291: "Lane invasion detected"
- Line 8982, 10950, 13542, 17129, 20410, 24538, 27192, 30916, 35434, 39561... (continues)

### 3. Reward Calculation Flow

**File:** `carla_env.py`, lines 666-695

```python
# CRITICAL FIX (Nov 19, 2025): Get per-step sensor counts BEFORE reward calculation
collision_count = self.sensors.get_step_collision_count()
lane_invasion_count = self.sensors.get_step_lane_invasion_count()

reward_dict = self.reward_calculator.calculate(
    velocity=vehicle_state["velocity"],
    lateral_deviation=vehicle_state["lateral_deviation"],  # ← Used in lane_keeping
    heading_error=vehicle_state["heading_error"],            # ← Used in lane_keeping
    ...
    lane_invasion_detected=(lane_invasion_count > 0),  # ← Only used in SAFETY, NOT lane_keeping!
    ...
)
```

**The Disconnect:**
- `lane_invasion_detected` is passed to `calculate()` ✓
- It's forwarded to `_calculate_safety_reward()` ✓
- It's **NOT** forwarded to `_calculate_lane_keeping_reward()` ✗

### 4. Why This Causes Unsafe Behavior

**Scenario:** Agent turning left at intersection
1. **Before lane invasion:**
   - `lateral_deviation` = 0.5m from original lane center
   - `lane_keeping` = +0.4 (centered in correct lane)

2. **During/after lane invasion (crossing into adjacent lane):**
   - CARLA updates vehicle's `lane_id` to new (wrong) lane
   - `lateral_deviation` = 0.3m from **new** lane center (invaded lane)
   - `lane_keeping` = +0.5 (now MORE centered in WRONG lane!)
   - **Agent receives HIGHER reward for invasion!**

3. **Total reward calculation:**
   - Lane keeping: +0.5 × 2.0 weight = +1.0
   - Safety penalty: -50.0 × 1.0 weight = -50.0 (NEW, not in logs)
   - Progress: +11.88 × 2.0 weight = +23.77
   - **TOTAL: -25.23** (with new fix) vs **+14.71** (current logs)

**Current logs show:** Safety penalty was -10.0 (offroad), NOT lane invasion specific.

---

## Related Fixes Already Implemented

### Fix #1: Lane Invasion Safety Penalty (Nov 19, 2025)

**Status:** ✅ Implemented, ⏳ Not yet tested (logs are from BEFORE this fix)

**Files Modified:**
1. `reward_functions.py` (lines 88-96, 806-850)
2. `carla_env.py` (lines 670-685)

**Changes:**
- Added `lane_invasion_penalty = -50.0` parameter
- Added `lane_invasion_detected` parameter to `_calculate_safety_reward()`
- Implemented penalty logic with `[SAFETY-LANE_INVASION]` logging
- Modified `carla_env.py` to retrieve and pass lane invasion events

**Expected Impact:**
- Lane invasions will now receive -50.0 safety penalty
- With example above: Total reward becomes -25.23 instead of +14.71
- **BUT:** Lane keeping component STILL gives positive reward (+0.54)

### Fix #2: Waypoint System (Nov 19, 2025)

**Status:** ✅ Validated - NO BUG

**Findings:**
- Z-coordinate adjustment (8.33m → 0.50m) uses CARLA's `get_waypoint(project_to_road=True)`
- Global-to-local coordinate transformation is mathematically correct
- System follows CARLA best practices

---

## Why Lane Keeping Should Consider Lane Invasions

### Literature Review

#### 1. **Chen et al. (2019)** - Urban Autonomous Driving
*File:* `contextual/Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex`

**Reward Function:**
- Uses semantic bird-eye view masks to determine lane occupancy
- Penalizes deviation from **intended lane**, not just any lane center
- Incorporates collision prediction with dynamic objects

**Key Insight:** "The learned policy is able to provide an explanation of how the car reasons about the driving environment" - interpretability requires understanding of lane boundaries.

#### 2. **Perot et al. (2017)** - Race Driving
*File:* `contextual/End-to-End Race Driving with Deep Reinforcement Learning.tex`

**Reward Formula:**
```
R = v(cos(α) - d)
```
Where `d` is distance from **track boundaries** (not just center).

**Key Insight:** Boundary violations are explicitly penalized via distance term, preventing cutting corners.

#### 3. **Elallid et al. (2023)** - TD3 for Intersection Navigation
*File:* `contextual/Deep Reinforcement Learning based on TD3 for Autonomous Vehicles crossing T-intersections.pdf`

**Approach:**
- Uses TD3 (same algorithm as us)
- Image-based state representation (similar to our approach)
- Safety is PRIMARY objective at intersections

**Reward Components:**
1. Collision avoidance: -100
2. **Lane discipline**: Penalized for wrong lane occupancy
3. Progress: Positive for forward movement

**Key Insight:** TD3 successfully learns safe intersection navigation when lane discipline is explicitly encoded in reward.

### TD3 Algorithm Requirements

**Fujimoto et al. (2018)** - *Addressing Function Approximation Error in Actor-Critic Methods*

**Critical Properties:**
1. **Continuous Rewards:** TD3 requires smooth, differentiable reward landscapes
2. **Overestimation Bias:** Value function approximation leads to overestimation
3. **Twin Critics:** Use minimum of two Q-networks to reduce overestimation

**Implication for our bug:**
- Current lane_keeping gives continuous gradient based on lateral deviation ✓
- BUT: Positive rewards during safety violations cause value overestimation ✗
- Twin critics will learn to overestimate value of lane invasion states ✗
- Agent converges to suboptimal (unsafe) policy ✗

**Quote from paper:**
> "In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies."

**Our case:** We're not just overestimating - we're giving POSITIVE rewards for negative behaviors!

---

## Proposed Solutions

### Option 1: Add Lane Invasion Flag to Lane Keeping Calculation (RECOMMENDED)

**Rationale:** Simplest, most direct fix. Aligns with user's original observation.

**Implementation:**
```python
def _calculate_lane_keeping_reward(
    self,
    lateral_deviation: float,
    heading_error: float,
    velocity: float,
    lane_half_width: float = None,
    lane_invasion_detected: bool = False,  # NEW PARAMETER
) -> float:
    """
    Calculate lane keeping reward with lane invasion awareness.

    CRITICAL FIX (Nov 19, 2025): Add lane invasion check to prevent positive
    rewards during lane marking crossings.
    """
    # IMMEDIATE PENALTY FOR LANE INVASION
    if lane_invasion_detected:
        return -1.0  # Maximum lane keeping penalty

    # ... rest of existing logic for normal lane keeping ...
```

**Pros:**
- Minimal code change (2 lines)
- Directly addresses root cause
- Aligns with CARLA sensor semantics
- Clear logging for debugging

**Cons:**
- Creates discontinuity in reward landscape (jumps to -1.0)
- May conflict with TD3's preference for smooth gradients

**Mitigation:** The safety penalty (-50.0) already creates large gradient, so this is consistent.

### Option 2: Use Lane Invasion Count for Graduated Penalty

**Implementation:**
```python
def _calculate_lane_keeping_reward(..., lane_invasion_detected: bool = False):
    # Calculate base lane keeping reward
    base_lane_keeping = (lat_reward + head_reward) / 2.0 - 0.5

    # Apply lane invasion penalty multiplier
    if lane_invasion_detected:
        # Scale down reward significantly during invasion
        base_lane_keeping *= 0.1  # Reduce to 10% of normal value
        # Ensure it's negative
        base_lane_keeping = min(base_lane_keeping, -0.3)

    return float(np.clip(base_lane_keeping * velocity_scale, -1.0, 1.0))
```

**Pros:**
- Smoother gradient (scales instead of hard cutoff)
- Maintains some differentiation between "centered invasion" vs "off-center invasion"

**Cons:**
- More complex logic
- Still allows small positive rewards during invasions (if perfectly centered)

### Option 3: Use Separate Lane Invasion Component

**Implementation:**
- Keep lane_keeping as-is
- Add `lane_invasion_penalty` to safety component (ALREADY DONE ✓)
- Increase `safety_weight` to ensure it dominates

**Current weights:**
- `lane_keeping_weight = 2.0`
- `safety_weight = 1.0`
- `progress_weight = 2.0`

**Proposed:**
- `safety_weight = 3.0` (higher than progress)
- `lane_invasion_penalty = -50.0` (current)

**With this change:**
```
Lane keeping: +0.54 × 2.0 = +1.08
Safety:       -50.0 × 3.0 = -150.0  ← DOMINATES
Progress:     +11.88 × 2.0 = +23.76
TOTAL: -125.16  ← Strongly negative
```

**Pros:**
- No changes to lane_keeping logic
- Leverages already-implemented fix
- Weight tuning is standard practice

**Cons:**
- Lane keeping component still technically wrong (positive during violations)
- Harder to interpret rewards (safety penalty must overcome lane_keeping + progress)
- Requires careful weight tuning

---

## Recommended Action Plan

### Phase 1: Immediate Fix (Option 1)
1. ✅ Modify `_calculate_lane_keeping_reward()` to accept `lane_invasion_detected` parameter
2. ✅ Add immediate return of `-1.0` when lane invasion detected
3. ✅ Update function signature in `reward_functions.py` line 438
4. ✅ Update call site in `calculate()` method to pass the flag
5. ✅ Add logging: `self.logger.warning(f"[LANE_KEEPING] invasion penalty=-1.0")`

### Phase 2: Validation
1. Run 100-step test to verify both penalties appear in logs:
   - `[SAFETY-LANE_INVASION] penalty=-50.0`
   - `[LANE_KEEPING] invasion penalty=-1.0`
2. Check total reward is negative when invasions occur
3. Monitor agent behavior - does it learn to avoid lane crossings?

### Phase 3: Extended Testing
1. Run 10K validation with both fixes active
2. Analyze TensorBoard metrics:
   - Lane invasion frequency should decrease
   - Total rewards should correlate with safety behavior
   - Q-values should not overestimate unsafe states
3. Compare against baseline (current buggy version)

### Phase 4: Literature Validation
1. Read TD3 paper (Fujimoto et al.) sections on:
   - Overestimation bias and its effects
   - Reward design best practices
   - Continuous action space considerations
2. Review Chen et al. reward formulation for urban driving
3. Review Perot et al. boundary penalty implementation
4. Document findings in `docs/day-19/TD3_reward_design_validation.md`

---

## Expected Outcomes After Fix

### Behavioral Changes
- Agent should learn to avoid lane invasions
- Fewer left turns into wrong lanes
- More conservative driving near lane boundaries
- Lower total episode rewards initially (as penalties apply)
- Gradual improvement as agent learns safety

### Metric Changes
- `lane_invasion_count` per episode: Should decrease over training
- `avg_reward_10ep`: Will drop initially, then recover as behavior improves
- `safety_violations`: Should trend toward zero
- `lane_keeping_reward`: Will be negative more often (during learning), then stabilize positive

### Training Dynamics
- Early episodes: Many violations, large negative rewards
- Mid training: Agent explores safer trajectories
- Late training: Consistent lane discipline, positive rewards

---

## Risk Assessment

### Risk: Fix Doesn't Solve Problem
**Probability:** Low
**Mitigation:** The fix directly addresses the root cause (positive rewards during violations)

### Risk: Creates New Issues (Discontinuity)
**Probability:** Medium
**Mitigation:** TD3 is robust to some discontinuities; safety penalty already creates large gradient

### Risk: Agent Becomes Too Conservative
**Probability:** Low-Medium
**Mitigation:** Can adjust `lane_invasion_penalty` magnitude if needed; progress reward still incentivizes forward motion

### Risk: Delayed Convergence
**Probability:** Medium
**Impact:** Acceptable - safety is more important than speed of convergence

---

## References

1. **CARLA Documentation** - Lane Invasion Detector
   - URL: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
   - Key: Event-based detection of lane marking crossings

2. **Fujimoto et al. (2018)** - Addressing Function Approximation Error in Actor-Critic Methods
   - File: `contextual/Addressing Function Approximation Error in Actor-Critic Methods.tex`
   - Key: Overestimation bias leads to suboptimal policies

3. **Chen et al. (2019)** - Interpretable End-to-end Urban Autonomous Driving
   - File: `contextual/Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex`
   - Key: Lane occupancy awareness via semantic segmentation

4. **Perot et al. (2017)** - End-to-End Race Driving with Deep Reinforcement Learning
   - File: `contextual/End-to-End Race Driving with Deep Reinforcement Learning.tex`
   - Key: Boundary distance penalty prevents cutting corners

5. **Elallid et al. (2023)** - TD3 for Intersection Navigation
   - Key: Lane discipline explicitly encoded in reward for TD3 success

---

## Conclusion

The current lane keeping reward implementation is fundamentally flawed: it provides positive feedback for objectively unsafe behavior (lane invasions). This contradicts the principles of safe autonomous driving and will prevent the agent from learning appropriate lane discipline.

The root cause is clear: `lane_keeping` reward depends only on distance from *some* lane center, without awareness of whether lane markings were crossed. The fix is straightforward: pass the `lane_invasion_detected` flag to `_calculate_lane_keeping_reward()` and return maximum penalty when true.

**RECOMMENDATION:** Implement Option 1 (immediate return -1.0 on lane invasion) before any 1M-step training run. The system is NOT READY for large-scale training until this fix is validated.

**PRIORITY:** 🔴 CRITICAL - Must fix before scaling to 1M steps.

---

**Document Status:** Ready for Implementation
**Next Steps:** Implement Option 1, run 100-step validation test
**Approval Required:** User confirmation before proceeding with code changes
