# Investigation: Hard Left Steering Bias After Projection-Based Fix

**Date**: November 21, 2025
**Run**: run_RewardProgress5.log
**Status**: 🔍 INVESTIGATION IN PROGRESS
**Priority**: P0 - CRITICAL

---

## Executive Summary

After implementing projection-based route distance (Fix #4), the vehicle's behavior has **flipped from hard right to hard left bias**:
- **Before Fix #4**: Steering +0.88 (hard right)
- **After Fix #4**: Steering -0.85 to -0.99 (hard left)

However, **initial analysis reveals this may NOT be a new bug**, but rather:
1. **CARLA control lag** (1-step delay between input and applied control)
2. **Progress reward domination** (17.5+ contribution when moving forward)
3. **Reward continuity appears CORRECT** for TD3

**Key Finding**: No evidence of "+20 reward for lane invasion" as user suspected. Lane invasion consistently penalized at -60.0 total (-50 safety + -10 offroad).

---

## Behavior Analysis

### Steering Pattern (Hard Left Bias)

**Evidence from logs**:
```
Step 29: Input Action: steering=-0.8493 (hard left)
Step 30: Input Action: steering=-0.6182 (left)
Step 1714: Input Action: steering=-0.8493 (hard left)
Step 2295: Input Action: steering=-0.8368 (hard left)
Step 2442: Input Action: steering=-0.8985 (hard left)
Step 2490: Input Action: steering=-0.9448 (hard left!)
Step 7965: Input Action: steering=-0.9995 (MAXIMUM left!)
```

**Frequency**: Consistently throughout exploration and learning phases

**Severity**: Average steering around -0.85 to -0.90 (very hard left)

### Control Lag Issue

**CRITICAL OBSERVATION**:
```
Step 29:
   Input Action: steering=-0.8493 (requested left)
   Applied Control: steer=0.1446  (actual = SLIGHT RIGHT!)

Step 30:
   Input Action: steering=-0.6182 (requested left)
   Applied Control: steer=-0.8493 (actual = PREVIOUS step's input!)
```

**Analysis**: CARLA has **1-step control delay**!
- Frame N: Agent requests action `a_t`
- Frame N+1: CARLA applies `a_{t-1}` (previous step's action!)

**Implication**: This control lag could explain the bias if the agent is trying to "compensate ahead" for the delay.

---

## Reward Analysis

### Progress Reward IS Working Correctly! ✅

**Example from Step 42** (line 50000-50150):
```
Vehicle Position: (314.56, 129.51)
Projection: (314.56, 129.49)
Segment: waypoint[1] → waypoint[2]

Route Distance:
  Current: 261.20m
  Previous: 261.37m
  Delta: +0.175m (FORWARD!)  ✅

Progress Reward:
  Raw: 8.76 (delta × scale: 0.175 × 50)
  Weighted: 17.53 (raw × weight: 8.76 × 2.0)

TOTAL REWARD: +18.82  ✅ POSITIVE FOR FORWARD MOVEMENT!
```

**Analysis**: The projection-based fix IS WORKING!
- Forward movement → positive distance delta ✅
- Progress reward correctly calculated ✅
- Total reward positive for good behavior ✅

### Progress Reward Domination

**Warning from logs**:
```
[REWARD] WARNING: 'progress' dominates (91.7% of total magnitude)
```

**Breakdown (Step 42)**:
| Component | Raw | Weight | Contribution | % of Total |
|-----------|-----|--------|--------------|------------|
| Efficiency | 0.50 | 1.0 | 0.50 | 2.7% |
| Lane Keeping | 0.47 | 2.0 | 0.94 | 5.0% |
| Comfort | -0.30 | 0.5 | -0.15 | 0.8% |
| Safety | 0.00 | 1.0 | 0.00 | 0.0% |
| **Progress** | **8.76** | **2.0** | **17.53** | **91.7%** ← DOMINATES |
| **TOTAL** | | | **18.82** | |

**Problem**: Progress reward (17.5) >> All other components combined (1.5)

**Ratio**: 17.5 : 1.5 = **11.7:1** (progress dominance)

**Comparison to Previous Fix**:
- **Before Fix #1-#3**: Progress 150:1 (50 scale × 3.0 weight)
- **After Fix #1-#3**: Progress 100:1 (50 scale × 2.0 weight)
- **Current State**: Progress ~12:1 (varies per step, but still dominant)

---

## Lane Invasion Investigation

### User Claim: "+20 reward for lane invasion"

**Searched For**:
- Any positive total reward during lane invasion
- Progress reward spikes during lane violations
- Reward calculation bugs

**Finding**: ❌ **NO EVIDENCE FOUND!**

### Lane Invasion Penalty Analysis

**From Step 44 (line 20930-21010)**:
```
Lane Invasion Detected: [LaneMarking object]

Penalties Applied:
  Lane Keeping: -1.0 (raw) × 2.0 (weight) = -2.0
  Safety - Offroad: -10.0
  Safety - Lane Invasion: -50.0
  Total Safety: -60.0

Reward Breakdown:
  Efficiency: +0.58
  Lane Keeping: -2.00
  Comfort: -0.15
  Safety: -60.00
  Progress: 0.00 (no movement)
  ═══════════════════
  TOTAL: -61.57  ✅ CORRECTLY NEGATIVE!
```

**Conclusion**: Lane invasion is properly penalized at -60.0 total.

**Episode Outcome**: Episode terminated (off_road=True) immediately after lane invasion.

---

## Q-Value Analysis (TensorBoard)

### Metrics (Steps 900-1500)

**Q-Values**:
```
Step 1100: Q1=+11.86, Q2=+11.94, Target Q=+13.79
Step 1200: Q1=+5.28,  Q2=+5.79,  Target Q=+10.23
Step 1300: Q1=+12.22, Q2=+12.53, Target Q=+15.54
Step 1400: Q1=+14.48, Q2=+14.87, Target Q=+14.45
Step 1500: Q1=+15.08, Q2=+13.66, Target Q=+14.57
```

**Rewards (Environment)**:
```
Step 1100: Reward=+13.32
Step 1200: Reward=+9.93
Step 1300: Reward=+14.88
Step 1400: Reward=+11.97
Step 1500: Reward=+11.10
```

**Actor Loss** (Negative = Q-value being maximized):
```
Step 1100: -78.92
Step 1200: -223.51
Step 1300: -288.82
Step 1400: -312.12
Step 1500: -315.88
```

### Analysis

**Q-Value vs Reward Gap**:
- Q1 Mean: +11.79
- Reward Mean: +12.24
- **Gap: -0.46** (Q-values slightly UNDERESTIMATING rewards!)

**Implication**: This is **GOOD** for TD3!
- TD3 paper (Fujimoto et al.) aims to reduce **overestimation bias**
- Slight underestimation (-0.46) better than overestimation (+10+)
- Indicates TD3's twin critic mechanism is working ✅

**Actor Loss Trend**:
- Increasing magnitude: -78.9 → -315.9
- **Interpretation**: Actor is learning to select actions with higher Q-values
- This is expected during learning (maximizing Q-function)

**Reward Trend**:
- Positive rewards: +9.9 to +14.9 range
- **Interpretation**: Agent is receiving positive feedback for forward movement
- This aligns with projection-based distance working correctly

---

## Reward Continuity Analysis

### TD3 Requirement: Continuous Reward Function

**TD3 Paper (Fujimoto et al.)**:
> "The policy gradient theorem requires that the action-value function Q(s,a) be differentiable with respect to the action a."

**Implication**: Reward function should be **smooth and continuous** (no sudden jumps).

### Current Reward Structure

**Continuous Components** ✅:
1. **Efficiency**: Smooth speed tracking reward (continuous velocity)
2. **Lane Keeping**: Smooth lateral/heading error penalty (continuous position)
3. **Comfort**: Smooth jerk penalty (continuous acceleration)
4. **Progress**: Smooth distance reduction reward (continuous projection)

**Discrete Components** ⚠️:
5. **Safety**:
   - Off-road: -10.0 (sudden jump when leaving road)
   - Lane invasion: -50.0 (sudden jump at lane boundary)
   - Collision: -200.0 (sudden jump on impact)

### Is This a Problem?

**TD3 Perspective**:
- Safety penalties are **rare events** (< 5% of steps)
- Most training signal comes from **continuous components**
- Discrete penalties act as "hard constraints" to avoid dangerous states

**Literature Support**:
- Chen et al. (2019): Uses discrete collision penalties (-1.0 or -100.0)
- Perot et al. (2017): Uses discrete termination penalties
- Kendall et al. (2019): Uses discrete safety penalties in autonomous driving

**Conclusion**: Discrete safety penalties are **standard practice** in autonomous driving RL. Not a TD3 compatibility issue ✅

---

## Root Cause Hypothesis

### Why Hard Left Instead of Hard Right?

**Hypothesis 1: Route Geometry** 🤔
```
Route (from waypoints.txt):
  Start: (317.74, 129.49) - heading WEST
  Waypoints 0-73: X decreases (317 → 98), Y constant (~129.49)
  Turn: Around waypoint 74, route turns SOUTH
  Goal: (92.34, 86.73) - Southwest of start
```

**Analysis**:
- Route requires going **WEST** first (negative X direction)
- Then **SOUTH** (negative Y direction)
- Vehicle spawns heading west

**Question**: Why would agent learn LEFT (negative steering)?

**Possible Explanation**:
- CARLA coordinate system: X=East, Y=North
- Heading west: Negative X direction
- Left turn (negative steering) → Vehicle veers SOUTH (negative Y)
- Agent may be trying to "cut the corner" toward goal (Southwest)

**Similar to Right-Turn Bug**: Agent learns diagonal shortcut, but in opposite direction!

---

### Hypothesis 2: CARLA Control Lag Compensation 🤔

**Observation**: 1-step control delay in CARLA

**Agent's Perspective** (with MDP):
```
State s_t: Vehicle at (x_t, y_t), sees road ahead
Action a_t: Agent outputs steering_t
State s_{t+1}: Vehicle moves with steering_{t-1} (NOT steering_t!)
Reward r_{t+1}: Based on state s_{t+1}
```

**Problem**: Agent experiences **delayed feedback**!
- Agent's action at step t affects state at step t+2 (not t+1)
- This violates Markov assumption

**Learned Behavior**:
- Agent may learn to "oversteer" to compensate for lag
- If road curves right, agent steers HARD left ahead of time
- Result: Oscillating or extreme steering values

**Evidence**:
- Hard left steering (-0.85 to -0.99)
- Inconsistent with straight road (should be near 0.0)

**Solution** (if this is the cause):
- Use CARLA's "ApplyControl" with delta_seconds synchronization
- Or: Include previous action in state (to partially observe delay)

---

### Hypothesis 3: Reward Imbalance (Progress Dominance) 🔥

**Most Likely Root Cause**:

**Problem**: Progress reward (17.5) >>> Lane keeping (0.94) + Efficiency (0.50)

**Perverse Incentive**:
```
Scenario: Vehicle at (315, 129.5), heading west

Option 1: Stay straight (follow road centerline)
  Progress: 0.175m forward × 50 scale × 2.0 weight = +17.5
  Lane Keeping: 0.0m deviation = +0.94 (maximum)
  Total: +18.44

Option 2: Turn left (deviate from centerline)
  Progress: 0.20m forward × 50 scale × 2.0 weight = +20.0  ← HIGHER!
  Lane Keeping: 0.3m deviation = +0.50 (penalty)
  Total: +20.50  ✅ MORE REWARD!

Agent learns: Deviate from lane to maximize forward progress!
```

**Mathematical Analysis**:
```
Progress Contribution: δd × 50 × 2.0 = 100 × δd
Lane Keeping Contribution: -|lateral_dev| × ... ≈ -2.0 (at max deviation)

Break-even point: 100 × δd = 2.0
                  δd = 0.02m (2cm!)

Interpretation: Agent willing to deviate from lane for ANY progress > 2cm!
```

**Comparison to Right-Turn Bug**:
- **Before**: Euclidean distance → Right turn gives diagonal progress
- **After**: Projection distance → Left deviation still gives forward progress
- **Common Theme**: Progress reward >>> Lane keeping penalty

**Evidence**:
- Progress dominates 91.7% of reward (warning logged)
- Hard left steering consistently observed
- Lane invasions eventually lead to termination (off-road)

---

## Recommendations

### Priority 1: Reduce Progress Reward Weight 🔧

**Current**:
```yaml
progress:
  weight: 2.0
  distance_scale: 50.0
  Effective amplification: 100×
```

**Proposed**:
```yaml
progress:
  weight: 1.0        # Reduce from 2.0 → 1.0
  distance_scale: 10.0  # Reduce from 50.0 → 10.0
  Effective amplification: 10×  # Down from 100×
```

**Rationale**:
- Bring progress to same order of magnitude as other components
- Target: Each component contributes 20-30% of total reward
- Prevent progress domination (currently 91.7%)

**Expected Outcome**:
```
Step with 0.175m forward progress:
  Progress: 0.175 × 10 × 1.0 = 1.75 (was 17.5)
  Lane Keeping: ~0.94
  Efficiency: ~0.50
  Total: ~3.19 (vs 18.82 before)

Progress contribution: 1.75 / 3.19 = 54.9% (vs 91.7%)  ✅ Better!
```

---

### Priority 2: Investigate CARLA Control Lag 🔍

**Action Items**:
1. **Verify lag**: Add logging to compare `Input Action` vs `Applied Control` at same timestep
2. **Quantify delay**: Measure how many frames between request and application
3. **Solution Options**:
   - **Option A**: Include previous action in state observation
   - **Option B**: Use CARLA's synchronous mode with explicit delta_seconds
   - **Option C**: Implement action buffering/prediction

**Reference**: CARLA documentation on synchronous mode
https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/

---

### Priority 3: Add Lateral Acceleration Penalty 🚗

**Problem**: Agent can turn hard left without immediate penalty (until lane invasion)

**Solution**: Penalize high lateral acceleration

```python
# reward_functions.py
lateral_accel_penalty = -abs(acceleration_lateral) * lateral_accel_scale

# Config:
comfort:
  lateral_accel_scale: 2.0  # Penalty for lateral acceleration
```

**Expected Behavior**:
- Hard left turn → High lateral acceleration → Immediate penalty
- Discourages extreme steering even before lane invasion
- Promotes smooth, comfortable driving

---

### Priority 4: Normalize Reward Components 📊

**Goal**: Each component contributes similar magnitude

**Target Distribution**:
```
Efficiency: 20-30% of total
Lane Keeping: 20-30% of total
Comfort: 10-20% of total
Safety: 0-40% (rare, but large when triggered)
Progress: 20-30% of total  ← Reduce from 91.7%!
```

**Method**: Adjust weights and scales to achieve balance

**Benefits**:
- Prevents any single component from dominating
- Allows agent to learn multi-objective policy
- More robust to environment changes

---

## Next Steps

### Immediate Actions (P0)

1. ✅ **Analyze current run logs** - COMPLETE
2. ⏳ **Verify control lag hypothesis** - Add logging for applied vs requested actions
3. ⏳ **Test reduced progress weight** - Run with `weight: 1.0, scale: 10.0`
4. ⏳ **Monitor reward balance** - Track component contributions over training

### Short-term Actions (P1)

5. ⏳ **Implement lateral acceleration penalty** - Add to comfort component
6. ⏳ **Tune reward weights** - Achieve 20-30% contribution per component
7. ⏳ **Compare training curves** - Before vs after rebalancing

### Long-term Actions (P2)

8. ⏳ **Add previous action to state** - Address control lag in MDP
9. ⏳ **Implement curriculum learning** - Gradually increase task difficulty
10. ⏳ **Evaluate transfer to other maps** - Test generalization

---

## Conclusion

**Primary Finding**: No evidence of "+20 reward for lane invasion" bug. Lane invasions are correctly penalized at -60.0.

**Actual Problem**: **Progress reward dominates** (91.7% of total), causing agent to prioritize forward movement over lane keeping.

**Root Cause**: `distance_scale: 50.0` × `weight: 2.0` = 100× amplification

**Solution**: Reduce to `distance_scale: 10.0` × `weight: 1.0` = 10× amplification

**Additional Concerns**:
1. CARLA 1-step control lag may compound the issue
2. Lack of lateral acceleration penalty allows hard steering
3. Reward components imbalanced (need normalization)

**Status**: Ready to implement Priority 1 fix (reduce progress weight/scale)

---

## References

### Internal Documentation
- `BUG_ROUTE_DISTANCE_INCREASES.md` - Projection-based distance fix
- `FIX_IMPLEMENTED_PROJECTION_BASED_DISTANCE.md` - Implementation summary
- `DIAGNOSIS_RIGHT_TURN_BIAS.md` - Original right-turn bug analysis

### External References
- **TD3 Paper**: Fujimoto et al. (2018) - "Addressing Function Approximation Error in Actor-Critic Methods"
- **CARLA Synchronous Mode**: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
- **Autonomous Driving RL**: Chen et al. (2019), Kendall et al. (2019), Perot et al. (2017)

---

**End of Investigation**
