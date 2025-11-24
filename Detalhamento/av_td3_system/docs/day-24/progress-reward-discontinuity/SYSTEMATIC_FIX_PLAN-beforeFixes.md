# Systematic Fix Plan: Progress Reward Discontinuity

**Date**: November 24, 2025  
**Issue**: Progress reward discontinuity during normal driving  
**Priority**: P0 - CRITICAL (Affects TD3 learning stability)  
**Status**: 🔍 INVESTIGATION PLANNED

---

## 📋 Problem Statement

### User's Observation

> "The progress reward is not been continuous, while moving we are correctly receiving reward for progress but for a few seconds the progress reward goes to 0 than suddenly goes back to e.g 10 (for example: the car is properly moving progressing and is receiving e.g 10 reward for the forward moving, then it suddenly goes to zero for half a second and then goes back to e.g 10 reward)"

### Context

**When Issue Appears**:
- During normal forward driving (vehicle moving correctly)
- Progress reward oscillates: 10.0 → 0.0 → 10.0
- Duration: Half a second (approximately 10 steps at 20 FPS)

**When It Started**:
- After removing PBRS from `_calculate_progress_reward()` (Nov 23, 2025)
- Previous PBRS removal fixed "free reward for zero movement" bug
- But introduced new discontinuity problem

**Related Issues**:
- Hard left bias still persists (#file:INVESTIGATION_HARD_LEFT_BIAS.md)
- May be related to waypoint projection issues
- Could affect TD3 Q-value estimation (violates reward consistency)

---

## 🎯 Systematic Investigation Plan

### Phase 1: Documentation Research (30-45 min)

#### Task 1.1: Fetch TD3 Paper Requirements for Reward Consistency

**URL**: Already attached - #file:Addressing Function Approximation Error in Actor-Critic Methods.tex

**Search For**:
1. "reward" + "consistency" → What TD3 requires for stable learning
2. "overestimation bias" → How reward noise affects Q-values
3. "temporal difference" → Impact of reward discontinuity on TD learning

**Questions to Answer**:
- Does reward discontinuity violate TD3 assumptions?
- How much reward noise can TD3 tolerate?
- Should progress reward be smoothed?

**Expected Time**: 15 minutes

#### Task 1.2: Fetch CARLA Waypoint Manager Documentation

**URL**: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint

**Search Terms**:
- "waypoint" + "distance"
- "get_waypoint" + "project_to_road"
- "lane_type" + "waypoint"

**Questions to Answer**:
- Can `get_route_distance_to_goal()` return None or invalid values?
- What happens when vehicle temporarily off-route?
- How does waypoint projection behave during turns?

**Expected Time**: 15 minutes

#### Task 1.3: Review Related Papers on Reward Shaping

**Papers**:
1. #file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex
   - Search: "reward" + "progress" + "continuous"
   - How do they handle progress in urban scenarios?

2. #file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex
   - Search: "termination" + "learning rate"
   - Impact of discontinuous rewards on learning

**Expected Time**: 15 minutes

**Deliverable**: Document findings in `PHASE_1_DOCUMENTATION.md`

---

### Phase 2: Root Cause Investigation (1-1.5 hours)

#### Task 2.1: Add Diagnostic Logging to Progress Reward Calculation

**File**: `av_td3_system/src/environment/reward_functions.py`

**Add Logging**:
```python
def _calculate_progress_reward(self, distance_to_goal, waypoint_reached, goal_reached):
    # Log input values with high precision
    self.logger.debug(
        f"[PROGRESS-DEBUG] distance_to_goal={distance_to_goal}, "
        f"prev_distance={self.prev_distance_to_goal}, "
        f"waypoint_reached={waypoint_reached}"
    )
    
    # ... existing code ...
    
    # Log delta calculation
    if self.prev_distance_to_goal is not None:
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        self.logger.debug(
            f"[PROGRESS-DEBUG] distance_delta={distance_delta:.6f}m, "
            f"distance_reward_raw={distance_delta * self.distance_scale:.3f}"
        )
    
    # Log final components
    self.logger.debug(
        f"[PROGRESS-DEBUG] Components: distance={distance_component:.3f}, "
        f"waypoint={waypoint_component:.3f}, goal={goal_component:.3f}, "
        f"total={progress:.3f}"
    )
```

**Expected Time**: 15 minutes

#### Task 2.2: Add Diagnostic Logging to Waypoint Manager

**File**: `av_td3_system/src/environment/waypoint_manager.py`

**Add Logging**:
```python
def get_route_distance_to_goal(self, vehicle_location):
    # Log projection status
    self.logger.debug(
        f"[WAYPOINT-DEBUG] vehicle_location={vehicle_location}, "
        f"current_waypoint_idx={self.current_waypoint_idx}"
    )
    
    # ... projection logic ...
    
    # Log projection result
    self.logger.debug(
        f"[WAYPOINT-DEBUG] projected_point={projected_point}, "
        f"distance_to_projection={distance_to_projection:.3f}m, "
        f"route_distance={route_distance:.3f}m"
    )
    
    return route_distance
```

**Expected Time**: 15 minutes

#### Task 2.3: Manual Test with Enhanced Logging

**Procedure**:
1. Start CARLA: `./CarlaUE4.sh -quality-level=Low`
2. Run validation with logging:
   ```bash
   python scripts/validate_rewards_manual.py \
       --config config/baseline_config.yaml \
       --output-dir validation_logs/progress_discontinuity_test \
       --log-level DEBUG
   ```
3. Drive straight for 20 seconds at constant speed
4. Observe progress reward in HUD
5. Note exact steps where discontinuity occurs

**Data to Collect**:
- Step numbers where progress goes to 0
- `distance_to_goal` values before/during/after discontinuity
- `waypoint_reached` status
- Vehicle position and heading

**Expected Time**: 20 minutes

#### Task 2.4: Analyze Log Patterns

**Look For**:
1. **Hypothesis 1: Waypoint Transition**
   - Does discontinuity occur when `waypoint_reached=True`?
   - Check if `distance_to_goal` jumps during waypoint advancement

2. **Hypothesis 2: Invalid Distance Values**
   - Does `get_route_distance_to_goal()` return None or 0.0?
   - Check if condition on line 1001 triggers

3. **Hypothesis 3: Projection Failure**
   - Does vehicle temporarily project outside route?
   - Check if `distance_to_projection` exceeds threshold

4. **Hypothesis 4: prev_distance Reset**
   - Is `self.prev_distance_to_goal` set to None unexpectedly?
   - Check if this causes delta calculation to skip

**Expected Time**: 30 minutes

**Deliverable**: Document findings in `PHASE_2_ROOT_CAUSE.md` with evidence

---

### Phase 3: Solution Implementation (1-2 hours)

**NOTE**: Implementation depends on Phase 2 findings. Below are potential fixes:

#### Scenario A: Waypoint Transition Discontinuity

**Root Cause**: When advancing to next waypoint, route distance jumps

**Fix**: Smooth transition with interpolation
```python
def get_route_distance_to_goal(self, vehicle_location):
    # ... projection logic ...
    
    # If waypoint just advanced, interpolate distance
    if self.waypoint_just_advanced:
        # Use vehicle progress along segment for smooth transition
        route_distance = self._interpolate_route_distance(
            vehicle_location, projected_point, route_distance
        )
        self.waypoint_just_advanced = False
    
    return route_distance
```

#### Scenario B: Invalid Distance Handling

**Root Cause**: Line 1001 condition triggers, returns 0.0, breaks continuity

**Fix**: Use last valid distance instead of 0.0
```python
if distance_to_goal is None or distance_to_goal <= 0.0:
    # Instead of returning 0.0, use last valid distance with zero delta
    if self.prev_distance_to_goal is not None:
        # No progress this step, but don't break continuity
        self.logger.debug("[PROGRESS] Using last valid distance (no change)")
        return 0.0  # Zero progress reward, but preserve prev_distance
    else:
        # Truly first step, initialize
        self.prev_distance_to_goal = None
        return 0.0
```

#### Scenario C: Projection Threshold Issue

**Root Cause**: Vehicle briefly projects far from route, distance becomes invalid

**Fix**: Use temporal smoothing
```python
def get_route_distance_to_goal(self, vehicle_location):
    raw_distance = self._calculate_raw_route_distance(vehicle_location)
    
    # Smooth with exponential moving average
    if self.smoothed_route_distance is None:
        self.smoothed_route_distance = raw_distance
    else:
        alpha = 0.7  # Smoothing factor (0.7 = responsive, 0.3 = smooth)
        self.smoothed_route_distance = (
            alpha * raw_distance + (1 - alpha) * self.smoothed_route_distance
        )
    
    return self.smoothed_route_distance
```

**Expected Time**: 1-2 hours depending on complexity

---

### Phase 4: Validation Testing (1 hour)

#### Test 1: Continuity Verification

**Procedure**:
1. Run manual control with fix implemented
2. Drive straight for 60 seconds
3. Monitor progress reward in HUD
4. Record any discontinuities

**Success Criteria**:
- [ ] Progress reward changes smoothly (no sudden drops to 0)
- [ ] Reward value matches vehicle motion (forward = positive)
- [ ] No unexpected spikes or valleys

#### Test 2: Waypoint Transition

**Procedure**:
1. Drive along route through 10 waypoints
2. Monitor progress reward during waypoint advancement
3. Check for smooth transitions

**Success Criteria**:
- [ ] Progress reward continuous through waypoint transitions
- [ ] No discontinuity when `waypoint_reached=True`

#### Test 3: Edge Cases

**Test Cases**:
1. **Sharp Turn**: Does progress stay continuous during tight turn?
2. **Temporary Stop**: Does progress resume correctly after brief stop?
3. **Reverse**: Does negative progress work correctly?

**Expected Time**: 1 hour total

**Deliverable**: Validation report in `PHASE_4_VALIDATION.md`

---

### Phase 5: TD3 Impact Analysis (30 minutes)

#### Task 5.1: Review TD3 Paper on Reward Consistency

**Question**: Does continuous progress reward improve TD3 learning?

**Analysis**:
- Compare reward variance before/after fix
- Check if discontinuity causes Q-value overestimation
- Verify fix doesn't violate PBRS theory

#### Task 5.2: Update Documentation

**Files to Update**:
- `reward_functions.py` - Add comments explaining fix
- `SYSTEMATIC_PROGRESS_REWARD_ANALYSIS.md` - Document solution
- `IMPLEMENTATION_SUMMARY.md` - Add to fix list

**Expected Time**: 30 minutes

---

## 📊 Success Criteria

### Technical Requirements

- [ ] Progress reward is continuous during normal driving
- [ ] No sudden drops to 0 without valid reason
- [ ] Smooth transitions during waypoint advancement
- [ ] Fix doesn't break existing functionality (route distance, waypoint bonuses)

### TD3 Requirements (from Paper)

- [ ] Reward signal is consistent (no unexpected noise)
- [ ] Temporal difference error is bounded
- [ ] Q-value estimation not biased by discontinuity

### Scientific Reproducibility

- [ ] Fix documented with rationale
- [ ] Test results logged for paper
- [ ] Implementation references official docs

---

## 📚 References

### Official Documentation
- **CARLA Waypoint API**: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint
- **Gymnasium Env.step()**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

### Research Papers
- **TD3 Paper**: #file:Addressing Function Approximation Error in Actor-Critic Methods.tex
  - Section 3: Overestimation bias from reward noise
- **Interpretable E2E Driving**: #file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex
  - Reward design for urban scenarios
- **Lane Keeping Paper**: #file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex
  - Impact of reward termination on learning rate

### Related Issues
- **Previous Fixes**: #file:SYSTEMATIC_PROGRESS_REWARD_ANALYSIS.md
- **PBRS Removal**: #file:DIAGNOSIS_RIGHT_TURN_BIAS.md
- **Hard Left Bias**: #file:INVESTIGATION_HARD_LEFT_BIAS.md

---

## 🔄 Status Tracking

| Phase | Status | Time Spent | Notes |
|-------|--------|------------|-------|
| Phase 1: Documentation | Not Started | 0/45 min | Fetch TD3, CARLA, paper requirements |
| Phase 2: Investigation | Not Started | 0/90 min | Root cause analysis with logging |
| Phase 3: Implementation | Not Started | 0/120 min | Depends on Phase 2 findings |
| Phase 4: Validation | Not Started | 0/60 min | Manual testing scenarios |
| Phase 5: Documentation | Not Started | 0/30 min | Update docs and analyze TD3 impact |

**Total Estimated Time**: 5.75 hours (distributed across phases)

---

## 🚀 Next Actions

1. **Start Phase 1**: Fetch TD3 paper section on reward consistency
2. **Set Up Logging**: Add diagnostic logging to progress reward calculation
3. **Manual Test**: Drive straight and observe discontinuity with enhanced logs
4. **Analyze**: Identify exact root cause from log patterns
5. **Implement Fix**: Apply solution based on findings
6. **Validate**: Test continuity across all scenarios
7. **Document**: Update all relevant documentation

---

**Created**: November 24, 2025  
**Next Review**: After Phase 2 completion
