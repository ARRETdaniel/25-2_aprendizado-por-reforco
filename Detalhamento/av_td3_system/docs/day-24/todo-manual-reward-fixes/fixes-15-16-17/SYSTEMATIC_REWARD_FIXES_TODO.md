# Systematic Reward Validation Fixes - Todo List

**Date Created**: November 24, 2025  
**Priority**: CRITICAL (P0) - Blocking TD3/DDPG Training  
**Status**: 🔴 IN PROGRESS

---

## Overview

This document outlines a systematic approach to fixing reward function issues identified during manual validation testing using `validate_rewards_manual.py`. All fixes follow official CARLA documentation, Gymnasium standards, and the TD3 paper principles.

**Testing Environment**: CARLA 0.9.16 + Manual Control Interface (WSAD)

---

## Identified Issues (from Manual Testing Session)

### Issue 1.5: Safety Penalty Persistence After Recovery ⚠️ CRITICAL

**Symptom**: After recovery from lane invasion or offroad event, safety reward remains at `-0.607` even when:
- Vehicle is stopped at the **center of the correct lane**
- Vehicle has **fully recovered** to safe driving conditions
- Vehicle begins making **progress toward goal** (correct behavior)

**Current Behavior**:
```
[Manual Test Scenario]
1. Drive on sidewalk → safety = -10.0 (correct)
2. Return to lane center, stop → safety = -0.607 (WRONG - should be 0.0)
3. Start moving forward in correct lane → safety = -0.607 (WRONG - penalizing good behavior)
```

**Expected Behavior**:
```
[Expected Scenario]
1. Drive on sidewalk → safety = -10.0 (penalty for violation)
2. Return to lane center, stop → safety = 0.0 (recovery confirmed)
3. Start moving forward in correct lane → safety = 0.0 (no penalty for correct behavior)
```

**Root Cause Hypothesis**:
- Safety reward component may be caching or accumulating negative values
- Recovery logic in `OffroadDetector` or `LaneInvasionDetector` not properly clearing state
- Proximity-based penalties (PBRS) may persist without proper reset

**References to Check**:
- CARLA Lane Invasion Sensor: `https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector`
- CARLA Waypoint API: `https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint`
- TD3 Paper (Addressing Function Approximation Error): Reward signal must be consistent to prevent Q-value bias
- #file:TASK_1_OFFROAD_DETECTION_FIX.md: Previous fix for offroad detection - ensure compatibility

---

### Issue 1.6: Lane Invasion Detection Inconsistency ⚠️ CRITICAL

**Symptom**: Inconsistent lane invasion penalty application:

**Case A - Warning Logged, No Penalty Applied**:
```
Terminal Output:
WARNING:src.environment.sensors:Lane invasion detected: [<carla.libcarla.LaneMarking object at 0x7fb874e68e40>]

Manual Control HUD:
lane_keeping: -0.45
total_reward: -0.89
```
**Expected**: Should also show:
```
WARNING:src.environment.reward_functions:[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)
```

**Case B - Sometimes No Warning or Penalty**:
```
[Manual Test]
1. Vehicle crosses lane marking → No warning in terminal
2. HUD shows: lane_keeping = -0.45 (based on lateral deviation only)
3. No discrete -1.0 penalty applied
```

**Expected Behavior**:
```
[Expected for ANY Lane Marking Crossing]
1. Sensor detects event → WARNING:src.environment.sensors:Lane invasion detected
2. Reward function responds → WARNING:src.environment.reward_functions:[LANE_KEEPING] applying maximum penalty
3. HUD shows: lane_keeping = -1.0 (discrete penalty)
4. Total reward reflects penalty
```

**Root Cause Hypothesis**:
- Lane invasion sensor callback fires (`_on_lane_invasion`) but flag not properly propagated to reward calculation
- Race condition between sensor callback and `step()` method
- `is_invading_lane()` recovery logic may be clearing flag before reward calculation reads it
- Missing integration between `LaneInvasionDetector.get_step_invasion_count()` and lane keeping reward

**References to Check**:
- CARLA Lane Invasion Sensor Event Handling: `https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector`
- Thread-safe sensor data access patterns
- Gymnasium `step()` method timing guarantees: `https://gymnasium.farama.org/api/env/#gymnasium.Env.step`

---

### Issue 1.7: Safety Reward Behavior When Moving vs Stopped 🔍 INVESTIGATE

**Symptom**: Different safety reward values based on vehicle motion state (before any violations):

**When Stopped** (velocity = 0.0):
```
safety: -0.5
```

**When Moving** (velocity > 0.0, normal driving):
```
safety: 0.0
```

**Question**: Is this behavior **correct** according to RL/DRL best practices?

**Hypothesis Analysis**:

**Argument FOR this behavior (potential PBRS/stopping penalty)**:
- Discourages agent from staying stationary
- Provides incentive for progress
- Common in navigation tasks to prevent "freezing"

**Argument AGAINST this behavior**:
- Penalizing stopped state when no safety violation occurred seems incorrect
- May conflict with traffic scenarios requiring stops (traffic lights, stop signs)
- TD3 paper emphasizes consistent reward signals - why penalize safe stopped state?

**Papers to Review**:
1. **TD3 Paper** (#file:Addressing Function Approximation Error in Actor-Critic Methods.tex)
   - Section on reward design
   - Consistency requirements for value estimation

2. **Lane Keeping Paper** (#file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex)
   - Quote: "many termination cause low learning rate"
   - Does inappropriate penalty cause similar issues?

3. **Interpretable E2E Driving** (#file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex)
   - How do they handle stopped states in urban scenarios?
   - CARLA-based implementation - what's their approach?

**Action Items**:
- [ ] Fetch documentation on PBRS (Potential-Based Reward Shaping) best practices
- [ ] Review all three papers for guidance on stopped state rewards
- [ ] Check if this penalty comes from:
  - [ ] Progress reward component (should be separate)
  - [ ] Safety PBRS proximity penalties
  - [ ] Explicit stopping penalty
- [ ] Determine if this is a bug or feature (document rationale either way)

---

## Todo List - Systematic Fix Plan

### Phase 1: Documentation Research 📚 (Estimated: 30-45 minutes)

#### Task 1.1: Fetch CARLA Lane Invasion Sensor Documentation
- [ ] Fetch: `https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector`
- [ ] Focus areas:
  - Event callback timing and guarantees
  - Thread safety considerations
  - Best practices for reading sensor state in `step()` method
  - Relationship between event firing and state persistence
- [ ] Document findings in: `TASK_1.5_SAFETY_PERSISTENCE_FIX.md`

#### Task 1.2: Fetch CARLA Waypoint API Documentation (Offroad Detection)
- [ ] Fetch: `https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint`
- [ ] Focus: `get_waypoint()` with `project_to_road=False` behavior
- [ ] Verify: How to properly detect "vehicle returned to drivable surface"
- [ ] Document findings in: `TASK_1.5_SAFETY_PERSISTENCE_FIX.md`

#### Task 1.3: Review TD3 Paper on Reward Consistency
- [ ] Read #file:Addressing Function Approximation Error in Actor-Critic Methods.tex
- [ ] Focus sections:
  - Reward signal requirements for accurate Q-value estimation
  - Impact of reward noise/inconsistency on overestimation bias
  - Discussion on reward function design principles
- [ ] Extract relevant quotes for justification
- [ ] Document findings in: `TASK_1.7_STOPPING_PENALTY_ANALYSIS.md`

#### Task 1.4: Review Related Papers on Reward Design
- [ ] **Lane Keeping Paper** (#file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex)
  - How they handle lane invasion penalties
  - Stopping behavior in highway scenarios
- [ ] **Interpretable E2E Driving** (#file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex)
  - Urban scenario reward design (stops at intersections, traffic lights)
  - Safety reward component implementation
- [ ] Document findings in: `TASK_1.7_STOPPING_PENALTY_ANALYSIS.md`

---

### Phase 2: Issue Investigation 🔍 (Estimated: 1-2 hours)

#### Task 2.1: Debug Safety Persistence (Issue 1.5)
- [ ] **Read current implementation**:
  - [ ] `av_td3_system/src/environment/reward_functions.py` lines 659-859 (`_calculate_safety_reward`)
  - [ ] `av_td3_system/src/environment/sensors.py` - `OffroadDetector` class
  - [ ] `av_td3_system/src/environment/sensors.py` - `LaneInvasionDetector` recovery logic
- [ ] **Trace data flow**:
  - [ ] How `offroad_detected` flag is set/cleared
  - [ ] How PBRS proximity penalties are calculated and reset
  - [ ] Check if `-0.607` value corresponds to specific penalty component
- [ ] **Identify root cause**:
  - [ ] Is state being cached somewhere?
  - [ ] Is reset logic being called properly?
  - [ ] Are proximity penalties persisting after obstacle clears?
- [ ] **Document findings** in: `TASK_1.5_SAFETY_PERSISTENCE_FIX.md`

#### Task 2.2: Debug Lane Invasion Detection (Issue 1.6)
- [ ] **Read current implementation**:
  - [ ] `av_td3_system/src/environment/sensors.py` lines 569-719 (`LaneInvasionDetector`)
  - [ ] `av_td3_system/src/environment/reward_functions.py` lines 360-510 (lane keeping reward)
  - [ ] `av_td3_system/src/environment/carla_env.py` - check how `is_invading_lane()` is called
- [ ] **Test sensor callback**:
  - [ ] Add debug logging to `_on_lane_invasion()` callback
  - [ ] Add debug logging to `is_invading_lane()` method
  - [ ] Add debug logging to lane keeping reward calculation
  - [ ] Run manual test: cross lane marking, observe log sequence
- [ ] **Check timing issues**:
  - [ ] Verify `step_invasion_count` is set before reward calculation
  - [ ] Verify `reset_step_counter()` is called AFTER reward calculation, not before
  - [ ] Check thread lock acquisition order
- [ ] **Document findings** in: `TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md`

#### Task 2.3: Analyze Stopping Penalty (Issue 1.7)
- [ ] **Locate penalty source**:
  - [ ] Grep search for `-0.5` in reward_functions.py
  - [ ] Check all reward components when `velocity = 0.0`
  - [ ] Identify which component generates this value
- [ ] **Determine intent**:
  - [ ] Is this from progress reward (should be separate)?
  - [ ] Is this from safety PBRS (proximity penalty)?
  - [ ] Is this explicit stopping penalty (if so, where is it configured)?
- [ ] **Consult papers**:
  - [ ] What do TD3/related papers recommend?
  - [ ] Is this standard practice or bug?
- [ ] **Document findings** in: `TASK_1.7_STOPPING_PENALTY_ANALYSIS.md`

---

### Phase 3: Implementation 🛠️ (Estimated: 2-3 hours)

#### Task 3.1: Fix Safety Persistence (Issue 1.5)
**Prerequisites**: Task 2.1 complete

**Potential Fix Scenarios**:

**Scenario A: State Not Clearing**
```python
# If root cause is: OffroadDetector not clearing state after recovery
# File: av_td3_system/src/environment/sensors.py

class OffroadDetector:
    def check_offroad(self) -> bool:
        """Check if on drivable surface."""
        with self.offroad_lock:
            # ... existing waypoint check ...
            
            # FIX: Explicitly clear cached state when on drivable lane
            if waypoint.lane_type in drivable_lane_types:
                self.is_offroad = False  # ← Ensure flag is cleared
                return False
            else:
                self.is_offroad = True
                return True
```

**Scenario B: PBRS Proximity Penalty Persisting**
```python
# If root cause is: Proximity penalty calculated even when obstacle cleared
# File: av_td3_system/src/environment/reward_functions.py

def _calculate_safety_reward(self, ...):
    safety = 0.0
    
    # FIX: Only apply proximity penalty if obstacle is CURRENTLY present
    if distance_to_nearest_obstacle is not None:
        if distance_to_nearest_obstacle < 10.0:  # ← Check this condition
            proximity_penalty = -1.0 / max(distance_to_nearest_obstacle, 0.5)
            safety += proximity_penalty
        # ELSE: distance >= 10.0 → NO penalty (clear area)
    # ELSE: No obstacle detected → NO penalty
    
    return safety
```

**Scenario C: Reset Logic Not Called**
```python
# If root cause is: Episode reset not clearing sensor state
# File: av_td3_system/src/environment/carla_env.py

def reset(self, ...):
    # ... existing reset logic ...
    
    # FIX: Ensure ALL sensors reset state
    self.sensors.reset()  # ← Should call:
    # - self.offroad_detector.reset()
    # - self.lane_invasion_detector.reset()
    # - self.collision_detector.reset()
```

**Implementation Steps**:
- [ ] Identify which scenario (A, B, or C) matches root cause from Task 2.1
- [ ] Implement fix following official CARLA patterns
- [ ] Add comprehensive logging for debugging
- [ ] Add unit test for recovery scenario
- [ ] Update documentation in `TASK_1.5_SAFETY_PERSISTENCE_FIX.md`

---

#### Task 3.2: Fix Lane Invasion Detection (Issue 1.6)
**Prerequisites**: Task 2.2 complete

**Potential Fix Scenarios**:

**Scenario A: Step Counter Reset Timing**
```python
# If root cause is: reset_step_counter() called too early
# File: av_td3_system/src/environment/carla_env.py

def step(self, action):
    # ... apply action, tick simulation ...
    
    # 1. GET sensor data (invasion count should be available)
    lane_invasion_count = self.sensors.lane_invasion_detector.get_step_invasion_count()
    
    # 2. CALCULATE reward (uses invasion count)
    reward_dict = self.reward_calculator.calculate(
        ...,
        lane_invasion_detected=(lane_invasion_count > 0),  # ← Read before reset
    )
    
    # 3. RESET counter for NEXT step
    self.sensors.lane_invasion_detector.reset_step_counter()  # ← After reward calc
    
    return obs, reward, terminated, truncated, info
```

**Scenario B: Race Condition with Callback**
```python
# If root cause is: Callback fires after step() reads invasion_count
# File: av_td3_system/src/environment/sensors.py

class LaneInvasionDetector:
    def _on_lane_invasion(self, event):
        """Callback when lane marking crossed."""
        with self.invasion_lock:  # ← Thread-safe
            self.lane_invaded = True
            self.step_invasion_count = 1  # ← Atomic set
            self.invasion_event = event
```

**Scenario C: Recovery Clearing Flag Too Soon**
```python
# If root cause is: is_invading_lane() clears flag before reward reads it
# File: av_td3_system/src/environment/sensors.py

class LaneInvasionDetector:
    def is_invading_lane(self, lateral_deviation=None, lane_half_width=None):
        """Check invasion status."""
        with self.invasion_lock:
            # FIX: Only clear PERSISTENT flag, not STEP counter
            # step_invasion_count should remain until explicitly reset
            
            if lateral_deviation is not None and lane_half_width is not None:
                recovery_threshold = lane_half_width * 0.8
                
                if abs(lateral_deviation) < recovery_threshold:
                    self.lane_invaded = False  # ← Clear persistent flag only
                    # DO NOT clear step_invasion_count here!
            
            return self.lane_invaded  # Return persistent state
```

**Implementation Steps**:
- [ ] Identify which scenario matches root cause from Task 2.2
- [ ] Implement fix ensuring proper timing
- [ ] Add debug logging to verify event sequence:
  ```python
  self.logger.debug(f"[STEP-{step_num}] lane_invasion_count={count} BEFORE reward")
  # ... calculate reward ...
  self.logger.debug(f"[STEP-{step_num}] Resetting invasion counter")
  ```
- [ ] Test with manual control: cross marking → verify warning + penalty
- [ ] Update documentation in `TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md`

---

#### Task 3.3: Resolve Stopping Penalty (Issue 1.7)
**Prerequisites**: Task 2.3 complete

**Decision Tree**:

```
Is -0.5 penalty when stopped INTENTIONAL?
│
├─ YES (from papers/PBRS theory) → Document rationale, keep behavior
│   └─ Action: Write justification in TASK_1.7_STOPPING_PENALTY_ANALYSIS.md
│       - Quote relevant papers
│       - Explain why this encourages progress
│       - Note any scenarios where this might be problematic (traffic lights)
│
└─ NO (bug/unintended) → Fix implementation
    │
    ├─ Source: Progress reward giving negative when stopped
    │   └─ Fix: Progress should be 0.0 when no movement, not negative
    │
    ├─ Source: Safety PBRS proximity penalty when vehicle is parked
    │   └─ Fix: Only apply proximity penalty when velocity > threshold
    │
    └─ Source: Explicit stopping penalty in config
        └─ Fix: Remove or make conditional (only penalize if should be moving)
```

**Implementation (if bug confirmed)**:
```python
# Example Fix: If proximity penalty shouldn't apply when stopped
# File: av_td3_system/src/environment/reward_functions.py

def _calculate_safety_reward(self, ..., velocity, ...):
    safety = 0.0
    
    # PBRS proximity guidance
    if distance_to_nearest_obstacle is not None:
        # FIX: Only apply proximity penalty if vehicle is moving
        # Rationale: Stopped vehicle at safe distance shouldn't be penalized
        if velocity > 0.5 and distance_to_nearest_obstacle < 10.0:  # ← Add velocity check
            proximity_penalty = -1.0 / max(distance_to_nearest_obstacle, 0.5)
            safety += proximity_penalty
    
    return safety
```

**Implementation Steps**:
- [ ] Review paper findings from Task 1.3, 1.4
- [ ] Make decision: bug or feature?
- [ ] If bug: Implement fix with clear rationale
- [ ] If feature: Document why this is correct behavior
- [ ] Add test case for stopped state reward
- [ ] Update documentation in `TASK_1.7_STOPPING_PENALTY_ANALYSIS.md`

---

### Phase 4: Validation Testing 🧪 (Estimated: 1-2 hours)

#### Task 4.1: Create Test Scenarios

**Test Scenario 1: Lane Invasion → Recovery**
```bash
# Expected behavior after fixes:
1. Drive center lane → safety=0.0, lane_keeping=0.0 ✓
2. Cross lane marking → WARNING logged, lane_keeping=-1.0 ✓
3. Return to center (lat_dev < 0.7m) → safety=0.0, lane_keeping=0.0 ✓
4. Continue driving straight → safety=0.0 (NO persistence) ✓
```

**Test Scenario 2: Offroad → Recovery**
```bash
# Expected behavior after fixes:
1. Drive on sidewalk → safety=-10.0 ✓
2. Return to drivable lane → offroad_detected=False
3. Stop at lane center → safety=0.0 (or -0.5 if stopping penalty confirmed as feature)
4. Start moving forward → safety=0.0 (NO cached negative value) ✓
```

**Test Scenario 3: Stopped State Reward**
```bash
# Test stopping penalty behavior:
1. Drive normally at 30 km/h → safety=0.0
2. Gradual brake to full stop → safety=? (document observed value)
3. Remain stopped for 5 seconds → safety=? (should be consistent)
4. Accelerate back to 30 km/h → safety=0.0
```

**Test Scenario 4: Lane Invasion Detection Consistency**
```bash
# Verify every crossing triggers penalty:
1. Cross left lane marking → WARNING + penalty=-1.0 ✓
2. Return to center → penalty clears ✓
3. Cross right lane marking → WARNING + penalty=-1.0 ✓
4. Return to center → penalty clears ✓
5. Repeat 10 times → 100% detection rate ✓
```

#### Task 4.2: Run Manual Validation

- [ ] Start CARLA: `./CarlaUE4.sh -quality-level=Low`
- [ ] Run validation script:
  ```bash
  python scripts/validate_rewards_manual.py \
      --config config/baseline_config.yaml \
      --output-dir validation_logs/fixes_1.5_1.6_1.7
  ```
- [ ] Execute each test scenario
- [ ] Record results in validation log
- [ ] Take screenshots of HUD for documentation

#### Task 4.3: Analyze Validation Logs

- [ ] Run analysis script:
  ```bash
  python scripts/analyze_reward_validation.py \
      --log validation_logs/fixes_1.5_1.6_1.7/reward_validation_*.json \
      --output-dir validation_logs/fixes_1.5_1.6_1.7/analysis
  ```
- [ ] Review generated report for:
  - [ ] Zero critical issues
  - [ ] Safety reward returns to 0.0 after recovery
  - [ ] Lane invasion detection consistency (100%)
  - [ ] Stopping penalty behavior documented

---

### Phase 5: Documentation 📝 (Estimated: 30-60 minutes)

#### Task 5.1: Create Fix Documentation Files

- [ ] **TASK_1.5_SAFETY_PERSISTENCE_FIX.md**:
  - Problem statement
  - Root cause analysis
  - Solution implementation
  - Testing results
  - Before/after comparisons

- [ ] **TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md**:
  - Problem statement
  - Root cause analysis (timing/threading)
  - Solution implementation
  - Testing results (detection rate improvement)

- [ ] **TASK_1.7_STOPPING_PENALTY_ANALYSIS.md**:
  - Investigation findings
  - Paper references and quotes
  - Decision rationale (bug vs feature)
  - Implementation changes (if any)
  - Testing results

#### Task 5.2: Update Main Todo List

- [ ] Mark completed tasks as ✅
- [ ] Document any new issues discovered
- [ ] Update priority rankings if needed
- [ ] Create follow-up tasks if fixes reveal deeper issues

#### Task 5.3: Commit Changes

```bash
# Commit each fix separately for clear history

# Fix 1.5: Safety persistence
git add av_td3_system/src/environment/sensors.py
git add av_td3_system/src/environment/reward_functions.py
git add av_td3_system/docs/day-24/todo-manual-reward-fixes/TASK_1.5_SAFETY_PERSISTENCE_FIX.md
git commit -m "fix(rewards): clear safety penalty state after recovery to drivable lane

Root cause: [Document based on investigation]
Solution: [Document based on implementation]
Testing: Manual validation with 4 scenarios, 100% recovery rate

Refs: CARLA docs lane_invasion, TD3 paper reward consistency
Issue: 1.5 from manual validation session"

# Fix 1.6: Lane invasion detection
git add av_td3_system/src/environment/sensors.py
git add av_td3_system/src/environment/carla_env.py
git add av_td3_system/docs/day-24/todo-manual-reward-fixes/TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md
git commit -m "fix(sensors): ensure lane invasion penalty applies consistently

Root cause: [Document based on investigation]
Solution: [Document based on implementation]
Testing: 10 crossing tests, 100% detection and penalty application

Refs: CARLA lane_invasion sensor, threading best practices
Issue: 1.6 from manual validation session"

# Fix/Analysis 1.7: Stopping penalty
git add av_td3_system/docs/day-24/todo-manual-reward-fixes/TASK_1.7_STOPPING_PENALTY_ANALYSIS.md
# (+ any code changes if bug was fixed)
git commit -m "docs/fix(rewards): analyze and document stopping penalty behavior

Investigation: [Document findings]
Decision: [Bug vs feature]
Action: [Document fix or justification]
Testing: Stopped state scenarios

Refs: TD3 paper, Lane Keeping paper, PBRS theory
Issue: 1.7 from manual validation session"
```

---

## Success Criteria ✅

### Phase 1-2 Complete When:
- [ ] All 4 documentation URLs fetched and key points extracted
- [ ] All 3 papers reviewed for relevant guidance
- [ ] Root causes identified for all 3 issues
- [ ] Fix strategies documented with code examples

### Phase 3 Complete When:
- [ ] All fixes implemented with proper error handling
- [ ] Code follows CARLA/Gymnasium best practices
- [ ] Comprehensive logging added for debugging
- [ ] Unit tests created for fixed scenarios

### Phase 4 Complete When:
- [ ] All 4 test scenarios pass with expected behavior
- [ ] Validation logs show zero critical issues
- [ ] Analysis report confirms fixes resolved problems
- [ ] Screenshots/evidence collected for documentation

### Phase 5 Complete When:
- [ ] All 3 fix documentation files created
- [ ] Commits pushed with descriptive messages
- [ ] Main todo list updated
- [ ] Ready to proceed to next validation tasks

---

## Timeline Estimate

| Phase | Tasks | Estimated Time | Dependencies |
|-------|-------|----------------|--------------|
| **Phase 1** | Documentation Research | 30-45 min | None |
| **Phase 2** | Investigation | 1-2 hours | Phase 1 |
| **Phase 3** | Implementation | 2-3 hours | Phase 2 |
| **Phase 4** | Validation Testing | 1-2 hours | Phase 3 |
| **Phase 5** | Documentation | 30-60 min | Phase 4 |
| **TOTAL** | **All Phases** | **5-8.5 hours** | **~1 work day** |

---

## Risk Assessment

**Low Risk**:
- Issue 1.7 (stopping penalty) - analysis only, low code impact

**Medium Risk**:
- Issue 1.5 (safety persistence) - state management fix, well-scoped

**High Risk**:
- Issue 1.6 (lane invasion) - threading/timing issues can be subtle

**Mitigation Strategies**:
- Fetch official docs BEFORE implementing (avoid assumptions)
- Add extensive logging for debugging
- Test each fix independently
- Rollback capability (git commits per fix)

---

## Next Steps After Completion

After all 3 issues (1.5, 1.6, 1.7) are fixed and validated:

1. **Return to Main Todo List** (#file:README_REWARD_VALIDATION.md)
   - Task 2: Fix Comfort Reward Penalizing Normal Movement
   - Task 3: Fix Progress Reward Discontinuity
   - Task 4: Investigate Route Completion Reward

2. **Comprehensive Validation Session**
   - Run extended manual testing (2000+ steps)
   - All scenarios from reward_validation_guide.md
   - Generate final analysis report

3. **Proceed to Training**
   - TD3 training with validated reward function
   - DDPG baseline comparison
   - Document methodology for paper

---

**END OF TODO LIST**
