# Reward Function Validation Guide

## Overview

This guide provides a systematic approach to validating the reward function used in the TD3-based autonomous vehicle system. Proper reward validation ensures:

1. **Scientific Reproducibility**: All algorithms (TD3, DDPG, baseline) use correctly calculated rewards
2. **Training Success**: Incorrect rewards can prevent learning or cause dangerous behavior
3. **Paper Credibility**: Validated reward function strengthens methodology section

## Validation Tools

### 1. Manual Control Script (`validate_rewards_manual.py`)

Interactive PyGame-based interface for driving the vehicle manually while observing real-time reward calculations.

**Features**:
- WSAD keyboard control
- Real-time reward component display
- Scenario logging
- Episode management

**Controls**:
```
W/S     - Throttle/Brake
A/D     - Steer Left/Right
SPACE   - Hand Brake
R       - Reset Episode
P       - Pause/Resume Logging
Q       - Quit
1-5     - Trigger Test Scenarios
```

### 2. Analysis Script (`analyze_reward_validation.py`)

Post-session analysis tool that validates reward correlations, detects anomalies, and generates visualization plots.

**Outputs**:
- Statistical validation report (Markdown)
- Correlation plots
- Time-series visualizations
- Anomaly detection results

## Validation Methodology

### Phase 1: Basic Validation (30 minutes)

**Objective**: Verify reward function computes correctly under normal driving conditions.

#### 1.1 Start CARLA Simulator

```bash
# Terminal 1: Start CARLA server
cd /path/to/carla
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600
```

#### 1.2 Run Manual Control Script

```bash
# Terminal 2: Start validation script
cd /path/to/av_td3_system
python scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/session_01
```

#### 1.3 Test Basic Scenarios

Perform the following maneuvers while observing the HUD:

**Test 1: Lane Center Driving**
- Drive straight at constant speed (30 km/h)
- **Expected**: 
  - `lane_keeping_reward` ≈ 0 (minimal lateral deviation)
  - `efficiency_reward` > 0 (near target speed)
  - `comfort_penalty` ≈ 0 (smooth driving)
  - `safety_penalty` = 0 (no violations)

**Test 2: Lane Boundary Approach**
- Drive near lane edge (stay in lane)
- **Expected**:
  - `lane_keeping_reward` < 0 (increasing with distance from center)
  - Penalty should scale smoothly, not suddenly

**Test 3: Speed Variation**
- Accelerate from 0 to 50 km/h
- **Expected**:
  - `efficiency_reward` increases as speed approaches target
  - `efficiency_reward` decreases if exceeding target significantly

**Test 4: Smooth Deceleration**
- Gradual brake from 50 to 0 km/h
- **Expected**:
  - `comfort_penalty` ≈ 0 (smooth deceleration)
  - No inappropriate penalties

**Validation Checklist**:
- [ ] Lane keeping penalty increases with lateral deviation
- [ ] Efficiency reward peaks near target speed
- [ ] Comfort penalty remains near zero during smooth driving
- [ ] Total reward equals sum of components (check HUD)

### Phase 2: Edge Case Validation (1 hour)

**Objective**: Test reward behavior in challenging scenarios that may expose bugs.

#### 2.1 Intersection Navigation

**Critical Test** (relates to Bug #7 - waypoint issues at intersections):

```bash
# Continue from Phase 1 or restart
# Drive to intersection in Town01
```

**Test 5: Right Turn at Intersection**
- Approach intersection
- Signal and turn right smoothly
- **Expected**:
  - Lateral deviation may temporarily increase during turn
  - `lane_keeping_penalty` should NOT be excessively harsh during legal turn
  - After turn completion, penalty should return to normal

**Test 6: Left Turn at Intersection**
- Same as Test 5, but left turn
- **Watch for**: Inappropriate penalties during turn execution

**Validation**:
- [ ] Lateral deviation penalty appropriate during turns
- [ ] No excessive penalties for legal maneuvering
- [ ] Waypoint tracking works correctly at intersections

#### 2.2 Lane Change Scenarios

**Test 7: Smooth Lane Change**
- Signal lane change
- Execute smooth merge
- **Expected**:
  - Temporary increase in `lane_keeping_penalty`
  - Penalty reduces after lane change complete
  - `comfort_penalty` minimal if executed smoothly

**Test 8: Abrupt Lane Change**
- Quick lane change (simulate avoidance)
- **Expected**:
  - Higher `comfort_penalty` due to lateral acceleration
  - `lane_keeping_penalty` during transition

**Validation**:
- [ ] Lane change penalties temporary
- [ ] Comfort penalty scales with maneuver aggressiveness

#### 2.3 Safety Scenarios

**Test 9: Near Collision** (use scenario trigger key `3`)
- Approach obstacle/vehicle closely
- **Expected**:
  - `safety_penalty` << 0 (large negative)
  - Should trigger before actual collision if possible

**Test 10: Off-Road Detection** (use scenario trigger key `4`)
- Drive partially off-road
- **Expected**:
  - Large `safety_penalty`
  - Episode termination if fully off-road

**Validation**:
- [ ] Safety penalties trigger before collision
- [ ] Penalties proportional to danger level
- [ ] Termination occurs at appropriate thresholds

#### 2.4 Emergency Maneuvers

**Test 11: Emergency Brake** (use scenario trigger key `2`)
- Emergency brake from high speed
- **Expected**:
  - High `comfort_penalty` (large deceleration)
  - But NOT a `safety_penalty` (legal emergency stop)

**Validation**:
- [ ] Comfort penalty reflects jerk
- [ ] No inappropriate safety penalties for legal emergency maneuvers

### Phase 3: Statistical Analysis (30 minutes)

**Objective**: Verify reward correlations match design specifications.

#### 3.1 Run Analysis Script

```bash
# After completing Phase 1 & 2
python scripts/analyze_reward_validation.py \
    --log validation_logs/session_01/reward_validation_YYYYMMDD_HHMMSS.json \
    --output-dir validation_logs/session_01/analysis
```

#### 3.2 Review Generated Report

Check the markdown report for:

**Critical Issues** (🔴):
- Reward components don't sum to total
- Lane keeping penalty has positive correlation with deviation
- Safety penalties not triggering

**Warnings** (⚠️):
- Weak correlations
- Unusual distributions
- Statistical outliers

**Action Required**:
- If critical issues found → **STOP** → Fix reward function before training
- If warnings found → Investigate, may require reward tuning

#### 3.3 Examine Correlation Plots

**Plot 1: `lateral_deviation_correlation.png`**
- Should show **negative correlation** (line slopes downward)
- Higher deviation → more negative reward

**Plot 2: `speed_efficiency_correlation.png`**
- Should show peak near target speed
- Rewards decrease away from target

**Plot 3: `correlation_heatmap.png`**
- `abs_lateral_deviation` ↔ `lane_keeping_reward`: Strong negative
- `velocity` ↔ `efficiency_reward`: Positive up to target, then negative
- `comfort_penalty` ↔ velocity changes: Negative correlation

**Validation Checklist**:
- [ ] All correlation plots match expected patterns
- [ ] No unexpected positive correlations for penalties
- [ ] Heatmap shows logical relationships

### Phase 4: Scenario-Specific Testing (1 hour)

**Objective**: Validate reward behavior for paper-specific scenarios.

Based on your paper requirements, test:

#### 4.1 High Traffic Density Scenario

```bash
# Modify config to spawn NPCs
# Or manually add vehicles in CARLA
```

**Test 12: Car Following**
- Follow vehicle at safe distance
- **Expected**:
  - `efficiency_reward` may be lower (reduced speed)
  - NO `safety_penalty` if maintaining safe distance
  - Small `progress_reward` for maintaining route

**Test 13: Overtaking**
- Legal overtake of slow vehicle
- **Expected**:
  - Temporary `lane_keeping_penalty` during lane change
  - `efficiency_reward` increases after overtake (if speed increases)
  - No `safety_penalty` if executed safely

#### 4.2 Urban Navigation

**Test 14: Traffic Light Stop**
- Approach red traffic light
- Stop and wait
- **Expected**:
  - NO inappropriate penalty for stopping at red light
  - `efficiency_reward` should not heavily penalize legal stop
  - `progress_reward` continues during wait (or minimal penalty)

**Test 15: Pedestrian Crossing**
- Yield to pedestrian
- **Expected**:
  - NO `safety_penalty` for yielding
  - Legal yielding should not be punished

### Phase 5: Documentation for Paper (30 minutes)

**Objective**: Generate evidence and documentation for paper's methodology section.

#### 5.1 Aggregate Validation Results

Create summary document:

```markdown
# Reward Function Validation Results

## Validation Sessions
- **Date**: YYYY-MM-DD
- **Total Steps Validated**: XXXX
- **Scenarios Tested**: 15
- **Critical Issues Found**: 0
- **Warnings**: X

## Key Findings
1. Lane keeping penalty correlation: r = -0.XX (p < 0.001)
2. Efficiency reward peak at: XX km/h (target: 30 km/h)
3. Safety penalty activation: XX% of unsafe scenarios
4. Reward component summation residual: < 0.0001

## Validation Plots
[Include correlation plots, time series]

## Conclusion
Reward function validated against design specifications. 
All components compute correctly and correlate as expected.
Safe to proceed with TD3/DDPG training.
```

#### 5.2 Include in Paper

**Methodology Section**:
```latex
\subsection{Reward Function Validation}

Prior to training, we conducted systematic validation of the reward 
function using manual control sessions ($n=XX$ scenarios, $XXXX$ steps). 
Statistical analysis confirmed:

\begin{itemize}
    \item Lane keeping penalty strongly correlates with lateral 
          deviation ($r=-0.XX$, $p<0.001$)
    \item Efficiency reward maximizes near target velocity 
          ($v_{target}=30$ km/h)
    \item Safety penalties trigger appropriately in hazardous 
          scenarios ($100\%$ detection rate)
    \item Reward components sum correctly (residual $<10^{-4}$)
\end{itemize}

These validation steps ensure reproducibility and correctness of 
the reinforcement learning signal across all experimental conditions.
```

**Supplementary Materials**:
- Include correlation plots
- Provide validation report as supplementary PDF
- Share validation logs for reproducibility

## Common Issues and Solutions

### Issue 1: Reward Components Don't Sum to Total

**Symptom**: Large residual in summation check (>0.001)

**Possible Causes**:
- Missing component in calculation
- Duplicate component addition
- Numerical precision issues

**Solution**:
```python
# In reward_functions.py, verify:
total_reward = (
    efficiency_reward +
    lane_keeping_reward +
    comfort_penalty +
    safety_penalty +
    progress_reward
)
# Ensure ALL components included, none duplicated
```

### Issue 2: Lane Keeping Penalty Has Wrong Sign

**Symptom**: Positive correlation with lateral deviation

**Possible Causes**:
- Incorrect penalty sign in code
- Reward vs. penalty confusion

**Solution**:
```python
# Should be NEGATIVE for deviation:
lane_keeping_reward = -weight * abs(lateral_deviation)

# NOT:
lane_keeping_reward = weight * abs(lateral_deviation)  # WRONG
```

### Issue 3: Safety Penalty Too Weak

**Symptom**: Agent learns to collide because penalty insufficient

**Possible Causes**:
- Weight too small
- Penalty not activating

**Solution**:
```python
# Ensure collision penalty is dominant:
collision_penalty = -10.0  # Large enough to outweigh other rewards

# Verify activation:
if collision_detected:
    safety_penalty = collision_penalty
    assert abs(safety_penalty) > sum(all_other_rewards)
```

### Issue 4: Intersection Waypoint Issues (Bug #7)

**Symptom**: Large lateral deviation penalties during legal turns

**Possible Causes**:
- Waypoint tracking issues at intersections
- Incorrect reference path

**Solution**:
- Verify `waypoint_manager.py` handles intersection waypoints
- Check if lateral deviation calculated from correct reference
- May need special handling for turn maneuvers

## Advanced Validation

### Automated Testing

For continuous validation during development:

```python
# scripts/test_reward_function.py
import pytest
from src.environment.reward_functions import TD3RewardFunction

def test_lane_keeping_penalty_increases_with_deviation():
    """Lane keeping penalty should increase with lateral deviation."""
    reward_func = TD3RewardFunction(config)
    
    # Test multiple deviation values
    deviations = [0.0, 0.5, 1.0, 2.0]
    rewards = []
    
    for dev in deviations:
        state = create_mock_state(lateral_deviation=dev)
        reward_components = reward_func.calculate(state)
        rewards.append(reward_components['lane_keeping'])
    
    # Should be monotonically decreasing (more negative)
    for i in range(len(rewards) - 1):
        assert rewards[i] > rewards[i+1], "Lane keeping penalty should increase with deviation"

def test_efficiency_reward_peaks_at_target_speed():
    """Efficiency reward should peak near target speed."""
    reward_func = TD3RewardFunction(config)
    target_speed = 30.0 / 3.6  # 30 km/h in m/s
    
    speeds = [10, 20, 30, 40, 50]  # km/h
    rewards = []
    
    for speed_kmh in speeds:
        state = create_mock_state(velocity=speed_kmh / 3.6)
        reward_components = reward_func.calculate(state)
        rewards.append(reward_components['efficiency'])
    
    # Maximum should be at or near target
    max_idx = rewards.index(max(rewards))
    assert speeds[max_idx] == 30, f"Efficiency peaks at {speeds[max_idx]} km/h, expected 30 km/h"

# Run with: pytest scripts/test_reward_function.py
```

### Continuous Monitoring During Training

Add logging to track reward statistics:

```python
# In training loop
def log_reward_statistics(episode_rewards, episode_num):
    """Log reward component statistics for monitoring."""
    
    # Calculate statistics
    total_mean = np.mean([r['total'] for r in episode_rewards])
    efficiency_mean = np.mean([r['efficiency'] for r in episode_rewards])
    # ... etc.
    
    # Log to tensorboard
    writer.add_scalar('reward/total_mean', total_mean, episode_num)
    writer.add_scalar('reward/efficiency_mean', efficiency_mean, episode_num)
    
    # Alert if anomalies
    if abs(total_mean) > 100:  # Sanity check
        logging.warning(f"Unusual reward magnitude at episode {episode_num}: {total_mean}")
```

## Checklist: Ready for Training

Before starting TD3/DDPG training, ensure:

- [ ] Manual validation completed (Phases 1-4)
- [ ] Statistical analysis shows no critical issues
- [ ] Correlation plots match expected patterns
- [ ] All test scenarios passed
- [ ] Reward components sum correctly (residual < 0.001)
- [ ] Lane keeping penalty: strong negative correlation with deviation
- [ ] Efficiency reward: peaks near target speed
- [ ] Safety penalties: activate in dangerous scenarios
- [ ] Comfort penalties: scale with jerk/acceleration
- [ ] No unexpected reward outliers
- [ ] Edge cases tested (intersections, lane changes, emergencies)
- [ ] Documentation prepared for paper
- [ ] Validation logs saved for reproducibility

## References

### Official Documentation
- CARLA 0.9.16: https://carla.readthedocs.io/en/latest/
- Gymnasium: https://gymnasium.farama.org/
- TD3 Paper: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods"

### Related Work
- Lane keeping reward design: Referenced papers in `#file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex`
- Urban driving rewards: `#file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex`

### Project Files
- Reward function: `src/environment/reward_functions.py`
- Environment: `src/environment/carla_env.py`
- Waypoint manager: `src/environment/waypoint_manager.py`
- Baseline comparison: `FinalProject/module_7.py`

---

**Last Updated**: January 2025
**Author**: Based on CARLA documentation and reinforcement learning best practices
