# Quick Start Guide: Systematic Reward Validation Fixes

**Date**: November 24, 2025  
**Status**: Ready to Begin  
**Estimated Time**: 5-8.5 hours (1 work day)

---

## What We're Fixing

Three critical issues discovered during manual validation testing:

1. **Issue 1.5**: Safety penalty persists at `-0.607` after recovery (should return to `0.0`)
2. **Issue 1.6**: Lane invasion detection inconsistent (sometimes no warning/penalty)
3. **Issue 1.7**: Stopped state receives `-0.5` safety penalty (investigate if bug or feature)

---

## How to Start

### Step 1: Read the Master Plan
```bash
# Open comprehensive todo list
code av_td3_system/docs/day-24/todo-manual-reward-fixes/SYSTEMATIC_REWARD_FIXES_TODO.md
```

This document contains:
- Detailed problem statements with examples
- Investigation strategies for each issue
- Potential fix scenarios with code examples
- Testing procedures
- Success criteria

### Step 2: Begin Phase 1 (Documentation Research)

**Time**: 30-45 minutes  
**Goal**: Gather official documentation and paper guidance BEFORE coding

**Tasks**:
```bash
# Task 1.1: Fetch CARLA Lane Invasion Sensor docs
- Focus: Event timing, thread safety, state persistence
- URL: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

# Task 1.2: Fetch CARLA Waypoint API docs  
- Focus: get_waypoint() with project_to_road=False
- URL: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint

# Task 1.3: Review TD3 Paper
- Focus: Reward consistency requirements for Q-value estimation
- File: #file:Addressing Function Approximation Error in Actor-Critic Methods.tex

# Task 1.4: Review Related Papers
- Lane Keeping Paper: #file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex
- Interpretable E2E: #file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex
```

**Deliverable**: Document findings in respective `TASK_1.5_*.md`, `TASK_1.6_*.md`, `TASK_1.7_*.md` files

---

## Documentation Template

When you fetch documentation or read papers, use this template:

```markdown
# TASK_1.X_[ISSUE_NAME]_FIX.md

## Problem Statement
[Copy from SYSTEMATIC_REWARD_FIXES_TODO.md]

## Documentation Research

### CARLA Documentation
**Source**: [URL]
**Key Findings**:
- Finding 1: [Quote + explanation]
- Finding 2: [Quote + explanation]
**Implications for our fix**: [How this guides implementation]

### Paper References
**Paper**: [Title]
**Section**: [Number/name]
**Key Quote**: "[exact quote]"
**Relevance**: [How this applies to our issue]

## Root Cause Analysis
[Fill in during Phase 2]

## Solution Implementation
[Fill in during Phase 3]

## Testing Results
[Fill in during Phase 4]
```

---

## Work Session Checklist

### Before You Start Coding (Phase 1-2)
- [ ] CARLA lane invasion sensor documentation fetched and read
- [ ] CARLA waypoint API documentation fetched and read
- [ ] TD3 paper reviewed for reward consistency requirements
- [ ] Related papers reviewed for reward design guidance
- [ ] Root cause identified for Issue 1.5 (safety persistence)
- [ ] Root cause identified for Issue 1.6 (lane invasion detection)
- [ ] Root cause identified for Issue 1.7 (stopping penalty)
- [ ] Fix strategy documented with code examples

### During Implementation (Phase 3)
- [ ] Fix 1.5 implemented following CARLA patterns
- [ ] Fix 1.6 implemented with thread safety
- [ ] Fix 1.7 implemented (or documented if feature, not bug)
- [ ] Comprehensive logging added to all fixes
- [ ] Unit tests created for fixed scenarios
- [ ] Code reviewed for error handling

### After Implementation (Phase 4-5)
- [ ] Test Scenario 1: Lane invasion → recovery (penalty clears)
- [ ] Test Scenario 2: Offroad → recovery (no cached negative)
- [ ] Test Scenario 3: Stopped state (consistent reward)
- [ ] Test Scenario 4: Lane invasion consistency (100% detection)
- [ ] Analysis script run (zero critical issues)
- [ ] Documentation files created for all 3 fixes
- [ ] Changes committed separately with descriptive messages

---

## Important Reminders

### DO:
✅ Fetch documentation FIRST, code SECOND  
✅ Test each fix independently  
✅ Add extensive debug logging  
✅ Commit each fix separately  
✅ Document rationale with paper references

### DON'T:
❌ Skip documentation research phase  
❌ Implement fixes based on assumptions  
❌ Mix multiple fixes in one commit  
❌ Remove existing logging/comments  
❌ Change unrelated code

---

## Testing Pattern

For each fix, use this manual testing pattern:

```bash
# 1. Start CARLA
./CarlaUE4.sh -quality-level=Low

# 2. Run manual validation
cd av_td3_system
python scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/fix_1.X_test

# 3. Execute specific test scenario (see SYSTEMATIC_REWARD_FIXES_TODO.md)
# Example for Issue 1.5:
# - Drive on sidewalk → observe safety=-10.0
# - Return to lane center → observe safety should return to 0.0
# - Continue driving → verify no cached negative value

# 4. Review logs
cat validation_logs/fix_1.X_test/reward_validation_*.json | grep "safety"

# 5. Run analysis
python scripts/analyze_reward_validation.py \
    --log validation_logs/fix_1.X_test/reward_validation_*.json \
    --output-dir validation_logs/fix_1.X_test/analysis
```

---

## File References

**Master Todo**: `av_td3_system/docs/day-24/todo-manual-reward-fixes/SYSTEMATIC_REWARD_FIXES_TODO.md`

**Code Files to Modify**:
- `av_td3_system/src/environment/sensors.py` (LaneInvasionDetector, OffroadDetector)
- `av_td3_system/src/environment/reward_functions.py` (_calculate_safety_reward)
- `av_td3_system/src/environment/carla_env.py` (step method, reset method)

**Documentation to Create**:
- `TASK_1.5_SAFETY_PERSISTENCE_FIX.md`
- `TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md`
- `TASK_1.7_STOPPING_PENALTY_ANALYSIS.md`

**Previous Fix Reference**:
- `av_td3_system/docs/day-24/todo-manual-reward-fixes/TASK_1_OFFROAD_DETECTION_FIX.md`

---

## Next Steps After This Todo List

Once Issues 1.5, 1.6, 1.7 are fixed:

1. **Fix Comfort Reward** (penalizes normal movement)
2. **Fix Progress Reward** (discontinuity issues)
3. **Investigate Route Completion** (verify +100 bonus)
4. **Comprehensive Validation** (2000+ steps, all scenarios)
5. **Generate Paper Documentation** (plots, statistics, methodology)
6. **Begin TD3 Training** (with validated reward function)

---

## Support Resources

- **CARLA Documentation**: https://carla.readthedocs.io/en/latest/
- **Gymnasium API**: https://gymnasium.farama.org/api/env/
- **TD3 Paper**: #file:Addressing Function Approximation Error in Actor-Critic Methods.tex
- **Validation Guide**: #file:reward_validation_guide.md
- **Previous Fix**: #file:TASK_1_OFFROAD_DETECTION_FIX.md

---

**Ready to begin? Start with Phase 1: Documentation Research!**
