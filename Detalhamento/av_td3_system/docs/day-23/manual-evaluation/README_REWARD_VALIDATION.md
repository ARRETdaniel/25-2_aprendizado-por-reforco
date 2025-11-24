# Quick Start: Reward Validation System

## Purpose

Validate that your CARLA environment's reward function calculates correctly **before** starting expensive TD3/DDPG training. This ensures scientific reproducibility and prevents training failures due to incorrect reward signals.

## What You Need

### 1. CARLA 0.9.16 Running
```bash
cd /path/to/carla
./CarlaUE4.sh -quality-level=Low -windowed
```

### 2. Environment Configuration
Ensure your `config/baseline_config.yaml` exists and is properly configured.

### 3. Python Dependencies
```bash
pip install numpy pygame pyyaml pandas matplotlib seaborn scipy
```

## 5-Minute Quick Test

### Step 1: Run Unit Tests (2 minutes)

```bash
cd /path/to/av_td3_system
python scripts/test_reward_components.py
```

**Expected Output:**
```
BASIC REWARD FUNCTION TESTS (Manual)
======================================================================

TestLaneKeepingReward:
  ✓ test_zero_deviation_gives_zero_penalty
  ✓ test_penalty_increases_with_deviation
  ✓ test_symmetric_deviation_penalty
  ✓ test_penalty_bounded

...

RESULTS: 15/15 tests passed
✅ All tests passed!
```

If tests fail → Fix reward function logic before continuing.

### Step 2: Run Manual Validation (3 minutes test drive)

```bash
python scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/quick_test
```

**What to do:**
1. Drive straight using `W` (forward) and `A/D` (steering)
2. Watch HUD display reward components in real-time
3. Try lane boundary approach - see `lane_keeping_reward` become negative
4. Press `Q` to quit when satisfied

**What to look for:**
- ✅ `total_reward` equals sum of components
- ✅ `lane_keeping_reward` becomes more negative as you deviate from center
- ✅ `efficiency_reward` changes based on speed
- ✅ No sudden jumps or NaN values

### Step 3: Quick Analysis (Optional)

```bash
python scripts/analyze_reward_validation.py \
    --log validation_logs/quick_test/reward_validation_*.json
```

Check for critical issues (🔴). If none, you're good to proceed!

## Full Validation (2-3 hours)

For thorough validation before starting training, follow the comprehensive guide:

**📖 See: `docs/reward_validation_guide.md`**

This includes:
- Phase 1: Basic scenarios (30 min)
- Phase 2: Edge cases (1 hour)
- Phase 3: Statistical analysis (30 min)
- Phase 4: Scenario testing (1 hour)
- Phase 5: Paper documentation (30 min)

## Common Issues

### Issue: "CARLA connection refused"
**Solution:** Make sure CARLA server is running first:
```bash
./CarlaUE4.sh -quality-level=Low -windowed
```

### Issue: "Module 'carla_env' not found"
**Solution:** You need to integrate with your existing environment:
1. Ensure `src/environment/carla_env.py` exists
2. Modify it to include reward component logging (see REWARD_VALIDATION_SUMMARY.md)

### Issue: "Reward components don't sum to total"
**Solution:** Check your `reward_functions.py`:
```python
# Ensure all components are included:
total = (
    efficiency_reward +
    lane_keeping_reward +
    comfort_penalty +
    safety_penalty +
    progress_reward
)
# No component should be added twice
```

### Issue: "Lane keeping has positive correlation"
**Solution:** Check penalty sign:
```python
# Should be NEGATIVE:
lane_keeping_reward = -weight * abs(lateral_deviation)

# NOT positive:
lane_keeping_reward = weight * abs(lateral_deviation)  # WRONG!
```

## Files Created

```
av_td3_system/
├── scripts/
│   ├── validate_rewards_manual.py      # Manual control interface
│   ├── analyze_reward_validation.py    # Statistical analysis
│   └── test_reward_components.py       # Unit tests
├── docs/
│   ├── reward_validation_guide.md      # Complete methodology
│   └── REWARD_VALIDATION_SUMMARY.md    # Technical documentation
└── validation_logs/                    # Output directory (created automatically)
    └── session_XX/
        ├── reward_validation_*.json    # Raw data
        └── analysis/
            ├── validation_report_*.md  # Analysis report
            └── *.png                   # Plots
```

## Next Steps After Validation

Once validation confirms reward function is correct:

1. ✅ **Proceed with Training**
   ```bash
   python train_td3.py --config config/td3_config.yaml
   ```

2. 📊 **Monitor During Training**
   - Watch for reward distributions matching validation
   - Alert if sudden changes in reward statistics

3. 📝 **Document for Paper**
   - Include validation results in methodology
   - Attach plots to supplementary materials
   - Reference validation logs for reproducibility

## Questions?

1. **How long does full validation take?**  
   About 2-3 hours including analysis and documentation.

2. **Can I skip validation?**  
   Not recommended. Incorrect rewards can cause:
   - Training failure (no learning)
   - Dangerous behavior (wrong optimization)
   - Wasted compute time (days of training wasted)

3. **What if I find critical issues?**  
   Fix the reward function code, then re-run validation. Don't proceed to training with broken rewards.

4. **How often should I validate?**  
   - Always before starting training campaign
   - After any reward function modifications
   - If training behavior seems incorrect

5. **Do TD3, DDPG, and baseline use same rewards?**  
   Yes! That's why validation is critical - it affects all algorithms equally.

## Help & Documentation

- **Quick Start**: This file (README_REWARD_VALIDATION.md)
- **Complete Guide**: `docs/reward_validation_guide.md`
- **Technical Details**: `docs/REWARD_VALIDATION_SUMMARY.md`
- **CARLA Docs**: https://carla.readthedocs.io/en/latest/
- **Gymnasium Docs**: https://gymnasium.farama.org/

## Status Checklist

Before starting TD3 training, confirm:

- [ ] Unit tests pass (`test_reward_components.py`)
- [ ] Manual validation completed
- [ ] Statistical analysis shows no critical issues
- [ ] Correlation plots look correct
- [ ] Lane keeping: negative correlation with deviation
- [ ] Efficiency: peaks near target speed
- [ ] Safety penalties activate appropriately
- [ ] Components sum to total (residual < 0.001)
- [ ] Edge cases tested (intersections, lane changes)
- [ ] Validation results documented for paper

**If all checked → ✅ Ready for training!**

---

**Created**: January 2025  
**For**: CARLA TD3 Autonomous Vehicle Project  
**Goal**: Ensure reward function correctness before training
