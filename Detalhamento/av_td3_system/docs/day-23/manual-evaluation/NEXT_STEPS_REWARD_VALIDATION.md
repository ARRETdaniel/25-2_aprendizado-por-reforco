# Reward Validation Integration - Next Steps

## ✅ Completed: Code Integration

### Modified File: `src/environment/carla_env.py`

**Location:** Line ~787-807 in `step()` method

**Changes Made:**
```python
info = {
    "step": self.current_step,
    "reward_breakdown": reward_dict["breakdown"],  # Existing format (preserved)

    # NEW: Validation-friendly flat format for manual control
    "reward_components": {
        "total": reward,
        "efficiency": reward_dict["breakdown"]["efficiency"][2],
        "lane_keeping": reward_dict["breakdown"]["lane_keeping"][2],
        "comfort": reward_dict["breakdown"]["comfort"][2],
        "safety": reward_dict["breakdown"]["safety"][2],
        "progress": reward_dict["breakdown"]["progress"][2],
    },

    # NEW: State metrics for validation HUD display
    "state": {
        "velocity": vehicle_state["velocity"],
        "lateral_deviation": vehicle_state["lateral_deviation"],
        "heading_error": vehicle_state["heading_error"],
        "distance_to_goal": distance_to_goal,
    },

    # ... rest of info dict unchanged
}
```

**What This Achieves:**
1. ✅ **Gymnasium Best Practice**: Follows official standard for diagnostic info
2. ✅ **Backward Compatible**: Preserves existing `reward_breakdown` format
3. ✅ **Validation Ready**: Adds simple dict format for manual control script
4. ✅ **Paper Documentation**: Provides data for methodology validation section

---

## 📋 Next Steps: Validation Workflow

### Phase 1: Quick Validation Test (15 minutes)

#### Step 1: Start CARLA Server

```bash
# From #file:RUN-COMMAND.md pattern
docker run -d --name carla-server --runtime=nvidia --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound
```

**Verify CARLA is running:**
```bash
docker logs carla-server
# Should show: "Listening for the client at 0.0.0.0:2000"
```

#### Step 2: Run Unit Tests

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Run inside Docker container
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/test_reward_components.py
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

**If Tests Fail:**
- Review test output for specific failures
- Check reward function logic in `src/environment/reward_functions.py`
- Fix issues before proceeding to manual validation

#### Step 3: Run Quick Manual Validation (5 minutes)

```bash
# Enable X11 forwarding for PyGame window
xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

# Run manual control validation
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
  python3 scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/quick_test \
    --max-steps 500
```

**What to Do:**
1. **Drive straight** using `W` (forward) and `A/D` (steering)
2. **Watch HUD** display reward components in real-time
3. **Test lane deviation**: Drive near lane edge, observe `lane_keeping_reward` become negative
4. **Press `Q`** to quit when satisfied

**What to Look For:**
- ✅ `total_reward` displayed matches sum of components
- ✅ `lane_keeping_reward` becomes more negative as you deviate from center
- ✅ `efficiency_reward` changes based on speed
- ✅ No sudden jumps, NaN values, or crashes

**Common Issues & Solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| "CARLA connection refused" | Server not running | Check `docker logs carla-server` |
| "Module 'pygame' not found" | Missing dependency | Install in Docker image: `pip install pygame` |
| Black screen/no camera | Camera not initialized | Check logs for camera sensor errors |
| Reward components all zero | Info dict format wrong | Verify modification in `carla_env.py` |

#### Step 4: Quick Analysis (3 minutes)

```bash
# Run analysis on validation log
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/analyze_reward_validation.py \
    --log validation_logs/quick_test/reward_validation_*.json \
    --output-dir validation_logs/quick_test/analysis
```

**Check Generated Files:**
```bash
ls -la validation_logs/quick_test/analysis/
# Should contain:
#   - validation_report_*.md
#   - lateral_deviation_correlation.png
#   - speed_efficiency_correlation.png
#   - reward_components_timeline.png
#   - correlation_heatmap.png
```

**Review Report:**
```bash
cat validation_logs/quick_test/analysis/validation_report_*.md
```

**Look for:**
- 🔴 **Critical Issues**: Must fix before proceeding
- ⚠️ **Warnings**: Investigate but may proceed
- ℹ️ **Info**: Informational findings

**Decision Point:**
- **If critical issues found** → Fix reward function → Re-run validation
- **If only warnings** → Document and proceed to full validation
- **If all clear** → Proceed to Phase 2 (full validation)

---

### Phase 2: Full Validation Session (2-3 hours)

**Objective:** Comprehensive validation following official guide

**Reference:** `docs/reward_validation_guide.md`

**Phases:**
1. **Basic Validation** (30 min) - Normal driving scenarios
2. **Edge Case Validation** (1 hour) - Intersections, lane changes, emergencies
3. **Statistical Analysis** (30 min) - Correlation validation
4. **Scenario Testing** (1 hour) - High traffic, urban navigation
5. **Documentation** (30 min) - Prepare paper materials

**Commands:**

```bash
# Full validation session (no max-steps limit)
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
  python3 scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/session_$(date +%Y%m%d_%H%M%S)
```

**Test Scenarios Checklist:**

From `reward_validation_guide.md`:

**Phase 1: Basic Validation**
- [ ] Test 1: Lane center driving (30 km/h constant)
- [ ] Test 2: Lane boundary approach
- [ ] Test 3: Speed variation (0→50 km/h)
- [ ] Test 4: Smooth deceleration

**Phase 2: Edge Cases**
- [ ] Test 5: Right turn at intersection (Bug #7 area)
- [ ] Test 6: Left turn at intersection
- [ ] Test 7: Smooth lane change
- [ ] Test 8: Abrupt lane change
- [ ] Test 9: Near collision (scenario key `3`)
- [ ] Test 10: Off-road detection (scenario key `4`)
- [ ] Test 11: Emergency brake (scenario key `2`)

**After Each Session:**
```bash
# Run comprehensive analysis
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/analyze_reward_validation.py \
    --log validation_logs/session_*/reward_validation_*.json \
    --output-dir validation_logs/session_*/analysis
```

---

### Phase 3: Paper Documentation (30 minutes)

**Objective:** Prepare validation results for #file:ourPaper.tex

#### 3.1 Aggregate Results

Create summary document:

```bash
# Create paper documentation folder
mkdir -p validation_logs/paper_materials

# Copy key plots
cp validation_logs/session_*/analysis/lateral_deviation_correlation.png \
   validation_logs/paper_materials/
cp validation_logs/session_*/analysis/speed_efficiency_correlation.png \
   validation_logs/paper_materials/
cp validation_logs/session_*/analysis/correlation_heatmap.png \
   validation_logs/paper_materials/

# Copy validation report
cp validation_logs/session_*/analysis/validation_report_*.md \
   validation_logs/paper_materials/validation_summary.md
```

#### 3.2 Extract Statistics for Paper

From validation report, extract:
- Lane keeping correlation coefficient (e.g., r=-0.85, p<0.001)
- Efficiency reward peak speed (e.g., 30 km/h ± 2 km/h)
- Safety penalty detection rate (e.g., 100%)
- Component summation residual (e.g., <10^-4)

#### 3.3 Update Paper Methodology Section

Add to `#file:ourPaper.tex`:

```latex
\subsection{Reward Function Validation}

Prior to training, we conducted systematic validation of the reward
function using manual control sessions ($n=XX$ episodes, $XXXX$ steps).
Statistical analysis confirmed expected relationships:

\begin{itemize}
    \item Lane keeping penalty strongly correlates with lateral
          deviation ($r=-0.XX$, $p<0.001$)
    \item Efficiency reward maximizes near target velocity
          ($v_{target}=30$ km/h, Gaussian $\sigma=2$ km/h)
    \item Safety penalties trigger reliably in hazardous scenarios
          (collision detection rate: $100\%$)
    \item Reward components sum correctly (numerical residual
          $<10^{-4}$)
\end{itemize}

Validation results and raw logs are provided in supplementary materials.
```

---

## 🎯 Success Criteria

Before proceeding to TD3 training, ensure:

### Technical Validation
- [ ] Unit tests pass (15/15 tests)
- [ ] Manual validation completed without crashes
- [ ] All test scenarios executed
- [ ] Statistical analysis shows no critical issues (🔴)

### Correlation Validation
- [ ] Lane keeping ↔ lateral deviation: r < -0.7 (strong negative)
- [ ] Efficiency ↔ speed: peaks near 30 km/h
- [ ] Safety penalties activate in 100% of unsafe scenarios
- [ ] Component summation residual < 0.001

### Documentation
- [ ] Validation report generated
- [ ] Correlation plots created
- [ ] Paper methodology section updated
- [ ] Supplementary materials prepared

### Edge Cases
- [ ] Intersection navigation validated (Bug #7 area)
- [ ] Lane changes handled correctly
- [ ] Emergency maneuvers don't trigger inappropriate penalties

---

## 📚 Reference Documentation

### Internal Guides
- **Quick Start**: `docs/README_REWARD_VALIDATION.md`
- **Complete Guide**: `docs/reward_validation_guide.md`
- **Technical Details**: `docs/REWARD_VALIDATION_SUMMARY.md`

### Official Documentation
- **Gymnasium API**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step
- **CARLA 0.9.16**: https://carla.readthedocs.io/en/latest/
- **TD3 Paper**: Fujimoto et al., "Addressing Function Approximation Error"

### Related Papers
- #file:ourPaper.tex - Target paper
- #file:Addressing Function Approximation Error in Actor-Critic Methods.tex - TD3 guidance
- #file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex - Termination warning
- #file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex - Interpretability reference

---

## 🐛 Troubleshooting

### Issue: PyGame Window Not Appearing

**Symptoms:** Script runs but no visual window

**Solutions:**
```bash
# Check X11 forwarding
echo $DISPLAY  # Should show :0 or similar

# Re-enable xhost
xhost +local:docker

# Test with simple X11 app
docker run --rm -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ubuntu xeyes  # Should show eyes window
```

### Issue: "reward_components" Key Not Found

**Symptoms:** Manual control script crashes with KeyError

**Cause:** Old version of `carla_env.py` loaded

**Solutions:**
```bash
# Clear Python cache
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# Verify modification in carla_env.py
grep -A 10 "reward_components" src/environment/carla_env.py
# Should show the new dict structure
```

### Issue: Reward Components Don't Sum to Total

**Symptoms:** HUD shows total ≠ sum of components

**Diagnosis:**
```bash
# Enable debug logging
docker run ... \
  -e LOG_LEVEL=DEBUG \
  ... python3 scripts/validate_rewards_manual.py ...
```

**Check:** Reward function in `src/environment/reward_functions.py`

---

## ⏭️ After Validation: Training Pipeline

Once validation confirms reward correctness:

### 1. Proceed with TD3 Training

```bash
# From #file:RUN-COMMAND.md pattern
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 100000 \
    --eval-freq 20000 \
    --checkpoint-freq 10000 \
    --seed 42 \
    --device cuda \
    2>&1 | tee logs/train_td3_validated_$(date +%Y%m%d_%H%M%S).log
```

### 2. Monitor Training

```bash
# TensorBoard for real-time monitoring
tensorboard --logdir data/logs --port 6006

# Watch for reward statistics matching validation baselines
```

### 3. Compare Results

**Expected Outcomes** (from #file:ourPaper.tex):
- TD3 achieves higher success rate vs DDPG baseline
- Reduced critical safety events (60% reduction target)
- Improved policy stability

---

## 📊 Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| **Phase 0** | Code integration (DONE) | ✅ Complete |
| **Phase 1** | Quick validation test | 15 minutes |
| **Phase 2** | Full validation session | 2-3 hours |
| **Phase 3** | Paper documentation | 30 minutes |
| **Total** | **First Complete Validation** | **~3-4 hours** |

**Subsequent validations** (after reward function changes): ~1 hour

---

## 🎓 Scientific Justification

### Why This Approach Follows Best Practices

1. **Gymnasium Standard** (Official Docs)
   - ✅ Uses recommended `info` dict structure
   - ✅ Includes "individual reward terms" as specified

2. **TD3 Paper Principles** (#file:Addressing Function Approximation Error in Actor-Critic Methods.tex)
   - ✅ Validates reward signal quality (reduces variance)
   - ✅ Detects potential sources of approximation error

3. **Lane Keeping Paper Warning** (#file:End-to-End_Deep_Reinforcement_Learning_for_Lane_Keeping_Assist.tex)
   - ✅ Identifies termination patterns
   - ✅ Tunes penalties to balance learning vs safety

4. **Interpretability** (#file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex)
   - ✅ Provides explanation of agent optimization
   - ✅ Enables reward breakdown analysis

5. **Reproducibility** (#file:ourPaper.tex goal)
   - ✅ Documents exact reward calculation
   - ✅ Provides validation logs for peer review
   - ✅ Ensures TD3, DDPG, baseline use same correct reward

---

**Status**: ✅ Ready to begin Phase 1 (Quick Validation Test)

**Next Command to Run**:
```bash
# Step 1: Verify CARLA running
docker logs carla-server

# Step 2: Run unit tests
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/test_reward_components.py
```
