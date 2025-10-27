# TD3 Validation Training Guide (20k Steps)

**Purpose:** Comprehensive validation of the TD3 autonomous vehicle solution before committing to the full 1M-step training run on the university supercomputer.

**Date:** October 26, 2024  
**Status:** Ready to execute

---

## Overview

This validation run will train the TD3 agent for **20,000 timesteps** (~2-3 hours on laptop) with comprehensive data collection and automated analysis to verify:

1. ✅ **Reward function works correctly** - No "stand still" exploit regression
2. ✅ **CNN feature extraction provides useful visual information** - Not degenerate/constant
3. ✅ **Waypoint data is properly integrated** - Spatially sensible, vehicle-frame coordinates
4. ✅ **Agent is learning to navigate** - Metrics improve over time
5. ✅ **All components work together** - CARLA, environment, agent, training loop

---

## What Has Changed Since Last Validation

### Previous 800-Step Test (Oct 26, Earlier)
- **Purpose:** Quick fix verification
- **Duration:** 800 steps (~10 minutes)
- **Result:** ✅ Confirmed reward sign fixes work
- **Limitation:** Too short to verify learning

### Current 20k-Step Validation (Oct 26, Now)
- **Purpose:** Full solution validation before supercomputer deployment
- **Duration:** 20,000 steps (~2-3 hours)
- **Includes:**
  - Comprehensive data collection (CNN features, waypoints, rewards, vehicle state)
  - 10 evaluation checkpoints (every 2k steps)
  - 4 model checkpoints (every 5k steps)
  - Automated analysis script with pass/fail criteria
  - Visualization plots for all metrics
  - Detailed validation report

---

## Prerequisites

### 1. CARLA Server Must Be Running

Start CARLA server in a separate terminal:

```bash
docker run -d --name carla-server --rm \
  --network host --gpus all \
  -e SDL_VIDEODRIVER=offscreen \
  carlasim/carla:0.9.16 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

Verify it's running:
```bash
docker ps | grep carla-server
```

### 2. Docker Image Built

The TD3 training image should already be built:
```bash
docker images | grep td3-av-system:v2.0-python310
```

If not found, build it:
```bash
cd av_td3_system
docker build -t td3-av-system:v2.0-python310 -f docker/Dockerfile .
```

### 3. X11 Forwarding (Optional, for Debug Visualization)

Enable X server access:
```bash
xhost +local:docker
```

---

## Running the Validation Training

### Step 1: Navigate to Project Directory

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system
```

### Step 2: Execute Validation Training Script

```bash
./scripts/run_validation_training.sh
```

**What This Does:**
- Pre-flight checks (CARLA running, Docker image exists)
- Launches TD3 training for 20,000 steps
- Enables debug mode (detailed logging every 10 steps)
- Evaluates every 2,000 steps (10 total evaluations)
- Saves checkpoints every 5,000 steps (4 total)
- Logs everything to `data/logs/validation_training_20k_YYYYMMDD_HHMMSS.log`
- Writes TensorBoard logs for real-time monitoring

### Step 3: Monitor Training Progress

**Option A: Watch log file**
```bash
tail -f data/logs/validation_training_20k_*.log
```

**Option B: TensorBoard (in another terminal)**
```bash
cd av_td3_system
tensorboard --logdir data/logs
```
Then open browser to: http://localhost:6006

**Option C: Debug visualization (if X11 enabled)**
- OpenCV window will show camera feed and real-time metrics
- Press 'p' to pause, 'q' to quit

### Step 4: Wait for Completion

**Expected Timeline:**
- **Steps 1-10,000:** Exploration phase (random actions, filling replay buffer)
  - Expected: Mostly stationary or erratic movement
  - This is NORMAL - agent not learning yet
  
- **Steps 10,001-20,000:** Learning phase (policy updates)
  - Expected: Agent starts learning to move
  - Reward should improve gradually
  - Speed should increase

**Total Runtime:** ~2-3 hours on laptop CPU

---

## Analyzing the Results

### Automated Analysis

After training completes, run the analysis script:

```bash
python3 scripts/analyze_validation_run.py \
  --log-file data/logs/validation_training_20k_YYYYMMDD_HHMMSS.log \
  --output-dir data/validation_analysis
```

**What This Generates:**

1. **Validation Report** (`validation_report.txt`)
   - Pass/Fail for each validation criterion
   - Statistical analysis of metrics
   - Overall verdict: Ready for full training? Yes/No
   - Recommendations

2. **Visualization Plots** (`validation_analysis.png`)
   - Reward components over time
   - Total reward trajectory
   - Vehicle speed over time
   - CNN feature L2 norm
   - Episode rewards
   - Waypoint spatial distribution

### Manual Review

**Key Files to Check:**

1. **Training Log** (`data/logs/validation_training_20k_*.log`)
   - Search for errors or warnings
   - Check reward values (should be negative when stationary)
   - Verify CNN features vary (L2 norm changes)
   - Confirm waypoints ahead of vehicle (x > 0)

2. **TensorBoard Logs** (view in browser)
   - `train/episode_reward` - Should show upward trend
   - `progress/speed_kmh` - Should increase over time
   - `train/critic_loss` - Should decrease and stabilize
   - `eval/success_rate` - Should increase

3. **Checkpoints** (`data/checkpoints/`)
   - Should have 4 checkpoint files:
     - `td3_scenario_0_step_5000.pth`
     - `td3_scenario_0_step_10000.pth`
     - `td3_scenario_0_step_15000.pth`
     - `td3_scenario_0_step_20000.pth`

---

## Validation Criteria (Pass/Fail)

### ✅ Must Pass for Full Training Approval

**1. Reward Function Validation**
- [ ] Standing still (speed < 1 km/h) gives **negative reward** (around -53)
- [ ] Moving (speed > 5 km/h) gives **better reward** than stationary
- [ ] Safety component is **negative when stationary** (not positive)
- [ ] All reward components in expected ranges

**2. CNN Feature Validation**
- [ ] CNN features **not constant** (L2 norm std > 0.01)
- [ ] CNN features **not degenerate** (mean L2 norm > 0.1)
- [ ] Features show **temporal variation** across training

**3. Waypoint Validation**
- [ ] **>80% of waypoints ahead** of vehicle (x > 0 in vehicle frame)
- [ ] Waypoints within **reasonable distance** (< 50m typically)
- [ ] Waypoints show **spatial variation** (not fixed)

**4. Learning Progress Validation**
- [ ] Episode rewards show **upward trend** (late > early)
- [ ] Average speed **increases over time**
- [ ] Collision rate **decreases or stays low**

### ⚠️ Expected Behaviors (Not Failures)

- **Low speeds during exploration phase (steps 1-10k):** This is NORMAL
  - Agent using random actions to populate replay buffer
  - Vehicle will be mostly stationary or moving slowly
  - Rewards will be consistently negative
  - **Learning starts at step 10,001**

- **Initial learning instability (steps 10k-12k):** Common
  - Agent transitioning from random to learned policy
  - Rewards may fluctuate
  - Speed may vary widely
  - Should stabilize after a few thousand steps

---

## Decision Tree: Proceed to Full Training?

### ✅ GREEN LIGHT - Proceed with Full Training

**Criteria:**
- All 4 validation checks PASS
- No crashes or CARLA disconnections
- TensorBoard shows learning curves trending correctly
- Checkpoints saved successfully

**Action:**
```bash
# Full training command (1M steps, ~3-5 days on GPU)
./scripts/run_full_training.sh --scenario 0 --max-timesteps 1000000 --device cuda
```

### ⚠️ YELLOW LIGHT - Review and Decide

**Criteria:**
- 3/4 validation checks pass
- Learning progress shows improvement but not statistically significant
- Minor warnings but no critical failures

**Action:**
- Review specific warnings in validation report
- Consider extending validation to 50k steps for more data
- Consult with advisor before proceeding

### 🛑 RED LIGHT - DO NOT Proceed

**Criteria:**
- Any critical validation check FAILS
- Reward sign regression (standing still gives positive reward)
- CNN features degenerate (constant or near-zero)
- No learning improvement after 20k steps
- Crashes or CARLA disconnections

**Action:**
1. Review validation report for root cause
2. Fix identified issues
3. Re-run validation training
4. Only proceed after all checks pass

---

## Troubleshooting

### Training Won't Start

**Error:** `CARLA server is NOT running`
```bash
# Start CARLA server
docker run -d --name carla-server --rm \
  --network host --gpus all \
  -e SDL_VIDEODRIVER=offscreen \
  carlasim/carla:0.9.16 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen

# Wait 30 seconds for initialization
sleep 30
```

**Error:** `Docker image NOT found`
```bash
# Build image
cd av_td3_system
docker build -t td3-av-system:v2.0-python310 -f docker/Dockerfile .
```

### Training Crashes

**Symptom:** CARLA connection timeout
- Check CARLA logs: `docker logs carla-server`
- Restart CARLA server
- Re-run training (will resume from last checkpoint if available)

**Symptom:** Out of memory
- Reduce batch size in `config/td3_config.yaml` (256 → 128)
- Use CPU instead of GPU: change `--device cuda` to `--device cpu`

**Symptom:** OpenCV visualization error
- Disable debug mode: remove `--debug` flag from script
- Training will continue without visualization

### Reward Still Positive When Stationary

**This is a CRITICAL bug regression!**

1. Check `config/training_config.yaml`:
   ```yaml
   safety:
     weight: 100.0  # MUST be +100.0, not -100.0
   ```

2. If weight is negative, fix it:
   ```bash
   # Edit config
   nano config/training_config.yaml
   # Change safety.weight to +100.0
   ```

3. Re-run validation training from scratch

---

## Expected Output Summary

### Terminal Output (Sample)

```
[CHECK] ✓ CARLA server is running
[CHECK] ✓ Docker image found: td3-av-system:v2.0-python310

[TRAINING] Environment initialized successfully in 4.1 seconds!
[TRAINING] Beginning training from timestep 1 to 20,000

[EXPLORATION] Processing step 100/20,000...
[DEBUG Step 100] Act=[steer:+0.830, thr/brk:+0.999] | Rew= -53.00 | Speed= 1.1 km/h
   [Reward] Efficiency=-3.00 | Lane=+0.00 | Comfort=+0.00 | Safety=-50.00 | Progress=+0.01

[EXPLORATION] Processing step 10,000/20,000...
[PHASE TRANSITION] Starting LEARNING phase at step 10,001
[PHASE TRANSITION] Replay buffer size: 10,000

[LEARNING] Processing step 10,100/20,000...
[DEBUG Step 10100] Act=[steer:-0.245, thr/brk:+0.678] | Rew= -12.50 | Speed= 8.3 km/h
   [Reward] Efficiency=-1.20 | Lane=+0.50 | Comfort=-0.10 | Safety=+0.00 | Progress=+2.30

[EVAL] Evaluation at timestep 12,000...
[EVAL] Mean Reward: -125.45 | Success Rate: 20.0% | Avg Collisions: 0.40

[TRAINING] Training complete!
```

### Validation Report (Sample)

```
==================================================================================
TD3 VALIDATION TRAINING ANALYSIS REPORT
==================================================================================

[✓] Reward Function
    PASS: Reward function working correctly
      Standing still (< 1 km/h): -53.00 reward (negative ✓)
      Safety when stationary: -50.00 (negative ✓)
      Moving (> 5 km/h): -12.50 reward (better ✓)
      Improvement: 40.50 points

[✓] CNN Features
    PASS: CNN features working correctly
      Mean L2 norm: 0.333 (non-zero ✓)
      Std L2 norm: 0.042 (varying ✓)

[✓] Waypoints
    PASS: Waypoints spatially sensible
      Waypoints ahead: 98.5% (>80% ✓)
      Mean distance: 7.2m

[✓] Learning Progress
    PASS: Agent shows significant learning improvement
      Early episodes reward: -890.50
      Late episodes reward: -420.30
      Improvement: +470.20
      Statistical trend: Spearman ρ=0.652, p=0.001 ✓

==================================================================================
OVERALL VERDICT
==================================================================================

✓✓✓ PASS: TD3 solution is ready for full training ✓✓✓

Recommendations:
  1. Proceed with full 1M-step training on supercomputer
  2. Use same hyperparameters and configuration
  3. Monitor TensorBoard during training
  4. Save checkpoints every 50k-100k steps
```

---

## Next Steps After Validation

### If Validation Passes ✅

1. **Commit validated configuration to git:**
   ```bash
   git add config/ scripts/ docs/
   git commit -m "Validated TD3 configuration for 1M-step training"
   git push origin main
   ```

2. **Transfer to supercomputer:**
   ```bash
   rsync -avz av_td3_system/ username@supercomputer:/path/to/project/
   ```

3. **Launch full training on supercomputer:**
   ```bash
   # On supercomputer (adjust for SLURM/PBS)
   sbatch scripts/run_full_training_slurm.sh
   ```

4. **Monitor remotely:**
   - Set up TensorBoard forwarding: `ssh -L 6006:localhost:6006 username@supercomputer`
   - Check logs periodically
   - Download checkpoints for backup

### If Validation Fails 🛑

1. **Review validation report** - Identify root cause
2. **Fix identified issues** - Code, config, or hyperparameters
3. **Re-run validation** - Full 20k steps again
4. **Document fixes** - Update git commit messages
5. **Only proceed after PASS** - No shortcuts!

---

## Files Created by This Guide

```
av_td3_system/
├── scripts/
│   ├── run_validation_training.sh          (NEW) - Launch 20k validation run
│   ├── analyze_validation_run.py           (NEW) - Automated analysis script
│   └── train_td3.py                        (EXISTING) - Already has debug logging
├── config/
│   ├── training_config.yaml                (FIXED) - Reward weights corrected
│   ├── td3_config.yaml                     (EXISTING) - Algorithm config
│   └── carla_config.yaml                   (EXISTING) - CARLA settings
├── docs/
│   ├── VALIDATION_TRAINING_GUIDE.md        (NEW) - This document
│   ├── EMPIRICAL_VALIDATION_RESULTS.md     (PREVIOUS) - 800-step test results
│   ├── REWARD_FUNCTION_VALIDATION.md       (PREVIOUS) - Detailed analysis
│   └── TODO.md                             (UPDATED) - Task tracking
└── data/
    ├── logs/
    │   └── validation_training_20k_*.log   (GENERATED) - Training logs
    ├── checkpoints/
    │   └── td3_scenario_0_step_*.pth       (GENERATED) - Model checkpoints
    └── validation_analysis/
        ├── validation_report.txt           (GENERATED) - Analysis report
        └── validation_analysis.png         (GENERATED) - Visualization plots
```

---

## FAQ

**Q: Why 20k steps instead of 800 like before?**  
A: The 800-step test only verified reward signs. 20k steps verifies learning capability, which requires the agent to transition from exploration (steps 1-10k) to learning (steps 10k-20k) and show improvement.

**Q: Can I stop training early if I see it's working?**  
A: Not recommended. The automated analysis requires the full 20k steps to make statistically valid conclusions about learning progress.

**Q: What if my laptop can't handle 20k steps?**  
A: You can reduce to 15k minimum, but update the analysis script expectations. Better: use a more powerful machine or the university's GPU cluster.

**Q: How much disk space do I need?**  
A: ~500 MB for logs, checkpoints, and analysis results. Ensure you have 1-2 GB free to be safe.

**Q: Can I run multiple validation runs in parallel?**  
A: No - CARLA server can only handle one client at a time. Run sequentially.

**Q: Should I use GPU or CPU?**  
A: **CPU for validation** (saves GPU for CARLA). **GPU for full 1M-step training** (much faster).

---

**Document Version:** 1.0  
**Last Updated:** October 26, 2024  
**Author:** Daniel Terra  
**Status:** Ready for execution ✅
