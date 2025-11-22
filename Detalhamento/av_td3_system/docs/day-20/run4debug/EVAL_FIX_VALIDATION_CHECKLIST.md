# EVAL Fix - Validation Checklist

**Date**: November 20, 2025
**Status**: Ready for Testing

---

## ✅ Pre-Flight Checklist (Implementation)

All implementation steps completed:

- [x] **Step 1**: Removed `self.eval_tm_port` configuration
  - Location: `__init__()` lines ~148-156
  - Replaced with single `self.tm_port = 8000`
  - Updated comments to reference CARLA singleton world architecture

- [x] **Step 2**: Added `self.in_eval_phase` flag
  - Location: `__init__()` line ~307-311
  - Tracks evaluation mode state
  - Prevents EVAL experiences in replay buffer

- [x] **Step 3**: Rewrote `evaluate()` method
  - Location: Lines ~1288-1423
  - Uses `self.env` (no separate environment)
  - Added vehicle state validation (before/after)
  - Preserves episode count
  - Resets environment after EVAL
  - Returns `(eval_metrics, fresh_obs_dict)` tuple

- [x] **Step 4**: Updated training loop
  - Location: Lines ~1227-1283
  - Sets `in_eval_phase` flag
  - Unpacks `(eval_metrics, obs_dict)` from evaluate()
  - Resets episode tracking after EVAL
  - Continues training with fresh observation

- [x] **Step 5**: Vehicle state validation logging
  - Logs vehicle ID and is_alive before EVAL
  - Logs vehicle ID and is_alive after EVAL
  - Warns if vehicle destroyed during EVAL
  - Notes ID changes as expected (new spawn)

- [x] **Step 6**: Removed all TM port comments
  - No references to "Option A (Separate TM Ports)"
  - Updated to reference singleton world architecture
  - All comments cite EVAL_PHASE_SOLUTION_ANALYSIS.md

---

## 🧪 Test Plan

### Test #1: 100-Step Micro-Validation

**Purpose**: Verify no vehicle corruption with minimal runtime

**Command**:
```bash
cd /workspace/av_td3_system
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 100 \
  --eval-freq 50 \
  --num-eval-episodes 2 \
  --seed 42 \
  --debug
```

**Expected Timeline**:
- Steps 0-49: EXPLORATION phase (random forward actions)
- Step 50: **EVAL phase** (2 deterministic episodes)
  - Vehicle state logged BEFORE
  - 2 episodes run
  - Vehicle state logged AFTER
  - Fresh observation returned
- Steps 51-99: LEARNING phase (policy + noise, training)
- Step 100: **EVAL phase** (2 deterministic episodes again)

**Success Criteria**:
```
✅ No steering=-973852377088 corruption
✅ No "destroyed actor" errors
✅ Vehicle is_alive=True after EVAL
✅ Training continues after each EVAL
✅ TensorBoard logs created
```

**Expected Log Output**:
```
[EVAL] Starting evaluation phase (deterministic policy, no training)...
[EVAL] Vehicle state before evaluation:
[EVAL]   ID: 123
[EVAL]   is_alive: True
[EVAL] Resetting environment for fresh training episode...
[EVAL] Vehicle state after evaluation:
[EVAL]   ID: 456
[EVAL]   is_alive: True
[EVAL] Note: Vehicle ID changed (123 -> 456)
[EVAL]       This is EXPECTED after env.reset() (respawns actors)
[EVAL] Evaluation complete:
[EVAL]   Mean Reward: XX.XX
[EVAL]   Success Rate: X.X%
[EVAL]   Avg Collisions: X.XX
[EVAL] Returning to training...
```

**Estimated Runtime**: 5-10 minutes

---

### Test #2: 5K Full Validation

**Purpose**: Verify stability over multiple EVAL phases

**Command**:
```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 5000 \
  --eval-freq 1000 \
  --num-eval-episodes 10 \
  --seed 42
```

**Expected Timeline**:
- 5 EVAL phases at steps: 1000, 2000, 3000, 4000, 5000
- Each EVAL: 10 deterministic episodes
- Training continues between each EVAL

**Success Criteria**:
```
✅ All 5 EVAL phases complete successfully
✅ No vehicle corruption at any point
✅ Training continues after each EVAL
✅ TensorBoard shows continuous metrics
✅ No gaps in episode_reward curve
```

**Estimated Runtime**: 30-60 minutes

---

### Test #3: TensorBoard Metrics Verification

**Command**:
```bash
tensorboard --logdir data/logs/
```

**Open in browser**: http://localhost:6006

**Check SCALARS tab**:

#### Evaluation Metrics
- `eval/mean_reward`: Should appear at timesteps 50, 100 (Test #1) or 1000, 2000, ... (Test #2)
- `eval/success_rate`: Same timesteps
- `eval/avg_collisions`: Same timesteps
- `eval/avg_lane_invasions`: Same timesteps
- `eval/avg_episode_length`: Same timesteps

#### Training Metrics (Should NOT have gaps)
- `train/episode_reward`: Should continue smoothly across EVAL phases
- `train/critic_loss`: Should be logged continuously (every 100 steps after learning starts)
- `train/q1_value`: Should not spike or drop at EVAL transitions
- `train/actor_loss`: Should appear every N steps (delayed updates)

#### Gradient Metrics (If debug enabled)
- `gradients/actor_cnn_norm`: Should stay within bounds
- `gradients/critic_cnn_norm`: Should stay within bounds
- No sudden explosions at EVAL boundaries

**Success Criteria**:
```
✅ Eval metrics appear at correct timesteps
✅ Training metrics continuous (no gaps)
✅ No discontinuities at EVAL transitions
✅ Gradients stable across phases
```

---

## 🔍 What to Look For

### ✅ Good Signs
- Vehicle `is_alive=True` after every EVAL
- Vehicle ID changes noted as "EXPECTED"
- Smooth TensorBoard curves
- No CARLA timeout errors
- Training continues after EVAL

### 🚨 Bad Signs (Report if seen)
- Vehicle `is_alive=False` after EVAL
- Steering values like -973852377088
- Gear values like 5649815
- CARLA timeout errors
- Training stops after EVAL
- TensorBoard gaps at EVAL timesteps

---

## 📊 Before vs After Comparison

### Before (Broken)
```
[EVAL] Creating temporary evaluation environment (TM port 8050)...
[EVAL] Closing evaluation environment...
Applied Control: steer=-973852377088.0000  ❌ CORRUPTED!
Hand Brake: True, Reverse: True, Gear: 5649815  ❌ INVALID!
```

### After (Fixed)
```
[EVAL] Starting evaluation phase (deterministic policy, no training)...
[EVAL] Vehicle state before evaluation:
[EVAL]   ID: 123
[EVAL]   is_alive: True
[EVAL] Vehicle state after evaluation:
[EVAL]   ID: 456
[EVAL]   is_alive: True  ✅ VALID!
[EVAL] Returning to training...
```

---

## 📝 Test Execution Instructions

### 1. Activate Environment
```bash
cd /workspace/av_td3_system
# Activate your Python environment if needed
# conda activate your_env
```

### 2. Start CARLA Server
```bash
# In a separate terminal
cd /path/to/carla
./CarlaUE4.sh -RenderOffScreen
```

### 3. Run Test #1 (100-step)
```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 100 \
  --eval-freq 50 \
  --num-eval-episodes 2 \
  --seed 42 \
  --debug
```

### 4. Monitor Logs
Watch for:
- `[EVAL] Vehicle state before evaluation:`
- `[EVAL] Vehicle state after evaluation:`
- `[EVAL] Note: Vehicle ID changed`
- `[EVAL] Returning to training...`

### 5. Check TensorBoard (Optional)
```bash
tensorboard --logdir data/logs/
```

### 6. Run Test #2 (5K - if Test #1 passes)
```bash
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 5000 \
  --eval-freq 1000 \
  --num-eval-episodes 10 \
  --seed 42
```

---

## 📄 Documentation Reference

If issues occur, consult:
- `EVAL_PHASE_SOLUTION_ANALYSIS.md` - Comprehensive technical analysis
- `EVAL_ARCHITECTURE_COMPARISON.md` - Visual architecture diagrams
- `EVAL_SOLUTION_SUMMARY.md` - Executive summary
- `EVAL_FIX_IMPLEMENTATION_COMPLETE.md` - Implementation details

---

## ✅ Final Checklist

Before running tests:
- [ ] CARLA server is running
- [ ] Python environment activated
- [ ] No other training scripts running
- [ ] Sufficient disk space for logs
- [ ] GPU available for CARLA (if needed)

After Test #1 (100-step):
- [ ] No vehicle corruption errors
- [ ] Training completed to step 100
- [ ] Logs show vehicle state validation
- [ ] TensorBoard created in data/logs/

After Test #2 (5K):
- [ ] All 5 EVAL phases completed
- [ ] TensorBoard shows continuous metrics
- [ ] No gaps at EVAL transitions
- [ ] Final results saved to JSON

---

**Implementation Status**: ✅ COMPLETE
**Testing Status**: ⏭️ READY TO RUN
**Next Action**: Execute Test #1 (100-step micro-validation)
