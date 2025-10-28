# TD3 Training Recovery Plan - Action Items# Next Steps: Testing with Memory Optimizations



**Date**: October 26, 2025  ## Current Status ✅

**Status**: Training stopped at step 13,300 - Analysis complete

**Documents**: See `TRAINING_ANALYSIS_STEP_13300.md` for full analysis### Code Complete

- ✅ Waypoint passing threshold fixed (distance-based, 5m radius)

---- ✅ CNN feature extractor implemented (MobileNetV3-Small, 512 features)

- ✅ Visual TD3 agent wrapper created (handles Dict observations)

## 🎯 **Quick Summary**- ✅ Visual testing script ready (OpenCV display)

- ✅ Memory optimization applied (expandable_segments + batch_size=64)

### What Happened- ✅ Memory monitoring added (GPU memory logging)

- ✅ **Steps 11,200-11,500**: Agent learned to move (8-11 km/h, +0.72 reward) 🎉

- ❌ **Steps 11,600-13,300**: Regressed back to 0.0 km/h (learning instability)### Documentation Fetched

- **Checkpoint available**: td3_scenario_0_step_10000.pth (before learning started)- ✅ CARLA maps/waypoints navigation

- ✅ PyTorch pretrained models (MobileNet, ResNet)

### Critical Bugs Found- ✅ PyTorch CUDA memory management (expandable_segments)

1. ❌ **Evaluation function**: Uses wrong timeout (30k instead of 1k steps)

2. ❌ **Scenario config**: Mismatch (20 NPCs configured, 50 NPCs used)### Files Modified/Created

3. ⚠️ **Learning regression**: Agent forgot how to move after initial success1. `src/environment/waypoint_manager.py` - Distance-based threshold

2. `src/networks/cnn_extractor.py` - MobileNetV3 feature extractor

---3. `src/agents/visual_td3_agent.py` - Dict observation wrapper

4. `scripts/test_visual_navigation.py` - Visual test with monitoring

## 📋 **Priority Actions**5. `scripts/run_visual_test_docker.sh` - Launcher with memory optimization

6. `config/td3_config_lowmem.yaml` - Low-memory configuration

### 1. Fix Evaluation Function (`scripts/train_td3.py`)7. `docs/MEMORY_OPTIMIZATION.md` - Complete optimization guide



**Line ~771**: Change timeout from `max_timesteps` to `max_episode_steps`---



```python## Immediate Next Steps 🎯

# BEFORE (BUG):

max_eval_steps = self.max_timesteps  # Uses 30,000!### Step 1: Restart CARLA (Clear Memory Fragmentation)



# AFTER (FIX):Current CARLA process may have memory fragmentation from previous runs. Restart to get clean slate:

max_eval_steps = self.td3_config.get("training", {}).get("max_episode_steps", 1000)

``````bash

# Stop current CARLA

### 2. Use Separate Eval Environmentdocker-compose down



**Lines 749-801**: Don't reuse training environment during evaluation# Start fresh CARLA

docker-compose up -d carla

```python

def evaluate(self) -> dict:# Verify it's running

    # Create temporary eval environment (don't reuse self.env)docker ps | grep carla

    eval_env = CARLANavigationEnv(nvidia-smi  # Should show fresh memory allocation

        carla_config_path=self.carla_config,```

        td3_config_path=self.agent_config,

        training_config_path="config/training_config.yaml",**Expected:**

    )- CARLA memory usage: ~5.4GB

    - No fragmentation from previous runs

    max_eval_steps = self.td3_config.get("training", {}).get("max_episode_steps", 1000)

    ### Step 2: Run Visual Test with Optimizations

    # ... run evaluations with eval_env ...

    ```bash

    eval_env.close()cd /workspace/av_td3_system

    return results

```# Make script executable

chmod +x scripts/run_visual_test_docker.sh

### 3. Fix Scenario Configuration

# Run visual test (100 steps with OpenCV display)

Check `config/carla_config.yaml` - ensure scenarios are properly defined:bash scripts/run_visual_test_docker.sh

```

```yaml

scenarios:**What this does:**

  - id: 01. Checks CARLA is running

    npcs: 202. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (+ other optimizations)

    spawn_points: random3. Launches Docker container with X11 forwarding

  - id: 14. Runs `test_visual_navigation.py` for 100 steps

    npcs: 505. Displays live camera feed with metrics

  - id: 2

    npcs: 100**Expected Output:**

``````

========================================

### 4. Address Learning Regression🐳 TD3 Visual Navigation - Docker Launcher

========================================

**Option A** (Recommended): Reduce learning rate✅ CARLA server running

✅ DISPLAY=:0

Edit `config/td3_config.yaml`:🧠 Applying PyTorch CUDA memory optimizations:

```yaml   - expandable_segments: Reduce fragmentation by 60-80%

algorithm:   - max_split_size_mb: Prevent splitting blocks > 128MB

  learning_rate: 0.0001  # Reduced from 0.0003   - garbage_collection_threshold: Aggressive memory reclaim at 80%



networks:🚀 Starting visual navigation test...

  actor:

    learning_rate: 0.0001================================================================================

  critic:🎥 TD3 VISUAL NAVIGATION TESTER

    learning_rate: 0.0001================================================================================

```Timestamp: 2025-01-XX XX:XX:XX

Max steps: 100

**Option B**: Increase explorationDisplay info overlay: True

```yaml

algorithm:🧠 PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8

  exploration_noise: 0.2  # Increased from 0.1🧠 Initial GPU Memory - Allocated: 0.50GB | Reserved: 0.50GB | Peak: 0.50GB

```

🌍 Initializing CARLA environment...

**Option C**: Strengthen movement incentive✅ Environment ready

```yaml🧠 After CARLA env GPU Memory - Allocated: 0.50GB | Reserved: 0.50GB | Peak: 0.50GB

reward:

  weights:🤖 Initializing TD3 agent...

    efficiency: 2.0  # Doubled from 1.0🧠 After TD3 agent GPU Memory - Allocated: 0.55GB | Reserved: 0.56GB | Peak: 0.55GB

    progress: 10.0  # Doubled from 5.0✅ Agent ready (device: cuda:0)

```

🎬 Starting episode...

---[OpenCV window opens showing front camera view]

Step 1/100 | Speed: 0.0 km/h | Reward: 0.00 | Steering: 0.00 | Throttle: 0.00

## 🚀 **Resume Training**...

```

After fixes, resume from checkpoint:

### Step 3: Verify Results

```bash

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system**Success Criteria:**

- [ ] No CUDA Out of Memory error

python3 scripts/train_td3.py \- [ ] OpenCV window displays camera feed

  --checkpoint data/checkpoints/td3_scenario_0_step_10000.pth \- [ ] Episode runs for >1 step (confirms waypoint fix)

  --scenario 0 \- [ ] Episode completes 100 steps or reaches goal

  --max-timesteps 50000 \  # Extended from 30k- [ ] Memory stays below 6GB throughout

  --eval-freq 5000 \

  --checkpoint-freq 10000 \**Check memory during run:**

  --device cpu \```bash

  2>&1 | tee training_50k_fixed.log# In another terminal

```watch -n 1 nvidia-smi

```

**Why step_10000 checkpoint?**

- Last checkpoint before learning started**Expected memory usage:**

- Before regression occurred```

- Clean slate with new learning parametersCARLA:        5.4GB  (90%)

PyTorch:      0.2GB  (3%)

---Overhead:     0.1GB  (2%)

Total:        5.7GB  (95% of 6GB) ✅

## 📊 **Monitor Progress**```



### TensorBoard### Step 4: Analyze Results

```bash

tensorboard --logdir data/logs --port 6006After successful run, check:

```

1. **Waypoint Fix:**

**Watch for**:   - Episode lasted >1 step

- `train/episode_reward` - Should increase   - Vehicle progressed through waypoints properly

- `progress/speed_kmh` - Should stay > 5 km/h (no regression to 0.0!)   - No immediate jump to last waypoint

- `train/critic_loss` - Should decrease

2. **Memory Optimization:**

### Checkpoints   - No OOM errors

- Step 15,000: Check if agent still moving   - Memory logs show stable usage

- Step 20,000: Evaluate performance   - expandable_segments reduced fragmentation

- Step 30,000: Compare with previous run

- Step 50,000: Final evaluation3. **Agent Behavior:**

   - Agent takes actions (steering, throttle)

---   - Actions are reasonable (not NaN or extreme)

   - Agent responds to camera input

## ✅ **Success Criteria**

---

### After 20k steps:

- ✅ Agent consistently moves (>5 km/h)## If Still OOM ⚠️

- ✅ No regression to 0.0 km/h for >1000 steps

- ✅ Episode rewards improving### Option A: Reduce Batch Size Further

```yaml

### After 50k steps:# Edit config/td3_config_lowmem.yaml

- ✅ Average speed 8-12 km/h (near target)batch_size: 32  # Down from 64

- ✅ Success rate > 5%```

- ✅ Evaluation mean reward > -200

### Option B: CARLA Low Quality

---```bash

# Edit docker-compose.yml

## 📚 **Literature Reference**command: ./CarlaUE4.sh -RenderOffScreen -quality-level=Low

```

| Paper | Training Steps | Note |

|-------|----------------|------|Expected savings: ~500-700MB

| Pérez-Gil et al. (2022) | 1M steps | CARLA, 85% success |

| Elallid et al. (2023) | 500k steps | TD3-CARLA, stable |### Option C: CPU Agent (Last Resort)

| Fujimoto et al. (2018) | 1M steps | Original TD3 |```python

# Edit test_visual_navigation.py

**Takeaway**: 30k steps likely too short. Aim for **100k-500k** minimum.agent = TD3Agent(..., device='cpu')

```

---

Very slow but guaranteed to work.

## 🔍 **Testing Checkpoint**

---

Before resuming training, test the saved checkpoint:

## After Successful Test ✅

```bash

python3 scripts/evaluate_td3.py \### Next: Prepare for Training

  --checkpoint data/checkpoints/td3_scenario_0_step_10000.pth \

  --scenario 0 \1. **Update training script** with memory optimization:

  --num-episodes 20 \   ```bash

  --output-dir data/evaluation_step_10000   # Add to scripts/train_td3.py or use environment variable

```   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

   ```

**Expected**: Random-like behavior (checkpoint from exploration phase)

2. **Use low-memory config** for initial test:

---   ```bash

   python scripts/train_td3.py \

## 📝 **Next Report**     --config config/td3_config_lowmem.yaml \

     --scenario scenario1 \

After starting training, report:     --max-steps 1000  # Short test run

1. ✅ Code fixes applied (which ones?)   ```

2. ✅ Training command used

3. ⏳ Status after 1,000 steps (moving?)3. **Monitor memory** during training:

4. ⏳ Status after 15,000 steps (regression check)   ```bash

5. ⏳ Final evaluation at 50,000 steps   watch -n 1 nvidia-smi

   ```

---

4. **If stable** → proceed with full training:

**Full Analysis**: See `TRAINING_ANALYSIS_STEP_13300.md`     ```bash

**Priority**: HIGH - Fix bugs before resuming     python scripts/train_td3.py \

**Estimated Time**: 1-2 hours fixes + 6-12 hours training     --config config/td3_config_lowmem.yaml \

     --scenario scenario1 \
     --max-steps 1000000  # Full 1M steps
   ```

---

## Timeline

### Immediate (Today)
- [ ] Restart CARLA (5 min)
- [ ] Run visual test (10 min)
- [ ] Verify results (5 min)

### If Successful (Next)
- [ ] Test training script (1 hour for 1K steps)
- [ ] Start full training (40-50 hours per scenario)
- [ ] Monitor and collect results

### Backup Plan (If OOM Persists)
- [ ] Use university HPC cluster (A100 40GB GPU)
- [ ] No memory constraints
- [ ] Can use original config (batch_size=256)

---

## Key Commands Summary

```bash
# 1. Restart CARLA
docker-compose down && docker-compose up -d carla

# 2. Run visual test
bash scripts/run_visual_test_docker.sh

# 3. Monitor memory
watch -n 1 nvidia-smi

# 4. Check logs
docker logs td3-av-system-visual

# 5. If successful, test training
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
python scripts/train_td3.py --config config/td3_config_lowmem.yaml --max-steps 1000
```

---

## Expected Outcome 🎉

**Best Case:**
- ✅ Visual test runs successfully
- ✅ Waypoint fix confirmed (episode >1 step)
- ✅ Memory optimization works (no OOM)
- ✅ Ready for full training

**Worst Case:**
- ❌ Still OOM with all optimizations
- ➡️ Use university HPC cluster for training
- ➡️ Continue development on local machine

**Most Likely:**
- ✅ Visual test succeeds with optimizations
- ✅ Training works with batch_size=64
- ✅ Slightly slower than batch_size=256 but acceptable
- ✅ Full training possible on 6GB GPU

---

*Ready to execute! Run Step 1 when ready.*
