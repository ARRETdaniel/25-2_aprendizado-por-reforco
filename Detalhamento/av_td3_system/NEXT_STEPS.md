# Next Steps: Testing with Memory Optimizations

## Current Status ✅

### Code Complete
- ✅ Waypoint passing threshold fixed (distance-based, 5m radius)
- ✅ CNN feature extractor implemented (MobileNetV3-Small, 512 features)
- ✅ Visual TD3 agent wrapper created (handles Dict observations)
- ✅ Visual testing script ready (OpenCV display)
- ✅ Memory optimization applied (expandable_segments + batch_size=64)
- ✅ Memory monitoring added (GPU memory logging)

### Documentation Fetched
- ✅ CARLA maps/waypoints navigation
- ✅ PyTorch pretrained models (MobileNet, ResNet)
- ✅ PyTorch CUDA memory management (expandable_segments)

### Files Modified/Created
1. `src/environment/waypoint_manager.py` - Distance-based threshold
2. `src/networks/cnn_extractor.py` - MobileNetV3 feature extractor
3. `src/agents/visual_td3_agent.py` - Dict observation wrapper
4. `scripts/test_visual_navigation.py` - Visual test with monitoring
5. `scripts/run_visual_test_docker.sh` - Launcher with memory optimization
6. `config/td3_config_lowmem.yaml` - Low-memory configuration
7. `docs/MEMORY_OPTIMIZATION.md` - Complete optimization guide

---

## Immediate Next Steps 🎯

### Step 1: Restart CARLA (Clear Memory Fragmentation)

Current CARLA process may have memory fragmentation from previous runs. Restart to get clean slate:

```bash
# Stop current CARLA
docker-compose down

# Start fresh CARLA
docker-compose up -d carla

# Verify it's running
docker ps | grep carla
nvidia-smi  # Should show fresh memory allocation
```

**Expected:**
- CARLA memory usage: ~5.4GB
- No fragmentation from previous runs

### Step 2: Run Visual Test with Optimizations

```bash
cd /workspace/av_td3_system

# Make script executable
chmod +x scripts/run_visual_test_docker.sh

# Run visual test (100 steps with OpenCV display)
bash scripts/run_visual_test_docker.sh
```

**What this does:**
1. Checks CARLA is running
2. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (+ other optimizations)
3. Launches Docker container with X11 forwarding
4. Runs `test_visual_navigation.py` for 100 steps
5. Displays live camera feed with metrics

**Expected Output:**
```
========================================
🐳 TD3 Visual Navigation - Docker Launcher
========================================
✅ CARLA server running
✅ DISPLAY=:0
🧠 Applying PyTorch CUDA memory optimizations:
   - expandable_segments: Reduce fragmentation by 60-80%
   - max_split_size_mb: Prevent splitting blocks > 128MB
   - garbage_collection_threshold: Aggressive memory reclaim at 80%

🚀 Starting visual navigation test...

================================================================================
🎥 TD3 VISUAL NAVIGATION TESTER
================================================================================
Timestamp: 2025-01-XX XX:XX:XX
Max steps: 100
Display info overlay: True

🧠 PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8
🧠 Initial GPU Memory - Allocated: 0.50GB | Reserved: 0.50GB | Peak: 0.50GB

🌍 Initializing CARLA environment...
✅ Environment ready
🧠 After CARLA env GPU Memory - Allocated: 0.50GB | Reserved: 0.50GB | Peak: 0.50GB

🤖 Initializing TD3 agent...
🧠 After TD3 agent GPU Memory - Allocated: 0.55GB | Reserved: 0.56GB | Peak: 0.55GB
✅ Agent ready (device: cuda:0)

🎬 Starting episode...
[OpenCV window opens showing front camera view]
Step 1/100 | Speed: 0.0 km/h | Reward: 0.00 | Steering: 0.00 | Throttle: 0.00
...
```

### Step 3: Verify Results

**Success Criteria:**
- [ ] No CUDA Out of Memory error
- [ ] OpenCV window displays camera feed
- [ ] Episode runs for >1 step (confirms waypoint fix)
- [ ] Episode completes 100 steps or reaches goal
- [ ] Memory stays below 6GB throughout

**Check memory during run:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Expected memory usage:**
```
CARLA:        5.4GB  (90%)
PyTorch:      0.2GB  (3%)
Overhead:     0.1GB  (2%)
Total:        5.7GB  (95% of 6GB) ✅
```

### Step 4: Analyze Results

After successful run, check:

1. **Waypoint Fix:**
   - Episode lasted >1 step
   - Vehicle progressed through waypoints properly
   - No immediate jump to last waypoint

2. **Memory Optimization:**
   - No OOM errors
   - Memory logs show stable usage
   - expandable_segments reduced fragmentation

3. **Agent Behavior:**
   - Agent takes actions (steering, throttle)
   - Actions are reasonable (not NaN or extreme)
   - Agent responds to camera input

---

## If Still OOM ⚠️

### Option A: Reduce Batch Size Further
```yaml
# Edit config/td3_config_lowmem.yaml
batch_size: 32  # Down from 64
```

### Option B: CARLA Low Quality
```bash
# Edit docker-compose.yml
command: ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

Expected savings: ~500-700MB

### Option C: CPU Agent (Last Resort)
```python
# Edit test_visual_navigation.py
agent = TD3Agent(..., device='cpu')
```

Very slow but guaranteed to work.

---

## After Successful Test ✅

### Next: Prepare for Training

1. **Update training script** with memory optimization:
   ```bash
   # Add to scripts/train_td3.py or use environment variable
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
   ```

2. **Use low-memory config** for initial test:
   ```bash
   python scripts/train_td3.py \
     --config config/td3_config_lowmem.yaml \
     --scenario scenario1 \
     --max-steps 1000  # Short test run
   ```

3. **Monitor memory** during training:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **If stable** → proceed with full training:
   ```bash
   python scripts/train_td3.py \
     --config config/td3_config_lowmem.yaml \
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
