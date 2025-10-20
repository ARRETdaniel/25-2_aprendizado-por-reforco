# Phase 3 Training Execution Guide

**Status**: ✅ Phase 2 Complete - Ready for Phase 3
**Estimated Duration**: 25 days (250-310 GPU hours)
**GPU Requirement**: NVIDIA RTX 2060 6GB or better
**Start Date**: Based on availability
**End Date**: Results ready for Phase 4 analysis

---

## Overview

Phase 3 executes the main training loop for all agents. This phase trains two deep reinforcement learning agents (TD3 and DDPG) on three traffic scenarios and evaluates the classical IDM+MOBIL baseline. This is the most computationally expensive phase (250+ GPU hours) and must run continuously.

### Key Milestones

1. **TD3 Training**: Complete all 3 scenarios (120-150 GPU hours)
2. **DDPG Training**: Complete all 3 scenarios (120-150 GPU hours)
3. **Baseline Evaluation**: IDM+MOBIL on all scenarios (~10 hours)
4. **Checkpoint Verification**: All models saved and ready for Phase 4

---

## Task #9: Train TD3 on 3 Traffic Scenarios

### Objective

Train TD3 agent on scenarios with increasing traffic complexity (20, 50, 100 NPC vehicles).

### Configuration

```yaml
Algorithm: TD3 (Twin Delayed DDPG)
Timesteps: 1,000,000 per scenario
Exploration Phase: 10,000 steps (random actions)
Batch Size: 256
Replay Buffer Size: 1,000,000
Learning Rate: 3e-4
Soft Update (tau): 0.005
Discount (gamma): 0.99
Policy Update Frequency: 2 (delayed updates)
Target Smoothing: noise=0.2, clip=0.5
Evaluation Frequency: Every 5,000 steps
Checkpoint Frequency: Every 10,000 steps
```

### Pre-Training Checklist

- [ ] CARLA 0.9.16 server installed and tested
- [ ] CARLA running on GPU with sufficient memory
- [ ] PyTorch with CUDA verified
- [ ] av_td3_system code structure validated
- [ ] Configuration files reviewed and customized if needed
- [ ] Disk space verified: minimum 50GB for checkpoints + logs
- [ ] Training environment isolated (no other GPU processes)

### Execution Commands

**Scenario 0 (Low Traffic: 20 vehicles):**

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Option 1: Direct execution
python scripts/train_td3.py \
  --scenario 0 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/td3_scenario_0

# Option 2: Background execution with nohup
nohup python scripts/train_td3.py \
  --scenario 0 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/td3_scenario_0 \
  > td3_scenario_0.log 2>&1 &
```

**Scenario 1 (Medium Traffic: 50 vehicles):**

```bash
# Wait for Scenario 0 to complete OR run in parallel if GPU memory allows
python scripts/train_td3.py \
  --scenario 1 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/td3_scenario_1
```

**Scenario 2 (High Traffic: 100 vehicles):**

```bash
# Wait for Scenario 1 to complete OR run in parallel if GPU memory allows
python scripts/train_td3.py \
  --scenario 2 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/td3_scenario_2
```

### Expected Output

**Per scenario:**

```
logs/td3_scenario_0/
├── checkpoints/
│   ├── td3_checkpoint_10000.pth
│   ├── td3_checkpoint_20000.pth
│   ├── ...
│   └── td3_checkpoint_1000000.pth  (final model)
├── tensorboard/
│   └── events.out.tfevents.xxx
├── training_log.json
└── results.csv
```

**Monitoring:**

```bash
# In separate terminal, monitor training in real-time
tensorboard --logdir=./logs/td3_scenario_0/tensorboard

# Or check the progress file
tail -f ./logs/td3_scenario_0/training_log.json
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size in config (e.g., 128 instead of 256) |
| CARLA connection timeout | Restart CARLA server, verify port 2000 is open |
| Slow training speed | Check GPU utilization with `nvidia-smi`, reduce logging frequency |
| Training crashes | Check logs, ensure sufficient disk space, validate CARLA environment |

### Expected Training Time

- **Scenario 0** (20 NPC): ~40-45 GPU hours
- **Scenario 1** (50 NPC): ~45-50 GPU hours
- **Scenario 2** (100 NPC): ~50-60 GPU hours
- **Total**: ~140-160 GPU hours

---

## Task #10: Train DDPG on 3 Traffic Scenarios

### Objective

Train DDPG baseline on identical scenarios to TD3 for fair algorithmic comparison.

### Configuration

```yaml
Algorithm: DDPG (Deep Deterministic Policy Gradient)
Timesteps: 1,000,000 per scenario
[All other parameters identical to TD3]
Key Difference: Single critic, immediate policy updates (policy_freq=1)
```

### Execution Commands

**Scenario 0 (Low Traffic):**

```bash
python scripts/train_ddpg.py \
  --scenario 0 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/ddpg_scenario_0
```

**Scenario 1 (Medium Traffic):**

```bash
python scripts/train_ddpg.py \
  --scenario 1 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/ddpg_scenario_1
```

**Scenario 2 (High Traffic):**

```bash
python scripts/train_ddpg.py \
  --scenario 2 \
  --seed 42 \
  --max-timesteps 1e6 \
  --device cuda \
  --log-dir ./logs/ddpg_scenario_2
```

### Expected Output

Same structure as TD3 but in `logs/ddpg_scenario_X/` directories:

```
logs/ddpg_scenario_0/
├── checkpoints/
│   └── ddpg_checkpoint_1000000.pth
├── tensorboard/
└── training_log.json
```

### Expected Training Time

- Same as TD3: ~140-160 GPU hours total
- Can run in parallel with TD3 if GPU memory allows (requires 2x6GB VRAM)

---

## Task #11: Evaluate IDM+MOBIL Baseline

### Objective

Run evaluation on classical IDM+MOBIL baseline to establish performance benchmark for comparison.

### Configuration

```yaml
Algorithm: IDM+MOBIL (deterministic, no learning)
Episodes: 20 per scenario
Metrics: Safety, Efficiency, Comfort
v_desired: 25 m/s (paper value)
a_max: 1.5 m/s²
b: 2.0 m/s²
```

### Execution Commands

**Scenario 0 (Low Traffic):**

```bash
python scripts/evaluate.py \
  --scenario 0 \
  --agent idm \
  --num-episodes 20 \
  --output ./results/idm_scenario_0.csv
```

**Scenario 1 (Medium Traffic):**

```bash
python scripts/evaluate.py \
  --scenario 1 \
  --agent idm \
  --num-episodes 20 \
  --output ./results/idm_scenario_1.csv
```

**Scenario 2 (High Traffic):**

```bash
python scripts/evaluate.py \
  --scenario 2 \
  --agent idm \
  --num-episodes 20 \
  --output ./results/idm_scenario_2.csv
```

### Expected Output

```
results/idm_scenario_0.csv
Sample columns:
Episode,Success,CollisionsCount,AvgSpeed_kmh,JerkLongitudinal_ms3,JerkLateral_ms3,RouteCompletionTime_s,OffRoadCount,LaneChangeCount
1,1,0,27.4,0.73,0.31,145.2,0,12
2,1,0,27.6,0.71,0.28,144.8,0,11
...
20,1,0,27.5,0.75,0.30,145.0,0,12
```

### Expected Results (from Paper)

```
Scenario 0-2 (All):
- Success Rate: 100% (all 20 episodes)
- Average Speed: 27.5 km/h
- Jerk (Longitudinal): 0.75 m/s³
- Jerk (Lateral): 0.30 m/s³
- Collisions: 0
- Time to Complete: ~145 seconds
```

### Expected Evaluation Time

- **Total**: ~10 hours (quick baseline since no learning)
- Can run while TD3/DDPG still training

---

## Quality Checks During Training

### Monitor Every 2-4 Hours

1. **GPU Utilization**
   ```bash
   nvidia-smi
   ```
   Expected: ~90-100% GPU usage, <5GB memory free

2. **Training Progress**
   ```bash
   # Check latest checkpoint timestamp
   ls -lh ./logs/td3_scenario_0/checkpoints/ | tail -5

   # Should see new checkpoints every 5-10 minutes
   ```

3. **TensorBoard Monitoring**
   - Open browser to `http://localhost:6006`
   - Check learning curves are trending downward
   - Verify loss values are stable and decreasing

4. **Disk Space**
   ```bash
   df -h
   # Ensure >10GB free space remains
   ```

5. **Log Files**
   ```bash
   # Check for errors
   tail -100 ./logs/td3_scenario_0/training_log.json
   grep -i "error\|exception" *.log
   ```

---

## Handling Training Interruptions

### If Training Crashes

1. **Restart from Last Checkpoint:**
   ```bash
   python scripts/train_td3.py \
     --scenario 0 \
     --seed 42 \
     --max-timesteps 1e6 \
     --device cuda \
     --resume-from ./logs/td3_scenario_0/checkpoints/td3_checkpoint_XXXXX.pth
   ```

2. **Check Checkpoint Integrity:**
   ```bash
   python -c "
   import torch
   ckpt = torch.load('./logs/td3_scenario_0/checkpoints/td3_checkpoint_XXXXX.pth')
   print('Keys:', list(ckpt.keys()))
   print('Actor params:', ckpt['actor'].keys())
   "
   ```

3. **Verify Disk Space:**
   ```bash
   # Clear old logs if needed
   du -sh ./logs
   rm -rf ./logs/*/tensorboard  # Can recreate later
   ```

---

## Pre-Phase 4 Verification

After all training completes, verify artifacts:

```bash
# 1. Check all checkpoints exist
ls -lh ./logs/td3_scenario_*/checkpoints/td3_checkpoint_1000000.pth
ls -lh ./logs/ddpg_scenario_*/checkpoints/ddpg_checkpoint_1000000.pth

# 2. Verify checkpoint sizes (should be ~20MB each)
du -sh ./logs/*/checkpoints/

# 3. Verify evaluation results
ls -lh ./results/idm_scenario_*.csv

# 4. Verify logs and TensorBoard events
find ./logs -name "events.out.tfevents.*" | wc -l

# 5. Test loading a checkpoint
python scripts/test_checkpoint_load.py \
  ./logs/td3_scenario_0/checkpoints/td3_checkpoint_1000000.pth
```

---

## Parallel Execution Strategy

### Option 1: Sequential (Safest)
- Day 1-7: TD3 all scenarios (160 GPU hours)
- Day 8-14: DDPG all scenarios (160 GPU hours)
- Day 15: IDM baseline (10 hours)
- **Total**: ~25 days

### Option 2: Parallel on Dual GPUs (if available)
- Day 1-7: TD3 on GPU0 + DDPG on GPU1 (both 160 hours in parallel)
- Day 8: IDM baseline
- **Total**: ~8 days

### Option 3: Parallel within GPU memory (current setup)
- Monitor GPU memory during training
- If <4GB free, training is stabilized
- Can typically do one scenario at a time safely

---

## Expected Outcomes

**After Phase 3 Completion:**

```
av_td3_system/
├── logs/
│   ├── td3_scenario_0/
│   │   ├── checkpoints/     (100 checkpoint files)
│   │   ├── tensorboard/
│   │   └── training_log.json
│   ├── td3_scenario_1/      (same structure)
│   ├── td3_scenario_2/      (same structure)
│   ├── ddpg_scenario_0/     (same structure)
│   ├── ddpg_scenario_1/     (same structure)
│   └── ddpg_scenario_2/     (same structure)
├── results/
│   ├── idm_scenario_0.csv
│   ├── idm_scenario_1.csv
│   └── idm_scenario_2.csv
└── data/
    └── training_summary.json
```

**Key Deliverables:**

- ✅ 6 trained agent checkpoints (TD3 x3 + DDPG x3)
- ✅ 6 training logs with learning curves (TensorBoard)
- ✅ 3 baseline evaluation results
- ✅ ~2.5 GB of checkpoint data
- ✅ Ready for Phase 4 evaluation and analysis

---

## Notes for Research Reproducibility

1. **Seed**: Always use `--seed 42` for reproducibility
2. **Hardware**: Document GPU used, memory allocated, CPU specs
3. **CARLA Version**: Verify CARLA 0.9.16 used for all training
4. **Environment**: Save exact package versions with `pip freeze > requirements_phase3.txt`
5. **Random Seeds**: Set everywhere:
   - Python: `random.seed(42)`
   - NumPy: `np.random.seed(42)`
   - PyTorch: `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`
6. **Determinism**: CARLA should be in synchronous mode (no race conditions)

---

## Next: Phase 4 After Training Completes

Once all training completes:

1. **Task #12**: Comprehensive evaluation on final checkpoints
2. **Task #13**: Statistical significance testing
3. **Task #14**: Generate publication-quality plots
4. **Task #15**: Qualitative analysis

See `Phase 4 Analysis Guide` for detailed instructions.

---

**Phase 3 Status**: 🚀 Ready to Launch Training
**Estimated Completion**: Within 25 days
**Next Checkpoint**: Phase 4 evaluation results available
