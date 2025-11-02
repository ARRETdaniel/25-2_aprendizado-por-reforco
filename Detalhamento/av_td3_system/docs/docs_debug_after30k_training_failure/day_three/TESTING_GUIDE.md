# Bug #13 Fix - Testing Instructions

**Date:** 2025-11-01
**Status:** ✅ Phase 3 Complete - Ready for Testing

---

## Quick Start: Run Diagnostic Test

```bash
# Navigate to project
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Start CARLA server (if not running)
docker start carla-server && sleep 10

# Run 1000-step diagnostic test
python scripts/train_td3.py --scenario 0 --max-timesteps 1000 --debug --device cpu
```

**Duration:** ~15 minutes
**Expected:** Vehicle moves, rewards vary, CNN weights update

---

## Expected Output

### Initialization (Should See)

```
[AGENT] Initializing NatureCNN feature extractor...
[AGENT] CNN extractor initialized on cpu
[AGENT] CNN architecture: 4×84×84 → Conv layers → 512 features
[AGENT] CNN training mode: ENABLED (weights will be updated during training)
[AGENT] CNN passed to TD3Agent for end-to-end training
[AGENT] DictReplayBuffer enabled for gradient flow

[TD3Agent] Initializing TD3 agent...
[TD3Agent] CNN optimizer initialized with lr=0.0001
[TD3Agent] CNN mode: training (gradients enabled)
[TD3Agent] Using DictReplayBuffer for end-to-end CNN training
[TD3Agent] Expected memory usage: ~400.00 MB for 1000000 transitions
```

### During Training

```
[EXPLORATION] Processing step    100/1,000...
[LEARNING] Processing step    500/1,000...
```

---

## Success Criteria

✅ **Initialization:**
- CNN optimizer created with lr=0.0001
- DictReplayBuffer enabled message appears
- No errors during agent initialization

✅ **During Training:**
- Vehicle speed > 0 km/h (not frozen)
- Episode rewards vary (not constant -53)
- No gradient-related errors
- Replay buffer fills correctly

✅ **After Training:**
- CNN weights changed from initialization
- Training completed without crashes
- TensorBoard logs generated

---

## Troubleshooting

### Issue: "KeyError: 'image'"

**Cause:** Trying to store flattened states in DictReplayBuffer

**Fix:** Verify `replay_buffer.add()` receives `obs_dict` (not `flat_state`)

### Issue: CNN weights not changing

**Debug:**
```python
# Add after first training update
for name, param in agent.cnn_extractor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

**Expected:** All parameters have gradients > 0

### Issue: Memory usage too high

**Fix:** Reduce buffer size in config:
```yaml
training:
  buffer_size: 250000  # Reduced from 1000000
```

---

## Next: Full Training

After diagnostic test passes:

```bash
# Run full 30K-step training
python scripts/train_td3.py --scenario 0 --max-timesteps 30000 --seed 42 --device cpu
```

**Duration:** 2-4 hours
**Expected:** Vehicle speed > 5 km/h, rewards > -30,000, success rate > 5%

---

## Compare Results

| Metric | Old (Bug #13) | Expected (Fixed) |
|--------|---------------|------------------|
| Vehicle Speed | 0 km/h | > 5 km/h |
| Mean Reward | -52,700 | > -30,000 |
| Success Rate | 0% | > 5% |

---

**Ready to test!** 🚀
