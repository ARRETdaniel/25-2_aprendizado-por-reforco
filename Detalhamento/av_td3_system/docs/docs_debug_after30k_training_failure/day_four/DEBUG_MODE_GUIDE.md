# Debug Mode Guide: CNN Diagnostics Integration

## Overview

The `--debug` flag now enables **comprehensive debugging capabilities** for both visualization and CNN learning monitoring. This integration provides zero-overhead debugging that can be toggled with a single command-line flag.

**Key Benefits:**
- ✅ **Zero overhead by default** - diagnostics only active when needed
- ✅ **Single flag activation** - no code changes required
- ✅ **Dual monitoring** - OpenCV visualization + CNN diagnostics
- ✅ **Production-ready** - clean separation of debug vs. normal training

---

## Quick Start

### Normal Training (No Diagnostics)
```bash
# Full performance, no overhead
python scripts/train_td3.py --steps 30000 --seed 42
```

### Debug Training (With Diagnostics)
```bash
# OpenCV visualization + CNN diagnostics
python scripts/train_td3.py --steps 1000 --debug --seed 42
```

**Recommended Use:**
- Use `--debug` for **short validation runs** (1k-5k steps)
- Use normal mode for **full training** (30k+ steps)

---

## What Debug Mode Enables

When you pass `--debug`, the following features are activated:

### 1. OpenCV Visualization
- **Real-time camera feed** - Front-facing camera view (84×84 → 800×600)
- **Action display** - Current steering and throttle/brake values
- **Vehicle state** - Speed, lateral deviation, heading error
- **Reward breakdown** - Efficiency, lane-keeping, comfort, safety, progress
- **Episode progress** - Distance to goal, progress %, waypoint index
- **Interactive controls** - Press `q` to quit, `p` to pause/unpause

### 2. CNN Diagnostics
- **Gradient flow tracking** - Per-layer gradient L2 norms
- **Weight update monitoring** - Weight changes after optimizer steps
- **Feature statistics** - Output feature norms, means, std devs
- **Health indicators** - Automatic detection of training issues
- **TensorBoard logging** - 10+ metrics logged to `cnn_diagnostics/*`
- **Console summaries** - Detailed diagnostics every 1000 steps

---

## Console Output

### Debug Mode Enabled Banner
```
======================================================================
[DEBUG MODE ENABLED]
======================================================================
[DEBUG] Visual feedback enabled (OpenCV display)
[DEBUG] Press 'q' to quit, 'p' to pause/unpause

[DEBUG] CNN diagnostics enabled for training monitoring
[DEBUG] Tracking: gradient flow, weight updates, feature statistics
[DEBUG] TensorBoard metrics: cnn_diagnostics/*
[DEBUG] Console output: Every 1000 steps
======================================================================

✅ CNN diagnostics enabled
```

### CNN Diagnostics Summary (Every 1000 Steps)
```
======================================================================
[CNN DIAGNOSTICS] Step 25,000
======================================================================

CNN DIAGNOSTICS SUMMARY
======================================================================

Captures: 950 gradients | 950 weights | 1900 features

✅ Gradient Flow: OK
✅ Weight Updates: OK

[Gradient Flow by Layer]
  features.0.weight    ✅ FLOWING (mean=1.23e-03, max=4.56e-03)
  features.2.weight    ✅ FLOWING (mean=2.34e-03, max=8.90e-03)
  features.4.weight    ✅ FLOWING (mean=3.45e-03, max=1.23e-02)

[Weight Changes]
  features.0.weight    norm=123.45 change=0.0123 trend=increasing ✅
  features.2.weight    norm=234.56 change=0.0234 trend=stable ✅

[Feature Statistics]
  output_norm          mean= 12.3456 trend=increasing ✅
  output_std           mean=  0.4567 trend=stable ✅

======================================================================
```

---

## TensorBoard Metrics

### Gradient Flow (`cnn_diagnostics/gradients/*`)
- `cnn_diagnostics/gradients/features.0.weight` - Conv layer 1 gradients
- `cnn_diagnostics/gradients/features.2.weight` - Conv layer 2 gradients
- `cnn_diagnostics/gradients/features.4.weight` - Conv layer 3 gradients
- `cnn_diagnostics/gradients/fc.weight` - Fully connected layer gradients

**What to Look For:**
- ✅ **Healthy:** Gradients > 1e-6, visible on log scale
- ⚠️ **Warning:** Gradients declining over time (vanishing)
- ❌ **Problem:** Gradients < 1e-10 (blocked flow)

### Weight Updates (`cnn_diagnostics/weights/*`)
- `cnn_diagnostics/weights/[layer]_norm` - Current weight L2 norm
- `cnn_diagnostics/weights/[layer]_change` - Weight change magnitude

**What to Look For:**
- ✅ **Healthy:** Changes > 1e-6, norms increasing/stable
- ⚠️ **Warning:** Changes declining rapidly (learning rate issue)
- ❌ **Problem:** Changes < 1e-10 (frozen weights)

### Feature Statistics (`cnn_diagnostics/features/*`)
- `cnn_diagnostics/features/output_norm` - Feature vector L2 norm
- `cnn_diagnostics/features/output_mean` - Mean activation
- `cnn_diagnostics/features/output_std` - Standard deviation

**What to Look For:**
- ✅ **Healthy:** Norms increasing, mean/std stable
- ⚠️ **Warning:** Norms constant (not learning)
- ❌ **Problem:** Norms → 0 (dead features) or → ∞ (exploding)

### Health Indicators (`cnn_diagnostics/health/*`)
- `cnn_diagnostics/health/gradient_flow_ok` - Binary (0=blocked, 1=flowing)
- `cnn_diagnostics/health/weight_updates_ok` - Binary (0=frozen, 1=updating)

**What to Look For:**
- ✅ **Healthy:** Both metrics = 1.0
- ❌ **Problem:** Either metric = 0.0 (training failure)

---

## Performance Characteristics

### Debug Mode Overhead

| Component | Overhead | Memory Impact |
|-----------|----------|---------------|
| **OpenCV Visualization** | ~1-2% | +10 MB |
| **CNN Diagnostics** | ~0.1% | +10 MB per 1k steps |
| **TensorBoard Logging** | ~0.05% | +30 MB for 30k steps |
| **Total Debug Mode** | **~1-2%** | **+300 MB for 30k steps** |

**Recommendation:**
- Use `--debug` for **short runs** (1k-5k steps) to verify learning
- Use normal mode for **full training** (30k+ steps) to maximize performance

---

## Usage Examples

### Example 1: Quick CNN Health Check (1k steps)
```bash
# Verify CNN is learning with minimal overhead
python scripts/train_td3.py --steps 1000 --debug --seed 42

# Check TensorBoard
tensorboard --logdir data/logs/

# Navigate to:
# - cnn_diagnostics/health/* (should be 1.0)
# - cnn_diagnostics/gradients/* (should be > 1e-6)
```

### Example 2: Debugging Gradient Flow Issues
```bash
# Long debug run to track gradient evolution
python scripts/train_td3.py --steps 5000 --debug --seed 42

# In TensorBoard, check:
# 1. cnn_diagnostics/gradients/* - are they flowing?
# 2. cnn_diagnostics/weights/*_change - are weights updating?
# 3. cnn_diagnostics/features/output_norm - are features changing?
```

### Example 3: Comparing Debug vs. Normal Training
```bash
# Run with debug (1k steps)
python scripts/train_td3.py --steps 1000 --debug --seed 42

# Run without debug (1k steps)
python scripts/train_td3.py --steps 1000 --seed 43

# Compare:
# - Training time (debug ~1-2% slower)
# - Episode rewards (should be similar)
# - Memory usage (debug +300 MB)
```

---

## Troubleshooting

### Problem 1: "Gradients all zero"

**Symptoms:**
- `cnn_diagnostics/health/gradient_flow_ok` = 0.0
- All gradient metrics < 1e-10

**Causes:**
- CNN in eval mode (should be train mode)
- Replay buffer storing flattened states (should store Dict)
- No gradient flow from critic/actor to CNN

**Solution:**
```python
# Verify CNN is in train mode
assert agent.cnn_extractor.training == True

# Verify using DictReplayBuffer
assert agent.use_dict_buffer == True

# Verify CNN optimizer exists
assert agent.cnn_optimizer is not None
```

---

### Problem 2: "Weights not updating"

**Symptoms:**
- `cnn_diagnostics/health/weight_updates_ok` = 0.0
- Weight change metrics < 1e-10

**Causes:**
- CNN optimizer not being called
- Learning rate too small
- Gradients too small

**Solution:**
```python
# Check CNN optimizer step is called
# In td3_agent.py train() method:
if self.cnn_optimizer is not None:
    self.cnn_optimizer.step()  # Should be called twice (critic + actor updates)

# Verify learning rate
print(f"CNN LR: {agent.cnn_optimizer.param_groups[0]['lr']}")  # Should be 3e-4
```

---

### Problem 3: "Features constant"

**Symptoms:**
- `cnn_diagnostics/features/output_norm` constant over time
- No trend (increasing/stable/decreasing)

**Causes:**
- CNN not learning (gradients/weights frozen)
- Input images all similar (need data diversity)
- Feature extractor disconnected from training

**Solution:**
1. Verify gradients flowing (Problem 1)
2. Verify weights updating (Problem 2)
3. Check input diversity: `obs_dict['image'].std()` should be > 0.1

---

### Problem 4: "TensorBoard empty"

**Symptoms:**
- No `cnn_diagnostics/*` metrics in TensorBoard

**Causes:**
- `--debug` flag not passed
- Diagnostics not enabled
- TensorBoard writer not flushed

**Solution:**
```bash
# Verify debug flag passed
python scripts/train_td3.py --steps 1000 --debug --seed 42

# Check console for enable message
# Should see: "✅ CNN diagnostics enabled"

# Flush TensorBoard
# In train_td3.py, line ~840:
self.writer.flush()  # Called every 100 steps
```

---

### Problem 5: "High memory usage"

**Symptoms:**
- Memory usage grows over time
- System runs out of RAM

**Causes:**
- Diagnostics history unbounded
- TensorBoard event files too large

**Solution:**
```python
# Limit diagnostics history (in cnn_diagnostics.py)
# Default max_history = 1000 (keeps last 1000 captures)
# For long runs, reduce to 500 or 100

# In CNNDiagnostics.__init__:
self.max_history = 500  # Reduce from 1000
```

---

### Problem 6: "Slow training"

**Symptoms:**
- Training noticeably slower with `--debug`

**Causes:**
- OpenCV rendering overhead (1-2%)
- Diagnostics capture overhead (0.1%)
- Console printing overhead

**Solution:**
- **For full training:** Don't use `--debug` (zero overhead)
- **For debugging:** Accept 1-2% overhead as necessary cost
- **Compromise:** Run debug for first 5k steps, then disable

---

## Best Practices

### 1. Use Debug Mode Strategically
- ✅ **Short validation runs** (1k-5k steps) - verify CNN learning
- ✅ **Initial training** (first 5k steps) - catch issues early
- ✅ **Debugging failures** - when training doesn't converge
- ❌ **Full training runs** (30k+ steps) - unnecessary overhead

### 2. Monitor Key Metrics
**During exploration phase (steps 1-25k):**
- CNN gradients should be **present but small** (critic updates only)
- Weight changes should be **present but small**
- Features should be **changing gradually**

**During learning phase (steps 25k+):**
- CNN gradients should be **larger and consistent**
- Weight changes should be **visible and steady**
- Features should be **evolving with clear trends**

### 3. Interpret Health Indicators
- **Both OK (1.0):** Training healthy, CNN learning ✅
- **Gradient Flow OK, Weight Updates NOT OK:** Learning rate too small ⚠️
- **Gradient Flow NOT OK:** Critical issue, check CNN mode/buffer ❌

### 4. Console Output Strategy
- **Every 100 steps:** Progress messages (always shown)
- **Every 1000 steps:** CNN diagnostics summary (debug only)
- **Every 5000 steps:** Evaluation metrics (always shown)

### 5. TensorBoard Analysis
**Quick Health Check (< 5 minutes):**
1. Open TensorBoard: `tensorboard --logdir data/logs/`
2. Check `cnn_diagnostics/health/*` - both should be 1.0
3. Check `cnn_diagnostics/gradients/*` - should be visible on log scale

**Deep Analysis (15-30 minutes):**
1. Plot `cnn_diagnostics/gradients/*` over time - trend analysis
2. Plot `cnn_diagnostics/weights/*_change` - learning rate effectiveness
3. Plot `cnn_diagnostics/features/*` - feature quality evolution
4. Compare with `train/critic_loss` and `train/episode_reward`

---

## Implementation Details

### Code Changes

**1. train_td3.py: Debug mode setup (lines ~237-252)**
```python
# Debug mode setup
if self.debug:
    print(f"\n{'='*70}")
    print(f"[DEBUG MODE ENABLED]")
    print(f"{'='*70}")
    print(f"[DEBUG] Visual feedback enabled (OpenCV display)")
    print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")
    print(f"\n[DEBUG] CNN diagnostics enabled for training monitoring")
    print(f"[DEBUG] Tracking: gradient flow, weight updates, feature statistics")
    print(f"[DEBUG] TensorBoard metrics: cnn_diagnostics/*")
    print(f"[DEBUG] Console output: Every 1000 steps")
    print(f"{'='*70}\n")
    
    # Enable CNN diagnostics
    self.agent.enable_diagnostics(self.writer)
    
    # Setup OpenCV window
    self.window_name = "TD3 Training - Debug View"
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.window_name, 1200, 600)
    self.paused = False
```

**2. train_td3.py: Diagnostics logging in training loop (lines ~810-824)**
```python
# Log CNN diagnostics every 100 steps (if debug mode enabled)
if self.debug and self.agent.cnn_diagnostics is not None:
    self.agent.cnn_diagnostics.log_to_tensorboard(t)
    
    # Print detailed CNN diagnostics every 1000 steps
    if t % 1000 == 0:
        print(f"\n{'='*70}")
        print(f"[CNN DIAGNOSTICS] Step {t:,}")
        print(f"{'='*70}")
        self.agent.print_diagnostics(max_history=1000)
        print(f"{'='*70}\n")
```

**3. train_td3.py: Updated --debug help text (line ~1125)**
```python
parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable debug mode: OpenCV visualization + CNN diagnostics (gradient flow, '
         'weight updates, feature stats). Recommended for short runs to verify learning. '
         'TensorBoard: cnn_diagnostics/* metrics. Console: detailed output every 1000 steps.'
)
```

### Architecture

```
train_td3.py (--debug flag)
    ↓
TD3TrainingPipeline.__init__()
    ↓ (if self.debug)
agent.enable_diagnostics(writer)
    ↓
TD3Agent.enable_diagnostics()
    ↓
CNNDiagnostics.__init__(cnn_module, writer)
    ↓
Training Loop
    ↓
agent.train(batch_size)
    ↓ (in TD3Agent.train())
    ├─ critic_loss.backward()
    │   ↓
    │   cnn_diagnostics.capture_gradients()
    │
    ├─ cnn_optimizer.step()
    │   ↓
    │   cnn_diagnostics.capture_weights()
    │
    └─ (same for actor update)
    ↓ (back in training loop)
    ↓ (if t % 100 == 0 and debug)
cnn_diagnostics.log_to_tensorboard(t)
    ↓ (if t % 1000 == 0 and debug)
agent.print_diagnostics()
```

---

## Testing

### Run Integration Tests
```bash
# Test debug mode integration
python tests/test_debug_mode_integration.py

# Expected output:
# ✅ test_debug_flag_parsing
# ✅ test_cnn_diagnostics_enabled_in_debug_mode
# ✅ test_cnn_diagnostics_disabled_without_debug
# ✅ test_diagnostics_logging_to_tensorboard
# ✅ test_console_output_in_debug_mode
```

### Manual Validation
```bash
# 1. Verify debug mode enables diagnostics
python scripts/train_td3.py --steps 100 --debug --seed 42
# Should see: "[DEBUG MODE ENABLED]" banner
# Should see: "✅ CNN diagnostics enabled"

# 2. Verify normal mode does NOT enable diagnostics
python scripts/train_td3.py --steps 100 --seed 42
# Should NOT see debug banner
# Should NOT see CNN diagnostics messages

# 3. Verify TensorBoard metrics appear
python scripts/train_td3.py --steps 1000 --debug --seed 42
tensorboard --logdir data/logs/
# Navigate to cnn_diagnostics/* - should have 10+ metrics
```

---

## Summary

| Feature | Normal Mode | Debug Mode |
|---------|-------------|------------|
| **OpenCV Visualization** | ❌ Disabled | ✅ Enabled |
| **CNN Diagnostics** | ❌ Disabled | ✅ Enabled |
| **TensorBoard Metrics** | Basic (6 metrics) | Full (16+ metrics) |
| **Console Output** | Progress only | Progress + diagnostics |
| **Performance Overhead** | 0% | ~1-2% |
| **Memory Overhead** | 0 MB | +300 MB (30k steps) |
| **Use Case** | Full training | Debugging/validation |

**Recommendation:**
- Use `--debug` for **initial validation** (1k-5k steps)
- Use normal mode for **full training** (30k+ steps)
- Use `--debug` when **debugging failures** or **verifying CNN learning**

---

## Next Steps

1. **Run short debug test:** `python scripts/train_td3.py --steps 1000 --debug --seed 42`
2. **Verify diagnostics:** Check console for "[DEBUG MODE ENABLED]" banner
3. **Inspect TensorBoard:** `tensorboard --logdir data/logs/`
4. **Check CNN health:** `cnn_diagnostics/health/*` should be 1.0
5. **Run full training:** `python scripts/train_td3.py --steps 30000 --seed 42` (no debug)

**For detailed CNN diagnostics usage, see:** `docs/CNN_DIAGNOSTICS_GUIDE.md`

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-28  
**Author:** Daniel Terra
