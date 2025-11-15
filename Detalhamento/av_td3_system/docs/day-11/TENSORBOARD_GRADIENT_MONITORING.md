# TensorBoard Gradient Explosion Monitoring Guide

**Date**: November 12, 2025
**Purpose**: Real-time monitoring of gradient norms to detect gradient explosion
**Related**: GRADIENT_EXPLOSION_FIX.md (Solution A implementation)

---

## Quick Start

### 1. Launch TensorBoard

While training is running (or after), start TensorBoard:

```bash
# From the av_td3_system directory
tensorboard --logdir data/logs --port 6006
```

Then open your browser to: `http://localhost:6006`

---

### 2. Key Metrics to Monitor

Navigate to the **SCALARS** tab in TensorBoard. The following metrics are automatically logged:

#### 🚨 **Critical Metrics** (Gradient Explosion Detection)

| Metric Path | Description | Target Range | Alert Threshold |
|-------------|-------------|--------------|-----------------|
| `gradients/actor_cnn_norm` | Actor CNN gradient magnitude | **< 10,000** | > 50,000 (critical) |
| `gradients/critic_cnn_norm` | Critic CNN gradient magnitude | 200-2,000 | > 5,000 (warning) |
| `alerts/gradient_explosion_critical` | Binary alert (0 or 1) | **0** | 1 (immediate action) |
| `alerts/gradient_explosion_warning` | Binary alert (0 or 1) | **0** | 1 (monitor closely) |

#### 📊 **Supporting Metrics** (Context)

| Metric Path | Description | Expected Range |
|-------------|-------------|----------------|
| `gradients/actor_mlp_norm` | Actor MLP head gradient | 1,000-10,000 |
| `gradients/critic_mlp_norm` | Critic MLP head gradient | 1,000-10,000 |
| `train/q1_value` | Q-value magnitude | -1,000 to 1,000,000 |
| `train/actor_loss` | Actor loss (negative Q-value) | -1,000,000 to 0 |

---

## Monitoring During Training

### Real-Time Monitoring

**TensorBoard automatically updates** every 30 seconds. You'll see:

1. **Smooth line graphs** showing gradient norms over training steps
2. **Alert spikes** in the alerts metrics (0 → 1 transitions)
3. **Comparative trends** between actor CNN and critic CNN

### Visual Indicators

#### ✅ **Healthy Training** (Solution A Working)

```
Actor CNN Gradient Norm:
    Step 100:  2,000 ────▂▂▂▂▂▂▂▂▂▂▂▂▂─────
    Step 200:  3,500 ────▃▃▃▃▃▃▃▃▃▃▃▃▃─────
    Step 300:  4,200 ────▄▄▄▄▄▄▄▄▄▄▄▄▄─────
    Step 400:  5,100 ────▅▅▅▅▅▅▅▅▅▅▅▅▅─────
    Step 500:  6,800 ────▆▆▆▆▆▆▆▆▆▆▆▆▆─────

Pattern: Linear or slight growth, staying < 10,000
```

#### ⚠️ **Warning Signs** (Elevated but Not Critical)

```
Actor CNN Gradient Norm:
    Step 100:  2,000 ────▂─────────────────
    Step 200: 12,000 ────▇▇▇───────────────
    Step 300: 18,000 ────███───────────────
    Step 400: 25,000 ────███▇──────────────
    Step 500: 32,000 ────████──────────────

Pattern: Growing but < 50,000 threshold
Action: Monitor closely, prepare to implement gradient clipping
```

#### 🔴 **Critical Explosion** (Solution A Failed)

```
Actor CNN Gradient Norm:
    Step 100:   5,191 ▂──────────────────────────
    Step 200: 130,486 ████──────────────────────
    Step 300: 826,256 ████████████──────────────
    Step 400: 2,860,755 ████████████████████────
    Step 500: 7,475,702 ████████████████████████

Pattern: Exponential growth (1,440x increase)
Action: STOP TRAINING, implement Solution B (gradient clipping)
```

---

## TensorBoard Dashboard Setup

### Recommended View Configuration

1. **Create custom dashboard** for gradient monitoring:
   - Click gear icon (⚙️) → "Settings"
   - Enable "Reload data" (30s interval)
   - Smoothing: 0 (raw values to see spikes)

2. **Pin critical metrics** to the top:
   - `gradients/actor_cnn_norm` (primary concern)
   - `alerts/gradient_explosion_critical`
   - `train/q1_value` (Q-value context)

3. **Use logarithmic scale** for gradient plots:
   - Click on the Y-axis label → "Log scale"
   - This makes exponential growth patterns more visible

---

## Alert System

### Console Alerts (During Training)

When gradients exceed thresholds, you'll see console output:

#### Warning Alert (> 10,000)

```
⚠️  WARNING: Actor CNN gradient elevated at step 1,200: 15,432.67
```

#### Critical Alert (> 50,000)

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
🔴 CRITICAL ALERT: Actor CNN gradient explosion detected!
   Step: 1,500
   Actor CNN grad norm: 125,678.90
   Threshold: 50,000
   Recommendation: Stop training, implement Solution B (gradient clipping)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### TensorBoard Alerts

**Binary alerts appear as step functions**:

```
alerts/gradient_explosion_critical:

0 ─────────────┐
               │
1              └──────────────────

   Step 1,500: Alert triggered (gradient > 50,000)
```

---

## Comparing Runs

### Validation Run #2 vs Run #3

Use TensorBoard's **compare runs** feature:

1. Select multiple event files in the left sidebar
2. Enable "Show data download links"
3. Compare gradient norms side-by-side

**Expected Comparison**:

| Metric | Run #2 (No Fix) | Run #3 (Solution A) | Improvement |
|--------|-----------------|---------------------|-------------|
| Actor CNN grad @ step 100 | 5,191 | ~2,000 | ✅ 2.6x lower |
| Actor CNN grad @ step 500 | 7,475,702 | **< 10,000** | ✅ 747x lower |
| Growth pattern | Exponential (1440x) | Linear/stable | ✅ Explosion prevented |
| Training completion | ⚠️ Completed (risky) | ✅ Stable | ✅ Safe for 1M run |

---

## Detailed Metrics Guide

### 1. Actor CNN Gradient Norm (`gradients/actor_cnn_norm`)

**What it measures**: L2 norm of gradients flowing through Actor CNN layers

**Calculation**:
```python
actor_cnn_grad_norm = sum(
    p.grad.norm().item() for p in self.actor_cnn.parameters() if p.grad is not None
)
```

**Healthy range**: **< 10,000**

**Warning signs**:
- Exponential growth pattern (each 100 steps showing 5x+ increase)
- Sudden spikes (> 3x previous value)
- Sustained levels above 10,000

**Why it matters**: Actor CNN gradients exploded from 5,191 → 7,475,702 in Run #2. This metric is the **primary indicator** of gradient explosion and the main target of Solution A.

---

### 2. Critic CNN Gradient Norm (`gradients/critic_cnn_norm`)

**What it measures**: L2 norm of gradients flowing through Critic CNN layers

**Healthy range**: **200-2,000**

**Comparison**: Critic CNN was **stable** in Run #2 (233-1,256 range) while Actor CNN exploded. This suggests the issue is specific to policy gradients, not value gradients.

**Why it matters**: Provides a **baseline** to compare actor vs critic gradient behavior. If critic is stable but actor explodes, it confirms the policy learning rate issue.

---

### 3. Gradient Explosion Alerts

#### `alerts/gradient_explosion_critical` (Binary: 0 or 1)

**Triggered when**: `actor_cnn_grad_norm > 50,000`

**Action required**:
1. Stop training immediately
2. Implement Solution B (gradient clipping)
3. Consider Solution C (Q-value normalization)
4. Document failure in validation report

#### `alerts/gradient_explosion_warning` (Binary: 0 or 1)

**Triggered when**: `10,000 < actor_cnn_grad_norm ≤ 50,000`

**Action required**:
1. Continue training but monitor closely
2. Prepare gradient clipping implementation (Solution B)
3. Check Q-value magnitude (`train/q1_value`)
4. Consider reducing actor CNN learning rate further (e.g., 1e-5 → 5e-6)

---

### 4. Q-Value Magnitude (`train/q1_value`)

**What it measures**: Mean Q-value predicted by Critic Q1 network

**Healthy range**: -1,000 to 1,000,000

**Connection to gradients**: Actor loss = -mean(Q(s, μ(s))). High Q-values (11M in Run #2) amplify actor gradients, causing explosion.

**Expected behavior**: With Solution A (slower actor CNN learning), actor will "pull down" Q-values more gradually, preventing explosion.

---

## Troubleshooting

### Issue 1: TensorBoard not showing gradients/actor_cnn_norm

**Cause**: Debug mode not enabled or training hasn't entered learning phase

**Solution**:
```bash
# Ensure --debug flag is set
python3 scripts/train_td3.py --debug --max-timesteps 1000

# Check training has passed learning_starts
grep "PHASE TRANSITION" validation_1k_3.log
```

---

### Issue 2: Alerts not appearing in TensorBoard

**Cause**: Bug fix not applied (writer parameter missing)

**Solution**:
```bash
# Verify fix applied
grep "self.writer.add_scalar('alerts" scripts/train_td3.py

# Should show:
#   self.writer.add_scalar('alerts/gradient_explosion_critical', ...
#   self.writer.add_scalar('alerts/gradient_explosion_warning', ...
```

---

### Issue 3: Gradient norms show as NaN

**Cause**: Gradient explosion already occurred, network parameters corrupted

**Solution**:
1. Stop training immediately
2. Revert to last checkpoint before explosion
3. Implement gradient clipping (Solution B)
4. Reduce all learning rates by 10x

---

## Advanced: Custom TensorBoard Plugins

### Export Data for Analysis

**Download gradient history**:

1. In TensorBoard, click on metric → "Show data download links"
2. Download CSV for `gradients/actor_cnn_norm`
3. Analyze in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('gradients_actor_cnn_norm.csv')

# Check for exponential growth
df['growth_factor'] = df['Value'].pct_change() + 1
is_exponential = (df['growth_factor'] > 5).sum() > 3

if is_exponential:
    print("⚠️ EXPONENTIAL GROWTH DETECTED!")
else:
    print("✅ Gradient growth is stable")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Step'], df['Value'], label='Actor CNN Grad Norm')
plt.axhline(y=10000, color='orange', linestyle='--', label='Warning threshold')
plt.axhline(y=50000, color='red', linestyle='--', label='Critical threshold')
plt.yscale('log')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.legend()
plt.title('Actor CNN Gradient Monitoring')
plt.savefig('gradient_monitoring.png')
```

---

## Success Criteria for 1K Test #3

### Primary Metrics ✅

1. **Actor CNN grad norm < 10,000** throughout 1K steps
2. **No critical alerts** (alerts/gradient_explosion_critical = 0)
3. **Growth factor < 2x** per 100 steps (no exponential pattern)
4. **Training completes** without NaN/Inf

### Comparison to Run #2 ✅

| Step | Run #2 (No Fix) | Run #3 (Target) | Improvement |
|------|-----------------|-----------------|-------------|
| 100  | 5,191 | < 5,000 | ≥ 1x better |
| 200  | 130,486 | < 10,000 | ≥ 13x better |
| 300  | 826,256 | < 10,000 | ≥ 82x better |
| 400  | 2,860,755 | < 10,000 | ≥ 286x better |
| 500  | 7,475,702 | **< 10,000** | **≥ 747x better** |

### Visual Confirmation ✅

**Expected TensorBoard plot**:

```
10,000 ┤──────────────────────────────────────────────
       │                                     ╭────────
       │                                ╭────╯
 5,000 ┤                           ╭────╯
       │                      ╭────╯
       │                 ╭────╯
 2,000 ┤            ╭────╯
       │       ╭────╯
       │  ╭────╯
     0 ┼──╯──────────────────────────────────────────
       0   100   200   300   400   500   600   700   800

Pattern: Linear or slight curve, NO exponential explosion
```

---

## Monitoring Checklist for 1K Test #3

**Before training**:
- [ ] Launch TensorBoard: `tensorboard --logdir data/logs --port 6006`
- [ ] Open browser to `http://localhost:6006`
- [ ] Verify previous run data visible (Run #2 baseline)
- [ ] Enable "Reload data" in TensorBoard settings

**During training** (check every 5 minutes):
- [ ] `gradients/actor_cnn_norm` staying < 10,000
- [ ] No alert spikes in `alerts/gradient_explosion_critical`
- [ ] Console shows no 🔴 CRITICAL alerts
- [ ] Training progressing normally (step count increasing)

**After training**:
- [ ] Export gradient data to CSV
- [ ] Compare Run #2 vs Run #3 side-by-side
- [ ] Calculate growth factors (should be < 2x per 100 steps)
- [ ] Document results in VALIDATION_1K_RUN3_RESULTS.md
- [ ] Approve or reject Solution A based on data

---

## References

1. **GRADIENT_EXPLOSION_FIX.md**: Full technical analysis of gradient explosion
2. **CHANGELOG_GRADIENT_FIX.md**: Implementation details of Solution A
3. **VALIDATION_1K_RUN2_ANALYSIS.md**: Baseline data showing gradient explosion
4. **TensorBoard Documentation**: https://www.tensorflow.org/tensorboard
5. **PyTorch Gradient Debugging**: https://pytorch.org/docs/stable/notes/autograd.html

---

**Status**: ✅ Real-time monitoring implemented and ready for 1K Test #3
**Next Action**: Run 1K validation with `--debug` flag to populate TensorBoard
**Expected Duration**: 30 minutes to complete + 10 minutes for analysis
