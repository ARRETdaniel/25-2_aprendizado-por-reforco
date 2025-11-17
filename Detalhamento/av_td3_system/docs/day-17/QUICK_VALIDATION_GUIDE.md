# QUICK VALIDATION GUIDE - Gradient Clipping Fixes

**Purpose**: Quick reference for validating gradient clipping implementation
**Date**: 2025-11-17
**Estimated Time**: 1 hour (training) + 30 minutes (analysis)

---

## ✅ What Was Changed

1. **Actor Gradient Clipping**: max_norm=1.0 (CRITICAL FIX)
2. **Critic Gradient Clipping**: max_norm=10.0 (stability enhancement)
3. **Actor CNN Learning Rate**: 1e-5 → 1e-4 (10× increase)
4. **Configuration**: Added explicit gradient_clipping parameters

---

## 🚀 Quick Start: Run Validation

```bash
cd /workspace/av_td3_system

# Run 5K validation with all fixes
python scripts/train_td3.py \
  --scenario 0 \
  --seed 42 \
  --max-timesteps 5000 \
  --eval-freq 5000 \
  --checkpoint-freq 5000 \
  --debug
```

---

## 📊 Success Criteria (Check TensorBoard)

### CRITICAL (MUST PASS)

| Metric | Baseline (5K) | Expected (5K Validation) | Status |
|--------|---------------|-------------------------|--------|
| **Actor CNN Grad Norm (mean)** | 1,826,337 | **<1.0** | ⬜ |
| **Actor CNN Grad Norm (max)** | 8,199,994 | **<1.5** | ⬜ |
| **Actor Loss (final)** | -7,607,850 | **<1,000** | ⬜ |
| **Gradient Explosion Alerts** | 22/25 (88%) | **0/25 (0%)** | ⬜ |

### IMPORTANT (SHOULD PASS)

| Metric | Baseline (5K) | Expected (5K Validation) | Status |
|--------|---------------|-------------------------|--------|
| **Critic CNN Grad Norm (mean)** | 5,897 | **<10,000** | ⬜ |
| **Q1 Value (increasing?)** | 20 → 71 ✅ | **Similar trend** | ⬜ |
| **Critic Loss (stable?)** | 121.87 ✅ | **<200** | ⬜ |

---

## 📈 How to Check Results

### Option 1: TensorBoard (Visual)

```bash
tensorboard --logdir data/logs/
# Open browser: http://localhost:6006
```

**Check these plots**:
- `gradients/actor_cnn_norm`: Should be FLAT LINE near 1.0 (not spiking to millions)
- `train/actor_loss`: Should be stable/decreasing (not diverging to -7M)
- `alerts/gradient_explosion_critical`: Should be ZERO events

### Option 2: Parse Logs (Automated)

```bash
# Get latest log directory
LOG_DIR=$(ls -td data/logs/TD3_scenario_0_npcs_20_* | head -1)

# Parse and summarize
python scripts/parse_tensorboard_logs.py \
  --log-dir $LOG_DIR \
  --output-dir docs/day-17/validation_results
```

---

## 🎯 Decision Matrix

### 🟢 GO for 1M Run (ALL CRITICAL pass)

- ✅ Actor CNN grad norm <1.0 mean
- ✅ Actor loss <1,000
- ✅ Zero gradient explosion alerts
- ✅ Q-values still increasing
- ✅ Critic stable

**Action**: Proceed with 1M-step production run (see Section 5.2 in IMPLEMENTATION doc)

---

### 🟡 PARTIAL SUCCESS (Some CRITICAL pass)

- ⚠️ Actor CNN grad norm 1.0-10.0 (clipping working, but needs tuning)
- ⚠️ Actor loss <10,000 (better, but not ideal)

**Action**: Adjust `actor_max_norm` from 1.0 to 5.0, re-run validation

---

### 🔴 NO-GO (CRITICAL failures)

- ❌ Actor CNN grad norm >100 (clipping NOT working)
- ❌ Actor loss still diverging
- ❌ Q-values collapsing

**Action**: Debug gradient clipping implementation (see Appendix B in IMPLEMENTATION doc)

---

## 📝 Quick Checklist

Before starting validation:

- [ ] Code changes committed to git
- [ ] Configuration file updated (td3_config.yaml)
- [ ] CARLA server accessible (localhost:2000)
- [ ] GPU available (check `nvidia-smi`)
- [ ] Disk space available (~5GB for logs)

After validation completes:

- [ ] Check TensorBoard plots (gradients, losses, Q-values)
- [ ] Verify zero gradient explosion alerts
- [ ] Compare with baseline metrics (CRITICAL_TENSORBOARD_ANALYSIS_5K_RUN.md)
- [ ] Document decision (GO/NO-GO) with evidence
- [ ] Update todo list with next steps

---

## 🔗 Full Documentation

For complete details, see:

- **IMPLEMENTATION_GRADIENT_CLIPPING_FIXES.md**: Complete implementation record
- **LITERATURE_VALIDATED_ACTOR_ANALYSIS.md**: Academic validation and analysis
- **CRITICAL_TENSORBOARD_ANALYSIS_5K_RUN.md**: Baseline metrics (BEFORE fixes)

---

**Expected Runtime**: ~1 hour for 5K steps (similar to baseline run)
