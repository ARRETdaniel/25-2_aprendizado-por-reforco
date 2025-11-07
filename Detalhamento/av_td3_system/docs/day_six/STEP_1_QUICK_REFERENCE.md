# Step 1: Quick Reference Card

**Date**: 2025-11-05
**Status**: ✅ VALIDATED

---

## 🎯 TL;DR

**Camera Pipeline**: ✅ CORRECT (95% confidence)
**Critical Bug**: 🔴 180° spawn misalignment
**Minor Issue**: 🟡 Vector size mismatch (23 vs 53)

---

## 📸 Camera Data Flow

```
CARLA → BGRA (256×144×4) uint8
  ↓ Drop alpha + BGR→RGB
RGB (256×144×3) uint8
  ↓ Grayscale (standard formula)
Gray (256×144) uint8
  ↓ Resize (INTER_AREA)
Gray (84×84) uint8
  ↓ Normalize [-1, 1]
Output (84×84) float32
  ↓ Stack 4 frames
CNN Input (4, 84, 84) float32 ✅
```

---

## ✅ What's Correct

- CARLA BGRA→RGB conversion
- Grayscale formula (CCIR 601)
- 84×84 resolution (DQN standard)
- [-1, 1] normalization (best practice)
- 4-frame stacking (temporal context)
- float32 type (GPU-efficient)
- Matches academic literature

---

## 🔴 Critical Issues

### Spawn Bug
```
Expected: Yaw -180°, Forward [-1, 0, 0]
Actual:   Yaw 0°,    Forward [1, 0, 0]
→ Vehicle faces backward (180° error)
```

**Fix**: `carla_env.py reset()` spawn calculation

---

## 🟡 Minor Issues

### Vector Size
```
Config:  (53,) = 3 + 25*2 waypoints
Actual:  (23,) = 3 + 10*2 waypoints
→ Missing 30 dimensions
```

**Fix**: Provide 25 waypoints OR update config

---

## 📊 Log Evidence

```
Initial:  [0.000, 0.000] - zeros (expected)
Step 0:   [-0.851, 0.608] - ✅ normalized
Step 1:   [-0.851, 0.631] - ✅ consistent
```

---

## 📚 Documentation

- Full Analysis: `STEP_1_OBSERVATION_ANALYSIS.md`
- Key Findings: `STEP_1_KEY_FINDINGS.md`
- Debug Log: Lines 1-300 of `DEBUG_validation_20251105_194845.log`

---

## 🚀 Next Actions

1. Fix spawn yaw calculation
2. Resolve vector waypoint count
3. Proceed to Step 2 (CNN)
