# Quick Fix Usage Guide

## Problem Fixed
✅ Diagnostic logs not appearing in validation script  
✅ Added `--log-level DEBUG` support to see waypoint blending logs

---

## How to Use

### 1. Start CARLA (Terminal 1)

```bash
cd /path/to/carla-0.9.16
./CarlaUE4.sh -quality-level=Low
```

### 2. Run Validation with DEBUG Logging (Terminal 2)

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

python scripts/validate_rewards_manual.py --log-level DEBUG
```

### 3. Drive and Observe Logs

**Controls:**
- W/A/S/D: Accelerate/Steer
- Space: Brake
- Q: Toggle reverse
- Esc: Quit

**Expected Logs (every step):**

```
15:30:45 - src.environment.waypoint_manager - DEBUG - [ROUTE_DISTANCE_BLEND] ON-ROUTE: dist_from_route=1.20m, using 100% projection=45.30m
15:30:46 - src.environment.waypoint_manager - DEBUG - [ROUTE_DISTANCE_BLEND] TRANSITION: dist_from_route=7.50m, blend=0.17, projection=44.20m, euclidean=42.10m, final=43.85m
```

---

## What to Check

### ✅ Success Indicators

1. **Logs appear** with `[ROUTE_DISTANCE_BLEND]` prefix
2. **Distance decreases smoothly** as you drive toward goal
3. **Progress reward changes continuously** (no 10.0 → 0.0 jumps)
4. **Smooth transitions** between ON-ROUTE → TRANSITION → FAR OFF-ROUTE

### ❌ Issues to Report

1. **No logs appear** even with `--log-level DEBUG`
2. **Distance jumps** unexpectedly in logs
3. **Reward still discontinuous** despite smooth blending
4. **Blend factor** outside [0, 1] range

---

## Changes Made

**File:** `scripts/validate_rewards_manual.py`

**Added:**
- `import logging` (line 22)
- `--log-level` argument (lines 453-460)
- `logging.basicConfig()` call (lines 464-471)

**Total:** ~15 lines added

---

## Next Steps

1. Run with DEBUG logging
2. Record observations (do logs appear? does discontinuity persist?)
3. Share terminal output if issues remain

---

## Reference

Full documentation: `docs/investigation/PHASE_4_LOGGING_FIX.md`
